"""Convert HumanML3D dataset to Stick-Gen canonical format.

HumanML3D Feature Layout (263 dimensions per frame):
- Root velocity/translation: dims 0-3 (4)
- Local joint positions: dims 4-69 (22 joints × 3)
- Local joint velocities: dims 70-135 (22 joints × 3)
- Joint rotations (6D continuous): dims 136-262 (22 joints × 6 - but only 21 used)

We extract the 22 joint positions and project to our 5-segment stick figure.
"""

import os
import re
import glob
import zipfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .schema import ActionType, ACTION_TO_IDX
from .validator import DataValidator
from .convert_amass import compute_basic_physics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HumanML3D joint indices (SMPL 22-joint model)
# We map these to our 5-segment stick figure
HUMANML3D_JOINTS = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
}

# Action keywords to ActionType mapping for text-based inference
ACTION_KEYWORDS = {
    # Walking/Running
    r'\bwalk(s|ing|ed)?\b': ActionType.WALK,
    r'\brun(s|ning|ran)?\b': ActionType.RUN,
    r'\bjog(s|ging|ged)?\b': ActionType.RUN,
    r'\bsprint(s|ing|ed)?\b': ActionType.SPRINT,
    r'\bstep(s|ping|ped)?\b': ActionType.WALK,
    r'\bpace(s|d|ing)?\b': ActionType.WALK,
    r'\bstroll(s|ing|ed)?\b': ActionType.WALK,
    r'\bmarch(es|ing|ed)?\b': ActionType.WALK,

    # Jumping
    r'\bjump(s|ing|ed)?\b': ActionType.JUMP,
    r'\bhop(s|ping|ped)?\b': ActionType.JUMP,
    r'\bleap(s|ing|ed|t)?\b': ActionType.JUMP,
    r'\bskip(s|ping|ped)?\b': ActionType.JUMP,

    # Dancing
    r'\bdanc(e|es|ing|ed|er)?\b': ActionType.DANCE,
    r'\bballet\b': ActionType.DANCE,
    r'\bwaltz(es|ing|ed)?\b': ActionType.DANCE,
    r'\bspin(s|ning)?\b': ActionType.DANCE,
    r'\btwirl(s|ing|ed)?\b': ActionType.DANCE,
    r'\bperform(s|ing|ed|ance)?\b': ActionType.DANCE,

    # Sitting/Standing
    r'\bsit(s|ting|sat)?\b': ActionType.SIT,
    r'\bstand(s|ing|stood)?\b': ActionType.STAND,
    r'\bkneel(s|ing|ed)?\b': ActionType.KNEEL,
    r'\blie(s)?\s*(down)?\b': ActionType.LIE_DOWN,
    r'\blay(s|ing)?\s*(down)?\b': ActionType.LIE_DOWN,

    # Punching/Kicking/Fighting
    r'\bpunch(es|ing|ed)?\b': ActionType.PUNCH,
    r'\bkick(s|ing|ed)?\b': ActionType.KICK,
    r'\bfight(s|ing|fought)?\b': ActionType.FIGHT,
    r'\bbox(es|ing|ed)?\b': ActionType.PUNCH,
    r'\bdodge(s|d|ing)?\b': ActionType.DODGE,

    # Waving/Gestures
    r'\bwave(s|d|ing)?\b': ActionType.WAVE,
    r'\bpoint(s|ing|ed)?\b': ActionType.POINT,
    r'\bclap(s|ping|ped)?\b': ActionType.CLAP,

    # Throwing/Catching
    r'\bthrow(s|ing|threw|n)?\b': ActionType.THROWING,
    r'\bcatch(es|ing|caught)?\b': ActionType.CATCHING,
    r'\bpitch(es|ing|ed)?\b': ActionType.PITCHING,

    # Climbing/Crawling
    r'\bclimb(s|ing|ed)?\b': ActionType.CLIMBING,
    r'\bcrawl(s|ing|ed)?\b': ActionType.CRAWLING,
    r'\bswim(s|ming|swam)?\b': ActionType.SWIMMING,

    # Eating/Drinking
    r'\beat(s|ing|ate)?\b': ActionType.EATING,
    r'\bdrink(s|ing|drank)?\b': ActionType.DRINKING,

    # Emotional
    r'\bcelebrat(e|es|ing|ed)?\b': ActionType.CELEBRATE,
    r'\bcry(ing|ied)?\b': ActionType.CRY,
    r'\blaugh(s|ing|ed)?\b': ActionType.LAUGH,
}


def _load_normalization(stats_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load normalization statistics with error handling."""
    mean_path = os.path.join(stats_dir, "Mean.npy")
    std_path = os.path.join(stats_dir, "Std.npy")

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        logger.warning(f"Missing normalization files in {stats_dir}")
        return None

    try:
        mean = np.load(mean_path)
        std = np.load(std_path)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    except Exception as e:
        logger.error(f"Failed to load normalization stats: {e}")
        return None


def _denorm(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Denormalize features using mean and std."""
    return arr * stats["std"] + stats["mean"]


def _load_texts_from_zip(zip_path: str) -> Dict[str, List[str]]:
    """Load HumanML3D texts.zip into {clip_id: [lines...]}.

    Each text file contains multiple annotations separated by '#'.
    Format: "description#start_time#end_time" or just "description"
    """
    mapping: Dict[str, List[str]] = {}
    if not os.path.exists(zip_path):
        logger.warning(f"texts.zip not found at {zip_path}")
        return mapping

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".txt"):
                    continue
                clip_id = os.path.splitext(os.path.basename(name))[0]
                try:
                    with zf.open(name) as f:
                        content = f.read().decode("utf-8")
                        # Parse annotations - each line may have format: "text#start#end"
                        lines = []
                        for line in content.strip().split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            # Extract just the text description (before any # delimiters)
                            parts = line.split('#')
                            if parts[0].strip():
                                lines.append(parts[0].strip())
                        mapping[clip_id] = lines
                except Exception as e:
                    logger.debug(f"Failed to read {name}: {e}")
    except Exception as e:
        logger.error(f"Failed to open texts.zip: {e}")

    return mapping


def _infer_action_from_text(texts: List[str]) -> ActionType:
    """Infer primary action type from text annotations using keyword matching."""
    if not texts:
        return ActionType.IDLE

    # Combine all texts for matching
    combined = ' '.join(texts).lower()

    # Count matches for each action
    action_scores: Dict[ActionType, int] = {}
    for pattern, action in ACTION_KEYWORDS.items():
        matches = re.findall(pattern, combined, re.IGNORECASE)
        if matches:
            action_scores[action] = action_scores.get(action, 0) + len(matches)

    if not action_scores:
        return ActionType.IDLE

    # Return the action with highest score
    return max(action_scores, key=action_scores.get)


def _features_to_stick(feats: np.ndarray) -> np.ndarray:
    """Map HumanML3D 263-dim features to stick figure [T, 20].

    HumanML3D feature layout (263 dims):
    - [0:4]: Root velocity and height
    - [4:67]: Local joint positions (21 joints × 3, relative to root)
    - [67:193]: Local joint velocities (21 joints × 3) + global velocities
    - [193:263]: Foot contact + joint rotations

    We extract joint positions and convert to our 5-segment format:
    - Segment 0: Torso (head to pelvis)
    - Segment 1: Left arm (shoulder to wrist)
    - Segment 2: Right arm (shoulder to wrist)
    - Segment 3: Left leg (hip to ankle)
    - Segment 4: Right leg (hip to ankle)

    Each segment: 4 values (x1, y1, x2, y2) = 20 total
    """
    T, D = feats.shape

    # Handle edge cases
    if T == 0:
        return np.zeros((1, 20), dtype=np.float32)

    # For standard HumanML3D (263 dims), extract joint positions
    if D >= 67:
        # Joint positions are in dims 4:67 (21 joints × 3)
        # We only need x and y (ignoring z for 2D stick figure)
        joint_pos = feats[:, 4:67].reshape(T, 21, 3)

        # Project 3D to 2D (use x and z as our x, y - y is height in SMPL)
        joint_2d = joint_pos[:, :, [0, 2]]  # [T, 21, 2]

        # Map to stick figure segments
        # Note: Joint indices shifted by 1 since root is not in local positions
        stick = np.zeros((T, 5, 4), dtype=np.float32)

        # Torso: neck (index 11) to pelvis (approximated as mean of hips)
        # In local coords, pelvis is at origin, so use spine/neck
        neck_idx = 11  # neck in 21-joint (0-indexed after root)
        spine_idx = 5   # spine2
        stick[:, 0, 0:2] = joint_2d[:, neck_idx]      # head/neck
        stick[:, 0, 2:4] = joint_2d[:, spine_idx] * 0  # pelvis at origin

        # Left arm: shoulder (15) to wrist (19)
        l_shoulder_idx = 15
        l_wrist_idx = 19
        stick[:, 1, 0:2] = joint_2d[:, l_shoulder_idx]
        stick[:, 1, 2:4] = joint_2d[:, l_wrist_idx]

        # Right arm: shoulder (16) to wrist (20)
        r_shoulder_idx = 16
        r_wrist_idx = 20
        stick[:, 2, 0:2] = joint_2d[:, r_shoulder_idx]
        stick[:, 2, 2:4] = joint_2d[:, r_wrist_idx]

        # Left leg: hip (0) to ankle (6)
        l_hip_idx = 0
        l_ankle_idx = 6
        stick[:, 3, 0:2] = joint_2d[:, l_hip_idx]
        stick[:, 3, 2:4] = joint_2d[:, l_ankle_idx]

        # Right leg: hip (1) to ankle (7)
        r_hip_idx = 1
        r_ankle_idx = 7
        stick[:, 4, 0:2] = joint_2d[:, r_hip_idx]
        stick[:, 4, 2:4] = joint_2d[:, r_ankle_idx]

        # Normalize to reasonable range (-1 to 1 ish)
        stick = stick / (np.abs(stick).max() + 1e-8) * 2.0

        return stick.reshape(T, 20).astype(np.float32)

    # Fallback for non-standard feature dimensions
    if D >= 20:
        arr = feats[:, :20]
    else:
        pad = np.zeros((T, 20 - D), dtype=feats.dtype)
        arr = np.concatenate([feats, pad], axis=1)
    return arr.astype(np.float32)
def _generate_camera_from_motion(motion: torch.Tensor) -> Optional[torch.Tensor]:
    """Generate basic camera data based on motion trajectory.

    Creates smooth camera that follows the center of motion.
    Returns [T, 3] tensor with (x, y, zoom) per frame.
    """
    T = motion.shape[0]
    if T == 0:
        return None

    # Reshape to [T, 5, 4] to get segments
    motion_np = motion.numpy().reshape(T, 5, 4)

    # Compute center of mass (average of all joint positions)
    all_x = np.concatenate([motion_np[:, :, 0], motion_np[:, :, 2]], axis=1)  # [T, 10]
    all_y = np.concatenate([motion_np[:, :, 1], motion_np[:, :, 3]], axis=1)  # [T, 10]

    center_x = np.mean(all_x, axis=1)  # [T]
    center_y = np.mean(all_y, axis=1)  # [T]

    # Smooth the camera trajectory
    from scipy.ndimage import gaussian_filter1d
    try:
        center_x = gaussian_filter1d(center_x, sigma=5)
        center_y = gaussian_filter1d(center_y, sigma=5)
    except ImportError:
        # Fallback: simple moving average
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        center_x = np.convolve(center_x, kernel, mode='same')
        center_y = np.convolve(center_y, kernel, mode='same')

    # Zoom based on motion spread
    spread = np.std(all_x, axis=1) + np.std(all_y, axis=1)
    zoom = 1.0 / (spread + 0.5)  # Zoom out when motion spreads
    zoom = np.clip(zoom, 0.5, 2.0)

    camera = np.stack([center_x, center_y, zoom], axis=1).astype(np.float32)
    return torch.from_numpy(camera)


def _build_sample(feats: np.ndarray,
                  texts: List[str],
                  clip_id: str,
                  stats_fps: int = 20,
                  include_camera: bool = False) -> Dict[str, Any]:
    """Build a canonical sample dict from HumanML3D features.

    Args:
        feats: Denormalized feature array [T, 263]
        texts: List of text annotations
        clip_id: Unique clip identifier
        stats_fps: Frame rate of the data
        include_camera: Whether to generate camera data

    Returns:
        Sample dict with motion, physics, actions, etc.
    """
    motion_np = _features_to_stick(feats)
    motion = torch.from_numpy(motion_np)
    physics = compute_basic_physics(motion, fps=stats_fps)

    # Infer action from text annotations
    action_enum = _infer_action_from_text(texts)
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    # Use first text as primary description, keep all for training
    description = texts[0] if texts else f"A human motion clip from HumanML3D ({clip_id})."

    # Generate camera if requested
    camera = _generate_camera_from_motion(motion) if include_camera else None

    return {
        "description": description,
        "all_descriptions": texts,  # Keep all text annotations
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "action_label": action_enum.value,  # String label for debugging
        "camera": camera,
        "source": "humanml3d",
        "meta": {
            "clip_id": clip_id,
            "fps": stats_fps,
            "num_frames": T,
            "feature_dim": feats.shape[1] if len(feats.shape) > 1 else 0,
        },
    }


def _process_single_clip(args: Tuple[str, Dict[str, np.ndarray], Dict[str, List[str]], int, bool, float]) -> Optional[Dict[str, Any]]:
    """Process a single clip (for parallel processing).

    Returns sample dict or None if validation fails.
    """
    path, stats, text_map, fps, include_camera, physics_threshold = args

    clip_id = os.path.splitext(os.path.basename(path))[0]

    try:
        feats = np.load(path).astype(np.float32)

        # Validate input shape
        if len(feats.shape) != 2:
            logger.debug(f"Skipping {clip_id}: invalid shape {feats.shape}")
            return None

        if feats.shape[0] < 10:  # Minimum 10 frames
            logger.debug(f"Skipping {clip_id}: too few frames ({feats.shape[0]})")
            return None

        feats_denorm = _denorm(feats, stats)
        texts = text_map.get(clip_id, [])
        sample = _build_sample(feats_denorm, texts, clip_id, stats_fps=fps, include_camera=include_camera)

        # Physics validation with configurable threshold
        validator = DataValidator(fps=fps)
        # Temporarily increase thresholds for expressive motions
        original_max_vel = validator.max_velocity
        original_max_acc = validator.max_acceleration
        validator.max_velocity = original_max_vel * physics_threshold
        validator.max_acceleration = original_max_acc * physics_threshold

        ok, score, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            logger.debug(f"Skipping {clip_id}: {reason}")
            return None

        return sample

    except Exception as e:
        logger.warning(f"Error processing {clip_id}: {e}")
        return None


def convert_humanml3d(
    root_dir: str,
    output_path: str,
    fps: int = 20,
    max_clips: int = -1,
    include_camera: bool = False,
    physics_threshold: float = 2.0,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Convert HumanML3D preprocessed features into canonical schema.

    This implementation assumes the standard HumanML3D layout with
    `new_joint_vecs/` (feature vectors), `Mean.npy`, `Std.npy`, and `texts.zip`.

    Args:
        root_dir: Root directory containing HumanML3D data
        output_path: Output .pt file path
        fps: Frame rate (default 20 for HumanML3D)
        max_clips: Maximum clips to process (-1 for all)
        include_camera: Generate camera data from motion
        physics_threshold: Multiplier for physics validation thresholds
                          (higher = more lenient, good for expressive motions)
        num_workers: Number of parallel workers (1 = sequential)

    Returns:
        List of converted samples
    """
    logger.info(f"Converting HumanML3D from {root_dir}")

    # Load normalization stats
    stats = _load_normalization(root_dir)
    if stats is None:
        raise ValueError(f"Could not load normalization stats from {root_dir}")

    # Load text annotations
    texts_zip = os.path.join(root_dir, "texts.zip")
    text_map = _load_texts_from_zip(texts_zip)
    logger.info(f"Loaded {len(text_map)} text annotations")

    # Find feature files
    feats_dir = os.path.join(root_dir, "new_joint_vecs")
    if not os.path.exists(feats_dir):
        raise ValueError(f"Feature directory not found: {feats_dir}")

    paths = sorted(glob.glob(os.path.join(feats_dir, "*.npy")))
    logger.info(f"Found {len(paths)} feature files")

    if max_clips > 0:
        paths = paths[:max_clips]
        logger.info(f"Limited to {max_clips} clips")

    samples: List[Dict[str, Any]] = []
    skipped = 0

    if num_workers > 1:
        # Parallel processing
        args_list = [(p, stats, text_map, fps, include_camera, physics_threshold) for p in paths]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_clip, args): args[0] for args in args_list}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    samples.append(result)
                else:
                    skipped += 1
    else:
        # Sequential processing with progress
        for i, path in enumerate(paths):
            if i % 500 == 0:
                logger.info(f"Processing {i}/{len(paths)}...")

            result = _process_single_clip((path, stats, text_map, fps, include_camera, physics_threshold))
            if result is not None:
                samples.append(result)
            else:
                skipped += 1

    logger.info(f"Converted {len(samples)}/{len(paths)} clips ({skipped} skipped)")

    # Compute action distribution
    action_counts: Dict[str, int] = {}
    for s in samples:
        label = s.get("action_label", "idle")
        action_counts[label] = action_counts.get(label, 0) + 1
    logger.info(f"Action distribution: {action_counts}")

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    logger.info(f"Saved to {output_path}")

    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HumanML3D to Stick-Gen canonical schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python -m src.data_gen.convert_humanml3d --root data/humanml3d --output data/humanml3d_converted.pt

    # With camera generation and lenient physics
    python -m src.data_gen.convert_humanml3d --root data/humanml3d --output data/humanml3d.pt \\
        --include-camera --physics-threshold 3.0

    # Parallel processing for large datasets
    python -m src.data_gen.convert_humanml3d --root data/humanml3d --output data/humanml3d.pt \\
        --num-workers 8

Prerequisites:
    Download HumanML3D from: https://github.com/EricGuo5513/HumanML3D
    Required files: new_joint_vecs/*.npy, Mean.npy, Std.npy, texts.zip
        """
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of HumanML3D")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frame rate (default: 20)")
    parser.add_argument("--max-clips", type=int, default=-1,
                        help="Maximum clips to process (-1 for all)")
    parser.add_argument("--include-camera", action="store_true",
                        help="Generate camera data from motion trajectory")
    parser.add_argument("--physics-threshold", type=float, default=2.0,
                        help="Physics validation threshold multiplier (default: 2.0)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    convert_humanml3d(
        args.root,
        args.output,
        fps=args.fps,
        max_clips=args.max_clips,
        include_camera=args.include_camera,
        physics_threshold=args.physics_threshold,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
