

# Import centralized paths config
try:
    from ..config.paths import get_path
    DEFAULT_OUTPUT_PATH = str(get_path("humanml3d_canonical"))
except ImportError:
    DEFAULT_OUTPUT_PATH = "data/motions_processed/humanml3d/canonical.pt"

"""Convert HumanML3D dataset to Stick-Gen canonical format.

HumanML3D Feature Layout (263 dimensions per frame):
- Root velocity/translation: dims 0-3 (4)
- Local joint positions: dims 4-69 (22 joints × 3)
- Local joint velocities: dims 70-135 (22 joints × 3)
- Joint rotations (6D continuous): dims 136-262 (22 joints × 6 - but only 21 used)

We extract the 22 joint positions and project to our 5-segment stick figure.
"""

import glob
import logging
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

from .convert_amass import compute_basic_physics
from .joint_utils import (
    CanonicalJoints2D,
    clean_canonical_joints,
    joints_to_v3_segments_2d,
    normalize_skeleton_height,
    validate_v3_connectivity,
)
from .metadata_extractors import build_enhanced_metadata
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HumanML3D joint indices (SMPL 22-joint model)
# We map these to our 5-segment stick figure
HUMANML3D_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

# Action keywords to ActionType mapping for text-based inference
ACTION_KEYWORDS = {
    # Walking/Running
    r"\bwalk(s|ing|ed)?\b": ActionType.WALK,
    r"\brun(s|ning|ran)?\b": ActionType.RUN,
    r"\bjog(s|ging|ged)?\b": ActionType.RUN,
    r"\bsprint(s|ing|ed)?\b": ActionType.SPRINT,
    r"\bstep(s|ping|ped)?\b": ActionType.WALK,
    r"\bpace(s|d|ing)?\b": ActionType.WALK,
    r"\bstroll(s|ing|ed)?\b": ActionType.WALK,
    r"\bmarch(es|ing|ed)?\b": ActionType.WALK,
    # Jumping
    r"\bjump(s|ing|ed)?\b": ActionType.JUMP,
    r"\bhop(s|ping|ped)?\b": ActionType.JUMP,
    r"\bleap(s|ing|ed|t)?\b": ActionType.JUMP,
    r"\bskip(s|ping|ped)?\b": ActionType.JUMP,
    # Dancing
    r"\bdanc(e|es|ing|ed|er)?\b": ActionType.DANCE,
    r"\bballet\b": ActionType.DANCE,
    r"\bwaltz(es|ing|ed)?\b": ActionType.DANCE,
    r"\bspin(s|ning)?\b": ActionType.DANCE,
    r"\btwirl(s|ing|ed)?\b": ActionType.DANCE,
    r"\bperform(s|ing|ed|ance)?\b": ActionType.DANCE,
    # Sitting/Standing
    r"\bsit(s|ting|sat)?\b": ActionType.SIT,
    r"\bstand(s|ing|stood)?\b": ActionType.STAND,
    r"\bkneel(s|ing|ed)?\b": ActionType.KNEEL,
    r"\blie(s)?\s*(down)?\b": ActionType.LIE_DOWN,
    r"\blay(s|ing)?\s*(down)?\b": ActionType.LIE_DOWN,
    # Punching/Kicking/Fighting
    r"\bpunch(es|ing|ed)?\b": ActionType.PUNCH,
    r"\bkick(s|ing|ed)?\b": ActionType.KICK,
    r"\bfight(s|ing|fought)?\b": ActionType.FIGHT,
    r"\bbox(es|ing|ed)?\b": ActionType.PUNCH,
    r"\bdodge(s|d|ing)?\b": ActionType.DODGE,
    # Waving/Gestures
    r"\bwave(s|d|ing)?\b": ActionType.WAVE,
    r"\bpoint(s|ing|ed)?\b": ActionType.POINT,
    r"\bclap(s|ping|ped)?\b": ActionType.CLAP,
    # Throwing/Catching
    r"\bthrow(s|ing|threw|n)?\b": ActionType.THROWING,
    r"\bcatch(es|ing|caught)?\b": ActionType.CATCHING,
    r"\bpitch(es|ing|ed)?\b": ActionType.PITCHING,
    # Climbing/Crawling
    r"\bclimb(s|ing|ed)?\b": ActionType.CLIMBING,
    r"\bcrawl(s|ing|ed)?\b": ActionType.CRAWLING,
    r"\bswim(s|ming|swam)?\b": ActionType.SWIMMING,
    # Eating/Drinking
    r"\beat(s|ing|ate)?\b": ActionType.EATING,
    r"\bdrink(s|ing|drank)?\b": ActionType.DRINKING,
    # Emotional
    r"\bcelebrat(e|es|ing|ed)?\b": ActionType.CELEBRATE,
    r"\bcry(ing|ied)?\b": ActionType.CRY,
    r"\blaugh(s|ing|ed)?\b": ActionType.LAUGH,
}


def _compute_stats_from_features(feats_dir: str) -> dict[str, np.ndarray]:
    """Compute normalization statistics from raw feature files.

    Used when Mean.npy/Std.npy are missing or corrupted. Computes mean and std
    across all frames from all feature files in the directory, skipping any
    files that contain NaN or Inf values.

    Args:
        feats_dir: Directory containing .npy feature files.

    Returns:
        Dict with 'mean' and 'std' arrays of shape (D,).
    """
    npy_files = sorted(glob.glob(os.path.join(feats_dir, "*.npy")))
    if not npy_files:
        raise ValueError(f"No .npy files found in {feats_dir}")

    logger.info(f"Computing normalization stats from {len(npy_files)} feature files...")

    # First pass: filter out files with NaN/Inf and accumulate for mean
    sample = np.load(npy_files[0])
    D = sample.shape[1]

    valid_files: list[str] = []
    skipped_files = 0

    # Pre-filter files to exclude those with NaN/Inf
    for fpath in npy_files:
        data = np.load(fpath)
        if np.isfinite(data).all():
            valid_files.append(fpath)
        else:
            skipped_files += 1

    if skipped_files > 0:
        logger.warning(
            f"Skipped {skipped_files} files with NaN/Inf values during stats computation"
        )

    if not valid_files:
        raise ValueError("All feature files contain NaN/Inf values")

    logger.info(f"Using {len(valid_files)} valid files for stats computation")

    # Accumulate all valid features for computing mean and std
    # Use explicit accumulation to avoid numerical issues with sum(generator)
    weighted_sum = np.zeros(D, dtype=np.float64)
    total_count = 0

    chunk_size = 1000
    for i in range(0, len(valid_files), chunk_size):
        chunk_files = valid_files[i : i + chunk_size]
        chunk_feats = [np.load(f) for f in chunk_files]
        chunk_concat = np.concatenate(chunk_feats, axis=0)
        weighted_sum += np.sum(chunk_concat, axis=0)
        total_count += chunk_concat.shape[0]

    mean = (weighted_sum / total_count).astype(np.float64)

    # Second pass for std (using the computed mean)
    sum_sq = np.zeros(D, dtype=np.float64)
    for i in range(0, len(valid_files), chunk_size):
        chunk_files = valid_files[i : i + chunk_size]
        chunk_feats = [np.load(f) for f in chunk_files]
        chunk_concat = np.concatenate(chunk_feats, axis=0)
        sum_sq += np.sum((chunk_concat - mean) ** 2, axis=0)

    std = np.sqrt(sum_sq / total_count).astype(np.float32)
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    logger.info(
        f"Computed stats: mean range [{mean.min():.4f}, {mean.max():.4f}], "
        f"std range [{std.min():.4f}, {std.max():.4f}]"
    )

    return {"mean": mean.astype(np.float32), "std": std}


def _load_normalization(stats_dir: str) -> dict[str, np.ndarray] | None:
    """Load normalization statistics with error handling.

    If Mean.npy/Std.npy are missing or corrupted (contain NaN values),
    automatically recomputes statistics from the raw feature files in
    new_joint_vecs/ and caches them to Mean_computed.npy/Std_computed.npy.

    Args:
        stats_dir: Directory containing Mean.npy, Std.npy, and new_joint_vecs/.

    Returns:
        Dict with 'mean' and 'std' arrays, or None if stats cannot be obtained.
    """
    mean_path = os.path.join(stats_dir, "Mean.npy")
    std_path = os.path.join(stats_dir, "Std.npy")
    feats_dir = os.path.join(stats_dir, "new_joint_vecs")

    # Paths for cached computed stats
    mean_computed_path = os.path.join(stats_dir, "Mean_computed.npy")
    std_computed_path = os.path.join(stats_dir, "Std_computed.npy")

    # Check for pre-computed cached stats first
    if os.path.exists(mean_computed_path) and os.path.exists(std_computed_path):
        logger.info("Using cached computed normalization stats")
        mean = np.load(mean_computed_path)
        std = np.load(std_computed_path)
        std = np.where(std < 1e-8, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    # Try to load original stats
    stats_valid = False
    if os.path.exists(mean_path) and os.path.exists(std_path):
        try:
            mean = np.load(mean_path)
            std = np.load(std_path)

            # Check if stats are corrupted (contain NaN values)
            if np.isnan(mean).any() or np.isnan(std).any():
                nan_count_mean = np.sum(np.isnan(mean))
                nan_count_std = np.sum(np.isnan(std))
                logger.warning(
                    f"Mean.npy has {nan_count_mean}/{len(mean)} NaN values, "
                    f"Std.npy has {nan_count_std}/{len(std)} NaN values. "
                    "Will recompute from raw features."
                )
            elif np.isinf(mean).any() or np.isinf(std).any():
                logger.warning("Stats contain Inf values. Will recompute from raw features.")
            else:
                stats_valid = True
                # Avoid division by zero
                std = np.where(std < 1e-8, 1.0, std)
                return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        except Exception as e:
            logger.warning(f"Failed to load normalization stats: {e}")

    # Stats missing or corrupted - compute from raw features
    if not stats_valid:
        if not os.path.isdir(feats_dir):
            logger.error(f"Cannot compute stats: feature directory not found at {feats_dir}")
            return None

        try:
            stats = _compute_stats_from_features(feats_dir)

            # Cache computed stats for future runs
            np.save(mean_computed_path, stats["mean"])
            np.save(std_computed_path, stats["std"])
            logger.info(f"Cached computed stats to {mean_computed_path} and {std_computed_path}")

            return stats
        except Exception as e:
            logger.error(f"Failed to compute normalization stats: {e}")
            return None

    return None


def _denorm(arr: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Denormalize features using mean and std."""
    return arr * stats["std"] + stats["mean"]


def _parse_text_content(content: str) -> list[str]:
    """Parse text annotation content into list of descriptions.

    Each line may have format: "text#start#end" or just "text"
    """
    lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Extract just the text description (before any # delimiters)
        parts = line.split("#")
        if parts[0].strip():
            lines.append(parts[0].strip())
    return lines


def _load_texts_from_dir(texts_dir: str) -> dict[str, list[str]]:
    """Load HumanML3D texts from a directory of .txt files."""
    mapping: dict[str, list[str]] = {}
    if not os.path.isdir(texts_dir):
        return mapping

    for txt_file in glob.glob(os.path.join(texts_dir, "*.txt")):
        clip_id = os.path.splitext(os.path.basename(txt_file))[0]
        try:
            with open(txt_file, encoding="utf-8") as f:
                content = f.read()
                lines = _parse_text_content(content)
                if lines:
                    mapping[clip_id] = lines
        except Exception as e:
            logger.debug(f"Failed to read {txt_file}: {e}")

    return mapping


def _load_texts_from_zip(zip_path: str) -> dict[str, list[str]]:
    """Load HumanML3D texts.zip into {clip_id: [lines...]}.

    Each text file contains multiple annotations separated by '#'.
    Format: "description#start_time#end_time" or just "description"
    """
    mapping: dict[str, list[str]] = {}
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
                        lines = _parse_text_content(content)
                        if lines:
                            mapping[clip_id] = lines
                except Exception as e:
                    logger.debug(f"Failed to read {name}: {e}")
    except Exception as e:
        logger.error(f"Failed to open texts.zip: {e}")

    return mapping


def _load_texts(root_dir: str) -> dict[str, list[str]]:
    """Load text annotations from either texts/ directory or texts.zip.

    Prefers texts/ directory if it exists (more common in unpacked datasets).
    """
    texts_dir = os.path.join(root_dir, "texts")
    if os.path.isdir(texts_dir):
        logger.info(f"Loading texts from directory: {texts_dir}")
        return _load_texts_from_dir(texts_dir)

    texts_zip = os.path.join(root_dir, "texts.zip")
    if os.path.exists(texts_zip):
        logger.info(f"Loading texts from zip: {texts_zip}")
        return _load_texts_from_zip(texts_zip)

    logger.warning(f"No texts found in {root_dir} (checked texts/ and texts.zip)")
    return {}


def _infer_action_from_text(texts: list[str]) -> ActionType:
    """Infer primary action type from text annotations using keyword matching."""
    if not texts:
        return ActionType.IDLE

    # Combine all texts for matching
    combined = " ".join(texts).lower()

    # Count matches for each action
    action_scores: dict[ActionType, int] = {}
    for pattern, action in ACTION_KEYWORDS.items():
        matches = re.findall(pattern, combined, re.IGNORECASE)
        if matches:
            action_scores[action] = action_scores.get(action, 0) + len(matches)

    if not action_scores:
        return ActionType.IDLE

    # Return the action with highest score
    return max(action_scores, key=action_scores.get)


def _features_to_stick(feats: np.ndarray) -> np.ndarray:
    """Map HumanML3D features to v3 stick figure ``[T, 48]``.

    HumanML3D feature layout (263 dims):

    * ``[0:4]``: Root velocity and height.
    * ``[4:67]``: Local joint positions for 21 joints × 3 (relative to root).
    * Remaining dims: joint velocities, contacts, and rotations.

    We extract the 21 joint positions, project (x, z) → 2D, build canonical
    joints, and convert to the v3 12‑segment skeleton via
    :func:`joints_to_v3_segments_2d`.
    """

    T, D = feats.shape

    # Handle edge cases
    if T == 0:
        return np.zeros((1, 48), dtype=np.float32)

    # For standard HumanML3D (263 dims), extract joint positions
    if D >= 67:
        # Joint positions are in dims 4:67 (21 joints × 3).
        joint_pos = feats[:, 4:67].reshape(T, 21, 3)

        # Project 3D to 2D (use x and z as our x, y - y is height in SMPL).
        joint_2d = joint_pos[:, :, [0, 2]]  # [T, 21, 2]

        # Local indices follow the HumanML3D layout *without* the root pelvis.
        l_hip = joint_2d[:, 0]
        r_hip = joint_2d[:, 1]
        l_knee = joint_2d[:, 3]
        r_knee = joint_2d[:, 4]
        l_ankle = joint_2d[:, 6]
        r_ankle = joint_2d[:, 7]
        neck = joint_2d[:, 11]
        head_center = joint_2d[:, 14]
        l_shoulder = joint_2d[:, 15]
        r_shoulder = joint_2d[:, 16]
        l_elbow = joint_2d[:, 17]
        r_elbow = joint_2d[:, 18]
        l_wrist = joint_2d[:, 19]
        r_wrist = joint_2d[:, 20]

        pelvis_center = 0.5 * (l_hip + r_hip)
        chest = 0.5 * (l_shoulder + r_shoulder)

        canonical_joints: CanonicalJoints2D = {
            "pelvis_center": pelvis_center,
            "chest": chest,
            "neck": neck,
            "head_center": head_center,
            "l_shoulder": l_shoulder,
            "r_shoulder": r_shoulder,
            "l_elbow": l_elbow,
            "r_elbow": r_elbow,
            "l_wrist": l_wrist,
            "r_wrist": r_wrist,
            "l_hip": l_hip,
            "r_hip": r_hip,
            "l_knee": l_knee,
            "r_knee": r_knee,
            "l_ankle": l_ankle,
            "r_ankle": r_ankle,
        }

        # Drop frames with non-finite canonical joints up front. Real HumanML3D
        # clips occasionally contain NaNs in a few frames; we prefer to remove
        # those frames rather than reject the entire clip.
        stacked = np.stack(list(canonical_joints.values()), axis=1)  # [T, J, 2]
        finite_mask = np.isfinite(stacked).all(axis=(1, 2))
        if not np.any(finite_mask):
            raise ValueError("Non-finite canonical joints for all frames")
        if not finite_mask.all():
            for name in canonical_joints.keys():
                canonical_joints[name] = canonical_joints[name][finite_mask]

        # 1. Clean geometric inconsistencies (alignment, clamping, head fix)
        canonical_joints = clean_canonical_joints(canonical_joints)

        # 2. Normalize height to ~1.8m (standard units) to fix physics scale issues
        # This is critical for KIT-ML which often triggers velocity checks due to scale
        canonical_joints = normalize_skeleton_height(
            canonical_joints, target_height=1.8
        )

        # 3. Convert to v3 segments and enforce connectivity. Any problem here is
        # surfaced as a dedicated exception so upstream callers can categorize
        # skip reasons (e.g. "connectivity" vs "physics").
        segments = joints_to_v3_segments_2d(canonical_joints, flatten=True)
        validate_v3_connectivity(segments)
        return segments.astype(np.float32)

    # Fallback for non-standard feature dimensions: keep data but ensure 48 dims.
    if D >= 48:
        arr = feats[:, :48]
    else:
        pad = np.zeros((T, 48 - D), dtype=feats.dtype)
        arr = np.concatenate([feats, pad], axis=1)
    return arr.astype(np.float32)


def _generate_camera_from_motion(motion: torch.Tensor) -> torch.Tensor | None:
    """Generate basic camera data based on motion trajectory.

    Creates smooth camera that follows the center of motion.
    Returns [T, 3] tensor with (x, y, zoom) per frame.
    """
    T = motion.shape[0]
    if T == 0:
        return None

    motion_np = motion.numpy()
    if motion_np.shape[1] % 4 != 0:
        raise ValueError(
            "Camera generation expects motion with last dim as segments * 4, "
            f"got {motion_np.shape[1]}",
        )

    num_segments = motion_np.shape[1] // 4
    segments = motion_np.reshape(T, num_segments, 4)

    # Compute center of mass (average of all segment endpoints)
    all_x = np.concatenate([segments[:, :, 0], segments[:, :, 2]], axis=1)
    all_y = np.concatenate([segments[:, :, 1], segments[:, :, 3]], axis=1)

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
        center_x = np.convolve(center_x, kernel, mode="same")
        center_y = np.convolve(center_y, kernel, mode="same")

    # Zoom based on motion spread
    spread = np.std(all_x, axis=1) + np.std(all_y, axis=1)
    zoom = 1.0 / (spread + 0.5)  # Zoom out when motion spreads
    zoom = np.clip(zoom, 0.5, 2.0)

    camera = np.stack([center_x, center_y, zoom], axis=1).astype(np.float32)
    return torch.from_numpy(camera)


def _build_sample(
    feats: np.ndarray,
    texts: list[str],
    clip_id: str,
    stats_fps: int = 20,
    include_camera: bool = False,
) -> dict[str, Any]:
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
    description = (
        texts[0] if texts else f"A human motion clip from HumanML3D ({clip_id})."
    )

    # Generate camera if requested
    camera = _generate_camera_from_motion(motion) if include_camera else None

    # Build enhanced metadata (HumanML3D uses 20 FPS consistently)
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=stats_fps,
        description=description,
        original_fps=stats_fps,  # HumanML3D is already at native fps
        original_num_frames=T,
    )

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
        "enhanced_meta": enhanced_meta.model_dump(),
    }


def _process_single_clip(
    args: tuple[str, dict[str, np.ndarray], dict[str, list[str]], int, bool, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Process a single clip (for parallel processing).

    Returns ``(sample, skip_reason)`` where ``sample`` is ``None`` when the clip
    is skipped and ``skip_reason`` is a short category string suitable for
    aggregation (e.g. ``"too_few_frames"``, ``"physics_velocity"``).
    """
    path, stats, text_map, fps, include_camera, physics_threshold = args

    clip_id = os.path.splitext(os.path.basename(path))[0]

    try:
        feats = np.load(path).astype(np.float32)

        # Validate input shape
        if len(feats.shape) != 2:
            logger.debug(f"Skipping {clip_id}: invalid shape {feats.shape}")
            return None, "invalid_shape"

        if feats.shape[0] < 10:  # Minimum 10 frames
            logger.debug(f"Skipping {clip_id}: too few frames ({feats.shape[0]})")
            return None, "too_few_frames"

        feats_denorm = _denorm(feats, stats)
        texts = text_map.get(clip_id, [])
        sample = _build_sample(
            feats_denorm, texts, clip_id, stats_fps=fps, include_camera=include_camera
        )

        # Physics validation with configurable threshold
        validator = DataValidator(fps=fps)
        # Temporarily increase thresholds for expressive motions
        original_max_vel = validator.max_velocity
        original_max_acc = validator.max_acceleration
        validator.max_velocity = original_max_vel * physics_threshold
        validator.max_acceleration = original_max_acc * physics_threshold

        ok, _score, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            logger.debug(f"Skipping {clip_id}: {reason}")
            if "Velocity limit exceeded" in reason:
                return None, "physics_velocity"
            if "Acceleration limit exceeded" in reason:
                return None, "physics_acceleration"
            return None, "physics_other"

        return sample, None

    except Exception as e:  # noqa: BLE001
        msg = str(e)
        logger.warning(f"Error processing {clip_id}: {msg}")

        # Provide more informative skip reasons for downstream aggregation.
        if "v3 connectivity violated" in msg:
            return None, "connectivity"
        if "Non-finite canonical joints" in msg:
            return None, "non_finite_joints"
        if "non-finite coordinates in segments" in msg:
            return None, "non_finite_segments"

        return None, "exception"


def convert_humanml3d(
    root_dir: str,
    output_path: str,
    fps: int = 20,
    max_clips: int = -1,
    include_camera: bool = False,
    physics_threshold: float = 2.0,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
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

    # Load text annotations (supports both texts/ directory and texts.zip)
    text_map = _load_texts(root_dir)
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

    samples: list[dict[str, Any]] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}

    def _record_skip(reason: str | None) -> None:
        if reason is None:
            return
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if num_workers > 1:
        # Parallel processing
        args_list = [
            (p, stats, text_map, fps, include_camera, physics_threshold) for p in paths
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_clip, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                result, reason = future.result()
                if result is not None:
                    samples.append(result)
                else:
                    skipped += 1
                    _record_skip(reason)
    else:
        # Sequential processing with progress
        for i, path in enumerate(paths):
            if i % 500 == 0:
                logger.info(f"Processing {i}/{len(paths)}...")

            result, reason = _process_single_clip(
                (path, stats, text_map, fps, include_camera, physics_threshold)
            )
            if result is not None:
                samples.append(result)
            else:
                skipped += 1
                _record_skip(reason)

    logger.info(f"Converted {len(samples)}/{len(paths)} clips ({skipped} skipped)")
    if skipped > 0:
        logger.info("Skip reasons: %s", skip_reasons)

    # Compute action distribution
    action_counts: dict[str, int] = {}
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
        """,
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of HumanML3D"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument("--fps", type=int, default=20, help="Frame rate (default: 20)")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=-1,
        help="Maximum clips to process (-1 for all)",
    )
    parser.add_argument(
        "--include-camera",
        action="store_true",
        help="Generate camera data from motion trajectory",
    )
    parser.add_argument(
        "--physics-threshold",
        type=float,
        default=2.0,
        help="Physics validation threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
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
