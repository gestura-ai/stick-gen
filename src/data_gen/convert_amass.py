"""
AMASS to Stick Figure Converter

Converts AMASS motion capture data (SMPL format) to stick figure format.
Maps 22 SMPL joints to 5 stick figure lines.

Usage:
    converter = AMASSConverter()
    motion_tensor = converter.convert_sequence('path/to/amass_file.npz')
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator


class AMASSConverter:
    """Convert AMASS SMPL data to stick figure format"""

    # Map SMPL 22 joints to stick figure components
    # SMPL joint indices: https://github.com/vchoutas/smplx
    SMPL_TO_STICK_MAPPING = {
        'head': 15,          # SMPL head joint
        'left_shoulder': 16,  # Left shoulder
        'right_shoulder': 17, # Right shoulder
        'left_hip': 1,        # Left hip
        'right_hip': 2,       # Right hip
        'left_hand': 20,      # Left wrist/hand
        'right_hand': 21,     # Right wrist/hand
        'left_foot': 7,       # Left ankle/foot
        'right_foot': 8       # Right ankle/foot
    }

    def __init__(self, smpl_model_path: str = 'data/smpl_models'):
        """
        Initialize AMASS converter

        Args:
            smpl_model_path: Path to SMPL model files
        """
        self.smpl_model_path = smpl_model_path
        self.smpl_model = None
        self.smplx_model = None
        self.current_model_type = None

    def _detect_format(self, npz_path: str) -> str:
        """
        Detect AMASS file format (SMPL+H or SMPL-X)

        Args:
            npz_path: Path to AMASS .npz file

        Returns:
            'smplh' or 'smplx'
        """
        data = np.load(npz_path, allow_pickle=True)
        poses_shape = data['poses'].shape
        pose_params = poses_shape[1] if len(poses_shape) > 1 else 0

        if pose_params == 156:
            return 'smplh'
        elif pose_params == 162:
            return 'smplx'
        elif pose_params == 165:
            # SMPL-X with separate components (root_orient + pose_body + pose_hand + pose_jaw + pose_eye)
            # 3 + 63 + 90 + 3 + 6 = 165
            return 'smplx'
        elif pose_params == 72:
            return 'smpl'
        else:
            raise ValueError(f"Unknown AMASS format: {pose_params} pose parameters")

    def _load_smpl_model(self, model_type: str = 'smplh'):
        """
        Lazy load SMPL model (requires smplx library)

        Args:
            model_type: 'smplh' or 'smplx'
        """
        # Check if we already have the right model loaded
        if model_type == 'smplh' and self.smpl_model is not None:
            return
        if model_type == 'smplx' and self.smplx_model is not None:
            return

        try:
            import smplx

            if model_type == 'smplh':
                # Try neutral first, fall back to male if not available
                import os
                neutral_path = os.path.join(self.smpl_model_path, 'smplh', 'SMPLH_NEUTRAL.pkl')
                gender = 'neutral' if os.path.exists(neutral_path) else 'male'

                self.smpl_model = smplx.create(
                    self.smpl_model_path,
                    model_type='smplh',
                    gender=gender,
                    use_face_contour=False,
                    use_pca=False
                )
                self.current_model_type = 'smplh'
                print(f"✓ SMPL+H model loaded successfully (gender: {gender})")
            elif model_type == 'smplx':
                # Try neutral first, fall back to male if not available
                import os
                neutral_path = os.path.join(self.smpl_model_path, 'smplx', 'SMPLX_NEUTRAL.npz')
                neutral_pkl_path = os.path.join(self.smpl_model_path, 'smplx', 'SMPLX_NEUTRAL.pkl')

                if os.path.exists(neutral_path) or os.path.exists(neutral_pkl_path):
                    gender = 'neutral'
                else:
                    gender = 'male'

                self.smplx_model = smplx.create(
                    self.smpl_model_path,
                    model_type='smplx',
                    gender=gender,
                    use_face_contour=False,
                    use_pca=False,
                    num_betas=10
                )
                self.current_model_type = 'smplx'
                print(f"✓ SMPL-X model loaded successfully (gender: {gender})")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except ImportError as exc:
            raise ImportError(
                "smplx library not installed. Run: pip install smplx"
            ) from exc
        except Exception as e:
            raise RuntimeError(f"Failed to load SMPL model: {e}") from e

    def load_amass_sequence(self, npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Load AMASS .npz file and detect format

        Args:
            npz_path: Path to AMASS .npz file

        Returns:
            poses: [num_frames, 156 or 162 or 165] - SMPL pose parameters
            trans: [num_frames, 3] - Global translation
            betas: [10 or 16] - Body shape parameters
            model_type: 'smplh' or 'smplx'
        """
        data = np.load(npz_path, allow_pickle=True)

        # Detect format
        model_type = self._detect_format(npz_path)

        # Extract pose parameters
        poses = data['poses']  # [num_frames, 156 or 162 or 165]
        trans = data['trans']  # [num_frames, 3] - global translation
        betas = data['betas'] if 'betas' in data else np.zeros(10)  # Body shape

        return poses, trans, betas, model_type

    def smpl_to_stick_figure(self, smpl_joints: np.ndarray) -> np.ndarray:
        """
        Convert SMPL 22 joints to stick figure 5 lines

        Args:
            smpl_joints: [num_frames, 22, 3] - SMPL joint positions (x, y, z)

        Returns:
            stick_lines: [num_frames, 5, 4] - 5 lines (x1, y1, x2, y2)
        """
        num_frames = smpl_joints.shape[0]
        stick_lines = np.zeros((num_frames, 5, 4))

        for f in range(num_frames):
            joints = smpl_joints[f]  # [22, 3]

            # Extract 2D projection (x, y) - ignore z depth
            head = joints[self.SMPL_TO_STICK_MAPPING['head'], :2]
            l_shoulder = joints[self.SMPL_TO_STICK_MAPPING['left_shoulder'], :2]
            r_shoulder = joints[self.SMPL_TO_STICK_MAPPING['right_shoulder'], :2]
            l_hip = joints[self.SMPL_TO_STICK_MAPPING['left_hip'], :2]
            r_hip = joints[self.SMPL_TO_STICK_MAPPING['right_hip'], :2]
            l_hand = joints[self.SMPL_TO_STICK_MAPPING['left_hand'], :2]
            r_hand = joints[self.SMPL_TO_STICK_MAPPING['right_hand'], :2]
            l_foot = joints[self.SMPL_TO_STICK_MAPPING['left_foot'], :2]
            r_foot = joints[self.SMPL_TO_STICK_MAPPING['right_foot'], :2]

            # Compute stick figure center points
            torso_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2

            # Define 5 lines (matching stick-gen schema)
            stick_lines[f, 0] = [*head, *torso_center]      # Line 0: Head to torso
            stick_lines[f, 1] = [*torso_center, *l_hand]    # Line 1: Left arm
            stick_lines[f, 2] = [*torso_center, *r_hand]    # Line 2: Right arm
            stick_lines[f, 3] = [*hip_center, *l_foot]      # Line 3: Left leg
            stick_lines[f, 4] = [*hip_center, *r_foot]      # Line 4: Right leg

        return stick_lines

    def convert_sequence(
        self,
        npz_path: str,
        target_fps: int = 25,
        target_duration: float = 10.0
    ) -> torch.Tensor:
        """
        Convert AMASS sequence to stick figure format
        Automatically detects and handles both SMPL+H and SMPL-X formats

        Args:
            npz_path: Path to AMASS .npz file
            target_fps: Target frames per second (default: 25)
            target_duration: Target duration in seconds (default: 10.0)

        Returns:
            motion_tensor: [250, 20] - stick figure motion
        """
        # Load AMASS data and detect format
        poses, trans, betas, model_type = self.load_amass_sequence(npz_path)

        # Load appropriate SMPL model
        self._load_smpl_model(model_type)

        # Select the right model
        model = self.smpl_model if model_type == 'smplh' else self.smplx_model

        # Extract pose parameters based on format
        pose_params = poses.shape[1]

        if model_type == 'smplh':
            # SMPL+H: 156 params (3 global + 63 body + 90 hands)
            body_pose = torch.tensor(poses[:, 3:66], dtype=torch.float32)  # Body pose (21 joints × 3)
            global_orient = torch.tensor(poses[:, :3], dtype=torch.float32)  # Root orientation
            left_hand_pose = torch.tensor(poses[:, 66:111], dtype=torch.float32)  # Left hand (15 joints × 3)
            right_hand_pose = torch.tensor(poses[:, 111:156], dtype=torch.float32)  # Right hand (15 joints × 3)
        elif model_type == 'smplx':
            if pose_params == 165:
                # SMPL-X: 165 params (3 global + 63 body + 90 hands + 3 jaw + 6 eyes)
                # For stick figures, we only need body and hand poses
                body_pose = torch.tensor(poses[:, 3:66], dtype=torch.float32)  # Body pose (21 joints × 3)
                global_orient = torch.tensor(poses[:, :3], dtype=torch.float32)  # Root orientation
                left_hand_pose = torch.tensor(poses[:, 66:111], dtype=torch.float32)  # Left hand (15 joints × 3)
                right_hand_pose = torch.tensor(poses[:, 111:156], dtype=torch.float32)  # Right hand (15 joints × 3)
                # Ignore jaw (156:159) and eyes (159:165) for now - will be used in Phase 5-7
            elif pose_params == 162:
                # SMPL-X: 162 params (3 global + 63 body + 90 hands + 3 jaw + 3 leye + 3 reye)
                body_pose = torch.tensor(poses[:, 3:66], dtype=torch.float32)  # Body pose (21 joints × 3)
                global_orient = torch.tensor(poses[:, :3], dtype=torch.float32)  # Root orientation
                left_hand_pose = torch.tensor(poses[:, 66:111], dtype=torch.float32)  # Left hand (15 joints × 3)
                right_hand_pose = torch.tensor(poses[:, 111:156], dtype=torch.float32)  # Right hand (15 joints × 3)
                # Ignore jaw (156:159), left eye (159:162)
            else:
                raise ValueError(f"Unexpected SMPL-X parameter count: {pose_params}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        transl = torch.tensor(trans, dtype=torch.float32)

        # Body shape parameters - use only first 10 for SMPL+H
        betas_tensor = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)  # [1, 10]

        # Process in batches to avoid memory issues
        batch_size = 64
        num_frames = poses.shape[0]
        all_joints = []

        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                batch_body_pose = body_pose[i:end_idx]
                batch_global_orient = global_orient[i:end_idx]
                batch_left_hand = left_hand_pose[i:end_idx]
                batch_right_hand = right_hand_pose[i:end_idx]
                batch_transl = transl[i:end_idx]

                # Repeat betas for batch
                batch_betas = betas_tensor.repeat(end_idx - i, 1)

                if model_type == 'smplh':
                    output = model(
                        body_pose=batch_body_pose,
                        global_orient=batch_global_orient,
                        transl=batch_transl,
                        left_hand_pose=batch_left_hand,
                        right_hand_pose=batch_right_hand,
                        betas=batch_betas
                    )
                else:  # smplx
                    output = model(
                        body_pose=batch_body_pose,
                        global_orient=batch_global_orient,
                        transl=batch_transl,
                        left_hand_pose=batch_left_hand,
                        right_hand_pose=batch_right_hand,
                        betas=batch_betas
                    )

                all_joints.append(output.joints.numpy())

        smpl_joints = np.concatenate(all_joints, axis=0)  # [num_frames, 73, 3]

        # Extract only the first 22 joints (body joints) for both formats
        # SMPL+H has 52 joints (22 body + 30 hands)
        # SMPL-X has 54 joints (22 body + 30 hands + 2 jaw)
        smpl_joints = smpl_joints[:, :22, :]  # [num_frames, 22, 3]

        # Convert to stick figure
        stick_lines = self.smpl_to_stick_figure(smpl_joints)

        # Resample to target FPS and duration
        target_frames = int(target_fps * target_duration)  # 250 frames
        current_frames = stick_lines.shape[0]

        if current_frames != target_frames:
            # Resample using linear interpolation
            indices = np.linspace(0, current_frames - 1, target_frames)
            stick_lines_resampled = np.zeros((target_frames, 5, 4))

            for i in range(5):
                for j in range(4):
                    stick_lines_resampled[:, i, j] = np.interp(
                        indices,
                        np.arange(current_frames),
                        stick_lines[:, i, j]
                    )

            stick_lines = stick_lines_resampled

        # Flatten to [250, 20]
        motion_tensor = torch.tensor(
            stick_lines.reshape(target_frames, 20),
            dtype=torch.float32
        )

        return motion_tensor


# AMASS category to ActionType mapping
AMASS_ACTION_MAPPING = {
    # Locomotion
    'walk': ActionType.WALK,
    'run': ActionType.RUN,
    'jog': ActionType.RUN,
    'sprint': ActionType.SPRINT,
    'jump': ActionType.JUMP,
    'hop': ActionType.JUMP,
    'leap': ActionType.JUMP,

    # Sports
    'kick': ActionType.KICKING,
    'throw': ActionType.THROWING,
    'catch': ActionType.CATCHING,
    'punch': ActionType.PUNCH,
    'basketball': ActionType.SHOOTING,
    'soccer': ActionType.KICKING,
    'baseball': ActionType.THROWING,

    # Social gestures
    'wave': ActionType.WAVE,
    'clap': ActionType.CLAP,
    'point': ActionType.POINT,
    'greet': ActionType.WAVE,
    'hello': ActionType.WAVE,

    # Postures
    'sit': ActionType.SIT,
    'stand': ActionType.STAND,
    'kneel': ActionType.KNEEL,
    'lie': ActionType.LIE_DOWN,
    'crouch': ActionType.KNEEL,  # Map to KNEEL (closest available)

    # Emotional/expressive
    'dance': ActionType.DANCE,
    'celebrate': ActionType.CELEBRATE,
    'cheer': ActionType.CELEBRATE,  # Map to CELEBRATE (closest available)

    # Combat
    'fight': ActionType.PUNCH,
    'dodge': ActionType.DODGE,
    'block': ActionType.DODGE,  # Map to DODGE (closest available)

    # Default
    'idle': ActionType.IDLE,
    'neutral': ActionType.IDLE,
    'rest': ActionType.IDLE,
}


def infer_action_from_filename(npz_path: str) -> ActionType:
    """
    Infer action type from AMASS filename

    Args:
        npz_path: Path to AMASS .npz file

    Returns:
        ActionType enum

    Example:
        'CMU/01/01_01_walk.npz' → ActionType.WALK
        'BMLmovi/Subject_1_F_MoSh/Subject_1_F_1_poses.npz' → ActionType.IDLE
    """
    filename = Path(npz_path).stem.lower()

    # Check for action keywords in filename
    for keyword, action in AMASS_ACTION_MAPPING.items():
        if keyword in filename:
            return action

    # Default to IDLE if no match
    return ActionType.IDLE


def generate_description_from_action(action: ActionType) -> str:
    """
    Generate natural language description from action type

    Args:
        action: ActionType enum

    Returns:
        Natural language description
    """
    import random

    DESCRIPTION_TEMPLATES = {
        ActionType.WALK: [
            "A person walking",
            "Someone walking forward",
            "A figure walking casually",
            "Walking motion",
        ],
        ActionType.RUN: [
            "A person running",
            "Someone running fast",
            "A figure sprinting",
            "Running motion",
        ],
        ActionType.JUMP: [
            "A person jumping",
            "Someone jumping up",
            "A figure leaping",
            "Jumping motion",
        ],
        ActionType.DANCE: [
            "A person dancing",
            "Someone dancing energetically",
            "A figure performing dance moves",
            "Dancing motion",
        ],
        ActionType.WAVE: [
            "A person waving",
            "Someone waving hello",
            "A figure waving their hand",
            "Waving gesture",
        ],
        # Add more templates as needed
    }

    templates = DESCRIPTION_TEMPLATES.get(action, [f"A person performing {action.value}"])
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Canonical dataset export helpers
# ---------------------------------------------------------------------------


def compute_basic_physics(motion: torch.Tensor, fps: int = 25) -> torch.Tensor:
    """Compute simple physics features from stick-figure motion.

    Supports both single-actor ``[T, 20]`` and multi-actor ``[T, A, 20]`` shapes.
    We approximate each actor's position as the mean of all joint endpoints
    (10 points from 5 line segments) and derive velocity/acceleration.
    """

    if motion.ndim == 2:
        # [T, 20] -> [T, 1, 20] for unified handling
        motion_reshaped = motion.unsqueeze(1)
        single_actor = True
    elif motion.ndim == 3 and motion.shape[2] == 20:
        # [T, A, 20]
        motion_reshaped = motion
        single_actor = False
    else:
        raise ValueError(
            f"Expected motion shape [T, 20] or [T, A, 20], got {tuple(motion.shape)}"
        )

    T, A, _ = motion_reshaped.shape
    coords = motion_reshaped.view(T, A, 10, 2)  # endpoints (x, y)
    pos = coords.mean(dim=2)  # [T, A, 2]

    vel = torch.zeros_like(pos)
    acc = torch.zeros_like(pos)

    if T > 1:
        vel[1:] = (pos[1:] - pos[:-1]) * fps
    if T > 2:
        acc[1:] = (vel[1:] - vel[:-1]) * fps

    # Assume unit mass for now; momentum = velocity
    mx = vel[..., 0]
    my = vel[..., 1]

    physics = torch.stack(
        [
            vel[..., 0],
            vel[..., 1],
            acc[..., 0],
            acc[..., 1],
            mx,
            my,
        ],
        dim=-1,
    )  # [T, A, 6]

    if single_actor:
        # squeeze actor dimension back out -> [T, 6]
        physics = physics.squeeze(1)

    return physics


def build_canonical_sample(
    motion: torch.Tensor,
    npz_path: str,
    fps: int = 25,
) -> dict[str, Any]:
    """Build a canonical sample dict for a single AMASS sequence.

    The output matches the format documented in docs/features/DATA_SCHEMA.md
    and is suitable for validation + training dataset construction.
    """

    action = infer_action_from_filename(npz_path)
    desc = generate_description_from_action(action)
    physics = compute_basic_physics(motion, fps=fps)

    T = motion.shape[0]
    action_idx = ACTION_TO_IDX[action]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    rel_path = os.path.relpath(npz_path, start=os.getcwd())

    sample: dict[str, Any] = {
        "description": desc,
        "motion": motion,  # [T, 20]
        "actions": actions,  # [T]
        "physics": physics,  # [T, 6]
        "camera": torch.zeros(T, 3),  # [T, 3] (no camera in AMASS)
        "source": "amass",
        "meta": {
            "file": rel_path,
        },
    }

    return sample


def convert_amass_dataset(
    amass_root: str = "data/amass",
    output_path: str = "data/motions_processed/amass/canonical.pt",
    smpl_model_path: str = "data/smpl_models",
    target_fps: int = 25,
    target_duration: float = 10.0,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    """Convert AMASS dataset into canonical schema and save to disk.

    Returns the list of valid samples that were written.
    """

    converter = AMASSConverter(smpl_model_path=smpl_model_path)
    validator = DataValidator(fps=target_fps)

    amass_root_path = Path(amass_root)
    if not amass_root_path.exists():
        raise FileNotFoundError(f"AMASS root not found: {amass_root}")

    npz_files = sorted(str(p) for p in amass_root_path.rglob("*.npz"))
    if max_files is not None:
        npz_files = npz_files[: max_files]

    samples: list[dict[str, Any]] = []
    num_total = 0
    num_valid = 0

    for npz_path in npz_files:
        num_total += 1
        try:
            motion = converter.convert_sequence(
                npz_path,
                target_fps=target_fps,
                target_duration=target_duration,
            )  # [T, 20]

            sample = build_canonical_sample(motion, npz_path, fps=target_fps)
            # For raw dataset conversion we enforce physics sanity checks but
            # allow skeleton variability (2D projection + dataset noise).
            is_valid, score, reason = validator.check_physics_consistency(
                sample["physics"]
            )

            if not is_valid:
                print(f"[AMASS] Skipping invalid sequence {npz_path}: {reason}")
                continue

            samples.append(sample)
            num_valid += 1
        except Exception as e:
            print(f"[AMASS] Error processing {npz_path}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)

    print(f"[AMASS] Saved {num_valid}/{num_total} valid samples to {output_path}")

    return samples


def main() -> None:
    """CLI entrypoint for converting AMASS to canonical format."""

    import argparse

    parser = argparse.ArgumentParser(description="Convert AMASS to canonical schema")
    parser.add_argument(
        "--amass-root",
        type=str,
        default="data/amass",
        help="Root directory of AMASS .npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/motions_processed/amass/canonical.pt",
        help="Output path for canonical dataset",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default="data/smpl_models",
        help="Path to SMPL/SMPL-X model files",
    )
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument("--target-duration", type=float, default=10.0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to process (for testing)",
    )

    args = parser.parse_args()

    convert_amass_dataset(
        amass_root=args.amass_root,
        output_path=args.output,
        smpl_model_path=args.smpl_model_path,
        target_fps=args.target_fps,
        target_duration=args.target_duration,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
