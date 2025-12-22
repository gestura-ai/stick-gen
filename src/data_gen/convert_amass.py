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

from .metadata_extractors import build_enhanced_metadata
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator

# Import centralized paths config
try:
    from ..config.paths import get_path
    DEFAULT_OUTPUT_PATH = str(get_path("amass_canonical"))
except ImportError:
    # Fallback for backward compatibility
    DEFAULT_OUTPUT_PATH = "data/motions_processed/amass/canonical.pt"


class AMASSConverter:
    """Convert AMASS SMPL data to stick figure format"""

    # Map SMPL 22 joints to stick figure components
    # SMPL joint indices: https://github.com/vchoutas/smplx
    SMPL_TO_STICK_MAPPING = {
        "head": 15,  # SMPL head joint
        "left_shoulder": 16,  # Left shoulder
        "right_shoulder": 17,  # Right shoulder
        "left_hip": 1,  # Left hip
        "right_hip": 2,  # Right hip
        "left_hand": 20,  # Left wrist/hand
        "right_hand": 21,  # Right wrist/hand
        "left_foot": 7,  # Left ankle/foot
        "right_foot": 8,  # Right ankle/foot
    }

    def __init__(self, smpl_model_path: str = "data/smpl_models"):
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
        poses_shape = data["poses"].shape
        pose_params = poses_shape[1] if len(poses_shape) > 1 else 0

        if pose_params == 156:
            return "smplh"
        elif pose_params == 162:
            return "smplx"
        elif pose_params == 165:
            # SMPL-X with separate components (root_orient + pose_body + pose_hand + pose_jaw + pose_eye)
            # 3 + 63 + 90 + 3 + 6 = 165
            return "smplx"
        elif pose_params == 72:
            return "smpl"
        else:
            raise ValueError(f"Unknown AMASS format: {pose_params} pose parameters")

    def _load_smpl_model(self, model_type: str = "smplh"):
        """
        Lazy load SMPL model (requires smplx library)

        Args:
            model_type: 'smplh' or 'smplx'
        """
        # Check if we already have the right model loaded
        if model_type == "smplh" and self.smpl_model is not None:
            return
        if model_type == "smplx" and self.smplx_model is not None:
            return

        try:
            import smplx

            if model_type == "smplh":
                # Try neutral first, fall back to male if not available
                neutral_path = os.path.join(
                    self.smpl_model_path, "smplh", "SMPLH_NEUTRAL.pkl"
                )
                gender = "neutral" if os.path.exists(neutral_path) else "male"

                self.smpl_model = smplx.create(
                    self.smpl_model_path,
                    model_type="smplh",
                    gender=gender,
                    use_face_contour=False,
                    use_pca=False,
                )
                self.current_model_type = "smplh"
                print(f"✓ SMPL+H model loaded successfully (gender: {gender})")
            elif model_type == "smplx":
                # Try neutral first, fall back to male if not available
                neutral_path = os.path.join(
                    self.smpl_model_path, "smplx", "SMPLX_NEUTRAL.npz"
                )
                neutral_pkl_path = os.path.join(
                    self.smpl_model_path, "smplx", "SMPLX_NEUTRAL.pkl"
                )

                if os.path.exists(neutral_path) or os.path.exists(neutral_pkl_path):
                    gender = "neutral"
                else:
                    gender = "male"

                self.smplx_model = smplx.create(
                    self.smpl_model_path,
                    model_type="smplx",
                    gender=gender,
                    use_face_contour=False,
                    use_pca=False,
                    num_betas=10,
                    num_expression_coeffs=10,
                )
                self.current_model_type = "smplx"
                print(f"✓ SMPL-X model loaded successfully (gender: {gender})")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except ImportError as exc:
            raise ImportError(
                "smplx library not installed. Run: pip install smplx"
            ) from exc
        except Exception as e:
            raise RuntimeError(f"Failed to load SMPL model: {e}") from e

    def _check_pose_stability(
        self,
        poses: np.ndarray,
        trans: np.ndarray,
        fps: float = 120.0,
        threshold_multiplier: float = 1.0,
    ) -> tuple[bool, float]:
        """
        Early detection of unstable/noisy motion data.

        Computes frame-to-frame pose acceleration in axis-angle space to detect
        sequences with extreme noise or corruption. This is a fast heuristic check
        to skip problematic files early before expensive SMPL processing.

        Thresholds are calibrated based on empirical analysis of AMASS data at 120fps:
        - Translation acceleration: median ~37 m/s², p99 ~580 m/s²
        - Pose acceleration: median ~3500 rad/s², p99 ~15500 rad/s²

        Args:
            poses: Raw pose array [N, params] in axis-angle representation
            trans: Translation array [N, 3]
            fps: Frame rate of the source data (AMASS is typically 120fps)
            threshold_multiplier: Multiplier for thresholds (higher = more permissive)

        Returns:
            (is_stable, max_metric): Stability check result and max acceleration metric
        """
        if poses.shape[0] < 3:
            return True, 0.0

        dt = 1.0 / fps

        # Check translation acceleration
        trans_vel = np.diff(trans, axis=0) / dt  # [N-1, 3]
        trans_acc = np.diff(trans_vel, axis=0) / dt  # [N-2, 3]
        trans_acc_mag = np.linalg.norm(trans_acc, axis=1)  # [N-2]
        max_trans_acc = float(np.max(trans_acc_mag)) if len(trans_acc_mag) > 0 else 0.0

        # Check pose acceleration (axis-angle space)
        # Focus on body pose (indices 3:66) - most indicative of motion quality
        body_poses = poses[:, 3:66] if poses.shape[1] >= 66 else poses[:, 3:]
        pose_vel = np.diff(body_poses, axis=0) / dt
        pose_acc = np.diff(pose_vel, axis=0) / dt
        pose_acc_mag = np.linalg.norm(pose_acc, axis=1)
        max_pose_acc = float(np.max(pose_acc_mag)) if len(pose_acc_mag) > 0 else 0.0

        # Base thresholds: ~3x the p99 values to catch only extreme outliers
        # These allow 99%+ of valid AMASS files through while catching corrupt data
        # Translation: 1500 m/s² (3x the p99 of ~580)
        # Pose: 50000 rad/s² (3x the p99 of ~15500)
        base_trans_threshold = 1500.0
        base_pose_threshold = 50000.0

        trans_threshold = base_trans_threshold * threshold_multiplier
        pose_threshold = base_pose_threshold * threshold_multiplier

        is_stable = max_trans_acc < trans_threshold and max_pose_acc < pose_threshold

        # Return a normalized metric for reporting (trans_acc + pose_acc/30 roughly same scale)
        return is_stable, max(max_trans_acc, max_pose_acc / 30.0)

    def load_amass_sequence(
        self,
        npz_path: str,
        check_stability: bool = False,
        stability_threshold: float = 1.0,
    ) -> tuple[dict[str, np.ndarray], str]:
        """
        Load AMASS .npz file and detect format

        Args:
            npz_path: Path to AMASS .npz file
            check_stability: If True, perform early stability check and raise
                ValueError for unstable sequences
            stability_threshold: Multiplier for stability thresholds (higher = more permissive)

        Returns:
            data: Dict with pose components (root_orient, pose_body, pose_hand, trans, betas, etc.)
            model_type: 'smplh' or 'smplx'
        """
        raw_data = np.load(npz_path, allow_pickle=True)

        # Detect format
        model_type = self._detect_format(npz_path)

        # Early stability check on raw pose data (before expensive SMPL processing)
        if check_stability and "poses" in raw_data and "trans" in raw_data:
            is_stable, max_val = self._check_pose_stability(
                raw_data["poses"],
                raw_data["trans"],
                threshold_multiplier=stability_threshold,
            )
            if not is_stable:
                raise ValueError(f"Unstable motion detected (early check): {max_val:.1f}")

        # Extract pose components - prefer explicit fields over slicing poses array
        # Different AMASS subsets have different poses array layouts, but explicit
        # fields (when present) are always consistent
        data: dict[str, np.ndarray] = {}

        data["trans"] = raw_data["trans"]
        data["betas"] = raw_data["betas"] if "betas" in raw_data else np.zeros(10)

        # Check for explicit pose component fields (SMPL-X stage II files)
        if "root_orient" in raw_data and "pose_body" in raw_data:
            data["root_orient"] = raw_data["root_orient"]
            data["pose_body"] = raw_data["pose_body"]

            # Hand poses - may be combined or separate
            if "pose_hand" in raw_data:
                pose_hand = raw_data["pose_hand"]
                data["left_hand_pose"] = pose_hand[:, :45]
                data["right_hand_pose"] = pose_hand[:, 45:]
            elif "left_hand_pose" in raw_data:
                data["left_hand_pose"] = raw_data["left_hand_pose"]
                data["right_hand_pose"] = raw_data["right_hand_pose"]
            else:
                # No hand data - use zeros
                num_frames = data["root_orient"].shape[0]
                data["left_hand_pose"] = np.zeros((num_frames, 45))
                data["right_hand_pose"] = np.zeros((num_frames, 45))
        else:
            # Fall back to slicing the poses array (SMPL+H and some SMPL-X files)
            poses = raw_data["poses"]
            pose_params = poses.shape[1]

            data["root_orient"] = poses[:, :3]
            data["pose_body"] = poses[:, 3:66]

            if model_type == "smplh" and pose_params == 156:
                # SMPL+H: 156 params (3 global + 63 body + 45 left + 45 right)
                data["left_hand_pose"] = poses[:, 66:111]
                data["right_hand_pose"] = poses[:, 111:156]
            elif pose_params >= 156:
                # SMPL-X: hands at 66:156 (90 params)
                data["left_hand_pose"] = poses[:, 66:111]
                data["right_hand_pose"] = poses[:, 111:156]
            else:
                # Insufficient pose data - use zeros for hands
                num_frames = poses.shape[0]
                data["left_hand_pose"] = np.zeros((num_frames, 45))
                data["right_hand_pose"] = np.zeros((num_frames, 45))

        return data, model_type

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
            head = joints[self.SMPL_TO_STICK_MAPPING["head"], :2]
            l_shoulder = joints[self.SMPL_TO_STICK_MAPPING["left_shoulder"], :2]
            r_shoulder = joints[self.SMPL_TO_STICK_MAPPING["right_shoulder"], :2]
            l_hip = joints[self.SMPL_TO_STICK_MAPPING["left_hip"], :2]
            r_hip = joints[self.SMPL_TO_STICK_MAPPING["right_hip"], :2]
            l_hand = joints[self.SMPL_TO_STICK_MAPPING["left_hand"], :2]
            r_hand = joints[self.SMPL_TO_STICK_MAPPING["right_hand"], :2]
            l_foot = joints[self.SMPL_TO_STICK_MAPPING["left_foot"], :2]
            r_foot = joints[self.SMPL_TO_STICK_MAPPING["right_foot"], :2]

            # Compute stick figure center points
            torso_center = (l_shoulder + r_shoulder) / 2
            hip_center = (l_hip + r_hip) / 2

            # Define 5 lines (matching stick-gen schema)
            stick_lines[f, 0] = [*head, *torso_center]  # Line 0: Head to torso
            stick_lines[f, 1] = [*torso_center, *l_hand]  # Line 1: Left arm
            stick_lines[f, 2] = [*torso_center, *r_hand]  # Line 2: Right arm
            stick_lines[f, 3] = [*hip_center, *l_foot]  # Line 3: Left leg
            stick_lines[f, 4] = [*hip_center, *r_foot]  # Line 4: Right leg

        return stick_lines

    def convert_sequence(
        self,
        npz_path: str,
        target_fps: int = 25,
        target_duration: float = 10.0,
        check_stability: bool = False,
        stability_threshold: float = 1.0,
    ) -> torch.Tensor:
        """
        Convert AMASS sequence to stick figure format
        Automatically detects and handles both SMPL+H and SMPL-X formats

        Args:
            npz_path: Path to AMASS .npz file
            target_fps: Target frames per second (default: 25)
            target_duration: Target duration in seconds (default: 10.0)
            check_stability: If True, perform early stability check on raw pose
                data before expensive SMPL processing
            stability_threshold: Multiplier for stability thresholds (higher = more permissive)

        Returns:
            motion_tensor: [250, 20] - stick figure motion
        """
        # Load AMASS data and detect format (with optional early stability check)
        data, model_type = self.load_amass_sequence(
            npz_path,
            check_stability=check_stability,
            stability_threshold=stability_threshold,
        )

        # Load appropriate SMPL model
        self._load_smpl_model(model_type)

        # Select the right model
        model = self.smpl_model if model_type == "smplh" else self.smplx_model

        # Extract pose components from pre-parsed data dict
        global_orient = torch.tensor(data["root_orient"], dtype=torch.float32)
        body_pose = torch.tensor(data["pose_body"], dtype=torch.float32)
        left_hand_pose = torch.tensor(data["left_hand_pose"], dtype=torch.float32)
        right_hand_pose = torch.tensor(data["right_hand_pose"], dtype=torch.float32)
        transl = torch.tensor(data["trans"], dtype=torch.float32)

        # Body shape parameters - use only first 10
        betas = data["betas"]
        betas_tensor = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(
            0
        )  # [1, 10]

        # Process in batches to avoid memory issues
        batch_size = 64
        num_frames = global_orient.shape[0]
        all_joints = []

        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                current_batch_size = end_idx - i
                batch_body_pose = body_pose[i:end_idx]
                batch_global_orient = global_orient[i:end_idx]
                batch_left_hand = left_hand_pose[i:end_idx]
                batch_right_hand = right_hand_pose[i:end_idx]
                batch_transl = transl[i:end_idx]

                # Repeat betas for batch
                batch_betas = betas_tensor.repeat(current_batch_size, 1)

                if model_type == "smplh":
                    output = model(
                        body_pose=batch_body_pose,
                        global_orient=batch_global_orient,
                        transl=batch_transl,
                        left_hand_pose=batch_left_hand,
                        right_hand_pose=batch_right_hand,
                        betas=batch_betas,
                    )
                else:  # smplx
                    # SMPL-X requires all pose parameters with matching batch sizes
                    # Create zero tensors for jaw, eyes, and expression
                    batch_jaw_pose = torch.zeros(current_batch_size, 3)
                    batch_leye_pose = torch.zeros(current_batch_size, 3)
                    batch_reye_pose = torch.zeros(current_batch_size, 3)
                    batch_expression = torch.zeros(current_batch_size, 10)

                    output = model(
                        body_pose=batch_body_pose,
                        global_orient=batch_global_orient,
                        transl=batch_transl,
                        left_hand_pose=batch_left_hand,
                        right_hand_pose=batch_right_hand,
                        jaw_pose=batch_jaw_pose,
                        leye_pose=batch_leye_pose,
                        reye_pose=batch_reye_pose,
                        expression=batch_expression,
                        betas=batch_betas,
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
                        indices, np.arange(current_frames), stick_lines[:, i, j]
                    )

            stick_lines = stick_lines_resampled

        # Flatten to [250, 20]
        motion_tensor = torch.tensor(
            stick_lines.reshape(target_frames, 20), dtype=torch.float32
        )

        return motion_tensor


# AMASS category to ActionType mapping
AMASS_ACTION_MAPPING = {
    # Locomotion
    "walk": ActionType.WALK,
    "run": ActionType.RUN,
    "jog": ActionType.RUN,
    "sprint": ActionType.SPRINT,
    "jump": ActionType.JUMP,
    "hop": ActionType.JUMP,
    "leap": ActionType.JUMP,
    # Sports
    "kick": ActionType.KICKING,
    "throw": ActionType.THROWING,
    "catch": ActionType.CATCHING,
    "punch": ActionType.PUNCH,
    "basketball": ActionType.SHOOTING,
    "soccer": ActionType.KICKING,
    "baseball": ActionType.THROWING,
    # Social gestures
    "wave": ActionType.WAVE,
    "clap": ActionType.CLAP,
    "point": ActionType.POINT,
    "greet": ActionType.WAVE,
    "hello": ActionType.WAVE,
    # Postures
    "sit": ActionType.SIT,
    "stand": ActionType.STAND,
    "kneel": ActionType.KNEEL,
    "lie": ActionType.LIE_DOWN,
    "crouch": ActionType.KNEEL,  # Map to KNEEL (closest available)
    # Emotional/expressive
    "dance": ActionType.DANCE,
    "celebrate": ActionType.CELEBRATE,
    "cheer": ActionType.CELEBRATE,  # Map to CELEBRATE (closest available)
    # Combat
    "fight": ActionType.PUNCH,
    "dodge": ActionType.DODGE,
    "block": ActionType.DODGE,  # Map to DODGE (closest available)
    # Default
    "idle": ActionType.IDLE,
    "neutral": ActionType.IDLE,
    "rest": ActionType.IDLE,
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

    templates = DESCRIPTION_TEMPLATES.get(
        action, [f"A person performing {action.value}"]
    )
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
    original_fps: int | None = None,
    original_num_frames: int | None = None,
    betas: np.ndarray | None = None,
    gender: str | None = None,
) -> dict[str, Any]:
    """Build a canonical sample dict for a single AMASS sequence.

    The output matches the format documented in docs/features/DATA_SCHEMA.md
    and is suitable for validation + training dataset construction.

    Args:
        motion: Motion tensor [T, 20]
        npz_path: Path to source .npz file
        fps: Current frame rate
        original_fps: Original source frame rate before resampling
        original_num_frames: Original frame count before processing
        betas: SMPL body shape parameters (if available)
        gender: Subject gender from metadata (if available)

    Returns:
        Canonical sample dict with enhanced metadata
    """
    action = infer_action_from_filename(npz_path)
    desc = generate_description_from_action(action)
    physics = compute_basic_physics(motion, fps=fps)

    T = motion.shape[0]
    action_idx = ACTION_TO_IDX[action]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    rel_path = os.path.relpath(npz_path, start=os.getcwd())

    # Build enhanced metadata
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=fps,
        description=desc,
        original_fps=original_fps,
        original_num_frames=original_num_frames,
        betas=betas,
        gender=gender,
    )

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
        "enhanced_meta": enhanced_meta.model_dump(),
    }

    return sample


def convert_amass_dataset(
    amass_root: str = "data/amass",
    output_path: str | None = None,
    smpl_model_path: str = "data/smpl_models",
    target_fps: int = 25,
    target_duration: float = 10.0,
    max_files: int | None = None,
    physics_threshold: float = 1.5,
    stability_threshold: float = 1.0,
    early_stability_check: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Convert AMASS dataset into canonical schema and save to disk.

    Args:
        amass_root: Root directory of AMASS dataset
        output_path: Output .pt file path (defaults to config path)
        smpl_model_path: Path to SMPL models
        target_fps: Target frame rate for output
        target_duration: Target duration in seconds
        max_files: Maximum number of files to process (None = all)
        physics_threshold: Multiplier for physics validation thresholds.
            Higher values allow more extreme motions (default 1.5 for MoCap data).
        stability_threshold: Multiplier for early stability check thresholds.
            Higher values allow more extreme raw pose accelerations (default 1.0).
        early_stability_check: If True, perform fast stability check on raw pose
            data before expensive SMPL processing (skips noisy files early).
        verbose: If True, print each skipped file. If False, aggregate counts.

    Returns:
        The list of valid samples that were written.
    """

    # Use default from paths config if not specified
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    converter = AMASSConverter(smpl_model_path=smpl_model_path)
    validator = DataValidator(fps=target_fps)

    # Apply physics threshold multiplier (MoCap data can have higher acceleration)
    validator.max_velocity = validator.base_max_velocity * physics_threshold
    validator.max_acceleration = validator.base_max_acceleration * physics_threshold

    amass_root_path = Path(amass_root)
    if not amass_root_path.exists():
        raise FileNotFoundError(f"AMASS root not found: {amass_root}")

    # Filter out metadata-only files that don't contain motion data:
    # - shape.npz: body shape parameters only
    # - neutral_stagei.npz: marker/beta calibration data
    # - *_stagei.npz: stage 1 fitting data (no poses)
    skip_patterns = {"shape.npz"}
    npz_files = sorted(
        str(p)
        for p in amass_root_path.rglob("*.npz")
        if p.name not in skip_patterns and not p.name.endswith("_stagei.npz")
    )
    if max_files is not None:
        npz_files = npz_files[:max_files]

    samples: list[dict[str, Any]] = []
    num_total = 0
    num_valid = 0
    num_early_skipped = 0
    num_physics_skipped = 0
    num_errors = 0
    total_files = len(npz_files)

    print(f"[AMASS] Processing {total_files} files...")

    for idx, npz_path in enumerate(npz_files):
        # Progress logging every 500 files
        if idx % 500 == 0:
            print(f"[AMASS] Processing {idx}/{total_files} ({100*idx/total_files:.1f}%)...")
        num_total += 1
        try:
            # Extract source metadata for enhanced metadata
            raw_data = np.load(npz_path, allow_pickle=True)
            source_fps = int(raw_data.get("mocap_frame_rate", raw_data.get("fps", 120)))
            source_num_frames = raw_data["poses"].shape[0] if "poses" in raw_data else None
            betas = raw_data["betas"] if "betas" in raw_data else None
            gender = str(raw_data["gender"]) if "gender" in raw_data else None

            motion = converter.convert_sequence(
                npz_path,
                target_fps=target_fps,
                target_duration=target_duration,
                check_stability=early_stability_check,
                stability_threshold=stability_threshold,
            )  # [T, 20]

            sample = build_canonical_sample(
                motion,
                npz_path,
                fps=target_fps,
                original_fps=source_fps,
                original_num_frames=source_num_frames,
                betas=betas,
                gender=gender,
            )
            # For raw dataset conversion we enforce physics sanity checks but
            # allow skeleton variability (2D projection + dataset noise).
            is_valid, score, reason = validator.check_physics_consistency(
                sample["physics"]
            )

            if not is_valid:
                num_physics_skipped += 1
                if verbose:
                    print(f"[AMASS] Skipping invalid sequence {npz_path}: {reason}")
                continue

            samples.append(sample)
            num_valid += 1
        except ValueError as e:
            # Early stability check failures
            if "Unstable motion" in str(e):
                num_early_skipped += 1
                if verbose:
                    print(f"[AMASS] Early skip {npz_path}: {e}")
            else:
                num_errors += 1
                if verbose:
                    print(f"[AMASS] Error processing {npz_path}: {e}")
        except Exception as e:
            num_errors += 1
            if verbose:
                print(f"[AMASS] Error processing {npz_path}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)

    # Summary statistics
    print(f"[AMASS] Conversion complete:")
    print(f"  - Valid samples: {num_valid}/{num_total}")
    if num_early_skipped > 0:
        print(f"  - Early stability skips: {num_early_skipped}")
    if num_physics_skipped > 0:
        print(f"  - Physics validation skips: {num_physics_skipped}")
    if num_errors > 0:
        print(f"  - Processing errors: {num_errors}")
    print(f"  - Output: {output_path}")

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
        default=DEFAULT_OUTPUT_PATH,
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
    parser.add_argument(
        "--physics-threshold",
        type=float,
        default=1.5,
        help="Multiplier for physics validation thresholds (higher = more permissive)",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=1.0,
        help="Multiplier for early stability check thresholds (higher = more permissive)",
    )
    parser.add_argument(
        "--no-early-check",
        action="store_true",
        help="Disable early stability check on raw pose data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each skipped file (default: aggregate counts only)",
    )

    args = parser.parse_args()

    convert_amass_dataset(
        amass_root=args.amass_root,
        output_path=args.output,
        smpl_model_path=args.smpl_model_path,
        target_fps=args.target_fps,
        target_duration=args.target_duration,
        max_files=args.max_files,
        physics_threshold=args.physics_threshold,
        stability_threshold=args.stability_threshold,
        early_stability_check=not args.no_early_check,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
