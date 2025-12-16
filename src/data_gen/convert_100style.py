"""
100STYLE Dataset Converter

Converts 100STYLE BVH motion capture files to Stick-Gen format.
100STYLE contains 100+ motion styles captured from professional actors.

Requirements:
    pip install bvh scipy

Joint Mapping:
    BVH Skeleton (Standard) -> Stick Figure (10 joints)
    - Hips -> Hip center
    - Spine/Chest -> Torso
    - Head -> Head
    - LeftShoulder/LeftArm -> Left shoulder/hand
    - RightShoulder/RightArm -> Right shoulder/hand
    - LeftUpLeg/LeftFoot -> Left hip/foot
    - RightUpLeg/RightFoot -> Right hip/foot
"""

import glob
import os

import numpy as np
import torch
from tqdm import tqdm

try:
    from bvh import Bvh
    from scipy.spatial.transform import Rotation as R
    BVH_AVAILABLE = True
except ImportError:
    BVH_AVAILABLE = False
    print("Warning: BVH packages not available. Install with: pip install bvh scipy")
    # Define placeholder for type hints when scipy is not available
    class R:
        """Placeholder for scipy.spatial.transform.Rotation when not installed."""
        @staticmethod
        def identity():
            return None
        @staticmethod
        def from_euler(*args, **kwargs):
            return None



# 100STYLE BVH joint names (standard CMU/Mixamo skeleton)
BVH_JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot"
]

# Mapping from BVH joints to Stick Figure joints (indices)
# Stick figure: [head, neck, hip, l_shoulder, r_shoulder, l_hand, r_hand, l_hip, r_hip, l_foot, r_foot]
# We use 10 joints (5 pairs of x,y coordinates = 20 values)
JOINT_MAPPING = {
    "Head": 0,
    "Neck": 1,
    "Hips": 2,
    "LeftShoulder": 3,
    "RightShoulder": 4,
    "LeftHand": 5,
    "RightHand": 6,
    "LeftUpLeg": 7,  # Hip joint
    "RightUpLeg": 8,
    "LeftFoot": 9,
    "RightFoot": 10  # We'll use 10 joints for full body, flatten to 20 coords
}


class BVHForwardKinematics:
    """Compute forward kinematics from BVH rotations to world positions."""

    def __init__(self, bvh: 'Bvh'):
        self.bvh = bvh
        self.joint_tree = self._build_joint_tree()

    def _build_joint_tree(self) -> dict[str, dict]:
        """Build hierarchical joint tree from BVH."""
        tree = {}
        for joint in self.bvh.get_joints():
            name = joint.name
            parent = self.bvh.joint_parent(name)
            offset = np.array(self.bvh.joint_offset(name))
            tree[name] = {
                "parent": parent.name if parent else None,
                "offset": offset,
                "children": []
            }

        # Build children lists
        for name, data in tree.items():
            parent = data["parent"]
            if parent and parent in tree:
                tree[parent]["children"].append(name)

        return tree

    def compute_frame_positions(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Compute world positions for all joints at a given frame."""
        positions = {}

        def compute_recursive(joint_name: str, parent_pos: np.ndarray, parent_rot: np.ndarray):
            joint = self.joint_tree[joint_name]

            # Get local offset
            offset = joint["offset"]

            # Get rotation channels for this joint at this frame
            rotation = self._get_joint_rotation(joint_name, frame_idx)

            # Apply parent rotation to offset
            rotated_offset = parent_rot.apply(offset)

            # Compute world position
            world_pos = parent_pos + rotated_offset
            positions[joint_name] = world_pos

            # Combine rotations
            combined_rot = parent_rot * rotation

            # Recurse to children
            for child in joint["children"]:
                compute_recursive(child, world_pos, combined_rot)

        # Start from root (usually "Hips")
        root_name = self.bvh.get_joint(self.bvh.get_joints_names()[0]).name
        root_pos = self._get_root_position(frame_idx)
        root_rot = self._get_joint_rotation(root_name, frame_idx)

        positions[root_name] = root_pos

        for child in self.joint_tree[root_name]["children"]:
            compute_recursive(child, root_pos, root_rot)

        return positions

    def _get_root_position(self, frame_idx: int) -> np.ndarray:
        """Get root joint position (translation channels)."""
        root = self.bvh.get_joints_names()[0]
        try:
            x = self.bvh.frame_joint_channel(frame_idx, root, "Xposition", 0)
            y = self.bvh.frame_joint_channel(frame_idx, root, "Yposition", 0)
            z = self.bvh.frame_joint_channel(frame_idx, root, "Zposition", 0)
            return np.array([x, y, z])
        except Exception:
            return np.zeros(3)

    def _get_joint_rotation(self, joint_name: str, frame_idx: int) -> R:
        """Get rotation for a joint at a frame."""
        try:
            # Try to get Euler angles (most common in BVH)
            rx = self.bvh.frame_joint_channel(frame_idx, joint_name, "Xrotation", 0)
            ry = self.bvh.frame_joint_channel(frame_idx, joint_name, "Yrotation", 0)
            rz = self.bvh.frame_joint_channel(frame_idx, joint_name, "Zrotation", 0)

            # BVH typically uses ZXY order
            return R.from_euler('ZXY', [rz, rx, ry], degrees=True)
        except Exception:
            return R.identity()


def extract_stick_figure_pose(positions: dict[str, np.ndarray], scale: float = 0.01) -> np.ndarray:
    """
    Extract stick figure joint positions from full skeleton.

    Returns:
        np.ndarray: Shape (10, 2) for 10 joints × (x, y) coordinates
    """
    pose = np.zeros((10, 2))

    # Map joints (use first 10 that are available)
    joint_order = ["Head", "Neck", "Hips", "LeftShoulder", "RightShoulder",
                   "LeftHand", "RightHand", "LeftFoot", "RightFoot", "Spine"]

    for i, joint_name in enumerate(joint_order[:10]):
        if joint_name in positions:
            pos_3d = positions[joint_name]
            # Project to 2D (use X and Y, ignore Z for basic 2D)
            pose[i, 0] = pos_3d[0] * scale  # X
            pose[i, 1] = pos_3d[1] * scale  # Y (height)

    return pose


def convert_bvh_file(bvh_path: str, target_fps: int = 25) -> dict | None:
    """
    Convert a single BVH file to Stick-Gen format.

    Returns:
        dict with 'motion' (tensor), 'style' (str), 'filename' (str)
    """
    if not BVH_AVAILABLE:
        return None

    try:
        with open(bvh_path) as f:
            bvh = Bvh(f.read())

        # Get file info
        filename = os.path.basename(bvh_path)
        style = filename.split("_")[0] if "_" in filename else "unknown"

        # Get frame info
        n_frames = bvh.nframes
        frame_time = bvh.frame_time
        source_fps = 1.0 / frame_time if frame_time > 0 else 30.0

        # Initialize FK solver
        fk = BVHForwardKinematics(bvh)

        # Extract poses for all frames
        all_poses = []
        for frame_idx in range(n_frames):
            positions = fk.compute_frame_positions(frame_idx)
            pose = extract_stick_figure_pose(positions)
            all_poses.append(pose)

        motion = np.stack(all_poses, axis=0)  # [n_frames, 10, 2]

        # Resample to target FPS if needed
        if abs(source_fps - target_fps) > 1:
            motion = resample_motion(motion, source_fps, target_fps)

        # Flatten to [n_frames, 20]
        motion = motion.reshape(motion.shape[0], -1)

        return {
            "motion": torch.from_numpy(motion).float(),
            "style": style,
            "filename": filename,
            "n_frames": motion.shape[0],
            "fps": target_fps
        }

    except Exception as e:
        print(f"Error processing {bvh_path}: {e}")
        return None



def resample_motion(motion: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """
    Resample motion data to target FPS using linear interpolation.

    Args:
        motion: [n_frames, 10, 2] motion data
        source_fps: Original frame rate
        target_fps: Target frame rate

    Returns:
        Resampled motion array
    """
    n_frames = motion.shape[0]
    duration = n_frames / source_fps
    target_frames = int(duration * target_fps)

    # Create interpolation indices
    source_times = np.linspace(0, 1, n_frames)
    target_times = np.linspace(0, 1, target_frames)

    # Interpolate each joint coordinate
    resampled = np.zeros((target_frames, motion.shape[1], motion.shape[2]))
    for j in range(motion.shape[1]):
        for c in range(motion.shape[2]):
            resampled[:, j, c] = np.interp(target_times, source_times, motion[:, j, c])

    return resampled


def convert_100style(input_dir: str = "data/100STYLE",
                     output_path: str = "data/100style_processed.pt",
                     target_fps: int = 25,
                     max_files: int | None = None) -> None:
    """
    Convert 100STYLE dataset to Stick-Gen format.

    Args:
        input_dir: Directory containing BVH files
        output_path: Output path for processed data
        target_fps: Target frame rate for output
        max_files: Maximum number of files to process (None for all)
    """
    if not BVH_AVAILABLE:
        print("Error: BVH packages not available. Install with: pip install bvh scipy")
        return

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Please download 100STYLE dataset.")
        print("Download from: https://www.ianxmason.com/100style/")
        return

    bvh_files = glob.glob(os.path.join(input_dir, "**/*.bvh"), recursive=True)
    if max_files:
        bvh_files = bvh_files[:max_files]

    print(f"Found {len(bvh_files)} BVH files.")

    processed_data = []
    styles_found = set()

    for bvh_file in tqdm(bvh_files, desc="Processing 100STYLE"):
        result = convert_bvh_file(bvh_file, target_fps)
        if result:
            processed_data.append(result)
            styles_found.add(result["style"])

    print(f"\nProcessed {len(processed_data)} sequences.")
    print(f"Styles found: {sorted(styles_found)}")

    # Save processed data
    if processed_data:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            "sequences": processed_data,
            "styles": list(styles_found),
            "fps": target_fps,
            "n_sequences": len(processed_data)
        }, output_path)
        print(f"Saved to {output_path}")
    else:
        print("No data processed. Check BVH files and dependencies.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert 100STYLE BVH to Stick-Gen format")
    parser.add_argument("--input", type=str, default="data/100STYLE",
                       help="Input directory with BVH files")
    parser.add_argument("--output", type=str, default="data/100style_processed.pt",
                       help="Output file path")
    parser.add_argument("--fps", type=int, default=25,
                       help="Target frame rate")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum files to process (for testing)")

    args = parser.parse_args()

    convert_100style(
        input_dir=args.input,
        output_path=args.output,
        target_fps=args.fps,
        max_files=args.max_files
    )
