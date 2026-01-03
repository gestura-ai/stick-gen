"""100STYLE BVH dataset converter.

This module converts 100STYLE BVH motion-capture files into the Stick-Gen
canonical **v3 12-segment (48D)** stick-figure representation.

The legacy 5-segment / 20D representation is only retained for backwards
compatibility with old experiments and renderer/export tooling. New code
should always use the v3 48-dimensional format for training and evaluation.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data_gen.joint_utils import (
    CanonicalJoints2D,
    joints_to_v3_segments_2d,
    validate_v3_connectivity,
)

try:  # pragma: no cover - exercised indirectly in integration flows
    from bvh import Bvh
    from scipy.spatial.transform import Rotation as R

    BVH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    BVH_AVAILABLE = False

    class R:  # type: ignore[too-many-instance-attributes]
        """Placeholder for :class:`scipy.spatial.transform.Rotation`.

        This avoids import-time failures when the optional BVH dependencies
        are not installed. All methods simply return neutral rotations.
        """

        @staticmethod
        def identity() -> "R":  # type: ignore[override]
            return R()

        @staticmethod
        def from_euler(*_args: Any, **_kwargs: Any) -> "R":  # type: ignore[override]
            return R()

        def apply(self, vec: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
            return vec

        def __mul__(self, other: "R") -> "R":  # pragma: no cover - trivial
            return other


class BVHForwardKinematics:
    """Compute forward kinematics from BVH joint rotations to world positions.

    This is a lightweight FK helper tailored for 100STYLE's CMU-style
    skeletons. It exposes a single method,
    :meth:`compute_frame_positions`, which returns a mapping from joint
    names to 3D positions in world coordinates for a given frame index.
    """

    def __init__(self, bvh: "Bvh") -> None:  # type: ignore[name-defined]
        self._bvh = bvh
        self._joint_tree = self._build_joint_tree()

    def _build_joint_tree(self) -> dict[str, dict[str, Any]]:
        tree: dict[str, dict[str, Any]] = {}
        for joint in self._bvh.get_joints():
            name = joint.name
            parent = self._bvh.joint_parent(name)
            offset = np.asarray(self._bvh.joint_offset(name), dtype=np.float32)
            tree[name] = {
                "parent": parent.name if parent else None,
                "offset": offset,
                "children": [],
            }

        for name, data in tree.items():
            parent = data["parent"]
            if parent and parent in tree:
                tree[parent]["children"].append(name)

        return tree

    def _get_root_position(self, frame_idx: int) -> np.ndarray:
        root_name = self._bvh.get_joints_names()[0]
        try:
            x = self._bvh.frame_joint_channel(frame_idx, root_name, "Xposition", 0)
            y = self._bvh.frame_joint_channel(frame_idx, root_name, "Yposition", 0)
            z = self._bvh.frame_joint_channel(frame_idx, root_name, "Zposition", 0)
            return np.array([x, y, z], dtype=np.float32)
        except Exception:  # pragma: no cover - defensive
            return np.zeros(3, dtype=np.float32)

    def _get_joint_rotation(self, joint_name: str, frame_idx: int) -> "R":
        try:
            rx = self._bvh.frame_joint_channel(frame_idx, joint_name, "Xrotation", 0)
            ry = self._bvh.frame_joint_channel(frame_idx, joint_name, "Yrotation", 0)
            rz = self._bvh.frame_joint_channel(frame_idx, joint_name, "Zrotation", 0)
            # BVH commonly uses ZXY order
            return R.from_euler("ZXY", [rz, rx, ry], degrees=True)
        except Exception:  # pragma: no cover - defensive
            return R.identity()

    def compute_frame_positions(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return world-space joint positions for ``frame_idx``.

        Args:
            frame_idx: Index of the frame in the BVH sequence.
        """

        positions: dict[str, np.ndarray] = {}

        def _compute_recursive(joint_name: str, parent_pos: np.ndarray, parent_rot: "R") -> None:
            joint = self._joint_tree[joint_name]
            offset = joint["offset"]
            rotation = self._get_joint_rotation(joint_name, frame_idx)

            rotated_offset = parent_rot.apply(offset)
            world_pos = parent_pos + rotated_offset
            positions[joint_name] = world_pos

            combined_rot = parent_rot * rotation
            for child in joint["children"]:
                _compute_recursive(child, world_pos, combined_rot)

        root_name = self._bvh.get_joints_names()[0]
        root_pos = self._get_root_position(frame_idx)
        root_rot = self._get_joint_rotation(root_name, frame_idx)

        positions[root_name] = root_pos
        for child in self._joint_tree[root_name]["children"]:
            _compute_recursive(child, root_pos, root_rot)

        return positions


def extract_v3_segments_from_positions(
    positions: dict[str, np.ndarray], *, scale: float = 0.01
) -> np.ndarray:
    """Extract a v3 12-segment stick figure from full-skeleton positions.

    The mapping follows the canonical 2D joint set used across the
    project. We project BVH joints into 2D using an orthographic camera
    and then build 12 line segments via :func:`joints_to_v3_segments_2d`.

    Returns:
        Array of shape ``(12, 4)`` for 12 segments with
        ``(x1, y1, x2, y2)`` coordinates.
    """

    def get_joint_2d(name: str) -> np.ndarray:
        if name in positions:
            pos_3d = positions[name]
            return np.array([pos_3d[0] * scale, pos_3d[1] * scale], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    pelvis_center = get_joint_2d("Hips")[None, :]
    neck = get_joint_2d("Neck")[None, :]
    head_center = get_joint_2d("Head")[None, :]

    l_shoulder = get_joint_2d("LeftShoulder")[None, :]
    r_shoulder = get_joint_2d("RightShoulder")[None, :]
    chest = (l_shoulder + r_shoulder) / 2.0

    l_elbow = get_joint_2d("LeftForeArm")[None, :]
    r_elbow = get_joint_2d("RightForeArm")[None, :]
    l_wrist = get_joint_2d("LeftHand")[None, :]
    r_wrist = get_joint_2d("RightHand")[None, :]

    l_hip = get_joint_2d("LeftUpLeg")[None, :]
    r_hip = get_joint_2d("RightUpLeg")[None, :]
    l_knee = get_joint_2d("LeftLeg")[None, :]
    r_knee = get_joint_2d("RightLeg")[None, :]
    l_ankle = get_joint_2d("LeftFoot")[None, :]
    r_ankle = get_joint_2d("RightFoot")[None, :]

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

    segments = joints_to_v3_segments_2d(canonical_joints, flatten=False)
    segments_flat = segments.reshape(segments.shape[0], -1)
    validate_v3_connectivity(segments_flat)
    return segments[0]


def resample_motion(motion: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample motion sequence to ``target_fps`` using linear interpolation.

    Args:
        motion: Array of shape ``[T, 12, 4]`` with v3 segments.
        source_fps: Original frame rate.
        target_fps: Desired frame rate.
    """

    n_frames = motion.shape[0]
    duration = n_frames / float(source_fps)
    target_frames = max(int(round(duration * float(target_fps))), 1)

    source_times = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    target_times = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)

    resampled = np.zeros((target_frames, motion.shape[1], motion.shape[2]), dtype=np.float32)
    for j in range(motion.shape[1]):
        for c in range(motion.shape[2]):
            resampled[:, j, c] = np.interp(target_times, source_times, motion[:, j, c])

    return resampled


def convert_bvh_file(bvh_path: str, target_fps: int = 25) -> dict[str, Any] | None:
    """Convert a single 100STYLE BVH file to v3 stick-figure motion.

    Returns a dict with fields:

    * ``motion``: ``[T, 48]`` float32 tensor.
    * ``style``: style string inferred from the filename prefix.
    * ``filename``: basename of the BVH file.
    * ``n_frames``: number of frames in the converted sequence.
    * ``fps``: target frame rate.
    """

    if not BVH_AVAILABLE:
        return None

    try:
        with open(bvh_path, encoding="utf-8") as f:
            bvh = Bvh(f.read())  # type: ignore[name-defined]

        filename = os.path.basename(bvh_path)
        style = filename.split("_")[0] if "_" in filename else "unknown"

        n_frames = bvh.nframes
        frame_time = float(getattr(bvh, "frame_time", 1.0 / 30.0))
        source_fps = 1.0 / frame_time if frame_time > 0 else 30.0

        fk = BVHForwardKinematics(bvh)

        all_poses: list[np.ndarray] = []
        for frame_idx in range(n_frames):
            positions = fk.compute_frame_positions(frame_idx)
            pose = extract_v3_segments_from_positions(positions)
            all_poses.append(pose)

        motion = np.stack(all_poses, axis=0)  # [T, 12, 4]

        if abs(source_fps - target_fps) > 1.0:
            motion = resample_motion(motion, source_fps, float(target_fps))

        motion = motion.reshape(motion.shape[0], -1)  # [T, 48]

        return {
            "motion": torch.from_numpy(motion).float(),
            "style": style,
            "filename": filename,
            "n_frames": motion.shape[0],
            "fps": target_fps,
        }
    except Exception as exc:  # pragma: no cover - defensive around third-party BVH
        print(f"Error processing {bvh_path}: {exc}")
        return None


def convert_100style(
    input_dir: str = "data/100STYLE",
    output_path: str = "data/100style_processed.pt",
    target_fps: int = 25,
    max_files: int | None = None,
) -> None:
    """Convert the 100STYLE BVH dataset into v3 48D Stick-Gen format.

    The resulting file is a simple :func:`torch.save` blob containing a
    list of per-sequence dicts produced by :func:`convert_bvh_file` plus
    some aggregate metadata.
    """

    if not BVH_AVAILABLE:
        print("Error: BVH packages not available. Install with: pip install bvh scipy")
        return

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Please download 100STYLE dataset.")
        print("Download from: https://www.ianxmason.com/100style/")
        return

    bvh_files = glob.glob(os.path.join(input_dir, "**", "*.bvh"), recursive=True)
    if max_files is not None:
        bvh_files = bvh_files[: max(0, max_files)]

    print(f"Found {len(bvh_files)} BVH files.")

    processed_data: list[dict[str, Any]] = []
    styles_found: set[str] = set()

    for bvh_file in tqdm(bvh_files, desc="Processing 100STYLE"):
        result = convert_bvh_file(bvh_file, target_fps=target_fps)
        if result is not None:
            processed_data.append(result)
            styles_found.add(result["style"])

    print(f"\nProcessed {len(processed_data)} sequences.")
    print(f"Styles found: {sorted(styles_found)}")

    if not processed_data:
        print("No data processed. Check BVH files and dependencies.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "sequences": processed_data,
            "styles": sorted(styles_found),
            "fps": target_fps,
            "n_sequences": len(processed_data),
        },
        output_path,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 100STYLE BVH to Stick-Gen v3 48D format",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/100STYLE",
        help="Input directory with BVH files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/100style_processed.pt",
        help="Output file path",
    )
    parser.add_argument("--fps", type=int, default=25, help="Target frame rate")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to process (for testing)",
    )

    args = parser.parse_args()

    convert_100style(
        input_dir=args.input,
        output_path=args.output,
        target_fps=args.fps,
        max_files=args.max_files,
    )
