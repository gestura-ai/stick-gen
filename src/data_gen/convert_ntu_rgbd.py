import os
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from .schema import ActionType, ACTION_TO_IDX
from .validator import DataValidator
from .convert_amass import compute_basic_physics


NTU_ACTION_LABELS: Dict[int, str] = {
    # Official NTU RGB+D 60 action names (1-indexed). We only use them
    # to build simple textual descriptions and coarse ActionType mapping.
    1: "drink water",
    2: "eat meal",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop object",
    6: "pickup object",
    7: "throw object",
    8: "sitting down",
    9: "standing up",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear shoe",
    17: "take off shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on hat",
    21: "take off hat",
    22: "cheer up",
    23: "hand waving",
    24: "kicking",
    25: "reach into pocket",
    26: "hopping",
    27: "jump up",
    28: "make phone call",
    29: "play with phone",
    30: "type on keyboard",
    31: "point to something",
    32: "taking selfie",
    33: "check time",
    34: "rub hands",
    35: "nod head",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put palms together",
    40: "cross hands in front",
    41: "sneeze or cough",
    42: "stagger",
    43: "falling",
    44: "touch head",
    45: "touch chest",
    46: "touch back",
    47: "touch neck",
    48: "nausea or vomiting",
    49: "use fan",
    50: "punching",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back",
    54: "point finger at other",
    55: "hugging other person",
    56: "giving object",
    57: "touching pocket",
    58: "shaking hands",
    59: "walking towards",
    60: "walking apart",
}


def _parse_filename(path: str) -> Tuple[int, int, int, int, int, int]:
    """Parse NTU file name `SssCcccPpppRrrrAaaa.skeleton`.

    Returns (setup_id, camera_id, performer_id, replication_id, action_id, ignored_idx).
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # Example: S001C001P001R001A001
    setup = int(name[1:4])
    camera = int(name[5:8])
    performer = int(name[9:12])
    replication = int(name[13:16])
    action = int(name[17:20])
    return setup, camera, performer, replication, action, 0


def _read_skeleton_file(path: str) -> np.ndarray:
    """Read a single NTU `.skeleton` file into an array [T, 25, 3].

    Follows the official NTU format (see `read_skeleton_file.m`). We keep only
    the 3D joint positions and ignore orientations and other metadata.
    For frames with multiple bodies we keep the first body only.
    """
    with open(path, "r") as f:
        num_frames = int(f.readline())
        frames: List[np.ndarray] = []
        for _ in range(num_frames):
            num_bodies = int(f.readline())
            body_joints = None
            for b in range(num_bodies):
                # Body info line (10 numbers), we ignore content.
                _ = f.readline()
                num_joints = int(f.readline())
                joints = []
                for _ in range(num_joints):
                    parts = f.readline().strip().split()
                    # x, y, z are first 3 floats
                    x, y, z = map(float, parts[:3])
                    joints.append((x, y, z))
                joints_arr = np.asarray(joints, dtype=np.float32)
                if body_joints is None:
                    body_joints = joints_arr
            if body_joints is None:
                # No body this frame: pad with zeros
                body_joints = np.zeros((25, 3), dtype=np.float32)
            frames.append(body_joints)
    return np.stack(frames, axis=0)  # [T, 25, 3]


# Mapping from NTU joints (0-24) to a coarse stick figure.
# We approximate torso/head and limb endpoints.
NTU_TO_STICK = {
    "head": 3,          # Head
    "torso": 1,         # SpineBase
    "left_hand": 7,     # HandLeft
    "right_hand": 11,   # HandRight
    "left_foot": 19,    # FootLeft
    "right_foot": 23,   # FootRight
}


def joints_to_stick(joints: np.ndarray) -> np.ndarray:
    """Convert [T, 25, 3] joints to [T, 20] stick lines (5 segments)."""
    T = joints.shape[0]
    out = np.zeros((T, 5, 4), dtype=np.float32)
    head = joints[:, NTU_TO_STICK["head"], :2]
    torso = joints[:, NTU_TO_STICK["torso"], :2]
    lh = joints[:, NTU_TO_STICK["left_hand"], :2]
    rh = joints[:, NTU_TO_STICK["right_hand"], :2]
    lf = joints[:, NTU_TO_STICK["left_foot"], :2]
    rf = joints[:, NTU_TO_STICK["right_foot"], :2]

    out[:, 0] = np.concatenate([head, torso], axis=-1)
    out[:, 1] = np.concatenate([torso, lh], axis=-1)
    out[:, 2] = np.concatenate([torso, rh], axis=-1)
    out[:, 3] = np.concatenate([torso, lf], axis=-1)
    out[:, 4] = np.concatenate([torso, rf], axis=-1)
    return out.reshape(T, 20)


def _action_to_enum(action_id: int) -> ActionType:
    name = NTU_ACTION_LABELS.get(action_id, "unknown")
    s = name.lower()
    if any(k in s for k in ["walk", "hopping", "jump"]):
        return ActionType.WALK
    if any(k in s for k in ["run", "stagger", "fall"]):
        return ActionType.RUN
    if any(k in s for k in ["punch", "kicking other", "push", "hug"]):
        return ActionType.FIGHT
    if any(k in s for k in ["clap", "cheer", "dance"]):
        return ActionType.DANCE
    return ActionType.IDLE


def _build_canonical_sample(joints: np.ndarray,
                            meta: Dict[str, Any],
                            fps: int = 30) -> Dict[str, Any]:
    motion_np = joints_to_stick(joints)  # [T, 20]
    motion = torch.from_numpy(motion_np)
    physics = compute_basic_physics(motion, fps=fps)

    action_id = meta["action_id"]
    action_enum = _action_to_enum(action_id)
    action_idx = ACTION_TO_IDX[action_enum]
    actions = torch.full((motion.shape[0],), action_idx, dtype=torch.long)

    label = NTU_ACTION_LABELS.get(action_id, f"action {action_id}")
    description = f"A person performing the NTU RGB+D action: {label}."

    sample = {
        "description": description,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "camera": None,
        "source": "ntu_rgbd",
        "meta": meta,
    }
    return sample


def convert_ntu_rgbd(root_dir: str,
                     output_path: str,
                     fps: int = 30,
                     max_files: int = -1) -> None:
    pattern = os.path.join(root_dir, "nturgb+d_skeletons", "S*.skeleton")
    paths = sorted(glob.glob(pattern))
    if max_files > 0:
        paths = paths[:max_files]

    validator = DataValidator(fps=fps)
    samples: List[Dict[str, Any]] = []

    for path in paths:
        setup, camera, performer, replication, action, _ = _parse_filename(path)
        joints = _read_skeleton_file(path)
        meta = {
            "path": os.path.relpath(path, root_dir),
            "setup_id": setup,
            "camera_id": camera,
            "performer_id": performer,
            "replication_id": replication,
            "action_id": action,
            "fps": fps,
        }
        sample = _build_canonical_sample(joints, meta, fps=fps)
        # For dataset ingestion we only require physics sanity checks; skeleton
        # consistency in 2D projection can legitimately vary across views.
        ok, score, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            continue
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert NTU RGB+D to canonical schema")
    parser.add_argument("--ntu-root", type=str, required=True,
                        help="Root of NTU_RGB_D (directory containing ntu-rgbd/ or nturgb+d_skeletons)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=-1)
    args = parser.parse_args()

    root = args.ntu_root
    if os.path.isdir(os.path.join(root, "nturgb+d_skeletons")):
        ntu_root = root
    else:
        ntu_root = os.path.join(root, "ntu-rgbd")

    convert_ntu_rgbd(ntu_root, args.output, fps=args.fps, max_files=args.max_files)


if __name__ == "__main__":
    main()

