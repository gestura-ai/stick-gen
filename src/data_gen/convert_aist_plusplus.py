import glob
import os
import pickle
from typing import Any

import numpy as np
import torch

from .convert_amass import compute_basic_physics
from .metadata_extractors import build_enhanced_metadata
from .schema import ACTION_TO_IDX, ActionType, MusicMetadata
from .validator import DataValidator

# Very lightweight AIST++ converter that uses keypoints3d_optim to avoid
# depending on SMPL models. We approximate our 5-line stick figure from 3D
# keypoints then drop Z to obtain 2D.


# Mapping indices from AIST++ keypoints (COCO-ish 3D, 17 joints) to endpoints.
# This mapping is approximate and follows the AIST++ documentation.
AIST_TO_STICK = {
    "head": 0,  # Nose / head
    "torso": 8,  # Mid-hip / pelvis
    "left_hand": 9,  # Left wrist
    "right_hand": 10,  # Right wrist
    "left_foot": 15,  # Left ankle
    "right_foot": 16,  # Right ankle
}


def keypoints3d_to_stick(keypoints: np.ndarray) -> np.ndarray:
    """Convert [T, 17, 3] to [T, 20] stick-figure representation."""
    T = keypoints.shape[0]
    out = np.zeros((T, 5, 4), dtype=np.float32)
    head = keypoints[:, AIST_TO_STICK["head"], :2]
    torso = keypoints[:, AIST_TO_STICK["torso"], :2]
    lh = keypoints[:, AIST_TO_STICK["left_hand"], :2]
    rh = keypoints[:, AIST_TO_STICK["right_hand"], :2]
    lf = keypoints[:, AIST_TO_STICK["left_foot"], :2]
    rf = keypoints[:, AIST_TO_STICK["right_foot"], :2]

    out[:, 0] = np.concatenate([head, torso], axis=-1)
    out[:, 1] = np.concatenate([torso, lh], axis=-1)
    out[:, 2] = np.concatenate([torso, rh], axis=-1)
    out[:, 3] = np.concatenate([torso, lf], axis=-1)
    out[:, 4] = np.concatenate([torso, rf], axis=-1)
    return out.reshape(T, 20)


def _load_aist_keypoints(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    # Official AIST++ format uses keypoints3d in `keypoints3d` or `keypoints3d_optim`.
    if "keypoints3d_optim" in data:
        kp = data["keypoints3d_optim"]
    else:
        kp = data["keypoints3d"]
    return kp.astype(np.float32)


def _infer_camera_feature(seq_name: str) -> torch.Tensor:
    """Return a simple per-frame camera feature [T, 3].

    For now we only encode which camera id (01-09) is used and map it to a
    discrete embedding in [0, 1]. This is a placeholder but keeps the field
    structurally present.
    """
    # Sequence names in AIST++ look like 'gBR_sBM_c01_d04_mBR0_ch01', so cXX
    cam_id = 1
    for part in seq_name.split("_"):
        if part.startswith("c") and part[1:].isdigit():
            cam_id = int(part[1:])
            break
    # Map cam_id 1-9 -> scalar in [0, 1]
    cam_scalar = (cam_id - 1) / 8.0
    # Caller will expand this to correct T.
    return torch.tensor([cam_scalar, 0.0, 0.0], dtype=torch.float32)


def _parse_aist_sequence_name(seq_name: str) -> dict[str, str | None]:
    """Parse AIST++ sequence name for music and dance metadata.

    AIST++ sequence names follow the format:
        gBR_sBM_c01_d04_mBR0_ch01
        |   |   |   |   |    |
        |   |   |   |   |    +-- choreography ID
        |   |   |   |   +------- music ID (e.g., mBR0 = break music 0)
        |   |   |   +----------- dancer ID
        |   |   +--------------- camera ID
        |   +------------------- dance setting (BM=basic, FM=freestyle, etc.)
        +----------------------- genre (BR=break, PO=pop, etc.)

    Returns:
        Dict with parsed metadata fields
    """
    parts = seq_name.split("_")
    result: dict[str, str | None] = {
        "genre": None,
        "setting": None,
        "camera_id": None,
        "dancer_id": None,
        "music_id": None,
        "choreo_id": None,
    }

    if len(parts) >= 1:
        result["genre"] = parts[0][1:] if parts[0].startswith("g") else parts[0]
    if len(parts) >= 2:
        result["setting"] = parts[1][1:] if parts[1].startswith("s") else parts[1]
    if len(parts) >= 3:
        result["camera_id"] = parts[2]
    if len(parts) >= 4:
        result["dancer_id"] = parts[3]
    if len(parts) >= 5:
        result["music_id"] = parts[4]
    if len(parts) >= 6:
        result["choreo_id"] = parts[5]

    return result


# AIST++ genre abbreviations to full names
AIST_GENRE_MAP = {
    "BR": "break",
    "PO": "pop",
    "LO": "lock",
    "MH": "middle_hip_hop",
    "LH": "LA_hip_hop",
    "HO": "house",
    "WA": "waack",
    "KR": "krump",
    "JS": "street_jazz",
    "JB": "ballet_jazz",
}


def _build_sample(
    keypoints: np.ndarray, seq_name: str, meta: dict[str, Any], fps: int = 60
) -> dict[str, Any]:
    motion_np = keypoints3d_to_stick(keypoints)
    motion = torch.from_numpy(motion_np)
    physics = compute_basic_physics(motion, fps=fps)

    # Dance actions for AIST++
    action_enum = ActionType.DANCE
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    cam_feat = _infer_camera_feature(seq_name)
    camera = cam_feat.unsqueeze(0).expand(T, -1)

    # Parse sequence name for music metadata
    parsed = _parse_aist_sequence_name(seq_name)
    genre_abbr = parsed.get("genre")
    genre = AIST_GENRE_MAP.get(genre_abbr, genre_abbr) if genre_abbr else None

    description = f"A person dancing {genre or 'a dance'} in the AIST++ dataset."

    # Build enhanced metadata with music info
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=fps,
        description=description,
        original_fps=fps,  # AIST++ is at native fps
        original_num_frames=T,
    )
    # Add music metadata specific to AIST++
    enhanced_meta.music = MusicMetadata(
        bpm=None,  # Would require loading music annotation files
        beat_frames=None,  # Would require loading beat annotations
        genre=genre,
    )

    return {
        "description": description,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "camera": camera,
        "source": "aist_plusplus",
        "meta": {**meta, "parsed_name": parsed},
        "enhanced_meta": enhanced_meta.model_dump(),
    }


def convert_aist_plusplus(
    root_dir: str, output_path: str, fps: int = 60, max_files: int = -1
) -> None:
    """Convert AIST++ keypoints3d to canonical schema.

    Expects `keypoints3d/` (or `keypoints3d_optim/`) under `root_dir`.
    """
    kp_dir = os.path.join(root_dir, "keypoints3d_optim")
    if not os.path.isdir(kp_dir):
        kp_dir = os.path.join(root_dir, "keypoints3d")

    paths = sorted(glob.glob(os.path.join(kp_dir, "*.pkl")))
    if max_files > 0:
        paths = paths[:max_files]

    validator = DataValidator(fps=fps)
    # AIST++ keypoints are in a dataset-specific coordinate system where
    # velocities/accelerations are much larger than the default thresholds.
    # Loosen them here so we do not discard all clips while still catching
    # extreme numerical glitches.
    validator.max_velocity = 300.0
    validator.max_acceleration = 6000.0
    samples: list[dict[str, Any]] = []

    for path in paths:
        seq_name = os.path.splitext(os.path.basename(path))[0]
        keypoints = _load_aist_keypoints(path)
        meta = {
            "sequence": seq_name,
            "keypoints_path": os.path.relpath(path, root_dir),
            "fps": fps,
        }
        sample = _build_sample(keypoints, seq_name, meta, fps=fps)
        # Allow some limb-length variability due to 3D->2D projection and
        # keypoint noise; enforce only physics constraints here.
        ok, score, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            continue
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert AIST++ to canonical schema")
    parser.add_argument(
        "--aist-root", type=str, required=True, help="Root directory of aist_plusplus"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-files", type=int, default=-1)
    args = parser.parse_args()

    convert_aist_plusplus(
        args.aist_root, args.output, fps=args.fps, max_files=args.max_files
    )


if __name__ == "__main__":
    main()
