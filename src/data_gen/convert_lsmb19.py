import os
from typing import List, Dict, Any, Iterator, Tuple

import numpy as np
import torch

from .schema import ACTION_TO_IDX
from .validator import DataValidator
from .convert_amass import compute_basic_physics
from .convert_ntu_rgbd import NTU_ACTION_LABELS, _action_to_enum, joints_to_stick


def _parse_sample_header(header: str) -> Dict[str, Any]:
    """Parse a training-sample header like `S001C003P007R002A043_61`.

    The prefix is the usual NTU RGB+D clip id (SssCcccPpppRrrrAaaa) and the
    suffix after the underscore is the starting frame index in the long
    CS/CV sequence. We expose both the structured NTU fields and the raw id.
    """
    header = header.strip()
    base, offset_str = header.split("_")
    setup = int(base[1:4])
    camera = int(base[5:8])
    performer = int(base[9:12])
    replication = int(base[13:16])
    action = int(base[17:20])
    start_frame = int(offset_str)
    return {
        "ntu_id": base,
        "setup_id": setup,
        "camera_id": camera,
        "performer_id": performer,
        "replication_id": replication,
        "action_id": action,
        "start_frame_global": start_frame,
    }


def _parse_frame_line(line: str) -> np.ndarray:
    """Parse one skeleton frame line into [25, 3] joints.

    Lines look like: "0.0, 0.0, 0.0; x, y, z; ..." (25 segments). We are
    conservative and pad/truncate to 25 joints if needed.
    """
    s = line.strip()
    parts = [p for p in s.split(";") if p.strip()]
    coords: List[List[float]] = []
    for p in parts:
        xyz = [float(x) for x in p.split(",") if x.strip()]
        if not xyz:
            continue
        if len(xyz) >= 3:
            xyz = xyz[:3]
        else:
            # Right-pad if somehow shorter.
            xyz = xyz + [0.0] * (3 - len(xyz))
        coords.append(xyz)
    if not coords:
        return np.zeros((25, 3), dtype=np.float32)
    joints = np.asarray(coords, dtype=np.float32)
    if joints.shape[0] < 25:
        pad = np.zeros((25 - joints.shape[0], 3), dtype=np.float32)
        joints = np.concatenate([joints, pad], axis=0)
    elif joints.shape[0] > 25:
        joints = joints[:25]
    return joints


def _iter_training_samples(path: str,
                           protocol: str,
                           max_samples: int = -1) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
    """Yield (meta, joints[T,25,3]) from an LSMB19 `*_training_samples.txt` file."""
    if not os.path.exists(path):
        return

    current_header: str | None = None
    frames: List[np.ndarray] = []
    yielded = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s[0] == "S":
                # Starting a new sample.
                if current_header is not None and frames:
                    meta = _parse_sample_header(current_header)
                    meta["protocol"] = protocol
                    joints = np.stack(frames, axis=0)  # [T, 25, 3]
                    yielded += 1
                    yield meta, joints
                    if max_samples > 0 and yielded >= max_samples:
                        return
                current_header = s
                frames = []
            else:
                frames.append(_parse_frame_line(s))

    if current_header is not None and frames:
        meta = _parse_sample_header(current_header)
        meta["protocol"] = protocol
        joints = np.stack(frames, axis=0)
        yield meta, joints


def _build_canonical_sample_from_joints(joints: np.ndarray,
                                        meta: Dict[str, Any],
                                        fps: int) -> Dict[str, Any]:
    """Convert [T,25,3] NTU-style joints to our canonical sample dict."""
    motion_np = joints_to_stick(joints)  # [T, 20]
    motion = torch.from_numpy(motion_np)
    physics = compute_basic_physics(motion, fps=fps)

    action_id = int(meta["action_id"])
    action_enum = _action_to_enum(action_id)
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    label = NTU_ACTION_LABELS.get(action_id, f"action {action_id}")
    proto = meta.get("protocol", "?")
    description = (
        f"LSMB19 {proto.upper()} sample from NTU RGB+D action {action_id}: {label}. "
        f"Sequence length {T} frames."
    )

    sample = {
        "description": description,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "camera": None,
        "source": "lsmb19",
        "meta": meta,
    }
    return sample


def convert_lsmb19(data_root: str,
                   output_path: str,
                   fps: int = 30,
                   max_samples: int = -1) -> None:
    """Convert LSMB19 training samples into the canonical schema.

    We use the pre-segmented `cs_training_samples.txt` and
    `cv_training_samples.txt` files produced by the benchmark authors. Each
    sample block is:

        SssCcccPpppRrrrAaaa_offset   # NTU clip id + starting frame index
        <frame 0 joints>
        <frame 1 joints>
        ...

    where every frame line contains 25 semicolon-separated `x, y, z` triplets.
    We interpret joints as NTU RGB+D joints and reuse the NTU converter's
    stick-figure mapping and action heuristics.
    """
    cs_train = os.path.join(data_root, "cs_training_samples.txt")
    cv_train = os.path.join(data_root, "cv_training_samples.txt")

    validator = DataValidator(fps=fps)
    samples: List[Dict[str, Any]] = []

    remaining = max_samples if max_samples > 0 else None

    for protocol, path in (("cs", cs_train), ("cv", cv_train)):
        if remaining is not None and remaining <= 0:
            break
        if not os.path.exists(path):
            continue

        per_proto_max = remaining if remaining is not None else -1
        for meta, joints in _iter_training_samples(path, protocol, max_samples=per_proto_max):
            sample = _build_canonical_sample_from_joints(joints, meta, fps=fps)
            # For LSMB19 we keep only the physics-based filter here to avoid
            # over-penalising projection-induced limb-length variation.
            ok, score, reason = validator.check_physics_consistency(sample["physics"])
            if not ok:
                continue
            samples.append(sample)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert LSMB19 to canonical schema")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of lsmb19-mocap")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Optional global cap on number of samples to convert")
    args = parser.parse_args()

    convert_lsmb19(args.root, args.output, fps=args.fps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

