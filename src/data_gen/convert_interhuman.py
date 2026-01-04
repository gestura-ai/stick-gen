import glob
import os
import pickle
from typing import Any

import numpy as np
import torch

from .convert_amass import AMASSConverter, compute_basic_physics
from .metadata_extractors import build_enhanced_metadata, compute_interaction_metadata
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator



# Import centralized paths config
try:
    from ..config.paths import get_path
    DEFAULT_OUTPUT_PATH = str(get_path("interhuman_canonical"))
except ImportError:
    DEFAULT_OUTPUT_PATH = "data/motions_processed/interhuman/canonical.pt"

def _person_dict_to_motion(
    person: dict[str, Any], converter: AMASSConverter
) -> torch.Tensor:
    """Convert a single person's SMPL-like parameters into v3 stick motion.

    The InterHuman motions store SMPL-style parameters per person:

      - trans: [T, 3]
      - root_orient: [T, 3]
      - pose_body: [T, 63]
      - betas: [10]

    We reuse the SMPL-X model and v3 stick-figure mapping from
    :class:`AMASSConverter` to obtain 3D joints and then 2D v3 segments with
    12 connected limbs (48 float coordinates per frame).

    Raises:
        ValueError: If person dict is None or missing required fields
        RuntimeError: If SMPL-X model is not available
    """
    # Validate person dict
    if person is None:
        raise ValueError("Person data is None")
    if not isinstance(person, dict):
        raise ValueError(f"Person data is not a dict: {type(person)}")

    # Check required fields
    required_fields = ["trans", "root_orient", "pose_body"]
    for field in required_fields:
        if field not in person or person[field] is None:
            raise ValueError(f"Missing required field '{field}' in person data")

    # Ensure SMPL-X model is loaded (will no-op if already present)
    converter._load_smpl_model("smplx")
    model = converter.smplx_model
    if model is None:
        raise RuntimeError("SMPL-X model not available for InterHuman conversion")

    trans = torch.tensor(person["trans"], dtype=torch.float32)
    root_orient = torch.tensor(person["root_orient"], dtype=torch.float32)
    body_pose = torch.tensor(person["pose_body"], dtype=torch.float32)

    # Normalize body pose to [T, 63] if provided as [T, 21, 3]
    if body_pose.ndim == 3:
        T = body_pose.shape[0]
        body_pose = body_pose.view(T, -1)
    else:
        T = trans.shape[0]

    betas = person.get("betas")
    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    betas_1 = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)  # [1, 10]

    # Get default poses from model (single frame templates)
    left_hand_1 = (
        model.left_hand_pose.clone()
        if hasattr(model, "left_hand_pose")
        else torch.zeros(1, 45, dtype=torch.float32)
    )
    right_hand_1 = (
        model.right_hand_pose.clone()
        if hasattr(model, "right_hand_pose")
        else torch.zeros(1, 45, dtype=torch.float32)
    )
    jaw_1 = (
        model.jaw_pose.clone()
        if hasattr(model, "jaw_pose")
        else torch.zeros(1, 3, dtype=torch.float32)
    )
    leye_1 = (
        model.leye_pose.clone()
        if hasattr(model, "leye_pose")
        else torch.zeros(1, 3, dtype=torch.float32)
    )
    reye_1 = (
        model.reye_pose.clone()
        if hasattr(model, "reye_pose")
        else torch.zeros(1, 3, dtype=torch.float32)
    )
    expression_1 = (
        model.expression.clone()
        if hasattr(model, "expression")
        else torch.zeros(1, 10, dtype=torch.float32)
    )

    # Batched SMPL-X forward pass with chunking for memory efficiency
    # Process in chunks to avoid OOM on very long sequences
    # 512 frames balances throughput vs memory (most clips are <500 frames)
    CHUNK_SIZE = 512
    joints_list = []

    with torch.no_grad():
        for start in range(0, T, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, T)
            chunk_size = end - start

            # Repeat static tensors to match chunk size
            output = model(
                body_pose=body_pose[start:end],
                global_orient=root_orient[start:end],
                transl=trans[start:end],
                left_hand_pose=left_hand_1.repeat(chunk_size, 1),
                right_hand_pose=right_hand_1.repeat(chunk_size, 1),
                jaw_pose=jaw_1.repeat(chunk_size, 1),
                leye_pose=leye_1.repeat(chunk_size, 1),
                reye_pose=reye_1.repeat(chunk_size, 1),
                expression=expression_1.repeat(chunk_size, 1),
                betas=betas_1.repeat(chunk_size, 1),
            )
            joints_list.append(output.joints[:, :22, :].cpu().numpy())  # [chunk, 22, 3]

    joints = np.concatenate(joints_list, axis=0)  # [T, 22, 3]

    # Use the v3 12-segment mapping from the AMASS converter to obtain a
    # fully connected stick-figure representation.
    segments = converter.smpl_to_v3_segments_2d(joints)  # [T, 48]
    motion = torch.from_numpy(segments.astype(np.float32))  # [T, 48]
    return motion


def _load_motion_pair_from_pkl(
    pkl_path: str, converter: AMASSConverter
) -> torch.Tensor:
    """Load an InterHuman motion .pkl and pack two actors into ``[T, 2, 48]``.

    Raises:
        ValueError: If pickle data is malformed or missing person data
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if obj is None:
        raise ValueError(f"Pickle file contains None: {pkl_path}")
    if not isinstance(obj, dict):
        raise ValueError(f"Pickle file does not contain a dict: {pkl_path}")

    # Check for required person fields
    if "person1" not in obj or obj["person1"] is None:
        raise ValueError(f"Missing 'person1' data in {pkl_path}")
    if "person2" not in obj or obj["person2"] is None:
        raise ValueError(f"Missing 'person2' data in {pkl_path}")

    person1 = obj["person1"]
    person2 = obj["person2"]

    m1 = _person_dict_to_motion(person1, converter)  # [T1, 48]
    m2 = _person_dict_to_motion(person2, converter)  # [T2, 48]

    # Ensure equal length (clip to shortest if needed)
    T = min(m1.shape[0], m2.shape[0])
    if m1.shape[0] != T:
        m1 = m1[:T]
    if m2.shape[0] != T:
        m2 = m2[:T]

    stacked = torch.stack([m1, m2], dim=1)  # [T, 2, 48]
    return stacked


def _load_texts(annots_dir: str, clip_id: str) -> list[str]:
    """Load textual annotations for a given clip.

    The InterHuman dataset typically stores one text file per clip in
    `annots/CLIP_ID.txt`, each line being a separate natural language
    description.
    """
    path = os.path.join(annots_dir, f"{clip_id}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def _build_sample(
    motion: torch.Tensor, texts: list[str], meta: dict[str, Any], fps: int = 30
) -> dict[str, Any]:
    # motion: [T, 2, 48] in v3 12-segment schema
    physics = compute_basic_physics(motion, fps=fps)  # [T, 2, 6]

    # For now treat all InterHuman clips as generic interactions.
    action_enum = ActionType.FIGHT
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T, 2), action_idx, dtype=torch.long)

    description = (
        texts[0] if texts else "Two people interacting in the InterHuman dataset."
    )

    # Build enhanced metadata with interaction info
    enhanced_meta = build_enhanced_metadata(
        motion=motion,
        fps=fps,
        description=description,
        original_fps=fps,
        original_num_frames=T,
        is_multi_actor=True,
    )
    # Compute interaction-specific metadata
    enhanced_meta.interaction = compute_interaction_metadata(motion)

    return {
        "description": description,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "camera": None,
        "source": "interhuman",
        "meta": meta,
        "enhanced_meta": enhanced_meta.model_dump(),
    }


def convert_interhuman(
    data_root: str,
    output_path: str,
    fps: int = 30,
    max_clips: int = -1,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Convert InterHuman motions into canonical schema.

    Supports the official InterHuman release layout under ``data_root``:

      - motions/*.pkl              (per-clip SMPL-style parameters)
      - annots/*.txt               (per-clip text descriptions)
      - split/train.txt, val.txt, test.txt (optional)

    Args:
        data_root: Root directory of InterHuman dataset
        output_path: Output .pt file path
        fps: Target frame rate
        max_clips: Maximum clips to process (-1 for all)
        verbose: If True, print each skipped/error file

    Returns:
        List of converted samples
    """
    import sys

    motions_dir = os.path.join(data_root, "motions")
    annots_dir = os.path.join(data_root, "annots")

    # Check if motions directory exists
    if not os.path.isdir(motions_dir):
        print(f"[InterHuman] ERROR: Motions directory not found: {motions_dir}")
        print(f"[InterHuman] Checked data_root: {data_root}")
        return []

    motion_files = sorted(glob.glob(os.path.join(motions_dir, "*.pkl")))
    if max_clips > 0:
        motion_files = motion_files[:max_clips]

    total_files = len(motion_files)

    # Early exit if no files found
    if total_files == 0:
        print(f"[InterHuman] WARNING: No .pkl files found in {motions_dir}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save([], output_path)
        print(f"[InterHuman] Saved empty dataset to {output_path}")
        return []

    print(f"[InterHuman] Processing {total_files} motion files...")
    print(f"[InterHuman] Motions dir: {motions_dir}")
    print(f"[InterHuman] Annotations dir: {annots_dir}")
    sys.stdout.flush()

    validator = DataValidator(fps=fps)
    converter = AMASSConverter()  # Reuse SMPL-X + stick-figure utilities
    samples: list[dict[str, Any]] = []
    num_errors = 0
    num_physics_skipped = 0

    for i, motion_path in enumerate(motion_files):
        # Progress logging: first file, then every 100 files for better visibility
        if i == 0 or i % 100 == 0 or i == total_files - 1:
            pct = 100 * i / total_files if total_files > 0 else 0
            print(f"[InterHuman] Processing {i}/{total_files} ({pct:.1f}%)...")
            sys.stdout.flush()

        clip_id = os.path.splitext(os.path.basename(motion_path))[0]

        try:
            # Load metadata once to retrieve mocap framerate for physics
            with open(motion_path, "rb") as f:
                obj = pickle.load(f)

            # Handle None or non-dict pickle data
            if obj is None or not isinstance(obj, dict):
                if verbose:
                    print(f"[InterHuman] Skipping {clip_id}: invalid pickle data")
                num_errors += 1
                continue

            mocap_fps = obj.get("mocap_framerate", fps)
            try:
                fps_clip = int(round(float(mocap_fps)))
            except (ValueError, TypeError):
                fps_clip = fps

            motion = _load_motion_pair_from_pkl(motion_path, converter)  # [T, 2, 20]
            texts = _load_texts(annots_dir, clip_id)
            meta = {
                "clip_id": clip_id,
                "motion_path": os.path.relpath(motion_path, data_root),
                "mocap_framerate": mocap_fps,
                "frames": obj.get("frames"),
            }
            sample = _build_sample(motion, texts, meta, fps=fps_clip)

            # Multi-actor motions already satisfy a structural layout; here we only
            # enforce basic physics bounds to avoid discarding valid interactions
            # due to 2D projection artifacts.
            # Pass the clip's actual fps to the validator so thresholds are scaled
            # correctly. InterHuman data is often at 60fps, which produces higher
            # velocity/acceleration values for the same physical motion compared
            # to 20-25fps datasets like HumanML3D.
            ok, score, reason = validator.check_physics_consistency(
                sample["physics"], clip_fps=fps_clip
            )
            if not ok:
                if verbose:
                    print(f"[InterHuman] Skipping {clip_id}: {reason}")
                num_physics_skipped += 1
                continue

            samples.append(sample)

        except Exception as e:
            if verbose:
                print(f"[InterHuman] Error processing {clip_id}: {e}")
            num_errors += 1
            continue

    # Summary output
    total = len(motion_files)
    print(
        f"[InterHuman] Conversion complete:\n"
        f"  - Valid samples: {len(samples)}/{total}\n"
        f"  - Physics skipped: {num_physics_skipped}\n"
        f"  - Errors: {num_errors}"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    print(f"[InterHuman] Output: {output_path}")

    return samples


def main() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Convert InterHuman to canonical schema"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of InterHuman Dataset",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file path"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each skipped/error file",
    )
    args = parser.parse_args()

    convert_interhuman(
        args.data_root,
        args.output,
        fps=args.fps,
        max_clips=args.max_clips,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
