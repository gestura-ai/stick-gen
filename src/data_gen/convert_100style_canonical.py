"""Canonical 100STYLE converter

Loads the text-based 100STYLE representation from ``data/100Style`` and
exports canonical samples compatible with ``docs/features/DATA_SCHEMA.md``.

This focuses on the public text format (InputTrain.txt / Tr_Va_Te_Frames.txt)
so we do not depend on BVH parsing libraries at runtime.
"""

import os
from typing import Any

import numpy as np
import torch

from .convert_amass import compute_basic_physics
from .schema import ACTION_TO_IDX, ActionType
from .validator import DataValidator


def _load_100style_txt(
    style_dir: str,
    target_frames: int = 250,
    max_sequences: int = 500,
) -> list[dict[str, Any]]:
    """Load 100STYLE motions from the public text format.

    This is a trimmed-down version of ``load_100style_txt`` in
    ``scripts/prepare_data.py`` that only keeps the information we need
    for canonical export.
    """
    input_file = os.path.join(style_dir, "InputTrain.txt")
    frames_file = os.path.join(style_dir, "Tr_Va_Te_Frames.txt")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"InputTrain.txt not found in {style_dir}")

    # Optional: frame ranges (not strictly required for fixed-length slices)
    frame_ranges: list[tuple[int, int]] = []
    if os.path.exists(frames_file):
        with open(frames_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        start, end = int(parts[0]), int(parts[1])
                        frame_ranges.append((start, end))
                    except ValueError:
                        continue

    # Load a bounded number of rows so we do not read the full ~21GB file
    try:
        max_rows = target_frames * max_sequences
        data = np.loadtxt(input_file, dtype=np.float32, max_rows=max_rows)
    except Exception as e:  # pragma: no cover - defensive path
        raise RuntimeError(
            f"Failed to load 100STYLE InputTrain.txt: {e}"
        ) from e

    if data.ndim != 2 or data.shape[0] < target_frames:
        raise ValueError(f"Unexpected 100STYLE data shape: {data.shape}")

    # 100STYLE format: 48 trajectory cols + remaining bone features
    if data.shape[1] <= 48:
        raise ValueError(
            f"Expected > 48 columns (trajectory + bones), got {data.shape[1]}"
        )

    bone_data = data[:, 48:]
    num_sequences = bone_data.shape[0] // target_frames
    num_sequences = min(num_sequences, max_sequences)

    sequences: list[dict[str, Any]] = []
    for i in range(num_sequences):
        start = i * target_frames
        end = start + target_frames
        seq = bone_data[start:end]
        if seq.shape[0] != target_frames:
            continue

        # Take first 20 features and reshape into [T, 20]
        if seq.shape[1] >= 20:
            motion_np = seq[:, :20].copy()
        else:  # pad
            motion_np = np.zeros((target_frames, 20), dtype=np.float32)
            motion_np[:, : seq.shape[1]] = seq

        motion = torch.from_numpy(motion_np).float()  # [T, 20]
        sequences.append(
            {
                "motion": motion,
                "style": f"style_{i}",
                "source": "100style_txt",
                "sequence_idx": i,
                "frame_range": frame_ranges[i] if i < len(frame_ranges) else None,
            }
        )

    return sequences


_STYLE_DESCRIPTIONS: dict[str, str] = {
    "angry": "A person moving with angry, aggressive body language",
    "happy": "A person moving with happy, joyful energy",
    "sad": "A person moving with sad, dejected posture",
    "neutral": "A person walking with neutral body language",
    "proud": "A person walking with proud, confident posture",
    "tired": "A person moving with tired, exhausted movements",
    "old": "An elderly person walking slowly",
    "young": "A young person moving with energetic steps",
    "drunk": "A person stumbling with unsteady movements",
    "sneaky": "A person moving stealthily and cautiously",
}


def _style_to_action(style: str) -> ActionType:
    """Map 100STYLE style labels to an ActionType.

    100STYLE is primarily locomotion with different styles, so we treat
    all of them as WALK for now and keep the fine-grained style in
    ``meta``.
    """
    return ActionType.WALK


def _build_canonical_sample(item: dict[str, Any], fps: int = 25) -> dict[str, Any]:
    motion = item["motion"]
    if not isinstance(motion, torch.Tensor):
        motion = torch.as_tensor(motion, dtype=torch.float32)
    if motion.dim() == 1:
        motion = motion.view(-1, 20)

    T = motion.shape[0]
    style = str(item.get("style", "neutral")).lower()

    # Map numeric style labels to the description dictionary in a stable way
    if style.startswith("style_"):
        try:
            idx = int(style.split("_")[1])
        except ValueError:
            idx = 0
        keys = sorted(_STYLE_DESCRIPTIONS.keys())
        style = keys[idx % len(keys)]

    description = _STYLE_DESCRIPTIONS.get(
        style, f"A person performing {style} movement"
    )

    action = _style_to_action(style)
    action_idx = ACTION_TO_IDX[action]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    physics = compute_basic_physics(motion, fps=fps)

    sample: dict[str, Any] = {
        "description": description,
        "motion": motion,
        "actions": actions,
        "physics": physics,
        "camera": torch.zeros(T, 3),
        "source": "100style",
        "meta": {
            "style": style,
            "raw_source": item.get("source", "100style_txt"),
            "sequence_idx": item.get("sequence_idx"),
            "frame_range": item.get("frame_range"),
        },
    }
    return sample


def convert_100style_canonical(
    style_dir: str = "data/100Style",
    output_path: str = "data/motions_processed/100style/canonical.pt",
    target_frames: int = 250,
    fps: int = 25,
    max_sequences: int = 500,
) -> list[dict[str, Any]]:
    """Convert 100STYLE into the canonical schema and save to disk."""
    sequences = _load_100style_txt(style_dir, target_frames, max_sequences)
    validator = DataValidator(fps=fps)

    samples: list[dict[str, Any]] = []
    for item in sequences:
        sample = _build_canonical_sample(item, fps=fps)
        # Only enforce physics constraints at conversion time; skeleton length
        # consistency is difficult to interpret from these engineered features.
        ok, _, reason = validator.check_physics_consistency(sample["physics"])
        if not ok:
            # Skip invalid samples but keep going
            # (detailed reason can be printed here if needed)
            continue
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(samples, output_path)
    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 100STYLE (txt) to canonical schema",
    )
    parser.add_argument("--style-dir", type=str, default="data/100Style")
    parser.add_argument(
        "--output",
        type=str,
        default="data/motions_processed/100style/canonical.pt",
    )
    parser.add_argument("--target-frames", type=int, default=250)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-sequences", type=int, default=500)
    args = parser.parse_args()

    convert_100style_canonical(
        style_dir=args.style_dir,
        output_path=args.output,
        target_frames=args.target_frames,
        fps=args.fps,
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()

