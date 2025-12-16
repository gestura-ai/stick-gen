"""Convert KIT-ML dataset to Stick-Gen canonical format.

KIT-ML uses a similar feature layout to HumanML3D with 251 dimensions:
- Root features: dims 0-3
- Joint positions: dims 4-66 (21 joints × 3)
- Joint velocities: dims 67-195 (21 joints × 3 + others)
- Additional features: dims 196-250

We reuse the action inference from HumanML3D since both have text annotations.
"""

import glob
import logging
import os
from typing import Any

import numpy as np
import torch

from .convert_amass import compute_basic_physics
from .convert_humanml3d import (
    _features_to_stick as humanml3d_features_to_stick,
)
from .convert_humanml3d import (
    _infer_action_from_text,
)
from .schema import ACTION_TO_IDX
from .validator import DataValidator

logger = logging.getLogger(__name__)


def _load_normalization(stats_dir: str) -> dict[str, np.ndarray] | None:
    """Load normalization statistics with error handling."""
    mean_path = os.path.join(stats_dir, "Mean.npy")
    std_path = os.path.join(stats_dir, "Std.npy")

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        logger.warning(f"Missing normalization files in {stats_dir}")
        return None

    try:
        mean = np.load(mean_path)
        std = np.load(std_path)
        std = np.where(std < 1e-8, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    except Exception as e:
        logger.error(f"Failed to load normalization stats: {e}")
        return None


def _denorm(arr: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return arr * stats["std"] + stats["mean"]


def _features_to_stick(feats: np.ndarray) -> np.ndarray:
    """Map KIT-ML features to stick figure [T, 20].

    KIT-ML has 251 dimensions, similar structure to HumanML3D.
    We use the HumanML3D converter for compatible feature layouts.
    """
    T, D = feats.shape

    # KIT-ML has similar structure - use HumanML3D converter if dimensions match
    if D >= 67:
        return humanml3d_features_to_stick(feats)

    # Fallback for non-standard dimensions
    if D >= 20:
        arr = feats[:, :20]
    else:
        pad = np.zeros((T, 20 - D), dtype=feats.dtype)
        arr = np.concatenate([feats, pad], axis=1)
    return arr.astype(np.float32)


def _build_sample(
    feats: np.ndarray,
    texts: list[str],
    clip_id: str,
    fps: int = 30,
) -> dict[str, Any]:
    """Build canonical sample from KIT-ML features."""
    motion_np = _features_to_stick(feats)
    motion = torch.from_numpy(motion_np)
    physics = compute_basic_physics(motion, fps=fps)

    # Infer action from text
    action_enum = _infer_action_from_text(texts)
    action_idx = ACTION_TO_IDX[action_enum]
    T = motion.shape[0]
    actions = torch.full((T,), action_idx, dtype=torch.long)

    description = texts[0] if texts else f"A human motion clip from KIT-ML ({clip_id})."

    return {
        "description": description,
        "all_descriptions": texts,
        "motion": motion,
        "physics": physics,
        "actions": actions,
        "action_label": action_enum.value,
        "camera": None,
        "source": "kit_ml",
        "meta": {
            "clip_id": clip_id,
            "fps": fps,
            "num_frames": T,
            "feature_dim": feats.shape[1] if len(feats.shape) > 1 else 0,
        },
    }


def _load_texts(texts_dir: str) -> dict[str, list[str]]:
    """Load KIT-ML text annotations.

    Returns dict mapping clip_id to list of text descriptions.
    """
    mapping: dict[str, list[str]] = {}
    if not os.path.isdir(texts_dir):
        logger.warning(f"Texts directory not found: {texts_dir}")
        return mapping

    for path in glob.glob(os.path.join(texts_dir, "*.txt")):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            # Parse multiple annotations (one per line or # separated)
            lines = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    # Handle # delimited format
                    parts = line.split('#')
                    if parts[0].strip():
                        lines.append(parts[0].strip())
            mapping[clip_id] = lines
        except Exception as e:
            logger.debug(f"Failed to read {path}: {e}")

    return mapping


def convert_kit_ml(
    root_dir: str,
    output_path: str,
    fps: int = 30,
    max_clips: int = -1,
    physics_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Convert KIT-ML preprocessed features into canonical schema.

    Assumes a layout similar to HumanML3D: `new_joint_vecs/` feature vectors,
    `Mean.npy`, `Std.npy`, and per-clip text files in `texts/`.

    Args:
        root_dir: Root directory containing KIT-ML data
        output_path: Output .pt file path
        fps: Frame rate (default 30 for KIT-ML)
        max_clips: Maximum clips to process (-1 for all)
        physics_threshold: Physics validation threshold multiplier

    Returns:
        List of converted samples
    """
    logger.info(f"Converting KIT-ML from {root_dir}")

    stats = _load_normalization(root_dir)
    if stats is None:
        raise ValueError(f"Could not load normalization stats from {root_dir}")

    text_map = _load_texts(os.path.join(root_dir, "texts"))
    logger.info(f"Loaded {len(text_map)} text annotations")

    feats_dir = os.path.join(root_dir, "new_joint_vecs")
    if not os.path.exists(feats_dir):
        raise ValueError(f"Feature directory not found: {feats_dir}")

    paths = sorted(glob.glob(os.path.join(feats_dir, "*.npy")))
    logger.info(f"Found {len(paths)} feature files")

    if max_clips > 0:
        paths = paths[:max_clips]

    validator = DataValidator(fps=fps)
    # KIT-ML feature space has different magnitudes; adjust thresholds
    validator.max_velocity *= physics_threshold
    validator.max_acceleration *= physics_threshold

    samples: list[dict[str, Any]] = []
    skipped = 0

    for i, path in enumerate(paths):
        if i % 200 == 0:
            logger.info(f"Processing {i}/{len(paths)}...")

        clip_id = os.path.splitext(os.path.basename(path))[0]

        try:
            feats = np.load(path).astype(np.float32)

            if len(feats.shape) != 2 or feats.shape[0] < 10:
                skipped += 1
                continue

            feats_denorm = _denorm(feats, stats)
            texts = text_map.get(clip_id, [])
            sample = _build_sample(feats_denorm, texts, clip_id, fps=fps)

            ok, score, reason = validator.check_physics_consistency(sample["physics"])
            if not ok:
                logger.debug(f"Skipping {clip_id}: {reason}")
                skipped += 1
                continue

            samples.append(sample)

        except Exception as e:
            logger.warning(f"Error processing {clip_id}: {e}")
            skipped += 1

    logger.info(f"Converted {len(samples)}/{len(paths)} clips ({skipped} skipped)")

    # Report action distribution
    action_counts: dict[str, int] = {}
    for s in samples:
        label = s.get("action_label", "idle")
        action_counts[label] = action_counts.get(label, 0) + 1
    logger.info(f"Action distribution: {action_counts}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    logger.info(f"Saved to {output_path}")

    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert KIT-ML to Stick-Gen canonical schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of KIT-ML")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate (default: 30)")
    parser.add_argument("--max-clips", type=int, default=-1,
                        help="Maximum clips to process (-1 for all)")
    parser.add_argument("--physics-threshold", type=float, default=2.0,
                        help="Physics validation threshold multiplier")
    args = parser.parse_args()

    convert_kit_ml(
        args.root,
        args.output,
        fps=args.fps,
        max_clips=args.max_clips,
        physics_threshold=args.physics_threshold,
    )


if __name__ == "__main__":
    main()

