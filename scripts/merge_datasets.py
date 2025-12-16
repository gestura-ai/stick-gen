#!/usr/bin/env python3
"""Merge multiple motion datasets into unified training data.

This script combines motion data from various sources (HumanML3D, KIT-ML, BABEL,
BEAT, AMASS, AIST++, synthetic, etc.) into a single dataset with:
- Source balancing to prevent any single source from dominating
- Quality filtering using artifact detection and realism scoring
- Action label distribution reporting
- Optional FID-like statistics for quality assessment

Usage:
    python scripts/merge_datasets.py \
        --inputs data/humanml3d.pt data/kit_ml.pt data/babel.pt \
        --output data/merged_dataset.pt \
        --balance-sources \
        --max-source-fraction 0.3

Example with all sources:
    python scripts/merge_datasets.py \
        --inputs \
            data/humanml3d.pt \
            data/kit_ml.pt \
            data/babel.pt \
            data/beat.pt \
            data/amass.pt \
            data/aist_plusplus.pt \
            data/synthetic.pt \
        --output data/merged_all.pt \
        --balance-sources \
        --filter-artifacts \
        --compute-stats
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch

from src.data_gen.curation import (
    CurationConfig,
    _get_source,
    balance_by_source,
    filter_by_artifacts,
    filter_by_length,
)
from src.eval.metrics import (
    compute_dataset_fid_statistics,
    compute_motion_diversity,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load a .pt dataset file."""
    if not Path(path).exists():
        logger.warning(f"Dataset not found: {path}")
        return []

    try:
        data = torch.load(path, weights_only=False)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            logger.warning(f"Unexpected format in {path}")
            return []
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return []


def merge_datasets(
    input_paths: list[str],
    output_path: str,
    balance_sources: bool = True,
    max_source_fraction: float = 0.3,
    filter_artifacts_enabled: bool = False,
    max_artifact_score: float = 0.4,
    min_frames: int = 25,
    max_frames: int = 500,
    compute_stats: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge multiple datasets with optional filtering and balancing.

    Returns:
        Statistics dictionary
    """
    rng = random.Random(seed)
    all_samples: list[dict[str, Any]] = []
    source_counts_input: dict[str, int] = {}

    # Load all datasets
    for path in input_paths:
        logger.info(f"Loading {path}...")
        samples = load_dataset(path)

        if samples:
            # Tag samples with their file source if not already tagged
            for s in samples:
                if "source" not in s:
                    # Infer source from filename
                    fname = Path(path).stem.lower()
                    s["source"] = fname

            all_samples.extend(samples)

            # Count by source
            for s in samples:
                src = _get_source(s)
                source_counts_input[src] = source_counts_input.get(src, 0) + 1

    logger.info(f"Loaded {len(all_samples)} total samples")
    logger.info(f"Input source distribution: {source_counts_input}")

    # Create config for filtering
    cfg = CurationConfig(
        min_frames=min_frames,
        max_frames=max_frames,
        max_artifact_score=max_artifact_score,
        balance_by_source=balance_sources,
        max_source_fraction=max_source_fraction,
    )

    # Apply length filter
    filtered, dropped_length = filter_by_length(all_samples, cfg)
    logger.info(
        f"After length filter: {len(filtered)} samples ({dropped_length} dropped)"
    )

    # Apply artifact filter if enabled
    dropped_artifacts = 0
    if filter_artifacts_enabled:
        filtered, dropped_artifacts = filter_by_artifacts(filtered, cfg)
        logger.info(
            f"After artifact filter: {len(filtered)} samples ({dropped_artifacts} dropped)"
        )

    # Apply source balancing if enabled
    if balance_sources:
        filtered = balance_by_source(filtered, cfg, rng)
        logger.info(f"After source balancing: {len(filtered)} samples")
    else:
        rng.shuffle(filtered)

    # Compute output statistics
    source_counts_output: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for s in filtered:
        src = _get_source(s)
        source_counts_output[src] = source_counts_output.get(src, 0) + 1

        action = s.get("action_label", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

    logger.info(f"Output source distribution: {source_counts_output}")
    logger.info(f"Output action distribution: {action_counts}")

    # Compute optional quality statistics
    diversity_stats = None
    if compute_stats and len(filtered) > 10:
        logger.info("Computing quality statistics...")
        motions = [s["motion"] for s in filtered[:1000] if "motion" in s]
        if motions:
            diversity_stats = compute_motion_diversity(motions)
            compute_dataset_fid_statistics(motions)
            logger.info(
                f"Diversity score: {diversity_stats.get('diversity_score', 0):.4f}"
            )

    # Save merged dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(filtered, output_path)
    logger.info(f"Saved {len(filtered)} samples to {output_path}")

    # Build stats dictionary
    stats = {
        "total_input": len(all_samples),
        "total_output": len(filtered),
        "dropped_length": dropped_length,
        "dropped_artifacts": dropped_artifacts,
        "source_distribution_input": source_counts_input,
        "source_distribution_output": source_counts_output,
        "action_distribution": action_counts,
        "config": {
            "balance_sources": balance_sources,
            "max_source_fraction": max_source_fraction,
            "filter_artifacts": filter_artifacts_enabled,
            "max_artifact_score": max_artifact_score,
            "min_frames": min_frames,
            "max_frames": max_frames,
        },
    }

    if diversity_stats:
        stats["diversity"] = {
            "score": diversity_stats.get("diversity_score", 0),
            "std": diversity_stats.get("diversity_std", 0),
            "num_samples_analyzed": diversity_stats.get("num_samples", 0),
        }

    # Save stats JSON
    stats_path = Path(output_path).with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Stats saved to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple motion datasets into unified training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic merge
    python scripts/merge_datasets.py \\
        --inputs data/humanml3d.pt data/kit_ml.pt \\
        --output data/merged.pt

    # Merge with source balancing and artifact filtering
    python scripts/merge_datasets.py \\
        --inputs data/*.pt \\
        --output data/merged_filtered.pt \\
        --balance-sources \\
        --filter-artifacts \\
        --max-source-fraction 0.25

    # Merge with quality statistics
    python scripts/merge_datasets.py \\
        --inputs data/humanml3d.pt data/babel.pt data/beat.pt \\
        --output data/merged_quality.pt \\
        --compute-stats
        """,
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to .pt dataset files to merge",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--balance-sources",
        action="store_true",
        help="Balance samples across data sources",
    )
    parser.add_argument(
        "--max-source-fraction",
        type=float,
        default=0.3,
        help="Maximum fraction for any single source (default: 0.3)",
    )
    parser.add_argument(
        "--filter-artifacts",
        action="store_true",
        help="Filter samples with motion artifacts",
    )
    parser.add_argument(
        "--max-artifact-score",
        type=float,
        default=0.4,
        help="Maximum artifact score to keep (default: 0.4)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=25,
        help="Minimum sequence length in frames (default: 25)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum sequence length in frames (default: 500)",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute diversity and FID statistics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    merge_datasets(
        input_paths=args.inputs,
        output_path=args.output,
        balance_sources=args.balance_sources,
        max_source_fraction=args.max_source_fraction,
        filter_artifacts_enabled=args.filter_artifacts,
        max_artifact_score=args.max_artifact_score,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        compute_stats=args.compute_stats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
