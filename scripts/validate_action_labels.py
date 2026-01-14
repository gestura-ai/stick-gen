#!/usr/bin/env python3
"""Validate action label distribution in datasets.

This script analyzes action label distribution in Stick-Gen datasets
to measure the effectiveness of embedding-based action classification.

Usage:
    python scripts/validate_action_labels.py data/processed/merged_canonical.pt
    python scripts/validate_action_labels.py data/processed/*.pt --compare
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_dataset(path: str) -> dict[str, Any]:
    """Analyze action label distribution in a dataset.

    Args:
        path: Path to .pt dataset file.

    Returns:
        Dictionary with analysis results.
    """
    logger.info(f"Loading {path}...")
    samples = torch.load(path, weights_only=False)

    if not samples:
        return {"path": path, "total": 0, "error": "Empty dataset"}

    total = len(samples)
    action_counts: Counter = Counter()
    source_counts: Counter = Counter()
    unknown_samples: list[dict] = []

    for sample in samples:
        action = sample.get("action_label", "unknown")
        if action is None:
            action = "unknown"
        action_counts[action] += 1

        source = sample.get("source", "unknown")
        source_counts[source] += 1

        # Track unknown samples for debugging
        if action == "unknown" and len(unknown_samples) < 10:
            unknown_samples.append({
                "source": source,
                "description": str(sample.get("description", ""))[:100],
            })

    unknown_count = action_counts.get("unknown", 0)
    unknown_pct = (unknown_count / total * 100) if total > 0 else 0

    return {
        "path": path,
        "total": total,
        "unknown_count": unknown_count,
        "unknown_pct": unknown_pct,
        "action_counts": dict(action_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "unknown_samples": unknown_samples,
    }


def print_analysis(result: dict[str, Any]) -> None:
    """Print analysis results in a readable format."""
    print(f"\n{'='*60}")
    print(f"Dataset: {result['path']}")
    print(f"{'='*60}")

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Total samples: {result['total']:,}")
    print(f"Unknown labels: {result['unknown_count']:,} ({result['unknown_pct']:.1f}%)")

    # Action distribution
    print(f"\nAction Distribution (top 15):")
    for i, (action, count) in enumerate(result["action_counts"].items()):
        if i >= 15:
            break
        pct = count / result["total"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {action:20s}: {count:6,} ({pct:5.1f}%) {bar}")

    # Source distribution
    print(f"\nSource Distribution:")
    for source, count in result["source_counts"].items():
        pct = count / result["total"] * 100
        print(f"  {source:25s}: {count:6,} ({pct:5.1f}%)")

    # Unknown samples
    if result["unknown_samples"]:
        print(f"\nSample unknown entries:")
        for i, sample in enumerate(result["unknown_samples"]):
            print(f"  [{i+1}] {sample['source']}: {sample['description']}")


def main():
    parser = argparse.ArgumentParser(description="Validate action label distribution")
    parser.add_argument("paths", nargs="+", help="Dataset .pt files to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare multiple datasets")
    args = parser.parse_args()

    results = []
    for path in args.paths:
        if not Path(path).exists():
            logger.warning(f"File not found: {path}")
            continue
        result = analyze_dataset(path)
        results.append(result)
        print_analysis(result)

    if args.compare and len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for r in results:
            name = Path(r["path"]).stem
            print(f"  {name:30s}: {r['unknown_pct']:5.1f}% unknown ({r['total']:,} samples)")


if __name__ == "__main__":
    main()

