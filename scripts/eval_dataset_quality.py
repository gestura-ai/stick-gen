#!/usr/bin/env python3
"""Dataset-level quality evaluation for canonical Stick-Gen datasets.

This script runs motion, camera, and physics metrics from ``src.eval.metrics``
on a saved ``.pt`` dataset of canonical samples and produces a JSON summary
report.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import (  # noqa: E402
    compute_camera_metrics,
    compute_motion_temporal_metrics,
    compute_physics_consistency_metrics,
)


def evaluate_dataset(data_path: str, max_samples: int | None = None) -> dict:
    """Compute dataset-level motion, camera, and physics quality metrics."""
    data = torch.load(data_path)
    if max_samples is not None:
        data = data[:max_samples]

    motion_metrics = []
    camera_metrics = []
    physics_metrics = []

    for sample in tqdm(data, desc="Evaluating dataset"):
        motion = sample.get("motion")
        if motion is not None:
            motion_metrics.append(compute_motion_temporal_metrics(motion))

        camera = sample.get("camera")
        if camera is not None:
            camera_metrics.append(compute_camera_metrics(camera))

        physics = sample.get("physics")
        if physics is not None:
            physics_metrics.append(compute_physics_consistency_metrics(physics))

    results: dict[str, object] = {"num_samples": len(data)}

    if motion_metrics:
        results["motion"] = {
            "smoothness_score_mean": float(np.mean([m["smoothness_score"] for m in motion_metrics])),
            "smoothness_score_std": float(np.std([m["smoothness_score"] for m in motion_metrics])),
            "mean_velocity_mean": float(np.mean([m["mean_velocity"] for m in motion_metrics])),
            "mean_acceleration_mean": float(np.mean([m["mean_acceleration"] for m in motion_metrics])),
            "mean_jerk_mean": float(np.mean([m["mean_jerk"] for m in motion_metrics])),
        }

    if camera_metrics:
        shot_types = [m["shot_type"] for m in camera_metrics]
        motion_types = [m["motion_type"] for m in camera_metrics]
        shot_counts = Counter(shot_types)
        motion_counts = Counter(motion_types)
        total = max(len(camera_metrics), 1)

        results["camera"] = {
            "mean_stability_score": float(np.mean([m["stability_score"] for m in camera_metrics])),
            "mean_zoom_range": float(np.mean([m["zoom_range"] for m in camera_metrics])),
            "shot_type_distribution": {k: v / total for k, v in shot_counts.items()},
            "motion_type_distribution": {k: v / total for k, v in motion_counts.items()},
        }

    if physics_metrics:
        valid_flags = [1.0 if m["is_valid"] else 0.0 for m in physics_metrics]
        results["physics"] = {
            "validator_score_mean": float(np.mean([m["validator_score"] for m in physics_metrics])),
            "validator_valid_fraction": float(np.mean(valid_flags)),
            "mean_velocity_mean": float(np.mean([m["mean_velocity"] for m in physics_metrics])),
            "mean_acceleration_mean": float(np.mean([m["mean_acceleration"] for m in physics_metrics])),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dataset-level motion/camera/physics quality")
    parser.add_argument("--data", type=str, required=True, help="Path to canonical .pt dataset")
    parser.add_argument("--output", type=str, default="dataset_quality.json", help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples")

    args = parser.parse_args()

    print("=" * 60)
    print("Stick-Gen Dataset Quality Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.data}")

    results = evaluate_dataset(args.data, max_samples=args.max_samples)
    results["metadata"] = {"data": args.data, "max_samples": args.max_samples}

    print("\nMotion smoothness (mean):", results.get("motion", {}).get("smoothness_score_mean", "n/a"))
    print("Physics validator score (mean):", results.get("physics", {}).get("validator_score_mean", "n/a"))
    print("Camera stability (mean):", results.get("camera", {}).get("mean_stability_score", "n/a"))

    print(f"\nSaving report to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()

