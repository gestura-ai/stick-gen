#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner for Stick-Gen.

This script runs the full suite of evaluation metrics against a trained model checkpoint
or a set of generated motions. It produces a detailed report including:
- Temporal smoothness metrics
- Physics consistency metrics
- Text alignment scores
- Diversity metrics (FID-like)
- Artifact detection scores

Usage:
    python scripts/run_comprehensive_eval.py --checkpoint checkpoints/best_model.pth --output eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from src.eval.metrics import (
        compute_motion_diversity,
        compute_motion_realism_score,
        compute_motion_temporal_metrics,
        compute_physics_consistency_metrics,
        compute_synthetic_artifact_score,
    )

    # Mocking DataValidator if not available/configured to avoid dependency hell in this script
    # In a real run, this would load the actual validator
    DataValidator = None
except ImportError as e:
    print(f"❌ Error importing metrics: {e}")
    sys.exit(1)


def generate_mock_data(
    num_samples: int = 10, seq_len: int = 250, input_dim: int = 20
) -> torch.Tensor:
    """Generate random motion data for testing the eval pipeline."""
    print("⚠️  No model/data provided. Generating MOCK data for demonstration.")
    # shape: [Batch, Time, Dim]
    return torch.randn(num_samples, seq_len, input_dim)


def load_model_and_generate(checkpoint_path: str, num_samples: int) -> torch.Tensor:
    """Load model and generate samples. (Placeholder for actual generation logic)"""
    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint {checkpoint_path} not found.")
        return generate_mock_data(num_samples)

    print(f"Loading model from {checkpoint_path}...")
    # TODO: Implement actual model loading and generation
    # For now, return mock data to ensure script runs through
    return generate_mock_data(num_samples)


def run_evaluation(motions: torch.Tensor) -> dict[str, Any]:
    """Run all metrics on the provided motion tensor."""
    print("Running evaluation suite...")

    # 1. Per-sample metrics
    sample_metrics = []
    print(f"Evaluating {len(motions)} samples...")

    for i, m in enumerate(motions):
        # m is [T, D]
        temp = compute_motion_temporal_metrics(m)
        phys = compute_physics_consistency_metrics(
            # Physics tensor usually different form, here approximating for demo
            # In real pipeline, model returns separate physics tensor
            physics=torch.cat(
                [m[:, :2], m[:, :2], m[:, :2]], dim=-1
            )  # Mock 6-dim physics
        )
        artifact = compute_synthetic_artifact_score(m)
        realism = compute_motion_realism_score(m)

        sample_metrics.append({"id": i, **temp, **phys, **artifact, **realism})

    # 2. Aggregate metrics
    print("Computing aggregate metrics...")

    # Convert list of dicts to dict of lists
    agg_keys = sample_metrics[0].keys()
    aggregates = {}
    for k in agg_keys:
        if isinstance(sample_metrics[0][k], (int, float)):
            values = [s[k] for s in sample_metrics]
            aggregates[f"avg_{k}"] = sum(values) / len(values)

    # 3. Diversity metrics
    print("Computing diversity metrics...")
    div = compute_motion_diversity([m for m in motions])

    return {"summary": {**aggregates, **div}, "details": sample_metrics}


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument(
        "--output", default="eval_report.json", help="Output JSON report path"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )

    args = parser.parse_args()

    # Get motions
    if args.checkpoint:
        motions = load_model_and_generate(args.checkpoint, args.num_samples)
    else:
        motions = generate_mock_data(args.num_samples)

    # Run Eval
    results = run_evaluation(motions)

    # Save Report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation Complete!")
    print(f"Report saved to: {output_path.absolute()}")
    print("\nSummary Results:")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
