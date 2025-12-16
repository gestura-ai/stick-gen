#!/usr/bin/env python3
"""
Stick-Gen Evaluation Pipeline
Gestura AI - https://gestura.ai

Comprehensive evaluation metrics:
- MSE (Mean Squared Error) for pose reconstruction
- Temporal consistency (motion smoothness)
- Action accuracy (action prediction)
- Physics validation (velocity, acceleration, momentum)
- FID-like motion quality score

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data data/train_data_final.pt
"""

import os
import sys
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import StickFigureTransformer
from src.data_gen.schema import NUM_ACTIONS, IDX_TO_ACTION
from src.eval.metrics import (
    compute_motion_temporal_metrics,
    compute_physics_consistency_metrics,
    compute_camera_metrics,
)


class EvaluationDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(self, data_path: str):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item["motion"]
        embedding = item["embedding"]
        actions = item.get("actions", torch.zeros(motion.shape[0], dtype=torch.long))
        physics = item.get("physics", torch.zeros(motion.shape[0], 6))
        camera = item.get("camera", torch.zeros(motion.shape[0], 3))

        return (
            motion[:-1],
            embedding,
            motion[1:],
            actions[:-1],
            physics[:-1],
            camera[:-1],
        )


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Squared Error."""
    return nn.MSELoss()(predictions, targets).item()


def compute_temporal_consistency(predictions: torch.Tensor) -> dict:
    """Compute temporal consistency metrics using src.eval.metrics.

    ``predictions`` is expected in [seq, batch, dim] format as produced by the
    model. We adapt it to the canonical [..., T, D] shape expected by
    ``compute_motion_temporal_metrics``.
    """

    # [seq, batch, dim] -> [batch, seq, dim]
    metrics = compute_motion_temporal_metrics(predictions.permute(1, 0, 2))
    return metrics


def compute_action_accuracy(
    action_logits: torch.Tensor, action_targets: torch.Tensor
) -> dict:
    """Compute action prediction accuracy."""
    # action_logits: [seq, batch, num_actions]
    # action_targets: [batch, seq]

    predictions = action_logits.argmax(dim=-1)  # [seq, batch]
    predictions = predictions.permute(1, 0)  # [batch, seq]

    correct = (predictions == action_targets).float()

    # Per-action accuracy
    action_accuracies = {}
    for action_idx in range(NUM_ACTIONS):
        mask = action_targets == action_idx
        if mask.sum() > 0:
            action_name = IDX_TO_ACTION.get(action_idx, f"action_{action_idx}")
            action_accuracies[action_name] = correct[mask].mean().item()

    return {
        "overall_accuracy": correct.mean().item(),
        "per_action_accuracy": action_accuracies,
    }


def compute_physics_metrics(
    physics_output: torch.Tensor, physics_targets: torch.Tensor
) -> dict:
    """Compute physics validation metrics.

    This preserves the original MSE-based metrics and augments them with the
    validator-backed consistency metrics from ``src.eval.metrics``.
    """

    # Extract components
    pred_v = physics_output[:, :, :2]  # velocity
    pred_a = physics_output[:, :, 2:4]  # acceleration
    pred_m = physics_output[:, :, 4:6]  # momentum

    target_v = physics_targets[:, :, :2]
    target_a = physics_targets[:, :, 2:4]
    target_m = physics_targets[:, :, 4:6]

    # MSE for each component
    velocity_mse = nn.MSELoss()(pred_v, target_v).item()
    acceleration_mse = nn.MSELoss()(pred_a, target_a).item()
    momentum_mse = nn.MSELoss()(pred_m, target_m).item()

    # Gravity consistency (ay should be ~-9.8 when airborne)
    pred_ay = physics_output[:, :, 3]
    gravity_error = torch.abs(pred_ay - (-9.8)).mean().item()

    # Velocity-acceleration consistency
    dt = 1.0 / 25.0
    expected_v = pred_v[:-1] + pred_a[:-1] * dt
    consistency_error = nn.MSELoss()(pred_v[1:], expected_v).item()

    # Momentum conservation
    momentum_change = torch.norm(pred_m[1:] - pred_m[:-1], dim=-1)
    momentum_conservation = momentum_change.mean().item()

    base_metrics = {
        "velocity_mse": velocity_mse,
        "acceleration_mse": acceleration_mse,
        "momentum_mse": momentum_mse,
        "gravity_error": gravity_error,
        "consistency_error": consistency_error,
        "momentum_conservation": momentum_conservation,
        "physics_score": 1.0 / (1.0 + velocity_mse + acceleration_mse + momentum_mse),
    }

    # Validator-backed physics consistency metrics on the *predicted* physics.
    validator_stats = compute_physics_consistency_metrics(physics_output)
    base_metrics["validator"] = validator_stats
    return base_metrics


def compute_diversity_metrics(all_predictions: list) -> dict:
    """Compute diversity of generated motions."""
    if len(all_predictions) < 2:
        return {"diversity_score": 0.0}

    # Stack all predictions
    stacked = torch.stack(all_predictions)  # [num_samples, seq, dim]

    # Compute pairwise distances
    flat = stacked.reshape(len(all_predictions), -1)  # [num_samples, seq*dim]

    # Sample pairs for efficiency
    num_pairs = min(1000, len(all_predictions) * (len(all_predictions) - 1) // 2)
    distances = []

    for _ in range(num_pairs):
        i, j = np.random.choice(len(all_predictions), 2, replace=False)
        dist = torch.norm(flat[i] - flat[j]).item()
        distances.append(dist)

    return {
        "mean_pairwise_distance": np.mean(distances),
        "std_pairwise_distance": np.std(distances),
        "diversity_score": np.mean(distances) / (np.std(distances) + 1e-6),
    }


def evaluate_model(model, loader, device) -> dict:
    """Run full evaluation on a dataset."""
    model.eval()

    all_mse = []
    all_temporal = []
    all_action_correct = []
    all_physics = []
    all_camera = []
    all_predictions = []

    with torch.no_grad():
        for motion, embedding, targets, actions, physics, camera in tqdm(
            loader, desc="Evaluating"
        ):
            motion = motion.permute(1, 0, 2).to(device)
            embedding = embedding.to(device)
            targets = targets.permute(1, 0, 2).to(device)
            actions = actions.to(device)
            physics = physics.permute(1, 0, 2).to(device)
            camera = camera.permute(1, 0, 2).to(device)

            outputs = model(
                motion, embedding, return_all_outputs=True, camera_data=camera
            )

            # MSE
            mse = compute_mse(outputs["pose"], targets)
            all_mse.append(mse)

            # Temporal consistency
            temporal = compute_temporal_consistency(outputs["pose"])
            all_temporal.append(temporal)

            # Action accuracy
            if "action_logits" in outputs:
                action_metrics = compute_action_accuracy(
                    outputs["action_logits"], actions
                )
                all_action_correct.append(action_metrics["overall_accuracy"])

            # Physics
            if "physics" in outputs:
                phys_metrics = compute_physics_metrics(outputs["physics"], physics)
                all_physics.append(phys_metrics)

            # Camera metrics (per-sample, using eval toolkit)
            # camera: [seq, batch, 3]
            for b in range(camera.shape[1]):
                cam_seq = camera[:, b, :]
                cam_stats = compute_camera_metrics(cam_seq)
                all_camera.append(cam_stats)

            # Store predictions for diversity
            all_predictions.append(outputs["pose"].cpu())

    # Aggregate metrics
    results = {
        "mse": {
            "mean": np.mean(all_mse),
            "std": np.std(all_mse),
            "min": np.min(all_mse),
            "max": np.max(all_mse),
        },
        "temporal_consistency": {
            "smoothness_score": np.mean([t["smoothness_score"] for t in all_temporal]),
            "mean_velocity": np.mean([t["mean_velocity"] for t in all_temporal]),
            "mean_jerk": np.mean([t["mean_jerk"] for t in all_temporal]),
        },
    }

    if all_action_correct:
        results["action_accuracy"] = {
            "mean": np.mean(all_action_correct),
            "std": np.std(all_action_correct),
        }

    if all_physics:
        results["physics"] = {
            "velocity_mse": np.mean([p["velocity_mse"] for p in all_physics]),
            "physics_score": np.mean([p["physics_score"] for p in all_physics]),
            "gravity_error": np.mean([p["gravity_error"] for p in all_physics]),
            # Aggregate validator-backed metrics
            "validator_score": np.mean(
                [p["validator"]["validator_score"] for p in all_physics]
            ),
            "validator_valid_fraction": np.mean(
                [1.0 if p["validator"]["is_valid"] else 0.0 for p in all_physics]
            ),
        }

    if all_camera:
        # Shot/motion type distributions and stability stats
        shot_types = [m["shot_type"] for m in all_camera]
        motion_types = [m["motion_type"] for m in all_camera]

        shot_counts = Counter(shot_types)
        motion_counts = Counter(motion_types)
        total = max(len(all_camera), 1)

        results["camera"] = {
            "mean_stability_score": np.mean([m["stability_score"] for m in all_camera]),
            "mean_zoom_range": np.mean([m["zoom_range"] for m in all_camera]),
            "shot_type_distribution": {k: v / total for k, v in shot_counts.items()},
            "motion_type_distribution": {
                k: v / total for k, v in motion_counts.items()
            },
        }

    # Diversity
    flat_predictions = [
        p.reshape(-1, p.shape[-1]) for batch in all_predictions for p in batch
    ]
    if len(flat_predictions) > 10:
        results["diversity"] = compute_diversity_metrics(flat_predictions[:100])

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stick-Gen model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/train_data_final.pt",
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Data split to evaluate on",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Stick-Gen Evaluation Pipeline")
    print("by Gestura AI")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("config", {})
    if not model_config:
        # Default config
        model_config = {
            "input_dim": 20,
            "d_model": 384,
            "nhead": 12,
            "num_layers": 8,
            "output_dim": 20,
            "embedding_dim": 1024,
            "dropout": 0.1,
            "num_actions": NUM_ACTIONS,
        }

    # Initialize model
    model = StickFigureTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    print(f"\nLoading data: {args.data}")
    dataset = EvaluationDataset(args.data)

    # Split data
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # Reproducible split
    )

    if args.split == "train":
        eval_set = train_set
    elif args.split == "val":
        eval_set = val_set
    elif args.split == "test":
        eval_set = test_set
    else:
        eval_set = dataset

    print(f"Evaluating on {args.split} split: {len(eval_set)} samples")

    loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_model(model, loader, device)

    # Add metadata
    results["metadata"] = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "split": args.split,
        "num_samples": len(eval_set),
        "model_params": sum(p.numel() for p in model.parameters()),
    }

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nMSE: {results['mse']['mean']:.6f} ± {results['mse']['std']:.6f}")
    print(
        f"Smoothness Score: {results['temporal_consistency']['smoothness_score']:.4f}"
    )

    if "action_accuracy" in results:
        print(f"Action Accuracy: {results['action_accuracy']['mean']:.4f}")

    if "physics" in results:
        print(f"Physics Score: {results['physics']['physics_score']:.4f}")

    if "diversity" in results:
        print(f"Diversity Score: {results['diversity']['diversity_score']:.4f}")

    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
