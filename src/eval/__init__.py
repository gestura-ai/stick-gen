"""Evaluation and metrics toolkit for Stick-Gen.

This package provides reusable metrics for motion quality,
physics consistency, camera behavior, and text / embedding
alignment. High-level scripts (e.g. scripts/evaluate.py)
are expected to call into these helpers.

Key modules:
- metrics: Motion temporal metrics, camera metrics, physics consistency
- reporting: HTML report generation
- safety_critic: Robustness checking and degenerate output detection
"""

from src.eval.metrics import (
    compute_camera_metrics,
    compute_motion_temporal_metrics,
    compute_physics_consistency_metrics,
    compute_text_alignment_from_embeddings,
)
from src.eval.safety_critic import (
    SafetyCritic,
    SafetyCriticConfig,
    SafetyCriticResult,
    SafetyIssue,
    SafetyIssueType,
    batch_evaluate_safety,
    evaluate_motion_safety,
)

__all__ = [
    # Metrics
    "compute_motion_temporal_metrics",
    "compute_camera_metrics",
    "compute_physics_consistency_metrics",
    "compute_text_alignment_from_embeddings",
    # Safety Critic
    "SafetyCritic",
    "SafetyCriticConfig",
    "SafetyCriticResult",
    "SafetyIssue",
    "SafetyIssueType",
    "evaluate_motion_safety",
    "batch_evaluate_safety",
]
