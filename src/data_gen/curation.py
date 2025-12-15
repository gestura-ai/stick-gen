"""Dataset curation utilities for preparing pretraining and SFT splits.

This module operates on canonical stick-figure samples (list of dicts) that
already contain physics, actions, camera, and auto-annotator fields such as
``quality_score`` and ``annotations``.

Enhanced curation features:
- Synthetic artifact detection (jitter, explosions, static poses)
- Motion realism scoring
- Multi-source dataset balancing
- Quality tier classification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Optional

import random
import torch

from .validator import DataValidator
from .schema import ActionType
from src.eval.metrics import (
    compute_camera_metrics,
    compute_synthetic_artifact_score,
    compute_motion_realism_score,
)

logger = logging.getLogger(__name__)


@dataclass
class CurationConfig:
    """Configuration for dataset curation thresholds and balancing."""

    # Quality thresholds
    min_quality_pretrain: float = 0.5
    min_quality_sft: float = 0.8
    min_camera_stability_sft: float = 0.6

    # Realism thresholds (from new metrics)
    min_realism_pretrain: float = 0.3
    min_realism_sft: float = 0.6
    max_artifact_score: float = 0.5  # Reject samples with high artifact scores

    # Balancing
    balance_max_fraction: float = 0.3  # max fraction for any dominant action in SFT
    balance_by_source: bool = True  # Balance across dataset sources
    max_source_fraction: float = 0.4  # max fraction from any single source

    # Synthetic data handling
    synthetic_quality_penalty: float = 0.1  # Reduce quality score for synthetic
    prefer_mocap: bool = True  # Prefer motion capture over synthetic in SFT

    # Filtering
    min_frames: int = 25  # Minimum sequence length (1 second at 25fps)
    max_frames: int = 500  # Maximum sequence length (20 seconds)

    # Source weights for quality combination
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "humanml3d": 1.0,
        "kit_ml": 1.0,
        "amass": 0.95,
        "aist_plusplus": 0.9,
        "interhuman": 0.85,
        "ntu_rgbd": 0.8,
        "100style": 0.75,
        "babel": 1.0,
        "beat": 0.9,
        "synthetic": 0.6,
    })


def load_canonical_datasets(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load and concatenate canonical samples from one or more .pt files.

    Each path should contain either:
      - a list[dict] of samples; or
      - a dict with a "sequences" key pointing to that list.
    """

    all_samples: List[Dict[str, Any]] = []
    for p in paths:
        data = torch.load(p)
        if isinstance(data, dict) and "sequences" in data:
            seqs = data["sequences"]
        else:
            seqs = data
        if not isinstance(seqs, list):
            raise ValueError(f"Expected list of dicts in {p}, got {type(seqs)}")
        all_samples.extend(seqs)
    return all_samples


def _get_quality(sample: Dict[str, Any]) -> Optional[float]:
    if "quality_score" in sample:
        try:
            return float(sample["quality_score"])
        except Exception:
            return None
    ann = sample.get("annotations") or {}
    q = ann.get("quality") or {}
    score = q.get("score")
    return float(score) if score is not None else None


def _camera_stability(sample: Dict[str, Any]) -> Optional[float]:
    cam = sample.get("camera")
    if cam is None:
        return None
    try:
        metrics = compute_camera_metrics(torch.as_tensor(cam))
        return float(metrics.get("stability_score", 0.0))
    except Exception:
        return None


def _dominant_action(sample: Dict[str, Any]) -> str:
    ann = sample.get("annotations") or {}
    acts = ann.get("actions") or {}
    dom = acts.get("dominant") or []
    if dom:
        return str(dom[0])
    # Fallback: infer from per-frame tensor if available
    actions = sample.get("actions")
    if actions is None:
        return "unknown"
    try:
        tens = torch.as_tensor(actions, dtype=torch.long)
        if tens.ndim > 1:
            tens = tens.reshape(-1)
        if tens.numel() == 0:
            return "unknown"
        idx = int(torch.mode(tens).values.item())
        # Map index back to enum name when possible
        if 0 <= idx < len(ActionType):
            return list(ActionType)[idx].value
        return str(idx)
    except Exception:
        return "unknown"


def _get_source(sample: Dict[str, Any]) -> str:
    """Extract data source from sample."""
    source = sample.get("source", "unknown")
    if source in ["dataset_generator", "programmatic"]:
        return "synthetic"
    return str(source).lower()


def _get_sequence_length(sample: Dict[str, Any]) -> int:
    """Get number of frames in sample."""
    motion = sample.get("motion")
    if motion is None:
        return 0
    try:
        if hasattr(motion, 'shape'):
            return motion.shape[0]
        return len(motion)
    except Exception:
        return 0


def _compute_motion_quality(sample: Dict[str, Any]) -> Tuple[float, float]:
    """Compute realism score and artifact score for motion.

    Returns (realism_score, artifact_score).
    """
    motion = sample.get("motion")
    if motion is None:
        return 0.5, 0.0  # Neutral defaults

    try:
        motion_tensor = torch.as_tensor(motion, dtype=torch.float32)

        realism = compute_motion_realism_score(motion_tensor)
        artifacts = compute_synthetic_artifact_score(motion_tensor)

        return realism["realism_score"], artifacts["artifact_score"]
    except Exception:
        return 0.5, 0.0


def _compute_combined_quality(
    sample: Dict[str, Any],
    cfg: CurationConfig,
) -> Optional[float]:
    """Compute combined quality score using multiple signals.

    Combines:
    - Base quality score (from auto-annotator or manual)
    - Motion realism score
    - Source weight
    - Artifact penalty
    """
    base_q = _get_quality(sample)
    if base_q is None:
        # Try to compute from motion realism if no base quality
        realism, artifact = _compute_motion_quality(sample)
        if artifact > cfg.max_artifact_score:
            return None  # Too many artifacts
        base_q = realism

    # Get source weight
    source = _get_source(sample)
    source_weight = cfg.source_weights.get(source, 0.7)

    # Compute realism and artifact scores
    realism, artifact = _compute_motion_quality(sample)

    # Apply synthetic penalty
    if source == "synthetic":
        base_q -= cfg.synthetic_quality_penalty

    # Combine scores
    combined = (
        0.5 * base_q +
        0.3 * realism * source_weight +
        0.2 * (1.0 - min(artifact, 1.0))
    )

    return max(0.0, min(1.0, combined))


def filter_by_length(
    samples: Sequence[Dict[str, Any]],
    cfg: CurationConfig,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter samples by sequence length.

    Returns (filtered_samples, num_dropped).
    """
    filtered = []
    dropped = 0

    for s in samples:
        T = _get_sequence_length(s)
        if cfg.min_frames <= T <= cfg.max_frames:
            filtered.append(s)
        else:
            dropped += 1

    return filtered, dropped


def filter_by_artifacts(
    samples: Sequence[Dict[str, Any]],
    cfg: CurationConfig,
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter samples with too many motion artifacts.

    Returns (filtered_samples, num_dropped).
    """
    filtered = []
    dropped = 0

    for s in samples:
        _, artifact_score = _compute_motion_quality(s)
        if artifact_score <= cfg.max_artifact_score:
            filtered.append(s)
        else:
            dropped += 1

    return filtered, dropped


def balance_by_source(
    samples: List[Dict[str, Any]],
    cfg: CurationConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Balance samples across data sources.

    Ensures no single source dominates the dataset.
    """
    if not cfg.balance_by_source:
        return samples

    # Group by source
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        source = _get_source(s)
        buckets.setdefault(source, []).append(s)

    total = len(samples)
    max_per_source = int(cfg.max_source_fraction * total)

    balanced = []
    for source, bucket in buckets.items():
        rng.shuffle(bucket)
        take = min(len(bucket), max_per_source)
        balanced.extend(bucket[:take])

    rng.shuffle(balanced)
    return balanced


def curate_samples(
    samples: Sequence[Dict[str, Any]],
    cfg: CurationConfig,
    seed: int = 42,
    use_enhanced_filtering: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Curate samples into pretraining and SFT splits.

    Enhanced version with:
    - Sequence length filtering
    - Artifact detection and filtering
    - Multi-source balancing
    - Combined quality scoring

    Returns (pretrain, sft, stats).
    """
    rng = random.Random(seed)
    validator = DataValidator()

    total = len(samples)
    working_samples = list(samples)

    # Track dropped counts
    dropped_length = 0
    dropped_artifacts = 0
    dropped_missing_quality = 0
    dropped_invalid_physics = 0
    dropped_low_realism = 0

    # Enhanced filtering pipeline
    if use_enhanced_filtering:
        # 1. Filter by length
        working_samples, dropped_length = filter_by_length(working_samples, cfg)
        logger.info(f"Length filter: kept {len(working_samples)}, dropped {dropped_length}")

        # 2. Filter by artifacts (expensive, do sampling for large datasets)
        if len(working_samples) > 10000:
            # Sample-based filtering for large datasets
            sample_size = min(2000, len(working_samples))
            sample_indices = rng.sample(range(len(working_samples)), sample_size)
            artifact_samples = [working_samples[i] for i in sample_indices]
            _, artifact_ratio = filter_by_artifacts(artifact_samples, cfg)
            # Estimate and log
            est_artifacts = int(artifact_ratio / sample_size * len(working_samples))
            logger.info(f"Estimated artifact samples: ~{est_artifacts} (sampled {sample_size})")
        else:
            working_samples, dropped_artifacts = filter_by_artifacts(working_samples, cfg)
            logger.info(f"Artifact filter: kept {len(working_samples)}, dropped {dropped_artifacts}")

    pretrain: List[Dict[str, Any]] = []
    sft_candidates: List[Tuple[Dict[str, Any], str, float]] = []

    for s in working_samples:
        # Compute combined quality if enhanced filtering enabled
        if use_enhanced_filtering:
            q = _compute_combined_quality(s, cfg)
        else:
            q = _get_quality(s)

        if q is None:
            dropped_missing_quality += 1
            continue

        # Physics validation
        ok, _, _ = validator.validate(s)
        if not ok:
            dropped_invalid_physics += 1
            continue

        # Realism check for enhanced mode
        if use_enhanced_filtering:
            realism, _ = _compute_motion_quality(s)
            if realism < cfg.min_realism_pretrain:
                dropped_low_realism += 1
                continue

        if q >= cfg.min_quality_pretrain:
            pretrain.append(s)

        # SFT requires higher quality and realism
        if use_enhanced_filtering:
            realism, _ = _compute_motion_quality(s)
            meets_realism = realism >= cfg.min_realism_sft
        else:
            meets_realism = True

        stab = _camera_stability(s)
        meets_camera = stab is None or stab >= cfg.min_camera_stability_sft

        if q >= cfg.min_quality_sft and meets_realism and meets_camera:
            # Prefer mocap for SFT
            source = _get_source(s)
            if cfg.prefer_mocap and source == "synthetic":
                # Still include but with lower priority (handled in sorting)
                priority = q * 0.8
            else:
                priority = q
            dom_act = _dominant_action(s)
            sft_candidates.append((s, dom_act, priority))

    # Balance SFT by dominant action label
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s, label, _ in sft_candidates:
        buckets.setdefault(label, []).append(s)

    sft: List[Dict[str, Any]] = []
    total_sft_candidates = len(sft_candidates)
    if total_sft_candidates:
        max_per_action = max(1, int(cfg.balance_max_fraction * total_sft_candidates))
        for label, bucket in buckets.items():
            rng.shuffle(bucket)
            take = min(len(bucket), max_per_action)
            sft.extend(bucket[:take])

    # Balance by source if enabled
    if use_enhanced_filtering:
        pretrain = balance_by_source(pretrain, cfg, rng)
        sft = balance_by_source(sft, cfg, rng)
    else:
        rng.shuffle(pretrain)
        rng.shuffle(sft)

    def _split_stats(split: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not split:
            return {"num_samples": 0}
        qualities = [float(_get_quality(s) or 0.0) for s in split]
        mean_q = float(sum(qualities) / len(qualities)) if qualities else 0.0

        # Action distribution by dominant label
        action_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        for s in split:
            lab = _dominant_action(s)
            action_counts[lab] = action_counts.get(lab, 0) + 1
            src = _get_source(s)
            source_counts[src] = source_counts.get(src, 0) + 1

        total_c = float(len(split))
        action_dist = {k: v / total_c for k, v in action_counts.items()}
        source_dist = {k: v / total_c for k, v in source_counts.items()}

        return {
            "num_samples": len(split),
            "mean_quality": mean_q,
            "action_distribution": action_dist,
            "source_distribution": source_dist,
        }

    stats = {
        "total_input": total,
        "dropped_length": dropped_length,
        "dropped_artifacts": dropped_artifacts,
        "dropped_missing_quality": dropped_missing_quality,
        "dropped_invalid_physics": dropped_invalid_physics,
        "dropped_low_realism": dropped_low_realism,
        "pretrain": _split_stats(pretrain),
        "sft": _split_stats(sft),
    }

    return pretrain, sft, stats

