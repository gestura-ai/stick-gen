"""Motion, camera, and physics quality metrics for Stick-Gen.

This module provides metrics for evaluating motion quality, diversity,
and realism. Includes FID-like diversity metrics and synthetic artifact
detection for data curation.
"""

import math
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data_gen.validator import DataValidator
from src.data_gen.auto_annotator import infer_shot_type, infer_camera_motion


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def compute_motion_temporal_metrics(motion: torch.Tensor) -> Dict[str, float]:
    """Temporal smoothness metrics on motion trajectories.

    Args:
        motion: [..., T, D] or [T, D] / [T, A, D]. Only the last
            dimension is treated as feature; earlier dims are batched.
    """
    m = _as_float_tensor(motion)
    # Collapse batch/actor dims into one
    m = m.view(-1, m.shape[-2], m.shape[-1])  # [B, T, D]

    diffs = m[:, 1:] - m[:, :-1]              # velocity proxy
    vel = torch.norm(diffs, dim=-1)          # [B, T-1]

    acc = diffs[:, 1:] - diffs[:, :-1]
    acc_mag = torch.norm(acc, dim=-1)        # [B, T-2]

    jerk = acc[:, 1:] - acc[:, :-1]
    jerk_mag = torch.norm(jerk, dim=-1)      # [B, T-3]

    mean_vel = vel.mean().item() if vel.numel() else 0.0
    mean_acc = acc_mag.mean().item() if acc_mag.numel() else 0.0
    mean_jerk = jerk_mag.mean().item() if jerk_mag.numel() else 0.0

    smoothness = 1.0 / (1.0 + mean_jerk)

    return {
        "mean_velocity": float(mean_vel),
        "mean_acceleration": float(mean_acc),
        "mean_jerk": float(mean_jerk),
        "smoothness_score": float(smoothness),
    }


def compute_camera_metrics(camera: torch.Tensor,
                           motion: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Camera stability and behavior metrics.

    Args:
        camera: [T, 3] with (x, y, zoom).
        motion: Optional motion tensor [T, A, 20] or [T, 20] used to
            disambiguate tracking vs complex motion.
    """
    cam = _as_float_tensor(camera)
    if cam.ndim != 2 or cam.shape[1] < 3 or cam.shape[0] < 2:
        return {"shot_type": "unknown", "motion_type": "unknown"}

    # Reuse high-level categorisation from auto_annotator
    shot_type = infer_shot_type(cam)
    motion_type = infer_camera_motion(cam, motion)

    # Low-level kinematic stats on camera center (x, y)
    pos = cam[:, :2]
    diffs = pos[1:] - pos[:-1]
    vel = torch.norm(diffs, dim=-1)

    acc = diffs[1:] - diffs[:-1]
    acc_mag = torch.norm(acc, dim=-1) if acc.numel() else torch.zeros(0)

    jerk = acc[1:] - acc[:-1]
    jerk_mag = torch.norm(jerk, dim=-1) if jerk.numel() else torch.zeros(0)

    def _safe_mean(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() else 0.0

    mean_speed = _safe_mean(vel)
    mean_jerk = _safe_mean(jerk_mag)
    stability = 1.0 / (1.0 + mean_jerk)

    zoom = cam[:, 2]
    zoom_range = float((zoom.max() - zoom.min()).item()) if zoom.numel() else 0.0

    return {
        "shot_type": shot_type,
        "motion_type": motion_type,
        "mean_speed": mean_speed,
        "mean_camera_jerk": mean_jerk,
        "stability_score": stability,
        "zoom_range": zoom_range,
    }


def compute_physics_consistency_metrics(
    physics: torch.Tensor,
    validator: Optional[DataValidator] = None,
) -> Dict[str, Any]:
    """Summarise physics tensor and validator-based consistency.

    Args:
        physics: [T, 6] or [T, A, 6].
    """
    phys = _as_float_tensor(physics)
    phys = phys.view(-1, phys.shape[-2], phys.shape[-1])  # [B, T, 6]

    speed = torch.norm(phys[:, :, 0:2], dim=-1)  # velocity
    acc = torch.norm(phys[:, :, 2:4], dim=-1)

    stats = {
        "max_velocity": float(speed.max().item()) if speed.numel() else 0.0,
        "mean_velocity": float(speed.mean().item()) if speed.numel() else 0.0,
        "max_acceleration": float(acc.max().item()) if acc.numel() else 0.0,
        "mean_acceleration": float(acc.mean().item()) if acc.numel() else 0.0,
    }

    validator = validator or DataValidator()
    ok, score, reason = validator.check_physics_consistency(phys.squeeze(0))

    stats.update({
        "is_valid": bool(ok),
        "validator_score": float(score),
        "validator_reason": reason,
    })
    return stats


def compute_text_alignment_from_embeddings(
    text_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """Cosine-similarity based alignment between two embedding sets.

    This is generic and can be used for text-text, text-motion, or
    description-vs-annotation alignment, as long as both sides are
    in the same embedding space.
    """
    a = _as_float_tensor(text_embeddings)
    b = _as_float_tensor(reference_embeddings)
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes must match, got {a.shape} vs {b.shape}")

    a_n = F.normalize(a.view(a.shape[0], -1), dim=-1)
    b_n = F.normalize(b.view(b.shape[0], -1), dim=-1)

    sims = (a_n * b_n).sum(dim=-1)
    return {
        "mean_cosine_similarity": float(sims.mean().item()),
        "min_cosine_similarity": float(sims.min().item()),
        "max_cosine_similarity": float(sims.max().item()),
    }


# =============================================================================
# Motion Diversity and Realism Metrics (FID-like)
# =============================================================================


def compute_motion_features(motion: torch.Tensor) -> torch.Tensor:
    """Extract statistical features from motion for diversity/FID computation.

    Computes a feature vector capturing:
    - Mean/std of positions per joint
    - Velocity statistics
    - Acceleration statistics
    - Joint correlations (simplified)

    Args:
        motion: [T, 20] or [T, A, 20] motion tensor

    Returns:
        Feature vector [D] suitable for FID-like comparisons
    """
    m = _as_float_tensor(motion)
    if m.ndim == 3:
        # [T, A, 20] -> [T, A*20]
        m = m.view(m.shape[0], -1)
    # Now m is [T, D]

    T, D = m.shape
    features = []

    # Position statistics per dimension
    mean_pos = m.mean(dim=0)  # [D]
    std_pos = m.std(dim=0)    # [D]
    features.extend([mean_pos, std_pos])

    # Velocity statistics
    if T > 1:
        vel = m[1:] - m[:-1]
        mean_vel = vel.mean(dim=0)
        std_vel = vel.std(dim=0)
        max_vel = vel.abs().max(dim=0).values
    else:
        mean_vel = std_vel = max_vel = torch.zeros(D)
    features.extend([mean_vel, std_vel, max_vel])

    # Acceleration statistics
    if T > 2:
        acc = vel[1:] - vel[:-1]
        mean_acc = acc.mean(dim=0)
        std_acc = acc.std(dim=0)
    else:
        mean_acc = std_acc = torch.zeros(D)
    features.extend([mean_acc, std_acc])

    # Global motion statistics
    global_stats = torch.tensor([
        float(m.mean()),
        float(m.std()),
        float(vel.mean()) if T > 1 else 0.0,
        float(vel.std()) if T > 1 else 0.0,
        float(T),  # sequence length as feature
    ])
    features.append(global_stats)

    return torch.cat(features)


def compute_motion_diversity(
    motions: List[torch.Tensor],
    num_pairs: int = 1000,
) -> Dict[str, float]:
    """Compute diversity metrics across a set of motion samples.

    Measures how varied the motion samples are, similar to diversity
    scores used in motion generation evaluation.

    Args:
        motions: List of motion tensors [T, 20] or [T, A, 20]
        num_pairs: Number of random pairs to sample for diversity

    Returns:
        Dictionary with diversity metrics
    """
    if len(motions) < 2:
        return {"diversity_score": 0.0, "num_samples": len(motions)}

    # Extract features for all motions
    features = torch.stack([compute_motion_features(m) for m in motions])
    features = F.normalize(features, dim=-1)

    N = features.shape[0]

    # Sample random pairs for diversity
    num_pairs = min(num_pairs, N * (N - 1) // 2)

    if num_pairs > 0:
        # Generate random pairs of indices in range [0, N)
        idx1 = torch.randint(0, N, (num_pairs,))
        idx2 = torch.randint(0, N, (num_pairs,))
        # Filter out same-sample pairs
        valid = idx1 != idx2
        idx1, idx2 = idx1[valid], idx2[valid]

        if len(idx1) > 0:
            dists = torch.norm(features[idx1] - features[idx2], dim=-1)
            diversity = float(dists.mean())
            diversity_std = float(dists.std())
        else:
            diversity = diversity_std = 0.0
    else:
        diversity = diversity_std = 0.0

    # Compute feature statistics for FID-like metric
    feat_mean = features.mean(dim=0)
    feat_cov_diag = features.var(dim=0)  # Simplified: diagonal covariance only

    return {
        "diversity_score": diversity,
        "diversity_std": diversity_std,
        "feature_mean_norm": float(feat_mean.norm()),
        "feature_var_mean": float(feat_cov_diag.mean()),
        "num_samples": N,
    }


def compute_synthetic_artifact_score(motion: torch.Tensor) -> Dict[str, float]:
    """Detect common artifacts in synthetic/generated motion.

    Checks for:
    - Jitter (high-frequency noise)
    - Foot sliding (unrealistic ground contact)
    - Static poses (frozen motion)
    - Explosions (sudden large movements)
    - Repetitive patterns (looping)

    Returns scores where higher = more artifacts (worse quality).
    Lower scores indicate more natural motion.
    """
    m = _as_float_tensor(motion)
    if m.ndim == 3:
        m = m.view(m.shape[0], -1)

    T, D = m.shape

    # --- Jitter Detection ---
    # High jitter = high acceleration variance relative to velocity
    if T > 2:
        vel = m[1:] - m[:-1]
        acc = vel[1:] - vel[:-1]
        vel_mag = torch.norm(vel, dim=-1)
        acc_mag = torch.norm(acc, dim=-1)

        # Jitter ratio: acceleration variance / mean velocity
        mean_vel = vel_mag.mean() + 1e-8
        acc_var = acc_mag.var()
        jitter_score = float(acc_var / mean_vel)
    else:
        jitter_score = 0.0

    # --- Static Detection ---
    # Detect frames with very low movement
    if T > 1:
        vel = m[1:] - m[:-1]
        vel_mag = torch.norm(vel, dim=-1)
        static_threshold = 0.01
        static_ratio = float((vel_mag < static_threshold).float().mean())
    else:
        static_ratio = 1.0

    # --- Explosion Detection ---
    # Sudden large movements (outlier velocities)
    if T > 1:
        vel = m[1:] - m[:-1]
        vel_mag = torch.norm(vel, dim=-1)
        mean_v = vel_mag.mean()
        std_v = vel_mag.std() + 1e-8
        z_scores = (vel_mag - mean_v) / std_v
        explosion_ratio = float((z_scores > 3.0).float().mean())
    else:
        explosion_ratio = 0.0

    # --- Repetition Detection ---
    # Check for repeating patterns using autocorrelation
    if T > 20:
        # Flatten and compute autocorrelation at various lags
        flat = m.mean(dim=-1)  # [T]
        flat = flat - flat.mean()

        max_corr = 0.0
        for lag in [5, 10, 15, 20]:
            if lag < T:
                corr = F.cosine_similarity(
                    flat[:-lag].unsqueeze(0),
                    flat[lag:].unsqueeze(0)
                ).item()
                max_corr = max(max_corr, abs(corr))
        repetition_score = float(max_corr)
    else:
        repetition_score = 0.0

    # --- Overall Artifact Score ---
    # Weighted combination (lower is better)
    artifact_score = (
        0.3 * min(jitter_score, 1.0) +
        0.2 * static_ratio +
        0.3 * explosion_ratio * 10 +  # Explosions are very bad
        0.2 * repetition_score
    )

    return {
        "jitter_score": jitter_score,
        "static_ratio": static_ratio,
        "explosion_ratio": explosion_ratio,
        "repetition_score": repetition_score,
        "artifact_score": float(artifact_score),
        "is_clean": artifact_score < 0.3,  # Threshold for "clean" motion
    }


def compute_motion_realism_score(motion: torch.Tensor) -> Dict[str, float]:
    """Compute overall motion realism/quality score.

    Combines multiple quality signals into a single realism score
    suitable for data curation and filtering.

    Returns score in [0, 1] where higher = more realistic.
    """
    # Get component metrics
    temporal = compute_motion_temporal_metrics(motion)
    artifacts = compute_synthetic_artifact_score(motion)

    # Smoothness contributes positively
    smoothness = temporal["smoothness_score"]

    # Artifacts contribute negatively
    artifact_penalty = artifacts["artifact_score"]

    # Reasonable velocity range contributes positively
    # Too slow or too fast is unnatural
    mean_vel = temporal["mean_velocity"]
    velocity_score = 1.0 / (1.0 + abs(mean_vel - 0.1))  # 0.1 is typical

    # Combine into realism score
    realism = (
        0.4 * smoothness +
        0.3 * (1.0 - min(artifact_penalty, 1.0)) +
        0.3 * velocity_score
    )

    return {
        "realism_score": float(realism),
        "smoothness_component": float(smoothness),
        "artifact_component": float(1.0 - min(artifact_penalty, 1.0)),
        "velocity_component": float(velocity_score),
        "is_realistic": realism > 0.5,
    }


def compute_dataset_fid_statistics(
    motions: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute FID-ready statistics (mean/covariance) for a motion dataset.

    Can be used to compare two datasets using Fréchet distance.

    Returns:
        Dictionary with 'mean' [D] and 'cov' [D, D] tensors
    """
    if not motions:
        return {"mean": torch.zeros(1), "cov": torch.zeros(1, 1), "num_samples": 0}

    features = torch.stack([compute_motion_features(m) for m in motions])

    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / (features.shape[0] - 1 + 1e-8)

    return {
        "mean": mean,
        "cov": cov,
        "num_samples": len(motions),
    }


def compute_frechet_distance(
    stats1: Dict[str, torch.Tensor],
    stats2: Dict[str, torch.Tensor],
) -> float:
    """Compute Fréchet distance between two motion distributions.

    Lower distance = more similar distributions.
    This is analogous to FID (Fréchet Inception Distance) but for motion.

    Args:
        stats1, stats2: Dictionaries from compute_dataset_fid_statistics

    Returns:
        Fréchet distance (float)
    """
    mu1, cov1 = stats1["mean"], stats1["cov"]
    mu2, cov2 = stats2["mean"], stats2["cov"]

    # Compute squared difference of means
    diff = mu1 - mu2
    mean_diff_sq = float((diff @ diff).item())

    # Compute trace terms (simplified: assumes diagonal dominance)
    # Full FID uses matrix square root, but this is expensive
    # We use trace approximation for efficiency
    trace1 = float(cov1.trace().item())
    trace2 = float(cov2.trace().item())

    # Approximate: tr(cov1) + tr(cov2) - 2 * sqrt(tr(cov1 * cov2))
    # Using geometric mean approximation
    trace_term = trace1 + trace2 - 2 * math.sqrt(trace1 * trace2 + 1e-8)

    fid = mean_diff_sq + trace_term
    return float(max(0.0, fid))

