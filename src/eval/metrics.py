"""Motion, camera, and physics quality metrics for Stick-Gen.

This module provides metrics for evaluating motion quality, diversity,
and realism. Includes FID-like diversity metrics and synthetic artifact
detection for data curation.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F

from src.data_gen.auto_annotator import infer_camera_motion, infer_shot_type
from src.data_gen.validator import DataValidator


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _compute_v3_foot_skate_metrics(
    motion_flat: torch.Tensor,
) -> tuple[float, float]:
    """Compute a simple foot-skate metric for canonical v3 48D motion.

    This helper expects motion laid out as ``[T, 48]`` where the last
    dimension corresponds to the 12 v3 segments (each ``[x1, y1, x2, y2]``).
    It returns a dimensionless score where ``0.0`` indicates no observable
    sliding of the ankles during ground-contact frames and larger values
    indicate more drift.

    Args:
        motion_flat: Motion tensor of shape ``[T, 48]``.

    Returns:
        (foot_skate_score, contact_ratio):
            - foot_skate_score: Normalised mean ankle drift during stance
              (0 = best, higher = worse).
            - contact_ratio: Approximate fraction of frames classified as foot
              contact (used to decide whether the score is meaningful).
    """

    m = _as_float_tensor(motion_flat)
    if m.ndim != 2 or m.shape[1] != 48 or m.shape[0] < 2:
        return 0.0, 0.0

    T, _D = m.shape
    segs = m.view(T, 12, 4)  # [T, S, 4]

    # Reconstruct joints needed for foot-skate: head and ankles.
    head_center = segs[:, 0, 2:4]
    l_ankle = segs[:, 8, 2:4]
    r_ankle = segs[:, 10, 2:4]

    # Estimate body height from head to ankles to obtain a scale for
    # normalising drift (so metric is roughly scale-invariant).
    head_to_l = torch.linalg.norm(head_center - l_ankle, dim=-1)
    head_to_r = torch.linalg.norm(head_center - r_ankle, dim=-1)
    heights = torch.maximum(head_to_l, head_to_r)
    finite_mask = torch.isfinite(heights) & (heights > 1e-6)
    if not finite_mask.any():
        return 0.0, 0.0

    height = torch.median(heights[finite_mask])
    if not torch.isfinite(height) or float(height.item()) <= 0.0:
        height = torch.tensor(1.0, dtype=heights.dtype, device=heights.device)

    def _foot_score(ankle: torch.Tensor) -> tuple[float, float]:
        # ankle: [T, 2]
        y = ankle[:, 1]
        if not torch.isfinite(y).all():
            return 0.0, 0.0

        y_min = y.min()
        # Treat frames near the global ankle minimum as approximate contact.
        y_tol = 0.05 * float(height.item())
        contact_mask = y <= (y_min + y_tol)

        if T < 2:
            return 0.0, float(contact_mask.float().mean().item())

        pair_mask = contact_mask[:-1] & contact_mask[1:]
        contact_ratio = float(contact_mask.float().mean().item())
        if pair_mask.sum() == 0:
            return 0.0, contact_ratio

        deltas = ankle[1:] - ankle[:-1]  # [T-1, 2]
        drift = torch.linalg.norm(deltas[pair_mask], dim=-1)
        mean_drift = drift.mean() / (height + 1e-6)
        return float(mean_drift.item()), contact_ratio

    l_score, l_contact = _foot_score(l_ankle)
    r_score, r_contact = _foot_score(r_ankle)

    if l_contact == 0.0 and r_contact == 0.0:
        return 0.0, 0.0

    foot_score = max(l_score, r_score)
    contact_ratio = max(l_contact, r_contact)
    return float(foot_score), float(contact_ratio)


def compute_motion_temporal_metrics(motion: torch.Tensor) -> dict[str, float]:
    """Temporal smoothness metrics on motion trajectories.

    Args:
        motion: [..., T, D] or [T, D] / [T, A, D]. Only the last
            dimension is treated as feature; earlier dims are batched.
    """
    m = _as_float_tensor(motion)
    # Collapse batch/actor dims into one
    m = m.view(-1, m.shape[-2], m.shape[-1])  # [B, T, D]

    diffs = m[:, 1:] - m[:, :-1]  # velocity proxy
    vel = torch.norm(diffs, dim=-1)  # [B, T-1]

    acc = diffs[:, 1:] - diffs[:, :-1]
    acc_mag = torch.norm(acc, dim=-1)  # [B, T-2]

    jerk = acc[:, 1:] - acc[:, :-1]
    jerk_mag = torch.norm(jerk, dim=-1)  # [B, T-3]

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


def compute_camera_metrics(
    camera: torch.Tensor, motion: torch.Tensor | None = None
) -> dict[str, Any]:
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
    torch.norm(acc, dim=-1) if acc.numel() else torch.zeros(0)

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
    validator: DataValidator | None = None,
) -> dict[str, Any]:
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

    stats.update(
        {
            "is_valid": bool(ok),
            "validator_score": float(score),
            "validator_reason": reason,
        }
    )
    return stats


def compute_text_alignment_from_embeddings(
    text_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> dict[str, float]:
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
    std_pos = m.std(dim=0)  # [D]
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
    global_stats = torch.tensor(
        [
            float(m.mean()),
            float(m.std()),
            float(vel.mean()) if T > 1 else 0.0,
            float(vel.std()) if T > 1 else 0.0,
            float(T),  # sequence length as feature
        ]
    )
    features.append(global_stats)

    return torch.cat(features)


def compute_motion_diversity(
    motions: list[torch.Tensor],
    num_pairs: int = 1000,
) -> dict[str, float]:
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


def compute_synthetic_artifact_score(motion: torch.Tensor) -> dict[str, float]:
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
                    flat[:-lag].unsqueeze(0), flat[lag:].unsqueeze(0)
                ).item()
                max_corr = max(max_corr, abs(corr))
        repetition_score = float(max_corr)
    else:
        repetition_score = 0.0

    # --- Foot-Skate Detection (v3-only, optional) ---
    # When motion is in canonical v3 layout ([T, 48]), compute a simple
    # ankle-drift metric during stance frames. For other dimensionalities the
    # score is 0 and does not affect legacy behaviour.
    foot_skate_score, foot_contact_ratio = _compute_v3_foot_skate_metrics(m)
    foot_penalty = (
        min(foot_skate_score * 5.0, 1.0) if foot_contact_ratio > 0.0 else 0.0
    )

    # --- Overall Artifact Score ---
    # Weighted combination (lower is better). For legacy 20D motion we keep the
    # original weighting; for canonical v3 48D motion we include foot-skate as
    # an additional penalty while keeping weights normalised.
    if D == 48 and foot_contact_ratio > 0.0:
        explosion_term = min(explosion_ratio * 10.0, 1.0)
        artifact_score = (
            0.25 * min(jitter_score, 1.0)
            + 0.2 * static_ratio
            + 0.25 * explosion_term
            + 0.15 * repetition_score
            + 0.15 * foot_penalty
        )
    else:
        artifact_score = (
            0.3 * min(jitter_score, 1.0)
            + 0.2 * static_ratio
            + 0.3 * explosion_ratio * 10  # Explosions are very bad
            + 0.2 * repetition_score
        )

    return {
        "jitter_score": jitter_score,
        "static_ratio": static_ratio,
        "explosion_ratio": explosion_ratio,
        "repetition_score": repetition_score,
        "foot_skate_score": float(foot_skate_score),
        "foot_contact_ratio": float(foot_contact_ratio),
        "artifact_score": float(artifact_score),
        "is_clean": artifact_score < 0.3,  # Threshold for "clean" motion
    }


# Environment-specific expected velocity ranges
# Maps environment_type to (expected_velocity, tolerance)
ENVIRONMENT_VELOCITY_EXPECTATIONS = {
    # Low velocity environments
    "underwater": (0.04, 0.03),  # Very slow movement
    "ocean_surface": (0.06, 0.04),
    "river": (0.05, 0.03),
    "lake": (0.05, 0.03),
    "pool": (0.05, 0.03),
    "swamp": (0.06, 0.04),
    # Zero/micro-gravity - floating
    "space_vacuum": (0.02, 0.02),  # Minimal movement
    "asteroid": (0.03, 0.02),
    "cloud_realm": (0.04, 0.03),
    # Low gravity - higher velocities possible
    "moon": (0.15, 0.1),
    "mars": (0.12, 0.08),
    "alien_planet_low_g": (0.15, 0.1),
    # Ice - faster sliding movement
    "rink": (0.15, 0.1),
    "arctic": (0.12, 0.08),
    "ice_realm": (0.12, 0.08),
    # Sports venues - faster movement
    "stadium": (0.12, 0.08),
    "arena": (0.12, 0.08),
    "track": (0.15, 0.1),
    "field": (0.12, 0.08),
    # Default Earth-normal
    "earth_normal": (0.1, 0.07),
    "grassland": (0.1, 0.07),
    "forest": (0.08, 0.05),
    "city_street": (0.1, 0.07),
}


def compute_motion_realism_score(
    motion: torch.Tensor, environment_type: str | None = None
) -> dict[str, float]:
    """Compute overall motion realism/quality score.

    Combines multiple quality signals into a single realism score
    suitable for data curation and filtering.

    Args:
        motion: Motion tensor [T, D] or [T, A, D]
        environment_type: Optional environment type for velocity expectations

    Returns score in [0, 1] where higher = more realistic.
    """
    # Get component metrics
    temporal = compute_motion_temporal_metrics(motion)
    artifacts = compute_synthetic_artifact_score(motion)

    # Smoothness contributes positively
    smoothness = temporal["smoothness_score"]

    # Artifacts contribute negatively
    artifact_penalty = artifacts["artifact_score"]

    # Environment-aware velocity scoring
    # Get expected velocity for this environment (default to Earth-normal)
    expected_vel, tolerance = ENVIRONMENT_VELOCITY_EXPECTATIONS.get(
        environment_type, (0.1, 0.07)
    )
    mean_vel = temporal["mean_velocity"]
    # Score based on deviation from expected velocity
    # Higher tolerance means more forgiving scoring
    velocity_deviation = abs(mean_vel - expected_vel)
    velocity_score = 1.0 / (1.0 + velocity_deviation / max(tolerance, 0.01))

    # Combine into realism score
    realism = (
        0.4 * smoothness
        + 0.3 * (1.0 - min(artifact_penalty, 1.0))
        + 0.3 * velocity_score
    )

    return {
        "realism_score": float(realism),
        "smoothness_component": float(smoothness),
        "artifact_component": float(1.0 - min(artifact_penalty, 1.0)),
        "velocity_component": float(velocity_score),
        "is_realistic": realism > 0.5,
        "environment_type": environment_type,
        "expected_velocity": expected_vel,
    }


def compute_dataset_fid_statistics(
    motions: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
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
    stats1: dict[str, torch.Tensor],
    stats2: dict[str, torch.Tensor],
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
