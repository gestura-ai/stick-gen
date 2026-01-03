"""Metadata extraction utilities for enhanced sample metadata.

This module provides functions to compute motion characteristics,
extract temporal information, estimate quality metrics, and infer
emotional content from motion data and text descriptions.

All functions return values compatible with the Pydantic models
defined in src/data_gen/schema.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.data_gen.schema import (
    EmotionMetadata,
    EnhancedSampleMetadata,
    InteractionMetadata,
    MotionStyleMetadata,
    MusicMetadata,
    QualityMetadata,
    SubjectMetadata,
    TemporalMetadata,
)


# =============================================================================
# Motion Style Computation
# =============================================================================


def compute_tempo(motion: Tensor | np.ndarray, fps: int = 25) -> float:
    """Compute motion tempo based on velocity oscillation frequency.

    Uses autocorrelation of velocity magnitude to find dominant period.
    Normalizes to 0-1 scale where:
        0.0 = very slow (tai chi, ~0.5 Hz)
        0.5 = walking (~1.5 Hz)
        1.0 = sprinting (~3+ Hz)

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]`` where ``D``
            is the flattened stick-figure dimension (e.g. 20 for the legacy
            5-segment layout or 48 for the v3 12-segment layout).
        fps: Frame rate of the motion data

    Returns:
        Tempo value normalized to 0-1
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    # Handle multi-actor: average across actors
    if motion.dim() == 3:
        motion = motion.mean(dim=1)

    if motion.shape[0] < 10:
        return 0.5  # Default for very short sequences

    # Compute velocity magnitudes
    velocities = motion[1:] - motion[:-1]  # [T-1, 20]
    vel_mag = velocities.norm(dim=-1)  # [T-1]

    # Use autocorrelation to find dominant frequency
    vel_centered = vel_mag - vel_mag.mean()
    autocorr = torch.nn.functional.conv1d(
        vel_centered.unsqueeze(0).unsqueeze(0),
        vel_centered.unsqueeze(0).unsqueeze(0),
        padding=len(vel_centered) - 1,
    ).squeeze()

    # Find first peak after lag 0 (indicates dominant period)
    autocorr = autocorr[len(vel_centered) - 1 :]  # Keep positive lags only
    min_lag = max(2, fps // 10)  # At least 0.1 sec
    max_lag = min(len(autocorr), fps * 2)  # Max 2 sec period

    if max_lag <= min_lag:
        return 0.5

    search_range = autocorr[min_lag:max_lag]
    if len(search_range) == 0:
        return 0.5

    peak_idx = search_range.argmax().item() + min_lag
    dominant_period = peak_idx / fps  # seconds

    # Convert period to frequency, then normalize
    # 0.5 Hz (2s period) -> 0.0, 3 Hz (0.33s period) -> 1.0
    frequency = 1.0 / max(dominant_period, 0.1)
    tempo = np.clip((frequency - 0.5) / 2.5, 0.0, 1.0)

    return float(tempo)


def compute_energy_level(motion: Tensor | np.ndarray, fps: int = 25) -> float:
    """Compute motion energy level based on velocity and acceleration.

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]`` where ``D``
            is the flattened stick-figure dimension.
        fps: Frame rate of the motion data

    Returns:
        Energy level normalized to 0-1
            0.0 = idle/resting
            0.5 = moderate activity (walking)
            1.0 = intense activity (running, jumping)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() == 3:
        motion = motion.mean(dim=1)

    if motion.shape[0] < 3:
        return 0.0

    # Compute velocity and acceleration
    velocities = motion[1:] - motion[:-1]  # [T-1, 20]
    accelerations = velocities[1:] - velocities[:-1]  # [T-2, 20]

    # Scale by fps for consistent units
    vel_magnitude = velocities.norm(dim=-1).mean().item() * fps
    acc_magnitude = accelerations.norm(dim=-1).mean().item() * (fps**2)

    # Empirical normalization based on typical motion ranges
    # Walking: vel ~2-3, acc ~5-10
    # Running: vel ~6-10, acc ~30-50
    energy_raw = vel_magnitude * 0.1 + acc_magnitude * 0.01
    energy = np.clip(energy_raw / 2.0, 0.0, 1.0)

    return float(energy)


def compute_smoothness(motion: Tensor | np.ndarray, fps: int = 25) -> float:
    """Compute motion smoothness based on jerk (derivative of acceleration).

    Lower jerk = higher smoothness. Normalized so:
        0.0 = very jerky/mechanical motion
        0.5 = natural motion
        1.0 = extremely smooth (tai chi, slow dance)

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]``
        fps: Frame rate of the motion data

    Returns:
        Smoothness value normalized to 0-1
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() == 3:
        motion = motion.mean(dim=1)

    if motion.shape[0] < 4:
        return 0.5

    # Compute jerk (third derivative)
    velocities = motion[1:] - motion[:-1]
    accelerations = velocities[1:] - velocities[:-1]
    jerks = accelerations[1:] - accelerations[:-1]

    # Scale by fps^3 for consistent units
    jerk_magnitude = jerks.norm(dim=-1).mean().item() * (fps**3)

    # Inverse relationship: high jerk = low smoothness
    # Empirical: natural motion ~1e4-1e5, jerky ~1e6+
    smoothness = 1.0 - np.clip(jerk_magnitude / 1e6, 0.0, 1.0)

    return float(smoothness)


def compute_motion_style(
    motion: Tensor | np.ndarray, fps: int = 25
) -> MotionStyleMetadata:
    """Compute all motion style metrics for a motion sequence.

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]``
        fps: Frame rate of the motion data

    Returns:
        MotionStyleMetadata with tempo, energy_level, and smoothness
    """
    return MotionStyleMetadata(
        tempo=compute_tempo(motion, fps),
        energy_level=compute_energy_level(motion, fps),
        smoothness=compute_smoothness(motion, fps),
    )


# =============================================================================
# Temporal Metadata Extraction
# =============================================================================


def extract_temporal_metadata(
    original_fps: int | None = None,
    original_num_frames: int | None = None,
    original_duration_sec: float | None = None,
) -> TemporalMetadata:
    """Create temporal metadata from source timing information.

    Can compute duration from fps and frame count if not provided.

    Args:
        original_fps: Original frame rate before resampling
        original_num_frames: Original frame count before padding/truncation
        original_duration_sec: Original duration in seconds

    Returns:
        TemporalMetadata with available timing information
    """
    # Compute duration if not provided but we have fps and frames
    if original_duration_sec is None and original_fps and original_num_frames:
        original_duration_sec = original_num_frames / original_fps

    return TemporalMetadata(
        original_fps=original_fps,
        original_duration_sec=original_duration_sec,
        original_num_frames=original_num_frames,
    )


# =============================================================================
# Quality Metrics Computation
# =============================================================================


def compute_marker_quality(motion: Tensor | np.ndarray, fps: int = 25) -> float:
    """Estimate marker/joint quality from motion noise level.

    Lower noise = higher quality. Based on high-frequency jitter detection.

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]``
        fps: Frame rate of the motion data

    Returns:
        Quality score 0-1 (higher = cleaner data)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() == 3:
        motion = motion.mean(dim=1)

    if motion.shape[0] < 5:
        return 0.5

    # High-frequency noise detection via second derivative variance
    velocities = motion[1:] - motion[:-1]
    accelerations = velocities[1:] - velocities[:-1]

    # High variance in acceleration at high fps = noisy markers
    acc_variance = accelerations.var().item() * (fps**2)

    # Empirical scaling: clean MoCap ~0.1-1, noisy ~10+
    quality = 1.0 - np.clip(acc_variance / 100.0, 0.0, 1.0)

    return float(quality)


def compute_quality_metadata(
    motion: Tensor | np.ndarray,
    fps: int = 25,
    reconstruction_confidence: float | None = None,
) -> QualityMetadata:
    """Compute quality metadata for a motion sequence.

    Args:
        motion: Motion tensor for marker quality computation
        fps: Frame rate
        reconstruction_confidence: Optional external confidence score

    Returns:
        QualityMetadata with computed and provided metrics
    """
    return QualityMetadata(
        reconstruction_confidence=reconstruction_confidence,
        marker_quality=compute_marker_quality(motion, fps),
        physics_score=None,  # Set by validator if available
    )


# =============================================================================
# Subject Demographics Estimation
# =============================================================================


def estimate_height_from_betas(betas: np.ndarray, is_smplx: bool = False) -> float:
    """Estimate height in cm from SMPL/SMPL-X body shape parameters.

    Beta[0] primarily encodes height variation from the mean body.

    Args:
        betas: SMPL body shape parameters (first 10 typically)
        is_smplx: Whether using SMPL-X (slightly different scaling)

    Returns:
        Estimated height in centimeters
    """
    # SMPL neutral body is approximately 170cm
    base_height = 170.0

    # Beta[0] approximately scales ±20cm across typical population
    # Empirical scaling factor (may need calibration per dataset)
    scale_factor = 10.0 if not is_smplx else 8.0

    if len(betas) > 0:
        height_adjustment = float(betas[0]) * scale_factor
    else:
        height_adjustment = 0.0

    return base_height + height_adjustment


def estimate_subject_metadata(
    betas: np.ndarray | None = None,
    gender: str | None = None,
    age_group: str | None = None,
    is_smplx: bool = False,
) -> SubjectMetadata:
    """Create subject metadata from available demographic information.

    Args:
        betas: SMPL body shape parameters for height estimation
        gender: Known gender ("male", "female", or "unknown")
        age_group: Known age group ("child", "adult", "elderly", "unknown")
        is_smplx: Whether using SMPL-X body model

    Returns:
        SubjectMetadata with available demographic information
    """
    height_cm = None
    if betas is not None and len(betas) > 0:
        height_cm = estimate_height_from_betas(betas, is_smplx)

    return SubjectMetadata(
        height_cm=height_cm,
        gender=gender,
        age_group=age_group,
    )


# =============================================================================
# Emotion Inference
# =============================================================================

# Emotion keywords with (valence, arousal) mappings
EMOTION_KEYWORDS: dict[str, tuple[float, float]] = {
    # Positive high arousal
    "happy": (0.7, 0.6),
    "excited": (0.8, 0.9),
    "joyful": (0.8, 0.7),
    "enthusiastic": (0.7, 0.8),
    "energetic": (0.5, 0.9),
    # Positive low arousal
    "calm": (0.4, 0.2),
    "relaxed": (0.5, 0.2),
    "peaceful": (0.6, 0.1),
    "content": (0.5, 0.3),
    # Negative high arousal
    "angry": (-0.6, 0.9),
    "furious": (-0.8, 1.0),
    "aggressive": (-0.5, 0.8),
    "frustrated": (-0.4, 0.7),
    "fighting": (-0.3, 0.9),
    # Negative low arousal
    "sad": (-0.7, 0.3),
    "depressed": (-0.8, 0.2),
    "tired": (-0.2, 0.1),
    "bored": (-0.3, 0.2),
    # Neutral
    "neutral": (0.0, 0.4),
    "walking": (0.1, 0.4),
    "standing": (0.0, 0.2),
}


def infer_emotion_from_text(description: str) -> EmotionMetadata:
    """Infer emotion from text description using keyword matching.

    Searches for emotion-related keywords and returns the average
    valence/arousal if multiple are found.

    Args:
        description: Text description of the motion

    Returns:
        EmotionMetadata with inferred emotion_label, valence, arousal
    """
    if not description:
        return EmotionMetadata()

    description_lower = description.lower()

    found_emotions: list[tuple[str, float, float]] = []
    for keyword, (valence, arousal) in EMOTION_KEYWORDS.items():
        if keyword in description_lower:
            found_emotions.append((keyword, valence, arousal))

    if not found_emotions:
        return EmotionMetadata()

    # Average the found emotions
    avg_valence = sum(e[1] for e in found_emotions) / len(found_emotions)
    avg_arousal = sum(e[2] for e in found_emotions) / len(found_emotions)

    # Map to FacialExpression categories based on valence/arousal
    if avg_valence > 0.3 and avg_arousal > 0.5:
        emotion_label = "excited"
    elif avg_valence > 0.3:
        emotion_label = "happy"
    elif avg_valence < -0.3 and avg_arousal > 0.5:
        emotion_label = "angry"
    elif avg_valence < -0.3:
        emotion_label = "sad"
    else:
        emotion_label = "neutral"

    return EmotionMetadata(
        emotion_label=emotion_label,
        valence=round(avg_valence, 2),
        arousal=round(avg_arousal, 2),
    )


def infer_emotion_from_motion(
    motion: Tensor | np.ndarray, fps: int = 25
) -> EmotionMetadata:
    """Infer emotion from motion characteristics.

    Uses energy level and body expansion to estimate valence/arousal:
    - High energy = high arousal
    - Expansive motion (limbs spread) = positive valence
    - Contracted motion = negative valence

    Args:
        motion: Motion tensor of shape ``[T, D]`` or ``[T, A, D]``
        fps: Frame rate

    Returns:
        EmotionMetadata with motion-inferred values
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() == 3:
        motion = motion.mean(dim=1)

    if motion.shape[0] < 5:
        return EmotionMetadata()

    # Energy level maps to arousal
    energy = compute_energy_level(motion, fps)
    arousal = energy

    # Estimate valence from body expansion
    # Higher variance in limb positions = more expansive = more positive
    limb_variance = motion.var(dim=0).mean().item()
    # Normalize: typical sitting ~0.01, dancing ~0.1
    expansion = np.clip(limb_variance / 0.1, 0.0, 1.0)
    valence = (expansion - 0.5) * 2  # Map to [-1, 1]

    # Determine emotion label
    if valence > 0.3 and arousal > 0.5:
        emotion_label = "excited"
    elif valence > 0.3:
        emotion_label = "happy"
    elif valence < -0.3 and arousal > 0.5:
        emotion_label = "angry"
    elif valence < -0.3:
        emotion_label = "sad"
    else:
        emotion_label = "neutral"

    return EmotionMetadata(
        emotion_label=emotion_label,
        valence=round(float(valence), 2),
        arousal=round(float(arousal), 2),
    )


# =============================================================================
# Interaction Metadata Computation
# =============================================================================


def compute_interaction_metadata(
    motion: Tensor | np.ndarray,
    contact_threshold: float = 0.5,
) -> InteractionMetadata:
    """Compute interaction metadata from multi-actor motion.

    Args:
        motion: Multi-actor motion tensor of shape ``[T, A, D]`` where
            ``A >= 2`` and ``D`` is the flattened stick-figure dimension
            (e.g. 20 or 48).
        contact_threshold: Distance threshold for contact detection in the
            normalized stick-space units.

    Returns:
        InteractionMetadata with contact_frames, role, and type
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if motion.dim() != 3 or motion.shape[1] < 2:
        return InteractionMetadata()

    T, A, D = motion.shape

    # Compute distance between actors (using first actor pair)
    actor1 = motion[:, 0, :]  # [T, D]
    actor2 = motion[:, 1, :]  # [T, D]

    if D % 4 != 0:
        raise ValueError(
            "compute_interaction_metadata expects last dimension to be a "
            f"multiple of 4 (segments x 4 endpoints), got D={D}"
        )

    num_segments = D // 4

    # Compute per-segment distances and average across segments. This is
    # simplified but works for both 5-segment and 12-segment layouts.
    distances = (actor1 - actor2).view(T, num_segments, 4).norm(dim=-1).mean(dim=-1)

    # Contact frames are where distance < threshold
    contact_mask = distances < contact_threshold
    contact_frames = contact_mask.nonzero(as_tuple=True)[0].tolist()

    # Determine role based on velocity variance (leader moves more)
    actor1_vel = (actor1[1:] - actor1[:-1]).norm(dim=-1).var().item()
    actor2_vel = (actor2[1:] - actor2[:-1]).norm(dim=-1).var().item()

    if actor1_vel > actor2_vel * 1.2:
        role = "leader"
    elif actor2_vel > actor1_vel * 1.2:
        role = "follower"
    else:
        role = "symmetric"

    return InteractionMetadata(
        contact_frames=contact_frames if contact_frames else None,
        interaction_role=role,
        interaction_type=None,  # Requires external annotation
    )


# =============================================================================
# Unified Enhanced Metadata Builder
# =============================================================================


def build_enhanced_metadata(
    motion: Tensor | np.ndarray,
    fps: int = 25,
    description: str | None = None,
    original_fps: int | None = None,
    original_num_frames: int | None = None,
    betas: np.ndarray | None = None,
    gender: str | None = None,
    is_multi_actor: bool = False,
) -> EnhancedSampleMetadata:
    """Build complete enhanced metadata for a motion sample.

    Computes all applicable metadata based on available inputs.

    Args:
        motion: Motion tensor ``[T, D]`` or ``[T, A, D]`` where ``D`` is the
            flattened stick-figure dimension (20 for v1, 48 for v3, etc.).
        fps: Current frame rate
        description: Text description for emotion inference
        original_fps: Source frame rate before resampling
        original_num_frames: Source frame count
        betas: SMPL body shape parameters
        gender: Subject gender if known
        is_multi_actor: Whether this is multi-actor data

    Returns:
        EnhancedSampleMetadata with all computed fields
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    # Motion style (always computed)
    motion_style = compute_motion_style(motion, fps)

    # Temporal metadata
    temporal = extract_temporal_metadata(
        original_fps=original_fps,
        original_num_frames=original_num_frames,
    )

    # Quality metrics
    quality = compute_quality_metadata(motion, fps)

    # Subject demographics (if betas available)
    subject = None
    if betas is not None:
        subject = estimate_subject_metadata(betas=betas, gender=gender)

    # Emotion (prefer text-based if description available)
    emotion = None
    if description:
        emotion = infer_emotion_from_text(description)
        if emotion.emotion_label is None:
            emotion = infer_emotion_from_motion(motion, fps)
    else:
        emotion = infer_emotion_from_motion(motion, fps)

    # Interaction (only for multi-actor)
    interaction = None
    if is_multi_actor and motion.dim() == 3:
        interaction = compute_interaction_metadata(motion)

    return EnhancedSampleMetadata(
        motion_style=motion_style,
        subject=subject,
        music=None,  # Requires external data (AIST++)
        interaction=interaction,
        temporal=temporal,
        quality=quality,
        emotion=emotion,
    )

