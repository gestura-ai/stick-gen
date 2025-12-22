"""Unit tests for metadata extraction functions.

Tests all metadata extraction functions in src/data_gen/metadata_extractors.py
with synthetic motion data.
"""

import pytest
import torch

from src.data_gen.metadata_extractors import (
    build_enhanced_metadata,
    compute_energy_level,
    compute_interaction_metadata,
    compute_marker_quality,
    compute_motion_style,
    compute_quality_metadata,
    compute_smoothness,
    compute_tempo,
    extract_temporal_metadata,
    infer_emotion_from_motion,
    infer_emotion_from_text,
)
from src.data_gen.schema import (
    EmotionMetadata,
    EnhancedSampleMetadata,
    InteractionMetadata,
    MotionStyleMetadata,
    QualityMetadata,
    TemporalMetadata,
)


class TestComputeTempo:
    """Tests for compute_tempo function."""

    def test_returns_float_in_range(self):
        """Tempo should be in [0, 1] range."""
        motion = torch.randn(100, 20)
        tempo = compute_tempo(motion, fps=25)
        assert isinstance(tempo, float)
        assert 0.0 <= tempo <= 1.0

    def test_constant_motion_valid_tempo(self):
        """Constant (non-oscillating) motion should return a valid tempo."""
        # Constant motion - same value every frame
        motion = torch.ones(100, 20) * 0.5
        tempo = compute_tempo(motion, fps=25)
        # Should be a valid value in range (may default to 0.5 for edge cases)
        assert 0.0 <= tempo <= 1.0

    def test_handles_short_sequence(self):
        """Should handle sequences shorter than FFT window."""
        motion = torch.randn(10, 20)
        tempo = compute_tempo(motion, fps=25)
        assert 0.0 <= tempo <= 1.0

    def test_multi_actor_format(self):
        """Should handle [T, A, 20] multi-actor format."""
        motion = torch.randn(100, 2, 20)
        tempo = compute_tempo(motion, fps=25)
        assert 0.0 <= tempo <= 1.0


class TestComputeEnergyLevel:
    """Tests for compute_energy_level function."""

    def test_returns_float_in_range(self):
        """Energy level should be in [0, 1] range."""
        motion = torch.randn(100, 20)
        energy = compute_energy_level(motion, fps=25)
        assert isinstance(energy, float)
        assert 0.0 <= energy <= 1.0

    def test_static_motion_low_energy(self):
        """Static motion should have low energy."""
        motion = torch.zeros(100, 20)
        energy = compute_energy_level(motion, fps=25)
        assert energy < 0.1

    def test_high_velocity_high_energy(self):
        """Fast-moving motion should have high energy."""
        # Create motion with high velocity (large frame-to-frame changes)
        motion = torch.cumsum(torch.randn(100, 20) * 2.0, dim=0)
        energy = compute_energy_level(motion, fps=25)
        assert energy > 0.5


class TestComputeSmoothness:
    """Tests for compute_smoothness function."""

    def test_returns_float_in_range(self):
        """Smoothness should be in [0, 1] range."""
        motion = torch.randn(100, 20)
        smoothness = compute_smoothness(motion, fps=25)
        assert isinstance(smoothness, float)
        assert 0.0 <= smoothness <= 1.0

    def test_smooth_motion_high_score(self):
        """Smooth interpolated motion should have high smoothness."""
        # Linear interpolation (very smooth)
        t = torch.linspace(0, 1, 100).unsqueeze(1)
        motion = t * torch.randn(1, 20) + (1 - t) * torch.randn(1, 20)
        smoothness = compute_smoothness(motion, fps=25)
        assert smoothness > 0.7

    def test_jittery_motion_low_score(self):
        """High-frequency noise should have low smoothness."""
        # Alternating values (maximum jitter)
        motion = torch.randn(100, 20)
        motion[::2] *= -5  # Flip every other frame
        smoothness = compute_smoothness(motion, fps=25)
        assert smoothness < 0.5


class TestComputeMotionStyle:
    """Tests for compute_motion_style function."""

    def test_returns_motion_style_metadata(self):
        """Should return MotionStyleMetadata instance."""
        motion = torch.randn(100, 20)
        style = compute_motion_style(motion, fps=25)
        assert isinstance(style, MotionStyleMetadata)

    def test_all_fields_populated(self):
        """All style fields should be populated."""
        motion = torch.randn(100, 20)
        style = compute_motion_style(motion, fps=25)
        assert style.tempo is not None
        assert style.energy_level is not None
        assert style.smoothness is not None

    def test_all_fields_in_range(self):
        """All style fields should be in [0, 1]."""
        motion = torch.randn(100, 20)
        style = compute_motion_style(motion, fps=25)
        assert 0.0 <= style.tempo <= 1.0
        assert 0.0 <= style.energy_level <= 1.0
        assert 0.0 <= style.smoothness <= 1.0


class TestExtractTemporalMetadata:
    """Tests for extract_temporal_metadata function."""

    def test_returns_temporal_metadata(self):
        """Should return TemporalMetadata instance."""
        temporal = extract_temporal_metadata(
            original_fps=30, original_num_frames=300
        )
        assert isinstance(temporal, TemporalMetadata)

    def test_computes_duration(self):
        """Duration should be frames / fps."""
        temporal = extract_temporal_metadata(
            original_fps=30, original_num_frames=300
        )
        assert temporal.original_duration_sec == pytest.approx(10.0)

    def test_handles_none_fps(self):
        """Should handle None fps gracefully."""
        temporal = extract_temporal_metadata(
            original_fps=None, original_num_frames=100
        )
        assert temporal.original_fps is None
        assert temporal.original_duration_sec is None


class TestComputeMarkerQuality:
    """Tests for compute_marker_quality function."""

    def test_returns_float_in_range(self):
        """Quality should be in [0, 1] range."""
        motion = torch.randn(100, 20)
        quality = compute_marker_quality(motion, fps=25)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_smooth_motion_high_quality(self):
        """Smooth motion should have high quality."""
        t = torch.linspace(0, 1, 100).unsqueeze(1)
        motion = t * torch.randn(1, 20) + (1 - t) * torch.randn(1, 20)
        quality = compute_marker_quality(motion, fps=25)
        assert quality > 0.5


class TestComputeQualityMetadata:
    """Tests for compute_quality_metadata function."""

    def test_returns_quality_metadata(self):
        """Should return QualityMetadata instance."""
        motion = torch.randn(100, 20)
        quality = compute_quality_metadata(motion, fps=25)
        assert isinstance(quality, QualityMetadata)

    def test_marker_quality_populated(self):
        """marker_quality should be populated."""
        motion = torch.randn(100, 20)
        quality = compute_quality_metadata(motion, fps=25)
        assert quality.marker_quality is not None


class TestInferEmotionFromText:
    """Tests for infer_emotion_from_text function."""

    def test_returns_emotion_metadata(self):
        """Should return EmotionMetadata instance."""
        emotion = infer_emotion_from_text("A person walking happily")
        assert isinstance(emotion, EmotionMetadata)

    def test_positive_keywords(self):
        """Positive words should give positive valence."""
        emotion = infer_emotion_from_text("A person dancing joyfully and happily")
        assert emotion.valence is not None
        assert emotion.valence > 0

    def test_negative_keywords(self):
        """Negative words should give negative valence."""
        emotion = infer_emotion_from_text("A person sadly walking away in anger")
        assert emotion.valence is not None
        assert emotion.valence < 0

    def test_neutral_text(self):
        """Neutral text should give neutral valence."""
        emotion = infer_emotion_from_text("A person standing")
        assert emotion.emotion_label == "neutral"


class TestInferEmotionFromMotion:
    """Tests for infer_emotion_from_motion function."""

    def test_returns_emotion_metadata(self):
        """Should return EmotionMetadata instance."""
        motion = torch.randn(100, 20)
        emotion = infer_emotion_from_motion(motion, fps=25)
        assert isinstance(emotion, EmotionMetadata)

    def test_arousal_in_range(self):
        """Arousal should be in [0, 1]."""
        motion = torch.randn(100, 20)
        emotion = infer_emotion_from_motion(motion, fps=25)
        assert emotion.arousal is not None
        assert 0.0 <= emotion.arousal <= 1.0


class TestComputeInteractionMetadata:
    """Tests for compute_interaction_metadata function."""

    def test_returns_interaction_metadata(self):
        """Should return InteractionMetadata instance."""
        motion = torch.randn(100, 2, 20)  # 2 actors
        interaction = compute_interaction_metadata(motion)
        assert isinstance(interaction, InteractionMetadata)

    def test_single_actor_returns_empty(self):
        """Single actor motion should return empty interaction metadata."""
        motion = torch.randn(100, 20)
        interaction = compute_interaction_metadata(motion)
        # Single actor returns InteractionMetadata with all None fields
        assert interaction is not None
        assert interaction.contact_frames is None
        assert interaction.interaction_role is None

    def test_detects_contact_frames(self):
        """Should detect frames where actors are close."""
        # Create motion where actors start far and come close
        motion = torch.zeros(100, 2, 20)
        motion[:, 0, :] = 0.0  # Actor 1 stationary
        motion[:50, 1, :] = 10.0  # Actor 2 far away initially
        motion[50:, 1, :] = 0.01  # Actor 2 comes very close
        interaction = compute_interaction_metadata(motion)
        assert interaction is not None
        assert interaction.contact_frames is not None
        assert len(interaction.contact_frames) > 0


class TestBuildEnhancedMetadata:
    """Tests for build_enhanced_metadata function."""

    def test_returns_enhanced_sample_metadata(self):
        """Should return EnhancedSampleMetadata instance."""
        motion = torch.randn(100, 20)
        enhanced = build_enhanced_metadata(motion=motion, fps=25)
        assert isinstance(enhanced, EnhancedSampleMetadata)

    def test_motion_style_always_populated(self):
        """motion_style should always be computed."""
        motion = torch.randn(100, 20)
        enhanced = build_enhanced_metadata(motion=motion, fps=25)
        assert enhanced.motion_style is not None

    def test_temporal_from_params(self):
        """Temporal metadata should use provided params."""
        motion = torch.randn(100, 20)
        enhanced = build_enhanced_metadata(
            motion=motion,
            fps=25,
            original_fps=30,
            original_num_frames=120,
        )
        assert enhanced.temporal is not None
        assert enhanced.temporal.original_fps == 30
        assert enhanced.temporal.original_num_frames == 120

    def test_emotion_from_description(self):
        """Emotion should be inferred from description."""
        motion = torch.randn(100, 20)
        enhanced = build_enhanced_metadata(
            motion=motion,
            fps=25,
            description="A person jumping happily",
        )
        assert enhanced.emotion is not None

    def test_model_dump_serializable(self):
        """model_dump() output should be JSON-serializable."""
        import json

        motion = torch.randn(100, 20)
        enhanced = build_enhanced_metadata(
            motion=motion,
            fps=25,
            description="Walking",
            original_fps=30,
            original_num_frames=100,
        )
        dump = enhanced.model_dump()
        # Should not raise
        json.dumps(dump)

    def test_multi_actor_computes_style(self):
        """Multi-actor motion should still compute motion style."""
        motion = torch.randn(100, 2, 20)
        enhanced = build_enhanced_metadata(motion=motion, fps=25)
        assert enhanced.motion_style is not None
        assert 0.0 <= enhanced.motion_style.energy_level <= 1.0

