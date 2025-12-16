"""
Unit tests for the Safety Critic module.

Tests cover:
- Frozen motion detection
- Repetitive motion detection
- Jittery motion detection
- Physics violation detection
- Ground penetration detection
- Overall scoring and safety decisions
- Batch evaluation
"""

import torch

from src.eval.safety_critic import (
    SafetyCritic,
    SafetyCriticConfig,
    SafetyCriticResult,
    SafetyIssueType,
    batch_evaluate_safety,
    evaluate_motion_safety,
)


class TestSafetyCriticConfig:
    """Tests for SafetyCriticConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SafetyCriticConfig()
        assert config.frozen_velocity_threshold == 0.01
        assert config.max_velocity == 15.0
        assert config.rejection_severity_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SafetyCriticConfig(
            frozen_velocity_threshold=0.05,
            max_velocity=20.0,
            rejection_severity_threshold=0.5,
        )
        assert config.frozen_velocity_threshold == 0.05
        assert config.max_velocity == 20.0
        assert config.rejection_severity_threshold == 0.5


class TestSafetyCriticFrozenMotion:
    """Tests for frozen motion detection."""

    def test_frozen_motion_detected(self):
        """Test that completely frozen motion is detected."""
        critic = SafetyCritic()
        # Create frozen motion (all zeros)
        motion = torch.zeros(100, 20)
        result = critic.evaluate(motion)

        assert not result.is_safe
        assert any(i.issue_type == SafetyIssueType.MOTION_FROZEN for i in result.issues)

    def test_normal_motion_not_frozen(self):
        """Test that normal motion is not flagged as frozen."""
        critic = SafetyCritic()
        # Create motion with movement
        motion = torch.randn(100, 20) * 0.1
        motion = torch.cumsum(motion, dim=0)  # Cumulative sum for smooth motion
        result = critic.evaluate(motion)

        assert not any(
            i.issue_type == SafetyIssueType.MOTION_FROZEN for i in result.issues
        )

    def test_short_sequence_not_frozen(self):
        """Test that very short sequences don't trigger frozen detection."""
        critic = SafetyCritic()
        motion = torch.zeros(1, 20)  # Single frame
        result = critic.evaluate(motion)

        assert not any(
            i.issue_type == SafetyIssueType.MOTION_FROZEN for i in result.issues
        )


class TestSafetyCriticRepetitiveMotion:
    """Tests for repetitive motion detection."""

    def test_repetitive_motion_detected(self):
        """Test that repetitive motion is detected."""
        critic = SafetyCritic(
            SafetyCriticConfig(
                repetition_window=10,
                repetition_min_cycles=3,
                repetition_similarity_threshold=0.99,
            )
        )
        # Create repetitive motion (same pattern repeated)
        pattern = torch.randn(10, 20)
        motion = pattern.repeat(10, 1)  # Repeat 10 times
        result = critic.evaluate(motion)

        assert any(
            i.issue_type == SafetyIssueType.MOTION_REPETITIVE for i in result.issues
        )

    def test_varied_motion_not_repetitive(self):
        """Test that varied motion is not flagged as repetitive."""
        critic = SafetyCritic()
        # Create varied motion
        motion = torch.randn(100, 20)
        result = critic.evaluate(motion)

        assert not any(
            i.issue_type == SafetyIssueType.MOTION_REPETITIVE for i in result.issues
        )


class TestSafetyCriticJitteryMotion:
    """Tests for jittery motion detection."""

    def test_jittery_motion_detected(self):
        """Test that jittery motion is detected."""
        critic = SafetyCritic(
            SafetyCriticConfig(
                jitter_acceleration_threshold=10.0,
                jitter_frame_ratio=0.2,
            )
        )
        # Create jittery motion (high frequency noise)
        motion = torch.randn(100, 20) * 100  # Large random jumps
        result = critic.evaluate(motion)

        assert any(
            i.issue_type == SafetyIssueType.MOTION_JITTERY for i in result.issues
        )

    def test_smooth_motion_not_jittery(self):
        """Test that smooth motion is not flagged as jittery."""
        critic = SafetyCritic()
        # Create smooth motion (low frequency)
        t = torch.linspace(0, 2 * 3.14159, 100).unsqueeze(1)
        motion = torch.sin(t) * torch.randn(1, 20) * 0.1
        result = critic.evaluate(motion)

        assert not any(
            i.issue_type == SafetyIssueType.MOTION_JITTERY for i in result.issues
        )


class TestSafetyCriticPhysicsViolations:
    """Tests for physics violation detection."""

    def test_velocity_exceeded_detected(self):
        """Test that excessive velocity is detected."""
        critic = SafetyCritic(SafetyCriticConfig(max_velocity=10.0))
        motion = torch.zeros(100, 20)
        # Physics: [vx, vy, ax, ay, mx, my]
        physics = torch.zeros(100, 6)
        physics[:, 0] = 20.0  # vx = 20 m/s (exceeds 10)

        result = critic.evaluate(motion, physics)

        assert any(
            i.issue_type == SafetyIssueType.PHYSICS_VELOCITY_EXCEEDED
            for i in result.issues
        )

    def test_normal_velocity_ok(self):
        """Test that normal velocity passes."""
        critic = SafetyCritic(SafetyCriticConfig(max_velocity=15.0))
        motion = torch.zeros(100, 20)
        physics = torch.zeros(100, 6)
        physics[:, 0] = 5.0  # vx = 5 m/s (within limit)

        result = critic.evaluate(motion, physics)

        assert not any(
            i.issue_type == SafetyIssueType.PHYSICS_VELOCITY_EXCEEDED
            for i in result.issues
        )


class TestSafetyCriticGroundPenetration:
    """Tests for ground penetration detection."""

    def test_ground_penetration_detected(self):
        """Test that ground penetration is detected."""
        critic = SafetyCritic(SafetyCriticConfig(ground_y_threshold=-0.1))
        # Motion format: 5 limbs * 4 coords (x1, y1, x2, y2) = 20
        motion = torch.zeros(100, 20)
        motion[:, 1] = -0.5  # y1 of first limb below ground

        result = critic.evaluate(motion)

        assert any(
            i.issue_type == SafetyIssueType.PHYSICS_GROUND_PENETRATION
            for i in result.issues
        )

    def test_above_ground_ok(self):
        """Test that motion above ground passes."""
        critic = SafetyCritic()
        motion = torch.ones(100, 20) * 0.5  # All positive y values

        result = critic.evaluate(motion)

        assert not any(
            i.issue_type == SafetyIssueType.PHYSICS_GROUND_PENETRATION
            for i in result.issues
        )


class TestSafetyCriticQualityScore:
    """Tests for quality score checking."""

    def test_low_quality_flagged(self):
        """Test that low quality score is flagged."""
        critic = SafetyCritic(SafetyCriticConfig(min_quality_score=0.5))
        motion = torch.randn(100, 20) * 0.01  # Normal motion
        motion = torch.cumsum(motion, dim=0)

        result = critic.evaluate(motion, quality_score=0.2)

        assert any(
            i.issue_type == SafetyIssueType.QUALITY_BELOW_THRESHOLD
            for i in result.issues
        )

    def test_high_quality_ok(self):
        """Test that high quality score passes."""
        critic = SafetyCritic(SafetyCriticConfig(min_quality_score=0.5))
        motion = torch.randn(100, 20) * 0.01
        motion = torch.cumsum(motion, dim=0)

        result = critic.evaluate(motion, quality_score=0.8)

        assert not any(
            i.issue_type == SafetyIssueType.QUALITY_BELOW_THRESHOLD
            for i in result.issues
        )


class TestSafetyCriticOverallScoring:
    """Tests for overall scoring and safety decisions."""

    def test_perfect_motion_is_safe(self):
        """Test that perfect motion is marked as safe."""
        critic = SafetyCritic()
        # Create smooth, varied motion
        t = torch.linspace(0, 4 * 3.14159, 100).unsqueeze(1)
        motion = torch.sin(t) * torch.randn(1, 20) * 0.1
        motion = torch.cumsum(motion, dim=0)

        result = critic.evaluate(motion)

        assert result.is_safe
        assert result.overall_score > 0.5

    def test_multiple_issues_reduce_score(self):
        """Test that multiple issues reduce the overall score."""
        critic = SafetyCritic()
        # Create problematic motion
        motion = torch.zeros(100, 20)  # Frozen
        motion[:, 1] = -0.5  # Ground penetration

        result = critic.evaluate(motion)

        assert len(result.issues) >= 2
        assert result.overall_score < 0.5

    def test_rejection_reasons(self):
        """Test that rejection reasons are properly generated."""
        critic = SafetyCritic()
        motion = torch.zeros(100, 20)  # Frozen motion

        result = critic.evaluate(motion)
        reasons = result.get_rejection_reasons()

        assert len(reasons) > 0
        assert any("frozen" in r.lower() for r in reasons)


class TestSafetyCriticConvenienceFunctions:
    """Tests for convenience functions."""

    def test_evaluate_motion_safety(self):
        """Test the evaluate_motion_safety convenience function."""
        motion = torch.randn(100, 20) * 0.01
        motion = torch.cumsum(motion, dim=0)

        result = evaluate_motion_safety(motion)

        assert isinstance(result, SafetyCriticResult)
        assert hasattr(result, "is_safe")
        assert hasattr(result, "overall_score")

    def test_batch_evaluate_safety(self):
        """Test the batch_evaluate_safety function."""
        samples = [
            {"motion": torch.randn(50, 20) * 0.01},
            {"motion": torch.zeros(50, 20)},  # Frozen
            {"motion": torch.randn(50, 20) * 0.01, "quality_score": 0.9},
        ]

        result = batch_evaluate_safety(samples)

        assert "total_samples" in result
        assert result["total_samples"] == 3
        assert "safe_count" in result
        assert "safe_ratio" in result
        assert "per_sample_results" in result
        assert len(result["per_sample_results"]) == 3


class TestSafetyCriticMultiActor:
    """Tests for multi-actor motion handling."""

    def test_multi_actor_motion(self):
        """Test that multi-actor motion is handled correctly."""
        critic = SafetyCritic()
        # [T, A, D] format with 2 actors
        motion = torch.randn(100, 2, 20) * 0.01
        motion = torch.cumsum(motion, dim=0)

        result = critic.evaluate(motion)

        assert isinstance(result, SafetyCriticResult)

    def test_single_actor_normalized(self):
        """Test that single-actor motion is normalized correctly."""
        critic = SafetyCritic()
        # [T, D] format
        motion = torch.randn(100, 20) * 0.01
        motion = torch.cumsum(motion, dim=0)

        result = critic.evaluate(motion)

        assert isinstance(result, SafetyCriticResult)
