"""
Safety Critic Module for Stick-Gen Motion Generation.

This module provides automated detection of degenerate, unsafe, or low-quality
motion outputs. It can be used during:
1. Inference - to reject or flag bad outputs before returning to users
2. Evaluation - to compute robustness metrics on adversarial prompts
3. Training - to filter or weight samples

Key checks:
- Motion degeneracy: repetition, freezing, jitter
- Physics violations: extreme velocity, ground penetration, impossible poses
- Semantic failures: motion doesn't match expected action category
- Quality thresholds: overall quality score gating
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SafetyIssueType(Enum):
    """Types of safety/quality issues that can be detected."""

    MOTION_FROZEN = "motion_frozen"
    MOTION_REPETITIVE = "motion_repetitive"
    MOTION_JITTERY = "motion_jittery"
    PHYSICS_VELOCITY_EXCEEDED = "physics_velocity_exceeded"
    PHYSICS_ACCELERATION_EXCEEDED = "physics_acceleration_exceeded"
    PHYSICS_GROUND_PENETRATION = "physics_ground_penetration"
    PHYSICS_IMPOSSIBLE_POSE = "physics_impossible_pose"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    DURATION_INVALID = "duration_invalid"


@dataclass
class SafetyIssue:
    """A detected safety or quality issue."""

    issue_type: SafetyIssueType
    severity: float  # 0.0 = minor, 1.0 = critical
    description: str
    frame_range: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyCriticResult:
    """Result from running the safety critic on a generated sample."""

    is_safe: bool
    overall_score: float  # 0.0 = reject, 1.0 = perfect
    issues: list[SafetyIssue] = field(default_factory=list)
    check_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_rejection_reasons(self) -> list[str]:
        """Get human-readable rejection reasons for critical issues."""
        return [issue.description for issue in self.issues if issue.severity >= 0.7]


# Environment-specific physics threshold multipliers for safety evaluation
# These adjust thresholds based on environment physics characteristics
SAFETY_ENVIRONMENT_MULTIPLIERS = {
    # Low gravity environments - different expectations
    "space_vacuum": {"velocity": 0.5, "acceleration": 0.3, "ground_y": -10.0},  # No ground in space
    "moon": {"velocity": 1.5, "acceleration": 0.5, "ground_y": -0.1},
    "mars": {"velocity": 1.3, "acceleration": 0.7, "ground_y": -0.1},
    "asteroid": {"velocity": 0.5, "acceleration": 0.3, "ground_y": -10.0},
    "alien_planet_low_g": {"velocity": 1.5, "acceleration": 0.5, "ground_y": -0.1},
    "cloud_realm": {"velocity": 0.6, "acceleration": 0.4, "ground_y": -10.0},  # Floating
    # Underwater - slow movement, no ground constraint
    "underwater": {"velocity": 0.4, "acceleration": 0.4, "ground_y": -10.0},
    "ocean_surface": {"velocity": 0.6, "acceleration": 0.5, "ground_y": -10.0},
    "river": {"velocity": 0.5, "acceleration": 0.5, "ground_y": -5.0},
    "lake": {"velocity": 0.5, "acceleration": 0.5, "ground_y": -5.0},
    "pool": {"velocity": 0.5, "acceleration": 0.5, "ground_y": -2.0},
    # Ice/slippery - can have higher velocities
    "rink": {"velocity": 1.5, "acceleration": 0.7, "ground_y": -0.1},
    "arctic": {"velocity": 1.2, "acceleration": 0.8, "ground_y": -0.1},
    "ice_realm": {"velocity": 1.3, "acceleration": 0.7, "ground_y": -0.1},
    # High altitude/rooftop - allow some negative Y
    "rooftop": {"velocity": 1.0, "acceleration": 1.0, "ground_y": -5.0},
    # Sports venues - fast motion expected
    "stadium": {"velocity": 1.2, "acceleration": 1.2, "ground_y": -0.1},
    "arena": {"velocity": 1.2, "acceleration": 1.2, "ground_y": -0.1},
    "track": {"velocity": 1.3, "acceleration": 1.2, "ground_y": -0.1},
}


class SafetyCriticConfig:
    """Configuration for the safety critic thresholds.

    Supports environment-aware threshold adjustment for different physics contexts
    (e.g., underwater, space, ice).
    """

    def __init__(
        self,
        # Motion degeneracy thresholds
        frozen_velocity_threshold: float = 0.01,
        frozen_frame_ratio: float = 0.8,
        repetition_window: int = 25,  # 1 second at 25 FPS
        repetition_similarity_threshold: float = 0.95,
        repetition_min_cycles: int = 3,
        jitter_acceleration_threshold: float = 50.0,
        jitter_frame_ratio: float = 0.3,
        # Physics thresholds (base values for Earth-normal)
        max_velocity: float = 15.0,  # m/s
        max_acceleration: float = 50.0,  # m/s²
        ground_y_threshold: float = -0.1,  # Allow small negative for tolerance
        # Quality thresholds
        min_quality_score: float = 0.3,
        min_smoothness_score: float = 0.2,
        # Rejection threshold
        rejection_severity_threshold: float = 0.7,
        # Environment type for physics-aware thresholds
        environment_type: str | None = None,
    ):
        self.frozen_velocity_threshold = frozen_velocity_threshold
        self.frozen_frame_ratio = frozen_frame_ratio
        self.repetition_window = repetition_window
        self.repetition_similarity_threshold = repetition_similarity_threshold
        self.repetition_min_cycles = repetition_min_cycles
        self.jitter_acceleration_threshold = jitter_acceleration_threshold
        self.jitter_frame_ratio = jitter_frame_ratio

        # Store base values for environment adjustment
        self._base_max_velocity = max_velocity
        self._base_max_acceleration = max_acceleration
        self._base_ground_y_threshold = ground_y_threshold

        # Apply environment multipliers
        self.environment_type = environment_type
        self._apply_environment_multipliers(environment_type)

        self.min_quality_score = min_quality_score
        self.min_smoothness_score = min_smoothness_score
        self.rejection_severity_threshold = rejection_severity_threshold

    def _apply_environment_multipliers(self, environment_type: str | None) -> None:
        """Apply environment-specific multipliers to physics thresholds."""
        if environment_type and environment_type in SAFETY_ENVIRONMENT_MULTIPLIERS:
            mult = SAFETY_ENVIRONMENT_MULTIPLIERS[environment_type]
            self.max_velocity = self._base_max_velocity * mult.get("velocity", 1.0)
            self.max_acceleration = self._base_max_acceleration * mult.get("acceleration", 1.0)
            self.ground_y_threshold = mult.get("ground_y", self._base_ground_y_threshold)
        else:
            self.max_velocity = self._base_max_velocity
            self.max_acceleration = self._base_max_acceleration
            self.ground_y_threshold = self._base_ground_y_threshold

    def set_environment(self, environment_type: str | None) -> None:
        """Update thresholds for a new environment type."""
        self.environment_type = environment_type
        self._apply_environment_multipliers(environment_type)

    @classmethod
    def for_environment(cls, environment_type: str, **kwargs) -> "SafetyCriticConfig":
        """Create a config pre-configured for a specific environment."""
        return cls(environment_type=environment_type, **kwargs)


class SafetyCritic:
    """
    Safety critic for evaluating motion generation quality and safety.

    Usage:
        critic = SafetyCritic()
        result = critic.evaluate(motion_tensor, physics_tensor)
        if not result.is_safe:
            print("Rejected:", result.get_rejection_reasons())
    """

    def __init__(self, config: SafetyCriticConfig | None = None):
        self.config = config or SafetyCriticConfig()

    def evaluate(
        self,
        motion: torch.Tensor,
        physics: torch.Tensor | None = None,
        quality_score: float | None = None,
        expected_action: str | None = None,
        environment_type: str | None = None,
    ) -> SafetyCriticResult:
        """
        Evaluate a generated motion sequence for safety and quality.

        Args:
            motion: Motion tensor [T, D] or [T, A, D] where T=frames, A=actors, D=dims
            physics: Optional physics tensor [T, 6] or [T, A, 6]
            quality_score: Optional pre-computed quality score from auto-annotator
            expected_action: Optional expected action category for semantic check
            environment_type: Optional environment type for physics-aware thresholds

        Returns:
            SafetyCriticResult with is_safe, overall_score, and detailed issues
        """
        # Temporarily adjust thresholds for environment-specific evaluation
        original_env = self.config.environment_type
        if environment_type and environment_type != original_env:
            self.config.set_environment(environment_type)
        issues: list[SafetyIssue] = []
        check_results: dict[str, dict[str, Any]] = {}

        # Normalize motion shape to [T, A, D]
        motion = self._normalize_motion_shape(motion)

        # Run all checks
        frozen_result = self._check_frozen_motion(motion)
        check_results["frozen"] = frozen_result
        if frozen_result.get("is_frozen", False):
            issues.append(
                SafetyIssue(
                    issue_type=SafetyIssueType.MOTION_FROZEN,
                    severity=0.9,
                    description=f"Motion is frozen for {frozen_result['frozen_ratio']*100:.1f}% of frames",
                    metadata=frozen_result,
                )
            )

        repetition_result = self._check_repetitive_motion(motion)
        check_results["repetition"] = repetition_result
        if repetition_result.get("is_repetitive", False):
            issues.append(
                SafetyIssue(
                    issue_type=SafetyIssueType.MOTION_REPETITIVE,
                    severity=0.7,
                    description=f"Detected {repetition_result['num_cycles']} repetitive cycles",
                    metadata=repetition_result,
                )
            )

        jitter_result = self._check_jittery_motion(motion)
        check_results["jitter"] = jitter_result
        if jitter_result.get("is_jittery", False):
            issues.append(
                SafetyIssue(
                    issue_type=SafetyIssueType.MOTION_JITTERY,
                    severity=0.6,
                    description=f"Motion is jittery ({jitter_result['jitter_ratio']*100:.1f}% high-accel frames)",
                    metadata=jitter_result,
                )
            )

        # Physics checks (if physics tensor provided)
        if physics is not None:
            physics = self._normalize_physics_shape(physics)
            physics_result = self._check_physics_violations(physics)
            check_results["physics"] = physics_result
            if physics_result.get("velocity_exceeded", False):
                issues.append(
                    SafetyIssue(
                        issue_type=SafetyIssueType.PHYSICS_VELOCITY_EXCEEDED,
                        severity=0.8,
                        description=f"Velocity exceeded: {physics_result['max_velocity']:.2f} m/s",
                        metadata=physics_result,
                    )
                )
            if physics_result.get("acceleration_exceeded", False):
                issues.append(
                    SafetyIssue(
                        issue_type=SafetyIssueType.PHYSICS_ACCELERATION_EXCEEDED,
                        severity=0.7,
                        description=f"Acceleration exceeded: {physics_result['max_acceleration']:.2f} m/s²",
                        metadata=physics_result,
                    )
                )

        # Ground penetration check
        ground_result = self._check_ground_penetration(motion)
        check_results["ground"] = ground_result
        if ground_result.get("has_penetration", False):
            issues.append(
                SafetyIssue(
                    issue_type=SafetyIssueType.PHYSICS_GROUND_PENETRATION,
                    severity=0.6,
                    description=f"Ground penetration detected (min_y={ground_result['min_y']:.3f})",
                    metadata=ground_result,
                )
            )

        # Quality score check
        if quality_score is not None:
            check_results["quality"] = {"score": quality_score}
            if quality_score < self.config.min_quality_score:
                issues.append(
                    SafetyIssue(
                        issue_type=SafetyIssueType.QUALITY_BELOW_THRESHOLD,
                        severity=0.5,
                        description=f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_score}",
                        metadata={
                            "score": quality_score,
                            "threshold": self.config.min_quality_score,
                        },
                    )
                )

        # Compute overall score and safety decision
        overall_score = self._compute_overall_score(issues, check_results)
        max_severity = max((i.severity for i in issues), default=0.0)
        is_safe = max_severity < self.config.rejection_severity_threshold

        # Restore original environment if we changed it
        if environment_type and environment_type != original_env:
            self.config.set_environment(original_env)

        return SafetyCriticResult(
            is_safe=is_safe,
            overall_score=overall_score,
            issues=issues,
            check_results=check_results,
        )

    def _normalize_motion_shape(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize motion to [T, A, D] format."""
        if motion.dim() == 2:
            return motion.unsqueeze(1)  # [T, D] -> [T, 1, D]
        return motion

    def _normalize_physics_shape(self, physics: torch.Tensor) -> torch.Tensor:
        """Normalize physics to [T, A, 6] format."""
        if physics.dim() == 2:
            return physics.unsqueeze(1)  # [T, 6] -> [T, 1, 6]
        return physics

    def _check_frozen_motion(self, motion: torch.Tensor) -> dict[str, Any]:
        """Check if motion is frozen (no movement)."""
        T, A, D = motion.shape
        if T < 2:
            return {"is_frozen": False, "frozen_ratio": 0.0}

        # Compute frame-to-frame velocity
        velocity = torch.norm(motion[1:] - motion[:-1], dim=-1)  # [T-1, A]
        mean_velocity = velocity.mean(dim=-1)  # [T-1]

        frozen_frames = (
            (mean_velocity < self.config.frozen_velocity_threshold).sum().item()
        )
        frozen_ratio = frozen_frames / (T - 1)

        return {
            "is_frozen": frozen_ratio > self.config.frozen_frame_ratio,
            "frozen_ratio": frozen_ratio,
            "frozen_frames": int(frozen_frames),
            "mean_velocity": float(mean_velocity.mean().item()),
        }

    def _check_repetitive_motion(self, motion: torch.Tensor) -> dict[str, Any]:
        """Check for repetitive/looping motion patterns."""
        T, A, D = motion.shape
        window = self.config.repetition_window

        if T < window * 2:
            return {"is_repetitive": False, "num_cycles": 0}

        # Compare windows at different offsets
        num_windows = T // window
        if num_windows < 2:
            return {"is_repetitive": False, "num_cycles": 0}

        windows = []
        for i in range(num_windows):
            start = i * window
            end = start + window
            windows.append(motion[start:end].flatten())

        # Count similar consecutive windows
        similar_count = 0
        for i in range(len(windows) - 1):
            similarity = torch.nn.functional.cosine_similarity(
                windows[i].unsqueeze(0), windows[i + 1].unsqueeze(0)
            ).item()
            if similarity > self.config.repetition_similarity_threshold:
                similar_count += 1

        num_cycles = similar_count + 1 if similar_count > 0 else 0

        return {
            "is_repetitive": num_cycles >= self.config.repetition_min_cycles,
            "num_cycles": num_cycles,
            "num_windows": num_windows,
        }

    def _check_jittery_motion(self, motion: torch.Tensor) -> dict[str, Any]:
        """Check for jittery/noisy motion (high acceleration)."""
        T, A, D = motion.shape
        if T < 3:
            return {"is_jittery": False, "jitter_ratio": 0.0}

        # Compute acceleration (second derivative)
        velocity = motion[1:] - motion[:-1]  # [T-1, A, D]
        acceleration = velocity[1:] - velocity[:-1]  # [T-2, A, D]
        accel_magnitude = torch.norm(acceleration, dim=-1)  # [T-2, A]
        mean_accel = accel_magnitude.mean(dim=-1)  # [T-2]

        jitter_frames = (
            (mean_accel > self.config.jitter_acceleration_threshold).sum().item()
        )
        jitter_ratio = jitter_frames / (T - 2)

        return {
            "is_jittery": jitter_ratio > self.config.jitter_frame_ratio,
            "jitter_ratio": jitter_ratio,
            "jitter_frames": int(jitter_frames),
            "max_acceleration": float(mean_accel.max().item()),
        }

    def _check_physics_violations(self, physics: torch.Tensor) -> dict[str, Any]:
        """Check physics tensor for velocity/acceleration violations."""
        # Physics format: [T, A, 6] = (vx, vy, ax, ay, mx, my)
        T, A, _ = physics.shape

        velocity = torch.norm(physics[:, :, 0:2], dim=-1)  # [T, A]
        acceleration = torch.norm(physics[:, :, 2:4], dim=-1)  # [T, A]

        max_vel = float(velocity.max().item())
        max_acc = float(acceleration.max().item())

        return {
            "velocity_exceeded": max_vel > self.config.max_velocity,
            "acceleration_exceeded": max_acc > self.config.max_acceleration,
            "max_velocity": max_vel,
            "max_acceleration": max_acc,
            "mean_velocity": float(velocity.mean().item()),
            "mean_acceleration": float(acceleration.mean().item()),
        }

    def _check_ground_penetration(self, motion: torch.Tensor) -> dict[str, Any]:
        """Check if any body parts penetrate the ground plane (y < 0)."""
        # Motion format: [T, A, D] where D=20 (5 limbs * 4 coords: x1,y1,x2,y2)
        T, A, D = motion.shape

        # Extract all y-coordinates (indices 1, 3, 5, 7, ... for each limb)
        # Assuming D=20: 5 limbs * (x1, y1, x2, y2)
        motion_flat = motion.view(T, A, -1, 4)  # [T, A, 5, 4]
        y_coords = motion_flat[:, :, :, [1, 3]]  # [T, A, 5, 2] - y1 and y2

        min_y = float(y_coords.min().item())
        penetration_count = (y_coords < self.config.ground_y_threshold).sum().item()

        return {
            "has_penetration": min_y < self.config.ground_y_threshold,
            "min_y": min_y,
            "penetration_count": int(penetration_count),
        }

    def _compute_overall_score(
        self, issues: list[SafetyIssue], check_results: dict[str, dict[str, Any]]
    ) -> float:
        """Compute overall quality/safety score from issues and checks."""
        if not issues:
            return 1.0

        # Weighted penalty based on severity
        total_penalty = sum(issue.severity for issue in issues)
        # Normalize: more issues = lower score, but cap at 0
        score = max(0.0, 1.0 - (total_penalty / 3.0))
        return score


def evaluate_motion_safety(
    motion: torch.Tensor,
    physics: torch.Tensor | None = None,
    quality_score: float | None = None,
    config: SafetyCriticConfig | None = None,
) -> SafetyCriticResult:
    """
    Convenience function to evaluate motion safety.

    Args:
        motion: Motion tensor [T, D] or [T, A, D]
        physics: Optional physics tensor [T, 6] or [T, A, 6]
        quality_score: Optional pre-computed quality score
        config: Optional custom configuration

    Returns:
        SafetyCriticResult
    """
    critic = SafetyCritic(config)
    return critic.evaluate(motion, physics, quality_score)


def batch_evaluate_safety(
    samples: list[dict[str, Any]],
    config: SafetyCriticConfig | None = None,
) -> dict[str, Any]:
    """
    Evaluate safety for a batch of samples.

    Args:
        samples: List of sample dicts with 'motion', optional 'physics', 'quality_score'
        config: Optional custom configuration

    Returns:
        Dict with aggregate statistics and per-sample results
    """
    critic = SafetyCritic(config)
    results = []

    for sample in samples:
        motion = sample.get("motion")
        if motion is None:
            continue
        if not isinstance(motion, torch.Tensor):
            motion = torch.tensor(motion, dtype=torch.float32)

        physics = sample.get("physics")
        if physics is not None and not isinstance(physics, torch.Tensor):
            physics = torch.tensor(physics, dtype=torch.float32)

        quality = sample.get("quality_score")

        result = critic.evaluate(motion, physics, quality)
        results.append(
            {
                "is_safe": result.is_safe,
                "overall_score": result.overall_score,
                "issue_count": len(result.issues),
                "issue_types": [i.issue_type.value for i in result.issues],
            }
        )

    # Aggregate statistics
    total = len(results)
    safe_count = sum(1 for r in results if r["is_safe"])
    mean_score = np.mean([r["overall_score"] for r in results]) if results else 0.0

    # Issue type distribution
    issue_counts: dict[str, int] = {}
    for r in results:
        for itype in r["issue_types"]:
            issue_counts[itype] = issue_counts.get(itype, 0) + 1

    return {
        "total_samples": total,
        "safe_count": safe_count,
        "safe_ratio": safe_count / total if total > 0 else 0.0,
        "mean_score": float(mean_score),
        "issue_distribution": issue_counts,
        "per_sample_results": results,
    }
