from typing import Any, Optional

import torch

# Environment-specific physics threshold multipliers
# These adjust validation thresholds based on environment physics
ENVIRONMENT_VELOCITY_MULTIPLIERS = {
    # Low gravity environments allow faster movement
    "space_vacuum": 0.5,  # Very low expected velocity in zero-g (floating)
    "moon": 1.5,  # Higher velocities possible in low-g
    "mars": 1.3,
    "asteroid": 0.5,
    "alien_planet_low_g": 1.5,
    "cloud_realm": 0.6,  # Floating motion
    # High resistance environments have slower movement
    "underwater": 0.4,  # Much slower underwater
    "ocean_surface": 0.6,
    "river": 0.5,
    "lake": 0.5,
    "pool": 0.5,
    "swamp": 0.6,
    # Slippery surfaces can have higher velocities (sliding)
    "rink": 1.5,  # Ice skating can be fast
    "arctic": 1.2,
    "ice_realm": 1.3,
    # Normal environments
    "earth_normal": 1.0,
    "grassland": 1.0,
    "forest": 0.9,
    "desert": 0.9,  # Sand slows movement
    "beach": 0.9,
    "mountain": 0.8,  # Uphill effort
    "city_street": 1.0,
    "stadium": 1.2,  # Sports can be fast
    "arena": 1.2,
    "track": 1.3,  # Running track
    "field": 1.1,
}

ENVIRONMENT_ACCELERATION_MULTIPLIERS = {
    # Low gravity = lower acceleration needed
    "space_vacuum": 0.3,
    "moon": 0.5,
    "mars": 0.7,
    "asteroid": 0.3,
    "alien_planet_low_g": 0.5,
    "cloud_realm": 0.4,
    # High resistance = lower acceleration possible
    "underwater": 0.4,
    "ocean_surface": 0.5,
    "river": 0.5,
    "lake": 0.5,
    "pool": 0.5,
    "swamp": 0.5,
    # Low friction = can have sudden direction changes
    "rink": 0.7,  # Less friction means less sudden stops
    "arctic": 0.8,
    "ice_realm": 0.7,
    # Normal environments
    "earth_normal": 1.0,
    "stadium": 1.2,
    "arena": 1.2,
    "track": 1.2,
}


class DataValidator:
    """
    Validates generated stick figure motion data for physical realism and structural consistency.

    Supports environment-aware validation with adjusted thresholds based on
    environment type (e.g., underwater, space, ice).
    """

    def __init__(self, fps: int = 25, environment_type: Optional[str] = None):
        self.fps = fps
        self.dt = 1.0 / fps
        self.environment_type = environment_type

        # Base thresholds (for Earth-normal conditions)
        self.base_max_velocity = 20.0  # Units/sec (approx 13m/s)
        self.base_max_acceleration = 100.0  # Units/sec^2

        # Apply environment multipliers
        vel_mult = ENVIRONMENT_VELOCITY_MULTIPLIERS.get(environment_type, 1.0)
        acc_mult = ENVIRONMENT_ACCELERATION_MULTIPLIERS.get(environment_type, 1.0)

        self.max_velocity = self.base_max_velocity * vel_mult
        self.max_acceleration = self.base_max_acceleration * acc_mult

        self.limb_length_tolerance = (
            0.15  # 15% variance allowed in limb length (interpolation artifacts)
        )
        # Minimum allowed distance between different actors (for multi-character scenes).
        # This is deliberately very small and intended as a soft diagnostic; callers
        # can choose whether/how to use it.
        self.min_interactor_distance = 0.02
        self.max_collision_fraction = 0.1  # Max fraction of frames with near-collisions

        # Standard Limb Indices in 20-float array (5 lines * 4 coords)
        # 0: Head-Torso
        # 1: Left Arm
        # 2: Right Arm
        # 3: Left Leg
        # 4: Right Leg
        self.num_limbs = 5

    def set_environment(self, environment_type: Optional[str]) -> None:
        """Update validator thresholds for a new environment type."""
        self.environment_type = environment_type
        vel_mult = ENVIRONMENT_VELOCITY_MULTIPLIERS.get(environment_type, 1.0)
        acc_mult = ENVIRONMENT_ACCELERATION_MULTIPLIERS.get(environment_type, 1.0)
        self.max_velocity = self.base_max_velocity * vel_mult
        self.max_acceleration = self.base_max_acceleration * acc_mult

    def check_physics_consistency(
        self, physics_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Check if velocity and acceleration are within realistic bounds.

        Args:
            physics_tensor: [frames, actors, 6] or [frames, 6]
                (vx, vy, ax, ay, mx, my)

        Returns:
            (is_valid, score, reason)
        """

        # Support both [F, A, 6] (multi-actor) and [F, 6] (single-actor) formats
        if physics_tensor.dim() == 2:
            physics_tensor = physics_tensor.unsqueeze(1)  # [F, 1, 6]
        elif physics_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected physics tensor shape: {tuple(physics_tensor.shape)}",
            )

        # Magnitudes
        velocity = torch.linalg.norm(
            physics_tensor[:, :, 0:2], dim=2
        )  # [frames, actors]
        acceleration = torch.linalg.norm(
            physics_tensor[:, :, 2:4], dim=2
        )  # [frames, actors]

        max_v = velocity.max().item()
        max_a = acceleration.max().item()

        if max_v > self.max_velocity:
            return (
                False,
                0.0,
                f"Velocity limit exceeded: {max_v:.2f} > {self.max_velocity}",
            )

        if max_a > self.max_acceleration:
            return (
                False,
                0.0,
                f"Acceleration limit exceeded: {max_a:.2f} > {self.max_acceleration}",
            )

        # Score based on how close to limit (closer to 0 is 'safer', but we want 1.0 is good)
        # Simple linear penalty if we are somewhat high but valid?
        # For now, if valid, return 1.0. We can refine to punish "jittery" motion later.
        return True, 1.0, "Physics OK"

    def check_skeleton_consistency(
        self, motion_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Check if limb lengths remain consistent over time.

        Args:
            motion_tensor: [frames, actors, 20] or [frames, 20]

        Returns:
            (is_valid, score, reason)
        """

        # Support both [F, A, 20] (multi-actor) and [F, 20] (single-actor)
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(1)  # [F, 1, 20]
        elif motion_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected motion tensor shape: {tuple(motion_tensor.shape)}",
            )

        frames, actors, _ = motion_tensor.shape
        reshaped = motion_tensor.view(frames, actors, 5, 4)  # [F, A, Lines, 4]

        # Calculate length of each limb for each frame/actor
        # Length = sqrt((x2-x1)^2 + (y2-y1)^2)
        diffs = reshaped[:, :, :, 2:4] - reshaped[:, :, :, 0:2]  # [F, A, L, 2]
        lengths = torch.linalg.norm(diffs, dim=3)  # [F, A, L]

        # Check variance over time for each limb of each actor
        # Calculate mean length for each limb/actor
        mean_lengths = lengths.mean(dim=0)  # [A, L]

        # Verify valid actors (skip if actor is all zeros i.e. padding)
        # If mean length is effectively 0, it's a padded actor
        active_mask = mean_lengths.sum(dim=1) > 0.01  # [A] boolean

        for a in range(actors):
            if not active_mask[a]:
                continue

            for limb_idx in range(5):
                # Get lengths for this limb over time
                l_lengths = lengths[:, a, limb_idx]
                mean_l = mean_lengths[a, limb_idx]

                if mean_l < 0.001:  # Zero length limb? (e.g. head point?)
                    continue

                # Calculate max deviation
                deviations = torch.abs(l_lengths - mean_l) / mean_l
                max_dev = deviations.max().item()

                if max_dev > self.limb_length_tolerance:
                    return (
                        False,
                        0.0,
                        f"Skeleton inconsistency: Actor {a} Limb {limb_idx} varies by {max_dev*100:.1f}%",
                    )

        return True, 1.0, "Skeleton OK"

    def check_interaction_consistency(
        self, motion_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Optional check for multi-actor interactions (e.g., collisions).

        This treats each actor as a single point (average of all segment endpoints)
        and measures how often different actors come extremely close to each other.

        Args:
            motion_tensor: [frames, actors, 20] or [frames, 20]

        Returns:
            (is_valid, score, reason)

        Notes:
            - This is *not* used inside ``validate`` to avoid over-constraining
              datasets that legitimately contain contact (e.g., hugs, grappling).
            - Callers (evaluation code, dataset curation) can use it as a
              diagnostic for extreme or degenerate overlaps.
        """

        # Normalize shape to [F, A, 20]
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(1)
        elif motion_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected motion tensor shape: {tuple(motion_tensor.shape)}",
            )

        F, A, _ = motion_tensor.shape
        if A <= 1:
            # Single-actor: nothing to check.
            return True, 1.0, "Single actor (no interactions)"

        # [F, A, 5, 4]
        segments = motion_tensor.view(F, A, 5, 4)
        starts = segments[..., 0:2]  # [F, A, 5, 2]
        ends = segments[..., 2:4]  # [F, A, 5, 2]
        points = torch.cat([starts, ends], dim=2)  # [F, A, 10, 2]
        centers = points.mean(dim=2)  # [F, A, 2]

        # Pairwise distances between actors per frame: [F, A, A]
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        dists = torch.linalg.norm(diff, dim=-1)

        # Ignore self-distances along the diagonal.
        mask = ~torch.eye(A, dtype=torch.bool, device=dists.device)
        dists_pairs = dists[:, mask].view(F, -1)  # [F, A*(A-1)]

        min_dist = dists_pairs.min().item()
        # Fraction of frames where any pair is closer than the min_interactor_distance
        near_mask = dists_pairs < self.min_interactor_distance
        frames_with_collision = near_mask.any(dim=1)
        collision_fraction = frames_with_collision.float().mean().item()

        if collision_fraction > self.max_collision_fraction:
            return (
                False,
                0.0,
                (
                    f"Interaction inconsistency: {collision_fraction*100:.1f}% frames "
                    f"have actors nearer than {self.min_interactor_distance:.3f} (min={min_dist:.4f})"
                ),
            )

        return True, 1.0, "Interactions OK"

    def check_motion_style_ranges(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate motion style metadata values are within expected ranges.

        Args:
            enhanced_meta: Enhanced metadata dict (from sample["enhanced_meta"])

        Returns:
            (is_valid, score, reason)
        """
        if enhanced_meta is None:
            return True, 1.0, "No enhanced metadata (optional)"

        motion_style = enhanced_meta.get("motion_style")
        if motion_style is None:
            return True, 1.0, "No motion style metadata (optional)"

        issues: list[str] = []
        for field in ["tempo", "energy_level", "smoothness"]:
            value = motion_style.get(field)
            if value is not None:
                if not (0.0 <= value <= 1.0):
                    issues.append(f"{field}={value:.3f} out of [0,1]")

        if issues:
            return False, 0.0, f"Motion style range error: {'; '.join(issues)}"
        return True, 1.0, "Motion style ranges OK"

    def check_temporal_metadata(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate temporal metadata values are sensible.

        Args:
            enhanced_meta: Enhanced metadata dict (from sample["enhanced_meta"])

        Returns:
            (is_valid, score, reason)
        """
        if enhanced_meta is None:
            return True, 1.0, "No enhanced metadata (optional)"

        temporal = enhanced_meta.get("temporal")
        if temporal is None:
            return True, 1.0, "No temporal metadata (optional)"

        issues: list[str] = []

        fps = temporal.get("original_fps")
        if fps is not None and (fps < 1 or fps > 240):
            issues.append(f"original_fps={fps} out of [1,240]")

        num_frames = temporal.get("original_num_frames")
        if num_frames is not None and num_frames < 1:
            issues.append(f"original_num_frames={num_frames} < 1")

        duration = temporal.get("original_duration_sec")
        if duration is not None and duration < 0:
            issues.append(f"original_duration_sec={duration} < 0")

        if issues:
            return False, 0.0, f"Temporal metadata error: {'; '.join(issues)}"
        return True, 1.0, "Temporal metadata OK"

    def check_quality_ranges(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate quality metadata values are within expected ranges.

        Args:
            enhanced_meta: Enhanced metadata dict (from sample["enhanced_meta"])

        Returns:
            (is_valid, score, reason)
        """
        if enhanced_meta is None:
            return True, 1.0, "No enhanced metadata (optional)"

        quality = enhanced_meta.get("quality")
        if quality is None:
            return True, 1.0, "No quality metadata (optional)"

        issues: list[str] = []
        for field in ["reconstruction_confidence", "marker_quality"]:
            value = quality.get(field)
            if value is not None:
                if not (0.0 <= value <= 1.0):
                    issues.append(f"{field}={value:.3f} out of [0,1]")

        if issues:
            return False, 0.0, f"Quality range error: {'; '.join(issues)}"
        return True, 1.0, "Quality ranges OK"

    def check_enhanced_metadata(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate all enhanced metadata fields.

        Args:
            enhanced_meta: Enhanced metadata dict (from sample["enhanced_meta"])

        Returns:
            (is_valid, score, reason)
        """
        checks = [
            self.check_motion_style_ranges(enhanced_meta),
            self.check_temporal_metadata(enhanced_meta),
            self.check_quality_ranges(enhanced_meta),
        ]

        for is_valid, score, reason in checks:
            if not is_valid:
                return is_valid, score, reason

        return True, 1.0, "Enhanced metadata OK"

    def validate(self, sample: dict[str, Any]) -> tuple[bool, float, str]:
        """
        Validate a generated sample.

        Args:
            sample: Dictionary containing 'motion', 'physics', etc.
                    Optionally contains 'environment_type' for environment-aware validation.
                    Optionally contains 'enhanced_meta' for metadata validation.

        Returns:
            (is_valid, score, reason)
        """
        # Temporarily adjust thresholds if sample has environment metadata
        sample_env = sample.get("environment_type")
        original_env = self.environment_type
        if sample_env and sample_env != self.environment_type:
            self.set_environment(sample_env)

        try:
            physics_ok, phys_score, phys_reason = self.check_physics_consistency(
                sample["physics"]
            )
            if not physics_ok:
                return False, phys_score, phys_reason

            skel_ok, skel_score, skel_reason = self.check_skeleton_consistency(
                sample["motion"]
            )
            if not skel_ok:
                return False, skel_score, skel_reason

            # Validate enhanced metadata if present
            enhanced_meta = sample.get("enhanced_meta")
            if enhanced_meta is not None:
                meta_ok, meta_score, meta_reason = self.check_enhanced_metadata(
                    enhanced_meta
                )
                if not meta_ok:
                    return False, meta_score, meta_reason
                # Include metadata score in average
                final_score = (phys_score + skel_score + meta_score) / 3.0
            else:
                # Combined score (average if both valid)
                final_score = (phys_score + skel_score) / 2.0

            return True, final_score, "Valid"
        finally:
            # Restore original environment
            if sample_env and sample_env != original_env:
                self.set_environment(original_env)
