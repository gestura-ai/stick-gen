from typing import Any, Optional

import math
import numpy as np
import torch


# Environment-specific physics threshold multipliers
# These adjust validation thresholds based on environment physics.
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
    """Validate stick-figure motion for physical and structural realism.

    The validator operates on motion tensors interpreted as concatenated 2D
    line segments of the form ``[..., segments * 4]`` where each segment is
    ``(x1, y1, x2, y2)``. It supports both single-actor ``[F, D]`` and
    multi-actor ``[F, A, D]`` layouts and can be configured for different
    environment types (e.g., underwater, space, ice).
    """

    def __init__(self, fps: int = 25, environment_type: Optional[str] = None) -> None:
        """Initialize a new :class:`DataValidator`.

        Args:
            fps: Frames per second for the motion sequences.
            environment_type: Optional environment type key used to scale
                physics thresholds via ``ENVIRONMENT_*_MULTIPLIERS``.
        """

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

        # 15% variance allowed in limb length (for interpolation artifacts).
        self.limb_length_tolerance = 0.15

        # Minimum allowed distance between different actors (for multi-character
        # scenes). This is deliberately small and intended as a soft
        # diagnostic; callers can choose whether/how to use it.
        self.min_interactor_distance = 0.02
        # Max fraction of frames that may contain near-collisions.
        self.max_collision_fraction = 0.1

    # ---------------------------------------------------------------------
    # Environment helpers
    # ---------------------------------------------------------------------
    def set_environment(self, environment_type: Optional[str]) -> None:
        """Update validator thresholds for a new environment type.

        Args:
            environment_type: New environment key or ``None`` for defaults.
        """

        self.environment_type = environment_type
        vel_mult = ENVIRONMENT_VELOCITY_MULTIPLIERS.get(environment_type, 1.0)
        acc_mult = ENVIRONMENT_ACCELERATION_MULTIPLIERS.get(environment_type, 1.0)
        self.max_velocity = self.base_max_velocity * vel_mult
        self.max_acceleration = self.base_max_acceleration * acc_mult

    # ---------------------------------------------------------------------
    # Core physics and structure checks
    # ---------------------------------------------------------------------
    def check_physics_consistency(
        self, physics_tensor: torch.Tensor, clip_fps: Optional[int] = None
    ) -> tuple[bool, float, str]:
        """Check that velocity and acceleration are within realistic bounds.

        Args:
            physics_tensor: Tensor of shape ``[F, A, 6]`` or ``[F, 6]`` with
                components ``(vx, vy, ax, ay, mx, my)``.
            clip_fps: Optional fps of the specific clip. If provided and differs
                from the validator's base fps, thresholds are scaled accordingly.
                This is important because:
                - Velocity scales linearly with fps (v = dx * fps)
                - Acceleration scales quadratically with fps (a = dv * fps = dx * fps^2)
                So the same physical motion at 60fps will have 2.4x higher velocity
                and 5.76x higher acceleration values compared to 25fps.

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        # Support both [F, A, 6] (multi-actor) and [F, 6] (single-actor) formats.
        # Convert lists and numpy arrays to tensors if needed.
        if isinstance(physics_tensor, list):
            physics_tensor = torch.tensor(physics_tensor, dtype=torch.float32)
        if isinstance(physics_tensor, np.ndarray):
            physics_tensor = torch.from_numpy(physics_tensor)
        if physics_tensor.dim() == 2:
            physics_tensor = physics_tensor.unsqueeze(1)  # [F, 1, 6]
        elif physics_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected physics tensor shape: {tuple(physics_tensor.shape)}",
            )

        # Compute fps-aware threshold scaling
        # If clip_fps is provided and differs from validator's base fps,
        # scale thresholds to account for the fps difference.
        if clip_fps is not None and clip_fps != self.fps:
            fps_ratio = clip_fps / self.fps
            # Velocity scales linearly with fps
            vel_scale = fps_ratio
            # Acceleration scales quadratically with fps
            acc_scale = fps_ratio ** 2
        else:
            vel_scale = 1.0
            acc_scale = 1.0

        effective_max_velocity = self.max_velocity * vel_scale
        effective_max_acceleration = self.max_acceleration * acc_scale

        # Magnitudes
        velocity = torch.linalg.norm(physics_tensor[:, :, 0:2], dim=2)  # [F, A]
        acceleration = torch.linalg.norm(physics_tensor[:, :, 2:4], dim=2)  # [F, A]

        max_v = velocity.max().item()
        max_a = acceleration.max().item()

        if max_v > effective_max_velocity:
            return (
                False,
                0.0,
                f"Velocity limit exceeded: {max_v:.2f} > {effective_max_velocity:.2f}",
            )

        if max_a > effective_max_acceleration:
            return (
                False,
                0.0,
                f"Acceleration limit exceeded: {max_a:.2f} > {effective_max_acceleration:.2f}",
            )

        # For now, if valid, return a neutral score of 1.0. This can be refined
        # later to penalise jittery motion.
        return True, 1.0, "Physics OK"

    def check_skeleton_consistency(
        self, motion_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Check that limb lengths remain consistent over time.

        Args:
            motion_tensor: Tensor of shape ``[F, A, D]`` or ``[F, D]`` where
                ``D`` is a multiple of 4 (segments × 4 endpoints).

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        # Support both [F, A, D] (multi-actor) and [F, D] (single-actor) layouts.
        # Convert lists and numpy arrays to tensors if needed.
        if isinstance(motion_tensor, list):
            motion_tensor = torch.tensor(motion_tensor, dtype=torch.float32)
        if isinstance(motion_tensor, np.ndarray):
            motion_tensor = torch.from_numpy(motion_tensor)
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(1)  # [F, 1, D]
        elif motion_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected motion tensor shape: {tuple(motion_tensor.shape)}",
            )

        frames, actors, dim = motion_tensor.shape
        if dim % 4 != 0:
            return (
                False,
                0.0,
                "Skeleton consistency expects last dimension to be a multiple "
                f"of 4 (segments × 4 endpoints), got D={dim}",
            )

        num_segments = dim // 4
        reshaped = motion_tensor.view(frames, actors, num_segments, 4)  # [F, A, S, 4]

        # Length = sqrt((x2-x1)^2 + (y2-y1)^2)
        diffs = reshaped[:, :, :, 2:4] - reshaped[:, :, :, 0:2]  # [F, A, S, 2]
        lengths = torch.linalg.norm(diffs, dim=3)  # [F, A, S]

        # Mean length for each segment/actor.
        mean_lengths = lengths.mean(dim=0)  # [A, S]

        # Skip actors that are effectively padding (all-zero limbs).
        active_mask = mean_lengths.sum(dim=1) > 0.01  # [A]

        for actor_idx in range(actors):
            if not active_mask[actor_idx]:
                continue

            for seg_idx in range(num_segments):
                seg_lengths = lengths[:, actor_idx, seg_idx]
                mean_length = mean_lengths[actor_idx, seg_idx]

                if mean_length < 0.001:
                    # Degenerate or zero-length segment; ignore.
                    continue

                deviations = torch.abs(seg_lengths - mean_length) / mean_length
                max_dev = deviations.max().item()

                if max_dev > self.limb_length_tolerance:
                    return (
                        False,
                        0.0,
                        "Skeleton inconsistency: Actor "
                        f"{actor_idx} Segment {seg_idx} varies by "
                        f"{max_dev*100:.1f}%",
                    )

        return True, 1.0, "Skeleton OK"

    def check_interaction_consistency(
        self, motion_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Optional check for multi-actor interactions (e.g., collisions).

        This treats each actor as a single point (average of all segment
        endpoints) and measures how often different actors come extremely close
        to each other.

        Args:
            motion_tensor: Tensor of shape ``[F, A, D]`` or ``[F, D]`` where
                ``D`` is a multiple of 4 (segments × 4 endpoints).

        Returns:
            Tuple ``(is_valid, score, reason)``.

        Notes:
            - This is *not* used inside :meth:`validate` to avoid
              over-constraining datasets that legitimately contain contact
              (e.g., hugs, grappling).
            - Callers (evaluation code, dataset curation) can use it as a
              diagnostic for extreme or degenerate overlaps.
        """

        # Normalise to [F, A, D].
        # Convert lists and numpy arrays to tensors if needed.
        if isinstance(motion_tensor, list):
            motion_tensor = torch.tensor(motion_tensor, dtype=torch.float32)
        if isinstance(motion_tensor, np.ndarray):
            motion_tensor = torch.from_numpy(motion_tensor)
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(1)
        elif motion_tensor.dim() != 3:
            return (
                False,
                0.0,
                f"Unexpected motion tensor shape: {tuple(motion_tensor.shape)}",
            )

        frames, actors, dim = motion_tensor.shape
        if actors <= 1:
            # Single-actor: nothing to check.
            return True, 1.0, "Single actor (no interactions)"

        if dim % 4 != 0:
            return (
                False,
                0.0,
                "Interaction consistency expects last dimension to be a "
                f"multiple of 4 (segments × 4 endpoints), got D={dim}",
            )

        num_segments = dim // 4
        segments = motion_tensor.view(frames, actors, num_segments, 4)
        starts = segments[..., 0:2]
        ends = segments[..., 2:4]
        points = torch.cat([starts, ends], dim=2)  # [F, A, 2S, 2]
        centers = points.mean(dim=2)  # [F, A, 2]

        # Pairwise distances between actors per frame: [F, A, A].
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        dists = torch.linalg.norm(diff, dim=-1)

        # Ignore self-distances along the diagonal.
        mask = ~torch.eye(actors, dtype=torch.bool, device=dists.device)
        dists_pairs = dists[:, mask].view(frames, -1)

        min_dist = dists_pairs.min().item()
        # Fraction of frames where any pair is closer than the
        # ``min_interactor_distance``.
        near_mask = dists_pairs < self.min_interactor_distance
        frames_with_collision = near_mask.any(dim=1)
        collision_fraction = frames_with_collision.float().mean().item()

        if collision_fraction > self.max_collision_fraction:
            return (
                False,
                0.0,
                (
                    "Interaction inconsistency: "
                    f"{collision_fraction*100:.1f}% frames have actors nearer "
                    f"than {self.min_interactor_distance:.3f} "
                    f"(min={min_dist:.4f})"
                ),
            )

        return True, 1.0, "Interactions OK"

    def check_joint_angles_v3(
        self, motion_tensor: torch.Tensor
    ) -> tuple[bool, float, str]:
        """Check basic elbow and knee joint angles for v3 12-segment motion.

        This check is deliberately lightweight and only runs when
        ``motion_tensor`` is in the canonical single-actor v3 layout
        (``[frames, 48]`` or ``[frames, 1, 48]``). For all other shapes or
        dimensionalities the check is skipped and a neutral score of ``1.0``
        is returned.

        The goal is to catch obviously collapsed joints (near-0 degree
        angles) at elbows and knees while respecting the requirement that
        joints remain connected in 2D and 2.5D/3D space.

        Args:
            motion_tensor: Motion tensor with concatenated 2D segment endpoints.

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        # Normalise to a single-actor [F, D] view.
        # Convert lists and numpy arrays to tensors if needed.
        if isinstance(motion_tensor, list):
            motion_tensor = torch.tensor(motion_tensor, dtype=torch.float32)
        if isinstance(motion_tensor, np.ndarray):
            motion_tensor = torch.from_numpy(motion_tensor)
        if motion_tensor.dim() == 2:
            frames, dim = motion_tensor.shape
            motion_flat = motion_tensor
        elif motion_tensor.dim() == 3 and motion_tensor.shape[1] == 1:
            frames, _actors, dim = motion_tensor.shape
            motion_flat = motion_tensor[:, 0, :]
        else:
            # Multi-actor or unexpected layout: skip angle check.
            return True, 1.0, "Joint angle check skipped (multi-actor or shape)"

        if dim != 48 or frames < 1:
            # Only canonical v3 (12 segments × 4) is supported here.
            return True, 1.0, "Joint angle check skipped (non-v3 dimensionality)"

        # [F, 48] -> [F, 12, 4]
        segs = motion_flat.view(frames, 12, 4)

        # Reconstruct the minimal joint set needed for elbow and knee angles.
        l_shoulder = segs[:, 3, 0:2]
        l_elbow = segs[:, 3, 2:4]
        l_wrist = segs[:, 4, 2:4]

        r_shoulder = segs[:, 5, 0:2]
        r_elbow = segs[:, 5, 2:4]
        r_wrist = segs[:, 6, 2:4]

        l_hip = segs[:, 11, 0:2]
        r_hip = segs[:, 11, 2:4]
        l_knee = segs[:, 7, 2:4]
        l_ankle = segs[:, 8, 2:4]
        r_knee = segs[:, 9, 2:4]
        r_ankle = segs[:, 10, 2:4]

        def _joint_angle(
            parent: torch.Tensor, center: torch.Tensor, child: torch.Tensor
        ) -> torch.Tensor:
            """Compute angle at ``center`` between segments.

            The angle is formed by ``(parent - center)`` and ``(child - center)``.
            Frames where either limb segment is effectively zero-length are
            assigned a neutral 90-degree angle so they do not influence the
            violation ratio.
            """

            v1 = parent - center
            v2 = child - center
            v1_norm = torch.linalg.norm(v1, dim=-1)
            v2_norm = torch.linalg.norm(v2, dim=-1)
            denom = v1_norm * v2_norm

            valid = denom > 1e-6
            cos_theta = torch.ones_like(denom)
            if valid.any():
                cos_theta_valid = (v1[valid] * v2[valid]).sum(dim=-1) / denom[valid]
                cos_theta_valid = torch.clamp(cos_theta_valid, -1.0, 1.0)
                cos_theta[valid] = cos_theta_valid

            theta = torch.acos(cos_theta)  # Radians in [0, pi].
            angle_deg = theta * (180.0 / math.pi)

            # Fill invalid entries with a neutral 90 degrees.
            angle_deg = torch.where(
                valid, angle_deg, torch.full_like(angle_deg, 90.0)
            )
            return angle_deg

        # Elbow and knee angles per frame.
        l_elbow_angle = _joint_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = _joint_angle(r_shoulder, r_elbow, r_wrist)
        l_knee_angle = _joint_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = _joint_angle(r_hip, r_knee, r_ankle)

        angles = torch.stack(
            [l_elbow_angle, r_elbow_angle, l_knee_angle, r_knee_angle], dim=1
        )  # [F, 4]

        # Enforce a conservative lower bound to catch near-collapsed joints.
        # Fully extended 180-degree poses are allowed since 2D projection can
        # flatten otherwise healthy 3D poses.
        min_angle = 5.0  # degrees
        invalid = angles < min_angle

        total = angles.numel()
        if total == 0:
            return True, 1.0, "Joint angle check skipped (no joints)"

        num_invalid = int(invalid.sum().item())
        violation_fraction = float(num_invalid) / float(total)

        if violation_fraction == 0.0:
            return True, 1.0, "Joint angles OK"

        # Soft scoring: allow small fractions of unusual poses but fail if a
        # significant portion of frames are collapsed.
        score = max(0.0, 1.0 - violation_fraction / 0.5)

        if violation_fraction > 0.1:
            return (
                False,
                float(score),
                (
                    "Joint angle inconsistency: "
                    f"{violation_fraction*100:.1f}% elbow/knee angles below "
                    f"{min_angle:.1f} degrees"
                ),
            )

        return (
            True,
            float(score),
            (
                "Joint angles mostly OK ("
                f"{violation_fraction*100:.1f}% below {min_angle:.1f} degrees)"
                ")"
            ),
        )

    # ------------------------------------------------------------------
    # Enhanced metadata checks
    # ------------------------------------------------------------------
    def check_motion_style_ranges(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate motion-style metadata values are within expected ranges.

        Args:
            enhanced_meta: Optional enhanced metadata dictionary from
                ``sample["enhanced_meta"]``.

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        if enhanced_meta is None:
            return True, 1.0, "No enhanced metadata (optional)"

        motion_style = enhanced_meta.get("motion_style")
        if motion_style is None:
            return True, 1.0, "No motion style metadata (optional)"

        issues: list[str] = []
        for field in ["tempo", "energy_level", "smoothness"]:
            value = motion_style.get(field)
            if value is not None and not (0.0 <= value <= 1.0):
                issues.append(f"{field}={value:.3f} out of [0,1]")

        if issues:
            return False, 0.0, f"Motion style range error: {'; '.join(issues)}"

        return True, 1.0, "Motion style ranges OK"

    def check_temporal_metadata(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate temporal metadata values are sensible.

        Args:
            enhanced_meta: Optional enhanced metadata dictionary from
                ``sample["enhanced_meta"]``.

        Returns:
            Tuple ``(is_valid, score, reason)``.
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
            enhanced_meta: Optional enhanced metadata dictionary from
                ``sample["enhanced_meta"]``.

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        if enhanced_meta is None:
            return True, 1.0, "No enhanced metadata (optional)"

        quality = enhanced_meta.get("quality")
        if quality is None:
            return True, 1.0, "No quality metadata (optional)"

        issues: list[str] = []
        for field in ["reconstruction_confidence", "marker_quality"]:
            value = quality.get(field)
            if value is not None and not (0.0 <= value <= 1.0):
                issues.append(f"{field}={value:.3f} out of [0,1]")

        if issues:
            return False, 0.0, f"Quality range error: {'; '.join(issues)}"

        return True, 1.0, "Quality ranges OK"

    def check_enhanced_metadata(
        self, enhanced_meta: dict[str, Any] | None
    ) -> tuple[bool, float, str]:
        """Validate all enhanced metadata fields.

        Args:
            enhanced_meta: Optional enhanced metadata dictionary from
                ``sample["enhanced_meta"]``.

        Returns:
            Tuple ``(is_valid, score, reason)``.
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

    # ------------------------------------------------------------------
    # Top-level sample validation
    # ------------------------------------------------------------------
    def validate(self, sample: dict[str, Any]) -> tuple[bool, float, str]:
        """Validate a generated sample.

        Args:
            sample: Dictionary containing at least ``"motion"`` and
                ``"physics"``. It may optionally contain ``"environment_type"``
                and ``"enhanced_meta"``.

        Returns:
            Tuple ``(is_valid, score, reason)``.
        """

        # Temporarily adjust thresholds if the sample has environment metadata.
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

            # Advanced v3 structural check: basic elbow/knee joint angles.
            angles_ok, angles_score, angles_reason = self.check_joint_angles_v3(
                sample["motion"]
            )
            if not angles_ok:
                return False, angles_score, angles_reason

            # Validate enhanced metadata if present.
            enhanced_meta = sample.get("enhanced_meta")
            if enhanced_meta is not None:
                meta_ok, meta_score, meta_reason = self.check_enhanced_metadata(
                    enhanced_meta
                )
                if not meta_ok:
                    return False, meta_score, meta_reason

                final_score = (phys_score + skel_score + meta_score) / 3.0
            else:
                final_score = (phys_score + skel_score) / 2.0

            return True, final_score, "Valid"
        finally:
            # Restore original environment.
            if sample_env and sample_env != original_env:
                self.set_environment(original_env)
