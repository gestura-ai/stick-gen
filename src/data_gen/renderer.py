import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from .camera import Camera
from .schema import (
    ACTION_VELOCITIES,
    ActionType,
    Actor,
    ActorType,
    EnvironmentProfile,
    EnvironmentType,
    FaceFeatures,
    FacialExpression,
    MotionProfile,
    MouthShape,
    ObjectType,
    Scene,
    combine_environment_profiles,
    get_environment_profile,
    get_motion_profile,
)


# Environment-specific visual configurations for rendering
ENVIRONMENT_VISUAL_CONFIGS = {
    # Format: (background_color, tint_color, tint_alpha, particle_type)
    # particle_type: None, "rain", "snow", "bubbles", "stars", "ash", "leaves", "sand"
    "earth_normal": ("#f0f8ff", None, 0.0, None),  # Light sky blue, no tint
    "underwater": ("#001a33", "#0066cc", 0.15, "bubbles"),  # Deep blue with bubble particles
    "space": ("#000000", "#0a0a20", 0.1, "stars"),  # Black with star particles
    "moon": ("#1a1a1a", "#404040", 0.1, None),  # Dark gray lunar surface
    "forest": ("#1a3d1a", "#228b22", 0.08, "leaves"),  # Forest green with falling leaves
    "desert": ("#f4a460", "#daa520", 0.1, "sand"),  # Sandy with sand particles
    "arctic": ("#e0f0ff", "#ffffff", 0.15, "snow"),  # Ice white with snow
    "volcanic": ("#1a0a0a", "#ff4500", 0.12, "ash"),  # Dark with orange glow and ash
    "urban": ("#2d3748", "#708090", 0.08, None),  # City gray
    "cyberpunk": ("#0a0514", "#ff00ff", 0.1, None),  # Dark purple with neon tint
    "fantasy": ("#1a0a2e", "#ff80c0", 0.08, None),  # Magical purple-pink
    "horror": ("#0a0505", "#4a0000", 0.15, None),  # Dark red
    "sunset": ("#4b0082", "#ff7f50", 0.12, None),  # Indigo with coral tint
    "sunrise": ("#191970", "#ffd700", 0.1, None),  # Navy with golden tint
    "storm": ("#0d1117", "#4a5568", 0.1, "rain"),  # Dark with rain
    "cave": ("#050505", "#3d3d3d", 0.1, None),  # Very dark
    "beach": ("#87ceeb", "#f4a460", 0.05, None),  # Sky blue with sandy tint
    "mountain": ("#4682b4", "#696969", 0.05, None),  # Steel blue with gray
    "swamp": ("#1a2f1a", "#6b8e23", 0.12, None),  # Murky green
    "jungle": ("#002600", "#32cd32", 0.1, "leaves"),  # Deep green with leaves
    # Weather overlays
    "rain": ("#2d3748", "#4a5568", 0.08, "rain"),
    "snow": ("#e0f0ff", "#ffffff", 0.1, "snow"),
    "fog": ("#808080", "#c0c0c0", 0.2, None),
    "wind": ("#f0f8ff", None, 0.0, "leaves"),
    "sandstorm": ("#daa520", "#cd853f", 0.25, "sand"),
}


class RenderStyle:
    NORMAL = "normal"
    SKETCH = "sketch"
    INK = "ink"
    NEON = "neon"


class StickFigure:
    def __init__(
        self,
        actor: Actor,
        environment_type: EnvironmentType = EnvironmentType.EARTH_NORMAL,
        weather_type: EnvironmentType | None = None,
    ):
        self.id = actor.id
        self.actor_type = actor.actor_type
        self.color = actor.color
        self.initial_pos = np.array(
            [actor.initial_position.x, actor.initial_position.y]
        )
        self.pos = self.initial_pos.copy()  # Current position (will be updated)
        self.scale = actor.scale
        self.actions = actor.actions  # List of (time, action) tuples
        self.current_action = ActionType.IDLE
        self.frame_idx = 0

        # Spatial movement support
        self.velocity = (
            np.array(actor.velocity) if actor.velocity else np.array([0.0, 0.0])
        )
        self.movement_path = actor.movement_path  # List of (time, Position) waypoints
        self.last_update_time = 0.0

        # Facial expression support (Phase 5)
        self.facial_expression = actor.facial_expression
        self.face_features = (
            actor.face_features
            if actor.face_features
            else FaceFeatures(expression=actor.facial_expression)
        )

        # Expression transition support (Phase 5.2)
        self.target_expression = actor.facial_expression
        self.transition_start_time = 0.0
        self.transition_duration = 0.3  # 0.3 seconds for smooth transitions
        self.is_transitioning = False
        self.previous_features = None

        # Actor-type-specific motion profile
        self.motion_profile: MotionProfile = get_motion_profile(self.actor_type)
        self._last_quantized_t = 0.0  # For time quantization
        self._rng = np.random.default_rng(hash(actor.id) % (2**31))  # Deterministic noise

        # Environment physics profile (combines base environment + optional weather)
        base_env = get_environment_profile(environment_type)
        weather_env = get_environment_profile(weather_type) if weather_type else None
        self.environment_profile: EnvironmentProfile = combine_environment_profiles(
            base_env, weather_env
        )
        self.environment_type = environment_type
        self.weather_type = weather_type

    def _apply_time_quantization(self, t: float) -> float:
        """Quantize time for mechanical/stepped motion (robots)."""
        if self.motion_profile.time_quantization > 0:
            step = self.motion_profile.time_quantization
            return np.floor(t / step) * step
        return t

    def _apply_smoothness(self, value: float, t: float) -> float:
        """Apply smoothness modifier to oscillating values."""
        smoothness = self.motion_profile.smoothness
        if smoothness >= 1.0:
            return value  # Normal or extra smooth
        # Stepped motion: quantize the sinusoidal output
        # smoothness=0 means full quantization, smoothness=1 means no quantization
        if smoothness < 0.5:
            # Strong quantization for mechanical motion
            steps = 8  # Number of discrete positions
            quantized = np.round(value * steps) / steps
            # Blend between quantized and smooth based on smoothness
            blend = smoothness * 2  # 0-0.5 -> 0-1
            return quantized * (1 - blend) + value * blend
        return value

    def _get_joint_noise(self) -> float:
        """Get random noise for mechanical jitter."""
        if self.motion_profile.joint_noise > 0:
            return self._rng.uniform(
                -self.motion_profile.joint_noise,
                self.motion_profile.joint_noise
            )
        return 0.0

    # === Environment Physics Methods ===

    def _get_effective_speed(self, base_speed: float) -> float:
        """Apply environment modifiers to movement speed."""
        env = self.environment_profile
        # Stack actor and environment speed modifiers
        actor_freq = self.motion_profile.frequency_multiplier
        env_speed = env.movement_speed_scale
        # Friction affects speed (lower friction = harder to accelerate)
        friction_effect = 0.5 + 0.5 * env.friction  # 0.5-1.0 range
        # Air resistance slows movement
        air_effect = 1.0 / (1.0 + (env.air_resistance - 1.0) * 0.3)
        # Crowd density reduces speed
        crowd_effect = 1.0 - env.crowd_density * 0.4
        # Temperature affects energy
        temp_effect = env.temperature_modifier

        return base_speed * actor_freq * env_speed * friction_effect * air_effect * crowd_effect * temp_effect

    def _get_effective_step_height(self, base_height: float) -> float:
        """Apply environment modifiers to step/leg lift height."""
        env = self.environment_profile
        actor_amp = self.motion_profile.amplitude_modifier
        env_step = env.step_height_scale
        # Gravity affects how high legs lift
        gravity_effect = 1.0 + (1.0 - env.gravity_scale) * 0.3

        return base_height * actor_amp * env_step * gravity_effect

    def _get_effective_jump_height(self, base_height: float) -> float:
        """Apply environment modifiers to jump height."""
        env = self.environment_profile
        actor_amp = self.motion_profile.amplitude_modifier
        # Jump height inversely proportional to gravity
        gravity_effect = 1.0 / max(env.gravity_scale, 0.01)
        env_jump = env.jump_height_scale

        return base_height * actor_amp * gravity_effect * env_jump

    def _get_effective_gravity(self) -> float:
        """Get effective gravity for physics calculations."""
        return 9.8 * self.environment_profile.gravity_scale

    def _get_buoyancy_offset(self, t: float) -> float:
        """Get vertical offset from buoyancy (floating in water/space)."""
        env = self.environment_profile
        if env.buoyancy > 0:
            # Gentle bobbing motion in buoyant environments
            bob_amplitude = env.buoyancy * 0.2  # Max 0.2 units of bob
            bob_freq = 0.5  # Slow, gentle bob
            return bob_amplitude * np.sin(2 * np.pi * bob_freq * t)
        return 0.0

    def _get_wind_offset(self, t: float) -> tuple[float, float]:
        """Get lateral offset from wind."""
        env = self.environment_profile
        if env.wind_strength > 0:
            # Wind causes swaying and drift
            sway_amp = env.wind_strength * 0.15
            sway_freq = 0.8 + self._rng.uniform(-0.1, 0.1)
            sway = sway_amp * np.sin(2 * np.pi * sway_freq * t + env.wind_direction)
            # Wind direction affects x/y components
            x_offset = sway * np.cos(env.wind_direction)
            y_offset = sway * np.sin(env.wind_direction) * 0.3  # Less vertical effect
            return x_offset, y_offset
        return 0.0, 0.0

    def _get_balance_sway(self, t: float) -> float:
        """Get balance difficulty sway (slippery surfaces, unstable ground)."""
        env = self.environment_profile
        if env.balance_difficulty > 1.0:
            # Increased sway on difficult balance surfaces
            difficulty_factor = (env.balance_difficulty - 1.0) * 0.1
            sway_freq = 1.5 + self._rng.uniform(-0.2, 0.2)
            return difficulty_factor * np.sin(2 * np.pi * sway_freq * t)
        return 0.0

    def _get_friction_slide(self) -> float:
        """Get slide factor for low-friction surfaces (ice, wet)."""
        env = self.environment_profile
        if env.friction < 0.5:
            # Low friction = sliding motion
            return 1.0 - env.friction * 2  # 0.0-1.0 slide factor
        return 0.0

    def _apply_environment_to_pose(
        self, lines: list, head_center: np.ndarray, t: float
    ) -> tuple[list, np.ndarray]:
        """Apply environment physics to final pose (buoyancy, wind, balance)."""
        env = self.environment_profile

        # Get environmental offsets
        buoyancy_y = self._get_buoyancy_offset(t)
        wind_x, wind_y = self._get_wind_offset(t)
        balance_x = self._get_balance_sway(t)

        # Combine offsets
        total_x_offset = wind_x + balance_x
        total_y_offset = buoyancy_y + wind_y

        # Apply slope lean (lean into uphill)
        slope_lean = np.sin(env.slope_angle) * 0.2

        if abs(total_x_offset) > 0.001 or abs(total_y_offset) > 0.001 or abs(slope_lean) > 0.001:
            # Apply offsets to all line endpoints
            offset = np.array([total_x_offset + slope_lean, total_y_offset])
            adjusted_lines = []
            for start, end in lines:
                adjusted_lines.append((start + offset, end + offset))
            head_center = head_center + offset
            return adjusted_lines, head_center

        return lines, head_center

    def update_position(self, t: float, dt: float = 0.04, apply_physics: bool = True):
        """
        Update actor position based on movement path or action velocity

        Phase 2: Now includes physics constraints (gravity, ground collision)
        Phase 3: Environment physics modifiers (gravity, friction, buoyancy)

        Args:
            t: Current time in seconds
            dt: Time delta (1/fps, default 0.04 for 25fps)
            apply_physics: Whether to apply physics constraints (default: True)
        """
        env = self.environment_profile

        if self.movement_path and len(self.movement_path) > 0:
            # Use predefined movement path (waypoint interpolation)
            self.pos = self._interpolate_path(t)
        else:
            # Use action-based velocity
            current_action = self.get_current_action(t)
            base_velocity = ACTION_VELOCITIES.get(current_action, 0.0)

            # Apply environment speed modifiers
            velocity_magnitude = self._get_effective_speed(base_velocity)

            if velocity_magnitude > 0:
                # Move in the direction specified by self.velocity
                # If velocity is (0, 0), use default direction (right)
                if np.linalg.norm(self.velocity) > 0:
                    direction = self.velocity / np.linalg.norm(self.velocity)
                else:
                    direction = np.array([1.0, 0.0])  # Default: move right

                # Update position with environment-adjusted velocity
                self.pos += direction * velocity_magnitude * dt

                # Apply friction-based sliding (ice, wet surfaces)
                slide_factor = self._get_friction_slide()
                if slide_factor > 0 and hasattr(self, '_last_velocity'):
                    # Continue sliding in previous direction
                    slide_decay = 0.95  # Gradual slowdown
                    self._last_velocity *= slide_decay
                    self.pos += self._last_velocity * slide_factor * dt
                self._last_velocity = direction * velocity_magnitude

            # Phase 2/3: Apply physics constraints
            if apply_physics:
                # Apply gravity for jumping actions (with environment modifier)
                if current_action == ActionType.JUMP:
                    # Initialize physics velocity if not present
                    if not hasattr(self, "physics_velocity"):
                        self.physics_velocity = np.array([0.0, 0.0])

                    # Apply environment-scaled gravity
                    gravity = -self._get_effective_gravity()
                    self.physics_velocity[1] += gravity * dt

                    # Apply air resistance to physics velocity
                    air_drag = 1.0 - (env.air_resistance - 1.0) * 0.1 * dt
                    self.physics_velocity *= max(air_drag, 0.9)

                    # Apply buoyancy (upward force in water/space)
                    if env.buoyancy > 0:
                        buoyancy_force = env.buoyancy * 5.0  # Counteracts gravity
                        self.physics_velocity[1] += buoyancy_force * dt

                    # Update position based on physics velocity
                    self.pos += self.physics_velocity * dt

                # Apply ground constraint (prevent falling through ground)
                self.apply_ground_constraint(ground_level=0.0)

        self.last_update_time = t

    def detect_ground_collision(self, ground_level: float = 0.0) -> bool:
        """
        Phase 2: Detect if actor is colliding with ground

        Args:
            ground_level: Y-coordinate of ground (default: 0.0)

        Returns:
            True if feet are at or below ground level
        """
        # Feet position is at self.pos (hip level)
        # Legs extend down by ~1.0 * scale
        feet_y = self.pos[1] - 1.0 * self.scale
        return feet_y <= ground_level

    def detect_object_collision(self, obj_pos: np.ndarray, obj_radius: float) -> bool:
        """
        Phase 2: Detect collision with circular object

        Args:
            obj_pos: Object center position [x, y]
            obj_radius: Object collision radius

        Returns:
            True if actor's body overlaps with object
        """
        # Use actor's center (hip) for collision detection
        distance = np.linalg.norm(self.pos - obj_pos)
        actor_radius = 0.5 * self.scale  # Approximate actor as circle
        return distance < (actor_radius + obj_radius)

    def apply_ground_constraint(self, ground_level: float = 0.0):
        """
        Phase 2: Apply ground collision constraint

        Prevents actor from falling through ground by adjusting position

        Args:
            ground_level: Y-coordinate of ground (default: 0.0)
        """
        feet_y = self.pos[1] - 1.0 * self.scale
        if feet_y < ground_level:
            # Adjust position to keep feet on ground
            self.pos[1] = ground_level + 1.0 * self.scale
            # Zero out downward velocity if present
            if hasattr(self, "physics_velocity") and self.physics_velocity[1] < 0:
                self.physics_velocity[1] = 0.0

    def _interpolate_path(self, t: float) -> np.ndarray:
        """Interpolate position along movement path based on time"""
        if not self.movement_path or len(self.movement_path) == 0:
            return self.pos

        # Find the two waypoints to interpolate between
        if t <= self.movement_path[0][0]:
            # Before first waypoint
            pos = self.movement_path[0][1]
            return np.array([pos.x, pos.y])

        if t >= self.movement_path[-1][0]:
            # After last waypoint
            pos = self.movement_path[-1][1]
            return np.array([pos.x, pos.y])

        # Find surrounding waypoints
        for i in range(len(self.movement_path) - 1):
            t1, pos1 = self.movement_path[i]
            t2, pos2 = self.movement_path[i + 1]

            if t1 <= t <= t2:
                # Linear interpolation
                alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                x = pos1.x + alpha * (pos2.x - pos1.x)
                y = pos1.y + alpha * (pos2.y - pos1.y)
                return np.array([x, y])

        # Fallback
        return self.pos

    def get_current_action(self, t: float) -> ActionType:
        """Determine current action based on time"""
        current = ActionType.IDLE
        for start_time, action in self.actions:
            if t >= start_time:
                current = action
        return current

    def update_expression(self, t: float, new_expression: FacialExpression):
        """
        Update facial expression with smooth transition (Phase 5.2)

        Args:
            t: Current time
            new_expression: Target expression to transition to
        """

        if new_expression != self.facial_expression:
            # Start transition
            self.previous_features = self.face_features
            self.target_expression = new_expression
            self.transition_start_time = t
            self.is_transitioning = True
            self.facial_expression = new_expression

    def get_interpolated_features(self, t: float) -> FaceFeatures:
        """
        Get interpolated facial features during transition (Phase 5.2)

        Returns current facial features, interpolating during transitions.

        Args:
            t: Current time

        Returns:
            FaceFeatures with interpolated values
        """
        from .story_engine import EXPRESSION_FEATURES

        if not self.is_transitioning:
            return self.face_features

        # Calculate transition progress
        elapsed = t - self.transition_start_time
        progress = min(elapsed / self.transition_duration, 1.0)

        if progress >= 1.0:
            # Transition complete
            self.is_transitioning = False
            self.face_features = EXPRESSION_FEATURES[self.facial_expression]
            return self.face_features

        # Interpolate between previous and target features
        target_features = EXPRESSION_FEATURES[self.target_expression]

        if self.previous_features is None:
            return target_features

        # Linear interpolation for eyebrow angle
        eyebrow_angle = (
            self.previous_features.eyebrow_angle * (1 - progress)
            + target_features.eyebrow_angle * progress
        )

        # Linear interpolation for mouth openness
        mouth_openness = (
            self.previous_features.mouth_openness * (1 - progress)
            + target_features.mouth_openness * progress
        )

        # Transition eye type and mouth shape at 50% progress
        eye_type = (
            target_features.eye_type
            if progress > 0.5
            else self.previous_features.eye_type
        )
        mouth_shape = (
            target_features.mouth_shape
            if progress > 0.5
            else self.previous_features.mouth_shape
        )

        return FaceFeatures(
            expression=self.target_expression,
            eye_type=eye_type,
            eyebrow_angle=eyebrow_angle,
            mouth_shape=mouth_shape,
            mouth_openness=mouth_openness,
        )

    def get_pose(self, t: float, dt: float = 0.04) -> tuple[list, np.ndarray]:
        """
        Get pose based on current action and time, with actor-type-specific modifiers.

        Args:
            t: Current time in seconds
            dt: Time delta for position update
        """
        # Apply time quantization for mechanical motion (robots)
        anim_t = self._apply_time_quantization(t)

        # Apply frequency modifier for actor-type-specific timing
        anim_t *= self.motion_profile.frequency_multiplier

        # Update position based on movement
        self.update_position(t, dt)

        self.current_action = self.get_current_action(t)

        # Apply posture modifiers from motion profile
        mp = self.motion_profile

        # Crouch factor lowers the stance
        crouch_offset = mp.crouch_factor * 0.5 * self.scale

        # Forward lean tilts the body
        lean_x = np.sin(mp.forward_lean) * 0.3 * self.scale
        lean_y = (1 - np.cos(mp.forward_lean)) * 0.2 * self.scale

        # Base skeleton positions with motion profile modifiers
        torso_height = 1.0 * mp.torso_scale
        head_height = 1.5 * mp.torso_scale

        hip = self.pos - np.array([0, crouch_offset])
        neck = hip + np.array([lean_x, torso_height - lean_y]) * self.scale
        head_center = hip + np.array([lean_x, head_height - lean_y]) * self.scale

        # Action-specific animations (pass modified time for actor-specific rhythms)
        if self.current_action == ActionType.WALK:
            lines, head = self._animate_walk(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.RUN:
            lines, head = self._animate_run(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.SPRINT:
            lines, head = self._animate_sprint(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.JUMP:
            lines, head = self._animate_jump(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.WAVE:
            lines, head = self._animate_wave(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.BATTING:
            lines, head = self._animate_batting(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.PITCHING:
            lines, head = self._animate_pitching(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.CATCHING:
            lines, head = self._animate_catching(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.RUNNING_BASES:
            lines, head = self._animate_run(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.FIELDING:
            lines, head = self._animate_fielding(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.THROWING:
            lines, head = self._animate_throwing(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.KICKING:
            lines, head = self._animate_kicking(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.SIT:
            lines, head = self._animate_sit(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.EATING:
            lines, head = self._animate_eating(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.TALK:
            lines, head = self._animate_talk(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.LOOKING_AROUND:
            lines, head = self._animate_looking_around(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.FIGHT:
            lines, head = self._animate_fight(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.DANCE:
            lines, head = self._animate_dance(anim_t, neck, hip, head_center)
        elif self.current_action == ActionType.TYPING:
            lines, head = self._animate_typing(anim_t, neck, hip, head_center)
        else:  # IDLE and others
            lines, head = self._animate_idle(anim_t, neck, hip, head_center)

        # Apply post-processing motion profile modifiers (actor-specific)
        lines = self._apply_motion_profile_to_pose(lines, hip)

        # Apply environment physics effects (buoyancy, wind, balance, slope)
        lines, head = self._apply_environment_to_pose(lines, head, t)

        return lines, head

    def _apply_motion_profile_to_pose(
        self, lines: list, hip: np.ndarray
    ) -> list:
        """
        Apply actor-type-specific motion modifiers to the pose.

        This post-processes the animation output to add:
        - Limb length scaling
        - Stance width adjustment
        - Joint noise (mechanical jitter)
        - Amplitude modification
        """
        mp = self.motion_profile

        # Skip if all modifiers are at baseline
        if (mp.limb_length_scale == 1.0 and mp.stance_width == 1.0 and
            mp.amplitude_modifier == 1.0 and mp.joint_noise == 0.0):
            return lines

        modified_lines = []

        for i, (start, end) in enumerate(lines):
            new_start = start.copy()
            new_end = end.copy()

            # Lines 1-2 are legs (hip to feet), lines 3-4 are arms
            is_leg = (i == 1 or i == 2)
            is_arm = (i == 3 or i == 4)

            if is_leg or is_arm:
                # Calculate limb vector from hip/neck
                limb_vec = end - start
                limb_length = np.linalg.norm(limb_vec)

                if limb_length > 0:
                    # Apply limb length scaling
                    limb_dir = limb_vec / limb_length
                    new_length = limb_length * mp.limb_length_scale
                    new_end = start + limb_dir * new_length

                    # Apply amplitude modifier (affects how far limbs swing from rest)
                    rest_end = start + limb_dir * new_length
                    swing = new_end - rest_end
                    new_end = rest_end + swing * mp.amplitude_modifier

            # Apply stance width for legs
            if is_leg:
                # Adjust horizontal spread
                center_x = hip[0]
                new_end[0] = center_x + (new_end[0] - center_x) * mp.stance_width

            # Apply joint noise
            if mp.joint_noise > 0:
                noise = np.array([self._get_joint_noise(), self._get_joint_noise()])
                new_end = new_end + noise

            modified_lines.append((new_start, new_end))

        return modified_lines

    def _animate_idle(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Idle animation - slight breathing motion with actor-type modifiers"""
        mp = self.motion_profile

        # Breathing amplitude affected by sway_amplitude (robots breathe less)
        breathe = np.sin(t * 2) * 0.05 * mp.sway_amplitude

        # Apply smoothness (mechanical actors have stepped breathing)
        breathe = self._apply_smoothness(breathe, t)

        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale
        left_hand = neck + np.array([-0.6, -0.8 + breathe]) * self.scale
        right_hand = neck + np.array([0.6, -0.8 + breathe]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_walk(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Walking animation with actor-type modifiers"""
        mp = self.motion_profile

        # Base swing values
        leg_swing_raw = np.sin(t * 8) * 0.4 * mp.amplitude_modifier
        arm_swing_raw = np.sin(t * 8 + mp.phase_offset) * 0.3 * mp.amplitude_modifier

        # Apply smoothness modifier (robots have stepped motion)
        leg_swing = self._apply_smoothness(leg_swing_raw, t)
        arm_swing = self._apply_smoothness(arm_swing_raw, t)

        left_foot = hip + np.array([-0.3 + leg_swing, -1.5]) * self.scale
        right_foot = hip + np.array([0.3 - leg_swing, -1.5]) * self.scale
        left_hand = neck + np.array([-0.5 - arm_swing, -0.6]) * self.scale
        right_hand = neck + np.array([0.5 + arm_swing, -0.6]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_run(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Running animation with actor-type modifiers"""
        mp = self.motion_profile

        # Base swing values with amplitude modifier
        leg_swing_raw = np.sin(t * 15) * 0.7 * mp.amplitude_modifier
        arm_swing_raw = np.sin(t * 15 + mp.phase_offset) * 0.5 * mp.amplitude_modifier
        bob_raw = abs(np.sin(t * 15)) * 0.1 * mp.sway_amplitude

        # Apply smoothness modifier
        leg_swing = self._apply_smoothness(leg_swing_raw, t)
        arm_swing = self._apply_smoothness(arm_swing_raw, t)
        bob = self._apply_smoothness(bob_raw, t)

        # Adjust body position for running
        hip_adjusted = hip + np.array([0, bob])
        neck_adjusted = neck + np.array([0, bob])
        head_adjusted = head_center + np.array([0, bob])

        left_foot = hip_adjusted + np.array([-0.3 + leg_swing, -1.5]) * self.scale
        right_foot = hip_adjusted + np.array([0.3 - leg_swing, -1.5]) * self.scale
        left_hand = neck_adjusted + np.array([-0.6 - arm_swing, -0.4]) * self.scale
        right_hand = neck_adjusted + np.array([0.6 + arm_swing, -0.4]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_sprint(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Sprinting animation - even faster and more aggressive than running"""
        # Much faster frequency (20 vs 15 for run)
        leg_swing = np.sin(t * 20) * 0.9  # More exaggerated leg movement
        arm_swing = np.sin(t * 20) * 0.7  # More exaggerated arm movement
        bob = abs(np.sin(t * 20)) * 0.15  # More vertical movement

        # Forward lean for sprinting
        lean = 0.2

        # Adjust body position for sprinting
        hip_adjusted = hip + np.array([lean, bob])
        neck_adjusted = neck + np.array([lean, bob])
        head_adjusted = head_center + np.array([lean, bob])

        left_foot = hip_adjusted + np.array([-0.4 + leg_swing, -1.5]) * self.scale
        right_foot = hip_adjusted + np.array([0.4 - leg_swing, -1.5]) * self.scale
        left_hand = neck_adjusted + np.array([-0.8 - arm_swing, -0.3]) * self.scale
        right_hand = neck_adjusted + np.array([0.8 + arm_swing, -0.3]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_jump(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Jumping animation with realistic physics (3-phase: anticipation → flight → landing)"""
        jump_cycle = 1.0  # 1 second per jump
        jump_progress = (t % jump_cycle) / jump_cycle

        # Phase 1: Anticipation (crouch) - 0 to 0.2
        if jump_progress < 0.2:
            phase = jump_progress / 0.2
            crouch = phase * 0.3  # Crouch down
            jump_height = 0
            arm_height = -0.3 - crouch * 0.5  # Arms down during crouch
            leg_bend = crouch * 0.3
        # Phase 2: Flight (parabolic trajectory) - 0.2 to 0.7
        elif jump_progress < 0.7:
            phase = (jump_progress - 0.2) / 0.5
            # Parabolic trajectory: y = -4(x-0.5)^2 + 1
            jump_height = -4 * (phase - 0.5) ** 2 + 1.0
            jump_height *= 0.8  # Scale to reasonable height
            crouch = 0
            arm_height = 0.5  # Arms up during flight
            leg_bend = 0
        # Phase 3: Landing - 0.7 to 1.0
        else:
            phase = (jump_progress - 0.7) / 0.3
            jump_height = (1 - phase) * 0.2  # Small bounce on landing
            crouch = phase * 0.2  # Slight crouch on landing
            arm_height = 0.5 - phase * 0.8  # Arms come down
            leg_bend = phase * 0.2

        # Adjust body position
        hip_adjusted = hip + np.array([0, jump_height - crouch]) * self.scale
        neck_adjusted = neck + np.array([0, jump_height - crouch]) * self.scale
        head_adjusted = head_center + np.array([0, jump_height - crouch]) * self.scale

        # Legs - bent during crouch/landing, straight during flight
        left_foot = hip_adjusted + np.array([-0.3, -1.5 + leg_bend]) * self.scale
        right_foot = hip_adjusted + np.array([0.3, -1.5 + leg_bend]) * self.scale

        # Arms - down during crouch, up during flight
        left_hand = neck_adjusted + np.array([-0.5, arm_height]) * self.scale
        right_hand = neck_adjusted + np.array([0.5, arm_height]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_wave(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Waving animation"""
        wave = np.sin(t * 10) * 0.3
        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale
        left_hand = neck + np.array([-0.6, -0.8]) * self.scale
        right_hand = (
            neck + np.array([0.8, 0.5 + wave]) * self.scale
        )  # Right hand waving

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_batting(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Batting animation - realistic swing timing (0.4s cycle)"""
        swing_cycle = 0.4  # 400ms total swing (realistic baseball swing)
        swing_progress = (t % swing_cycle) / swing_cycle

        # Multi-phase swing: load (0-0.2) → stride (0.2-0.4) → swing (0.4-0.8) → follow-through (0.8-1.0)
        if swing_progress < 0.2:
            # Load phase - weight back
            angle = -0.3
            hip_shift = -0.1
        elif swing_progress < 0.4:
            # Stride phase - step forward
            angle = -0.2
            hip_shift = 0.0
        elif swing_progress < 0.8:
            # Swing phase - rapid rotation
            phase = (swing_progress - 0.4) / 0.4
            angle = -0.2 + phase * np.pi * 1.2  # Fast swing
            hip_shift = 0.1
        else:
            # Follow-through
            angle = np.pi
            hip_shift = 0.2

        hip_adjusted = hip + np.array([hip_shift, 0]) * self.scale

        left_foot = hip_adjusted + np.array([-0.5, -1.5]) * self.scale
        right_foot = hip_adjusted + np.array([0.5, -1.5]) * self.scale

        # Bat swing motion with realistic arc
        left_hand = (
            neck
            + np.array([-0.3 + np.cos(angle) * 0.8, -0.2 + np.sin(angle) * 0.5])
            * self.scale
        )
        right_hand = (
            neck
            + np.array([0.3 + np.cos(angle) * 1.0, -0.4 + np.sin(angle) * 0.7])
            * self.scale
        )

        lines = [
            (neck, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck, left_hand),
            (neck, right_hand),
            (left_hand, right_hand),  # Bat
        ]
        return lines, head_center

    def _animate_pitching(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Pitching animation"""
        pitch_phase = (t % 2.0) / 2.0

        left_foot = hip + np.array([-0.4, -1.5]) * self.scale
        right_foot = hip + np.array([0.6, -1.3]) * self.scale
        left_hand = neck + np.array([-0.5, -0.5]) * self.scale

        # Throwing arm motion
        if pitch_phase < 0.5:
            # Wind up
            right_hand = neck + np.array([0.3, 0.8]) * self.scale
        else:
            # Release
            right_hand = neck + np.array([0.8, -0.3]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_catching(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Catching animation"""
        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale

        # Arms up ready to catch
        left_hand = neck + np.array([-0.4, 0.5]) * self.scale
        right_hand = neck + np.array([0.4, 0.5]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_fielding(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Fielding animation - crouched position"""
        crouch = 0.3
        hip_adjusted = hip + np.array([0, -crouch])
        neck_adjusted = neck + np.array([0, -crouch])
        head_adjusted = head_center + np.array([0, -crouch])

        left_foot = hip_adjusted + np.array([-0.5, -1.3]) * self.scale
        right_foot = hip_adjusted + np.array([0.5, -1.3]) * self.scale
        left_hand = neck_adjusted + np.array([-0.5, -0.8]) * self.scale
        right_hand = neck_adjusted + np.array([0.5, -0.8]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_throwing(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Throwing animation"""
        throw_phase = (t % 1.5) / 1.5

        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale
        left_hand = neck + np.array([-0.5, -0.5]) * self.scale

        if throw_phase < 0.4:
            right_hand = neck + np.array([0.2, 0.6]) * self.scale
        else:
            right_hand = neck + np.array([0.9, 0.2]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_kicking(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Kicking animation"""
        kick_phase = (t % 1.5) / 1.5

        left_foot = hip + np.array([-0.3, -1.5]) * self.scale

        if kick_phase < 0.5:
            right_foot = hip + np.array([0.3, -1.5]) * self.scale
        else:
            right_foot = hip + np.array([0.8, -1.0]) * self.scale  # Kicking leg

        left_hand = neck + np.array([-0.5, -0.5]) * self.scale
        right_hand = neck + np.array([0.5, -0.5]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_sit(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Sitting animation"""
        sit_offset = 0.5
        hip_adjusted = hip + np.array([0, -sit_offset])
        neck_adjusted = neck + np.array([0, -sit_offset])
        head_adjusted = head_center + np.array([0, -sit_offset])

        # Legs bent at 90 degrees
        left_foot = hip_adjusted + np.array([-0.8, -0.8]) * self.scale
        right_foot = hip_adjusted + np.array([0.8, -0.8]) * self.scale
        left_knee = hip_adjusted + np.array([-0.4, -0.4]) * self.scale
        right_knee = hip_adjusted + np.array([0.4, -0.4]) * self.scale

        left_hand = neck_adjusted + np.array([-0.5, -0.6]) * self.scale
        right_hand = neck_adjusted + np.array([0.5, -0.6]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_knee),
            (left_knee, left_foot),
            (hip_adjusted, right_knee),
            (right_knee, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_eating(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Eating animation - sitting with hand to mouth, includes pauses for chewing"""
        sit_offset = 0.5
        hip_adjusted = hip + np.array([0, -sit_offset])
        neck_adjusted = neck + np.array([0, -sit_offset])
        head_adjusted = head_center + np.array([0, -sit_offset])

        left_foot = hip_adjusted + np.array([-0.8, -0.8]) * self.scale
        right_foot = hip_adjusted + np.array([0.8, -0.8]) * self.scale
        left_knee = hip_adjusted + np.array([-0.4, -0.4]) * self.scale
        right_knee = hip_adjusted + np.array([0.4, -0.4]) * self.scale

        # Eating cycle with pauses: reach (0-0.3) → to mouth (0.3-0.5) → chew (0.5-0.8) → lower (0.8-1.0)
        eat_cycle = 2.0  # 2 seconds per eating cycle
        eat_progress = (t % eat_cycle) / eat_cycle

        if eat_progress < 0.3:
            # Reaching for food
            phase = eat_progress / 0.3
            hand_y = -0.6 + phase * 0.3
        elif eat_progress < 0.5:
            # Bringing to mouth
            phase = (eat_progress - 0.3) / 0.2
            hand_y = -0.3 + phase * 0.8
        elif eat_progress < 0.8:
            # Chewing (hold at mouth with small variation)
            chew = np.sin(t * 15) * 0.05  # Small chewing motion
            hand_y = 0.5 + chew
        else:
            # Lowering hand
            phase = (eat_progress - 0.8) / 0.2
            hand_y = 0.5 - phase * 1.1

        left_hand = neck_adjusted + np.array([-0.5, -0.6]) * self.scale
        right_hand = neck_adjusted + np.array([0.2, hand_y]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_knee),
            (left_knee, left_foot),
            (hip_adjusted, right_knee),
            (right_knee, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_talk(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Talking animation - gesturing"""
        gesture = np.sin(t * 4) * 0.2

        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale
        left_hand = neck + np.array([-0.6, -0.3 + gesture]) * self.scale
        right_hand = neck + np.array([0.6, -0.3 - gesture]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_looking_around(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Looking around animation - head turns"""
        look = np.sin(t * 3) * 0.3
        head_adjusted = head_center + np.array([look, 0])

        left_foot = hip + np.array([-0.3, -1.5]) * self.scale
        right_foot = hip + np.array([0.3, -1.5]) * self.scale
        left_hand = neck + np.array([-0.5, -0.7]) * self.scale
        right_hand = neck + np.array([0.5, -0.7]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_adjusted

    def _animate_fight(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Fighting animation - punching"""
        punch_cycle = (t % 1.0) / 1.0

        left_foot = hip + np.array([-0.4, -1.5]) * self.scale
        right_foot = hip + np.array([0.4, -1.5]) * self.scale

        if punch_cycle < 0.5:
            left_hand = neck + np.array([-0.9, 0.2]) * self.scale
            right_hand = neck + np.array([0.5, -0.5]) * self.scale
        else:
            left_hand = neck + np.array([-0.5, -0.5]) * self.scale
            right_hand = neck + np.array([0.9, 0.2]) * self.scale

        lines = [
            (neck, hip),
            (hip, left_foot),
            (hip, right_foot),
            (neck, left_hand),
            (neck, right_hand),
        ]
        return lines, head_center

    def _animate_dance(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Dancing animation"""
        dance_x = np.sin(t * 5) * 0.3
        dance_y = abs(np.sin(t * 5)) * 0.2

        hip_adjusted = hip + np.array([dance_x, dance_y])
        neck_adjusted = neck + np.array([dance_x, dance_y])
        head_adjusted = head_center + np.array([dance_x, dance_y])

        left_foot = hip_adjusted + np.array([-0.4, -1.5]) * self.scale
        right_foot = hip_adjusted + np.array([0.4, -1.5]) * self.scale
        left_hand = (
            neck_adjusted + np.array([-0.7, 0.3 + np.sin(t * 5) * 0.3]) * self.scale
        )
        right_hand = (
            neck_adjusted + np.array([0.7, 0.3 - np.sin(t * 5) * 0.3]) * self.scale
        )

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_foot),
            (hip_adjusted, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _animate_typing(
        self, t: float, neck: np.ndarray, hip: np.ndarray, head_center: np.ndarray
    ):
        """Typing animation - sitting with hands down"""
        sit_offset = 0.5
        hip_adjusted = hip + np.array([0, -sit_offset])
        neck_adjusted = neck + np.array([0, -sit_offset])
        head_adjusted = head_center + np.array([0, -sit_offset])

        left_foot = hip_adjusted + np.array([-0.8, -0.8]) * self.scale
        right_foot = hip_adjusted + np.array([0.8, -0.8]) * self.scale
        left_knee = hip_adjusted + np.array([-0.4, -0.4]) * self.scale
        right_knee = hip_adjusted + np.array([0.4, -0.4]) * self.scale

        # Hands typing motion
        type_motion = np.sin(t * 10) * 0.05
        left_hand = neck_adjusted + np.array([-0.3, -0.9 + type_motion]) * self.scale
        right_hand = neck_adjusted + np.array([0.3, -0.9 - type_motion]) * self.scale

        lines = [
            (neck_adjusted, hip_adjusted),
            (hip_adjusted, left_knee),
            (left_knee, left_foot),
            (hip_adjusted, right_knee),
            (right_knee, right_foot),
            (neck_adjusted, left_hand),
            (neck_adjusted, right_hand),
        ]
        return lines, head_adjusted

    def _draw_face(self, ax, head_pos: np.ndarray, t: float):
        """
        Draw facial features on the stick figure head.

        Args:
            ax: Matplotlib axes to draw on
            head_pos: Position of the head center [x, y]
            t: Current time (for animation and transitions)
        """
        # Get interpolated features for smooth transitions (Phase 5.2)
        current_features = self.get_interpolated_features(t)

        # Head size based on scale
        head_radius = 0.25 * self.scale

        # Draw eyes
        self._draw_eyes(ax, head_pos, head_radius, current_features)

        # Draw eyebrows
        self._draw_eyebrows(ax, head_pos, head_radius, current_features)

        # Draw mouth
        self._draw_mouth(ax, head_pos, head_radius, t, current_features)

    def _draw_eyes(
        self, ax, head_pos: np.ndarray, head_radius: float, features: FaceFeatures
    ):
        """Draw eyes based on expression (Phase 5.2: uses interpolated features)"""
        eye_y_offset = 0.08 * self.scale
        eye_x_offset = 0.12 * self.scale
        eye_size = 0.04 * self.scale

        left_eye_pos = head_pos + np.array([-eye_x_offset, eye_y_offset])
        right_eye_pos = head_pos + np.array([eye_x_offset, eye_y_offset])

        eye_type = features.eye_type

        if eye_type == "dots":
            # Simple dots for neutral/happy
            ax.plot(
                left_eye_pos[0],
                left_eye_pos[1],
                "o",
                color=self.color,
                markersize=eye_size * 20,
            )
            ax.plot(
                right_eye_pos[0],
                right_eye_pos[1],
                "o",
                color=self.color,
                markersize=eye_size * 20,
            )

        elif eye_type == "curves":
            # Curved eyes for happy/excited
            theta = np.linspace(0, np.pi, 10)
            curve_width = eye_size * 1.5
            curve_height = eye_size * 0.8

            left_curve_x = left_eye_pos[0] + curve_width * np.cos(theta)
            left_curve_y = left_eye_pos[1] + curve_height * np.sin(theta)
            ax.plot(left_curve_x, left_curve_y, color=self.color, linewidth=1.5)

            right_curve_x = right_eye_pos[0] + curve_width * np.cos(theta)
            right_curve_y = right_eye_pos[1] + curve_height * np.sin(theta)
            ax.plot(right_curve_x, right_curve_y, color=self.color, linewidth=1.5)

        elif eye_type == "wide":
            # Wide eyes for surprised
            ax.plot(
                left_eye_pos[0],
                left_eye_pos[1],
                "o",
                color=self.color,
                markersize=eye_size * 30,
            )
            ax.plot(
                right_eye_pos[0],
                right_eye_pos[1],
                "o",
                color=self.color,
                markersize=eye_size * 30,
            )

        elif eye_type == "closed":
            # Closed eyes (horizontal lines)
            line_width = eye_size * 1.5
            ax.plot(
                [left_eye_pos[0] - line_width, left_eye_pos[0] + line_width],
                [left_eye_pos[1], left_eye_pos[1]],
                color=self.color,
                linewidth=1.5,
            )
            ax.plot(
                [right_eye_pos[0] - line_width, right_eye_pos[0] + line_width],
                [right_eye_pos[1], right_eye_pos[1]],
                color=self.color,
                linewidth=1.5,
            )

    def _draw_eyebrows(
        self, ax, head_pos: np.ndarray, head_radius: float, features: FaceFeatures
    ):
        """Draw eyebrows based on expression (Phase 5.2: uses interpolated features)"""
        eyebrow_y_offset = 0.18 * self.scale
        eyebrow_x_offset = 0.12 * self.scale
        eyebrow_width = 0.15 * self.scale

        # Convert angle from degrees to radians (from interpolated features)
        angle_rad = np.radians(features.eyebrow_angle)

        # Left eyebrow
        left_center = head_pos + np.array([-eyebrow_x_offset, eyebrow_y_offset])
        left_start = left_center + np.array(
            [-eyebrow_width, -eyebrow_width * np.tan(angle_rad)]
        )
        left_end = left_center + np.array(
            [eyebrow_width, eyebrow_width * np.tan(angle_rad)]
        )
        ax.plot(
            [left_start[0], left_end[0]],
            [left_start[1], left_end[1]],
            color=self.color,
            linewidth=1.5,
        )

        # Right eyebrow (mirrored)
        right_center = head_pos + np.array([eyebrow_x_offset, eyebrow_y_offset])
        right_start = right_center + np.array(
            [-eyebrow_width, eyebrow_width * np.tan(angle_rad)]
        )
        right_end = right_center + np.array(
            [eyebrow_width, -eyebrow_width * np.tan(angle_rad)]
        )
        ax.plot(
            [right_start[0], right_end[0]],
            [right_start[1], right_end[1]],
            color=self.color,
            linewidth=1.5,
        )

    def _draw_mouth(
        self,
        ax,
        head_pos: np.ndarray,
        head_radius: float,
        t: float,
        features: FaceFeatures,
    ):
        """
        Draw mouth based on expression and speech animation.

        Phase 7: Enhanced speech animation with cyclic mouth movements
        - Uses features.is_speaking to enable speech animation
        - Uses features.speech_cycle_speed for animation speed
        - Cycles through viseme shapes for realistic speech
        """
        mouth_y_offset = -0.12 * self.scale
        mouth_width = 0.18 * self.scale

        mouth_shape = features.mouth_shape
        mouth_center = head_pos + np.array([0, mouth_y_offset])

        # Phase 7: Speech animation with cyclic mouth movements
        if features.is_speaking:
            # Calculate speech cycle phase (0 to 1)
            cycle_phase = (t * features.speech_cycle_speed) % 1.0

            # Determine mouth openness based on sine wave
            # Creates smooth open-close cycles
            openness_min, openness_max = 0.2, 0.6  # Default range
            if hasattr(features, "mouth_openness"):
                # Use custom openness if specified
                openness_min = max(0.1, features.mouth_openness - 0.2)
                openness_max = min(1.0, features.mouth_openness + 0.2)

            # Sine wave for smooth cycling
            openness = openness_min + (openness_max - openness_min) * (
                0.5 + 0.5 * np.sin(2 * np.pi * cycle_phase)
            )

            # Draw animated mouth based on shape
            if mouth_shape in [MouthShape.SMALL_O, MouthShape.OPEN]:
                # Circular mouth for talking/whispering
                circle_radius = mouth_width * openness
                circle = patches.Circle(
                    mouth_center,
                    circle_radius,
                    fill=False,
                    edgecolor=self.color,
                    linewidth=1.5,
                )
                ax.add_patch(circle)
            elif mouth_shape == MouthShape.WIDE_OPEN:
                # Large circular mouth for shouting
                circle_radius = mouth_width * (0.6 + 0.3 * openness)
                circle = patches.Circle(
                    mouth_center,
                    circle_radius,
                    fill=False,
                    edgecolor=self.color,
                    linewidth=1.5,
                )
                ax.add_patch(circle)
            elif mouth_shape == MouthShape.SINGING:
                # Elliptical mouth for singing
                ellipse = patches.Ellipse(
                    mouth_center,
                    mouth_width * 1.2,
                    mouth_width * openness,
                    fill=False,
                    edgecolor=self.color,
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)
            else:
                # Fallback to simple animated circle
                circle_radius = mouth_width * openness
                circle = patches.Circle(
                    mouth_center,
                    circle_radius,
                    fill=False,
                    edgecolor=self.color,
                    linewidth=1.5,
                )
                ax.add_patch(circle)

        # Non-speaking mouth shapes (static)
        elif mouth_shape == MouthShape.CLOSED:
            # Simple horizontal line
            ax.plot(
                [mouth_center[0] - mouth_width, mouth_center[0] + mouth_width],
                [mouth_center[1], mouth_center[1]],
                color=self.color,
                linewidth=1.5,
            )

        elif mouth_shape == MouthShape.SMILE:
            # Upward curve (smile)
            self._draw_curved_mouth(ax, mouth_center, mouth_width, curve_direction=1)

        elif mouth_shape == MouthShape.FROWN:
            # Downward curve (frown)
            self._draw_curved_mouth(ax, mouth_center, mouth_width, curve_direction=-1)

        elif mouth_shape == MouthShape.OPEN:
            # Small circle (surprised)
            circle_radius = mouth_width * 0.4
            circle = patches.Circle(
                mouth_center,
                circle_radius,
                fill=False,
                edgecolor=self.color,
                linewidth=1.5,
            )
            ax.add_patch(circle)

        elif mouth_shape == MouthShape.WIDE_OPEN:
            # Large circle (shouting - static)
            circle_radius = mouth_width * 0.6
            circle = patches.Circle(
                mouth_center,
                circle_radius,
                fill=False,
                edgecolor=self.color,
                linewidth=1.5,
            )
            ax.add_patch(circle)

        elif mouth_shape == MouthShape.SMALL_O:
            # Small oval (static)
            circle_radius = mouth_width * 0.3
            circle = patches.Circle(
                mouth_center,
                circle_radius,
                fill=False,
                edgecolor=self.color,
                linewidth=1.5,
            )
            ax.add_patch(circle)

        elif mouth_shape == MouthShape.SINGING:
            # Wide oval (static)
            ellipse = patches.Ellipse(
                mouth_center,
                mouth_width * 1.2,
                mouth_width * 0.5,
                fill=False,
                edgecolor=self.color,
                linewidth=1.5,
            )
            ax.add_patch(ellipse)

    def _draw_curved_mouth(
        self, ax, mouth_center: np.ndarray, mouth_width: float, curve_direction: int
    ):
        """
        Helper to draw curved mouth (smile or frown)

        Args:
            ax: Matplotlib axes
            mouth_center: Center position of mouth
            mouth_width: Width of mouth
            curve_direction: 1 for smile (upward), -1 for frown (downward)
        """
        # Create a curved line using a quadratic bezier-like curve
        x_points = np.linspace(-mouth_width, mouth_width, 20)
        curve_height = 0.08 * self.scale * curve_direction
        y_points = -curve_height * (1 - (x_points / mouth_width) ** 2)

        curve_x = mouth_center[0] + x_points
        curve_y = mouth_center[1] + y_points

        ax.plot(curve_x, curve_y, color=self.color, linewidth=1.5)


class CinematicRenderer(StickFigure):
    """
    Advanced renderer with '2.5D' features:
    - Perspective projection (foreshortening)
    - Dynamic line width (depth cue)
    - Z-sorting (occlusion)
    """

    def __init__(
        self,
        actor: Actor,
        environment_type: EnvironmentType = EnvironmentType.EARTH_NORMAL,
        weather_type: EnvironmentType | None = None,
    ):
        super().__init__(actor, environment_type, weather_type)

    def project_3d_to_2d(self, x, y, z, camera_zoom=1.0):
        """
        Project 3D point to 2D with perspective.
        Simple weak perspective: x' = x * (f / (f + z))
        """
        focal_length = 10.0  # Arbitrary focal length
        # Avoid division by zero
        depth = max(focal_length + z, 0.1)
        scale = (focal_length / depth) * camera_zoom

        x_proj = x * scale
        y_proj = y * scale

        return x_proj, y_proj, scale

    def get_pose(self, t: float, dt: float = 0.04):
        """
        Get pose with perspective projection applied.
        Intercepts base 2D pose, adds Z-depth heuristics, and projects.
        """
        # Get base 2D lines from parent
        base_lines, head_center = super().get_pose(t, dt)

        # Define Z-depths for limbs (assuming standard order)
        # 0: Torso, 1: L-Leg, 2: R-Leg, 3: L-Arm, 4: R-Arm
        # Positive Z is closer to camera
        z_depths = [0.0, -0.2, 0.2, -0.3, 0.3]

        # Extend z_depths if there are extra lines (e.g. props)
        if len(base_lines) > len(z_depths):
            z_depths.extend([0.1] * (len(base_lines) - len(z_depths)))

        cinematic_lines = []

        # Camera parameters (could be passed in or stored)
        # For now, assume static camera at Z=-10 looking at origin
        camera_z = -10.0

        for i, (start, end) in enumerate(base_lines):
            z = z_depths[i]

            # Project start point
            # Relative to camera: point_z - camera_z
            # If point is at z=0, dist=10. If z=2, dist=8 (closer)

            # Simple projection: scale = focal / distance
            # distance = point.z - camera.z
            dist = z - camera_z
            scale = 10.0 / dist

            # Apply projection (assuming camera centered at 0,0)
            start_proj = start * scale
            end_proj = end * scale

            # Calculate line width based on depth
            # Base width is 2. Closer = thicker.
            width = 2.0 * scale

            cinematic_lines.append((start_proj, end_proj, width, z))

        # Project head
        head_dist = 0.0 - camera_z  # Head at Z=0 roughly
        head_scale = 10.0 / head_dist
        head_proj = head_center * head_scale

        # Sort lines by Z (painters algorithm)
        # Draw furthest first (lowest Z)
        cinematic_lines.sort(key=lambda x: x[3])

        return cinematic_lines, head_proj


class Renderer:
    def __init__(self, width=640, height=480, style: str = RenderStyle.NORMAL):
        self.width = width
        self.height = height
        self.style = style
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Initialize Camera
        self.camera = Camera(width=10.0, height=10.0)

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Set background based on style
        if self.style == RenderStyle.NEON:
            self.fig.patch.set_facecolor("black")
            self.ax.set_facecolor("black")

    def _apply_style(self, artist):
        """Apply artistic style to a matplotlib artist"""
        if self.style == RenderStyle.SKETCH:
            artist.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="gray", alpha=0.5),
                    path_effects.Normal(),
                ]
            )
        elif self.style == RenderStyle.INK:
            artist.set_linewidth(artist.get_linewidth() * 1.5)
            artist.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=artist.get_linewidth() * 1.2,
                        foreground=artist.get_color(),
                        alpha=0.8,
                    ),
                    path_effects.Normal(),
                ]
            )
        elif self.style == RenderStyle.NEON:
            # Glow effect
            color = artist.get_color()
            artist.set_path_effects(
                [
                    path_effects.Stroke(linewidth=5, foreground=color, alpha=0.1),
                    path_effects.Stroke(linewidth=3, foreground=color, alpha=0.3),
                    path_effects.Normal(),
                ]
            )

    def _get_environment_visuals(self, env_type: EnvironmentType, weather_type=None):
        """Get visual configuration for environment type."""
        env_name = env_type.value if hasattr(env_type, 'value') else str(env_type)
        config = ENVIRONMENT_VISUAL_CONFIGS.get(env_name, ENVIRONMENT_VISUAL_CONFIGS["earth_normal"])
        bg_color, tint_color, tint_alpha, particle_type = config

        # Override with weather if specified
        if weather_type:
            weather_name = weather_type.value if hasattr(weather_type, 'value') else str(weather_type)
            if weather_name in ENVIRONMENT_VISUAL_CONFIGS:
                _, w_tint, w_alpha, w_particles = ENVIRONMENT_VISUAL_CONFIGS[weather_name]
                if w_tint:
                    tint_color = w_tint
                    tint_alpha = max(tint_alpha, w_alpha)
                if w_particles:
                    particle_type = w_particles

        return bg_color, tint_color, tint_alpha, particle_type

    def _draw_environment_tint(self, tint_color: str, tint_alpha: float, xmin, xmax, ymin, ymax):
        """Draw a semi-transparent color overlay for environment tinting."""
        if tint_color and tint_alpha > 0:
            tint_rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                facecolor=tint_color, alpha=tint_alpha, zorder=100
            )
            self.ax.add_patch(tint_rect)

    def _draw_particles(self, particle_type: str, t: float, xmin, xmax, ymin, ymax):
        """Draw environment-specific particle effects."""
        if not particle_type:
            return

        rng = np.random.default_rng(int(t * 10) % 1000)  # Deterministic per-frame
        num_particles = 30

        if particle_type == "rain":
            for _ in range(num_particles):
                x = rng.uniform(xmin, xmax)
                y_start = rng.uniform(ymin, ymax)
                length = rng.uniform(0.2, 0.5)
                self.ax.plot([x, x - 0.05], [y_start, y_start - length],
                           color='#6699cc', alpha=0.4, linewidth=1, zorder=90)

        elif particle_type == "snow":
            for _ in range(num_particles):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.02, 0.08)
                self.ax.add_patch(patches.Circle((x, y), size, color='white', alpha=0.7, zorder=90))

        elif particle_type == "bubbles":
            for _ in range(num_particles // 2):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.03, 0.1)
                self.ax.add_patch(patches.Circle((x, y), size, facecolor='none',
                                                edgecolor='#66ccff', alpha=0.5, linewidth=1, zorder=90))

        elif particle_type == "stars":
            for _ in range(num_particles):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.01, 0.04)
                alpha = rng.uniform(0.3, 1.0)
                self.ax.add_patch(patches.Circle((x, y), size, color='white', alpha=alpha, zorder=5))

        elif particle_type == "ash":
            for _ in range(num_particles // 2):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.02, 0.06)
                gray = rng.uniform(0.3, 0.6)
                self.ax.add_patch(patches.Circle((x, y), size, color=str(gray), alpha=0.6, zorder=90))

        elif particle_type == "leaves":
            for _ in range(num_particles // 3):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.03, 0.08)
                color = rng.choice(['#228b22', '#32cd32', '#6b8e23', '#8b4513'])
                self.ax.add_patch(patches.Circle((x, y), size, color=color, alpha=0.6, zorder=90))

        elif particle_type == "sand":
            for _ in range(num_particles):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                size = rng.uniform(0.01, 0.03)
                self.ax.add_patch(patches.Circle((x, y), size, color='#daa520', alpha=0.4, zorder=90))

    def render_scene(
        self,
        scene: Scene,
        output_path: str,
        camera_mode: str = "static",
        cinematic: bool = False,
    ):
        # Get environment info from scene
        env_type = getattr(scene, 'environment_type', EnvironmentType.EARTH_NORMAL)
        weather_type = getattr(scene, 'weather_type', None)

        # Get environment-specific visual configuration
        bg_color, tint_color, tint_alpha, particle_type = self._get_environment_visuals(
            env_type, weather_type
        )

        if cinematic:
            actors = [
                CinematicRenderer(a, environment_type=env_type, weather_type=weather_type)
                for a in scene.actors
            ]
        else:
            actors = [
                StickFigure(a, environment_type=env_type, weather_type=weather_type)
                for a in scene.actors
            ]
        objects = scene.objects

        # Setup Camera based on mode
        if camera_mode == "dynamic":
            # Example dynamic behavior: track the first actor
            if actors:
                self.camera.track_actor(actors[0].id)

            # Or add a slow zoom in
            # self.camera.add_movement(Zoom((0,0), 0.8, 1.2, 0, scene.duration))

        # Override background for specific styles
        if self.style == RenderStyle.NEON:
            bg_color = "black"

        def update(frame):
            self.ax.clear()

            t = frame * 0.04  # 25 fps

            # Update Camera
            actors_dict = {a.id: a for a in actors}
            self.camera.update(t, actors_dict)
            xmin, xmax, ymin, ymax = self.camera.get_view_limits()

            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.axis("off")
            self.ax.set_facecolor(bg_color)
            self.fig.patch.set_facecolor(bg_color)

            # Draw background particles (stars, etc.) - behind everything
            if particle_type == "stars":
                self._draw_particles(particle_type, t, xmin, xmax, ymin, ymax)

            # Draw Background Objects
            self._draw_objects(objects, t)

            # Draw all actors
            for actor in actors:
                self._draw_actor(actor, t)

            # Draw foreground particles (rain, snow, bubbles, etc.) - in front of actors
            if particle_type and particle_type != "stars":
                self._draw_particles(particle_type, t, xmin, xmax, ymin, ymax)

            # Draw environment tint overlay (on top of everything)
            self._draw_environment_tint(tint_color, tint_alpha, xmin, xmax, ymin, ymax)

        ani = animation.FuncAnimation(
            self.fig, update, frames=int(scene.duration * 25), interval=40
        )
        ani.save(output_path, writer="ffmpeg", fps=25)
        plt.close()

    def _draw_objects(self, objects: list, t: float):
        """Draw all scene objects"""
        for obj in objects:
            # Update position if object has velocity
            obj_x = obj.position.x
            obj_y = obj.position.y
            if obj.velocity:
                obj_x += obj.velocity[0] * t
                obj_y += obj.velocity[1] * t

            if obj.type == ObjectType.TREE:
                self.ax.add_patch(
                    patches.Rectangle(
                        (obj_x - 0.1 * obj.scale, obj_y),
                        0.2 * obj.scale,
                        1.5 * obj.scale,
                        color="brown",
                    )
                )
                self.ax.add_patch(
                    patches.Circle(
                        (obj_x, obj_y + 1.5 * obj.scale), 0.8 * obj.scale, color="green"
                    )
                )

            elif obj.type == ObjectType.BUILDING:
                self.ax.add_patch(
                    patches.Rectangle(
                        (obj_x - 1.0 * obj.scale, obj_y),
                        2.0 * obj.scale,
                        3.0 * obj.scale,
                        color="gray",
                    )
                )
                for wx in [-0.6, 0.2]:
                    for wy in [0.5, 1.5, 2.5]:
                        self.ax.add_patch(
                            patches.Rectangle(
                                (obj_x + wx * obj.scale, obj_y + wy * obj.scale),
                                0.4 * obj.scale,
                                0.4 * obj.scale,
                                color="lightblue",
                            )
                        )

            elif obj.type in [
                ObjectType.BALL,
                ObjectType.BASEBALL,
                ObjectType.BASKETBALL,
                ObjectType.SOCCER_BALL,
            ]:
                self.ax.add_patch(
                    patches.Circle(
                        (obj_x, obj_y + 0.2 * obj.scale),
                        0.2 * obj.scale,
                        color=obj.color,
                    )
                )

            elif obj.type == ObjectType.BASE:
                self.ax.add_patch(
                    patches.Rectangle(
                        (obj_x - 0.15 * obj.scale, obj_y - 0.15 * obj.scale),
                        0.3 * obj.scale,
                        0.3 * obj.scale,
                        color=obj.color,
                    )
                )

            elif obj.type == ObjectType.LAPTOP:
                self.ax.add_patch(
                    patches.Rectangle(
                        (obj_x - 0.3 * obj.scale, obj_y + 0.5),
                        0.6 * obj.scale,
                        0.05 * obj.scale,
                        color="gray",
                    )
                )
                self.ax.plot(
                    [obj_x - 0.3 * obj.scale, obj_x - 0.3 * obj.scale],
                    [obj_y + 0.5, obj_y + 0.9],
                    color="gray",
                    lw=2,
                )

            elif obj.type == ObjectType.TABLE:
                self.ax.add_patch(
                    patches.Rectangle(
                        (obj_x - 0.8 * obj.scale, obj_y),
                        1.6 * obj.scale,
                        0.1 * obj.scale,
                        color=obj.color,
                    )
                )
                # Table legs
                for leg_x in [-0.7, 0.7]:
                    self.ax.plot(
                        [obj_x + leg_x * obj.scale, obj_x + leg_x * obj.scale],
                        [obj_y, obj_y - 0.5],
                        color=obj.color,
                        lw=2,
                    )

            elif obj.type == ObjectType.FOOD:
                self.ax.add_patch(
                    patches.Circle((obj_x, obj_y), 0.15 * obj.scale, color=obj.color)
                )

            elif obj.type == ObjectType.PLANET:
                self.ax.add_patch(
                    patches.Circle(
                        (obj_x, obj_y), 0.8 * obj.scale, color=obj.color, alpha=0.7
                    )
                )

            elif obj.type == ObjectType.STAR:
                # Draw star as small bright circle
                self.ax.add_patch(
                    patches.Circle(
                        (obj_x, obj_y), 0.1 * obj.scale, color=obj.color, alpha=0.9
                    )
                )

            elif obj.type == ObjectType.SPACESHIP:
                # Simple spaceship shape
                ship_points = np.array(
                    [
                        [obj_x, obj_y + 0.3 * obj.scale],
                        [obj_x - 0.4 * obj.scale, obj_y - 0.3 * obj.scale],
                        [obj_x + 0.4 * obj.scale, obj_y - 0.3 * obj.scale],
                    ]
                )
                self.ax.add_patch(patches.Polygon(ship_points, color=obj.color))

    def _draw_actor(self, actor: StickFigure, t: float):
        """
        Draw a single actor with expression transitions and speech animation.

        Phase 5.2: Expression transitions
        Phase 7: Speech animation
        """
        from .story_engine import ACTION_EXPRESSIONS, SPEECH_ANIMATION_CONFIG

        current_action = actor.get_current_action(t)
        if current_action != actor.current_action:
            # Action changed - trigger expression transition
            actor.current_action = current_action
            new_expression = ACTION_EXPRESSIONS.get(
                current_action, FacialExpression.NEUTRAL
            )
            actor.update_expression(t, new_expression)

            # Phase 7: Update speech animation parameters
            if current_action in SPEECH_ANIMATION_CONFIG:
                # Enable speech animation
                speech_config = SPEECH_ANIMATION_CONFIG[current_action]
                actor.face_features.is_speaking = True
                actor.face_features.speech_cycle_speed = speech_config["cycle_speed"]

                # Update mouth shape based on speech type
                # Use the first mouth shape in the cycle as the base
                if speech_config["mouth_shapes"]:
                    actor.face_features.mouth_shape = speech_config["mouth_shapes"][0]

                # Set mouth openness range
                openness_min, openness_max = speech_config["openness_range"]
                actor.face_features.mouth_openness = (openness_min + openness_max) / 2
            else:
                # Disable speech animation for non-speech actions
                actor.face_features.is_speaking = False

        lines, head = actor.get_pose(t)

        # Draw body lines
        for line_data in lines:
            if len(line_data) == 4:
                # Cinematic line: (start, end, width, z)
                start, end, width, z = line_data
                (line,) = self.ax.plot(
                    [start[0], end[0]], [start[1], end[1]], color=actor.color, lw=width
                )
            else:
                # Standard line: (start, end)
                start, end = line_data
                (line,) = self.ax.plot(
                    [start[0], end[0]], [start[1], end[1]], color=actor.color, lw=2
                )

            self._apply_style(line)

        # Draw head - different shapes for different actor types
        if actor.actor_type == ActorType.ALIEN:
            # Alien head - larger and oval
            ellipse = patches.Ellipse(
                (head[0], head[1]),
                0.5 * actor.scale,
                0.4 * actor.scale,
                color=actor.color,
                fill=False,
                lw=2,
            )
            self.ax.add_patch(ellipse)
            # Alien eyes
            self.ax.plot(
                [head[0] - 0.1, head[0] + 0.1],
                [head[1], head[1]],
                "o",
                color=actor.color,
                markersize=3,
            )
        else:
            # Human head - circle
            circle = plt.Circle(
                (head[0], head[1]),
                0.3 * actor.scale,
                color=actor.color,
                fill=False,
                lw=2,
            )
            self.ax.add_patch(circle)

            # Draw facial features (Phase 5)
            actor._draw_face(self.ax, head, t)

    def render_raw_frames(
        self,
        frames_data: list[list[float]],
        output_path: str,
        scene_context: Scene = None,
    ):
        # frames_data: List of [x1, y1, x2, y2, ...] (20 floats per frame)

        # Set background color based on theme
        bg_color = "white"
        if scene_context and scene_context.theme and "space" in scene_context.theme:
            bg_color = "black"

        def update(frame_idx):
            self.ax.clear()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.axis("off")
            self.ax.set_facecolor(bg_color)

            t = frame_idx * 0.04

            # Draw Background Objects if context provided
            if scene_context:
                self._draw_objects(scene_context.objects, t)

            frame = frames_data[frame_idx]
            # 20 floats -> 5 lines * 4 coords
            for i in range(0, len(frame), 4):
                x1, y1, x2, y2 = frame[i : i + 4]
                self.ax.plot([x1, x2], [y1, y2], color="black", lw=2)

        ani = animation.FuncAnimation(
            self.fig, update, frames=len(frames_data), interval=40
        )
        ani.save(output_path, writer="ffmpeg", fps=25)
        plt.close()
