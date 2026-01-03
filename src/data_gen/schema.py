from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class FacialExpression(str, Enum):
    """Facial expression types for stick figures"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    EXCITED = "excited"


class MouthShape(str, Enum):
    """Mouth shapes for speech animation and expressions"""

    CLOSED = "closed"  # Neutral closed mouth
    SMILE = "smile"  # Happy smile
    FROWN = "frown"  # Sad frown
    OPEN = "open"  # Surprised/shocked
    WIDE_OPEN = "wide_open"  # Shouting/very surprised
    SMALL_O = "small_o"  # Talking/whispering
    SINGING = "singing"  # Singing (wide oval)


class ActionType(str, Enum):
    # Basic actions
    IDLE = "idle"
    WALK = "walk"
    RUN = "run"
    SPRINT = "sprint"
    JUMP = "jump"

    # Social actions
    WAVE = "wave"
    TALK = "talk"
    SHOUT = "shout"  # Phase 7: Speech animation
    WHISPER = "whisper"  # Phase 7: Speech animation
    SING = "sing"  # Phase 7: Speech animation
    POINT = "point"
    CLAP = "clap"

    # Sports actions - Baseball
    BATTING = "batting"
    PITCHING = "pitching"
    CATCHING = "catching"
    RUNNING_BASES = "running_bases"
    FIELDING = "fielding"
    THROWING = "throwing"

    # Sports actions - General
    KICKING = "kicking"
    DRIBBLING = "dribbling"
    SHOOTING = "shooting"

    # Combat actions
    FIGHT = "fight"
    PUNCH = "punch"
    KICK = "kick"
    DODGE = "dodge"

    # Narrative actions
    SIT = "sit"
    STAND = "stand"
    KNEEL = "kneel"
    LIE_DOWN = "lie_down"
    EATING = "eating"
    DRINKING = "drinking"
    READING = "reading"
    TYPING = "typing"

    # Exploration actions
    LOOKING_AROUND = "looking_around"
    CLIMBING = "climbing"
    CRAWLING = "crawling"
    SWIMMING = "swimming"
    FLYING = "flying"

    # Emotional actions
    CELEBRATE = "celebrate"
    DANCE = "dance"
    CRY = "cry"
    LAUGH = "laugh"

    # Interactive Actions (Multi-Actor)
    HANDSHAKE = "handshake"
    HUG = "hug"
    HIGH_FIVE = "high_five"
    FIGHT_STANCE = "fight_stance"


# Action-to-index mapping for action embedding (Phase 1)
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ActionType)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}
NUM_ACTIONS = len(ActionType)


def generate_per_frame_actions(
    actions: list[tuple[float, ActionType]], fps: int = 25, total_duration: float = 10.0
) -> list[ActionType]:
    """
    Convert time-based action sequence to per-frame action sequence.

    Args:
        actions: List of (start_time, action) tuples
        fps: Frames per second (default: 25)
        total_duration: Total duration in seconds (default: 10.0)

    Returns:
        List of ActionType, one per frame (length = fps * total_duration)

    Example:
        >>> actions = [(0.0, ActionType.WALK), (5.0, ActionType.RUN)]
        >>> per_frame = generate_per_frame_actions(actions, fps=25, total_duration=10.0)
        >>> len(per_frame)
        250
    """
    total_frames = int(fps * total_duration)
    per_frame_actions = [ActionType.IDLE] * total_frames

    if not actions:
        return per_frame_actions

    # Sort actions by start time
    sorted_actions = sorted(actions, key=lambda x: x[0])

    # Fill in actions based on start times
    for i, (start_time, action) in enumerate(sorted_actions):
        start_frame = int(fps * start_time)

        # Determine end frame (either next action's start or total duration)
        if i + 1 < len(sorted_actions):
            end_time = sorted_actions[i + 1][0]
        else:
            end_time = total_duration

        end_frame = int(fps * end_time)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        # Fill frames with this action
        for frame_idx in range(start_frame, end_frame):
            per_frame_actions[frame_idx] = action

    return per_frame_actions


class ActorType(str, Enum):
    HUMAN = "human"
    ALIEN = "alien"
    ROBOT = "robot"
    ANIMAL = "animal"


class MotionProfile:
    """
    Actor-type-specific motion characteristics.

    These parameters modify how animations are generated to create
    distinct movement patterns for different actor types.
    """

    def __init__(
        self,
        # Timing and rhythm
        frequency_multiplier: float = 1.0,  # Animation speed modifier
        time_quantization: float = 0.0,  # Quantize time to steps (0=smooth, 0.1=100ms steps)

        # Motion amplitude and style
        amplitude_modifier: float = 1.0,  # Scale of motion swings
        sway_amplitude: float = 1.0,  # Organic sway/breathing
        joint_noise: float = 0.0,  # Random jitter per frame

        # Proportions
        limb_length_scale: float = 1.0,  # Arm/leg length modifier
        torso_scale: float = 1.0,  # Torso length modifier
        stance_width: float = 1.0,  # Leg spread modifier

        # Posture
        forward_lean: float = 0.0,  # Forward tilt in radians
        crouch_factor: float = 0.0,  # Lower stance (0-1)

        # Interpolation style
        smoothness: float = 1.0,  # 1.0=smooth sinusoid, 0.0=stepped/mechanical
        phase_offset: float = 0.0,  # Offset between arm/leg phases
    ):
        self.frequency_multiplier = frequency_multiplier
        self.time_quantization = time_quantization
        self.amplitude_modifier = amplitude_modifier
        self.sway_amplitude = sway_amplitude
        self.joint_noise = joint_noise
        self.limb_length_scale = limb_length_scale
        self.torso_scale = torso_scale
        self.stance_width = stance_width
        self.forward_lean = forward_lean
        self.crouch_factor = crouch_factor
        self.smoothness = smoothness
        self.phase_offset = phase_offset


# Motion profiles for each actor type
MOTION_PROFILES: dict[ActorType, MotionProfile] = {
    # Human: Natural, smooth, organic motion (baseline)
    ActorType.HUMAN: MotionProfile(
        frequency_multiplier=1.0,
        time_quantization=0.0,
        amplitude_modifier=1.0,
        sway_amplitude=1.0,
        joint_noise=0.0,
        limb_length_scale=1.0,
        torso_scale=1.0,
        stance_width=1.0,
        forward_lean=0.0,
        crouch_factor=0.0,
        smoothness=1.0,
        phase_offset=0.0,
    ),

    # Robot: Mechanical, stepped, precise movements
    ActorType.ROBOT: MotionProfile(
        frequency_multiplier=0.8,  # Slightly slower, more deliberate
        time_quantization=0.08,  # Quantize to ~12fps (servo-like)
        amplitude_modifier=0.85,  # Tighter, more controlled movements
        sway_amplitude=0.1,  # Almost no organic sway
        joint_noise=0.02,  # Slight mechanical jitter
        limb_length_scale=1.0,
        torso_scale=1.0,
        stance_width=1.1,  # Slightly wider, more stable stance
        forward_lean=0.0,
        crouch_factor=0.0,
        smoothness=0.3,  # Very stepped/mechanical
        phase_offset=0.05,  # Slight joint delay (servo lag)
    ),

    # Alien: Fluid but unusual, non-human rhythms
    ActorType.ALIEN: MotionProfile(
        frequency_multiplier=1.3,  # Slightly faster, unusual timing
        time_quantization=0.0,  # Smooth but eerie
        amplitude_modifier=1.2,  # More exaggerated movements
        sway_amplitude=0.6,  # Less breathing, more gliding
        joint_noise=0.015,  # Slight otherworldly shimmer
        limb_length_scale=1.15,  # Slightly longer limbs
        torso_scale=0.95,  # Slightly shorter torso
        stance_width=0.85,  # Narrower stance
        forward_lean=0.05,  # Slight forward lean
        crouch_factor=0.0,
        smoothness=1.2,  # Extra smooth, almost floating
        phase_offset=0.15,  # Asymmetric arm-leg timing
    ),

    # Animal: Lower, forward-leaning, natural but non-humanoid
    ActorType.ANIMAL: MotionProfile(
        frequency_multiplier=1.4,  # Faster gait
        time_quantization=0.0,
        amplitude_modifier=1.3,  # More dynamic movement
        sway_amplitude=0.8,
        joint_noise=0.01,
        limb_length_scale=0.9,  # Shorter limbs
        torso_scale=1.1,  # Longer torso
        stance_width=1.2,  # Wider stance
        forward_lean=0.15,  # Noticeable forward lean
        crouch_factor=0.2,  # Lower to ground
        smoothness=0.9,
        phase_offset=0.0,  # Natural gait phase
    ),
}


def get_motion_profile(actor_type: ActorType) -> MotionProfile:
    """Get the motion profile for an actor type."""
    return MOTION_PROFILES.get(actor_type, MOTION_PROFILES[ActorType.HUMAN])


class EnvironmentType(str, Enum):
    """Types of environments that affect physics and motion."""
    # Terrestrial
    EARTH_NORMAL = "earth_normal"
    GRASSLAND = "grassland"
    FOREST = "forest"
    DESERT = "desert"
    BEACH = "beach"
    MOUNTAIN = "mountain"
    ARCTIC = "arctic"
    JUNGLE = "jungle"
    SWAMP = "swamp"
    CAVE = "cave"
    VOLCANO = "volcano"

    # Urban
    CITY_STREET = "city_street"
    CITY_PARK = "city_park"
    ROOFTOP = "rooftop"
    ALLEY = "alley"
    SUBWAY = "subway"
    MALL = "mall"

    # Indoor
    OFFICE = "office"
    GYM = "gym"
    MUSEUM = "museum"
    RESTAURANT = "restaurant"
    THEATER = "theater"
    LIBRARY = "library"
    HOSPITAL = "hospital"
    FACTORY = "factory"
    WAREHOUSE = "warehouse"
    CLASSROOM = "classroom"

    # Aquatic
    UNDERWATER = "underwater"
    OCEAN_SURFACE = "ocean_surface"
    RIVER = "river"
    LAKE = "lake"
    POOL = "pool"

    # Space/Sci-Fi
    SPACE_VACUUM = "space_vacuum"
    MOON = "moon"
    MARS = "mars"
    ASTEROID = "asteroid"
    SPACESHIP_INTERIOR = "spaceship_interior"
    SPACE_STATION = "space_station"
    ALIEN_PLANET_LOW_G = "alien_planet_low_g"
    ALIEN_PLANET_HIGH_G = "alien_planet_high_g"

    # Fantasy
    CASTLE = "castle"
    DUNGEON = "dungeon"
    ENCHANTED_FOREST = "enchanted_forest"
    CLOUD_REALM = "cloud_realm"
    LAVA_REALM = "lava_realm"
    ICE_REALM = "ice_realm"

    # Weather Conditions (can combine with location)
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    FOGGY = "foggy"
    STORMY = "stormy"

    # Sports/Activity Venues
    STADIUM = "stadium"
    ARENA = "arena"
    RINK = "rink"
    COURT = "court"
    TRACK = "track"
    FIELD = "field"

    # Social
    CONCERT = "concert"
    PARTY = "party"
    WEDDING = "wedding"
    MARKET = "market"
    FESTIVAL = "festival"
    PROTEST = "protest"
    PARADE = "parade"


@dataclass
class EnvironmentProfile:
    """
    Physics and atmosphere parameters for different environments.
    These modify motion characteristics based on physical realism.
    """
    # Gravity effects
    gravity_scale: float = 1.0  # 0.16 (moon), 0.38 (mars), 1.0 (earth), 2.5 (high-g)

    # Surface/terrain friction
    friction: float = 1.0  # 0.1 (ice), 0.5 (wet), 1.0 (normal), 1.5 (sand/rough)

    # Atmosphere/medium resistance
    air_resistance: float = 1.0  # 0.0 (vacuum), 0.7 (thin), 1.0 (normal), 3.0 (underwater)

    # Buoyancy effect (vertical float tendency)
    buoyancy: float = 0.0  # 0.0 (normal), 0.5 (partial float), 1.0 (neutral buoyancy)

    # Movement speed modifier
    movement_speed_scale: float = 1.0  # 0.5 (difficult terrain), 1.0 (normal), 1.2 (smooth)

    # Step height (how high feet lift)
    step_height_scale: float = 1.0  # 0.5 (smooth floor), 1.0 (normal), 2.0 (rocky/obstacles)

    # Jump height modifier (combines with gravity)
    jump_height_scale: float = 1.0  # Based on gravity and surface

    # Landing impact (affects recovery time)
    landing_impact: float = 1.0  # 0.3 (soft/low-g), 1.0 (normal), 1.5 (hard surface)

    # Balance difficulty (affects sway and stability)
    balance_difficulty: float = 1.0  # 0.5 (stable), 1.0 (normal), 2.0 (unstable/slippery)

    # Visibility modifier (affects looking around behavior)
    visibility: float = 1.0  # 0.3 (fog/dark), 1.0 (clear)

    # Temperature effect (affects movement energy)
    temperature_modifier: float = 1.0  # 0.7 (cold/sluggish), 1.0 (normal), 1.1 (warm/energetic)

    # Wind effect (lateral movement influence)
    wind_strength: float = 0.0  # 0.0 (calm), 0.5 (breezy), 1.0 (strong wind)
    wind_direction: float = 0.0  # Angle in radians (0 = from left, π/2 = from front)

    # Terrain slope effect
    slope_angle: float = 0.0  # Radians, positive = uphill

    # Crowd density (affects movement freedom)
    crowd_density: float = 0.0  # 0.0 (empty), 0.5 (moderate), 1.0 (packed)

    # Noise level (affects behavior intensity)
    ambient_noise: float = 0.5  # 0.0 (silent), 0.5 (normal), 1.0 (loud)


# Environment profiles for each environment type
ENVIRONMENT_PROFILES: dict[EnvironmentType, EnvironmentProfile] = {
    # === TERRESTRIAL ===
    EnvironmentType.EARTH_NORMAL: EnvironmentProfile(),  # Baseline

    EnvironmentType.GRASSLAND: EnvironmentProfile(
        friction=0.9,
        step_height_scale=1.1,
    ),

    EnvironmentType.FOREST: EnvironmentProfile(
        friction=0.85,
        movement_speed_scale=0.85,
        step_height_scale=1.3,  # Roots, uneven ground
        visibility=0.7,
    ),

    EnvironmentType.DESERT: EnvironmentProfile(
        friction=0.6,  # Sand shifts
        movement_speed_scale=0.75,
        step_height_scale=1.4,  # Sinking in sand
        temperature_modifier=0.85,  # Heat exhaustion
        visibility=0.9,
    ),

    EnvironmentType.BEACH: EnvironmentProfile(
        friction=0.5,  # Wet sand
        movement_speed_scale=0.8,
        step_height_scale=1.2,
        wind_strength=0.3,
    ),

    EnvironmentType.MOUNTAIN: EnvironmentProfile(
        air_resistance=0.7,  # Thin air
        movement_speed_scale=0.7,
        step_height_scale=1.5,  # Rocky terrain
        jump_height_scale=0.85,  # Harder to jump at altitude
        balance_difficulty=1.3,
        slope_angle=0.3,  # ~17 degrees
        temperature_modifier=0.8,
    ),

    EnvironmentType.ARCTIC: EnvironmentProfile(
        friction=0.15,  # Ice
        movement_speed_scale=0.6,
        step_height_scale=1.2,
        balance_difficulty=2.0,
        temperature_modifier=0.6,
        wind_strength=0.5,
    ),

    EnvironmentType.JUNGLE: EnvironmentProfile(
        friction=0.7,  # Humid, muddy
        air_resistance=1.2,  # Dense vegetation
        movement_speed_scale=0.65,
        step_height_scale=1.6,  # Vines, roots
        visibility=0.5,
        temperature_modifier=0.9,
    ),

    EnvironmentType.SWAMP: EnvironmentProfile(
        friction=0.4,
        movement_speed_scale=0.5,
        step_height_scale=1.8,  # Pulling feet from mud
        balance_difficulty=1.5,
        visibility=0.6,
    ),

    EnvironmentType.CAVE: EnvironmentProfile(
        movement_speed_scale=0.8,
        step_height_scale=1.4,
        visibility=0.3,
        ambient_noise=0.2,
    ),

    EnvironmentType.VOLCANO: EnvironmentProfile(
        friction=0.8,
        movement_speed_scale=0.7,
        step_height_scale=1.5,
        temperature_modifier=0.7,  # Extreme heat
        visibility=0.6,  # Smoke
        ambient_noise=0.8,
    ),

    # === URBAN ===
    EnvironmentType.CITY_STREET: EnvironmentProfile(
        friction=1.1,  # Concrete
        movement_speed_scale=1.0,
        crowd_density=0.4,
        ambient_noise=0.7,
    ),

    EnvironmentType.CITY_PARK: EnvironmentProfile(
        friction=0.95,
        step_height_scale=1.1,
        crowd_density=0.2,
        ambient_noise=0.4,
    ),

    EnvironmentType.ROOFTOP: EnvironmentProfile(
        wind_strength=0.6,
        balance_difficulty=1.2,
        visibility=1.0,
    ),

    EnvironmentType.ALLEY: EnvironmentProfile(
        visibility=0.6,
        crowd_density=0.1,
        ambient_noise=0.3,
    ),

    EnvironmentType.SUBWAY: EnvironmentProfile(
        friction=1.0,
        visibility=0.7,
        crowd_density=0.6,
        ambient_noise=0.9,
    ),

    EnvironmentType.MALL: EnvironmentProfile(
        friction=1.1,  # Smooth floors
        movement_speed_scale=1.1,
        crowd_density=0.5,
        ambient_noise=0.6,
    ),

    # === INDOOR ===
    EnvironmentType.OFFICE: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.9,  # Confined spaces
        crowd_density=0.3,
        ambient_noise=0.3,
    ),

    EnvironmentType.GYM: EnvironmentProfile(
        friction=1.2,  # Rubber mats
        movement_speed_scale=1.1,
        jump_height_scale=1.1,
        ambient_noise=0.6,
    ),

    EnvironmentType.MUSEUM: EnvironmentProfile(
        friction=1.1,
        movement_speed_scale=0.7,  # Walking slowly
        ambient_noise=0.2,
    ),

    EnvironmentType.RESTAURANT: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.8,
        crowd_density=0.4,
        ambient_noise=0.5,
    ),

    EnvironmentType.THEATER: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.7,
        visibility=0.4,
        ambient_noise=0.1,
    ),

    EnvironmentType.LIBRARY: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.6,
        ambient_noise=0.1,
    ),

    EnvironmentType.HOSPITAL: EnvironmentProfile(
        friction=1.1,
        movement_speed_scale=0.8,
        ambient_noise=0.3,
    ),

    EnvironmentType.FACTORY: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.9,
        step_height_scale=1.2,
        ambient_noise=0.9,
    ),

    EnvironmentType.WAREHOUSE: EnvironmentProfile(
        friction=1.0,
        step_height_scale=1.3,
        visibility=0.7,
        ambient_noise=0.4,
    ),

    EnvironmentType.CLASSROOM: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.8,
        crowd_density=0.5,
        ambient_noise=0.4,
    ),

    # === AQUATIC ===
    EnvironmentType.UNDERWATER: EnvironmentProfile(
        gravity_scale=0.1,  # Near-neutral buoyancy
        friction=0.3,
        air_resistance=3.0,  # Water resistance
        buoyancy=0.9,
        movement_speed_scale=0.4,
        step_height_scale=0.5,  # Swimming motion
        jump_height_scale=0.3,
        landing_impact=0.2,
        balance_difficulty=0.5,  # Easier to balance in water
        visibility=0.5,
    ),

    EnvironmentType.OCEAN_SURFACE: EnvironmentProfile(
        gravity_scale=0.8,
        friction=0.2,
        buoyancy=0.6,
        movement_speed_scale=0.5,
        balance_difficulty=1.8,  # Waves
        wind_strength=0.4,
    ),

    EnvironmentType.RIVER: EnvironmentProfile(
        gravity_scale=0.9,
        friction=0.4,
        air_resistance=1.5,
        buoyancy=0.3,
        movement_speed_scale=0.6,
        balance_difficulty=1.5,
    ),

    EnvironmentType.LAKE: EnvironmentProfile(
        gravity_scale=0.85,
        friction=0.3,
        buoyancy=0.4,
        movement_speed_scale=0.55,
        balance_difficulty=1.2,
    ),

    EnvironmentType.POOL: EnvironmentProfile(
        gravity_scale=0.2,
        friction=0.25,
        air_resistance=2.5,
        buoyancy=0.8,
        movement_speed_scale=0.45,
        landing_impact=0.3,
    ),

    # === SPACE/SCI-FI ===
    EnvironmentType.SPACE_VACUUM: EnvironmentProfile(
        gravity_scale=0.0,  # Zero-G
        friction=0.0,
        air_resistance=0.0,
        buoyancy=1.0,  # Float freely
        movement_speed_scale=0.3,  # Slow, deliberate
        jump_height_scale=5.0,  # Infinite jump
        landing_impact=0.0,
        balance_difficulty=0.3,
    ),

    EnvironmentType.MOON: EnvironmentProfile(
        gravity_scale=0.16,
        friction=0.5,  # Regolith
        air_resistance=0.0,
        movement_speed_scale=0.6,
        step_height_scale=2.0,  # Bounding
        jump_height_scale=6.0,
        landing_impact=0.3,
    ),

    EnvironmentType.MARS: EnvironmentProfile(
        gravity_scale=0.38,
        friction=0.6,
        air_resistance=0.01,  # Thin atmosphere
        movement_speed_scale=0.75,
        step_height_scale=1.5,
        jump_height_scale=2.6,
        landing_impact=0.5,
        wind_strength=0.2,  # Dust storms
    ),

    EnvironmentType.ASTEROID: EnvironmentProfile(
        gravity_scale=0.05,
        friction=0.4,
        air_resistance=0.0,
        buoyancy=0.8,
        movement_speed_scale=0.4,
        jump_height_scale=10.0,
        balance_difficulty=2.0,
    ),

    EnvironmentType.SPACESHIP_INTERIOR: EnvironmentProfile(
        gravity_scale=1.0,  # Artificial gravity
        friction=1.0,
        movement_speed_scale=0.9,
        ambient_noise=0.4,
    ),

    EnvironmentType.SPACE_STATION: EnvironmentProfile(
        gravity_scale=0.8,  # Partial artificial gravity
        friction=1.0,
        movement_speed_scale=0.85,
        step_height_scale=1.2,
        ambient_noise=0.3,
    ),

    EnvironmentType.ALIEN_PLANET_LOW_G: EnvironmentProfile(
        gravity_scale=0.3,
        friction=0.7,
        air_resistance=0.8,
        movement_speed_scale=0.8,
        step_height_scale=1.8,
        jump_height_scale=3.3,
        visibility=0.8,
    ),

    EnvironmentType.ALIEN_PLANET_HIGH_G: EnvironmentProfile(
        gravity_scale=2.5,
        friction=1.2,
        air_resistance=1.5,
        movement_speed_scale=0.5,
        step_height_scale=0.6,
        jump_height_scale=0.4,
        landing_impact=2.0,
        balance_difficulty=1.5,
    ),

    # === FANTASY ===
    EnvironmentType.CASTLE: EnvironmentProfile(
        friction=1.0,
        step_height_scale=1.3,  # Stairs
        visibility=0.7,
        ambient_noise=0.3,
    ),

    EnvironmentType.DUNGEON: EnvironmentProfile(
        friction=0.8,
        movement_speed_scale=0.75,
        step_height_scale=1.4,
        visibility=0.2,
        ambient_noise=0.2,
    ),

    EnvironmentType.ENCHANTED_FOREST: EnvironmentProfile(
        gravity_scale=0.9,  # Slight magical lift
        friction=0.85,
        movement_speed_scale=0.9,
        step_height_scale=1.2,
        visibility=0.6,
    ),

    EnvironmentType.CLOUD_REALM: EnvironmentProfile(
        gravity_scale=0.4,
        friction=0.3,
        buoyancy=0.5,
        movement_speed_scale=0.7,
        step_height_scale=0.8,
        jump_height_scale=2.5,
        landing_impact=0.3,
        visibility=0.8,
    ),

    EnvironmentType.LAVA_REALM: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=0.6,
        step_height_scale=1.5,
        temperature_modifier=0.5,
        visibility=0.7,
        ambient_noise=0.8,
    ),

    EnvironmentType.ICE_REALM: EnvironmentProfile(
        friction=0.1,
        movement_speed_scale=0.5,
        balance_difficulty=2.5,
        temperature_modifier=0.5,
        visibility=0.9,
    ),

    # === WEATHER CONDITIONS ===
    EnvironmentType.RAINY: EnvironmentProfile(
        friction=0.6,
        movement_speed_scale=0.85,
        visibility=0.6,
        wind_strength=0.2,
    ),

    EnvironmentType.SNOWY: EnvironmentProfile(
        friction=0.4,
        movement_speed_scale=0.7,
        step_height_scale=1.4,
        visibility=0.5,
        temperature_modifier=0.7,
    ),

    EnvironmentType.WINDY: EnvironmentProfile(
        wind_strength=0.8,
        balance_difficulty=1.4,
        movement_speed_scale=0.85,
    ),

    EnvironmentType.FOGGY: EnvironmentProfile(
        visibility=0.2,
        movement_speed_scale=0.8,
    ),

    EnvironmentType.STORMY: EnvironmentProfile(
        friction=0.5,
        movement_speed_scale=0.6,
        visibility=0.3,
        wind_strength=1.0,
        balance_difficulty=1.8,
        ambient_noise=1.0,
    ),

    # === SPORTS/ACTIVITY VENUES ===
    EnvironmentType.STADIUM: EnvironmentProfile(
        friction=1.0,
        movement_speed_scale=1.1,
        crowd_density=0.7,
        ambient_noise=0.9,
    ),

    EnvironmentType.ARENA: EnvironmentProfile(
        friction=1.1,
        movement_speed_scale=1.0,
        crowd_density=0.6,
        ambient_noise=0.8,
    ),

    EnvironmentType.RINK: EnvironmentProfile(
        friction=0.08,  # Ice
        movement_speed_scale=1.2,  # Gliding
        balance_difficulty=2.0,
        crowd_density=0.3,
    ),

    EnvironmentType.COURT: EnvironmentProfile(
        friction=1.2,  # Hardwood
        movement_speed_scale=1.1,
        jump_height_scale=1.1,
        crowd_density=0.3,
    ),

    EnvironmentType.TRACK: EnvironmentProfile(
        friction=1.3,  # Rubberized
        movement_speed_scale=1.2,
        crowd_density=0.2,
    ),

    EnvironmentType.FIELD: EnvironmentProfile(
        friction=0.9,
        step_height_scale=1.1,
        crowd_density=0.1,
    ),

    # === SOCIAL ===
    EnvironmentType.CONCERT: EnvironmentProfile(
        movement_speed_scale=0.7,
        crowd_density=0.9,
        ambient_noise=1.0,
        visibility=0.6,
    ),

    EnvironmentType.PARTY: EnvironmentProfile(
        movement_speed_scale=0.9,
        crowd_density=0.6,
        ambient_noise=0.8,
        visibility=0.7,
    ),

    EnvironmentType.WEDDING: EnvironmentProfile(
        movement_speed_scale=0.7,
        crowd_density=0.5,
        ambient_noise=0.4,
    ),

    EnvironmentType.MARKET: EnvironmentProfile(
        movement_speed_scale=0.75,
        crowd_density=0.7,
        ambient_noise=0.7,
        step_height_scale=1.2,
    ),

    EnvironmentType.FESTIVAL: EnvironmentProfile(
        movement_speed_scale=0.8,
        crowd_density=0.8,
        ambient_noise=0.9,
    ),

    EnvironmentType.PROTEST: EnvironmentProfile(
        movement_speed_scale=0.6,
        crowd_density=0.9,
        ambient_noise=1.0,
        balance_difficulty=1.2,
    ),

    EnvironmentType.PARADE: EnvironmentProfile(
        movement_speed_scale=0.8,
        crowd_density=0.7,
        ambient_noise=0.8,
    ),
}


def get_environment_profile(env_type: EnvironmentType) -> EnvironmentProfile:
    """Get the environment profile for an environment type."""
    return ENVIRONMENT_PROFILES.get(env_type, ENVIRONMENT_PROFILES[EnvironmentType.EARTH_NORMAL])


def combine_environment_profiles(
    base: EnvironmentProfile,
    weather: EnvironmentProfile | None = None,
) -> EnvironmentProfile:
    """
    Combine a base environment with weather conditions.
    Weather modifiers are applied multiplicatively.
    """
    if weather is None:
        return base

    return EnvironmentProfile(
        gravity_scale=base.gravity_scale,  # Weather doesn't affect gravity
        friction=base.friction * weather.friction,
        air_resistance=base.air_resistance * weather.air_resistance,
        buoyancy=max(base.buoyancy, weather.buoyancy),
        movement_speed_scale=base.movement_speed_scale * weather.movement_speed_scale,
        step_height_scale=base.step_height_scale * weather.step_height_scale,
        jump_height_scale=base.jump_height_scale * weather.jump_height_scale,
        landing_impact=base.landing_impact * weather.landing_impact,
        balance_difficulty=base.balance_difficulty * weather.balance_difficulty,
        visibility=min(base.visibility, weather.visibility),
        temperature_modifier=base.temperature_modifier * weather.temperature_modifier,
        wind_strength=max(base.wind_strength, weather.wind_strength),
        wind_direction=weather.wind_direction if weather.wind_strength > base.wind_strength else base.wind_direction,
        slope_angle=base.slope_angle,
        crowd_density=base.crowd_density,
        ambient_noise=max(base.ambient_noise, weather.ambient_noise),
    )


class ObjectType(str, Enum):
    # Nature
    TREE = "tree"
    ROCK = "rock"
    GRASS = "grass"

    # Buildings
    BUILDING = "building"
    HOUSE = "house"
    STADIUM = "stadium"

    # Sports equipment
    BALL = "ball"
    BASEBALL = "baseball"
    BASKETBALL = "basketball"
    SOCCER_BALL = "soccer_ball"
    BAT = "bat"
    GLOVE = "glove"
    BASE = "base"

    # Tech
    LAPTOP = "laptop"
    PHONE = "phone"
    COMPUTER = "computer"

    # Furniture
    TABLE = "table"
    CHAIR = "chair"
    BENCH = "bench"

    # Food
    FOOD = "food"
    PLATE = "plate"
    CUP = "cup"

    # Space
    SPACESHIP = "spaceship"
    PLANET = "planet"
    STAR = "star"
    UFO = "ufo"


class FaceFeatures(BaseModel):
    """
    Defines facial feature parameters for stick figure expressions.

    This class controls the visual appearance of facial features including
    eyes, eyebrows, and mouth to create different expressions.

    Attributes:
        expression: The overall facial expression (NEUTRAL, HAPPY, SAD, etc.)
        eye_type: Visual style of eyes ("dots", "curves", "wide", "closed")
        eyebrow_angle: Angle of eyebrows in degrees (-30 to +30)
                      Negative = sad/worried, Positive = happy/surprised
        mouth_shape: Shape of the mouth (CLOSED, SMILE, FROWN, etc.)
        mouth_openness: How open the mouth is (0.0 to 1.0)
                       Used for speech animation
        is_speaking: Whether the character is currently speaking (Phase 7)
        speech_cycle_speed: Speed of speech animation cycle in Hz (Phase 7)
                           TALK: 8 Hz, SHOUT: 6 Hz, WHISPER: 10 Hz, SING: 4 Hz
    """

    expression: FacialExpression = FacialExpression.NEUTRAL
    eye_type: str = "dots"  # "dots", "curves", "wide", "closed"
    eyebrow_angle: float = 0.0  # -30 to +30 degrees
    mouth_shape: MouthShape = MouthShape.CLOSED
    mouth_openness: float = 0.0  # 0.0 (closed) to 1.0 (wide open)

    # Speech animation parameters (Phase 7)
    is_speaking: bool = False
    speech_cycle_speed: float = 8.0  # Hz (cycles per second)


class Position(BaseModel):
    x: float
    y: float


class SceneObject(BaseModel):
    id: str
    type: ObjectType
    position: Position
    color: str = "gray"
    scale: float = 1.0
    velocity: tuple[float, float] | None = None  # For dynamic objects (vx, vy)


class Actor(BaseModel):
    id: str
    actor_type: ActorType = ActorType.HUMAN
    color: str = "black"
    initial_position: Position
    actions: list[tuple[float, ActionType]] = []  # (start_time, action)
    team: str | None = None  # For team-based scenarios
    scale: float = 1.0
    velocity: tuple[float, float] | None = None  # (vx, vy) in units/second
    movement_path: list[tuple[float, Position]] | None = (
        None  # (time, position) waypoints
    )
    per_frame_actions: list[ActionType] | None = (
        None  # Per-frame action sequence (Phase 1)
    )

    # Facial expression fields (Phase 5)
    facial_expression: FacialExpression = FacialExpression.NEUTRAL
    face_features: FaceFeatures | None = None


class CameraKeyframe(BaseModel):
    frame: int
    x: float
    y: float
    zoom: float = 1.0
    interpolation: str = "linear"  # linear, smooth


class Scene(BaseModel):
    duration: float
    actors: list[Actor]
    objects: list[SceneObject] = []
    background_color: str = "white"
    description: str
    theme: str | None = None
    camera_keyframes: list[CameraKeyframe] = []
    environment_type: EnvironmentType = EnvironmentType.EARTH_NORMAL
    weather_type: EnvironmentType | None = None  # Optional weather overlay


# Action-to-velocity mapping (in units/second, where 1 unit ≈ 0.68m)
# Human scale: stick figure is ~2.5 units tall = 1.7m, so 1 unit = 0.68m
ACTION_VELOCITIES = {
    # Basic actions
    ActionType.IDLE: 0.0,
    ActionType.WALK: 1.9,  # 1.3 m/s / 0.68 = 1.9 units/s
    ActionType.RUN: 7.4,  # 5.0 m/s / 0.68 = 7.4 units/s
    ActionType.SPRINT: 11.8,  # 8.0 m/s / 0.68 = 11.8 units/s
    ActionType.JUMP: 0.0,  # Vertical, not horizontal
    # Social actions
    ActionType.WAVE: 0.0,
    ActionType.TALK: 0.0,
    ActionType.POINT: 0.0,
    ActionType.CLAP: 0.0,
    # Sports actions - Baseball
    ActionType.BATTING: 0.0,
    ActionType.PITCHING: 0.0,
    ActionType.CATCHING: 0.0,
    ActionType.RUNNING_BASES: 8.8,  # 6.0 m/s / 0.68 = 8.8 units/s
    ActionType.FIELDING: 0.0,
    ActionType.THROWING: 0.0,
    # Sports actions - General
    ActionType.KICKING: 0.0,
    ActionType.DRIBBLING: 1.5,  # Slow movement while dribbling
    ActionType.SHOOTING: 0.0,
    # Combat actions
    ActionType.FIGHT: 0.3,  # Slow movement during fight
    ActionType.PUNCH: 0.0,
    ActionType.KICK: 0.0,
    ActionType.DODGE: 2.0,  # Quick dodge movement
    # Narrative actions
    ActionType.SIT: 0.0,
    ActionType.STAND: 0.0,
    ActionType.KNEEL: 0.0,
    ActionType.LIE_DOWN: 0.0,
    ActionType.EATING: 0.0,
    ActionType.DRINKING: 0.0,
    ActionType.READING: 0.0,
    ActionType.TYPING: 0.0,
    # Exploration actions
    ActionType.LOOKING_AROUND: 0.0,
    ActionType.CLIMBING: 0.5,  # Slow upward movement
    ActionType.CRAWLING: 0.8,  # Slow forward movement
    ActionType.SWIMMING: 1.2,  # Moderate swimming speed
    ActionType.FLYING: 3.0,  # Flying movement
    # Emotional actions
    ActionType.CELEBRATE: 0.0,
    ActionType.DANCE: 0.5,  # Slow movement while dancing
    ActionType.CRY: 0.0,
    ActionType.LAUGH: 0.0,
}

# Object scales (relative to human height = 2.5 units)
# Human: 2.5 units = 1.7m, so 1 unit = 0.68m
OBJECT_SCALES = {
    # Furniture
    ObjectType.TABLE: 1.1,  # 0.75m height
    ObjectType.CHAIR: 0.65,  # 0.45m height
    ObjectType.BENCH: 0.65,  # 0.45m height
    # Nature
    ObjectType.TREE: 15.0,  # 10m (medium tree)
    ObjectType.ROCK: 1.5,  # 1m
    ObjectType.GRASS: 0.3,  # 0.2m
    # Buildings
    ObjectType.BUILDING: 30.0,  # 20m (small building)
    ObjectType.HOUSE: 15.0,  # 10m
    ObjectType.STADIUM: 50.0,  # 35m
    # Sports equipment
    ObjectType.BALL: 0.3,  # 0.22m diameter (generic)
    ObjectType.BASEBALL: 0.15,  # Small
    ObjectType.BASKETBALL: 0.35,  # 0.24m diameter
    ObjectType.SOCCER_BALL: 0.3,  # 0.22m diameter
    ObjectType.BAT: 1.3,  # 0.9m
    ObjectType.GLOVE: 0.4,  # 0.3m
    ObjectType.BASE: 0.3,  # 0.2m
    # Tech
    ObjectType.LAPTOP: 0.5,  # 0.35m
    ObjectType.COMPUTER: 0.8,  # 0.5m
    ObjectType.PHONE: 0.2,  # 0.15m
    # Space
    ObjectType.SPACESHIP: 25.0,  # 17m
    ObjectType.PLANET: 50.0,  # Background object
    ObjectType.STAR: 0.5,  # Visual element
    ObjectType.UFO: 15.0,  # 10m
    # Food
    ObjectType.FOOD: 0.3,  # 0.2m
    ObjectType.PLATE: 0.4,  # 0.3m diameter
    ObjectType.CUP: 0.2,  # 0.15m height
}


# =============================================================================
# Enhanced Sample Metadata Models
# =============================================================================
# These optional metadata models enrich canonical samples with additional
# information that can improve motion generation quality by providing
# richer conditioning signals during training and inference.


class MotionStyleMetadata(BaseModel):
    """Motion style characteristics computed from the motion data.

    These metrics help distinguish between different types of motion
    (e.g., athletic vs relaxed, smooth vs jerky) and can be used as
    conditioning signals for style-aware generation.

    Attributes:
        tempo: Motion speed/rhythm normalized to 0-1.
               0.0 = very slow (tai chi), 0.5 = walking, 1.0 = sprinting
        energy_level: Overall motion intensity normalized to 0-1.
                     0.0 = idle/resting, 0.5 = moderate activity, 1.0 = intense
        smoothness: Motion fluidity normalized to 0-1.
                   0.0 = jerky/mechanical, 0.5 = natural, 1.0 = very fluid
    """

    tempo: float | None = None
    energy_level: float | None = None
    smoothness: float | None = None


class SubjectMetadata(BaseModel):
    """Subject/performer demographic information.

    Extracted from SMPL body shape parameters or dataset annotations.
    Useful for body-aware motion generation.

    Attributes:
        height_cm: Estimated height in centimeters (typically 150-200)
        gender: Subject gender if known ("male", "female", "unknown")
        age_group: Age category ("child", "adult", "elderly", "unknown")
    """

    height_cm: float | None = None
    gender: str | None = None  # "male", "female", "unknown"
    age_group: str | None = None  # "child", "adult", "elderly", "unknown"


class MusicMetadata(BaseModel):
    """Music/rhythm information for dance sequences.

    Primarily used for AIST++ dataset to enable music-synchronized
    motion generation.

    Attributes:
        bpm: Beats per minute of the accompanying music
        beat_frames: List of frame indices where beats occur
        genre: Music genre (e.g., "break", "pop", "lock", "middle_hip_hop")
    """

    bpm: float | None = None
    beat_frames: list[int] | None = None
    genre: str | None = None


class InteractionMetadata(BaseModel):
    """Multi-actor interaction information.

    Used for InterHuman and other multi-person datasets to capture
    the relationship and coordination between actors.

    Attributes:
        contact_frames: Frame indices where actors are in contact/proximity
        interaction_role: Role in the interaction ("leader", "follower", "symmetric")
        interaction_type: Type of interaction (e.g., "handshake", "dance", "fight")
    """

    contact_frames: list[int] | None = None
    interaction_role: str | None = None  # "leader", "follower", "symmetric"
    interaction_type: str | None = None


class TemporalMetadata(BaseModel):
    """Original temporal information before resampling/padding.

    Preserves source timing information that may be lost during
    preprocessing. Useful for understanding motion pacing.

    Attributes:
        original_fps: Original frame rate of the source data
        original_duration_sec: Original duration in seconds
        original_num_frames: Original number of frames before processing
    """

    original_fps: int | None = None
    original_duration_sec: float | None = None
    original_num_frames: int | None = None


class QualityMetadata(BaseModel):
    """Data quality metrics for the motion sequence.

    Used to weight samples during training or filter low-quality data.

    Attributes:
        reconstruction_confidence: MoCap reconstruction quality (0-1)
        marker_quality: Joint position noise/jitter quality (0-1, higher=cleaner)
        physics_score: Physics validation score from DataValidator
    """

    reconstruction_confidence: float | None = None
    marker_quality: float | None = None
    physics_score: float | None = None


class EmotionMetadata(BaseModel):
    """Emotional characteristics of the motion.

    Inferred from text descriptions or motion patterns. Uses the
    valence-arousal model common in affective computing.

    Attributes:
        emotion_label: Primary emotion label from FacialExpression enum
        valence: Pleasantness dimension (-1.0 negative to 1.0 positive)
        arousal: Intensity/energy dimension (0.0 calm to 1.0 excited)
    """

    emotion_label: str | None = None  # From FacialExpression values
    valence: float | None = None  # -1.0 to 1.0
    arousal: float | None = None  # 0.0 to 1.0


class EnhancedSampleMetadata(BaseModel):
    """Container for all enhanced metadata fields.

    This unified container holds all optional metadata models.
    All fields are optional to maintain backward compatibility
    with existing samples that lack enhanced metadata.

    Usage in canonical samples:
        sample = {
            "motion": motion_tensor,
            "physics": physics_tensor,
            ...
            "enhanced_meta": EnhancedSampleMetadata(
                motion_style=MotionStyleMetadata(tempo=0.5, energy_level=0.7),
                temporal=TemporalMetadata(original_fps=120, original_duration_sec=5.2),
            ).model_dump()
        }
    """

    motion_style: MotionStyleMetadata | None = None
    subject: SubjectMetadata | None = None
    music: MusicMetadata | None = None
    interaction: InteractionMetadata | None = None
    temporal: TemporalMetadata | None = None
    quality: QualityMetadata | None = None
    emotion: EmotionMetadata | None = None


class MotionSample(BaseModel):
	"""Canonical v3 motion sample for a single actor sequence.

	Motion is encoded using the v3 12-segment stick-figure schema with
	48 floats per frame representing 12 segments of the form
	[x1, y1, x2, y2].
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	motion: torch.Tensor
	physics: torch.Tensor | None = None
	actions: torch.Tensor | None = None
	camera: torch.Tensor | None = None
	description: str
	source: str
	meta: dict[str, Any]
	enhanced_meta: EnhancedSampleMetadata | None = None

	# Schema metadata for downstream consumers (exporter, renderer, models)
	skeleton_type: str = "stick_figure_12_segment_v3"
	input_dim: int = 48
