from enum import Enum

from pydantic import BaseModel


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
