import random
import re
from .schema import (
    Scene, Actor, Position, ActionType, SceneObject, ObjectType, ActorType,
    FacialExpression, MouthShape, FaceFeatures
)

# Action-to-Expression mapping (Phase 5)
# Maps each action type to an appropriate facial expression
ACTION_EXPRESSIONS = {
    # Basic actions - mostly neutral
    ActionType.IDLE: FacialExpression.NEUTRAL,
    ActionType.WALK: FacialExpression.NEUTRAL,
    ActionType.RUN: FacialExpression.NEUTRAL,
    ActionType.SPRINT: FacialExpression.EXCITED,
    ActionType.JUMP: FacialExpression.EXCITED,

    # Social actions - happy/friendly
    ActionType.WAVE: FacialExpression.HAPPY,
    ActionType.TALK: FacialExpression.NEUTRAL,
    ActionType.SHOUT: FacialExpression.EXCITED,    # Phase 7: Speech animation
    ActionType.WHISPER: FacialExpression.NEUTRAL,  # Phase 7: Speech animation
    ActionType.SING: FacialExpression.HAPPY,       # Phase 7: Speech animation
    ActionType.POINT: FacialExpression.NEUTRAL,
    ActionType.CLAP: FacialExpression.HAPPY,

    # Sports actions - excited/focused
    ActionType.BATTING: FacialExpression.NEUTRAL,
    ActionType.PITCHING: FacialExpression.NEUTRAL,
    ActionType.CATCHING: FacialExpression.SURPRISED,
    ActionType.RUNNING_BASES: FacialExpression.EXCITED,
    ActionType.FIELDING: FacialExpression.NEUTRAL,
    ActionType.THROWING: FacialExpression.NEUTRAL,
    ActionType.KICKING: FacialExpression.NEUTRAL,
    ActionType.DRIBBLING: FacialExpression.NEUTRAL,
    ActionType.SHOOTING: FacialExpression.EXCITED,

    # Combat actions - angry/intense
    ActionType.FIGHT: FacialExpression.ANGRY,
    ActionType.PUNCH: FacialExpression.ANGRY,
    ActionType.KICK: FacialExpression.ANGRY,
    ActionType.DODGE: FacialExpression.SURPRISED,

    # Narrative actions - context-dependent
    ActionType.SIT: FacialExpression.NEUTRAL,
    ActionType.STAND: FacialExpression.NEUTRAL,
    ActionType.KNEEL: FacialExpression.SAD,
    ActionType.LIE_DOWN: FacialExpression.NEUTRAL,
    ActionType.EATING: FacialExpression.HAPPY,
    ActionType.DRINKING: FacialExpression.HAPPY,
    ActionType.READING: FacialExpression.NEUTRAL,
    ActionType.TYPING: FacialExpression.NEUTRAL,

    # Exploration actions - curious/surprised
    ActionType.LOOKING_AROUND: FacialExpression.SURPRISED,
    ActionType.CLIMBING: FacialExpression.NEUTRAL,
    ActionType.CRAWLING: FacialExpression.NEUTRAL,
    ActionType.SWIMMING: FacialExpression.NEUTRAL,
    ActionType.FLYING: FacialExpression.EXCITED,

    # Emotional actions - explicit emotions
    ActionType.CELEBRATE: FacialExpression.EXCITED,
    ActionType.DANCE: FacialExpression.HAPPY,
    ActionType.CRY: FacialExpression.SAD,
    ActionType.LAUGH: FacialExpression.HAPPY,
}

# Expression-to-FaceFeatures mapping
# Defines the visual parameters for each expression
EXPRESSION_FEATURES = {
    FacialExpression.NEUTRAL: FaceFeatures(
        expression=FacialExpression.NEUTRAL,
        eye_type="dots",
        eyebrow_angle=0.0,
        mouth_shape=MouthShape.CLOSED,
        mouth_openness=0.0
    ),
    FacialExpression.HAPPY: FaceFeatures(
        expression=FacialExpression.HAPPY,
        eye_type="curves",
        eyebrow_angle=10.0,
        mouth_shape=MouthShape.SMILE,
        mouth_openness=0.0
    ),
    FacialExpression.SAD: FaceFeatures(
        expression=FacialExpression.SAD,
        eye_type="dots",
        eyebrow_angle=-15.0,
        mouth_shape=MouthShape.FROWN,
        mouth_openness=0.0
    ),
    FacialExpression.SURPRISED: FaceFeatures(
        expression=FacialExpression.SURPRISED,
        eye_type="wide",
        eyebrow_angle=20.0,
        mouth_shape=MouthShape.OPEN,
        mouth_openness=0.5
    ),
    FacialExpression.ANGRY: FaceFeatures(
        expression=FacialExpression.ANGRY,
        eye_type="dots",
        eyebrow_angle=-20.0,
        mouth_shape=MouthShape.FROWN,
        mouth_openness=0.0
    ),
    FacialExpression.EXCITED: FaceFeatures(
        expression=FacialExpression.EXCITED,
        eye_type="wide",
        eyebrow_angle=15.0,
        mouth_shape=MouthShape.SMILE,
        mouth_openness=0.3
    ),
}

# Speech animation configuration (Phase 7)
# Maps speech actions to their animation parameters
SPEECH_ANIMATION_CONFIG = {
    ActionType.TALK: {
        'cycle_speed': 8.0,      # 8 Hz - normal talking speed
        'mouth_shapes': [MouthShape.SMALL_O, MouthShape.OPEN, MouthShape.CLOSED],
        'openness_range': (0.2, 0.5),  # Moderate mouth opening
    },
    ActionType.SHOUT: {
        'cycle_speed': 6.0,      # 6 Hz - slower, more emphatic
        'mouth_shapes': [MouthShape.WIDE_OPEN, MouthShape.OPEN],
        'openness_range': (0.6, 1.0),  # Wide mouth opening
    },
    ActionType.WHISPER: {
        'cycle_speed': 10.0,     # 10 Hz - faster, subtle movements
        'mouth_shapes': [MouthShape.SMALL_O, MouthShape.CLOSED],
        'openness_range': (0.1, 0.3),  # Small mouth opening
    },
    ActionType.SING: {
        'cycle_speed': 4.0,      # 4 Hz - slower, sustained notes
        'mouth_shapes': [MouthShape.SINGING, MouthShape.OPEN, MouthShape.SMALL_O],
        'openness_range': (0.4, 0.8),  # Varied mouth opening
    },
}

def create_actor_with_expression(
    actor_id: str,
    position: Position,
    actions: list,
    color: str = "black",
    actor_type: ActorType = ActorType.HUMAN,
    **kwargs
) -> Actor:
    """
    Helper function to create an Actor with appropriate facial expression.

    Automatically sets facial expression based on the first action in the action list.

    Args:
        actor_id: Unique identifier for the actor
        position: Initial position
        actions: List of (time, ActionType) tuples
        color: Actor color (default: "black")
        actor_type: Type of actor (default: HUMAN)
        **kwargs: Additional Actor parameters

    Returns:
        Actor with facial expression set
    """
    # Determine expression from first action
    if actions and len(actions) > 0:
        first_action = actions[0][1] if isinstance(actions[0], tuple) else actions[0]
        expression = ACTION_EXPRESSIONS.get(first_action, FacialExpression.NEUTRAL)
    else:
        expression = FacialExpression.NEUTRAL

    # Get face features for this expression
    face_features = EXPRESSION_FEATURES[expression]

    return Actor(
        id=actor_id,
        actor_type=actor_type,
        color=color,
        initial_position=position,
        actions=actions,
        facial_expression=expression,
        face_features=face_features,
        **kwargs
    )

class StoryGenerator:
    def __init__(self):
        pass

    def generate_random_scene(self) -> Scene:
        themes = ["nature", "city", "sports_baseball", "sports_soccer", "tech", "space", "narrative"]
        theme = random.choice(themes)
        return self._generate_scene_for_theme(theme)

    def generate_scene_from_prompt(self, prompt: str) -> Scene:
        """Parse prompt and generate appropriate scene"""
        prompt_lower = prompt.lower()

        # Detect theme
        theme = self._detect_theme(prompt_lower)

        # Detect number of actors
        num_actors = self._detect_actor_count(prompt_lower, theme)

        # Detect narrative elements
        narrative_beats = self._detect_narrative_beats(prompt_lower)

        return self._generate_scene_for_theme(
            theme,
            description_override=prompt,
            num_actors=num_actors,
            narrative_beats=narrative_beats
        )

    def _detect_theme(self, prompt: str) -> str:
        """Detect theme from prompt"""
        if any(word in prompt for word in ["baseball", "world series", "playoff", "batting", "pitching"]):
            return "sports_baseball"
        elif any(word in prompt for word in ["soccer", "football", "goal", "kick"]):
            return "sports_soccer"
        elif any(word in prompt for word in ["basketball", "dunk", "hoop"]):
            return "sports_basketball"
        elif any(word in prompt for word in ["space", "alien", "planet", "astronaut", "ufo"]):
            return "space"
        elif any(word in prompt for word in ["nature", "tree", "forest", "park"]):
            return "nature"
        elif any(word in prompt for word in ["city", "building", "urban", "street"]):
            return "city"
        elif any(word in prompt for word in ["tech", "laptop", "computer", "coding"]):
            return "tech"
        elif any(word in prompt for word in ["meet", "eat", "talk", "explore", "story"]):
            return "narrative"
        else:
            return random.choice(["nature", "city", "sports_baseball", "tech"])

    def _detect_actor_count(self, prompt: str, theme: str) -> int:
        """Detect number of actors from prompt"""
        # Check for explicit numbers
        numbers = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', prompt)
        if numbers:
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            for num in numbers:
                if num.isdigit():
                    return min(int(num), 20)
                elif num in word_to_num:
                    return word_to_num[num]

        # Check for team keywords
        if "teams" in prompt or "team" in prompt:
            if "baseball" in prompt:
                return 18  # 9 per team
            else:
                return 10  # 5 per team for other sports

        # Check for group keywords
        if any(word in prompt for word in ["crowd", "group", "many", "several"]):
            return random.randint(8, 15)

        # Check for multi-character indicators
        if any(word in prompt for word in ["alien", "aliens", "robot", "robots", "meet", "meets", "with them", "together"]):
            return max(2, random.randint(2, 4))  # At least 2 for interactions

        # Theme-based defaults
        if theme.startswith("sports_"):
            return random.randint(6, 12)
        elif theme == "narrative":
            return random.randint(2, 4)
        elif theme == "space":
            return random.randint(2, 4)  # Space scenes usually have multiple characters
        else:
            return random.randint(1, 3)

    def _detect_narrative_beats(self, prompt: str) -> list:
        """Detect narrative sequence from prompt"""
        beats = []

        # Common narrative patterns
        if "explore" in prompt or "exploring" in prompt:
            beats.append(("exploring", 0.0, 2.0))
        if "meet" in prompt or "meets" in prompt:
            beats.append(("meeting", 2.0, 4.0))
        if "eat" in prompt or "eats" in prompt or "meal" in prompt:
            beats.append(("eating", 4.0, 7.0))
        if "talk" in prompt or "conversation" in prompt:
            beats.append(("talking", 2.0, 5.0))
        if "fight" in prompt or "battle" in prompt:
            beats.append(("fighting", 1.0, 4.0))
        if "dance" in prompt or "dancing" in prompt:
            beats.append(("dancing", 0.0, 5.0))

        return beats

    def _generate_scene_for_theme(self, theme: str, description_override: str = None,
                                   num_actors: int = None, narrative_beats: list = None) -> Scene:
        """Generate scene based on theme with enhanced multi-actor support"""

        if num_actors is None:
            num_actors = random.randint(1, 3)

        # Adjust duration based on narrative complexity
        if narrative_beats and len(narrative_beats) > 0:
            max_beat_time = max(beat[2] for beat in narrative_beats)
            duration = max(max_beat_time + 1.0, 5.0)
        else:
            duration = random.uniform(3.0, 7.0)

        actors = []
        objects = []

        # Generate scene based on theme
        if theme == "sports_baseball":
            actors, objects, description = self._generate_baseball_scene(num_actors, duration)
        elif theme == "sports_soccer":
            actors, objects, description = self._generate_soccer_scene(num_actors, duration)
        elif theme == "space":
            actors, objects, description = self._generate_space_scene(num_actors, duration, narrative_beats)
        elif theme == "narrative":
            actors, objects, description = self._generate_narrative_scene(num_actors, duration, narrative_beats)
        elif theme == "nature":
            actors, objects, description = self._generate_nature_scene(num_actors, duration)
        elif theme == "city":
            actors, objects, description = self._generate_city_scene(num_actors, duration)
        elif theme == "tech":
            actors, objects, description = self._generate_tech_scene(num_actors, duration)
        else:
            actors, objects, description = self._generate_generic_scene(num_actors, duration)

        if description_override:
            description = description_override

        return Scene(
            duration=duration,
            actors=actors,
            objects=objects,
            description=description,
            theme=theme
        )

    def _generate_baseball_scene(self, num_actors: int, duration: float):
        """Generate baseball game scene with teams"""
        actors = []
        objects = []
        colors_team1 = ["red", "darkred"]
        colors_team2 = ["blue", "darkblue"]

        # Create baseball diamond
        for i, base_name in enumerate(["home", "first", "second", "third"]):
            angle = i * 90
            x = 2.5 * (1 if i % 2 == 1 else -1) if i > 0 else 0
            y = -2.5 if i == 0 else -2.5 + 2.5 * (i // 2)
            objects.append(SceneObject(
                id=f"base_{base_name}",
                type=ObjectType.BASE,
                position=Position(x=x, y=y),
                color="white",
                scale=0.3
            ))

        # Add baseball
        objects.append(SceneObject(
            id="baseball",
            type=ObjectType.BASEBALL,
            position=Position(x=0, y=-2.0),
            color="white",
            scale=0.2,
            velocity=(random.uniform(-1, 1), random.uniform(0.5, 2.0))
        ))

        # Ensure we have enough actors for teams
        num_actors = max(num_actors, 10)
        team_size = num_actors // 2

        # Team 1 (batting team) - positioned near home plate and bases
        for i in range(team_size):
            if i == 0:
                # Batter - runs the bases after hitting
                pos = Position(x=-0.5, y=-2.5)
                action_seq = [(0.0, ActionType.IDLE), (1.0, ActionType.BATTING), (2.0, ActionType.RUNNING_BASES)]

                # Movement path: home → 1st → 2nd → 3rd → home
                movement_path = [
                    (0.0, Position(x=-0.5, y=-2.5)),   # Home (start)
                    (2.0, Position(x=-0.5, y=-2.5)),   # Stay at home during batting
                    (3.5, Position(x=2.5, y=-2.5)),    # Run to 1st base
                    (5.0, Position(x=2.5, y=0.0)),     # Run to 2nd base
                    (6.5, Position(x=-2.5, y=0.0)),    # Run to 3rd base
                    (8.0, Position(x=-0.5, y=-2.5)),   # Run back to home
                ]
            else:
                # Other batters waiting or on bases
                pos = Position(x=-4 + i * 0.5, y=-3.0)
                action_seq = [(0.0, ActionType.IDLE), (3.0 + i * 0.5, ActionType.WALK)]

                # Simple walking movement
                movement_path = [
                    (0.0, Position(x=-4 + i * 0.5, y=-3.0)),
                    (3.0 + i * 0.5, Position(x=-4 + i * 0.5, y=-3.0)),
                    (duration, Position(x=-3 + i * 0.5, y=-2.5)),
                ]

            actors.append(Actor(
                id=f"team1_player_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors_team1),
                initial_position=pos,
                actions=action_seq,
                team="team1",
                movement_path=movement_path
            ))

        # Team 2 (fielding team) - positioned in field
        for i in range(team_size):
            if i == 0:
                # Pitcher
                pos = Position(x=0, y=-1.0)
                action_seq = [(0.0, ActionType.IDLE), (0.5, ActionType.PITCHING), (2.5, ActionType.FIELDING)]
            elif i == 1:
                # Catcher
                pos = Position(x=0.5, y=-2.7)
                action_seq = [(0.0, ActionType.IDLE), (1.5, ActionType.CATCHING)]
            else:
                # Fielders
                angle = (i - 2) * 40 - 60
                distance = random.uniform(3.0, 4.5)
                pos = Position(
                    x=distance * (0.5 if i % 2 == 0 else -0.5),
                    y=-1.0 + distance * 0.3
                )
                action_seq = [(0.0, ActionType.IDLE), (2.0 + i * 0.2, ActionType.FIELDING), (3.0, ActionType.THROWING)]

            actors.append(Actor(
                id=f"team2_player_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors_team2),
                initial_position=pos,
                actions=action_seq,
                team="team2"
            ))

        description = f"A baseball game with {num_actors} players on two teams"
        return actors, objects, description

    def _generate_space_scene(self, num_actors: int, duration: float, narrative_beats: list = None):
        """Generate space exploration scene"""
        actors = []
        objects = []

        # Add space objects
        objects.append(SceneObject(
            id="planet",
            type=ObjectType.PLANET,
            position=Position(x=3.5, y=2.0),
            color="purple",
            scale=1.5
        ))

        objects.append(SceneObject(
            id="spaceship",
            type=ObjectType.SPACESHIP,
            position=Position(x=-3.0, y=1.0),
            color="silver",
            scale=1.0
        ))

        # Add stars
        for i in range(5):
            objects.append(SceneObject(
                id=f"star_{i}",
                type=ObjectType.STAR,
                position=Position(x=random.uniform(-4, 4), y=random.uniform(0, 4)),
                color="yellow",
                scale=0.2
            ))

        # Human explorer
        human_actions = [(0.0, ActionType.WALK), (2.0, ActionType.LOOKING_AROUND)]
        if narrative_beats:
            for beat_name, start, end in narrative_beats:
                if "meeting" in beat_name:
                    human_actions.append((start, ActionType.WAVE))
                    human_actions.append((start + 0.5, ActionType.TALK))
                elif "eating" in beat_name:
                    human_actions.append((start, ActionType.SIT))
                    human_actions.append((start + 0.5, ActionType.EATING))

        # Movement path: exploring space, walking toward aliens/planet
        human_movement_path = [
            (0.0, Position(x=-2.0, y=-2.0)),    # Start at spaceship
            (2.0, Position(x=0.0, y=-1.5)),     # Walk to center
            (4.0, Position(x=1.5, y=-1.5)),     # Walk toward aliens
        ]
        if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
            human_movement_path.append((5.0, Position(x=0.0, y=-1.5)))  # Move to table

        actors.append(Actor(
            id="human_explorer",
            actor_type=ActorType.HUMAN,
            color="blue",
            initial_position=Position(x=-2.0, y=-2.0),
            actions=human_actions,
            movement_path=human_movement_path
        ))

        # Alien(s)
        num_aliens = min(num_actors - 1, 3)
        for i in range(num_aliens):
            alien_actions = [(0.0, ActionType.IDLE), (2.5, ActionType.WALK)]
            if narrative_beats:
                for beat_name, start, end in narrative_beats:
                    if "meeting" in beat_name:
                        alien_actions.append((start + 0.3, ActionType.WAVE))
                        alien_actions.append((start + 0.8, ActionType.TALK))
                    elif "eating" in beat_name:
                        alien_actions.append((start, ActionType.SIT))
                        alien_actions.append((start + 0.5, ActionType.EATING))

            # Alien movement: walk toward human
            alien_movement_path = [
                (0.0, Position(x=1.0 + i * 0.8, y=-2.0)),   # Start position
                (2.5, Position(x=1.0 + i * 0.8, y=-2.0)),   # Stay idle
                (4.0, Position(x=0.5 + i * 0.5, y=-1.5)),   # Walk toward human
            ]
            if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
                alien_movement_path.append((5.0, Position(x=0.3 + i * 0.4, y=-1.5)))  # Move to table

            actors.append(Actor(
                id=f"alien_{i}",
                actor_type=ActorType.ALIEN,
                color="green",
                initial_position=Position(x=1.0 + i * 0.8, y=-2.0),
                actions=alien_actions,
                scale=1.2,  # Aliens slightly larger
                movement_path=alien_movement_path
            ))

        # Add table and food if eating scene
        if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
            objects.append(SceneObject(
                id="table",
                type=ObjectType.TABLE,
                position=Position(x=0, y=-1.5),
                color="brown",
                scale=1.0
            ))
            objects.append(SceneObject(
                id="food",
                type=ObjectType.FOOD,
                position=Position(x=0, y=-0.8),
                color="orange",
                scale=0.5
            ))

        description = f"A space scene with {num_actors} characters"
        return actors, objects, description

    def _generate_narrative_scene(self, num_actors: int, duration: float, narrative_beats: list = None):
        """Generate general narrative scene"""
        actors = []
        objects = []
        colors = ["black", "red", "blue", "green", "purple"]

        for i in range(num_actors):
            actions = [(0.0, ActionType.IDLE)]

            if narrative_beats:
                for beat_name, start, end in narrative_beats:
                    if "exploring" in beat_name:
                        actions.append((start, ActionType.WALK))
                        actions.append((start + 0.5, ActionType.LOOKING_AROUND))
                    elif "meeting" in beat_name:
                        actions.append((start, ActionType.WAVE))
                        actions.append((start + 0.5, ActionType.TALK))
                    elif "eating" in beat_name:
                        actions.append((start, ActionType.SIT))
                        actions.append((start + 0.5, ActionType.EATING))
                    elif "fighting" in beat_name:
                        actions.append((start, ActionType.FIGHT))
                    elif "dancing" in beat_name:
                        actions.append((start, ActionType.DANCE))

            actors.append(Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors),
                initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                actions=actions
            ))

        description = f"A narrative scene with {num_actors} characters"
        return actors, objects, description

    def _generate_soccer_scene(self, num_actors: int, duration: float):
        """Generate soccer game scene"""
        actors = []
        objects = []

        # Add soccer ball
        objects.append(SceneObject(
            id="soccer_ball",
            type=ObjectType.SOCCER_BALL,
            position=Position(x=0, y=-2.0),
            color="white",
            scale=0.3,
            velocity=(random.uniform(-0.5, 0.5), random.uniform(0, 1.0))
        ))

        num_actors = max(num_actors, 6)
        team_size = num_actors // 2

        for i in range(team_size):
            actors.append(Actor(
                id=f"team1_player_{i}",
                actor_type=ActorType.HUMAN,
                color="red",
                initial_position=Position(x=-3 + i * 1.5, y=-2.0),
                actions=[(0.0, ActionType.RUN), (1.5, ActionType.KICKING), (3.0, ActionType.RUN)],
                team="team1"
            ))

        for i in range(team_size):
            actors.append(Actor(
                id=f"team2_player_{i}",
                actor_type=ActorType.HUMAN,
                color="blue",
                initial_position=Position(x=1 + i * 1.5, y=-2.0),
                actions=[(0.0, ActionType.RUN), (2.0, ActionType.KICKING)],
                team="team2"
            ))

        description = f"A soccer game with {num_actors} players"
        return actors, objects, description

    def _generate_nature_scene(self, num_actors: int, duration: float):
        """Generate nature scene"""
        actors = []
        objects = []

        # Add trees
        for i in range(random.randint(2, 4)):
            objects.append(SceneObject(
                id=f"tree_{i}",
                type=ObjectType.TREE,
                position=Position(x=random.uniform(-4, 4), y=random.uniform(-1, 0)),
                scale=random.uniform(0.8, 1.2)
            ))

        colors = ["black", "brown", "green"]
        for i in range(num_actors):
            action = random.choice([ActionType.WALK, ActionType.IDLE, ActionType.WAVE, ActionType.LOOKING_AROUND])
            actors.append(Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors),
                initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                actions=[(0.0, action), (2.0, ActionType.WALK)]
            ))

        description = f"A nature scene with trees and {num_actors} people"
        return actors, objects, description

    def _generate_city_scene(self, num_actors: int, duration: float):
        """Generate city scene"""
        actors = []
        objects = []

        # Add buildings
        for i in range(random.randint(1, 3)):
            objects.append(SceneObject(
                id=f"building_{i}",
                type=ObjectType.BUILDING,
                position=Position(x=random.uniform(-3, 3), y=random.uniform(-1, 0)),
                scale=random.uniform(0.8, 1.2)
            ))

        colors = ["black", "red", "blue", "gray"]
        for i in range(num_actors):
            action = random.choice([ActionType.WALK, ActionType.RUN, ActionType.TALK, ActionType.TYPING])
            actors.append(Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors),
                initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                actions=[(0.0, ActionType.WALK), (2.0, action)]
            ))

        description = f"A city scene with {num_actors} people"
        return actors, objects, description

    def _generate_tech_scene(self, num_actors: int, duration: float):
        """Generate tech scene"""
        actors = []
        objects = []

        # Add laptops
        for i in range(min(num_actors, 3)):
            objects.append(SceneObject(
                id=f"laptop_{i}",
                type=ObjectType.LAPTOP,
                position=Position(x=-2 + i * 2, y=-1.0),
                scale=1.0
            ))

        for i in range(num_actors):
            actors.append(Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                color="black",
                initial_position=Position(x=-2 + i * 1.5, y=-2.0),
                actions=[(0.0, ActionType.SIT), (0.5, ActionType.TYPING), (3.0, ActionType.TALK)]
            ))

        description = f"A tech scene with {num_actors} people working"
        return actors, objects, description

    def _generate_generic_scene(self, num_actors: int, duration: float):
        """Generate generic scene"""
        actors = []
        objects = []
        colors = ["black", "red", "blue", "green"]

        for i in range(num_actors):
            action = random.choice(list(ActionType))
            actors.append(Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                color=random.choice(colors),
                initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                actions=[(0.0, action)]
            ))

        description = f"A scene with {num_actors} stick figures"
        return actors, objects, description
