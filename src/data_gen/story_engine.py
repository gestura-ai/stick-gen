import random
import re

from .schema import (
    ActionType,
    Actor,
    ActorType,
    EnvironmentType,
    FaceFeatures,
    FacialExpression,
    MouthShape,
    ObjectType,
    Position,
    Scene,
    SceneObject,
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
    ActionType.SHOUT: FacialExpression.EXCITED,  # Phase 7: Speech animation
    ActionType.WHISPER: FacialExpression.NEUTRAL,  # Phase 7: Speech animation
    ActionType.SING: FacialExpression.HAPPY,  # Phase 7: Speech animation
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
        mouth_openness=0.0,
    ),
    FacialExpression.HAPPY: FaceFeatures(
        expression=FacialExpression.HAPPY,
        eye_type="curves",
        eyebrow_angle=10.0,
        mouth_shape=MouthShape.SMILE,
        mouth_openness=0.0,
    ),
    FacialExpression.SAD: FaceFeatures(
        expression=FacialExpression.SAD,
        eye_type="dots",
        eyebrow_angle=-15.0,
        mouth_shape=MouthShape.FROWN,
        mouth_openness=0.0,
    ),
    FacialExpression.SURPRISED: FaceFeatures(
        expression=FacialExpression.SURPRISED,
        eye_type="wide",
        eyebrow_angle=20.0,
        mouth_shape=MouthShape.OPEN,
        mouth_openness=0.5,
    ),
    FacialExpression.ANGRY: FaceFeatures(
        expression=FacialExpression.ANGRY,
        eye_type="dots",
        eyebrow_angle=-20.0,
        mouth_shape=MouthShape.FROWN,
        mouth_openness=0.0,
    ),
    FacialExpression.EXCITED: FaceFeatures(
        expression=FacialExpression.EXCITED,
        eye_type="wide",
        eyebrow_angle=15.0,
        mouth_shape=MouthShape.SMILE,
        mouth_openness=0.3,
    ),
}

# Speech animation configuration (Phase 7)
# Maps speech actions to their animation parameters
SPEECH_ANIMATION_CONFIG = {
    ActionType.TALK: {
        "cycle_speed": 8.0,  # 8 Hz - normal talking speed
        "mouth_shapes": [MouthShape.SMALL_O, MouthShape.OPEN, MouthShape.CLOSED],
        "openness_range": (0.2, 0.5),  # Moderate mouth opening
    },
    ActionType.SHOUT: {
        "cycle_speed": 6.0,  # 6 Hz - slower, more emphatic
        "mouth_shapes": [MouthShape.WIDE_OPEN, MouthShape.OPEN],
        "openness_range": (0.6, 1.0),  # Wide mouth opening
    },
    ActionType.WHISPER: {
        "cycle_speed": 10.0,  # 10 Hz - faster, subtle movements
        "mouth_shapes": [MouthShape.SMALL_O, MouthShape.CLOSED],
        "openness_range": (0.1, 0.3),  # Small mouth opening
    },
    ActionType.SING: {
        "cycle_speed": 4.0,  # 4 Hz - slower, sustained notes
        "mouth_shapes": [MouthShape.SINGING, MouthShape.OPEN, MouthShape.SMALL_O],
        "openness_range": (0.4, 0.8),  # Varied mouth opening
    },
}


# Weighted actor type selection for variety
# Weights control how often each actor type appears in generated scenes
ACTOR_TYPE_WEIGHTS = {
    ActorType.HUMAN: 0.50,   # 50% humans (most common)
    ActorType.ROBOT: 0.20,   # 20% robots
    ActorType.ALIEN: 0.20,   # 20% aliens
    ActorType.ANIMAL: 0.10,  # 10% animals
}


def random_actor_type(
    weights: dict[ActorType, float] | None = None,
    exclude: list[ActorType] | None = None,
) -> ActorType:
    """
    Select a random actor type with weighted probability.

    Args:
        weights: Custom weights dict (uses ACTOR_TYPE_WEIGHTS if None)
        exclude: Actor types to exclude from selection

    Returns:
        Randomly selected ActorType
    """
    w = weights or ACTOR_TYPE_WEIGHTS

    if exclude:
        w = {k: v for k, v in w.items() if k not in exclude}

    if not w:
        return ActorType.HUMAN  # Fallback

    # Normalize weights
    total = sum(w.values())
    types = list(w.keys())
    probs = [w[t] / total for t in types]

    return random.choices(types, weights=probs, k=1)[0]


def create_actor_with_expression(
    actor_id: str,
    position: Position,
    actions: list,
    color: str = "black",
    actor_type: ActorType | None = None,
    randomize_type: bool = False,
    **kwargs,
) -> Actor:
    """
    Helper function to create an Actor with appropriate facial expression.

    Automatically sets facial expression based on the first action in the action list.

    Args:
        actor_id: Unique identifier for the actor
        position: Initial position
        actions: List of (time, ActionType) tuples
        color: Actor color (default: "black")
        actor_type: Type of actor (if None and randomize_type=True, picks randomly)
        randomize_type: If True and actor_type is None, select random actor type
        **kwargs: Additional Actor parameters

    Returns:
        Actor with facial expression set
    """
    # Determine actor type
    if actor_type is None:
        if randomize_type:
            actor_type = random_actor_type()
        else:
            actor_type = ActorType.HUMAN

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
        **kwargs,
    )


class StoryGenerator:
    def __init__(self):
        pass

    def generate_random_scene(self) -> Scene:
        # Expanded theme list with 40+ environment types
        themes = [
            # Terrestrial outdoor
            "nature", "forest", "desert", "beach", "mountain", "arctic",
            "jungle", "swamp", "cave", "volcano",
            # Urban
            "city", "rooftop", "alley", "subway", "mall",
            # Indoor
            "office", "gym", "museum", "restaurant", "theater", "library",
            "hospital", "factory", "warehouse", "classroom",
            # Aquatic
            "underwater", "ocean_surface", "river", "pool",
            # Space/Sci-Fi
            "space", "moon", "mars", "spaceship", "space_station", "alien_planet",
            # Fantasy
            "castle", "dungeon", "enchanted_forest", "cloud_realm", "lava_realm", "ice_realm",
            # Sports
            "sports_baseball", "sports_soccer", "sports_basketball", "stadium", "rink", "track",
            # Social
            "concert", "party", "wedding", "market", "festival", "parade",
            # Weather variations
            "rainy_city", "snowy_mountain", "stormy_beach",
            # Narrative
            "narrative", "tech",
        ]
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
            narrative_beats=narrative_beats,
        )

    def _detect_theme(self, prompt: str) -> str:
        """Detect theme from prompt with expanded environment detection"""
        # Sports themes
        if any(word in prompt for word in ["baseball", "world series", "playoff", "batting", "pitching"]):
            return "sports_baseball"
        elif any(word in prompt for word in ["soccer", "football match", "goal kick"]):
            return "sports_soccer"
        elif any(word in prompt for word in ["basketball", "dunk", "hoop", "court"]):
            return "sports_basketball"
        elif any(word in prompt for word in ["ice rink", "hockey", "skating", "ice skating"]):
            return "rink"
        elif any(word in prompt for word in ["track", "running race", "sprint", "marathon"]):
            return "track"
        elif any(word in prompt for word in ["stadium", "arena", "sports venue"]):
            return "stadium"

        # Space/Sci-Fi themes
        elif any(word in prompt for word in ["moon", "lunar", "moonwalk"]):
            return "moon"
        elif any(word in prompt for word in ["mars", "martian", "red planet"]):
            return "mars"
        elif any(word in prompt for word in ["spaceship", "spacecraft", "rocket interior"]):
            return "spaceship"
        elif any(word in prompt for word in ["space station", "orbital"]):
            return "space_station"
        elif any(word in prompt for word in ["alien planet", "exoplanet"]):
            return "alien_planet"
        elif any(word in prompt for word in ["space", "alien", "planet", "astronaut", "ufo"]):
            return "space"

        # Aquatic themes
        elif any(word in prompt for word in ["underwater", "diving", "scuba", "ocean floor"]):
            return "underwater"
        elif any(word in prompt for word in ["ocean", "sea surface", "boat", "ship"]):
            return "ocean_surface"
        elif any(word in prompt for word in ["river", "stream", "creek"]):
            return "river"
        elif any(word in prompt for word in ["pool", "swimming pool", "swim"]):
            return "pool"

        # Fantasy themes
        elif any(word in prompt for word in ["castle", "kingdom", "throne"]):
            return "castle"
        elif any(word in prompt for word in ["dungeon", "underground", "crypt"]):
            return "dungeon"
        elif any(word in prompt for word in ["enchanted", "magical forest", "fairy"]):
            return "enchanted_forest"
        elif any(word in prompt for word in ["cloud", "sky realm", "heaven"]):
            return "cloud_realm"
        elif any(word in prompt for word in ["lava", "volcano", "magma"]):
            return "lava_realm"
        elif any(word in prompt for word in ["ice realm", "frozen", "glacier"]):
            return "ice_realm"

        # Terrestrial outdoor themes
        elif any(word in prompt for word in ["forest", "woods", "trees"]):
            return "forest"
        elif any(word in prompt for word in ["desert", "sand dunes", "sahara"]):
            return "desert"
        elif any(word in prompt for word in ["beach", "coast", "shore", "seaside"]):
            return "beach"
        elif any(word in prompt for word in ["mountain", "climb", "summit", "peak"]):
            return "mountain"
        elif any(word in prompt for word in ["arctic", "polar", "tundra", "snow"]):
            return "arctic"
        elif any(word in prompt for word in ["jungle", "rainforest", "tropical"]):
            return "jungle"
        elif any(word in prompt for word in ["swamp", "marsh", "bog"]):
            return "swamp"
        elif any(word in prompt for word in ["cave", "cavern", "grotto"]):
            return "cave"
        elif any(word in prompt for word in ["volcano", "volcanic"]):
            return "volcano"
        elif any(word in prompt for word in ["nature", "tree", "park"]):
            return "nature"

        # Urban themes
        elif any(word in prompt for word in ["rooftop", "roof"]):
            return "rooftop"
        elif any(word in prompt for word in ["alley", "backstreet"]):
            return "alley"
        elif any(word in prompt for word in ["subway", "metro", "underground train"]):
            return "subway"
        elif any(word in prompt for word in ["mall", "shopping center"]):
            return "mall"
        elif any(word in prompt for word in ["city", "building", "urban", "street"]):
            return "city"

        # Indoor themes
        elif any(word in prompt for word in ["office", "cubicle", "desk"]):
            return "office"
        elif any(word in prompt for word in ["gym", "workout", "fitness"]):
            return "gym"
        elif any(word in prompt for word in ["museum", "gallery", "exhibit"]):
            return "museum"
        elif any(word in prompt for word in ["restaurant", "dining", "cafe"]):
            return "restaurant"
        elif any(word in prompt for word in ["theater", "theatre", "stage", "performance"]):
            return "theater"
        elif any(word in prompt for word in ["library", "books", "reading"]):
            return "library"
        elif any(word in prompt for word in ["hospital", "clinic", "medical"]):
            return "hospital"
        elif any(word in prompt for word in ["factory", "manufacturing", "assembly"]):
            return "factory"
        elif any(word in prompt for word in ["warehouse", "storage"]):
            return "warehouse"
        elif any(word in prompt for word in ["classroom", "school", "lecture"]):
            return "classroom"

        # Social themes
        elif any(word in prompt for word in ["concert", "live music", "band"]):
            return "concert"
        elif any(word in prompt for word in ["party", "celebration"]):
            return "party"
        elif any(word in prompt for word in ["wedding", "ceremony", "bride"]):
            return "wedding"
        elif any(word in prompt for word in ["market", "bazaar", "stall"]):
            return "market"
        elif any(word in prompt for word in ["festival", "fair"]):
            return "festival"
        elif any(word in prompt for word in ["parade", "march", "procession"]):
            return "parade"

        # Weather combinations
        elif any(word in prompt for word in ["rainy", "rain"]) and "city" in prompt:
            return "rainy_city"
        elif any(word in prompt for word in ["snowy", "blizzard"]) and any(word in prompt for word in ["mountain", "peak"]):
            return "snowy_mountain"
        elif any(word in prompt for word in ["storm", "stormy"]) and any(word in prompt for word in ["beach", "coast"]):
            return "stormy_beach"

        # Tech and narrative
        elif any(word in prompt for word in ["tech", "laptop", "computer", "coding"]):
            return "tech"
        elif any(word in prompt for word in ["meet", "eat", "talk", "explore", "story"]):
            return "narrative"

        # Default: random from common themes
        else:
            return random.choice([
                "nature", "city", "forest", "beach", "office", "party",
                "space", "castle", "gym", "restaurant"
            ])

    def _detect_actor_count(self, prompt: str, theme: str) -> int:
        """Detect number of actors from prompt"""
        # Check for explicit numbers
        numbers = re.findall(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", prompt
        )
        if numbers:
            word_to_num = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
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
        if any(
            word in prompt
            for word in [
                "alien",
                "aliens",
                "robot",
                "robots",
                "meet",
                "meets",
                "with them",
                "together",
            ]
        ):
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

    # Theme to environment type mapping
    THEME_TO_ENVIRONMENT = {
        # Terrestrial outdoor
        "nature": EnvironmentType.GRASSLAND,
        "forest": EnvironmentType.FOREST,
        "desert": EnvironmentType.DESERT,
        "beach": EnvironmentType.BEACH,
        "mountain": EnvironmentType.MOUNTAIN,
        "arctic": EnvironmentType.ARCTIC,
        "jungle": EnvironmentType.JUNGLE,
        "swamp": EnvironmentType.SWAMP,
        "cave": EnvironmentType.CAVE,
        "volcano": EnvironmentType.VOLCANO,
        # Urban
        "city": EnvironmentType.CITY_STREET,
        "rooftop": EnvironmentType.ROOFTOP,
        "alley": EnvironmentType.ALLEY,
        "subway": EnvironmentType.SUBWAY,
        "mall": EnvironmentType.MALL,
        # Indoor
        "office": EnvironmentType.OFFICE,
        "gym": EnvironmentType.GYM,
        "museum": EnvironmentType.MUSEUM,
        "restaurant": EnvironmentType.RESTAURANT,
        "theater": EnvironmentType.THEATER,
        "library": EnvironmentType.LIBRARY,
        "hospital": EnvironmentType.HOSPITAL,
        "factory": EnvironmentType.FACTORY,
        "warehouse": EnvironmentType.WAREHOUSE,
        "classroom": EnvironmentType.CLASSROOM,
        # Aquatic
        "underwater": EnvironmentType.UNDERWATER,
        "ocean_surface": EnvironmentType.OCEAN_SURFACE,
        "river": EnvironmentType.RIVER,
        "pool": EnvironmentType.POOL,
        # Space/Sci-Fi
        "space": EnvironmentType.SPACE_VACUUM,
        "moon": EnvironmentType.MOON,
        "mars": EnvironmentType.MARS,
        "spaceship": EnvironmentType.SPACESHIP_INTERIOR,
        "space_station": EnvironmentType.SPACE_STATION,
        "alien_planet": EnvironmentType.ALIEN_PLANET_LOW_G,
        # Fantasy
        "castle": EnvironmentType.CASTLE,
        "dungeon": EnvironmentType.DUNGEON,
        "enchanted_forest": EnvironmentType.ENCHANTED_FOREST,
        "cloud_realm": EnvironmentType.CLOUD_REALM,
        "lava_realm": EnvironmentType.LAVA_REALM,
        "ice_realm": EnvironmentType.ICE_REALM,
        # Sports
        "sports_baseball": EnvironmentType.FIELD,
        "sports_soccer": EnvironmentType.FIELD,
        "sports_basketball": EnvironmentType.COURT,
        "stadium": EnvironmentType.STADIUM,
        "rink": EnvironmentType.RINK,
        "track": EnvironmentType.TRACK,
        # Social
        "concert": EnvironmentType.CONCERT,
        "party": EnvironmentType.PARTY,
        "wedding": EnvironmentType.WEDDING,
        "market": EnvironmentType.MARKET,
        "festival": EnvironmentType.FESTIVAL,
        "parade": EnvironmentType.PARADE,
        # Weather combinations
        "rainy_city": EnvironmentType.CITY_STREET,
        "snowy_mountain": EnvironmentType.MOUNTAIN,
        "stormy_beach": EnvironmentType.BEACH,
        # Tech and narrative
        "tech": EnvironmentType.OFFICE,
        "narrative": EnvironmentType.EARTH_NORMAL,
    }

    # Weather overlays for weather-themed scenes
    THEME_TO_WEATHER = {
        "rainy_city": EnvironmentType.RAINY,
        "snowy_mountain": EnvironmentType.SNOWY,
        "stormy_beach": EnvironmentType.STORMY,
    }

    def _generate_scene_for_theme(
        self,
        theme: str,
        description_override: str = None,
        num_actors: int = None,
        narrative_beats: list = None,
    ) -> Scene:
        """Generate scene based on theme with enhanced multi-actor support and environment physics"""

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

        # Get environment type and weather from theme
        environment_type = self.THEME_TO_ENVIRONMENT.get(theme, EnvironmentType.EARTH_NORMAL)
        weather_type = self.THEME_TO_WEATHER.get(theme, None)

        # Generate scene based on theme - route to appropriate generator
        if theme == "sports_baseball":
            actors, objects, description = self._generate_baseball_scene(num_actors, duration)
        elif theme == "sports_soccer":
            actors, objects, description = self._generate_soccer_scene(num_actors, duration)
        elif theme == "sports_basketball":
            actors, objects, description = self._generate_basketball_scene(num_actors, duration)
        elif theme in ("stadium", "rink", "track"):
            actors, objects, description = self._generate_sports_venue_scene(num_actors, duration, theme)
        elif theme == "space":
            actors, objects, description = self._generate_space_scene(num_actors, duration, narrative_beats)
        elif theme in ("moon", "mars", "spaceship", "space_station", "alien_planet"):
            actors, objects, description = self._generate_scifi_scene(num_actors, duration, theme)
        elif theme == "underwater":
            actors, objects, description = self._generate_underwater_scene(num_actors, duration)
        elif theme in ("ocean_surface", "river", "pool"):
            actors, objects, description = self._generate_aquatic_scene(num_actors, duration, theme)
        elif theme in ("castle", "dungeon", "enchanted_forest", "cloud_realm", "lava_realm", "ice_realm"):
            actors, objects, description = self._generate_fantasy_scene(num_actors, duration, theme)
        elif theme in ("forest", "jungle", "swamp", "cave", "volcano"):
            actors, objects, description = self._generate_wilderness_scene(num_actors, duration, theme)
        elif theme in ("desert", "beach", "mountain", "arctic"):
            actors, objects, description = self._generate_terrain_scene(num_actors, duration, theme)
        elif theme in ("office", "gym", "museum", "restaurant", "theater", "library", "hospital", "factory", "warehouse", "classroom"):
            actors, objects, description = self._generate_indoor_scene(num_actors, duration, theme)
        elif theme in ("rooftop", "alley", "subway", "mall"):
            actors, objects, description = self._generate_urban_scene(num_actors, duration, theme)
        elif theme in ("concert", "party", "wedding", "market", "festival", "parade"):
            actors, objects, description = self._generate_social_scene(num_actors, duration, theme)
        elif theme in ("rainy_city", "snowy_mountain", "stormy_beach"):
            actors, objects, description = self._generate_weather_scene(num_actors, duration, theme)
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
            theme=theme,
            environment_type=environment_type,
            weather_type=weather_type,
        )

    def _generate_baseball_scene(self, num_actors: int, duration: float):
        """Generate baseball game scene with teams"""
        actors = []
        objects = []
        colors_team1 = ["red", "darkred"]
        colors_team2 = ["blue", "darkblue"]

        # Create baseball diamond
        for i, base_name in enumerate(["home", "first", "second", "third"]):
            i * 90
            x = 2.5 * (1 if i % 2 == 1 else -1) if i > 0 else 0
            y = -2.5 if i == 0 else -2.5 + 2.5 * (i // 2)
            objects.append(
                SceneObject(
                    id=f"base_{base_name}",
                    type=ObjectType.BASE,
                    position=Position(x=x, y=y),
                    color="white",
                    scale=0.3,
                )
            )

        # Add baseball
        objects.append(
            SceneObject(
                id="baseball",
                type=ObjectType.BASEBALL,
                position=Position(x=0, y=-2.0),
                color="white",
                scale=0.2,
                velocity=(random.uniform(-1, 1), random.uniform(0.5, 2.0)),
            )
        )

        # Ensure we have enough actors for teams
        num_actors = max(num_actors, 10)
        team_size = num_actors // 2

        # Team 1 (batting team) - positioned near home plate and bases
        for i in range(team_size):
            if i == 0:
                # Batter - runs the bases after hitting
                pos = Position(x=-0.5, y=-2.5)
                action_seq = [
                    (0.0, ActionType.IDLE),
                    (1.0, ActionType.BATTING),
                    (2.0, ActionType.RUNNING_BASES),
                ]

                # Movement path: home → 1st → 2nd → 3rd → home
                movement_path = [
                    (0.0, Position(x=-0.5, y=-2.5)),  # Home (start)
                    (2.0, Position(x=-0.5, y=-2.5)),  # Stay at home during batting
                    (3.5, Position(x=2.5, y=-2.5)),  # Run to 1st base
                    (5.0, Position(x=2.5, y=0.0)),  # Run to 2nd base
                    (6.5, Position(x=-2.5, y=0.0)),  # Run to 3rd base
                    (8.0, Position(x=-0.5, y=-2.5)),  # Run back to home
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

            actors.append(
                Actor(
                    id=f"team1_player_{i}",
                    actor_type=ActorType.HUMAN,
                    color=random.choice(colors_team1),
                    initial_position=pos,
                    actions=action_seq,
                    team="team1",
                    movement_path=movement_path,
                )
            )

        # Team 2 (fielding team) - positioned in field
        for i in range(team_size):
            if i == 0:
                # Pitcher
                pos = Position(x=0, y=-1.0)
                action_seq = [
                    (0.0, ActionType.IDLE),
                    (0.5, ActionType.PITCHING),
                    (2.5, ActionType.FIELDING),
                ]
            elif i == 1:
                # Catcher
                pos = Position(x=0.5, y=-2.7)
                action_seq = [(0.0, ActionType.IDLE), (1.5, ActionType.CATCHING)]
            else:
                # Fielders
                (i - 2) * 40 - 60
                distance = random.uniform(3.0, 4.5)
                pos = Position(
                    x=distance * (0.5 if i % 2 == 0 else -0.5), y=-1.0 + distance * 0.3
                )
                action_seq = [
                    (0.0, ActionType.IDLE),
                    (2.0 + i * 0.2, ActionType.FIELDING),
                    (3.0, ActionType.THROWING),
                ]

            actors.append(
                Actor(
                    id=f"team2_player_{i}",
                    actor_type=ActorType.HUMAN,
                    color=random.choice(colors_team2),
                    initial_position=pos,
                    actions=action_seq,
                    team="team2",
                )
            )

        description = f"A baseball game with {num_actors} players on two teams"
        return actors, objects, description

    def _generate_space_scene(
        self, num_actors: int, duration: float, narrative_beats: list = None
    ):
        """Generate space exploration scene"""
        actors = []
        objects = []

        # Add space objects
        objects.append(
            SceneObject(
                id="planet",
                type=ObjectType.PLANET,
                position=Position(x=3.5, y=2.0),
                color="purple",
                scale=1.5,
            )
        )

        objects.append(
            SceneObject(
                id="spaceship",
                type=ObjectType.SPACESHIP,
                position=Position(x=-3.0, y=1.0),
                color="silver",
                scale=1.0,
            )
        )

        # Add stars
        for i in range(5):
            objects.append(
                SceneObject(
                    id=f"star_{i}",
                    type=ObjectType.STAR,
                    position=Position(x=random.uniform(-4, 4), y=random.uniform(0, 4)),
                    color="yellow",
                    scale=0.2,
                )
            )

        # Human explorer
        human_actions = [(0.0, ActionType.WALK), (2.0, ActionType.LOOKING_AROUND)]
        if narrative_beats:
            for beat_name, start, _end in narrative_beats:
                if "meeting" in beat_name:
                    human_actions.append((start, ActionType.WAVE))
                    human_actions.append((start + 0.5, ActionType.TALK))
                elif "eating" in beat_name:
                    human_actions.append((start, ActionType.SIT))
                    human_actions.append((start + 0.5, ActionType.EATING))

        # Movement path: exploring space, walking toward aliens/planet
        human_movement_path = [
            (0.0, Position(x=-2.0, y=-2.0)),  # Start at spaceship
            (2.0, Position(x=0.0, y=-1.5)),  # Walk to center
            (4.0, Position(x=1.5, y=-1.5)),  # Walk toward aliens
        ]
        if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
            human_movement_path.append((5.0, Position(x=0.0, y=-1.5)))  # Move to table

        actors.append(
            Actor(
                id="human_explorer",
                actor_type=ActorType.HUMAN,
                color="blue",
                initial_position=Position(x=-2.0, y=-2.0),
                actions=human_actions,
                movement_path=human_movement_path,
            )
        )

        # Alien(s)
        num_aliens = min(num_actors - 1, 3)
        for i in range(num_aliens):
            alien_actions = [(0.0, ActionType.IDLE), (2.5, ActionType.WALK)]
            if narrative_beats:
                for beat_name, start, _end in narrative_beats:
                    if "meeting" in beat_name:
                        alien_actions.append((start + 0.3, ActionType.WAVE))
                        alien_actions.append((start + 0.8, ActionType.TALK))
                    elif "eating" in beat_name:
                        alien_actions.append((start, ActionType.SIT))
                        alien_actions.append((start + 0.5, ActionType.EATING))

            # Alien movement: walk toward human
            alien_movement_path = [
                (0.0, Position(x=1.0 + i * 0.8, y=-2.0)),  # Start position
                (2.5, Position(x=1.0 + i * 0.8, y=-2.0)),  # Stay idle
                (4.0, Position(x=0.5 + i * 0.5, y=-1.5)),  # Walk toward human
            ]
            if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
                alien_movement_path.append(
                    (5.0, Position(x=0.3 + i * 0.4, y=-1.5))
                )  # Move to table

            actors.append(
                Actor(
                    id=f"alien_{i}",
                    actor_type=ActorType.ALIEN,
                    color="green",
                    initial_position=Position(x=1.0 + i * 0.8, y=-2.0),
                    actions=alien_actions,
                    scale=1.2,  # Aliens slightly larger
                    movement_path=alien_movement_path,
                )
            )

        # Add table and food if eating scene
        if narrative_beats and any("eating" in beat[0] for beat in narrative_beats):
            objects.append(
                SceneObject(
                    id="table",
                    type=ObjectType.TABLE,
                    position=Position(x=0, y=-1.5),
                    color="brown",
                    scale=1.0,
                )
            )
            objects.append(
                SceneObject(
                    id="food",
                    type=ObjectType.FOOD,
                    position=Position(x=0, y=-0.8),
                    color="orange",
                    scale=0.5,
                )
            )

        description = f"A space scene with {num_actors} characters"
        return actors, objects, description

    def _generate_narrative_scene(
        self, num_actors: int, duration: float, narrative_beats: list = None
    ):
        """Generate general narrative scene with varied actor types"""
        actors = []
        objects = []
        colors = ["black", "red", "blue", "green", "purple"]

        # Fully random actor type for narrative scenes
        actor_type = random_actor_type()

        for i in range(num_actors):
            actions = [(0.0, ActionType.IDLE)]

            if narrative_beats:
                for beat_name, start, _end in narrative_beats:
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

            actors.append(
                Actor(
                    id=f"actor_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                    actions=actions,
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A narrative scene with {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_soccer_scene(self, num_actors: int, duration: float):
        """Generate soccer game scene with varied actor types"""
        actors = []
        objects = []

        # Add soccer ball
        objects.append(
            SceneObject(
                id="soccer_ball",
                type=ObjectType.SOCCER_BALL,
                position=Position(x=0, y=-2.0),
                color="white",
                scale=0.3,
                velocity=(random.uniform(-0.5, 0.5), random.uniform(0, 1.0)),
            )
        )

        num_actors = max(num_actors, 6)
        team_size = num_actors // 2

        # Soccer scenes favor humans and robots
        actor_type = random_actor_type(weights={
            ActorType.HUMAN: 0.60,
            ActorType.ROBOT: 0.30,
            ActorType.ALIEN: 0.08,
            ActorType.ANIMAL: 0.02,
        })

        for i in range(team_size):
            actors.append(
                Actor(
                    id=f"team1_player_{i}",
                    actor_type=actor_type,
                    color="red",
                    initial_position=Position(x=-3 + i * 1.5, y=-2.0),
                    actions=[
                        (0.0, ActionType.RUN),
                        (1.5, ActionType.KICKING),
                        (3.0, ActionType.RUN),
                    ],
                    team="team1",
                )
            )

        for i in range(team_size):
            actors.append(
                Actor(
                    id=f"team2_player_{i}",
                    actor_type=actor_type,
                    color="blue",
                    initial_position=Position(x=1 + i * 1.5, y=-2.0),
                    actions=[(0.0, ActionType.RUN), (2.0, ActionType.KICKING)],
                    team="team2",
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A soccer game with {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_nature_scene(self, num_actors: int, duration: float):
        """Generate nature scene with varied actor types"""
        actors = []
        objects = []

        # Add trees
        for i in range(random.randint(2, 4)):
            objects.append(
                SceneObject(
                    id=f"tree_{i}",
                    type=ObjectType.TREE,
                    position=Position(x=random.uniform(-4, 4), y=random.uniform(-1, 0)),
                    scale=random.uniform(0.8, 1.2),
                )
            )

        colors = ["black", "brown", "green"]
        # Choose actor type for scene (all actors same type for coherence)
        actor_type = random_actor_type()

        for i in range(num_actors):
            action = random.choice(
                [
                    ActionType.WALK,
                    ActionType.IDLE,
                    ActionType.WAVE,
                    ActionType.LOOKING_AROUND,
                ]
            )
            actors.append(
                Actor(
                    id=f"actor_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                    actions=[(0.0, action), (2.0, ActionType.WALK)],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A nature scene with trees and {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_city_scene(self, num_actors: int, duration: float):
        """Generate city scene with varied actor types"""
        actors = []
        objects = []

        # Add buildings
        for i in range(random.randint(1, 3)):
            objects.append(
                SceneObject(
                    id=f"building_{i}",
                    type=ObjectType.BUILDING,
                    position=Position(x=random.uniform(-3, 3), y=random.uniform(-1, 0)),
                    scale=random.uniform(0.8, 1.2),
                )
            )

        colors = ["black", "red", "blue", "gray"]
        # City scenes favor humans and robots
        actor_type = random_actor_type(weights={
            ActorType.HUMAN: 0.50,
            ActorType.ROBOT: 0.35,
            ActorType.ALIEN: 0.10,
            ActorType.ANIMAL: 0.05,
        })

        for i in range(num_actors):
            action = random.choice(
                [ActionType.WALK, ActionType.RUN, ActionType.TALK, ActionType.TYPING]
            )
            actors.append(
                Actor(
                    id=f"actor_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                    actions=[(0.0, ActionType.WALK), (2.0, action)],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A city scene with {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_tech_scene(self, num_actors: int, duration: float):
        """Generate tech scene with varied actor types (favors robots)"""
        actors = []
        objects = []

        # Add laptops
        for i in range(min(num_actors, 3)):
            objects.append(
                SceneObject(
                    id=f"laptop_{i}",
                    type=ObjectType.LAPTOP,
                    position=Position(x=-2 + i * 2, y=-1.0),
                    scale=1.0,
                )
            )

        # Tech scenes favor humans and robots
        actor_type = random_actor_type(weights={
            ActorType.HUMAN: 0.45,
            ActorType.ROBOT: 0.40,
            ActorType.ALIEN: 0.10,
            ActorType.ANIMAL: 0.05,
        })

        for i in range(num_actors):
            actors.append(
                Actor(
                    id=f"actor_{i}",
                    actor_type=actor_type,
                    color="black",
                    initial_position=Position(x=-2 + i * 1.5, y=-2.0),
                    actions=[
                        (0.0, ActionType.SIT),
                        (0.5, ActionType.TYPING),
                        (3.0, ActionType.TALK),
                    ],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A tech scene with {num_actors} {actor_label} working"
        return actors, objects, description

    def _generate_generic_scene(self, num_actors: int, duration: float):
        """Generate generic scene with varied actor types"""
        actors = []
        objects = []
        colors = ["black", "red", "blue", "green"]

        # Fully random actor type for generic scenes
        actor_type = random_actor_type()

        for i in range(num_actors):
            action = random.choice(list(ActionType))
            actors.append(
                Actor(
                    id=f"actor_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=random.uniform(-3, 3), y=-2.0),
                    actions=[(0.0, action)],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A scene with {num_actors} {actor_label}"
        return actors, objects, description


    # ==================== NEW SCENE GENERATORS ====================

    def _generate_basketball_scene(self, num_actors: int, duration: float):
        """Generate basketball game scene"""
        actors = []
        objects = []
        colors_team1 = ["red", "darkred"]
        colors_team2 = ["blue", "darkblue"]

        # Add basketball
        objects.append(
            SceneObject(
                id="basketball",
                type=ObjectType.BASKETBALL,
                position=Position(x=0, y=-1.0),
                color="orange",
                scale=0.3,
            )
        )

        actor_type = random_actor_type()
        basketball_actions = [ActionType.RUN, ActionType.JUMP, ActionType.THROWING]

        for i in range(num_actors):
            team_colors = colors_team1 if i % 2 == 0 else colors_team2
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"player_{i}",
                    actor_type=actor_type,
                    color=random.choice(team_colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(basketball_actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"A basketball game with {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_sports_venue_scene(self, num_actors: int, duration: float, venue_type: str):
        """Generate sports venue scene (stadium, rink, track)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        if venue_type == "stadium":
            actions = [ActionType.WALK, ActionType.WAVE, ActionType.CLAP]
            description_prefix = "A crowded stadium with"
        elif venue_type == "rink":
            actions = [ActionType.RUN, ActionType.SPRINT]  # Skating-like motion
            description_prefix = "An ice rink with"
        else:  # track
            actions = [ActionType.RUN, ActionType.SPRINT]
            description_prefix = "A running track with"

        for i in range(num_actors):
            x_pos = -3 + (i * 2)
            actors.append(
                Actor(
                    id=f"athlete_{i}",
                    actor_type=actor_type,
                    color=random.choice(["red", "blue", "green", "yellow"]),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_scifi_scene(self, num_actors: int, duration: float, scifi_type: str):
        """Generate sci-fi scene (moon, mars, spaceship, space_station, alien_planet)"""
        actors = []
        objects = []

        # Sci-fi scenes favor robots and aliens
        actor_weights = {
            ActorType.HUMAN: 0.3,
            ActorType.ROBOT: 0.4,
            ActorType.ALIEN: 0.3,
        }
        actor_type = random.choices(
            list(actor_weights.keys()),
            weights=list(actor_weights.values()),
            k=1
        )[0]

        if scifi_type == "moon":
            actions = [ActionType.WALK, ActionType.JUMP]  # Low gravity bouncing
            description_prefix = "On the lunar surface,"
        elif scifi_type == "mars":
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND]
            description_prefix = "On the Martian landscape,"
        elif scifi_type == "spaceship":
            actions = [ActionType.WALK, ActionType.TYPING, ActionType.TALK]
            description_prefix = "Inside a spaceship,"
        elif scifi_type == "space_station":
            actions = [ActionType.WALK, ActionType.TYPING]
            description_prefix = "Aboard a space station,"
        else:  # alien_planet
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.RUN]
            description_prefix = "On an alien world,"

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"explorer_{i}",
                    actor_type=actor_type,
                    color=random.choice(["white", "silver", "blue", "orange"]),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label} explore"
        return actors, objects, description

    def _generate_underwater_scene(self, num_actors: int, duration: float):
        """Generate underwater scene with swimming motion"""
        actors = []
        objects = []

        # Underwater favors humans and animals
        actor_weights = {
            ActorType.HUMAN: 0.6,
            ActorType.ANIMAL: 0.3,
            ActorType.ROBOT: 0.1,
        }
        actor_type = random.choices(
            list(actor_weights.keys()),
            weights=list(actor_weights.values()),
            k=1
        )[0]

        # Swimming-like actions
        actions = [ActionType.WALK, ActionType.WAVE]  # Slow, floating motion

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            y_pos = random.uniform(-1.5, 0.5)  # Varied depths
            actors.append(
                Actor(
                    id=f"diver_{i}",
                    actor_type=actor_type,
                    color=random.choice(["blue", "cyan", "teal", "navy"]),
                    initial_position=Position(x=x_pos, y=y_pos),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"Underwater scene with {num_actors} {actor_label} swimming"
        return actors, objects, description

    def _generate_aquatic_scene(self, num_actors: int, duration: float, aquatic_type: str):
        """Generate aquatic scene (ocean_surface, river, pool)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        if aquatic_type == "ocean_surface":
            actions = [ActionType.WALK, ActionType.WAVE]
            description_prefix = "On the ocean surface,"
            colors = ["blue", "navy", "white"]
        elif aquatic_type == "river":
            actions = [ActionType.WALK, ActionType.RUN]
            description_prefix = "By the river,"
            colors = ["green", "brown", "blue"]
        else:  # pool
            actions = [ActionType.WALK, ActionType.JUMP]
            description_prefix = "At the swimming pool,"
            colors = ["red", "blue", "yellow", "orange"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"swimmer_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_fantasy_scene(self, num_actors: int, duration: float, fantasy_type: str):
        """Generate fantasy scene (castle, dungeon, enchanted_forest, cloud_realm, lava_realm, ice_realm)"""
        actors = []
        objects = []

        # Fantasy scenes have varied actor types
        actor_weights = {
            ActorType.HUMAN: 0.4,
            ActorType.ALIEN: 0.3,  # Magical creatures
            ActorType.ROBOT: 0.1,  # Golems
            ActorType.ANIMAL: 0.2,
        }
        actor_type = random.choices(
            list(actor_weights.keys()),
            weights=list(actor_weights.values()),
            k=1
        )[0]

        if fantasy_type == "castle":
            actions = [ActionType.WALK, ActionType.TALK, ActionType.WAVE]
            description_prefix = "In a grand castle,"
            colors = ["gold", "purple", "red", "blue"]
        elif fantasy_type == "dungeon":
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.FIGHT]
            description_prefix = "In a dark dungeon,"
            colors = ["gray", "brown", "black"]
        elif fantasy_type == "enchanted_forest":
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.DANCE]
            description_prefix = "In an enchanted forest,"
            colors = ["green", "purple", "blue", "gold"]
        elif fantasy_type == "cloud_realm":
            actions = [ActionType.WALK, ActionType.JUMP, ActionType.DANCE]
            description_prefix = "In the cloud realm,"
            colors = ["white", "gold", "lightblue"]
        elif fantasy_type == "lava_realm":
            actions = [ActionType.WALK, ActionType.JUMP, ActionType.RUN]
            description_prefix = "In the lava realm,"
            colors = ["red", "orange", "black"]
        else:  # ice_realm
            actions = [ActionType.WALK, ActionType.RUN]
            description_prefix = "In the frozen ice realm,"
            colors = ["white", "lightblue", "silver"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"adventurer_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label} on a quest"
        return actors, objects, description

    def _generate_wilderness_scene(self, num_actors: int, duration: float, wilderness_type: str):
        """Generate wilderness scene (forest, jungle, swamp, cave, volcano)"""
        actors = []
        objects = []

        # Wilderness scenes favor humans and animals
        actor_weights = {
            ActorType.HUMAN: 0.5,
            ActorType.ANIMAL: 0.4,
            ActorType.ALIEN: 0.1,
        }
        actor_type = random.choices(
            list(actor_weights.keys()),
            weights=list(actor_weights.values()),
            k=1
        )[0]

        if wilderness_type == "forest":
            actions = [ActionType.WALK, ActionType.RUN, ActionType.LOOKING_AROUND]
            description_prefix = "In a dense forest,"
            colors = ["green", "brown", "olive"]
        elif wilderness_type == "jungle":
            actions = [ActionType.WALK, ActionType.RUN, ActionType.JUMP]
            description_prefix = "In a tropical jungle,"
            colors = ["green", "brown", "yellow"]
        elif wilderness_type == "swamp":
            actions = [ActionType.WALK]  # Slow, careful movement
            description_prefix = "In a murky swamp,"
            colors = ["green", "brown", "gray"]
        elif wilderness_type == "cave":
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND]
            description_prefix = "Deep in a cave,"
            colors = ["gray", "brown", "black"]
        else:  # volcano
            actions = [ActionType.RUN, ActionType.JUMP]
            description_prefix = "Near an active volcano,"
            colors = ["red", "orange", "black"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"explorer_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label} exploring"
        return actors, objects, description

    def _generate_terrain_scene(self, num_actors: int, duration: float, terrain_type: str):
        """Generate terrain scene (desert, beach, mountain, arctic)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        if terrain_type == "desert":
            actions = [ActionType.WALK]  # Slow, trudging movement
            description_prefix = "In the scorching desert,"
            colors = ["tan", "brown", "orange"]
        elif terrain_type == "beach":
            actions = [ActionType.WALK, ActionType.RUN, ActionType.JUMP]
            description_prefix = "On a sunny beach,"
            colors = ["red", "blue", "yellow", "orange"]
        elif terrain_type == "mountain":
            actions = [ActionType.WALK, ActionType.JUMP]  # Climbing
            description_prefix = "On a mountain trail,"
            colors = ["gray", "brown", "green", "blue"]
        else:  # arctic
            actions = [ActionType.WALK]  # Careful movement on ice
            description_prefix = "In the frozen arctic,"
            colors = ["white", "blue", "gray"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"traveler_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label} traveling"
        return actors, objects, description

    def _generate_indoor_scene(self, num_actors: int, duration: float, indoor_type: str):
        """Generate indoor scene (office, gym, museum, restaurant, theater, library, hospital, factory, warehouse, classroom)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        indoor_configs = {
            "office": {
                "actions": [ActionType.WALK, ActionType.TYPING, ActionType.TALK, ActionType.SIT],
                "prefix": "In a busy office,",
                "colors": ["gray", "blue", "white", "black"],
            },
            "gym": {
                "actions": [ActionType.RUN, ActionType.JUMP, ActionType.WALK],
                "prefix": "At the gym,",
                "colors": ["red", "blue", "black", "gray"],
            },
            "museum": {
                "actions": [ActionType.WALK, ActionType.LOOKING_AROUND],
                "prefix": "In a museum,",
                "colors": ["gray", "brown", "navy"],
            },
            "restaurant": {
                "actions": [ActionType.SIT, ActionType.EATING, ActionType.TALK],
                "prefix": "At a restaurant,",
                "colors": ["red", "white", "black"],
            },
            "theater": {
                "actions": [ActionType.SIT, ActionType.CLAP, ActionType.WALK],
                "prefix": "In a theater,",
                "colors": ["red", "gold", "black"],
            },
            "library": {
                "actions": [ActionType.WALK, ActionType.SIT, ActionType.LOOKING_AROUND],
                "prefix": "In a quiet library,",
                "colors": ["brown", "gray", "navy"],
            },
            "hospital": {
                "actions": [ActionType.WALK, ActionType.SIT],
                "prefix": "In a hospital,",
                "colors": ["white", "blue", "green"],
            },
            "factory": {
                "actions": [ActionType.WALK, ActionType.TYPING],
                "prefix": "In a factory,",
                "colors": ["gray", "orange", "yellow"],
            },
            "warehouse": {
                "actions": [ActionType.WALK, ActionType.RUN],
                "prefix": "In a warehouse,",
                "colors": ["gray", "brown", "orange"],
            },
            "classroom": {
                "actions": [ActionType.SIT, ActionType.TALK, ActionType.WALK],
                "prefix": "In a classroom,",
                "colors": ["blue", "red", "green", "yellow"],
            },
        }

        config = indoor_configs.get(indoor_type, indoor_configs["office"])

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"person_{i}",
                    actor_type=actor_type,
                    color=random.choice(config["colors"]),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(config["actions"]))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{config['prefix']} {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_urban_scene(self, num_actors: int, duration: float, urban_type: str):
        """Generate urban scene (rooftop, alley, subway, mall)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        if urban_type == "rooftop":
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.TALK]
            description_prefix = "On a city rooftop,"
            colors = ["black", "gray", "blue"]
        elif urban_type == "alley":
            actions = [ActionType.WALK, ActionType.RUN, ActionType.LOOKING_AROUND]
            description_prefix = "In a dark alley,"
            colors = ["black", "gray", "brown"]
        elif urban_type == "subway":
            actions = [ActionType.WALK, ActionType.RUN, ActionType.SIT]
            description_prefix = "In the subway station,"
            colors = ["gray", "black", "blue", "red"]
        else:  # mall
            actions = [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.TALK]
            description_prefix = "In a shopping mall,"
            colors = ["red", "blue", "pink", "yellow"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"urbanite_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label}"
        return actors, objects, description

    def _generate_social_scene(self, num_actors: int, duration: float, social_type: str):
        """Generate social scene (concert, party, wedding, market, festival, parade)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        social_configs = {
            "concert": {
                "actions": [ActionType.DANCE, ActionType.JUMP, ActionType.CLAP, ActionType.WAVE],
                "prefix": "At a concert,",
                "colors": ["black", "red", "purple", "blue"],
            },
            "party": {
                "actions": [ActionType.DANCE, ActionType.TALK, ActionType.WALK],
                "prefix": "At a party,",
                "colors": ["red", "blue", "gold", "pink"],
            },
            "wedding": {
                "actions": [ActionType.WALK, ActionType.DANCE, ActionType.CLAP],
                "prefix": "At a wedding,",
                "colors": ["white", "black", "gold", "pink"],
            },
            "market": {
                "actions": [ActionType.WALK, ActionType.LOOKING_AROUND, ActionType.TALK],
                "prefix": "At a busy market,",
                "colors": ["red", "green", "brown", "orange"],
            },
            "festival": {
                "actions": [ActionType.WALK, ActionType.DANCE, ActionType.WAVE],
                "prefix": "At a festival,",
                "colors": ["red", "yellow", "green", "blue"],
            },
            "parade": {
                "actions": [ActionType.WALK, ActionType.WAVE, ActionType.DANCE],
                "prefix": "In a parade,",
                "colors": ["red", "blue", "gold", "white"],
            },
        }

        config = social_configs.get(social_type, social_configs["party"])

        for i in range(num_actors):
            x_pos = -3 + (i * 1.2)
            actors.append(
                Actor(
                    id=f"attendee_{i}",
                    actor_type=actor_type,
                    color=random.choice(config["colors"]),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(config["actions"]))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{config['prefix']} {num_actors} {actor_label} celebrating"
        return actors, objects, description

    def _generate_weather_scene(self, num_actors: int, duration: float, weather_type: str):
        """Generate weather-affected scene (rainy_city, snowy_mountain, stormy_beach)"""
        actors = []
        objects = []

        actor_type = random_actor_type()

        if weather_type == "rainy_city":
            actions = [ActionType.WALK, ActionType.RUN]  # Hurrying through rain
            description_prefix = "In a rainy city,"
            colors = ["gray", "black", "navy", "blue"]
        elif weather_type == "snowy_mountain":
            actions = [ActionType.WALK]  # Trudging through snow
            description_prefix = "On a snowy mountain,"
            colors = ["white", "blue", "gray", "red"]
        else:  # stormy_beach
            actions = [ActionType.RUN, ActionType.WALK]  # Battling wind
            description_prefix = "On a stormy beach,"
            colors = ["gray", "blue", "navy"]

        for i in range(num_actors):
            x_pos = -2 + (i * 1.5)
            actors.append(
                Actor(
                    id=f"traveler_{i}",
                    actor_type=actor_type,
                    color=random.choice(colors),
                    initial_position=Position(x=x_pos, y=-2.0),
                    actions=[(0.0, random.choice(actions))],
                )
            )

        actor_label = actor_type.value + "s" if num_actors > 1 else actor_type.value
        description = f"{description_prefix} {num_actors} {actor_label} braving the elements"
        return actors, objects, description