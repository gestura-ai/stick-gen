"""
Dynamic Prompt Generator for LLM-based scene generation.

Two modes of operation:
1. Random prompt generation (legacy) - generates diverse prompts from templates
2. Scene-based prompt generation (preferred) - extracts metadata from a procedurally
   generated scene and uses Grok to create a natural language description

The scene-based approach ensures text prompts accurately describe the motion data,
which is critical for training text-to-motion models.
"""

import json
import os
import random
from typing import TYPE_CHECKING, Optional

from .schema import ActionType, ActorType, FacialExpression


# ============================================================================
# PHASE 1: Emotional Context Mappings
# ============================================================================

# Map facial expressions to emotional adverbs for natural language
EMOTION_ADVERBS = {
    FacialExpression.NEUTRAL: ["calmly", "steadily", "quietly"],
    FacialExpression.HAPPY: ["happily", "joyfully", "cheerfully", "enthusiastically"],
    FacialExpression.SAD: ["sadly", "dejectedly", "glumly", "mournfully"],
    FacialExpression.SURPRISED: ["surprisingly", "unexpectedly", "suddenly"],
    FacialExpression.ANGRY: ["angrily", "furiously", "aggressively", "fiercely"],
    FacialExpression.EXCITED: ["excitedly", "eagerly", "energetically", "enthusiastically"],
}

# Map facial expressions to adjectives for actor descriptions
EMOTION_ADJECTIVES = {
    FacialExpression.NEUTRAL: ["calm", "composed", "relaxed"],
    FacialExpression.HAPPY: ["happy", "joyful", "cheerful", "delighted"],
    FacialExpression.SAD: ["sad", "dejected", "gloomy", "downcast"],
    FacialExpression.SURPRISED: ["surprised", "startled", "astonished"],
    FacialExpression.ANGRY: ["angry", "furious", "fierce", "aggressive"],
    FacialExpression.EXCITED: ["excited", "eager", "energetic", "animated"],
}


# ============================================================================
# PHASE 1: Temporal Context Helpers
# ============================================================================

def get_temporal_phrase(duration: float, start_time: float, total_duration: float) -> str:
    """
    Generate temporal phrases based on action timing.

    Args:
        duration: How long the action lasts in seconds
        start_time: When the action starts
        total_duration: Total scene duration

    Returns:
        Temporal phrase like "briefly", "for a while", "suddenly"
    """
    # Duration-based phrases
    if duration < 1.0:
        duration_phrase = random.choice(["briefly", "quickly", "momentarily"])
    elif duration < 3.0:
        duration_phrase = random.choice(["", "for a moment", ""])  # Often omit for medium
    else:
        duration_phrase = random.choice(["for a while", "steadily", "continuously"])

    # Timing-based phrases (when does action start?)
    relative_start = start_time / total_duration if total_duration > 0 else 0

    if relative_start < 0.1:
        timing_phrase = ""  # At the start, no special phrase needed
    elif relative_start > 0.7:
        timing_phrase = random.choice(["then", "finally", "eventually"])
    else:
        timing_phrase = random.choice(["then", "next", ""])

    return duration_phrase, timing_phrase


def extract_action_timing(actions: list, total_duration: float) -> list:
    """
    Extract timing information from action sequence.

    Args:
        actions: List of (start_time, ActionType) tuples
        total_duration: Total scene duration

    Returns:
        List of dicts with action, duration, start_time, and temporal phrases
    """
    if not actions:
        return []

    timing_info = []
    sorted_actions = sorted(actions, key=lambda x: x[0])

    for i, (start_time, action) in enumerate(sorted_actions):
        # Calculate duration (until next action or end of scene)
        if i + 1 < len(sorted_actions):
            end_time = sorted_actions[i + 1][0]
        else:
            end_time = total_duration

        duration = end_time - start_time
        duration_phrase, timing_phrase = get_temporal_phrase(
            duration, start_time, total_duration
        )

        timing_info.append({
            "action": action.value.replace("_", " "),
            "start_time": start_time,
            "duration": round(duration, 1),
            "duration_phrase": duration_phrase,
            "timing_phrase": timing_phrase,
        })

    return timing_info


# ============================================================================
# PHASE 3: Spatial Relationship Helpers
# ============================================================================

def infer_spatial_relationship(pos1: dict, pos2: dict) -> str:
    """
    Infer the spatial relationship between two actors based on their positions.

    Args:
        pos1: First actor's position {"x": float, "y": float}
        pos2: Second actor's position {"x": float, "y": float}

    Returns:
        Spatial phrase like "near", "far from", "to the left of", etc.
    """
    if not pos1 or not pos2:
        return ""

    x1, y1 = pos1.get("x", 0), pos1.get("y", 0)
    x2, y2 = pos2.get("x", 0), pos2.get("y", 0)

    # Calculate distance
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Determine relative position
    dx = x2 - x1
    dy = y2 - y1

    # Distance-based phrases
    if distance < 1.0:
        distance_phrase = random.choice(["next to", "beside", "close to"])
    elif distance < 3.0:
        distance_phrase = random.choice(["near", "not far from", "nearby"])
    elif distance < 6.0:
        distance_phrase = random.choice(["across from", "a few steps from"])
    else:
        distance_phrase = random.choice(["far from", "at a distance from"])

    # Direction-based phrases (optional addition)
    if abs(dx) > abs(dy) * 2:  # Primarily horizontal
        if dx > 0:
            direction_phrase = "to the right of"
        else:
            direction_phrase = "to the left of"
    elif abs(dy) > abs(dx) * 2:  # Primarily vertical
        if dy > 0:
            direction_phrase = "behind"
        else:
            direction_phrase = "in front of"
    else:
        direction_phrase = ""

    # Return either distance or direction phrase (not both - too wordy)
    if distance_phrase and random.random() < 0.7:
        return distance_phrase
    elif direction_phrase:
        return direction_phrase
    else:
        return distance_phrase


def get_spatial_context(actors_info: list) -> list:
    """
    Extract spatial relationships between actors.

    Args:
        actors_info: List of actor metadata dicts with position info

    Returns:
        List of spatial relationship descriptions
    """
    if len(actors_info) < 2:
        return []

    relationships = []
    for i in range(len(actors_info) - 1):
        pos1 = actors_info[i].get("position")
        pos2 = actors_info[i + 1].get("position")

        if pos1 and pos2:
            rel = infer_spatial_relationship(pos1, pos2)
            if rel:
                relationships.append({
                    "actor1_idx": i,
                    "actor2_idx": i + 1,
                    "relationship": rel,
                })

    return relationships


# ============================================================================
# PHASE 2: Verb Synonyms and Prompt Styles
# ============================================================================

class PromptStyle:
    """Defines different prompt generation styles for variety."""
    ACTION_FOCUSED = "action_focused"  # Emphasizes what characters DO
    NARRATIVE = "narrative"            # Story-like, more descriptive
    DESCRIPTIVE = "descriptive"        # Rich environmental/emotional detail
    SIMPLE = "simple"                  # Basic, short prompts

    @classmethod
    def all_styles(cls) -> list:
        return [cls.ACTION_FOCUSED, cls.NARRATIVE, cls.DESCRIPTIVE, cls.SIMPLE]

    @classmethod
    def random(cls) -> str:
        return random.choice(cls.all_styles())


# Verb synonyms for variety in template generation
VERB_SYNONYMS = {
    "idle": ["stands still", "remains motionless", "stays in place", "pauses"],
    "stand still": ["stands still", "remains stationary", "holds position", "stays put"],
    "walk": ["walks", "strolls", "paces", "ambles", "wanders", "meanders"],
    "run": ["runs", "sprints", "dashes", "races", "rushes", "bolts"],
    "sprint": ["sprints", "dashes", "races", "bolts", "speeds"],
    "jump": ["jumps", "leaps", "hops", "bounds", "springs"],
    "wave": ["waves", "gestures", "signals", "beckons", "greets"],
    "talk": ["talks", "chats", "converses", "speaks", "discusses"],
    "point": ["points", "gestures", "indicates", "motions toward"],
    "clap": ["claps", "applauds", "cheers"],
    "kick": ["kicks", "boots", "strikes", "sends", "launches"],
    "dribble": ["dribbles", "controls", "maneuvers", "handles"],
    "bat": ["bats", "swings", "hits", "strikes at"],
    "pitch": ["pitches", "throws", "hurls", "tosses"],
    "dance": ["dances", "grooves", "sways", "twirls", "moves rhythmically"],
    "fight": ["fights", "battles", "spars", "engages", "combats"],
    "dodge": ["dodges", "evades", "sidesteps", "ducks", "avoids"],
    "punch": ["punches", "strikes", "jabs", "swings at"],
    "throw": ["throws", "hurls", "tosses", "launches", "sends"],
    "catch": ["catches", "grabs", "receives", "snags"],
    "hug": ["hugs", "embraces", "holds", "wraps arms around"],
    "handshake": ["shakes hands", "greets formally", "exchanges handshakes"],
    "high five": ["high-fives", "slaps hands with", "celebrates with"],
    "sit": ["sits", "settles down", "takes a seat", "rests"],
    "type": ["types", "works at a keyboard", "enters data"],
    "push": ["pushes", "shoves", "presses", "moves forward"],
    "pull": ["pulls", "tugs", "draws", "hauls"],
    "lift": ["lifts", "raises", "hoists", "picks up"],
    "carry": ["carries", "holds", "transports", "bears"],
    "explore": ["explores", "investigates", "searches", "wanders through"],
    "fly": ["flies", "soars", "glides", "floats"],
    "teleport": ["teleports", "materializes", "appears suddenly", "vanishes and reappears"],
}


def get_verb_synonym(action: str, use_synonyms: bool = True) -> str:
    """Get a verb synonym for variety, or return the original."""
    if not use_synonyms:
        return action

    # Normalize action name
    action_lower = action.lower().replace("_", " ")

    synonyms = VERB_SYNONYMS.get(action_lower, [action])
    return random.choice(synonyms)


# Prompt style templates for different styles
STYLE_TEMPLATES = {
    PromptStyle.ACTION_FOCUSED: {
        "single_actor": "A {actor} {actions}.",
        "multi_actor": "{actor1} {actions1} while {actor2} {actions2}.",
        "environment_suffix": "",  # No environment for action-focused
    },
    PromptStyle.NARRATIVE: {
        "single_actor": "In the {environment}, a {actor} {actions}.",
        "multi_actor": "The scene unfolds in a {environment} where {actor1} {actions1} as {actor2} {actions2}.",
        "environment_suffix": "",
    },
    PromptStyle.DESCRIPTIVE: {
        "single_actor": "A {emotion} {actor} {temporal} {actions} in the {environment}.",
        "multi_actor": "A {emotion1} {actor1} {actions1} {temporal}, while a {emotion2} {actor2} {actions2} nearby in the {environment}.",
        "environment_suffix": "",
    },
    PromptStyle.SIMPLE: {
        "single_actor": "{actor} {actions}.",
        "multi_actor": "{actor1} {actions1} and {actor2} {actions2}.",
        "environment_suffix": " in a {environment}",
    },
}


if TYPE_CHECKING:
    from .schema import Scene


# Thematic settings that match story_engine themes
THEMES = [
    "nature",
    "city", 
    "sports_baseball",
    "sports_soccer",
    "tech",
    "space",
    "narrative",
]

# Environment/location descriptors by theme
ENVIRONMENTS = {
    "nature": ["forest", "park", "beach", "mountain trail", "garden", "meadow", "riverside"],
    "city": ["busy street", "rooftop", "coffee shop", "subway station", "plaza", "alley"],
    "sports_baseball": ["baseball diamond", "stadium", "batting cage", "dugout"],
    "sports_soccer": ["soccer field", "goal area", "sideline", "locker room"],
    "tech": ["office", "server room", "startup garage", "hackathon", "computer lab"],
    "space": ["spaceship", "alien planet", "space station", "moon base", "asteroid"],
    "narrative": ["living room", "restaurant", "classroom", "hospital", "wedding venue"],
}

# Action categories for prompt generation
ACTION_CATEGORIES = {
    "social": [ActionType.WAVE, ActionType.TALK, ActionType.POINT, ActionType.CLAP,
               ActionType.HANDSHAKE, ActionType.HUG, ActionType.HIGH_FIVE],
    "movement": [ActionType.WALK, ActionType.RUN, ActionType.SPRINT, ActionType.JUMP],
    "combat": [ActionType.FIGHT, ActionType.PUNCH, ActionType.KICK, ActionType.DODGE,
               ActionType.FIGHT_STANCE],
    "emotional": [ActionType.CELEBRATE, ActionType.DANCE, ActionType.CRY, ActionType.LAUGH],
    "sports": [ActionType.BATTING, ActionType.PITCHING, ActionType.CATCHING,
               ActionType.KICKING, ActionType.DRIBBLING, ActionType.SHOOTING],
    "exploration": [ActionType.LOOKING_AROUND, ActionType.CLIMBING, ActionType.CRAWLING,
                    ActionType.SWIMMING, ActionType.FLYING],
    "daily": [ActionType.SIT, ActionType.STAND, ActionType.EATING, ActionType.DRINKING,
              ActionType.READING, ActionType.TYPING],
}

# Narrative scenario templates (use {action_verb} for conjugated verbs)
SCENARIO_TEMPLATES = [
    "{num} {actors} {action_verb} in a {environment}",
    "A {actor} discovers {something} while {action_ing} through the {environment}",
    "A {actor} and a {actor2} have a {interaction} at the {environment}",
    "An epic {genre} scene where {actors} must {objective}",
    "A {actor} teaches a {actor2} how to {action} in the {environment}",
    "A group of {actors} {action_verb} together after {event}",
    "A {actor} nervously {action_verb} while waiting for {event}",
    "Two {actors} compete in a {competition} at the {environment}",
    "A {actor} celebrates {achievement} by {action_ing}",
    "A mysterious {actor} appears and {action_verb} {adverb}",
]

# Interaction types for multi-actor scenes
INTERACTIONS = [
    "heated argument", "friendly conversation", "dance-off", "arm wrestling match",
    "chess game", "cooking competition", "rescue mission", "surprise party",
    "awkward first meeting", "emotional reunion", "training session", "job interview",
]

# Story genres
GENRES = [
    "action", "comedy", "drama", "romance", "thriller", "sci-fi", "fantasy",
    "sports", "slice-of-life", "adventure", "mystery", "heist",
]

# Objectives for action scenes
OBJECTIVES = [
    "escape before time runs out", "find the hidden treasure", "save their friend",
    "win the championship", "solve the mystery", "complete the mission",
    "survive the night", "catch the thief", "deliver the package",
]

# Events/achievements
EVENTS = [
    "winning the game", "a successful presentation", "their graduation",
    "the big reveal", "an unexpected visitor", "the final countdown",
]

# Adverbs for variety
ADVERBS = ["frantically", "gracefully", "cautiously", "enthusiastically", "mysteriously"]


class DynamicPromptGenerator:
    """Generates diverse prompts for LLM scene generation."""
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the prompt generator.
        
        Args:
            cache_size: Number of unique prompts to cache (reduces duplicates)
        """
        self.cache_size = cache_size
        self.generated_prompts: set[str] = set()
        self.prompt_cache: list[str] = []
        
    def generate_prompt(self, theme: Optional[str] = None) -> str:
        """
        Generate a dynamic, contextual prompt.
        
        Args:
            theme: Optional theme to constrain the prompt
            
        Returns:
            A unique, descriptive prompt for LLM scene generation
        """
        # Try to generate a unique prompt (up to 10 attempts)
        for _ in range(10):
            prompt = self._create_prompt(theme)
            if prompt not in self.generated_prompts:
                self.generated_prompts.add(prompt)
                if len(self.generated_prompts) > self.cache_size:
                    # Remove oldest entries when cache is full
                    self.generated_prompts.pop()
                return prompt
        
        # If we can't generate unique, return anyway
        return prompt
    
    def _create_prompt(self, theme: Optional[str] = None) -> str:
        """Create a single prompt using templates and schema data."""
        if theme is None:
            theme = random.choice(THEMES)
            
        # Choose a random strategy
        strategy = random.choice([
            self._template_based_prompt,
            self._action_focused_prompt,
            self._character_focused_prompt,
            self._scenario_prompt,
        ])
        
        return strategy(theme)
    
    def _template_based_prompt(self, theme: str) -> str:
        """Generate prompt using scenario templates."""
        template = random.choice(SCENARIO_TEMPLATES)
        environment = random.choice(ENVIRONMENTS.get(theme, ENVIRONMENTS["narrative"]))

        # Get random actions
        category = random.choice(list(ACTION_CATEGORIES.keys()))
        action = random.choice(ACTION_CATEGORIES[category])
        action_name = action.value.replace("_", " ")
        action_verb = self._action_to_verb(action)

        # Get actor types
        actor_types = [ActorType.HUMAN, ActorType.ROBOT, ActorType.ALIEN, ActorType.ANIMAL]
        actor = random.choice(actor_types).value
        actor2 = random.choice(actor_types).value

        # Create -ing form for action
        action_ing = self._action_to_ing(action)

        # Fill template
        return template.format(
            num=random.choice(["Two", "Three", "A group of", "Several"]),
            actors=f"{actor}s",
            actor=actor,
            actor1=f"a {actor}",
            actor2=actor2,
            action=action_name,
            action_verb=action_verb,
            action_ing=action_ing,
            environment=environment,
            something=random.choice(["a hidden door", "an old map", "a strange device", "a secret message"]),
            interaction=random.choice(INTERACTIONS),
            genre=random.choice(GENRES),
            objective=random.choice(OBJECTIVES),
            event=random.choice(EVENTS),
            competition=random.choice(["dance battle", "race", "cooking contest", "talent show"]),
            achievement=random.choice(EVENTS),
            adverb=random.choice(ADVERBS),
        )

    def _action_to_ing(self, action: ActionType) -> str:
        """Convert ActionType to -ing form."""
        action_ing_forms = {
            ActionType.IDLE: "standing idle",
            ActionType.WALK: "walking",
            ActionType.RUN: "running",
            ActionType.SPRINT: "sprinting",
            ActionType.JUMP: "jumping",
            ActionType.WAVE: "waving",
            ActionType.TALK: "talking",
            ActionType.SHOUT: "shouting",
            ActionType.WHISPER: "whispering",
            ActionType.SING: "singing",
            ActionType.POINT: "pointing",
            ActionType.CLAP: "clapping",
            ActionType.BATTING: "batting",
            ActionType.PITCHING: "pitching",
            ActionType.CATCHING: "catching",
            ActionType.RUNNING_BASES: "running the bases",
            ActionType.FIELDING: "fielding",
            ActionType.THROWING: "throwing",
            ActionType.KICKING: "kicking",
            ActionType.DRIBBLING: "dribbling",
            ActionType.SHOOTING: "shooting",
            ActionType.FIGHT: "fighting",
            ActionType.PUNCH: "punching",
            ActionType.KICK: "kicking",
            ActionType.DODGE: "dodging",
            ActionType.SIT: "sitting",
            ActionType.STAND: "standing",
            ActionType.KNEEL: "kneeling",
            ActionType.LIE_DOWN: "lying down",
            ActionType.EATING: "eating",
            ActionType.DRINKING: "drinking",
            ActionType.READING: "reading",
            ActionType.TYPING: "typing",
            ActionType.LOOKING_AROUND: "looking around",
            ActionType.CLIMBING: "climbing",
            ActionType.CRAWLING: "crawling",
            ActionType.SWIMMING: "swimming",
            ActionType.FLYING: "flying",
            ActionType.CELEBRATE: "celebrating",
            ActionType.DANCE: "dancing",
            ActionType.CRY: "crying",
            ActionType.LAUGH: "laughing",
            ActionType.HANDSHAKE: "shaking hands",
            ActionType.HUG: "hugging",
            ActionType.HIGH_FIVE: "high-fiving",
            ActionType.FIGHT_STANCE: "taking a fighting stance",
        }
        return action_ing_forms.get(action, action.value.replace("_", " ") + "ing")
    
    def _action_focused_prompt(self, theme: str) -> str:
        """Generate prompt focused on specific actions."""
        # Pick 2-3 actions for a sequence
        num_actions = random.randint(2, 3)
        categories = random.sample(list(ACTION_CATEGORIES.keys()), min(num_actions, len(ACTION_CATEGORIES)))
        actions = [self._action_to_verb(random.choice(ACTION_CATEGORIES[cat])) for cat in categories]

        environment = random.choice(ENVIRONMENTS.get(theme, ENVIRONMENTS["narrative"]))
        actor = random.choice([ActorType.HUMAN, ActorType.ROBOT, ActorType.ALIEN]).value

        if len(actions) == 2:
            return f"A {actor} {actions[0]} then {actions[1]} in a {environment}"
        else:
            return f"A {actor} {actions[0]}, {actions[1]}, and finally {actions[2]} in a {environment}"

    def _action_to_verb(self, action: ActionType) -> str:
        """Convert ActionType to proper verb form."""
        action_verbs = {
            ActionType.IDLE: "stands idle",
            ActionType.WALK: "walks",
            ActionType.RUN: "runs",
            ActionType.SPRINT: "sprints",
            ActionType.JUMP: "jumps",
            ActionType.WAVE: "waves",
            ActionType.TALK: "talks",
            ActionType.SHOUT: "shouts",
            ActionType.WHISPER: "whispers",
            ActionType.SING: "sings",
            ActionType.POINT: "points",
            ActionType.CLAP: "claps",
            ActionType.BATTING: "bats",
            ActionType.PITCHING: "pitches",
            ActionType.CATCHING: "catches",
            ActionType.RUNNING_BASES: "runs the bases",
            ActionType.FIELDING: "fields",
            ActionType.THROWING: "throws",
            ActionType.KICKING: "kicks",
            ActionType.DRIBBLING: "dribbles",
            ActionType.SHOOTING: "shoots",
            ActionType.FIGHT: "fights",
            ActionType.PUNCH: "punches",
            ActionType.KICK: "kicks",
            ActionType.DODGE: "dodges",
            ActionType.SIT: "sits",
            ActionType.STAND: "stands",
            ActionType.KNEEL: "kneels",
            ActionType.LIE_DOWN: "lies down",
            ActionType.EATING: "eats",
            ActionType.DRINKING: "drinks",
            ActionType.READING: "reads",
            ActionType.TYPING: "types",
            ActionType.LOOKING_AROUND: "looks around",
            ActionType.CLIMBING: "climbs",
            ActionType.CRAWLING: "crawls",
            ActionType.SWIMMING: "swims",
            ActionType.FLYING: "flies",
            ActionType.CELEBRATE: "celebrates",
            ActionType.DANCE: "dances",
            ActionType.CRY: "cries",
            ActionType.LAUGH: "laughs",
            ActionType.HANDSHAKE: "shakes hands",
            ActionType.HUG: "hugs",
            ActionType.HIGH_FIVE: "high-fives",
            ActionType.FIGHT_STANCE: "takes a fighting stance",
        }
        return action_verbs.get(action, action.value.replace("_", " "))
    
    def _character_focused_prompt(self, theme: str) -> str:
        """Generate prompt focused on character interactions."""
        actors = [ActorType.HUMAN, ActorType.ROBOT, ActorType.ALIEN, ActorType.ANIMAL]
        actor1 = random.choice(actors).value
        actor2 = random.choice(actors).value
        
        environment = random.choice(ENVIRONMENTS.get(theme, ENVIRONMENTS["narrative"]))
        interaction = random.choice(INTERACTIONS)
        
        templates = [
            f"A {actor1} and a {actor2} engage in {interaction} at the {environment}",
            f"When a {actor1} meets a {actor2} at the {environment}, they {random.choice(['fight', 'dance', 'talk', 'compete'])}",
            f"A nervous {actor1} tries to impress a {actor2} at the {environment}",
            f"A {actor1} helps a lost {actor2} find their way through the {environment}",
            f"A {actor1} challenges a {actor2} to a duel at the {environment}",
        ]
        return random.choice(templates)
    
    def _scenario_prompt(self, theme: str) -> str:
        """Generate a complete scenario prompt."""
        genre = random.choice(GENRES)
        environment = random.choice(ENVIRONMENTS.get(theme, ENVIRONMENTS["narrative"]))
        objective = random.choice(OBJECTIVES)
        
        num_actors = random.choice([2, 3, 4])
        actor = random.choice([ActorType.HUMAN, ActorType.ROBOT, ActorType.ALIEN]).value
        
        templates = [
            f"A {genre} scene: {num_actors} {actor}s must {objective} in a {environment}",
            f"An intense {genre} moment where {num_actors} {actor}s face their greatest challenge at the {environment}",
            f"A {genre} story about {num_actors} {actor}s who learn to work together at the {environment}",
            f"The climax of a {genre} tale: {num_actors} {actor}s finally {objective}",
        ]
        return random.choice(templates)
    
    def get_stats(self) -> dict:
        """Return statistics about prompt generation."""
        return {
            "unique_prompts_generated": len(self.generated_prompts),
            "cache_size": self.cache_size,
        }


# Convenience function for simple usage
_default_generator = None

def generate_dynamic_prompt(theme: Optional[str] = None) -> str:
    """Generate a dynamic prompt using the default generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = DynamicPromptGenerator()
    return _default_generator.generate_prompt(theme)


class ScenePromptGenerator:
    """
    Generates natural language prompts from Scene metadata using Grok.

    This ensures the text prompt accurately describes the actual motion data,
    which is essential for training text-to-motion models.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._client = None
        self._cache: dict[str, str] = {}  # Cache descriptions by scene hash

    def _get_client(self):
        """Lazy initialization of OpenAI client for Grok."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("GROK_API_KEY")
                if api_key:
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.x.ai/v1"
                    )
                else:
                    if self.verbose:
                        print("[ScenePromptGenerator] No GROK_API_KEY, using template fallback")
            except ImportError:
                if self.verbose:
                    print("[ScenePromptGenerator] OpenAI not installed, using template fallback")
        return self._client

    def extract_scene_metadata(self, scene: "Scene") -> dict:
        """
        Extract structured metadata from a Scene object.

        Includes:
        - Actor info (type, color, actions, team, emotional state)
        - Action timing information (duration, start time, temporal phrases)
        - Objects and theme
        """
        actors_info = []
        for actor in scene.actors:
            # Get unique actions for this actor (simple list)
            actions = []
            for _, action_type in actor.actions:
                action_name = action_type.value.replace("_", " ")
                if action_name not in actions:
                    actions.append(action_name)

            # Extract detailed timing info for richer prompts
            action_timing = extract_action_timing(actor.actions, scene.duration)

            # Get emotional state from facial expression
            emotional_state = actor.facial_expression.value if actor.facial_expression else "neutral"

            # Get emotional adverb for this actor
            expr = actor.facial_expression if actor.facial_expression else FacialExpression.NEUTRAL
            emotional_adverb = random.choice(EMOTION_ADVERBS.get(expr, ["calmly"]))
            emotional_adjective = random.choice(EMOTION_ADJECTIVES.get(expr, ["calm"]))

            # Extract position for spatial relationships
            position = None
            if actor.initial_position:
                position = {
                    "x": actor.initial_position.x,
                    "y": actor.initial_position.y,
                }

            actors_info.append({
                "type": actor.actor_type.value,
                "color": actor.color,
                "actions": actions[:5],  # Limit to 5 main actions
                "action_timing": action_timing[:5],  # Detailed timing info
                "team": actor.team,
                "emotional_state": emotional_state,
                "emotional_adverb": emotional_adverb,
                "emotional_adjective": emotional_adjective,
                "position": position,
            })

        objects_info = [
            {"type": obj.type.value, "color": obj.color}
            for obj in scene.objects[:5]  # Limit objects
        ]

        # Extract spatial relationships between actors
        spatial_relationships = get_spatial_context(actors_info)

        return {
            "duration": round(scene.duration, 1),
            "theme": scene.theme or "general",
            "num_actors": len(scene.actors),
            "actors": actors_info[:6],  # Limit to 6 actors for prompt
            "objects": objects_info,
            "has_teams": any(a.team for a in scene.actors),
            "spatial_relationships": spatial_relationships,
        }

    def extract_raw_scene_metadata(self, scene: "Scene") -> dict:
        """
        Extract RAW scene metadata for Grok API - no pre-mapped vocabulary.

        Unlike extract_scene_metadata(), this method returns only raw data
        (emotional_state values, raw timing numbers, raw coordinates) without
        pre-computed adverbs, temporal phrases, or spatial relationships.

        This allows Grok to generate its own vocabulary, reducing author bias
        and increasing training data diversity.

        Args:
            scene: The Scene object to extract metadata from

        Returns:
            Dict with raw scene data for Grok to interpret freely
        """
        actors_info = []
        for actor in scene.actors:
            # Extract raw action timing (just numbers, no phrases)
            raw_actions = []
            sorted_actions = sorted(actor.actions, key=lambda x: x[0])

            for i, (start_time, action_type) in enumerate(sorted_actions):
                # Calculate duration
                if i + 1 < len(sorted_actions):
                    end_time = sorted_actions[i + 1][0]
                else:
                    end_time = scene.duration
                duration = end_time - start_time

                raw_actions.append({
                    "action": action_type.value.replace("_", " "),
                    "start_time": round(start_time, 1),
                    "duration_seconds": round(duration, 1),
                })

            # Extract raw position (just coordinates)
            raw_position = None
            if actor.initial_position:
                raw_position = {
                    "x": round(actor.initial_position.x, 2),
                    "y": round(actor.initial_position.y, 2),
                }

            actors_info.append({
                "type": actor.actor_type.value,
                "color": actor.color,
                "team": actor.team,
                # RAW emotional state - NOT pre-mapped to adverbs
                "emotional_state": (
                    actor.facial_expression.value
                    if actor.facial_expression else "neutral"
                ),
                # RAW action timing - NOT pre-mapped to phrases
                "actions": raw_actions[:5],
                # RAW position - NOT pre-mapped to spatial relationships
                "position": raw_position,
            })

        objects_info = [
            {"type": obj.type.value, "color": obj.color}
            for obj in scene.objects[:5]
        ]

        return {
            "duration_seconds": round(scene.duration, 1),
            "theme": scene.theme or "general",
            "num_actors": len(scene.actors),
            "actors": actors_info[:6],
            "objects": objects_info,
            "has_teams": any(a.team for a in scene.actors),
        }

    def generate_prompt_from_scene(
        self,
        scene: "Scene",
        style: str = None,
        use_synonyms: bool = True
    ) -> str:
        """
        Generate a natural language prompt that describes the scene.

        Args:
            scene: The Scene object to describe
            style: PromptStyle value (ACTION_FOCUSED, NARRATIVE, DESCRIPTIVE, SIMPLE)
                   If None, a random style is selected for variety
            use_synonyms: Whether to use verb synonyms for variety

        Uses Grok to create varied, natural descriptions.
        Falls back to template-based generation if API unavailable.
        """
        # Extract metadata for template fallback (includes pre-mapped vocabulary)
        metadata = self.extract_scene_metadata(scene)

        # Select style randomly if not specified
        if style is None:
            style = PromptStyle.random()
        metadata["style"] = style

        # Create a cache key from metadata (includes style)
        cache_key = json.dumps(metadata, sort_keys=True)
        if cache_key in self._cache:
            return self._cache[cache_key]

        client = self._get_client()

        if client:
            try:
                # Pass scene directly - _generate_with_grok uses raw metadata
                prompt = self._generate_with_grok(scene, client, style)
            except Exception as e:
                if self.verbose:
                    print(f"[ScenePromptGenerator] Grok error: {e}, using template")
                # Fallback uses pre-mapped metadata for deterministic templates
                prompt = self._generate_from_template(metadata, style, use_synonyms)
        else:
            # Template fallback uses pre-mapped metadata
            prompt = self._generate_from_template(metadata, style, use_synonyms)

        # Cache the result
        if len(self._cache) < 5000:
            self._cache[cache_key] = prompt

        return prompt

    def generate_prompts_from_scene(
        self,
        scene: "Scene",
        num_prompts: int = 3,
        use_synonyms: bool = True
    ) -> list:
        """
        Generate multiple diverse prompts for a single scene.

        This is useful for training data augmentation - the same motion
        can be described in multiple valid ways.

        Args:
            scene: The Scene object to describe
            num_prompts: Number of prompts to generate (default: 3)
            use_synonyms: Whether to use verb synonyms for variety

        Returns:
            List of unique prompt strings describing the scene
        """
        prompts = []
        styles_used = set()

        # Get all available styles
        all_styles = PromptStyle.all_styles()

        # Generate prompts with different styles
        for i in range(min(num_prompts, len(all_styles))):
            # Pick a style we haven't used yet
            available_styles = [s for s in all_styles if s not in styles_used]
            if not available_styles:
                break

            style = random.choice(available_styles)
            styles_used.add(style)

            # Generate prompt with this style
            prompt = self.generate_prompt_from_scene(
                scene, style=style, use_synonyms=use_synonyms
            )

            # Only add if unique
            if prompt not in prompts:
                prompts.append(prompt)

        # If we need more prompts, generate with synonyms variation
        attempts = 0
        while len(prompts) < num_prompts and attempts < 10:
            style = random.choice(all_styles)
            prompt = self.generate_prompt_from_scene(
                scene, style=style, use_synonyms=True
            )
            if prompt not in prompts:
                prompts.append(prompt)
            attempts += 1

        return prompts

    def _get_style_system_prompt(self, style: str) -> str:
        """
        Get style-specific system prompt for Grok.

        The prompt instructs Grok to generate its own vocabulary from RAW scene data:
        - emotional_state ("happy") → generate appropriate adverbs/adjectives
        - position coordinates → infer spatial relationships naturally
        - theme → create fitting environment descriptions
        - action timing → incorporate temporal flow naturally

        This reduces author bias by not constraining Grok to predefined word lists.
        """
        base_rules = """RULES FOR RAW DATA INTERPRETATION:
- The data contains RAW values - generate your OWN vocabulary choices:
  * emotional_state: "happy" → you choose: "joyfully", "gleefully", "with a smile", etc.
  * emotional_state: "angry" → you choose: "furiously", "aggressively", "with rage", etc.
  * position coordinates (x, y) → describe spatial relationships naturally: "nearby", "across the field", "side by side"
  * theme → create a fitting environment: "soccer" → "pitch", "field", "stadium", etc.
  * duration_seconds → incorporate pacing: short durations feel quick/sudden, long ones feel gradual/sustained

- Use VARIED VERBS - choose creatively, not from a fixed list
- Include COLORS of characters when distinctive
- Don't mention "stick figures" or "animation"
- Be creative with your word choices - maximize vocabulary diversity"""

        if style == PromptStyle.ACTION_FOCUSED:
            return f"""You are a motion caption writer focusing on ACTIONS.
Write a direct, action-focused sentence about what the characters DO.

{base_rules}
- Emphasize the sequence of actions
- Write 10-25 words
- Keep it punchy and direct

EXAMPLE: "A red player sprints forward and kicks the ball as a blue defender rushes to block."
BAD: "In the sunny stadium, a red player..." (too descriptive)"""

        elif style == PromptStyle.NARRATIVE:
            return f"""You are a storytelling motion caption writer.
Write a narrative sentence that feels like part of a story.

{base_rules}
- Set the scene with environment context
- Use story-like phrasing ("The scene unfolds...", "As the...", etc.)
- Write 15-35 words

EXAMPLE: "The scene unfolds on the pitch where an excited red striker races toward goal, narrowly evading a determined blue defender."
BAD: "A human runs and kicks." (too mechanical)"""

        elif style == PromptStyle.DESCRIPTIVE:
            return f"""You are a descriptive motion caption writer.
Write a richly detailed sentence focusing on emotions, environment, and atmosphere.

{base_rules}
- Include environmental details from the theme
- Describe emotional states and body language
- Use temporal words: "quickly", "slowly", "suddenly", "briefly"
- Write 20-40 words

EXAMPLE: "An excited red athlete sprints eagerly across the sunlit pitch, their movements swift and determined, while a calm blue opponent tracks their path methodically."
BAD: "A player runs and kicks the ball." (too simple)"""

        else:  # SIMPLE
            return f"""You are a concise motion caption writer.
Write a short, clear sentence describing the main action.

{base_rules}
- Keep it simple and direct
- Write 8-15 words
- Focus on the core action only

EXAMPLE: "A red player kicks the ball toward a blue defender."
BAD: "In the magnificent stadium under the bright afternoon sun..." (too wordy)"""

    def _generate_with_grok(
        self, scene: "Scene", client, style: str = None
    ) -> str:
        """
        Use Grok to generate a natural language scene description.

        Uses extract_raw_scene_metadata() to pass RAW data to Grok,
        allowing the LLM to generate its own vocabulary (adverbs,
        spatial phrases, environment descriptions) without author bias.
        """
        if style is None:
            style = PromptStyle.DESCRIPTIVE  # Default style

        # Use RAW metadata - no pre-mapped vocabulary
        raw_metadata = self.extract_raw_scene_metadata(scene)

        system_prompt = self._get_style_system_prompt(style)
        user_prompt = f"Generate a motion caption for this scene:\n{json.dumps(raw_metadata)}"

        response = client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=120,
            temperature=0.85,
        )

        result = response.choices[0].message.content.strip()
        # Remove quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]

        if self.verbose:
            print(f"[ScenePrompt] [{style}] Generated: {result[:60]}...")

        return result

    def _generate_from_template(
        self,
        metadata: dict,
        style: str = None,
        use_synonyms: bool = True
    ) -> str:
        """
        Fallback: generate prompt from templates when API unavailable.

        Now includes:
        - Emotional adverbs from facial expressions
        - Temporal phrases from action timing
        - Color and team context
        - Object and environment descriptions
        - Style-based formatting
        - Verb synonyms for variety
        """
        actors = metadata.get("actors", [])
        theme = metadata.get("theme", "general")
        objects = metadata.get("objects", [])
        has_teams = metadata.get("has_teams", False)
        spatial_relationships = metadata.get("spatial_relationships", [])

        if style is None:
            style = PromptStyle.DESCRIPTIVE

        if not actors:
            return "A figure stands in a scene."

        # Get environment for styles that need it
        env = self._theme_to_environment(theme, objects)
        object_phrase = self._objects_to_phrase(objects)

        # Build actor descriptions based on style
        if style == PromptStyle.SIMPLE:
            return self._generate_simple_template(actors, env, use_synonyms)
        elif style == PromptStyle.ACTION_FOCUSED:
            return self._generate_action_focused_template(
                actors, object_phrase, use_synonyms
            )
        elif style == PromptStyle.NARRATIVE:
            return self._generate_narrative_template(
                actors, env, object_phrase, use_synonyms, spatial_relationships
            )
        else:  # DESCRIPTIVE (default)
            return self._generate_descriptive_template(
                actors, env, object_phrase, has_teams, use_synonyms,
                spatial_relationships
            )

    def _generate_simple_template(
        self, actors: list, env: str, use_synonyms: bool
    ) -> str:
        """Generate a simple, short prompt."""
        parts = []
        for i, actor in enumerate(actors[:2]):
            actor_type = actor["type"]
            color = actor.get("color", "")
            actions = actor.get("actions", ["stand"])

            if color and color not in ["black", "gray"]:
                actor_desc = f"{color} {actor_type}"
            else:
                actor_desc = actor_type

            verb = get_verb_synonym(actions[0], use_synonyms)

            if i == 0:
                parts.append(f"A {actor_desc} {verb}")
            else:
                parts.append(f"a {actor_desc} {verb}")

        return " and ".join(parts) + f" in a {env}."

    def _generate_action_focused_template(
        self, actors: list, object_phrase: str, use_synonyms: bool
    ) -> str:
        """Generate an action-focused prompt emphasizing what characters do."""
        parts = []
        for i, actor in enumerate(actors[:2]):
            actor_type = actor["type"]
            color = actor.get("color", "")
            actions = actor.get("actions", ["stand"])

            if color and color not in ["black", "gray"]:
                actor_desc = f"{color} {actor_type}"
            else:
                actor_desc = actor_type

            # Build action sequence
            verb_parts = []
            for action in actions[:2]:
                verb_parts.append(get_verb_synonym(action, use_synonyms))

            action_text = " and ".join(verb_parts) if len(verb_parts) > 1 else verb_parts[0]

            if i == 0:
                parts.append(f"A {actor_desc} {action_text}")
            else:
                parts.append(f"a {actor_desc} {action_text}")

        result = " while ".join(parts) if len(parts) > 1 else parts[0]
        if object_phrase:
            result += object_phrase
        return result + "."

    def _generate_narrative_template(
        self, actors: list, env: str, object_phrase: str, use_synonyms: bool,
        spatial_relationships: list = None
    ) -> str:
        """Generate a story-like narrative prompt with spatial context."""
        if not actors:
            return "A figure stands in a scene."

        actor = actors[0]
        actor_type = actor["type"]
        color = actor.get("color", "")
        emotional_adj = actor.get("emotional_adjective", "")
        actions = actor.get("actions", ["stand"])

        if emotional_adj and emotional_adj not in ["calm", "neutral"]:
            if color and color not in ["black", "gray"]:
                actor_desc = f"{emotional_adj} {color} {actor_type}"
            else:
                actor_desc = f"{emotional_adj} {actor_type}"
        elif color and color not in ["black", "gray"]:
            actor_desc = f"{color} {actor_type}"
        else:
            actor_desc = actor_type

        verb = get_verb_synonym(actions[0], use_synonyms)

        if len(actors) > 1:
            actor2 = actors[1]
            actor2_type = actor2["type"]
            actor2_color = actor2.get("color", "")
            actor2_actions = actor2.get("actions", ["stand"])

            if actor2_color and actor2_color not in ["black", "gray"]:
                actor2_desc = f"{actor2_color} {actor2_type}"
            else:
                actor2_desc = actor2_type

            verb2 = get_verb_synonym(actor2_actions[0], use_synonyms)

            # Add spatial relationship if available
            spatial_phrase = ""
            if spatial_relationships:
                for rel in spatial_relationships:
                    if rel.get("actor1_idx") == 0 and rel.get("actor2_idx") == 1:
                        spatial_phrase = f" {rel['relationship']}"
                        break

            return f"The scene unfolds in a {env} where a {actor_desc} {verb}{spatial_phrase} as a {actor2_desc} {verb2}{object_phrase}."
        else:
            return f"In the {env}, a {actor_desc} {verb}{object_phrase}."

    def _generate_descriptive_template(
        self, actors: list, env: str, object_phrase: str,
        has_teams: bool, use_synonyms: bool,
        spatial_relationships: list = None
    ) -> str:
        """Generate a richly descriptive prompt with emotions, timing, and spatial context."""
        descriptions = []
        seen_teams = set()
        spatial_map = {}

        # Build spatial relationship map for quick lookup
        if spatial_relationships:
            for rel in spatial_relationships:
                key = (rel.get("actor1_idx"), rel.get("actor2_idx"))
                spatial_map[key] = rel.get("relationship", "")

        for i, actor in enumerate(actors[:3]):  # Max 3 actors in description
            actor_type = actor["type"]
            actor_color = actor.get("color", "")
            actor_team = actor.get("team")
            actions = actor.get("actions", ["stand"])
            action_timing = actor.get("action_timing", [])

            # Get emotional context
            emotional_adverb = actor.get("emotional_adverb", "")
            emotional_adj = actor.get("emotional_adjective", "")

            # Build actor description with emotion and color
            if emotional_adj and emotional_adj not in ["calm", "neutral"]:
                if actor_color and actor_color not in ["black", "gray"]:
                    actor_desc = f"{emotional_adj} {actor_color} {actor_type}"
                else:
                    actor_desc = f"{emotional_adj} {actor_type}"
            elif actor_color and actor_color not in ["black", "gray"]:
                actor_desc = f"{actor_color} {actor_type}"
            else:
                actor_desc = actor_type

            # Add team context for first actor of each team
            if has_teams and actor_team and actor_team not in seen_teams:
                seen_teams.add(actor_team)
                if len(seen_teams) == 1:
                    actor_desc = f"{actor_desc} from one team"
                else:
                    actor_desc = f"{actor_desc} from the opposing team"

            if i == 0:
                prefix = f"A {actor_desc}"
            else:
                # Add spatial relationship between this actor and previous
                spatial_phrase = spatial_map.get((i - 1, i), "")
                if spatial_phrase:
                    prefix = f"{spatial_phrase}, a {actor_desc}"
                else:
                    prefix = f"and a {actor_desc}"

            # Build action text with temporal phrases and synonyms
            action_text = self._build_action_text_with_timing(
                actions, action_timing, emotional_adverb, use_synonyms
            )

            descriptions.append(f"{prefix} {action_text}")

        base = " ".join(descriptions)
        if object_phrase:
            return f"{base}{object_phrase} in a {env}."
        else:
            return f"{base} in a {env}."

    def _build_action_text_with_timing(
        self, actions: list, action_timing: list, emotional_adverb: str,
        use_synonyms: bool = True
    ) -> str:
        """Build action description with temporal phrases, emotional adverbs, and synonyms."""
        if not actions:
            return "stands still"

        # Use timing info if available, otherwise fall back to simple list
        if action_timing and len(action_timing) >= len(actions):
            parts = []
            for i, timing in enumerate(action_timing[:3]):  # Max 3 actions
                action = timing.get("action", actions[i] if i < len(actions) else "stand")
                duration_phrase = timing.get("duration_phrase", "")
                timing_phrase = timing.get("timing_phrase", "")

                # Use synonym if enabled, otherwise use standard verb
                if use_synonyms:
                    verb = get_verb_synonym(action, use_synonyms)
                else:
                    verb = self._action_to_verb_template(action)

                # Add emotional adverb to first action only
                if i == 0 and emotional_adverb and emotional_adverb not in ["calmly"]:
                    verb = f"{emotional_adverb} {verb}"

                # Add timing phrase
                if timing_phrase and i > 0:
                    verb = f"{timing_phrase} {verb}"

                # Add duration phrase for short/long actions
                if duration_phrase and duration_phrase not in [""]:
                    verb = f"{verb} {duration_phrase}"

                parts.append(verb)

            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return f"{parts[0]}, {parts[1]}"
            else:
                return f"{parts[0]}, {parts[1]}, and {parts[2]}"
        else:
            # Fallback to simple action list
            if len(actions) == 1:
                verb = get_verb_synonym(actions[0], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[0])
                if emotional_adverb and emotional_adverb not in ["calmly"]:
                    return f"{emotional_adverb} {verb}"
                return verb
            elif len(actions) == 2:
                v1 = get_verb_synonym(actions[0], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[0])
                v2 = get_verb_synonym(actions[1], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[1])
                if emotional_adverb and emotional_adverb not in ["calmly"]:
                    return f"{emotional_adverb} {v1} then {v2}"
                return f"{v1} then {v2}"
            else:
                v1 = get_verb_synonym(actions[0], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[0])
                v2 = get_verb_synonym(actions[1], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[1])
                v3 = get_verb_synonym(actions[2], use_synonyms) if use_synonyms else self._action_to_verb_template(actions[2])
                return f"{v1}, {v2}, and {v3}"

    def _action_to_verb_template(self, action: str) -> str:
        """Convert action name to verb phrase."""
        verbs = {
            "idle": "stands still",
            "walk": "walks",
            "run": "runs",
            "sprint": "sprints",
            "jump": "jumps",
            "wave": "waves",
            "talk": "talks",
            "shout": "shouts",
            "point": "points",
            "clap": "claps",
            "batting": "bats",
            "pitching": "pitches",
            "catching": "catches",
            "running bases": "runs the bases",
            "fielding": "fields",
            "throwing": "throws",
            "kicking": "kicks",
            "dribbling": "dribbles",
            "shooting": "shoots",
            "fight": "fights",
            "punch": "punches",
            "kick": "kicks",
            "dodge": "dodges",
            "sit": "sits down",
            "stand": "stands up",
            "eating": "eats",
            "drinking": "drinks",
            "reading": "reads",
            "typing": "types",
            "looking around": "looks around",
            "climbing": "climbs",
            "crawling": "crawls",
            "swimming": "swims",
            "flying": "flies",
            "celebrate": "celebrates",
            "dance": "dances",
            "cry": "cries",
            "laugh": "laughs",
            "handshake": "shakes hands",
            "hug": "hugs",
            "high five": "high-fives",
            "fight stance": "takes a fighting stance",
        }
        return verbs.get(action.lower(), action + "s")

    def _objects_to_phrase(self, objects: list) -> str:
        """Convert objects to a descriptive phrase."""
        if not objects:
            return ""

        # Filter to interesting objects (sports equipment, etc.)
        interesting_types = {
            "baseball", "basketball", "soccer_ball", "ball", "bat", "glove",
            "laptop", "phone", "computer", "food", "plate", "cup",
            "spaceship", "ufo",
        }

        interesting = [obj for obj in objects if obj.get("type", "") in interesting_types]

        if not interesting:
            return ""

        # Describe first interesting object
        obj = interesting[0]
        obj_type = obj["type"].replace("_", " ")
        obj_color = obj.get("color", "")

        if obj_color and obj_color not in ["gray", "white"]:
            return f" with a {obj_color} {obj_type}"
        else:
            return f" with a {obj_type}"

    def _theme_to_environment(self, theme: str, objects: list = None) -> str:
        """Convert theme and objects to environment description."""
        objects = objects or []

        # First, try to infer environment from objects (more specific)
        object_types = {obj.get("type", "") for obj in objects}

        if "stadium" in object_types:
            return "stadium"
        if "building" in object_types or "house" in object_types:
            return random.choice(["city street", "urban area", "plaza"])
        if "tree" in object_types or "rock" in object_types:
            return random.choice(["forest", "park", "outdoor area"])
        if "spaceship" in object_types or "planet" in object_types or "ufo" in object_types:
            return random.choice(["space", "alien planet", "space station"])

        # Fall back to theme-based environment with more comprehensive mapping
        environments = {
            # Nature themes
            "nature": random.choice(["forest", "park", "meadow", "garden"]),
            # City themes
            "city": random.choice(["busy street", "plaza", "rooftop", "alley"]),
            # Sports themes - multiple variations to catch different naming
            "sports_baseball": random.choice(["baseball diamond", "stadium", "field"]),
            "baseball": random.choice(["baseball diamond", "stadium", "ballpark"]),
            "sports_soccer": random.choice(["soccer field", "goal area", "pitch"]),
            "soccer": random.choice(["soccer field", "pitch", "goal area"]),
            "sports": random.choice(["playing field", "arena", "court"]),
            # Tech themes
            "tech": random.choice(["office", "computer lab", "tech hub"]),
            # Space themes
            "space": random.choice(["spaceship", "space station", "alien planet"]),
            # Narrative themes
            "narrative": random.choice(["room", "restaurant", "home"]),
            # General fallback
            "general": random.choice(["open area", "scene", "setting"]),
        }
        return environments.get(theme, environments.get("general"))

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cached_prompts": len(self._cache),
            "has_grok_client": self._client is not None,
        }
