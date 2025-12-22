# Prompt Generation Architecture

## Overview

This document explains how text prompts are generated and associated with motion data in the Stick-Gen training pipeline. The key principle is **scene-first generation**: we first generate the motion data, then create a text prompt that accurately describes it.

## The Problem with Traditional Approaches

**Traditional (prompt-first) approach:**
```
Random prompt → LLM → Script → Motion data
     ↑                              ↑
 (user's idea)            (what actually happens)
     These may NOT match!
```

For example, if you prompt "an epic heist", the LLM might describe "a ninja sneaking past guards", but the procedural motion generator might produce slightly different actions.

**Our solution (scene-first):**
```
Procedural Scene Generation → Scene Object (with actors, actions, objects)
          ↓                              ↓
     Motion Tensors          Extract Metadata → Grok → Natural Language Prompt
          ↓                              ↓
     Training Sample ←←←← These are GUARANTEED to match! →→→→ Text Prompt
```

---

## Architecture Components

### 1. Scene Generation Layer

**File:** `src/data_gen/story_engine.py`

The `StoryGenerator` class creates `Scene` objects with:
- **Actors**: Type (human/robot/alien/animal), color, team, initial position
- **Actions**: Time-indexed action sequences per actor (e.g., `[(0.0, WALK), (2.0, WAVE)]`)
- **Objects**: Environment props (trees, buildings, sports equipment)
- **Theme**: Context like "soccer", "nature", "city", "space"

**Example Scene themes:**
- `soccer`: Teams of players with kicking, running, dribbling actions
- `baseball`: Pitcher, batter, fielders with sport-specific actions
- `nature`: People walking among trees
- `space`: Astronauts and aliens interacting

### 2. Scene Object Structure

**File:** `src/data_gen/schema.py`

```python
class Scene(BaseModel):
    duration: float           # Length in seconds
    actors: list[Actor]       # Characters with actions
    objects: list[SceneObject]  # Environment props
    background_color: str
    description: str          # Text prompt (set after generation)
    theme: str | None         # "soccer", "city", etc.

class Actor(BaseModel):
    id: str
    actor_type: ActorType     # HUMAN, ROBOT, ALIEN, ANIMAL
    color: str                # "red", "blue", etc.
    initial_position: Position
    actions: list[tuple[float, ActionType]]  # [(time, action), ...]
    team: str | None          # For team sports

class ActionType(str, Enum):
    # 40+ action types including:
    WALK, RUN, JUMP, WAVE, TALK, DANCE, FIGHT, KICK, BATTING, PITCHING, ...
```

### 3. Prompt Generation from Scene

**File:** `src/data_gen/prompt_generator.py` → `ScenePromptGenerator` class

This is where the magic happens. After a scene is generated:

1. **Extract Metadata**: Pull structured info from the Scene object
2. **Call Grok API**: Send metadata to generate natural language
3. **Fallback to Templates**: If API unavailable, use rule-based generation

```python
class ScenePromptGenerator:
    def extract_scene_metadata(self, scene: Scene) -> dict:
        """Extract structured data from scene."""
        return {
            "duration": scene.duration,
            "theme": scene.theme,
            "num_actors": len(scene.actors),
            "actors": [
                {
                    "type": actor.actor_type.value,  # "human", "robot"
                    "color": actor.color,            # "red", "blue"
                    "actions": [action.value for _, action in actor.actions],
                    "action_timing": [...],          # Duration and timing info
                    "team": actor.team,
                    "emotional_state": "happy",      # From facial_expression
                    "emotional_adverb": "happily",   # For action descriptions
                    "emotional_adjective": "excited", # For actor descriptions
                    "position": {"x": 0, "y": 2},    # For spatial relationships
                }
                for actor in scene.actors
            ],
            "objects": [{"type": obj.type.value, "color": obj.color} ...],
            "has_teams": any(a.team for a in scene.actors),
            "spatial_relationships": [...],  # Between-actor spatial context
        }

    def generate_prompt_from_scene(self, scene: Scene, style=None, use_synonyms=True) -> str:
        metadata = self.extract_scene_metadata(scene)
        # Try Grok first, fall back to templates
        return self._generate_with_grok(metadata, style) or self._generate_from_template(metadata, style, use_synonyms)

    def generate_prompts_from_scene(self, scene: Scene, num_prompts=3) -> list:
        """Generate multiple diverse prompts for training augmentation."""
        # Returns list of unique prompts using different styles
```

---

## Grok Integration

**API:** X.AI Grok (via OpenAI-compatible endpoint)
**Model:** `grok-4-1-fast`
**Environment Variable:** `GROK_API_KEY`

### Dual-Mode Metadata Extraction

To maximize vocabulary diversity and reduce author bias, the prompt generator uses **two different metadata extraction modes**:

#### 1. Raw Metadata (for Grok API)

When using the Grok API, we pass **raw scene data only** via `extract_raw_scene_metadata()`:

```python
{
    "duration_seconds": 5.0,
    "theme": "baseball",
    "actors": [
        {
            "type": "human",
            "color": "dark blue",
            "emotional_state": "happy",  # Raw value - NOT "happily"
            "actions": [
                {"action": "idle", "start_time": 0.0, "duration_seconds": 1.0},
                {"action": "pitching", "start_time": 1.0, "duration_seconds": 4.0}
            ],
            "position": {"x": 0, "y": -2.0},  # Raw coords - NOT "near" or "beside"
            "team": "team1"
        }
    ],
    "objects": [{"type": "baseball", "color": "white"}]
}
```

**Why raw data?** Grok generates its own vocabulary:
- `emotional_state: "happy"` → Grok chooses: "joyfully", "gleefully", "with enthusiasm", etc.
- `position: {x: 0, y: 2}` → Grok infers: "nearby", "across the field", "side by side"
- `theme: "baseball"` → Grok creates: "diamond", "ballpark", "pitcher's mound"

This eliminates author bias from predefined word lists and increases training data diversity.

#### 2. Pre-mapped Metadata (for Template Fallback)

When the Grok API is unavailable, we use `extract_scene_metadata()` which includes pre-computed vocabulary:

```python
{
    "actors": [
        {
            "emotional_state": "happy",
            "emotional_adverb": "happily",      # Pre-mapped
            "emotional_adjective": "excited",   # Pre-mapped
            "action_timing": [
                {"action": "idle", "duration_phrase": "briefly", "timing_phrase": ""}
            ]
        }
    ],
    "spatial_relationships": [
        {"actor1_idx": 0, "actor2_idx": 1, "relationship": "across from"}
    ]
}
```

**Static mappings used only for templates:**
- `EMOTION_ADVERBS`: Maps facial expressions to adverb lists
- `EMOTION_ADJECTIVES`: Maps expressions to adjective lists
- `VERB_SYNONYMS`: Maps action verbs to synonym lists
- `get_temporal_phrase()`: Generates duration/timing phrases
- `infer_spatial_relationship()`: Computes spatial context from positions

---

## Complete Data Flow Example

### Step 1: Scene Generation

```python
# In dataset_generator.py
story_engine = StoryGenerator()
scene = story_engine.generate_random_scene()
```

**Generated Scene:**
```python
Scene(
    duration=5.0,
    theme="baseball",
    actors=[
        Actor(
            id="pitcher",
            actor_type=ActorType.HUMAN,
            color="dark blue",
            initial_position=Position(x=0, y=-2.0),
            actions=[(0.0, ActionType.IDLE), (1.0, ActionType.PITCHING)],
            team="team1"
        ),
        Actor(
            id="batter",
            actor_type=ActorType.HUMAN,
            color="red",
            initial_position=Position(x=2, y=-2.0),
            actions=[(0.0, ActionType.IDLE), (1.5, ActionType.BATTING)],
            team="team2"
        ),
    ],
    objects=[
        SceneObject(type=ObjectType.BASEBALL, ...),
        SceneObject(type=ObjectType.BAT, ...),
    ],
    description="",  # Empty - will be filled next
)
```

### Step 2: Extract Metadata

```python
scene_prompt_generator = ScenePromptGenerator()
metadata = scene_prompt_generator.extract_scene_metadata(scene)
```

**Extracted Metadata:**
```json
{
    "duration": 5.0,
    "theme": "baseball",
    "num_actors": 2,
    "actors": [
        {"type": "human", "color": "dark blue", "actions": ["idle", "pitching"], "team": "team1"},
        {"type": "human", "color": "red", "actions": ["idle", "batting"], "team": "team2"}
    ],
    "objects": [
        {"type": "baseball", "color": "white"},
        {"type": "bat", "color": "brown"}
    ],
    "has_teams": true
}
```

### Step 3: Generate Natural Language Prompt

```python
text_prompt = scene_prompt_generator.generate_prompt_from_scene(scene)
scene.description = text_prompt
```

**With Grok API available:**
```
"A red batter prepares to swing as the dark blue pitcher winds up and throws the baseball."
```

**With Template Fallback (no API):**
```
"A human stands still then pitches and a human stands still then swings a bat in a baseball diamond."
```

### Step 4: Generate Motion Tensors

The same `Scene` object is used to create motion data:

```python
# Motion tensor: [250 frames, 6 actors, 20 joint values]
motion_tensor = torch.zeros((target_frames, max_actors, 20))

# For each actor, convert actions to joint positions
for actor_idx, actor in enumerate(scene.actors):
    for frame_idx in range(target_frames):
        current_action = get_action_at_frame(actor, frame_idx)
        joint_positions = action_to_joint_positions(current_action)
        motion_tensor[frame_idx, actor_idx, :] = joint_positions
```

### Step 5: Final Training Sample

```python
sample = {
    "text_prompt": "A red batter prepares to swing as the dark blue pitcher winds up...",
    "motion": motion_tensor,          # [250, 6, 20]
    "actions": action_tensor,         # [250, 6]
    "face": face_tensor,              # [250, 6, 7]
    "camera": camera_tensor,          # [250, 9]
    "actor_types": [0, 0, 0, 0, 0, 0], # All humans
    "num_actors": 2,
}
```

**The text prompt describes EXACTLY what's in the motion data.**

---

## LLM Usage Summary

| Component | LLM Used | Purpose | When Called |
|-----------|----------|---------|-------------|
| `LLMStoryGenerator` | Grok | Generate creative scene scripts | ~20% of samples (for variety) |
| `ScenePromptGenerator` | Grok | Describe existing scene in natural language | Every sample |
| Template Fallback | None | Rule-based description | When API unavailable |

### LLM Ratio Configuration

```python
generate_dataset(
    llm_ratio=0.2,      # 20% of scenes created by Grok for variety
    use_mock_llm=False, # Use real Grok API
)
```

- **80% of scenes**: Procedural generation (fast, deterministic)
- **20% of scenes**: LLM-generated scripts (creative, varied)
- **100% of prompts**: Generated from actual scene metadata (guaranteed alignment)

---

## Template Fallback System

When Grok API is unavailable, `ScenePromptGenerator._generate_from_template()` creates prompts using **all** scene metadata:

### Data Used in Template Generation

| Metadata Field | How It's Used |
|----------------|---------------|
| `actor.type` | Actor noun: "human", "robot", "alien" |
| `actor.color` | Color prefix: "**red** human", "**green** alien" |
| `actor.team` | Team context: "from one team", "from the opposing team" |
| `actor.actions` | Verb phrases: "runs then kicks", "stands still, pitches, and throws" |
| `actor.emotional_state` | Emotional adverbs: "happily walks", "angrily punches" |
| `actor.action_timing` | Temporal phrases: "briefly pauses", "for a while" |
| `actor.position` | Spatial context: "near", "beside", "across from" |
| `objects[].type` | Object phrase: "with a baseball", "with a soccer ball" |
| `theme` + `objects` | Environment: inferred from both theme AND objects |

### Prompt Styles

The generator supports 4 distinct styles for variety:

| Style | Description | Example |
|-------|-------------|---------|
| `SIMPLE` | Short, direct | "A red human walks in a park." |
| `ACTION_FOCUSED` | Emphasizes actions | "The excited red player sprints and kicks while a blue defender rushes to block." |
| `NARRATIVE` | Story-like | "The scene unfolds in a stadium where an excited striker races toward goal." |
| `DESCRIPTIVE` | Rich detail | "In a sunlit park, a cheerful red human happily strolls for a moment before waving enthusiastically." |

### Verb Synonyms

For variety, actions are mapped to synonyms:

| Action | Synonyms |
|--------|----------|
| walk | strolls, paces, ambles, wanders, meanders |
| run | sprints, dashes, races, rushes, bolts |
| kick | boots, strikes, sends, launches |
| wave | gestures, signals, greets, beckons |

### Example Template Output

**Scene metadata:**
```json
{
  "theme": "baseball",
  "actors": [
    {"type": "human", "color": "dark blue", "actions": ["idle", "pitching"], "team": "team1", "emotional_state": "excited"},
    {"type": "human", "color": "red", "actions": ["idle", "batting"], "team": "team2", "emotional_state": "focused"}
  ],
  "objects": [{"type": "baseball", "color": "white"}],
  "has_teams": true,
  "spatial_relationships": [{"actor1_idx": 0, "actor2_idx": 1, "relationship": "across from"}]
}
```

**Template output (DESCRIPTIVE style):**
```
"An excited dark blue human from one team pauses briefly then pitches eagerly,
across from, a focused red human from the opposing team waits then bats
with a baseball in a ballpark."
```

### More Examples

| Scene Type | Template Output |
|------------|-----------------|
| Soccer | "An excited red player from one team sprints then boots the ball while a determined blue defender from the opposing team rushes to intercept with a soccer ball in a pitch." |
| Space | "A curious green alien ambles then waves happily and a calm silver robot stands still then claps with a ufo in a space station." |
| Nature | "A relaxed brown human strolls peacefully for a while then looks around in a forest." |

---

## Configuration

### Environment Variables

```bash
# Required for Grok API
export GROK_API_KEY="xai-..."

# Optional: Use mock LLM for testing (no API calls)
export USE_MOCK_LLM=true
```

### Generation Parameters

```python
generate_dataset(
    num_samples=50000,    # Total samples
    llm_ratio=0.2,        # 20% LLM-generated scenes
    use_mock_llm=False,   # Use real API
    target_frames=250,    # 10 seconds at 25 FPS
    max_actors=6,
)
```

---

## Debugging & Verification

### Verify Prompt-Motion Alignment

```python
from src.data_gen.dataset_generator import generate_dataset

# Generate a small test set with verbose output
generate_dataset(
    output_path="test_alignment.pt",
    num_samples=5,
    verbose=True,
)
```

Look for output like:
```
[ScenePrompt] Generated: A red batter prepares to swing as the dark blue pitcher...
```

### Check Cached Prompts

```python
from src.data_gen.prompt_generator import ScenePromptGenerator

gen = ScenePromptGenerator(verbose=True)
print(f"Cached prompts: {gen.get_stats()['cached_prompts']}")
```

---

## Multi-Prompt Generation

For training data augmentation, the same scene can be described in multiple valid ways:

```python
from src.data_gen.prompt_generator import ScenePromptGenerator

gen = ScenePromptGenerator()
prompts = gen.generate_prompts_from_scene(scene, num_prompts=3)

# Returns 3 unique prompts with different styles:
# 1. "A red human walks in a park."  (SIMPLE)
# 2. "The scene unfolds in a park where a cheerful red human strolls..."  (NARRATIVE)
# 3. "In a sunlit park, an excited red human happily ambles..."  (DESCRIPTIVE)
```

This increases training diversity without generating new motion data.

---

## Summary

| Aspect | How It Works |
|--------|--------------|
| **Scene Source** | `StoryGenerator` (procedural) or `LLMStoryGenerator` (Grok) |
| **Prompt Source** | `ScenePromptGenerator.generate_prompt_from_scene()` |
| **Multi-Prompt** | `ScenePromptGenerator.generate_prompts_from_scene()` for augmentation |
| **Alignment** | Prompt is generated FROM scene metadata, guaranteeing match |
| **LLM Model** | Grok (`grok-4-1-fast`) via X.AI API |
| **Fallback** | Template-based generation when API unavailable |
| **Styles** | SIMPLE, ACTION_FOCUSED, NARRATIVE, DESCRIPTIVE |
| **Enrichment** | Emotional adverbs, temporal phrases, spatial relationships, verb synonyms |
| **Caching** | Scene metadata hashed → prompt cached (up to 5000 entries) |

