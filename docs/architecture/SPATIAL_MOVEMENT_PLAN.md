# Spatial Movement & Model Improvement Plan

## Executive Summary

This plan addresses the critical missing feature: **actors don't actually move through space**. Currently, actors animate in place (limbs move but position is static). This plan adds realistic spatial movement, object scaling, and state-of-the-art embeddings.

## Problem Statement

### Current Issues
1. **No spatial translation**: Actors with WALK/RUN actions stay in one spot
2. **Unrealistic object scales**: No standardized size relationships
3. **Static positions**: `initial_position` never changes during animation
4. **Suboptimal embeddings**: Using Qwen2.5-1.5B-Instruct (not top-tier for embeddings)

### User Expectation
> "When a man runs he runs from one spot to another and the scene needs to make you sense this"

## Research Findings (December 2025)

### Human Movement Speeds
- **Walking**: 1.2-1.4 m/s (average: 1.3 m/s)
- **Jogging**: 3-5 m/s
- **Running**: 5-8 m/s
- **Sprinting**: 8-12 m/s (elite: 12.4 m/s - Usain Bolt)

### Object Scales (relative to human height = 1.7m)
- **Human**: 1.7m height (our stick figure scale: 2.5 units head-to-toe)
- **Table**: 0.75m height (0.44x human = 1.1 units)
- **Chair**: 0.45m height (0.26x human = 0.65 units)
- **Car**: 1.5m height (0.88x human = 2.2 units)
- **Tree**: 5-15m height (3-9x human = 7.5-22.5 units)
- **Building**: 10-50m height (6-30x human = 15-75 units)
- **Spaceship**: 10-30m (6-18x human = 15-45 units)

### State-of-the-Art Embeddings (MTEB Leaderboard Dec 2025)

**Top 3 Models:**
1. **Google Gemini Embedding** (gemini-embedding-001) - #1 on MTEB, API-based
2. **Qwen3-Embedding-8B** - Top open-source, 8B parameters
3. **llama-embed-nemotron-8b** - NVIDIA, excellent performance

**Current Model:**
- Qwen2.5-1.5B-Instruct (1536-dim) - Good but not optimized for embeddings

**Recommendation:**
- **Option A**: Google Gemini Embedding API (best quality, requires API key)
- **Option B**: Qwen3-Embedding-8B (best open-source, larger model)
- **Option C**: Keep Qwen2.5-1.5B (free, local, "good enough")

**Note on Grok-4**: xAI's Grok-4 is a reasoning/chat model, not an embedding model. Your API key can be used for other purposes but not for embeddings.

## Proposed Solution

### Phase 1: Add Spatial Movement System

#### 1.1 Schema Changes (`schema.py`)

**Add to Actor class:**
```python
class Actor(BaseModel):
    id: str
    actor_type: ActorType = ActorType.HUMAN
    color: str = "black"
    initial_position: Position
    actions: List[Tuple[float, ActionType]] = []
    team: Optional[str] = None
    scale: float = 1.0
    # NEW FIELDS:
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) in units/second
    movement_path: Optional[List[Tuple[float, Position]]] = None  # (time, position) waypoints
```

**Add action-to-velocity mapping:**
```python
ACTION_VELOCITIES = {
    ActionType.WALK: 1.3,      # m/s
    ActionType.RUN: 5.0,       # m/s
    ActionType.RUNNING_BASES: 6.0,  # Faster running
    ActionType.SPRINT: 8.0,    # m/s
    ActionType.IDLE: 0.0,
    ActionType.SIT: 0.0,
    ActionType.STAND: 0.0,
    # ... all other actions default to 0.0
}
```

#### 1.2 Renderer Changes (`renderer.py`)

**Update StickFigure class:**
```python
class StickFigure:
    def __init__(self, actor: Actor):
        self.id = actor.id
        self.actor_type = actor.actor_type
        self.color = actor.color
        self.initial_pos = np.array([actor.initial_position.x, actor.initial_position.y])
        self.pos = self.initial_pos.copy()  # Current position (dynamic)
        self.scale = actor.scale
        self.actions = actor.actions
        self.current_action = ActionType.IDLE
        self.movement_path = actor.movement_path or []
        
    def update_position(self, t: float, dt: float = 0.04):
        """Update position based on current action and time"""
        # If movement_path exists, interpolate
        if self.movement_path:
            self.pos = self._interpolate_path(t)
        else:
            # Calculate velocity from current action
            velocity = ACTION_VELOCITIES.get(self.current_action, 0.0)
            if velocity > 0:
                # Move in a direction (for now, right)
                # TODO: Add direction parameter
                self.pos[0] += velocity * dt
```

#### 1.3 Story Engine Changes (`story_engine.py`)

**Generate movement paths for actors:**
```python
def _generate_baseball_scene(self, num_actors: int, duration: float):
    # ... existing code ...
    
    # Add movement for base runners
    for i in range(num_runners):
        # Create path: home → 1st → 2nd → 3rd → home
        movement_path = [
            (0.0, Position(x=-2.0, y=-2.0)),    # Start at home
            (2.0, Position(x=2.0, y=-2.0)),     # Run to 1st base
            (4.0, Position(x=2.0, y=2.0)),      # Run to 2nd base
            (6.0, Position(x=-2.0, y=2.0)),     # Run to 3rd base
            (8.0, Position(x=-2.0, y=-2.0)),    # Run home
        ]
        actors.append(Actor(
            id=f"runner_{i}",
            initial_position=Position(x=-2.0, y=-2.0),
            actions=[(0.0, ActionType.RUNNING_BASES)],
            movement_path=movement_path
        ))
```

### Phase 2: Realistic Object Scales

**Update object generation with realistic scales:**
```python
OBJECT_SCALES = {
    ObjectType.TABLE: 1.1,      # 0.75m / 0.68m per unit
    ObjectType.CHAIR: 0.65,     # 0.45m
    ObjectType.CAR: 2.2,        # 1.5m
    ObjectType.TREE: 15.0,      # 10m (medium tree)
    ObjectType.BUILDING: 30.0,  # 20m (small building)
    ObjectType.SPACESHIP: 25.0, # 17m
    ObjectType.PLANET: 50.0,    # Background object
    ObjectType.BASEBALL: 0.1,   # Small
    ObjectType.FOOD: 0.3,       # Small
}
```

### Phase 3: Upgrade Embeddings

**Three options (in order of quality):**

#### Option A: Google Gemini Embedding (BEST)
- **Pros**: #1 on MTEB leaderboard, state-of-the-art quality
- **Cons**: Requires API key, costs money
- **Embedding dim**: 768
- **Implementation**: Use Google AI API

#### Option B: Qwen3-Embedding-8B (BEST OPEN-SOURCE)
- **Pros**: Top open-source model, free, local
- **Cons**: Larger model (8B params), slower inference
- **Embedding dim**: 4096
- **Implementation**: Load from HuggingFace

#### Option C: Keep Qwen2.5-1.5B-Instruct (CURRENT)
- **Pros**: Already working, fast, free
- **Cons**: Not optimized for embeddings
- **Embedding dim**: 1536
- **Implementation**: No changes needed

**Recommendation**: Start with Option B (Qwen3-Embedding-8B) for best quality without API costs.

### Phase 4: Model Architecture Improvements

**Current model**: 5.2M parameters (d_model=256, layers=6, heads=8)

**Proposed improvements for position prediction:**

1. **Increase output dimension**: 
   - Current: 20 (10 joints × 2 coords)
   - New: 22 (10 joints × 2 coords + 1 position × 2 coords)

2. **Add position prediction head**:
   ```python
   self.position_decoder = nn.Sequential(
       nn.Linear(d_model, d_model // 2),
       nn.GELU(),
       nn.Dropout(dropout),
       nn.Linear(d_model // 2, 2)  # (x, y) position
   )
   ```

3. **Consider increasing capacity**:
   - d_model: 256 → 384 (50% increase)
   - Parameters: 5.2M → 11M
   - Better for learning complex movement patterns

## Expected Improvements

### Realism Metrics
| Metric | Current | After Improvements | Improvement |
|--------|---------|-------------------|-------------|
| Spatial Movement | 0/10 (static) | 9/10 (realistic) | +9 |
| Object Scale Accuracy | 5/10 (arbitrary) | 9/10 (realistic) | +4 |
| Embedding Quality | 7/10 (Qwen2.5) | 10/10 (Gemini/Qwen3) | +3 |
| Overall Realism | 7/10 | 9.5/10 | +2.5 |

### Training Data Changes
- **Current**: 5000 samples, static positions
- **New**: 5000-10000 samples, dynamic position trajectories
- **Training time**: 2-3x longer (more complex data)
- **Expected loss**: Similar or better with larger model

## Implementation Plan

### Step 1: Schema & Data Structure (1-2 hours)
- [ ] Add velocity and movement_path to Actor
- [ ] Define ACTION_VELOCITIES mapping
- [ ] Define OBJECT_SCALES mapping

### Step 2: Renderer Updates (2-3 hours)
- [ ] Implement update_position() method
- [ ] Add path interpolation
- [ ] Update get_pose() to use dynamic position

### Step 3: Story Engine Updates (3-4 hours)
- [ ] Generate movement paths for baseball (base running)
- [ ] Generate movement paths for space (exploring)
- [ ] Generate movement paths for other themes

### Step 4: Embedding Upgrade (1-2 hours)
- [ ] Choose embedding model (Gemini vs Qwen3 vs keep current)
- [ ] Update preprocess_embeddings.py
- [ ] Regenerate embeddings for all prompts

### Step 5: Model Architecture (1-2 hours)
- [ ] Add position prediction head
- [ ] Increase model capacity (optional)
- [ ] Update training script

### Step 6: Regenerate Training Data (2-4 hours)
- [ ] Run dataset_generator with new schema
- [ ] Generate 5000-10000 samples
- [ ] Verify position trajectories

### Step 7: Retrain Model (4-8 hours)
- [ ] Train with new data
- [ ] Monitor position prediction accuracy
- [ ] Save best checkpoint

### Step 8: Test & Validate (1-2 hours)
- [ ] Generate test videos
- [ ] Verify spatial movement
- [ ] Verify object scales
- [ ] Compare before/after

**Total estimated time**: 15-25 hours

## Questions for You

1. **Embedding model choice**: Which option do you prefer?
   - A) Google Gemini (best quality, requires API key)
   - B) Qwen3-Embedding-8B (best open-source)
   - C) Keep current Qwen2.5-1.5B

2. **Model size**: Should we increase model capacity to 11M params for better position prediction?

3. **Training data size**: Keep 5000 samples or increase to 10000 for more diversity?

4. **Grok-4 API**: You mentioned having a Grok-4 API key - would you like to use it for anything specific in this project? (Note: It's not for embeddings, but could be used for prompt enhancement or other tasks)

Let me know your preferences and I'll proceed with implementation!


