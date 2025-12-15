"""
Phase 1: Action-Conditioned Generation - Schema Extensions

This template shows the modifications needed for src/data_gen/schema.py
to support per-frame action sequences.

INTEGRATION INSTRUCTIONS:
1. Add ACTION_TO_IDX and IDX_TO_ACTION mappings after ActionType enum
2. Add per_frame_actions field to Actor dataclass
3. Update data generation to create per-frame action sequences
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ============================================================================
# STEP 1: Add after ActionType enum definition (around line 60)
# ============================================================================

# Action to index mapping for embedding layer
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ActionType)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}

# Total number of actions (should be 60 based on current ActionType enum)
NUM_ACTIONS = len(ActionType)

print(f"Loaded {NUM_ACTIONS} action types for action-conditioned generation")


# ============================================================================
# STEP 2: Modify Actor dataclass (around line 130)
# ============================================================================

@dataclass
class Actor:
    """
    Actor in the scene with action sequences
    
    MODIFICATIONS FOR PHASE 1:
    - Added per_frame_actions field for frame-by-frame action control
    """
    id: int
    actor_type: ActorType
    color: str
    initial_position: Position
    actions: List[Tuple[ActionType, float]]  # (action, duration) - existing field
    team: Optional[int] = None
    scale: float = 1.0
    velocity: Optional[Tuple[float, float]] = None
    movement_path: Optional[List[Position]] = None
    
    # NEW FIELD FOR PHASE 1: Per-frame action sequences
    per_frame_actions: Optional[List[ActionType]] = None
    """
    Per-frame action sequence for action-conditioned generation.
    Length should match sequence length (250 frames for 10s @ 25fps).
    
    If None, will be generated from self.actions during data generation.
    If provided, overrides self.actions for fine-grained control.
    
    Example:
        # Smooth transition: WALK (0-5s) → RUN (5-10s)
        per_frame_actions = [ActionType.WALK] * 125 + [ActionType.RUN] * 125
        
        # Complex sequence: IDLE → WALK → JUMP → WALK → IDLE
        per_frame_actions = (
            [ActionType.IDLE] * 50 +
            [ActionType.WALK] * 75 +
            [ActionType.JUMP] * 50 +
            [ActionType.WALK] * 50 +
            [ActionType.IDLE] * 25
        )
    """


# ============================================================================
# STEP 3: Helper function to generate per-frame actions from action list
# ============================================================================

def generate_per_frame_actions(
    actions: List[Tuple[ActionType, float]],
    fps: int = 25,
    total_duration: float = 10.0
) -> List[ActionType]:
    """
    Convert action list with durations to per-frame action sequence.
    
    Args:
        actions: List of (ActionType, duration_seconds) tuples
        fps: Frames per second (default: 25)
        total_duration: Total sequence duration in seconds (default: 10.0)
    
    Returns:
        List of ActionType, one per frame (length = fps * total_duration)
    
    Example:
        >>> actions = [(ActionType.WALK, 5.0), (ActionType.RUN, 5.0)]
        >>> per_frame = generate_per_frame_actions(actions, fps=25, total_duration=10.0)
        >>> len(per_frame)
        250
        >>> per_frame[0]
        <ActionType.WALK: 'walk'>
        >>> per_frame[125]
        <ActionType.RUN: 'run'>
    """
    total_frames = int(fps * total_duration)
    per_frame_actions = []
    
    current_frame = 0
    for action, duration in actions:
        num_frames = int(fps * duration)
        
        # Ensure we don't exceed total frames
        num_frames = min(num_frames, total_frames - current_frame)
        
        per_frame_actions.extend([action] * num_frames)
        current_frame += num_frames
        
        if current_frame >= total_frames:
            break
    
    # Pad with last action if needed
    if len(per_frame_actions) < total_frames:
        last_action = actions[-1][0] if actions else ActionType.IDLE
        per_frame_actions.extend([last_action] * (total_frames - len(per_frame_actions)))
    
    # Truncate if too long
    per_frame_actions = per_frame_actions[:total_frames]
    
    return per_frame_actions


# ============================================================================
# STEP 4: Update data generation to use per-frame actions
# ============================================================================

def create_actor_with_per_frame_actions(
    actor_id: int,
    actor_type: ActorType,
    actions: List[Tuple[ActionType, float]],
    **kwargs
) -> Actor:
    """
    Create Actor with automatically generated per-frame actions.
    
    Example:
        >>> actor = create_actor_with_per_frame_actions(
        ...     actor_id=0,
        ...     actor_type=ActorType.STICK_FIGURE,
        ...     actions=[(ActionType.WALK, 3.0), (ActionType.RUN, 7.0)],
        ...     color='blue',
        ...     initial_position=Position(x=0, y=0)
        ... )
        >>> len(actor.per_frame_actions)
        250
    """
    per_frame_actions = generate_per_frame_actions(actions)
    
    return Actor(
        id=actor_id,
        actor_type=actor_type,
        actions=actions,
        per_frame_actions=per_frame_actions,
        **kwargs
    )

