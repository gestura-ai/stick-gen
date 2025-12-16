#!/usr/bin/env python3
"""
Test Expression Transitions (Phase 5.2)

Tests smooth transitions between facial expressions when actions change.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_gen.renderer import Renderer
from src.data_gen.schema import ActionType, Position, Scene
from src.data_gen.story_engine import create_actor_with_expression


def test_expression_transition():
    """Test smooth transition between expressions"""
    print("Testing expression transitions...")

    # Create an actor that changes actions (and expressions) during the scene
    # Start HAPPY (waving), then transition to SAD (crying), then to EXCITED (celebrating)
    actor = create_actor_with_expression(
        actor_id="transition_actor",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.WAVE),  # HAPPY expression
            (2.0, ActionType.CRY),  # Transition to SAD at 2s
            (4.0, ActionType.CELEBRATE),  # Transition to EXCITED at 4s
        ],
        color="black",
    )

    # Manually update the actor's actions to trigger transitions
    # The renderer will detect action changes and update expressions

    scene = Scene(
        duration=6.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing smooth expression transitions: HAPPY → SAD → EXCITED",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "tests/outputs/test_expression_transition.mp4")
    print("✓ Saved to tests/outputs/test_expression_transition.mp4")


def test_multiple_actors_transitions():
    """Test multiple actors with different transition timings"""
    print("\nTesting multiple actors with staggered transitions...")

    # Actor 1: Quick transitions
    actor1 = create_actor_with_expression(
        actor_id="actor1",
        position=Position(x=-2, y=0),
        actions=[
            (0.0, ActionType.IDLE),  # NEUTRAL
            (1.0, ActionType.WAVE),  # HAPPY
            (2.0, ActionType.PUNCH),  # ANGRY
            (3.0, ActionType.CELEBRATE),  # EXCITED
        ],
        color="blue",
    )

    # Actor 2: Slower transitions
    actor2 = create_actor_with_expression(
        actor_id="actor2",
        position=Position(x=2, y=0),
        actions=[
            (0.0, ActionType.CELEBRATE),  # EXCITED
            (2.5, ActionType.CRY),  # SAD
            (5.0, ActionType.WAVE),  # HAPPY
        ],
        color="red",
    )

    scene = Scene(
        duration=7.0,
        actors=[actor1, actor2],
        objects=[],
        background_color="white",
        description="Two actors with different expression transition timings",
    )

    renderer = Renderer(width=800, height=480)
    renderer.render_scene(scene, "tests/outputs/test_multiple_transitions.mp4")
    print("✓ Saved to tests/outputs/test_multiple_transitions.mp4")


def test_rapid_transitions():
    """Test rapid expression changes"""
    print("\nTesting rapid expression transitions...")

    actor = create_actor_with_expression(
        actor_id="rapid_actor",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.IDLE),  # NEUTRAL
            (0.5, ActionType.LOOKING_AROUND),  # SURPRISED
            (1.0, ActionType.WAVE),  # HAPPY
            (1.5, ActionType.PUNCH),  # ANGRY
            (2.0, ActionType.CRY),  # SAD
            (2.5, ActionType.CELEBRATE),  # EXCITED
            (3.0, ActionType.IDLE),  # NEUTRAL
        ],
        color="purple",
    )

    scene = Scene(
        duration=4.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Rapid expression transitions every 0.5 seconds",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "tests/outputs/test_rapid_transitions.mp4")
    print("✓ Saved to tests/outputs/test_rapid_transitions.mp4")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPRESSION TRANSITION TESTS (Phase 5.2)")
    print("=" * 60)
    print()

    test_expression_transition()
    test_multiple_actors_transitions()
    test_rapid_transitions()

    print()
    print("=" * 60)
    print("All transition tests complete!")
    print("=" * 60)
    print("\nGenerated videos:")
    print("  - tests/outputs/test_expression_transition.mp4")
    print("  - tests/outputs/test_multiple_transitions.mp4")
    print("  - tests/outputs/test_rapid_transitions.mp4")
