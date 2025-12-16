#!/usr/bin/env python3
"""
Integration Test: All Features Working Together

Tests that all implemented features work correctly together:
- Phase 5: Basic Facial Expressions
- Phase 6: Expression Transitions
- Phase 7: Speech Animation
- Existing: Action system, rendering, scene generation

This test creates a complex scene with multiple actors performing
various actions with appropriate facial expressions and speech animation.
"""

from src.data_gen.renderer import Renderer
from src.data_gen.schema import ActionType, Position, Scene
from src.data_gen.story_engine import create_actor_with_expression


def test_multi_actor_scene_with_expressions():
    """
    Test a complex scene with multiple actors, various actions,
    facial expressions, and speech animation all working together.
    """
    print("Testing multi-actor scene with all features...")

    # Actor 1: Greeting and talking
    actor1 = create_actor_with_expression(
        actor_id="greeter",
        position=Position(x=-2, y=0),
        actions=[
            (0.0, ActionType.WAVE),  # Wave with happy expression
            (2.0, ActionType.TALK),  # Talk with speech animation
            (5.0, ActionType.CLAP),  # Clap with happy expression
        ],
        color="blue",
    )

    # Actor 2: Emotional journey
    actor2 = create_actor_with_expression(
        actor_id="emotional",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.STAND),  # Neutral
            (2.0, ActionType.JUMP),  # Excited
            (4.0, ActionType.SIT),  # Neutral
            (6.0, ActionType.SHOUT),  # Excited with speech
        ],
        color="red",
    )

    # Actor 3: Athletic performance
    actor3 = create_actor_with_expression(
        actor_id="athlete",
        position=Position(x=2, y=0),
        actions=[
            (0.0, ActionType.RUN),  # Neutral
            (2.0, ActionType.KICK),  # Excited
            (4.0, ActionType.CELEBRATE),  # Happy
            (6.0, ActionType.SING),  # Happy with speech
        ],
        color="green",
    )

    scene = Scene(
        duration=8.0,
        actors=[actor1, actor2, actor3],
        objects=[],
        background_color="white",
        description="Integration test: Multiple actors with expressions, transitions, and speech",
    )

    renderer = Renderer(width=960, height=480)
    renderer.render_scene(scene, "tests/outputs/test_integration_multi_actor.mp4")
    print("✓ Multi-actor integration test complete: tests/outputs/test_integration_multi_actor.mp4")


def test_all_speech_types_in_sequence():
    """Test all speech types in a single scene with smooth transitions."""
    print("Testing all speech types in sequence...")

    actor = create_actor_with_expression(
        actor_id="speaker",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.TALK),  # Normal talking
            (2.0, ActionType.WHISPER),  # Quiet speech
            (4.0, ActionType.SHOUT),  # Loud speech
            (6.0, ActionType.SING),  # Musical speech
            (8.0, ActionType.TALK),  # Back to normal
        ],
        color="purple",
    )

    scene = Scene(
        duration=10.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="All speech types: TALK → WHISPER → SHOUT → SING → TALK",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "tests/outputs/test_integration_all_speech.mp4")
    print("✓ All speech types test complete: tests/outputs/test_integration_all_speech.mp4")


def test_expression_variety():
    """Test a variety of expressions through different actions."""
    print("Testing expression variety...")

    actor = create_actor_with_expression(
        actor_id="expressive",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.WAVE),  # Happy
            (1.5, ActionType.JUMP),  # Excited
            (3.0, ActionType.STAND),  # Neutral
            (4.5, ActionType.CELEBRATE),  # Happy
            (6.0, ActionType.KICK),  # Excited
            (7.5, ActionType.CLAP),  # Happy
        ],
        color="orange",
    )

    scene = Scene(
        duration=9.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Expression variety: Happy, Excited, Neutral transitions",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "tests/outputs/test_integration_expressions.mp4")
    print("✓ Expression variety test complete: tests/outputs/test_integration_expressions.mp4")


def test_complex_interaction():
    """Test complex interaction between two actors with speech and expressions."""
    print("Testing complex interaction...")

    # Actor 1: Initiates conversation
    actor1 = create_actor_with_expression(
        actor_id="speaker1",
        position=Position(x=-1.5, y=0),
        actions=[
            (0.0, ActionType.WAVE),  # Greet
            (1.0, ActionType.TALK),  # Start talking
            (3.0, ActionType.POINT),  # Point at something
            (4.0, ActionType.SHOUT),  # Get excited
            (6.0, ActionType.CLAP),  # Celebrate
        ],
        color="blue",
    )

    # Actor 2: Responds
    actor2 = create_actor_with_expression(
        actor_id="speaker2",
        position=Position(x=1.5, y=0),
        actions=[
            (0.0, ActionType.STAND),  # Listen
            (1.5, ActionType.WAVE),  # Wave back
            (2.5, ActionType.TALK),  # Respond
            (4.5, ActionType.JUMP),  # Get excited
            (6.0, ActionType.CELEBRATE),  # Celebrate together
        ],
        color="red",
    )

    scene = Scene(
        duration=8.0,
        actors=[actor1, actor2],
        objects=[],
        background_color="white",
        description="Complex interaction: Two actors conversing with expressions and speech",
    )

    renderer = Renderer(width=960, height=480)
    renderer.render_scene(scene, "tests/outputs/test_integration_interaction.mp4")
    print("✓ Complex interaction test complete: tests/outputs/test_integration_interaction.mp4")


if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATION TEST: All Features Working Together")
    print("=" * 70)

    test_multi_actor_scene_with_expressions()
    test_all_speech_types_in_sequence()
    test_expression_variety()
    test_complex_interaction()

    print("\n" + "=" * 70)
    print("All integration tests complete!")
    print("=" * 70)
    print("\nGenerated test videos:")
    print("  - tests/outputs/test_integration_multi_actor.mp4")
    print("  - tests/outputs/test_integration_all_speech.mp4")
    print("  - tests/outputs/test_integration_expressions.mp4")
    print("  - tests/outputs/test_integration_interaction.mp4")
