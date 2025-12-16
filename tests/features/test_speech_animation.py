#!/usr/bin/env python3
"""
Test script for Phase 7: Speech Animation

Tests cyclic mouth movements for TALK, SHOUT, WHISPER, and SING actions.
"""

from src.data_gen.schema import ActionType, Position, Scene
from src.data_gen.story_engine import create_actor_with_expression
from src.data_gen.renderer import Renderer


def test_talk_animation():
    """Test normal talking animation (8 Hz cycle)"""
    print("Testing TALK animation...")

    actor = create_actor_with_expression(
        actor_id="talker",
        position=Position(x=0, y=0),
        actions=[(0.0, ActionType.TALK)],
        color="black",
    )

    scene = Scene(
        duration=4.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing TALK speech animation with 8 Hz cycle",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "test_speech_talk.mp4")
    print("✓ TALK animation test complete: test_speech_talk.mp4")


def test_shout_animation():
    """Test shouting animation (6 Hz cycle, wide mouth)"""
    print("Testing SHOUT animation...")

    actor = create_actor_with_expression(
        actor_id="shouter",
        position=Position(x=0, y=0),
        actions=[(0.0, ActionType.SHOUT)],
        color="black",
    )

    scene = Scene(
        duration=4.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing SHOUT speech animation with 6 Hz cycle and wide mouth",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "test_speech_shout.mp4")
    print("✓ SHOUT animation test complete: test_speech_shout.mp4")


def test_whisper_animation():
    """Test whispering animation (10 Hz cycle, small mouth)"""
    print("Testing WHISPER animation...")

    actor = create_actor_with_expression(
        actor_id="whisperer",
        position=Position(x=0, y=0),
        actions=[(0.0, ActionType.WHISPER)],
        color="black",
    )

    scene = Scene(
        duration=4.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing WHISPER speech animation with 10 Hz cycle and small mouth",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "test_speech_whisper.mp4")
    print("✓ WHISPER animation test complete: test_speech_whisper.mp4")


def test_sing_animation():
    """Test singing animation (4 Hz cycle, oval mouth)"""
    print("Testing SING animation...")

    actor = create_actor_with_expression(
        actor_id="singer",
        position=Position(x=0, y=0),
        actions=[(0.0, ActionType.SING)],
        color="black",
    )

    scene = Scene(
        duration=4.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing SING speech animation with 4 Hz cycle and oval mouth",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "test_speech_sing.mp4")
    print("✓ SING animation test complete: test_speech_sing.mp4")


def test_speech_transitions():
    """Test transitions between different speech types"""
    print("Testing speech transitions...")

    actor = create_actor_with_expression(
        actor_id="speaker",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.TALK),  # Normal talking
            (2.0, ActionType.SHOUT),  # Transition to shouting
            (4.0, ActionType.WHISPER),  # Transition to whispering
            (6.0, ActionType.SING),  # Transition to singing
        ],
        color="black",
    )

    scene = Scene(
        duration=8.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Testing speech transitions: TALK → SHOUT → WHISPER → SING",
    )

    renderer = Renderer(width=640, height=480)
    renderer.render_scene(scene, "test_speech_transitions.mp4")
    print("✓ Speech transitions test complete: test_speech_transitions.mp4")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 7: Speech Animation Tests")
    print("=" * 60)

    test_talk_animation()
    test_shout_animation()
    test_whisper_animation()
    test_sing_animation()
    test_speech_transitions()

    print("\n" + "=" * 60)
    print("All speech animation tests complete!")
    print("=" * 60)
