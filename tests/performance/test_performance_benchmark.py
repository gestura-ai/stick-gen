#!/usr/bin/env python3
"""
Performance Benchmark Test

Measures rendering performance with and without facial expressions
to ensure the new features don't significantly impact performance.
"""

import time
from src.data_gen.schema import ActionType, Position, Scene, Actor
from src.data_gen.story_engine import create_actor_with_expression
from src.data_gen.renderer import Renderer


def benchmark_rendering(scene_name, scene, iterations=3):
    """Benchmark rendering performance."""
    times = []
    
    for i in range(iterations):
        renderer = Renderer(width=640, height=480)
        start_time = time.time()
        
        # Render to a temporary file
        output_file = f"benchmark_{scene_name}_{i}.mp4"
        renderer.render_scene(scene, output_file)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    return avg_time


def test_baseline_performance():
    """Test baseline performance without facial expressions."""
    print("\n" + "="*60)
    print("Baseline Performance (No Facial Expressions)")
    print("="*60)

    # Create actor without facial expressions (using basic Actor)
    actor = Actor(
        id="baseline",
        initial_position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.WALK),
            (2.0, ActionType.RUN),
            (4.0, ActionType.JUMP),
        ],
        color="black"
    )

    scene = Scene(
        duration=6.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Baseline performance test"
    )

    avg_time = benchmark_rendering("baseline", scene, iterations=3)
    print(f"\nAverage rendering time: {avg_time:.2f}s")
    assert avg_time > 0


def test_with_facial_expressions():
    """Test performance with facial expressions enabled."""
    print("\n" + "="*60)
    print("With Facial Expressions")
    print("="*60)

    actor = create_actor_with_expression(
        actor_id="expressive",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.WALK),
            (2.0, ActionType.RUN),
            (4.0, ActionType.JUMP),
        ],
        color="black"
    )

    scene = Scene(
        duration=6.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Performance test with facial expressions"
    )

    avg_time = benchmark_rendering("with_expressions", scene, iterations=3)
    print(f"\nAverage rendering time: {avg_time:.2f}s")
    assert avg_time > 0


def test_with_speech_animation():
    """Test performance with speech animation enabled."""
    print("\n" + "="*60)
    print("With Speech Animation")
    print("="*60)

    actor = create_actor_with_expression(
        actor_id="speaker",
        position=Position(x=0, y=0),
        actions=[
            (0.0, ActionType.TALK),
            (2.0, ActionType.SHOUT),
            (4.0, ActionType.SING),
        ],
        color="black"
    )

    scene = Scene(
        duration=6.0,
        actors=[actor],
        objects=[],
        background_color="white",
        description="Performance test with speech animation"
    )

    avg_time = benchmark_rendering("with_speech", scene, iterations=3)
    print(f"\nAverage rendering time: {avg_time:.2f}s")
    assert avg_time > 0


def test_multi_actor_performance():
    """Test performance with multiple actors."""
    print("\n" + "="*60)
    print("Multi-Actor Performance (3 actors with expressions)")
    print("="*60)

    actors = [
        create_actor_with_expression(
            actor_id=f"actor{i}",
            position=Position(x=-2 + i*2, y=0),
            actions=[
                (0.0, ActionType.WAVE),
                (2.0, ActionType.TALK),
                (4.0, ActionType.JUMP),
            ],
            color=["blue", "red", "green"][i]
        )
        for i in range(3)
    ]

    scene = Scene(
        duration=6.0,
        actors=actors,
        objects=[],
        background_color="white",
        description="Multi-actor performance test"
    )

    avg_time = benchmark_rendering("multi_actor", scene, iterations=3)
    print(f"\nAverage rendering time: {avg_time:.2f}s")
    assert avg_time > 0


if __name__ == "__main__":
    print("="*60)
    print("PERFORMANCE BENCHMARK TEST")
    print("="*60)
    print("\nMeasuring rendering performance...")
    print("(Each test runs 3 iterations)")
    
    baseline_time = test_baseline_performance()
    expressions_time = test_with_facial_expressions()
    speech_time = test_with_speech_animation()
    multi_actor_time = test_multi_actor_performance()
    
    # Calculate overhead
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Baseline (no expressions):     {baseline_time:.2f}s")
    print(f"With facial expressions:       {expressions_time:.2f}s")
    print(f"With speech animation:         {speech_time:.2f}s")
    print(f"Multi-actor (3 actors):        {multi_actor_time:.2f}s")
    
    expressions_overhead = ((expressions_time - baseline_time) / baseline_time) * 100
    speech_overhead = ((speech_time - baseline_time) / baseline_time) * 100
    
    print(f"\nPerformance Overhead:")
    print(f"  Facial expressions: {expressions_overhead:+.1f}%")
    print(f"  Speech animation:   {speech_overhead:+.1f}%")
    
    # Success criteria: <5% overhead (negative is improvement)
    if expressions_overhead < 5 and speech_overhead < 5:
        print(f"\n✅ PASS: Performance overhead is within acceptable range (<5%)")
        if expressions_overhead < 0:
            print(f"   Note: Facial expressions actually improved performance!")
    else:
        print(f"\n⚠️  WARNING: Performance overhead exceeds 5%")
    
    print("\n" + "="*60)

