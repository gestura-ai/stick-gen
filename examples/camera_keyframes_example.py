"""
Camera Keyframes Example

Demonstrates camera movement definitions including Pan, Zoom, Track,
and CameraKeyframe scene integration.

Usage:
    python examples/camera_keyframes_example.py --output outputs/camera_demo.mp4
    python examples/camera_keyframes_example.py --mode pan --duration 5.0
    python examples/camera_keyframes_example.py --mode track --actor hero

Available modes: static, pan, zoom, track, combined
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.camera import (
    Camera,
    Pan,
    StaticCamera,
    Zoom,
)
from src.data_gen.renderer import Renderer
from src.data_gen.schema import (
    ActionType,
    Actor,
    ActorType,
    CameraKeyframe,
    Position,
    Scene,
)


def create_demo_scene() -> Scene:
    """Create a demo scene with actors for camera demonstrations."""
    hero = Actor(
        id="hero",
        actor_type=ActorType.HUMAN,
        initial_position=Position(x=-3.0, y=0.0),
        color="blue",
        actions=[
            (0.0, ActionType.WALK),
            (4.0, ActionType.WAVE),
            (6.0, ActionType.JUMP),
        ],
    )

    companion = Actor(
        id="companion",
        actor_type=ActorType.HUMAN,
        initial_position=Position(x=2.0, y=0.0),
        color="green",
        actions=[(0.0, ActionType.IDLE), (3.0, ActionType.DANCE)],
    )

    return Scene(
        description="Camera demonstration scene",
        actors=[hero, companion],
        duration=10.0,
        theme="default",
    )


def demo_static_camera():
    """Demonstrate static camera."""
    print("\n--- Static Camera Demo ---")
    camera = StaticCamera(x=0.0, y=0.0, zoom=1.0)

    # Camera state doesn't change over time
    for t in [0.0, 5.0, 10.0]:
        state = camera.get_state(t)
        print(f"  t={t:.1f}s: x={state.x:.2f}, y={state.y:.2f}, zoom={state.zoom:.2f}")


def demo_pan_camera():
    """Demonstrate pan movement."""
    print("\n--- Pan Camera Demo ---")
    pan = Pan(
        start_pos=(-4.0, 0.0),
        end_pos=(4.0, 0.0),
        start_time=0.0,
        duration=8.0,
        zoom=1.0,
    )

    # Camera pans from left to right
    for t in [0.0, 2.0, 4.0, 6.0, 8.0]:
        state = pan.get_state(t)
        if state:
            print(f"  t={t:.1f}s: x={state.x:.2f}, y={state.y:.2f}")


def demo_zoom_camera():
    """Demonstrate zoom movement."""
    print("\n--- Zoom Camera Demo ---")
    zoom = Zoom(
        center=(0.0, 0.0), start_zoom=0.8, end_zoom=2.0, start_time=0.0, duration=5.0
    )

    # Camera zooms in
    for t in [0.0, 1.0, 2.5, 4.0, 5.0]:
        state = zoom.get_state(t)
        if state:
            print(f"  t={t:.1f}s: zoom={state.zoom:.2f}")


def demo_camera_keyframes():
    """Demonstrate CameraKeyframe in scenes."""
    print("\n--- CameraKeyframe Scene Demo ---")

    scene = create_demo_scene()
    scene.camera_keyframes = [
        CameraKeyframe(frame=0, x=-2.0, y=0.0, zoom=1.0, interpolation="linear"),
        CameraKeyframe(frame=50, x=0.0, y=0.0, zoom=1.2, interpolation="smooth"),
        CameraKeyframe(frame=150, x=2.0, y=0.5, zoom=1.5, interpolation="smooth"),
        CameraKeyframe(frame=250, x=0.0, y=0.0, zoom=1.0, interpolation="linear"),
    ]

    print(f"  Scene: {scene.description}")
    print(f"  Actors: {len(scene.actors)}")
    print(f"  Camera Keyframes: {len(scene.camera_keyframes)}")
    for kf in scene.camera_keyframes:
        print(
            f"    Frame {kf.frame}: pos=({kf.x:.1f}, {kf.y:.1f}), "
            f"zoom={kf.zoom:.1f}, interp={kf.interpolation}"
        )


def demo_camera_controller():
    """Demonstrate Camera controller with multiple movements."""
    print("\n--- Camera Controller Demo ---")

    camera = Camera(width=10.0, height=10.0)

    # Add sequential movements
    camera.add_movement(Pan((-3, 0), (0, 0), 0.0, 3.0))  # Pan to center
    camera.add_movement(Zoom((0, 0), 1.0, 1.5, 3.0, 2.0))  # Then zoom in
    camera.add_movement(Pan((0, 0), (3, 0), 5.0, 3.0))  # Then pan right

    print("  Camera timeline:")
    for t in [0.0, 1.5, 3.0, 4.0, 6.0, 8.0]:
        camera.update(t)
        limits = camera.get_view_limits()
        print(
            f"    t={t:.1f}s: pos=({camera.state.x:.2f}, {camera.state.y:.2f}), "
            f"zoom={camera.state.zoom:.2f}, view=[{limits[0]:.1f}, {limits[1]:.1f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="Camera keyframes demonstration")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["static", "pan", "zoom", "keyframes", "controller", "all"],
        help="Camera demo mode",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional: render to video file"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Camera System Examples - by Gestura AI")
    print("=" * 50)

    demos = {
        "static": demo_static_camera,
        "pan": demo_pan_camera,
        "zoom": demo_zoom_camera,
        "keyframes": demo_camera_keyframes,
        "controller": demo_camera_controller,
    }

    if args.mode == "all":
        for demo_fn in demos.values():
            demo_fn()
    else:
        demos[args.mode]()

    if args.output:
        print(f"\n--- Rendering to {args.output} ---")
        scene = create_demo_scene()
        scene.camera_keyframes = [
            CameraKeyframe(frame=0, x=-2.0, y=0.0, zoom=1.0),
            CameraKeyframe(frame=125, x=0.0, y=0.0, zoom=1.3),
            CameraKeyframe(frame=250, x=2.0, y=0.0, zoom=1.0),
        ]
        renderer = Renderer()
        renderer.render_scene(scene, args.output, camera_mode="dynamic")
        print(f"✅ Saved to {args.output}")

    print("\n" + "=" * 50)
    print("Camera demonstrations complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
