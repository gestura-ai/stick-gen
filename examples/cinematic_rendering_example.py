"""
Cinematic Rendering Example

Demonstrates 2.5D perspective effects including:
- Perspective projection (foreshortening)
- Dynamic line width based on depth
- Z-sorting (painter's algorithm)

Usage:
    python examples/cinematic_rendering_example.py --output outputs/cinematic_demo.mp4
    python examples/cinematic_rendering_example.py --demo projection
    python examples/cinematic_rendering_example.py --demo comparison --output comparison.mp4
"""

import argparse
import sys
import math
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.schema import Scene, Actor, ActorType, Position, ActionType
from src.data_gen.renderer import Renderer


def demo_perspective_math():
    """Demonstrate the perspective projection mathematics."""
    print("\n" + "=" * 50)
    print("Perspective Projection Mathematics")
    print("=" * 50)

    # Camera parameters (from CinematicRenderer)
    focal_length = 10.0
    camera_z = -10.0

    print(f"\nCamera setup:")
    print(f"  • Focal length: {focal_length}")
    print(f"  • Camera Z position: {camera_z}")

    print("\nProjection formula: scale = focal_length / (z - camera_z)")

    # Test different Z-depths
    z_values = [-2.0, 0.0, 2.0, 5.0]
    base_point = (1.0, 1.0)  # Point at (1, 1) in 2D

    print(f"\nProjecting point ({base_point[0]}, {base_point[1]}) at various depths:")
    print("-" * 50)

    for z in z_values:
        dist = z - camera_z
        scale = focal_length / dist
        x_proj = base_point[0] * scale
        y_proj = base_point[1] * scale
        line_width = 2.0 * scale

        print(f"  Z={z:+.1f}: dist={dist:.1f}, scale={scale:.3f}")
        print(
            f"         projected=({x_proj:.3f}, {y_proj:.3f}), width={line_width:.3f}"
        )


def demo_z_depth_assignments():
    """Demonstrate Z-depth assignments for limbs."""
    print("\n" + "=" * 50)
    print("Z-Depth Assignments for Limbs")
    print("=" * 50)

    # Default Z-depths from CinematicRenderer
    limbs = [
        ("Torso", 0.0),
        ("Left Leg", -0.2),
        ("Right Leg", 0.2),
        ("Left Arm", -0.3),
        ("Right Arm", 0.3),
    ]

    print("\nPositive Z = Closer to camera")
    print("-" * 40)

    # Sort by Z to show draw order
    sorted_limbs = sorted(limbs, key=lambda x: x[1])

    print("\nDraw order (farthest to closest):")
    for i, (name, z) in enumerate(sorted_limbs, 1):
        indicator = "◀──" if z > 0 else "──▶" if z < 0 else "──●"
        print(f"  {i}. {name:12s} Z={z:+.1f} {indicator}")


def demo_z_sorting():
    """Demonstrate painter's algorithm Z-sorting."""
    print("\n" + "=" * 50)
    print("Z-Sorting (Painter's Algorithm)")
    print("=" * 50)

    # Simulated line data: (start, end, width, z)
    lines = [
        ("Torso", 0.0),
        ("L-Leg", -0.2),
        ("R-Leg", 0.2),
        ("L-Arm", -0.3),
        ("R-Arm", 0.3),
    ]

    print("\nOriginal order:")
    for name, z in lines:
        print(f"  {name}: Z={z:+.1f}")

    # Sort by Z ascending (farthest first)
    sorted_lines = sorted(lines, key=lambda x: x[1])

    print("\nSorted for rendering (farthest first):")
    for i, (name, z) in enumerate(sorted_lines, 1):
        print(f"  {i}. {name}: Z={z:+.1f}")

    print("\n✓ Elements drawn in this order ensure proper occlusion")


def create_demo_scene() -> Scene:
    """Create a scene for cinematic rendering demo."""
    actor = Actor(
        id="demo_actor",
        actor_type=ActorType.HUMAN,
        initial_position=Position(x=0.0, y=0.0),
        color="blue",
        actions=[
            (0.0, ActionType.WALK),
            (3.0, ActionType.JUMP),
            (5.0, ActionType.WAVE),
            (7.0, ActionType.DANCE),
        ],
    )

    return Scene(
        description="Cinematic rendering demonstration",
        actors=[actor],
        duration=10.0,
        theme="default",
    )


def demo_render_comparison(output_path: str = None):
    """Compare standard vs cinematic rendering."""
    print("\n" + "=" * 50)
    print("Standard vs Cinematic Rendering Comparison")
    print("=" * 50)

    scene = create_demo_scene()

    print("\nStandard Rendering (2D):")
    print("  • All limbs have same line width")
    print("  • No depth ordering")
    print("  • Flat appearance")

    print("\nCinematic Rendering (2.5D):")
    print("  • Variable line widths based on depth")
    print("  • Proper occlusion handling (Z-sorting)")
    print("  • Subtle foreshortening effects")
    print("  • Natural depth perception")

    if output_path:
        print(f"\n📽️  Rendering comparison to {output_path}...")
        renderer = Renderer()
        # Render with cinematic=True
        renderer.render_scene(scene, output_path, camera_mode="static", cinematic=True)
        print(f"✅ Saved cinematic render to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cinematic Rendering Example")
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["projection", "depths", "sorting", "comparison", "all"],
        help="Demo mode",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional: render to video file"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Cinematic Rendering (2.5D) - by Gestura AI")
    print("=" * 50)

    demos = {
        "projection": demo_perspective_math,
        "depths": demo_z_depth_assignments,
        "sorting": demo_z_sorting,
        "comparison": lambda: demo_render_comparison(args.output),
    }

    if args.demo == "all":
        demo_perspective_math()
        demo_z_depth_assignments()
        demo_z_sorting()
        demo_render_comparison(args.output)
    else:
        if args.demo == "comparison":
            demo_render_comparison(args.output)
        else:
            demos[args.demo]()

    print("\n" + "=" * 50)
    print("Cinematic rendering demonstration complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
