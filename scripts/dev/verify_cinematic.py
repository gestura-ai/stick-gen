import os
import sys

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data_gen.renderer import Renderer, RenderStyle
from src.data_gen.schema import ActionType, Actor, Position, Scene


def verify_cinematic():
    print("Verifying Cinematic Rendering...")

    # Create a simple scene
    actor = Actor(
        id="actor1",
        initial_position=Position(x=0, y=0),
        actions=[(0.0, ActionType.WALK), (2.0, ActionType.WAVE)],
        color="blue",
    )

    scene = Scene(
        description="A test scene for cinematic rendering",
        actors=[actor],
        duration=4.0,
        theme="park",
    )

    # Render with cinematic mode
    renderer = Renderer(width=640, height=480, style=RenderStyle.NORMAL)
    output_path = "cinematic_test.mp4"

    try:
        renderer.render_scene(scene, output_path, camera_mode="static", cinematic=True)
        print(f"Successfully rendered to {output_path}")

        # Check if file exists and has size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("Video file created successfully.")
        else:
            print("Error: Video file not created or empty.")

    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_cinematic()
