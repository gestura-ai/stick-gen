from src.data_gen.story_engine import StoryGenerator
from src.data_gen.renderer import Renderer


def main():
    print("Initializing Story Generator...")
    engine = StoryGenerator()

    print("Generating Random Scene...")
    scene = engine.generate_random_scene()
    print(f"Scene Description: {scene.description}")
    print(f"Duration: {scene.duration:.2f}s")
    for actor in scene.actors:
        print(
            f"  - Actor {actor.id}: {actor.actions[0][1]} at ({actor.initial_position.x:.2f}, {actor.initial_position.y:.2f})"
        )

    print("Rendering Scene...")
    renderer = Renderer()
    output_file = "output.mp4"
    renderer.render_scene(scene, output_file)

    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    main()
