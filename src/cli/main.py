import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.data_gen.dataset_generator import generate_dataset
from src.data_gen.parallax_augmentation import generate_parallax_for_dataset
from src.inference.generator import InferenceGenerator


def main_generate_animation(argv=None):
    """Default CLI: text -> stick-figure MP4."""
    parser = argparse.ArgumentParser(
        description="Stick Gen - Text to Stick Figure Animation"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the scene (e.g., 'A sports scene with a ball')",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="model_checkpoint.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--style",
        "-s",
        type=str,
        default="normal",
        choices=["normal", "sketch", "ink", "neon"],
        help="Rendering style",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Camera behavior",
    )
    parser.add_argument(
        "--story-mode",
        action="store_true",
        help="Use LLM Story Engine for script generation",
    )

    args = parser.parse_args(argv)

    print(f"Stick Gen: Generating animation for '{args.prompt}'...")
    print(
        f"Settings: Style={args.style}, Camera={args.camera}, Story Mode={args.story_mode}"
    )

    generator = InferenceGenerator(model_path=args.model)
    generator.generate(
        prompt=args.prompt,
        output_path=args.output,
        style=args.style,
        camera_mode=args.camera,
        use_llm=args.story_mode,
    )

    print(f"Success! Video saved to {args.output}")


def main_generate_data(argv=None):
    """Data generation & optional 2.5D parallax augmentation."""
    parser = argparse.ArgumentParser(
        description="Stick Gen - Dataset Generation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/medium.yaml",
        help="Path to data generation YAML config",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default=None,
        help="Override dataset .pt output path (defaults from config)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of base samples to generate",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable in-place motion augmentation inside dataset_generator",
    )
    parser.add_argument(
        "--augment-parallax",
        action="store_true",
        help="Generate 2.5D parallax PNG frames with Three.js",
    )
    parser.add_argument(
        "--views-per-motion",
        type=int,
        default=0,
        help="Number of parallax camera trajectories per actor motion",
    )
    parser.add_argument(
        "--frames-per-view",
        type=int,
        default=1,
        help=(
            "Number of rendered frames per camera trajectory; when >1, "
            "each view becomes a short camera path over the motion."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/2.5d_parallax",
        help="Output directory for parallax PNG frames",
    )
    parser.add_argument(
        "--parallax-node-script",
        type=str,
        default="src/data_gen/renderers/threejs_parallax_renderer.js",
        help="Path to the Node.js Three.js renderer script",
    )

    args = parser.parse_args(argv)

    augment_motion = not args.no_augment
    dataset_path = generate_dataset(
        config_path=args.config,
        num_samples=args.num_samples,
        output_path=args.dataset_output,
        augment=augment_motion,
    )

    if args.augment_parallax:
        views = args.views_per_motion or 1000
        frames_per_view = args.frames_per_view or 1
        print(
            f"Stick Gen: Generating parallax frames "
            f"(views_per_motion={views}, frames_per_view={frames_per_view}) "
            f"into {args.output}"
        )
        generate_parallax_for_dataset(
            dataset_path=dataset_path,
            output_dir=args.output,
            views_per_motion=views,
            node_script=args.parallax_node_script,
            frames_per_view=frames_per_view,
        )


def main():
    # Simple sub-command dispatch without breaking existing CLI:
    #   ./stick-gen \"prompt\" ...              -> animation
    #   ./stick-gen generate-data [options...] -> dataset + parallax
    if len(sys.argv) > 1 and sys.argv[1] == "generate-data":
        return main_generate_data(sys.argv[2:])
    return main_generate_animation(sys.argv[1:])


if __name__ == "__main__":
    main()
