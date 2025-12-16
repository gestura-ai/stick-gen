import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.inference.generator import InferenceGenerator


def main():
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

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
