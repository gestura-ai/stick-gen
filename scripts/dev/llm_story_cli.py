import argparse
import json
import os
import sys

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data_gen.llm_story_engine import LLMStoryGenerator


def run_llm(
    provider: str, model: str | None = None, prompt: str = "A ninja fighting a robot"
) -> None:
    """Manual CLI helper to exercise LLMStoryGenerator backends.

    This is intentionally separate from the pytest suite so it can be used
    for ad-hoc local verification (e.g. Grok / Ollama connectivity).
    """
    print(f"Testing LLM Provider: {provider}")
    if model:
        print(f"Model: {model}")

    generator = LLMStoryGenerator(provider=provider, model=model)

    print(f"\nGenerating script for prompt: '{prompt}'...")
    try:
        script = generator.generate_script(prompt)
        print("\n--- Generated Script ---")
        print(json.dumps(script.dict(), indent=2))

        print("\nConverting to scenes...")
        scenes = generator.script_to_scenes(script)
        print(f"Generated {len(scenes)} scenes.")
        for i, scene in enumerate(scenes):
            print(
                f"Scene {i+1}: {scene.description} ({len(scene.actors)} actors, {scene.duration}s)"
            )

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM Story Generator")
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "grok", "ollama"],
        help="LLM Provider",
    )
    parser.add_argument("--model", type=str, help="Specific model name")
    parser.add_argument(
        "--prompt", type=str, default="A ninja fighting a robot", help="Story prompt"
    )

    args = parser.parse_args()

    run_llm(args.provider, args.model, args.prompt)


if __name__ == "__main__":
    main()
