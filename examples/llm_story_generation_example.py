"""
LLM Story Generation Example

Generate stick figure animation scripts using LLM backends (Grok, Ollama, Mock).

Usage:
    # Using mock backend (default, no API key needed)
    python examples/llm_story_generation_example.py --prompt "A ninja heist"

    # Using Grok backend (requires GROK_API_KEY in .env)
    python examples/llm_story_generation_example.py --provider grok --prompt "Dance battle"

    # Using Ollama backend (requires Ollama server running)
    python examples/llm_story_generation_example.py --provider ollama --model llama3

Environment Variables:
    GROK_API_KEY: API key for Grok (X.AI) backend
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.llm_story_engine import (
    LLM_AVAILABLE,
    LLMStoryGenerator,
    ScriptSchema,
)


def print_script(script: ScriptSchema):
    """Pretty print a generated script."""
    print(f"\n{'='*50}")
    print(f"📽️  {script.title}")
    print(f"{'='*50}")
    print(f"\n📝 Description: {script.description}")
    print(f"⏱️  Duration: {script.duration}s")

    print(f"\n👥 Characters ({len(script.characters)}):")
    for char in script.characters:
        print(f"   • {char['name']} - {char['role']}")

    print(f"\n🎬 Scenes ({len(script.scenes)}):")
    for i, scene in enumerate(script.scenes, 1):
        print(f"   {i}. {scene['description']}")
        print(f"      Actions: {scene['action_sequence']}")

    if script.camera:
        print(f"\n📷 Camera Keyframes: {len(script.camera)}")


def demo_mock_backend():
    """Demonstrate mock backend with different prompts."""
    print("\n" + "=" * 50)
    print("Mock Backend Demonstrations")
    print("=" * 50)

    generator = LLMStoryGenerator(provider="mock")

    prompts = ["A bank heist movie", "An epic dance battle", "A random adventure"]

    for prompt in prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        script = generator.generate_script(prompt)
        print_script(script)


def demo_script_to_scenes(prompt: str, provider: str = "mock"):
    """Demonstrate converting script to Scene objects."""
    print("\n" + "=" * 50)
    print("Script to Scene Conversion")
    print("=" * 50)

    generator = LLMStoryGenerator(provider=provider)

    print(f"\n🎯 Prompt: '{prompt}'")
    script = generator.generate_script(prompt)
    print_script(script)

    print("\n📐 Converting to Scene objects...")
    scenes = generator.script_to_scenes(script)

    print(f"\n✅ Generated {len(scenes)} scene(s):")
    for i, scene in enumerate(scenes, 1):
        print(f"\n   Scene {i}:")
        print(f"   • Description: {scene.description}")
        print(f"   • Duration: {scene.duration}s")
        print(f"   • Actors: {len(scene.actors)}")
        print(f"   • Camera Keyframes: {len(scene.camera_keyframes)}")

        for actor in scene.actors:
            print(
                f"     - {actor.id}: {len(actor.actions)} actions, color={actor.color}"
            )


def demo_provider_fallback():
    """Demonstrate fallback behavior when provider unavailable."""
    print("\n" + "=" * 50)
    print("Provider Fallback Demonstration")
    print("=" * 50)

    print(f"\n📊 LLM Libraries Available: {LLM_AVAILABLE}")

    # Try each provider
    for provider in ["mock", "grok", "ollama"]:
        print(f"\n🔌 Attempting provider: {provider}")
        generator = LLMStoryGenerator(provider=provider)
        backend_type = type(generator.backend).__name__
        print(f"   Active backend: {backend_type}")


def main():
    parser = argparse.ArgumentParser(description="LLM Story Generation Example")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A ninja sneaking into a bank",
        help="Text prompt for story generation",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "grok", "ollama"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., grok-4-1-fast, llama3)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="full",
        choices=["mock", "convert", "fallback", "full"],
        help="Demo mode",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional: save script as JSON"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("LLM Story Generation - by Gestura AI")
    print("=" * 50)

    if args.demo == "mock":
        demo_mock_backend()
    elif args.demo == "convert":
        demo_script_to_scenes(args.prompt, args.provider)
    elif args.demo == "fallback":
        demo_provider_fallback()
    else:  # full
        demo_provider_fallback()
        demo_mock_backend()
        demo_script_to_scenes(args.prompt, args.provider)

    # Generate and optionally save
    if args.output:
        print(f"\n💾 Generating and saving to {args.output}...")
        generator = LLMStoryGenerator(provider=args.provider, model=args.model)
        script = generator.generate_script(args.prompt)

        with open(args.output, "w") as f:
            json.dump(script.model_dump(), f, indent=2)

        print(f"✅ Saved script to {args.output}")

    print("\n" + "=" * 50)
    print("LLM Story Generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
