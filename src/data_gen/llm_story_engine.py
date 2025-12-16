import json
import os
import random
from typing import Protocol

from pydantic import BaseModel

from .schema import ActionType, Actor, ActorType, CameraKeyframe, Position, Scene


class ScriptSchema(BaseModel):
    """Schema for LLM-generated scripts"""

    title: str
    description: str
    duration: float
    characters: list[dict]
    scenes: list[dict]
    actions: list[dict] = []
    camera: list[dict] = []


# Try importing LLM clients
try:
    import ollama
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM Import Error: {e}")
    LLM_AVAILABLE = False


class LLMBackend(Protocol):
    def generate_story(self, prompt: str) -> ScriptSchema: ...


class MockBackend:
    def generate_story(self, prompt: str) -> ScriptSchema:
        # Existing mock logic
        if "heist" in prompt.lower():
            return self._generate_heist_script()
        elif "dance" in prompt.lower():
            return self._generate_dance_battle_script()
        else:
            return self._generate_generic_script()

    def _generate_heist_script(self):
        return ScriptSchema(
            title="The Stick Heist",
            description="A bank heist gone wrong",
            duration=10.0,
            characters=[
                {"name": "Ninja", "role": "The Infiltrator"},
                {"name": "Guard", "role": "The Obstacle"},
            ],
            scenes=[
                {
                    "description": "Ninja sneaks past guard",
                    "action_sequence": "sneak, look_around",
                },
                {
                    "description": "Guard hears noise",
                    "action_sequence": "look_around, walk",
                },
                {"description": "Ninja attacks", "action_sequence": "jump, kick"},
                {"description": "Escape", "action_sequence": "run, jump"},
            ],
            actions=[],
            camera=[],
        )

    def _generate_dance_battle_script(self):
        return ScriptSchema(
            title="Stick Dance Off",
            description="An epic dance battle",
            duration=8.0,
            characters=[
                {"name": "Dancer1", "role": "Challenger"},
                {"name": "Dancer2", "role": "Defender"},
            ],
            scenes=[
                {"description": "Dancer1 starts", "action_sequence": "dance, wave"},
                {"description": "Dancer2 responds", "action_sequence": "dance, jump"},
                {"description": "Finale", "action_sequence": "dance, bow"},
            ],
            actions=[],
            camera=[],
        )

    def _generate_generic_script(self):
        return ScriptSchema(
            title="A Stick Story",
            description="A generic story",
            duration=5.0,
            characters=[{"name": "Hero", "role": "Protagonist"}],
            scenes=[
                {"description": "Hero enters", "action_sequence": "walk, wave"},
                {"description": "Hero acts", "action_sequence": "jump, run"},
                {"description": "Hero leaves", "action_sequence": "walk"},
            ],
            actions=[],
            camera=[],
        )


class GrokBackend:
    def __init__(self, model="grok-4-1-fast", fallback_to_mock=True, verbose=True):
        """
        Initialize Grok (X.AI) backend for LLM story generation.

        Args:
            model: Grok model name. Recommended: "grok-4-1-fast", "grok-3", or "grok-4"
            fallback_to_mock: If True, falls back to MockBackend on API errors (default: True)
            verbose: If True, prints detailed logging (default: True)
        """
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1"
        )
        self.model = model
        self.fallback_to_mock = fallback_to_mock
        self.verbose = verbose

        if self.verbose:
            api_key = os.getenv("GROK_API_KEY")
            if api_key:
                print(f"[Grok] Initialized with model: {model}")
                print(f"[Grok] API key: {api_key[:10]}...{api_key[-4:]}")
            else:
                print("[Grok] WARNING: GROK_API_KEY not set!")

    def generate_story(self, prompt: str) -> ScriptSchema:
        system_prompt = """You are a screenplay writer for a stick figure animation engine.
        Output strictly valid JSON matching this schema:
        {
          "title": "String",
          "description": "String",
          "duration": Float,
          "characters": [{"name": "String", "role": "String"}],
          "scenes": [{"description": "String", "action_sequence": "String (comma separated actions)"}],
          "actions": [],
          "camera": []
        }
        Available actions: walk, run, jump, wave, fight, dance, sneak, kick, punch.
        """

        try:
            if self.verbose:
                print(f"[Grok] Generating story for prompt: '{prompt[:50]}...'")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a script for: {prompt}"},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            # Ensure required fields
            if "actions" not in data:
                data["actions"] = []
            if "camera" not in data:
                data["camera"] = []

            script = ScriptSchema(**data)

            if self.verbose:
                print(f"[Grok] ✅ Successfully generated: '{script.title}'")

            return script

        except Exception as e:
            error_msg = str(e)

            # Provide detailed error diagnostics
            if "404" in error_msg:
                if "deprecated" in error_msg.lower():
                    print(f"[Grok] ❌ ERROR: Model '{self.model}' is deprecated")
                    print("[Grok] Suggestion: Use 'grok-3' or 'grok-4-1-fast' instead")
                elif "does not exist" in error_msg.lower():
                    print(
                        f"[Grok] ❌ ERROR: Model '{self.model}' not found or no access"
                    )
                    print("[Grok] Suggestion: Check your API key permissions")
                else:
                    print(f"[Grok] ❌ ERROR 404: {error_msg}")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print("[Grok] ❌ ERROR: Invalid or expired API key")
                print("[Grok] Get a new key from: https://console.x.ai/")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                print("[Grok] ❌ ERROR: Rate limit exceeded")
            else:
                print(f"[Grok] ❌ ERROR: {error_msg}")

            if self.fallback_to_mock:
                print("[Grok] ⚠️  Falling back to MockBackend")
                return MockBackend().generate_story(prompt)
            else:
                raise


class OllamaBackend:
    def __init__(self, model="llama3"):
        self.model = model

    def generate_story(self, prompt: str) -> ScriptSchema:
        system_prompt = """You are a screenplay writer for a stick figure animation engine.
        Output strictly valid JSON matching this schema:
        {
          "title": "String",
          "description": "String",
          "duration": Float,
          "characters": [{"name": "String", "role": "String"}],
          "scenes": [{"description": "String", "action_sequence": "String (comma separated actions)"}],
          "actions": [],
          "camera": []
        }
        Do not include markdown formatting or code blocks. Just raw JSON.
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a script for: {prompt}"},
                ],
            )
            content = response["message"]["content"]
            # Clean potential markdown
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            # Ensure required fields
            if "actions" not in data:
                data["actions"] = []
            if "camera" not in data:
                data["camera"] = []
            return ScriptSchema(**data)
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return MockBackend().generate_story(prompt)


class LLMStoryGenerator:
    def __init__(
        self, provider="mock", model=None, fallback_to_mock=True, verbose=True
    ):
        """
        Initialize LLM Story Generator with configurable backend.

        Args:
            provider: Backend provider - "mock", "grok", or "ollama"
            model: Model name (provider-specific)
            fallback_to_mock: If True, falls back to mock on errors (default: True)
            verbose: If True, prints detailed logging (default: True)
        """
        self.provider = provider
        self.verbose = verbose

        if provider == "grok" and LLM_AVAILABLE:
            if verbose:
                print("[LLMStoryGenerator] Using Grok backend")
            self.backend = GrokBackend(
                model=model or "grok-4-1-fast",
                fallback_to_mock=fallback_to_mock,
                verbose=verbose,
            )
        elif provider == "ollama" and LLM_AVAILABLE:
            if verbose:
                print("[LLMStoryGenerator] Using Ollama backend")
            self.backend = OllamaBackend(model or "llama3")
        else:
            if provider != "mock" and verbose:
                print(
                    f"[LLMStoryGenerator] Warning: Provider '{provider}' not available. Using mock."
                )
            self.backend = MockBackend()

    def generate_script(self, prompt: str) -> ScriptSchema:
        """Generate a script from a text prompt."""
        return self.backend.generate_story(prompt)

    def script_to_scenes(self, script: ScriptSchema) -> list[Scene]:
        """Convert a script into a list of Scene objects."""
        scenes = []

        # Create actors based on characters
        actors_map = {}
        for i, char in enumerate(script.characters):
            # Assign random colors and positions
            color = random.choice(["black", "blue", "red", "green", "purple"])
            pos = Position(x=random.uniform(-2, 2), y=0)
            actor = Actor(
                id=f"actor_{i}",
                actor_type=ActorType.HUMAN,
                initial_position=pos,
                color=color,
                actions=[],
            )
            actors_map[char["name"]] = actor

        # Process scenes
        current_time = 0.0
        for scene_data in script.scenes:
            actions = [a.strip() for a in scene_data["action_sequence"].split(",")]

            # Assign actions to actors (simplified: all actors do the sequence for now)
            # In a real engine, we'd parse who does what
            for actor in actors_map.values():
                for action_name in actions:
                    action_type = self._map_action(action_name)
                    if action_type:
                        actor.actions.append((current_time, action_type))
                        current_time += 2.0  # Assume 2 seconds per action

            # Add camera keyframes (cinematic touch)
            camera_keyframes = [
                # Start of scene
                CameraKeyframe(
                    frame=int((current_time - len(actions) * 2.0) * 10),
                    x=0,
                    y=0,
                    zoom=1.0,
                ),
                # End of scene
                CameraKeyframe(frame=int(current_time * 10), x=0, y=0, zoom=1.2),
            ]

            scene = Scene(
                description=scene_data["description"],
                actors=list(actors_map.values()),
                duration=current_time,
                theme="default",
                camera_keyframes=camera_keyframes,
            )
            scenes.append(scene)

        return scenes

    def _map_action(self, action_name: str) -> ActionType | None:
        """Map string action name to ActionType enum."""
        mapping = {
            "walk": ActionType.WALK,
            "run": ActionType.RUN,
            "jump": ActionType.JUMP,
            "wave": ActionType.WAVE,
            "fight": ActionType.FIGHT,
            "punch": ActionType.FIGHT,
            "kick": ActionType.KICKING,
            "dance": ActionType.DANCE,
            "sneak": ActionType.WALK,  # Fallback
            "look_around": ActionType.LOOKING_AROUND,
        }
        return mapping.get(action_name.lower().strip())
