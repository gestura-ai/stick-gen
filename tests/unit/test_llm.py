"""
Tests for LLM Story Generation System

Tests:
- MockBackend story generation
- ScriptSchema validation
- LLMStoryGenerator backend switching
- Script to Scene conversion
- Action mapping
"""

import sys

sys.path.insert(0, "/Users/bc/gestura/stick-gen")

import pytest
from unittest.mock import MagicMock, patch

from src.data_gen.llm_story_engine import (
    ScriptSchema,
    MockBackend,
    GrokBackend,
    OllamaBackend,
    LLMStoryGenerator,
    LLM_AVAILABLE,
)
from src.data_gen.schema import ActionType


class TestScriptSchema:
    """Tests for ScriptSchema Pydantic model"""

    def test_script_schema_creation(self):
        """Test creating a valid ScriptSchema"""
        script = ScriptSchema(
            title="Test Script",
            description="A test description",
            duration=5.0,
            characters=[{"name": "Hero", "role": "Protagonist"}],
            scenes=[{"description": "Hero walks", "action_sequence": "walk"}],
        )
        assert script.title == "Test Script"
        assert script.duration == 5.0
        assert len(script.characters) == 1
        assert len(script.scenes) == 1
        print("✓ ScriptSchema creation test passed")

    def test_script_schema_defaults(self):
        """Test default values for optional fields"""
        script = ScriptSchema(
            title="Test", description="Test", duration=1.0, characters=[], scenes=[]
        )
        assert script.actions == []
        assert script.camera == []
        print("✓ ScriptSchema defaults test passed")


class TestMockBackend:
    """Tests for MockBackend"""

    def test_mock_generates_heist_script(self):
        """Mock should generate heist-themed script for 'heist' prompt"""
        backend = MockBackend()
        script = backend.generate_story("A bank heist movie")

        assert "heist" in script.title.lower() or "heist" in script.description.lower()
        assert len(script.characters) >= 1
        assert len(script.scenes) >= 1
        print("✓ MockBackend heist script test passed")

    def test_mock_generates_dance_script(self):
        """Mock should generate dance-themed script for 'dance' prompt"""
        backend = MockBackend()
        script = backend.generate_story("An epic dance battle")

        assert "dance" in script.title.lower() or "dance" in script.description.lower()
        print("✓ MockBackend dance script test passed")

    def test_mock_generates_generic_script(self):
        """Mock should generate generic script for other prompts"""
        backend = MockBackend()
        script = backend.generate_story("Something random")

        assert isinstance(script, ScriptSchema)
        assert script.duration > 0
        print("✓ MockBackend generic script test passed")


class TestLLMStoryGenerator:
    """Tests for LLMStoryGenerator"""

    def test_generator_defaults_to_mock(self):
        """Generator should use MockBackend when provider is 'mock'"""
        generator = LLMStoryGenerator(provider="mock")
        assert isinstance(generator.backend, MockBackend)
        print("✓ LLMStoryGenerator mock default test passed")

    def test_generator_fallback_when_unavailable(self):
        """Generator should fallback to mock when LLM not available"""
        with patch("src.data_gen.llm_story_engine.LLM_AVAILABLE", False):
            generator = LLMStoryGenerator(provider="grok")
            assert isinstance(generator.backend, MockBackend)
        print("✓ LLMStoryGenerator fallback test passed")

    def test_generate_script_returns_valid_schema(self):
        """generate_script should return ScriptSchema"""
        generator = LLMStoryGenerator(provider="mock")
        script = generator.generate_script("Test prompt")

        assert isinstance(script, ScriptSchema)
        assert script.title is not None
        assert script.duration > 0
        print("✓ LLMStoryGenerator generate_script test passed")

    def test_script_to_scenes_conversion(self):
        """script_to_scenes should convert ScriptSchema to Scene objects"""
        generator = LLMStoryGenerator(provider="mock")
        script = generator.generate_script("A simple walk")
        scenes = generator.script_to_scenes(script)

        assert isinstance(scenes, list)
        assert len(scenes) > 0
        print("✓ LLMStoryGenerator script_to_scenes test passed")


class TestActionMapping:
    """Tests for action string to ActionType mapping"""

    def test_walk_action_mapping(self):
        """'walk' should map to ActionType.WALK"""
        generator = LLMStoryGenerator(provider="mock")
        action = generator._map_action("walk")
        assert action == ActionType.WALK
        print("✓ Walk action mapping test passed")

    def test_run_action_mapping(self):
        """'run' should map to ActionType.RUN"""
        generator = LLMStoryGenerator(provider="mock")
        action = generator._map_action("run")
        assert action == ActionType.RUN
        print("✓ Run action mapping test passed")

    def test_case_insensitive_mapping(self):
        """Action mapping should be case insensitive"""
        generator = LLMStoryGenerator(provider="mock")
        assert generator._map_action("JUMP") == ActionType.JUMP
        assert generator._map_action("Dance") == ActionType.DANCE
        print("✓ Case insensitive mapping test passed")

    def test_unknown_action_returns_none(self):
        """Unknown action should return None"""
        generator = LLMStoryGenerator(provider="mock")
        action = generator._map_action("unknown_action_xyz")
        assert action is None
        print("✓ Unknown action mapping test passed")


# Run tests if executed directly
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running LLM Story Generation Tests")
    print("=" * 50 + "\n")

    # Run all test classes
    for test_class in [
        TestScriptSchema,
        TestMockBackend,
        TestLLMStoryGenerator,
        TestActionMapping,
    ]:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                getattr(instance, method_name)()

    print("\n" + "=" * 50)
    print("All LLM Story Generation Tests Passed! ✓")
    print("=" * 50)
