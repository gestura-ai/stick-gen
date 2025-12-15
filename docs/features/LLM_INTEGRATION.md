# LLM Integration for Story Generation

This document describes the LLM-powered story generation system in Stick-Gen, which enables dynamic script creation from natural language prompts using various LLM backends.

## Overview

The LLM Story Engine (`src/data_gen/llm_story_engine.py`) generates structured animation scripts from text prompts. It supports multiple LLM backends and falls back to a mock backend when external services are unavailable.

## Supported Backends

### 1. Mock Backend (Default)
- **Purpose**: Fallback for testing and development
- **Requirements**: None (built-in)
- **Usage**: Activated when no LLM dependencies installed or explicitly requested

### 2. Grok Backend (X.AI)
- **Provider**: X.AI (formerly Twitter AI)
- **Model**: `grok-4-latest` (default)
- **API Endpoint**: `https://api.x.ai/v1`
- **Requirements**: `openai>=1.0.0`, `python-dotenv>=1.0.0`

### 3. Ollama Backend (Local)
- **Purpose**: Local inference with open-source models
- **Model**: `llama3` (default), supports any Ollama-compatible model
- **Requirements**: `ollama>=0.1.0` and Ollama server running locally

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# For Grok (X.AI) backend
GROK_API_KEY=your_grok_api_key_here

# Optional: Custom model overrides
GROK_MODEL=grok-4-latest
OLLAMA_MODEL=llama3
```

### Getting API Keys

**Grok API Key:**
1. Visit [X.AI Developer Portal](https://x.ai)
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy to your `.env` file

**Ollama Setup:**
1. Install Ollama: `brew install ollama` (macOS) or see [ollama.ai](https://ollama.ai)
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull llama3`

## ScriptSchema Format

LLM backends generate scripts matching this Pydantic schema:

```python
class ScriptSchema(BaseModel):
    title: str              # Animation title
    description: str        # Brief description
    duration: float         # Total duration in seconds
    characters: List[Dict]  # Character definitions
    scenes: List[Dict]      # Scene descriptions with actions
    actions: List[Dict]     # Additional action metadata (optional)
    camera: List[Dict]      # Camera keyframes (optional)
```

### Example Output

```json
{
  "title": "The Stick Heist",
  "description": "A bank heist gone wrong",
  "duration": 10.0,
  "characters": [
    {"name": "Ninja", "role": "The Infiltrator"},
    {"name": "Guard", "role": "The Obstacle"}
  ],
  "scenes": [
    {"description": "Ninja sneaks past guard", "action_sequence": "sneak, look_around"},
    {"description": "Guard hears noise", "action_sequence": "look_around, walk"},
    {"description": "Ninja attacks", "action_sequence": "jump, kick"},
    {"description": "Escape", "action_sequence": "run, jump"}
  ],
  "actions": [],
  "camera": []
}
```

## Available Actions

The following action keywords are recognized:

| Action | Maps To | Description |
|--------|---------|-------------|
| `walk` | `ActionType.WALK` | Basic walking |
| `run` | `ActionType.RUN` | Fast movement |
| `jump` | `ActionType.JUMP` | Jumping |
| `wave` | `ActionType.WAVE` | Waving gesture |
| `fight` | `ActionType.FIGHT` | Fighting stance |
| `punch` | `ActionType.FIGHT` | Punching (alias) |
| `kick` | `ActionType.KICKING` | Kicking motion |
| `dance` | `ActionType.DANCE` | Dancing |
| `sneak` | `ActionType.WALK` | Sneaking (fallback) |
| `look_around` | `ActionType.LOOKING_AROUND` | Looking around |

## Usage Examples

### Command Line

```bash
# Use LLM for story generation (default: mock if unavailable)
./stick-gen "A ninja sneaking into a bank" --use-llm

# Specify output file
./stick-gen "Two dancers in a competition" --use-llm --output dance_battle.mp4
```

### Python API

```python
from src.data_gen.llm_story_engine import LLMStoryGenerator

# Initialize with specific backend
generator = LLMStoryGenerator(provider="grok")  # or "ollama", "mock"

# Generate script from prompt
script = generator.generate_script("A robot learning to dance")

# Convert to Scene objects for rendering
scenes = generator.script_to_scenes(script)

# Access script properties
print(f"Title: {script.title}")
print(f"Duration: {script.duration}s")
print(f"Characters: {len(script.characters)}")
```

### With Custom Model

```python
# Grok with specific model
generator = LLMStoryGenerator(provider="grok", model="grok-4-latest")

# Ollama with custom model
generator = LLMStoryGenerator(provider="ollama", model="mistral")
```

## Adding New LLM Backends

To add a new backend, implement the `LLMBackend` protocol:

```python
from typing import Protocol
from .llm_story_engine import ScriptSchema

class LLMBackend(Protocol):
    def generate_story(self, prompt: str) -> ScriptSchema:
        ...

class MyCustomBackend:
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
    
    def generate_story(self, prompt: str) -> ScriptSchema:
        # Call your LLM API
        response = my_api_call(prompt)
        
        # Parse response into ScriptSchema
        return ScriptSchema(
            title=response.title,
            description=response.description,
            duration=response.duration,
            characters=response.characters,
            scenes=response.scenes,
            actions=[],
            camera=[]
        )
```

Then add to `LLMStoryGenerator.__init__()`:

```python
elif provider == "my_custom" and LLM_AVAILABLE:
    self.backend = MyCustomBackend(os.getenv("MY_API_KEY"))
```

## Error Handling

All backends implement graceful fallback:

1. **API Failure**: Falls back to `MockBackend`
2. **Invalid JSON**: Attempts to clean markdown formatting, then falls back
3. **Missing Fields**: Default values provided (`actions=[]`, `camera=[]`)

## Troubleshooting

### "LLM Import Error" Message
Install required dependencies:
```bash
pip install openai python-dotenv ollama
```

### Grok API Returns Error
- Verify `GROK_API_KEY` is set correctly
- Check API key hasn't expired
- Ensure sufficient API credits

### Ollama Connection Refused
- Start Ollama server: `ollama serve`
- Verify model is pulled: `ollama list`
- Check default port 11434 is accessible

