# Stick-Gen Examples

This directory contains example scripts demonstrating common use cases for the stick-gen model.

## Available Examples

### 1. Basic Generation (`basic_generation.py`)
Generate stick figure animations from text prompts using a trained model.

**Features:**
- Load model from checkpoint
- Generate animations from single prompts
- Save outputs as JSON or video
- Visualize generated animations

**Usage:**
```bash
python examples/basic_generation.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --prompt "A person walking forward" \
  --output outputs/walking.json
```

### 2. Batch Processing (`batch_processing.py`)
Process multiple prompts in batch for efficient generation.

**Features:**
- Load prompts from file or command line
- Batch generation with configurable batch size
- Progress tracking and ETA
- Export results in multiple formats

**Usage:**
```bash
# From file
python examples/batch_processing.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --prompts-file prompts.txt \
  --output-dir outputs/

# From command line
python examples/batch_processing.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --prompts "walking" "running" "jumping" \
  --output-dir outputs/
```

### 3. Fine-Tuning (`fine_tuning.py`)
Fine-tune a pre-trained model on custom datasets.

**Features:**
- Load pre-trained checkpoint
- Fine-tune on custom data
- Configurable learning rate and epochs
- Save fine-tuned checkpoints

**Usage:**
```bash
python examples/fine_tuning.py \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --data custom_data.pt \
  --output checkpoints/fine_tuned.pt \
  --epochs 10 \
  --learning-rate 1e-5
```

### 4. Camera Keyframes (`camera_keyframes_example.py`)
Demonstrate camera movement definitions and keyframe animation.

**Features:**
- Pan, Zoom, Track, Dolly, Crane, Orbit movements
- CameraKeyframe creation and interpolation
- Camera state visualization
- Integration with scene rendering

**Usage:**
```bash
python examples/camera_keyframes_example.py
```

**Documentation:** See [Camera System Guide](../docs/features/CAMERA_SYSTEM.md)

### 5. LLM Story Generation (`llm_story_generation_example.py`)
Generate complex narratives using LLM backends.

**Features:**
- Multiple backends: Grok (X.AI), Ollama (local), Mock (testing)
- ScriptSchema for structured narratives
- Script-to-scene conversion
- Action mapping from story context

**Usage:**
```bash
# With Mock backend (no API key needed)
python examples/llm_story_generation_example.py

# With Grok backend
export GROK_API_KEY=your_key
python examples/llm_story_generation_example.py --backend grok

# With Ollama backend
python examples/llm_story_generation_example.py --backend ollama
```

**Documentation:** See [LLM Integration Guide](../docs/features/LLM_INTEGRATION.md)

### 6. Cinematic Rendering (`cinematic_rendering_example.py`)
Render animations with 2.5D perspective effects.

**Features:**
- Perspective projection with focal length
- Z-depth ordering (painter's algorithm)
- Dynamic line width based on depth
- CinematicRenderer integration

**Usage:**
```bash
python examples/cinematic_rendering_example.py
```

**Documentation:** See [Cinematic Rendering Guide](../docs/features/CINEMATIC_RENDERING.md)

## Requirements

All examples require the same dependencies as the main project:

```bash
pip install -r requirements.txt
```

## Expected Outputs

### Basic Generation
- JSON file with motion data
- Optional: MP4 video visualization
- Console output with generation statistics

### Batch Processing
- Directory of JSON files (one per prompt)
- CSV file with generation metadata
- Progress bar during generation

### Fine-Tuning
- Fine-tuned model checkpoint
- Training logs and metrics
- Validation results

## Troubleshooting

### Out of Memory
- Use smaller batch size: `--batch-size 1`
- Use small variant: `--config configs/small.yaml`
- Reduce sequence length in config

### Slow Generation
- Use GPU if available
- Use large variant for better GPU utilization: `--config configs/large.yaml`
- Increase batch size if memory allows

### Poor Quality Outputs
- Check if model is fully trained
- Verify prompt format matches training data
- Try different random seeds: `--seed 42`

## Contributing

To add new examples:

1. Create a new Python script in this directory
2. Follow the existing code structure
3. Add documentation to this README
4. Include usage examples and expected outputs
5. Test with all three model variants (small, base, large)

## Support

For issues or questions:
- Check the main [documentation](../docs/)
- Open an issue on [GitHub](https://github.com/gestura-ai/stick-gen)
- See [troubleshooting guide](../docs/setup/TROUBLESHOOTING.md)

**Note:** Model weights are hosted on Hugging Face Hub under the [GesturaAI](https://huggingface.co/GesturaAI) organization.

