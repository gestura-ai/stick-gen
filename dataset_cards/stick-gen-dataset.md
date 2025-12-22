---
license: apache-2.0
task_categories:
  - text-to-video
  - video-classification
language:
  - en
tags:
  - stick-figure
  - animation
  - motion-synthesis
  - text-to-motion
  - pytorch
  - gestura
size_categories:
  - 10K<n<100K
pretty_name: Stick-Gen Motion Dataset
---

# Stick-Gen Motion Dataset

A synthetic dataset for training stick-figure animation models. Generated using the [Stick-Gen](https://github.com/gestura-ai/stick-gen) framework.

## Dataset Description

This dataset contains motion sequences for stick-figure characters with:
- **Text descriptions**: Natural language prompts describing the motion
- **Motion tensors**: Joint positions over time `[250 frames, 3 actors, 20 coords]`
- **Action labels**: Per-frame action classification `[250 frames, 3 actors]`
- **Physics tensors**: Velocity, acceleration, momentum `[250 frames, 3 actors, 6]`
- **Facial expressions**: Eye/mouth states `[250 frames, 3 actors, 7]`
- **Camera parameters**: Position and zoom `[250 frames, 3]`

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | ~50,000 (configurable) |
| Sequence Duration | 10 seconds |
| Frame Rate | 25 FPS |
| Max Actors | 3 per scene |
| Action Classes | 60 |

## Dataset Structure

### Streaming Format (Human-Readable)

```
stick-gen-dataset/
├── train_samples/
│   ├── generation_meta.json    # Generation metadata
│   ├── sample_00000000.json    # Individual samples
│   ├── sample_00000001.json
│   └── ...
└── README.md
```

### Sample JSON Format

```json
{
  "description": "A blue figure walks forward while waving",
  "motion": {"_type": "tensor", "shape": [250, 3, 20], "dtype": "torch.float32", "data": [...]},
  "actions": {"_type": "tensor", "shape": [250, 3], "dtype": "torch.int64", "data": [...]},
  "physics": {"_type": "tensor", "shape": [250, 3, 6], "dtype": "torch.float32", "data": [...]},
  "face": {"_type": "tensor", "shape": [250, 3, 7], "dtype": "torch.float32", "data": [...]},
  "camera": {"_type": "tensor", "shape": [250, 3], "dtype": "torch.float32", "data": [...]},
  "augmented": false,
  "annotations": {"shot_type": "medium", "camera_motion": "static", "quality_score": 0.85}
}
```

### Merged Format (For Training)

```python
import torch
data = torch.load("train_data.pt")
# data is a list of sample dictionaries
```

## Usage

### Loading with PyTorch

```python
import torch

# Load merged dataset
data = torch.load("train_data.pt", weights_only=False)
print(f"Loaded {len(data)} samples")

# Access a sample
sample = data[0]
print(f"Description: {sample['description']}")
print(f"Motion shape: {sample['motion'].shape}")  # [250, 3, 20]
```

### Loading Streaming Samples

```python
import json
import torch

def load_sample(filepath):
    with open(filepath) as f:
        data = json.load(f)
    # Convert tensors back
    for key, value in data.items():
        if isinstance(value, dict) and value.get("_type") == "tensor":
            data[key] = torch.tensor(value["data"])
    return data

sample = load_sample("train_samples/sample_00000000.json")
```

## Generation

Generated using the Stick-Gen data generation pipeline:

```bash
python -m src.data_gen.dataset_generator \
  --num-samples 50000 \
  --output data/train_data.pt \
  --streaming
```

## Motion Representation

Each frame contains 20 coordinates representing 5 body segments:
- **Head**: (x1, y1, x2, y2) - head position and size
- **Torso**: (x1, y1, x2, y2) - shoulder to hip
- **Left Arm**: (x1, y1, x2, y2) - shoulder to hand
- **Right Arm**: (x1, y1, x2, y2) - shoulder to hand
- **Left Leg**: (x1, y1, x2, y2) - hip to foot
- **Right Leg**: (x1, y1, x2, y2) - hip to foot

## License

Apache 2.0 - See [LICENSE](https://github.com/gestura-ai/stick-gen/blob/main/LICENSE)

## Citation

```bibtex
@software{stick_gen_2024,
  title = {Stick-Gen: Expressive Stick-Figure Animation Synthesis},
  author = {Gestura AI},
  year = {2024},
  url = {https://github.com/gestura-ai/stick-gen}
}
```

## Contact

- **Organization**: [Gestura AI](https://gestura.ai)
- **Email**: gestura@gestura.ai
- **GitHub**: [gestura-ai/stick-gen](https://github.com/gestura-ai/stick-gen)

