# Data Schema & Ingestion Plan

This document defines the **canonical on-disk schema** for real motion datasets and how they flow into training.

## 1. Canonical sample format (on disk)

Each processed dataset writes a file

- `data/motions_processed/<dataset>/canonical.pt`

containing a **list of Python dicts**. Each dict is a *scene-level sample* with the following keys:

- `"description"` (str): Natural-language description of the motion/scene.
- `"motion"` (torch.FloatTensor):
  - Shape: `[T, 48]` for single-actor (flattened 12 line segments × (x1, y1, x2, y2)) in the **v3 canonical schema**.
  - Optionally `[T, A, 48]` for multi-actor sources (e.g. InterHuman).
  - Legacy/export pipelines may still derive a `[T, 20]` / `[T, A, 20]` **v1 renderer schema** (5 segments × 4 coords) from this canonical representation for `.motion` files.
- `"actions"` (torch.LongTensor):
  - Shape: `[T]` (single-actor) or `[T, A]` (multi-actor).
  - Values are indices into `schema.ActionType` via `ACTION_TO_IDX`.
  - If per-frame labels are unavailable, fill with a constant action (e.g. `IDLE`).
- `"physics"` (torch.FloatTensor):
  - Shape: `[T, 6]` or `[T, A, 6]`.
  - Convention: `[vx, vy, ax, ay, mx, my]` per actor (velocity, acceleration, momentum of a representative point).
- `"camera"` (torch.FloatTensor):
  - Shape: `[T, 3]`.
  - Canonical layout: `[x, y, zoom]` in world units.
  - For datasets without cameras, use zeros.
- `"source"` (str): Short dataset identifier, e.g. `"amass"`, `"lsmb19"`, `"humanml3d"`.
- `"meta"` (dict, optional): Dataset-specific metadata (subject id, view id, clip id, split, etc.).
- `"enhanced_meta"` (dict, optional): Enhanced metadata for improved generation quality. See Section 5.

All tensors are **single-precision** (`torch.float32` for motion/physics/camera, `torch.int64` for actions).

## 2. Validation layer

All canonical samples must pass `src/data_gen/validator.py::DataValidator` before being written:

- `check_physics_consistency(physics)` now accepts `[T, 6]` and `[T, A, 6]`.
- `check_skeleton_consistency(motion)` now accepts `[T, D]` and `[T, A, D]` for any `D` where `D % 4 == 0`.
  - Canonical training uses `D = 48` (v3 schema). Legacy renderer/export paths may still use `D = 20` (v1-only renderer/export schema).

Converters are responsible for:

1. Building `motion` and (optionally) `physics` at the datasets native FPS.
2. Resampling/padding/truncating to the configured `target_frames` (e.g. 250) if needed.
3. Running `DataValidator` on each candidate sample and dropping or logging invalid ones.
4. Saving only valid samples into `canonical.pt`.

## 3. From canonical data to training datasets

Training uses `StickFigureDataset`, which expects per-sample dicts with **embeddings**:

- `"description"` (str)
- `"motion"` (FloatTensor `[T, 48]` in the v3 canonical schema)
- `"actions"` (LongTensor `[T]`)
- `"physics"` (FloatTensor `[T, 6]`)
- `"camera"` (FloatTensor `[T, 3]`)
- `"embedding"` (FloatTensor `[1024]`)
- `"source"` (str)

To keep converters focused on geometry, we introduce a generic helper script:

- `scripts/build_dataset_for_training.py`

This script will:

1. Load `canonical.pt` (list of dicts as defined above).
2. Use the same sentence-transformer model as `scripts/prepare_data.py` to compute
   a 1024-dim embedding for each `description`.
3. Attach `"embedding"` to each dict.
4. Save the result as a training-ready file, e.g.:

   - `data/motions_processed/<dataset>/train_data.pt`

After this step, `train_runpod.py` (and other training scripts) can point directly at
`train_data.pt` via the `--data-path` / config option and will receive samples in the
expected format.

## 4. Dataset-specific converters

For each real dataset we implement `src/data_gen/convert_<dataset>.py` that:

1. Reads raw data from `data/<dataset>/...`.
2. Maps native joints / skeletons / cameras / labels into the canonical keys above.
3. Runs `DataValidator` on each sample.
4. Writes `canonical.pt` into the appropriate `data/motions_processed/<dataset>/` folder.

Examples (names only; actual implementations live in `src/data_gen`):

- `convert_amass.py` 	6 AMASS SMPL/SMPLX to canonical stick figures.
- `convert_lsmb19.py` 	6 LSMB19 long-horizon NTU-based sequences.
- `convert_humanml3d.py` 	6 HumanML3D motion + text.
- `convert_kit_ml.py` 	6 KIT-ML motion + text.
- `convert_interhuman.py` 	6 InterHuman multi-human interactions.
- `convert_aist_plusplus.py` 	6 AIST++ dance + camera.
- `convert_ntu_rgbd.py` 	6 NTU RGB+D skeleton actions.
- `convert_100style_canonical.py` 	6 100STYLE canonicalized into this format.

Together, these pieces define a consistent path:

Raw dataset  →  `convert_<dataset>.py`  →  `canonical.pt`  →
`build_dataset_for_training.py`  →  `<dataset>/train_data.pt`  →  training.

## 5. Enhanced Sample Metadata

The `"enhanced_meta"` field provides optional metadata that enriches samples with
additional information for improved motion generation quality. All fields are
**optional** to maintain backward compatibility with existing samples.

Defined in `src/data_gen/schema.py` as Pydantic models, serialized as dicts:

### 5.1 Motion Style (`motion_style`)

Computed characteristics of the motion pattern:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `tempo` | float | 0.0-1.0 | Motion speed/rhythm (0=slow, 1=fast) |
| `energy_level` | float | 0.0-1.0 | Motion intensity (0=idle, 1=intense) |
| `smoothness` | float | 0.0-1.0 | Motion fluidity (0=jerky, 1=fluid) |

### 5.2 Subject Demographics (`subject`)

Performer characteristics (from SMPL betas or annotations):

| Field | Type | Description |
|-------|------|-------------|
| `height_cm` | float | Estimated height in centimeters |
| `gender` | str | "male", "female", or "unknown" |
| `age_group` | str | "child", "adult", "elderly", or "unknown" |

### 5.3 Music Metadata (`music`)

Music/rhythm information (primarily AIST++):

| Field | Type | Description |
|-------|------|-------------|
| `bpm` | float | Beats per minute |
| `beat_frames` | list[int] | Frame indices where beats occur |
| `genre` | str | Music genre (e.g., "break", "pop") |

### 5.4 Interaction Metadata (`interaction`)

Multi-actor relationship information (primarily InterHuman):

| Field | Type | Description |
|-------|------|-------------|
| `contact_frames` | list[int] | Frames where actors are in contact |
| `interaction_role` | str | "leader", "follower", or "symmetric" |
| `interaction_type` | str | Type of interaction (e.g., "handshake") |

### 5.5 Temporal Metadata (`temporal`)

Original timing before resampling:

| Field | Type | Description |
|-------|------|-------------|
| `original_fps` | int | Source frame rate |
| `original_duration_sec` | float | Original duration in seconds |
| `original_num_frames` | int | Original frame count |

### 5.6 Quality Metadata (`quality`)

Data quality metrics for training weights:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `reconstruction_confidence` | float | 0.0-1.0 | MoCap reconstruction quality |
| `marker_quality` | float | 0.0-1.0 | Joint noise level (higher=cleaner) |
| `physics_score` | float | ≥0.0 | Physics validation score |

### 5.7 Emotion Metadata (`emotion`)

Emotional characteristics (inferred from text or motion):

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `emotion_label` | str | - | Primary emotion (from FacialExpression) |
| `valence` | float | -1.0 to 1.0 | Pleasantness (-1=negative, 1=positive) |
| `arousal` | float | 0.0-1.0 | Intensity (0=calm, 1=excited) |

### Example Usage

```python
from src.data_gen.schema import (
    EnhancedSampleMetadata,
    MotionStyleMetadata,
    TemporalMetadata,
)

# Build enhanced metadata
enhanced = EnhancedSampleMetadata(
    motion_style=MotionStyleMetadata(
        tempo=0.6,
        energy_level=0.8,
        smoothness=0.7,
    ),
    temporal=TemporalMetadata(
        original_fps=120,
        original_duration_sec=5.2,
        original_num_frames=624,
    ),
)

# Add to canonical sample
sample = {
    "description": "A person running quickly",
    "motion": motion_tensor,
    "physics": physics_tensor,
    "actions": actions_tensor,
    "camera": camera_tensor,
    "source": "amass",
    "meta": {"file": "CMU/01/01_01.npz"},
    "enhanced_meta": enhanced.model_dump(),  # Serialize to dict
}
```

### Dataset Coverage

| Dataset | motion_style | temporal | quality | subject | emotion | music | interaction |
|---------|:------------:|:--------:|:-------:|:-------:|:-------:|:-----:|:-----------:|
| AMASS | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| HumanML3D | ✓ | ✓ | ✓ | ○ | ✓ | - | - |
| BABEL | ✓ | ✓ | - | - | ✓ | - | - |
| AIST++ | ✓ | ✓ | - | - | ✓ | ✓ | - |
| InterHuman | ✓ | ✓ | - | - | ✓ | - | ✓ |
| KIT-ML | ✓ | ✓ | - | - | ✓ | - | - |
| NTU RGB+D | ✓ | ✓ | - | - | ✓ | - | - |

Legend: ✓ = Supported, ○ = Partial, - = Not applicable

### Using Enhanced Metadata in Training

The enhanced metadata enables several training strategies for improved motion generation:

#### 1. Conditional Generation

Use metadata as additional conditioning signals:

```python
# In training dataloader
def collate_with_metadata(samples):
    batch = default_collate(samples)

    # Extract motion style for conditioning
    energy_levels = [s["enhanced_meta"]["motion_style"]["energy_level"]
                     for s in samples if s.get("enhanced_meta")]
    batch["energy_condition"] = torch.tensor(energy_levels)
    return batch
```

#### 2. Sample Weighting

Weight samples by quality during training:

```python
# Higher quality samples get more weight
def get_sample_weight(sample):
    meta = sample.get("enhanced_meta", {})
    quality = meta.get("quality", {})
    marker_quality = quality.get("marker_quality", 1.0) or 1.0
    return marker_quality  # 0-1 scale
```

#### 3. Style-Aware Augmentation

Use motion style to apply appropriate augmentations:

```python
# More aggressive augmentation for high-energy motion
def augment_motion(motion, enhanced_meta):
    energy = enhanced_meta["motion_style"]["energy_level"]
    noise_scale = 0.01 * (1 + energy)  # More noise for athletic motion
    return motion + torch.randn_like(motion) * noise_scale
```

#### 4. Emotion-Guided Generation

Condition generation on inferred emotion:

```python
# Map emotion to style tokens
EMOTION_TOKENS = {
    "happy": "<HAPPY>",
    "sad": "<SAD>",
    "neutral": "<NEUTRAL>",
    "excited": "<EXCITED>",
}

def add_emotion_prompt(description, enhanced_meta):
    emotion = enhanced_meta.get("emotion", {}).get("emotion_label", "neutral")
    return f"{EMOTION_TOKENS.get(emotion, '')} {description}"
```

#### 5. Backward Compatibility

All enhanced metadata is optional. Training code should handle missing fields:

```python
def safe_get_energy(sample):
    meta = sample.get("enhanced_meta") or {}
    style = meta.get("motion_style") or {}
    return style.get("energy_level", 0.5)  # Default to medium energy
```

#### 6. Diffusion Layer Conditioning

The diffusion refinement module (`src/model/diffusion.py`) supports style-conditioned
generation using enhanced metadata. Key components:

**StyleCondition**: Container for style signals

```python
from src.model.diffusion import StyleCondition

# Create from enhanced_meta
condition = StyleCondition.from_enhanced_meta(sample["enhanced_meta"])

# Or create directly
condition = StyleCondition(
    tempo=0.8,          # Motion speed (0-1)
    energy_level=0.6,   # Motion intensity (0-1)
    smoothness=0.9,     # Jerk-based smoothness (0-1)
    valence=0.2,        # Emotional positivity (-1 to 1)
    arousal=0.7,        # Emotional activation (0-1)
)
```

**Conditioned Refinement with Classifier-Free Guidance**:

```python
from src.model.diffusion import (
    PoseRefinementUNet,
    DiffusionRefinementModule,
    DDPMScheduler,
    extract_style_conditions_from_batch,
)

# Create conditioned UNet
unet = PoseRefinementUNet(
    pose_dim=48,  # v3 canonical 12-segment schema
    use_style_conditioning=True,  # Enable style conditioning
    style_emb_dim=128,
)

# Create refinement module
scheduler = DDPMScheduler(num_train_timesteps=1000)
module = DiffusionRefinementModule(unet, scheduler, device="cuda")

# Refine with classifier-free guidance
style_conditions = extract_style_conditions_from_batch(batch)
refined = module.refine_poses(
    transformer_output,
    style_conditions=style_conditions,
    guidance_scale=2.5,  # > 1.0 for stronger style adherence
    num_inference_steps=50,
)
```

**Training with CFG Dropout**:

```python
# Training automatically applies CFG dropout (10% by default)
result = module.train_step(
    clean_poses,
    optimizer,
    style_conditions=style_conditions,
)
```

