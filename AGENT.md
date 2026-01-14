# Stick Gen - Agent Context

## Project Overview
Stick Gen is a hybrid transformer model (~15.8M parameters) for generating realistic stick figure animations from text prompts. The system combines synthetic data generation with real motion capture data (AMASS) to achieve high-quality, physics-based motion. It features spatial movement, facial expressions, and state-of-the-art text embeddings.

## Latest Updates (December 2025)

### Major Improvements Implemented
1.  **Hybrid Training Strategy**: Combines 50k synthetic samples with 5.6k AMASS motion capture sequences (Total: ~55.6k samples).
2.  **15.8M Parameter Model**: Upgraded architecture with multi-task learning (6 decoder heads).
3.  **Facial Expressions & Speech**: 6 expression types (Happy, Sad, etc.) and 4 speech types with synchronized mouth movements.
4.  **BAAI/bge-large-en-v1.5 Embeddings**: Top-5 on MTEB leaderboard (1024-dim).
5.  **Spatial Movement System**: Actors move through space with realistic velocities and physics constraints.
6.  **Extended Sequences**: 10-second animations (250 frames @ 25fps).

### Current Training Status
-   **Phase 1**: Dataset Generation (Synthetic) - COMPLETE
-   **Phase 2**: Embedding Generation (BAAI/bge) - COMPLETE
-   **Phase 3**: AMASS Integration - COMPLETE
-   **Phase 4**: Model Training (50 epochs) - READY

## Repository Structure

```
stick-gen/
├── stick-gen              # CLI executable script
├── src/
│   ├── cli/              # Command-line interface
│   │   └── main.py       # Entry point for CLI
│   ├── data_gen/         # Data generation pipeline
│   │   ├── dataset_generator.py      # Synthetic data generation
│   │   ├── preprocess_embeddings.py  # Text embedding preprocessing
│   │   ├── convert_amass.py          # AMASS dataset conversion
│   │   ├── convert_100style.py       # 100STYLE BVH conversion (NEW)
│   │   ├── process_amass.py          # AMASS processing utilities
│   │   ├── renderer.py               # Scene rendering & cinematic effects
│   │   ├── camera.py                 # Camera system (Pan, Zoom, Dolly, etc.) (NEW)
│   │   ├── llm_story_engine.py       # LLM story generation (Grok, Ollama) (NEW)
│   │   ├── schema.py                 # Data structures for scenes/actors
│   │   └── story_engine.py           # Random scene generation logic
│   ├── model/            # Transformer model architecture
│   │   ├── transformer.py            # Model definition (15.8M params) + camera conditioning
│   │   └── diffusion.py              # Diffusion refinement (experimental)
│   ├── train/            # Training scripts
│   │   └── train.py                  # Multi-task training loop
│   └── inference/        # Inference pipeline
│       └── generator.py              # Text-to-animation generation
├── examples/             # Example scripts (NEW)
│   ├── basic_generation.py           # Basic animation generation
│   ├── camera_keyframes_example.py   # Camera movement demos
│   ├── llm_story_generation_example.py # LLM story generation
│   └── cinematic_rendering_example.py  # 2.5D rendering demos
├── data/                 # Training data
│   ├── train_data.pt                 # Raw synthetic data
│   ├── train_data_embedded.pt        # Preprocessed with embeddings
│   ├── amass_converted/              # Converted AMASS data
│   ├── 100STYLE/                     # 100STYLE BVH data (NEW)
│   └── train_data_final.pt           # Merged Hybrid Dataset
├── docs/features/        # Feature documentation (NEW)
│   ├── LLM_INTEGRATION.md            # LLM story generation guide
│   ├── CAMERA_SYSTEM.md              # Camera movements and keyframes
│   └── CINEMATIC_RENDERING.md        # 2.5D perspective rendering
├── scripts/              # Utility scripts
│   ├── training/                     # Training automation scripts
│   └── merge_amass_dataset.py        # Dataset merging script
├── model_checkpoint.pth  # Trained model weights
└── requirements.txt      # Python dependencies
```

## Key Components

### 1. Data Generation Pipeline (`src/data_gen/`)

#### Hybrid Dataset Strategy
-   **Synthetic Data**: 50k samples generated programmatically (actions, interactions).
-   **AMASS Data**: 5,592 real motion capture sequences from 12 datasets (CMU, MPI, etc.).
-   **Total Dataset**: ~55.6k samples (before augmentation).

#### `renderer.py` - Animation & Expressions
-   **Spatial Movement**: Updates actor position based on velocity or waypoints.
-   **Facial Expressions**:
    -   6 Types: NEUTRAL, HAPPY, SAD, SURPRISED, ANGRY, EXCITED.
    -   Smooth transitions (0.3s) between expressions.
    -   Speech animation (TALK, SHOUT, WHISPER, SING) with cyclic mouth movements.
-   **Physics**: Gravity, ground collision, and object interaction.

#### `schema.py` - Core Data Structures
-   **ACTION_VELOCITIES**: Mapped to real-world speeds (e.g., Sprinting: 8.0 m/s).
-   **OBJECT_SCALES**: Realistic sizes for 25+ objects.

#### `camera.py` - Camera System (NEW)
-   **CameraState**: Position (x, y) and zoom level per frame.
-   **CameraKeyframe**: Keyframe-based camera animation.
-   **Movement Types**: Static, Pan, Zoom, Track, Dolly, Crane, Orbit.
-   **Interpolation**: Smooth transitions between keyframes.

#### `llm_story_engine.py` - LLM Story Generation (NEW)
-   **Backends**: GrokBackend (X.AI), OllamaBackend (local), MockBackend (testing).
-   **ScriptSchema**: Structured JSON format for complex narratives.
-   **LLMStoryGenerator**: Converts text prompts to renderable scenes.
-   **Environment Variables**: `GROK_API_KEY`, `OLLAMA_HOST`.

#### `convert_100style.py` - 100STYLE Integration (NEW)
-   **BVHForwardKinematics**: Full skeleton-to-stick-figure conversion.
-   **Joint Mapping**: Standard CMU/Mixamo skeleton retargeting.
-   **Resampling**: Convert any FPS to target frame rate (25 FPS).
-   **100+ Motion Styles**: Depressed, Angry, Happy, Proud, etc.

### 2. Model (`src/model/transformer.py`)

#### Architecture (15.8M Parameters)
-   **d_model**: 384
-   **nhead**: 12
-   **num_layers**: 8
-   **embedding_dim**: 1024 (BAAI/bge-large-en-v1.5)

#### Camera Conditioning (NEW)
-   **camera_projection**: Linear layer projecting camera state (x, y, zoom) to d_model.
-   **camera_data**: Optional tensor [seq_len, batch, 3] passed to forward().
-   **Integration**: Camera context added to transformer hidden states.

#### Multi-Task Decoders (6 Heads)
1.  **Pose**: Joint positions (10-dim).
2.  **Position**: Global x, y coordinates.
3.  **Velocity**: Movement speed (vx, vy).
4.  **Action**: Per-frame action classification.
5.  **Physics**: Velocity, acceleration, momentum.
6.  **Environment**: Interaction context.

### 3. Training (`src/train/train.py`)

#### Configuration
-   **Epochs**: 50
-   **Batch Size**: 4 (Effective 16 with Gradient Accumulation)
-   **Loss Functions**:
    -   Pose MSE (1.0)
    -   Temporal Consistency (0.1) - Smoothness
    -   Physics/Velocity constraints

### 4. Inference (`src/inference/generator.py`)
-   Loads 15.8M parameter model.
-   Generates 10-second MP4 videos.
-   Automatically adds facial expressions based on action context.

## Usage

### Generate Animation
```bash
./stick-gen "Two teams playing against each other in a World Series playoff"
./stick-gen "A man exploring space and meets an alien" --output space.mp4
```

### Full Training Pipeline
```bash
cd scripts/training
./run_full_training_pipeline.sh
```
Runs: Synthetic Gen -> Embeddings -> AMASS Merge -> Training.

## Dependencies

-   **torch** (2.3.1+)
-   **transformers** (4.40+)
-   **numpy**, **matplotlib**, **pycairo**, **manim**, **tqdm**
-   **AMASS Dataset**: Requires SMPL+H compatible datasets.

## Development Workflow

1.  **Data Gen**: `python -m src.data_gen.dataset_generator`
2.  **Embeddings**: `python -m src.data_gen.preprocess_embeddings`
3.  **AMASS Merge**: `python scripts/merge_amass_dataset.py`
4.  **Train**: `python -m src.train.train`
5.  **Test**: `python test_improvements.py`

## Key Files to Understand

-   **`src/data_gen/renderer.py`**: Animation logic, facial expressions, cinematic rendering.
-   **`src/data_gen/camera.py`**: Camera movements (Pan, Zoom, Dolly, Crane, Orbit).
-   **`src/data_gen/llm_story_engine.py`**: LLM story generation backends.
-   **`src/data_gen/convert_100style.py`**: 100STYLE BVH conversion with forward kinematics.
-   **`src/model/transformer.py`**: Multi-head transformer with camera conditioning.
-   **`src/data_gen/convert_amass.py`**: Handling real motion capture data.
-   **`scripts/training/run_full_training_pipeline.sh`**: Master training script.

## Technical Details

### Spatial Movement
-   **Unit Scale**: 1 unit ≈ 0.68 meters.
-   **Update Frequency**: 25Hz (every 0.04s).

### Facial Expression System
-   **Implementation**: `FaceFeatures` class in `schema.py`.
-   **Rendering**: Drawn on top of head circle in `renderer.py`.
-   **Performance**: Negligible impact (<5% render time).

### Camera System
-   **6 Movement Types**: Static, Pan, Zoom, Track, Dolly, Crane, Orbit.
-   **Keyframe Animation**: CameraKeyframe with smooth interpolation.
-   **Model Integration**: Camera conditioning via `camera_data` tensor.

### Cinematic Rendering (2.5D)
-   **Perspective Projection**: `x' = x * (f / (f + z))` with focal length f.
-   **Z-Depth Ordering**: Painter's algorithm for proper limb occlusion.
-   **Dynamic Line Width**: Depth-based stroke thickness.

## Licensing
Stick-Gen is licensed under a **Prosocial Public License – PPL-3% (Gestura AI / Stick-Gen Variant)**: for research/OSS and
users below **USD $1M** annual revenue it behaves effectively like a permissive license, while organizations above that
threshold using Stick-Gen in commercial products are expected to share **3% of Attributable Revenue** or obtain a
separate commercial license; see `LICENSE` and `LICENSE_PPL_README.md` for details.

## Future Improvements
-   Multi-actor coordination and complex interaction logic.
-   Background environments and scenery generation.
-   Sound effects and music generation.
-   Real-time rendering and interactive preview.
