# Transformer Training Methodology with Curated Data

This document explains how the Stick-Gen transformer is trained on the new
**curated datasets**, how pretraining and SFT fit together, and how the
configuration files control this process.

For architectural details see:

- `TRAINING_ARCHITECTURE_CLARIFICATION.md`
- `CONFIGURATION.md`

## 1. Data representation at training time

`src/train/train.py` expects a single `.pt` file containing a list of samples,
each with at least:

- `motion`: `[T, D_motion]` tensor of stick-figure poses.
- `embedding`: `[D_text]` text embedding (BAAI/bge-large-en-v1.5, 1024-dim).
- `actions` (optional): `[T]` integer action labels.
- `physics` (optional): `[T, 6]` physics features.
- `camera` (optional): camera parameters per frame.

The loader does not care whether the file came from the legacy or curated
pipeline, as long as the structure matches the schema in
`docs/features/DATA_SCHEMA.md`.

The curated pipeline produces:

- `pretrain_data.pt`, `sft_data.pt` (canonical, no embeddings)
- `pretrain_data_embedded.pt`, `sft_data_embedded.pt` (training-ready)

## 2. Configuration overview

Each variant config (`configs/small.yaml`, `configs/base.yaml`,
`configs/large.yaml`) defines:

- `model.*`: dimensions, layers, dropout.
- `training.*`: batch size, gradient accumulation, epochs, learning rate,
  warmup schedule, max gradient norm.
- `loss_weights.*`: relative weighting of pose, temporal, action, physics, and
  diffusion losses.
- `diffusion.*`: whether diffusion refinement is enabled and its learning rate.
- `data.*`: default training dataset paths, including curated paths:
  - `data.train_data`: legacy training file (for backwards compatibility).
  - `data.curated_pretrain_data`: recommended curated pretraining dataset.
  - `data.curated_sft_data`: curated SFT dataset.


Parallax 2.5D augmentation is configured via additional `data` keys (see
`docs/architecture/RENDERING.md`): `data.use_parallax_augmentation`,
`data.parallax_root`, and `data.parallax_image_size`. These control whether
multi-view PNG frames and their `metadata.json` sidecars are used alongside
the standard motion-only `.pt` datasets for multimodal training.

`src/train/train.py` takes an optional `--data_path` CLI argument (or
`TRAIN_DATA_PATH` env var on RunPod) to override the dataset path from the
config. This is how you switch between legacy and curated datasets.

## 3. Pretraining on curated data

### 3.1 Goals

Pretraining on curated data aims to:

- Improve motion quality and physical plausibility.
- Reduce exposure to clearly invalid or noisy synthetic clips.
- Preserve broad coverage over actions and camera behaviors.

### 3.2 Recommended workflow (local)

1. Run the canonical generation + auto-annotation pipeline as usual.
2. Run `scripts/prepare_curated_datasets.py` with thresholds from
   `DATASETS_AND_CURATION.md` to create `pretrain_data.pt` and `sft_data.pt`.
3. Run `scripts/build_dataset_for_training.py` on `pretrain_data.pt` to produce
   `pretrain_data_embedded.pt` (and optionally the SFT split).
4. Launch training, pointing `--data_path` at the curated dataset, e.g.:

   ```bash
   python -m src.train.train \
     --config configs/large.yaml \
     --data_path data/curated/pretrain_data_embedded.pt \
     --checkpoint_dir outputs/ckpts_large_curated
   ```

By default, training starts from random initialization **unless** you specify a
checkpoint to resume from (see §3.4 for continued pretraining).

### 3.3 Interaction with loss weights

Curated data is generally higher-quality, so you can safely use stronger
regularization via loss weights, e.g. for the base/large models:

- Pose reconstruction loss: baseline weight (1.0).
- Temporal smoothness: moderate (e.g. `loss_weights.temporal = 0.5`).
- Action loss: non-trivial (e.g. `loss_weights.action` in the range 0.3-0.8) if
  actions are densely labeled.
- Physics loss: enable (`loss_weights.physics > 0`) once physics fields are
  trusted.
- Diffusion loss: only if `diffusion.enabled = true` and you want refinement.

These values are set in the configs and can be tuned per experiment.

### 3.4 Continued pretraining and checkpoint resume

`src/train/train.py` supports **resuming** from a previously saved checkpoint.
This enables true continued pretraining (and staging into SFT) without
restarting from random initialization.

Checkpoint resolution precedence is:

1. CLI flag `--resume_from <path>`
2. Environment variable `RESUME_FROM_CHECKPOINT`
3. Config key `training.resume_from` in `configs/*.yaml`

The checkpoint files saved during training contain at least:

- `epoch`: last completed epoch index (0-based).
- `global_step`: number of optimizer steps taken.
- `model_state_dict`: transformer weights.
- `optimizer_state_dict`: optimizer state.
- `scheduler_state_dict`: LR scheduler state.
- `best_val_loss`, `train_loss`, `val_loss`, and aux metrics.

When you resume, training continues from `epoch + 1` up to
`training.epochs` in the current config.

**Example: continued pretraining locally**

```bash
python -m src.train.train \
  --config configs/base.yaml \
  --data_path data/curated/pretrain_data_embedded.pt \
  --checkpoint_dir outputs/ckpts_base_curated_v2 \
  --resume_from outputs/ckpts_base_curated_v1/model_checkpoint_best.pth
```

You can also encode this in the config instead of the CLI:

```yaml
training:
  epochs: 80
  resume_from: "outputs/ckpts_base_curated_v1/model_checkpoint_best.pth"
```

or, for RunPod, via an environment variable (see `RUNPOD_DEPLOYMENT.md`):

```bash
export RESUME_FROM_CHECKPOINT=/runpod-volume/checkpoints/model_checkpoint_best.pth
```

## 4. SFT (instruction-style fine-tuning)

The curated SFT split (`sft_data_embedded.pt`) contains the highest-quality
clips with stricter quality and camera-stability thresholds and action
balancing. SFT typically:

- Uses fewer epochs and a smaller learning rate than pretraining.
- Emphasizes alignment with high-quality textual descriptions and rare actions.

### 4.1 SFT Configurations

Pre-built SFT configs are available:

- `configs/sft_small.yaml` - 7.2M/11.7M params, 15 epochs, lr=1e-4
- `configs/sft_medium.yaml` - 20.6M/25.1M params, 20 epochs, lr=1e-4
- `configs/sft_large.yaml` - 44.6M/71.3M params, 25 epochs, lr=5e-5

> See [../MODEL_SIZES.md](../MODEL_SIZES.md) for detailed parameter breakdowns.

These configs include:

- `training.stage: "sft"` - marks the training stage for logging/checkpoints.
- `training.init_from: null` - set to a pretrained checkpoint path for warm-start.
- `lora.*` - LoRA configuration (disabled by default).

### 4.2 init_from vs resume_from

There are two ways to load a checkpoint:

| Feature | `init_from` (SFT) | `resume_from` (Continue) |
|---------|-------------------|--------------------------|
| Model weights | ✓ Loaded | ✓ Loaded |
| Optimizer state | ✗ Fresh | ✓ Restored |
| Scheduler state | ✗ Fresh | ✓ Restored |
| Epoch counter | ✗ Starts at 0 | ✓ Continues from checkpoint |
| Use case | SFT from pretrained | Resume interrupted training |

**Precedence for init_from:**

1. CLI flag `--init_from <path>`
2. Environment variable `INIT_FROM_CHECKPOINT`
3. Config key `training.init_from` in `configs/*.yaml`

### 4.3 Running SFT

**Fresh SFT (no pretrained weights):**

```bash
python -m src.train.train \
  --config configs/sft_base.yaml \
  --data_path data/curated/sft_data_embedded.pt \
  --checkpoint_dir outputs/ckpts_sft_base
```

**Warm-start SFT from pretrained checkpoint:**

```bash
python -m src.train.train \
  --config configs/sft_base.yaml \
  --data_path data/curated/sft_data_embedded.pt \
  --checkpoint_dir outputs/ckpts_sft_base \
  --init_from outputs/ckpts_base_curated/model_checkpoint_best.pth
```

Or via config:

```yaml
training:
  stage: "sft"
  init_from: "outputs/ckpts_base_curated/model_checkpoint_best.pth"
```

### 4.4 LoRA (Low-Rank Adaptation)

For efficient fine-tuning with minimal trainable parameters, enable LoRA:

```yaml
lora:
  enabled: true
  rank: 8          # Low-rank dimension
  alpha: 16        # Scaling factor (alpha/rank)
  dropout: 0.05    # Dropout on LoRA layers
  target_modules:  # Regex patterns for layers to modify
    - "transformer_encoder"
    - "pose_decoder"
```

When LoRA is enabled:

1. LoRA adapters are injected into matching Linear layers.
2. Base model parameters are frozen.
3. Only LoRA parameters (A and B matrices) are trained.
4. Checkpoints include `lora_state_dict` for easy extraction.

**Example: SFT with LoRA**

```bash
# Set lora.enabled: true in configs/sft_base.yaml, then:
python -m src.train.train \
  --config configs/sft_base.yaml \
  --data_path data/curated/sft_data_embedded.pt \
  --checkpoint_dir outputs/ckpts_sft_base_lora \
  --init_from outputs/ckpts_base_curated/model_checkpoint_best.pth
```

LoRA typically reduces trainable parameters by 90-99%, enabling faster
training and lower memory usage while preserving most of the model's
pretrained knowledge.

### 4.5 RunPod SFT Workflow

Use the dedicated SFT entrypoint on RunPod:

```bash
# On RunPod, set environment variables:
export MODEL_VARIANT=sft_base
export INIT_FROM_CHECKPOINT=/runpod-volume/checkpoints/pretrain/model_checkpoint_best.pth
export USE_LORA=true  # Optional

# Run SFT entrypoint
./runpod/sft_entrypoint.sh
```

See `runpod/sft_entrypoint.sh` for full configuration options.

## 5. Diffusion refinement

If `diffusion.enabled: true` and the optional diffusion module is available,
`src/train/train.py` will:

- Instantiate a diffusion refinement module on top of the transformer output.
- Add a diffusion loss term scaled by `loss_weights.diffusion`.

Because diffusion focuses on fine-grained smoothing and realism, it pairs
well with curated data, which already filters out heavily corrupted samples.
For debugging and ablation, set `diffusion.enabled: false` and
`loss_weights.diffusion: 0.0`.

## 6. Relationship to RunPod workflows

On RunPod, the same methodology applies; the difference is how data paths and
configs are supplied:

- `runpod/deploy.sh` with `--curated`:
  - Runs the canonical -> curated -> embedding pipeline inside the data prep Pod.
  - Writes the final training file into the Network Volume (by default
    `data/train_data_final.pt`). When `USE_CURATED_DATA=true`, this file is the
    curated embedded dataset.
- Training Pods receive `TRAIN_DATA_PATH` pointing at that file and use the
  standard training entrypoint.

To run curated continued pretraining on RunPod:

1. Provision a volume with enough space (see `RUNPOD_SYSTEM_REQUIREMENTS.md`).
2. Run `./runpod/deploy.sh prep-data --volume-id <VOL> --curated`.
3. Once data prep completes, run `./runpod/deploy.sh auto-train-all --volume-id <VOL> --curated`.
4. Monitor training using the existing logging and checkpoint tools.

This yields the same training behavior as local curated runs, but fully
orchestrated on RunPod.

## 7. Robustness Evaluation and Safety Critic

The Stick-Gen pipeline includes a **Safety Critic** module for detecting
degenerate or low-quality motion outputs. This is critical for:

1. **Inference-time rejection**: Flagging outputs that fail quality checks
2. **Adversarial evaluation**: Testing model robustness against edge-case prompts
3. **Training data filtering**: Identifying problematic samples in datasets

### 7.1 Safety Critic Checks

The safety critic (`src/eval/safety_critic.py`) performs the following checks:

| Check | Description | Severity |
|-------|-------------|----------|
| Frozen Motion | Motion with near-zero velocity for >80% of frames | 0.9 (Critical) |
| Repetitive Motion | Detected cyclic patterns (≥3 similar windows) | 0.7 (High) |
| Jittery Motion | High acceleration in >30% of frames | 0.6 (Medium) |
| Velocity Exceeded | Physics velocity > 15 m/s | 0.8 (High) |
| Acceleration Exceeded | Physics acceleration > 50 m/s² | 0.7 (High) |
| Ground Penetration | Body parts below y = -0.1 | 0.6 (Medium) |
| Quality Below Threshold | Auto-annotator score < 0.3 | 0.5 (Medium) |

Outputs with any issue severity ≥ 0.7 are marked as **unsafe** by default.

### 7.2 Adversarial Prompt Suites

The file `configs/eval/prompt_suites.yaml` contains adversarial prompt suites:

- **adversarial_long_prompts**: Extremely long, detailed prompts
- **adversarial_contradictory**: Contradictory or impossible instructions
- **adversarial_extreme_actions**: Physically impossible actions
- **adversarial_ambiguous**: Vague, nonsensical, or empty prompts
- **adversarial_repetition_inducing**: Prompts designed to cause loops
- **adversarial_edge_cases**: Boundary conditions (zero duration, many actors)
- **adversarial_injection**: Prompt injection attempts
- **robustness_baseline**: Simple prompts that should always work

### 7.3 Running Adversarial Evaluation

```bash
# Evaluate all adversarial suites
python scripts/run_adversarial_eval.py --checkpoint checkpoints/best_model.pth

# Evaluate specific suites
python scripts/run_adversarial_eval.py \
    --checkpoint checkpoints/best_model.pth \
    --suites adversarial_contradictory adversarial_extreme_actions

# Custom output directory
python scripts/run_adversarial_eval.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir eval_results/adversarial
```

The script produces a JSON report with:
- Per-prompt safety results and issue types
- Per-suite aggregated statistics
- Overall robustness metrics (baseline vs adversarial safe ratios)
- Robustness gap (difference between baseline and adversarial performance)

### 7.4 Integrating Safety Checks at Inference

The `InferenceGenerator` class supports optional safety checking:

```python
from src.inference.generator import InferenceGenerator
from src.eval.safety_critic import SafetyCriticConfig

# Enable safety checking
generator = InferenceGenerator(
    model_path="checkpoints/best_model.pth",
    enable_safety_check=True,
    safety_config=SafetyCriticConfig(rejection_severity_threshold=0.6),
)

# Check motion safety manually
motion = torch.randn(250, 20)
is_safe, result = generator.check_motion_safety(motion)
if not is_safe:
    print("Rejected:", result.get_rejection_reasons())
```

### 7.5 Interpreting Robustness Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Baseline Safe Ratio | >95% | 80-95% | <80% |
| Adversarial Safe Ratio | >70% | 50-70% | <50% |
| Robustness Gap | <15% | 15-30% | >30% |

A large robustness gap indicates the model is overfitting to "easy" prompts
and may fail on real-world edge cases. Consider:

1. Adding adversarial examples to training data
2. Increasing regularization (dropout, weight decay)
3. Using data augmentation on prompts
4. Fine-tuning on curated SFT data with diverse prompts
