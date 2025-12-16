# RunPod System Requirements for Curated Continued Pretraining

This document summarizes **disk, compute, and cost considerations** for running Stick-Gen
continued pretraining on RunPod using the **curated dataset pipeline**.

## 1. Pipelines and assumptions

Three data prep modes share the same training code:

- **Legacy pipeline**
  - Raw data (100STYLE, AMASS, synthetic, etc.) → `scripts/prepare_data.py`
  - Output: `data/train_data_final.pt` (embedding-augmented)
- **Curated pipeline**
  - Canonical `.pt` from real datasets → `scripts/prepare_curated_datasets.py`
  - Outputs: `pretrain_data.pt`, `sft_data.pt`, `curation_stats.json`
  - Then `scripts/build_dataset_for_training.py` adds text embeddings.
- **2.5D Parallax Augmentation pipeline**
  - Canonical `.pt` → `stick-gen generate-data --augment-parallax`
  - Output: `data/2.5d_parallax/sample_XXXXXX/actor_Y/*.png` + `metadata.json`
  - Used by `MultimodalParallaxDataset` for image+motion multimodal training
  - Requires Node.js runtime and npm packages: `three`, `pngjs`, `gl` (headless-gl)

Assumptions in estimates below:

- 10-second sequences (≈250 frames) in the canonical schema (see `DATA_SCHEMA.md`).
- Embeddings from `BAAI/bge-large-en-v1.5` (1024-dim, float32).
- Default configs in `configs/small.yaml`, `configs/base.yaml`, `configs/large.yaml`.

## 2. Storage requirements

### 2.1 Per-sample size estimates

From `docs/training/CPU_TRAINING_PLAN.md` (synthetic + AMASS baseline):

- 500k canonical samples → `train_data.pt` ≈ **1.0–1.5 GB**
- 500k embedded samples → `train_data_embedded.pt` ≈ **2.0–2.5 GB**

Approximate per-sample sizes:

- Canonical only: **~2–3 kB / sample**
- Canonical + embedding: **~4–5 kB / sample**

For rough planning, use **5 kB per embedded training sample** as a safe upper bound.

### 2.2 Estimating sizes from `curation_stats.json`

`scripts/prepare_curated_datasets.py` writes `curation_stats.json` with:

- `pretrain.num_samples`
- `sft.num_samples`

Let:

- `N_pre = pretrain.num_samples`
- `N_sft = sft.num_samples`
- `N_total = N_pre + N_sft`

Then approximate sizes:

- **Embedded training set** (pretrain-only or merged):
  - `S_embedded_GB ≈ 5 kB × N_pre / 1e9 ≈ 5 × N_pre / 1e6  (GB)`
- **Canonical curated sets** (`pretrain_data.pt` + `sft_data.pt`):
  - `S_canonical_GB ≈ 3 kB × N_total / 1e9 ≈ 3 × N_total / 1e6  (GB)`

Example ballparks:

| Scenario | `N_pre` + `N_sft` | Canonical (`pretrain`+`sft`) | Embedded pretrain only |
|---------:|------------------:|------------------------------:|------------------------:|
| Small    | 0.5 M             | ~1.5–2 GB                     | ~2–2.5 GB               |
| Medium   | 2.0 M             | ~6–8 GB                       | ~8–10 GB                |
| Large    | 5.0 M             | ~15–20 GB                     | ~20–25 GB               |

### 2.3 2.5D Parallax augmentation storage

The parallax pipeline generates PNG frames for each `(sample, actor, view)` combination:

- Default: 250 views × 4 frames/view = **1,000 PNGs per actor**
- Each PNG: 256×256 RGB ≈ **50–100 KB** (compressed)
- Per-sample with 2 actors: ≈ **100–200 MB**

| Samples | Actors | Views | Frames/View | Estimated Size |
|--------:|-------:|------:|------------:|---------------:|
| 10k     | 1      | 250   | 4           | ~5–10 GB       |
| 50k     | 1      | 250   | 4           | ~25–50 GB      |
| 100k    | 2      | 250   | 4           | ~100–200 GB    |
| 500k    | 1      | 50    | 2           | ~25–50 GB      |

**Tip:** Use fewer `--views-per-motion` and `--frames-per-view` for development/testing.

### 2.4 Network volume sizing

A RunPod Network Volume must hold:

1. **Raw data** (text + motion + metadata)
   - Existing pipeline: ≈ **80–90 GB** (see `RUNPOD_DEPLOYMENT.md`).
2. **Canonical + curated datasets**
   - From table above: typically **< 25 GB** even for large curated runs.
3. **Training datasets with embeddings**
   - 1–25 GB depending on `N_pre` and whether you keep multiple versions.
4. **2.5D Parallax data** (if enabled)
   - 5–200 GB depending on sample count and view/frame settings.
5. **Checkpoints + logs**
   - Each checkpoint ≈ 250–300 MB (base/large models).
   - Keeping 5–10 checkpoints per variant + logs: plan **10–30 GB**.

**Recommended volume sizes:**

| Use case                                   | Suggested size |
|-------------------------------------------|---------------:|
| Legacy pipeline only, single run          | 120 GB         |
| Curated pipeline, ≤ 2M curated samples    | 160 GB         |
| Curated pipeline, up to ~5M samples + CKPT history | 200 GB |
| **With 2.5D parallax augmentation**       | **250 GB**     |

You can reduce requirements by:

- Deleting raw text after canonicalization.
- Pruning older checkpoints once best models are pushed to HuggingFace.
- Using lower `--views-per-motion` / `--frames-per-view` for parallax.

## 3. GPU / CPU requirements

### 3.1 Model sizes and memory

From `configs/*.yaml`:

| Variant | Params | Typical hardware | Notes |
|--------:|-------:|------------------|-------|
| small   | ~5.6M  | 4+ CPU cores, 8–16 GB RAM, GPU optional | CPU-focused, diffusion disabled. |
| base    | ~15.8M | 8+ CPU cores, 16–32 GB RAM, GPU optional (4 GB+) | Balanced default. |
| large   | ~28M   | 32 GB RAM, **8 GB+ VRAM** (RTX 3060 or better) | GPU-optimized, diffusion enabled. |

The curated pipeline does **not** change model sizes; it only improves data quality.

### 3.2 Steps per epoch

For all three variants the **effective batch size** is 64:

- small: `batch_size=1`, `grad_accum_steps=64` → 64
- base:  `batch_size=2`, `grad_accum_steps=32` → 64
- large: `batch_size=16`, `grad_accum_steps=4` → 64

For a dataset with `N_pre` pretraining samples:

- `steps_per_epoch ≈ ceil(N_pre / 64)`

Examples:

| `N_pre` | Steps / epoch |
|--------:|---------------:|
| 0.5 M   | ≈ 7,812        |
| 1.0 M   | ≈ 15,625       |
| 2.0 M   | ≈ 31,250       |

### 3.3 Estimating training time and cost

Let:

- `t_step` = average time per training step (seconds), measured on your GPU.
- `price` = GPU hourly price (e.g. RTX A4000 ≈ **$0.25/h**, L4 ≈ **$0.39/h**; see comments in `runpod/deploy.sh`).

Then:

- `hours_per_epoch = steps_per_epoch × t_step / 3600`
- `total_hours = hours_per_epoch × epochs`
- `cost ≈ total_hours × price`

**Worked example (ballpark):**

- Variant: **large** (100 epochs by default)
- Dataset: `N_pre = 1.0 M` → `steps_per_epoch ≈ 15,625`
- Assume `t_step = 0.10–0.20 s` on an RTX 4090 or A4000
- Then `hours_per_epoch ≈ 0.43–0.87 h`
- `total_hours ≈ 43–87 GPU-hours`
- At `$0.25/h` → **$11–22** for the pretraining run

For smaller variants or fewer epochs, costs drop proportionally. Always measure `t_step`
with a short smoke run on your chosen GPU before committing to long jobs.

## 4. Data preparation Pod requirements

The RunPod data prep Pod (see `runpod/data_prep_entrypoint.sh`):

- Uses the same GPU families as training (A4000/A4500/4000 Ada/A5000/L4/3090/4090).
- Main GPU work is **text embedding** with `BAAI/bge-large-en-v1.5`.
- Embedding memory footprint is modest (fits comfortably in **8 GB VRAM**).

For planning:

- Expect **1–3 GPU-hours** for embedding ≈0.5–1.0M samples.
- Time scales roughly linearly with the number of samples.

Combine these estimates with Section 3 to choose an appropriate GPU type and budget
for both **curated data prep** and **continued pretraining** runs on RunPod.

## 5. 2.5D Parallax Augmentation Requirements

The parallax pipeline uses a headless Three.js renderer to generate multi-view PNG frames.

### 5.1 System dependencies

The Docker image (`docker/Dockerfile`) includes all required dependencies:

- **Node.js 20.x LTS** (required for `threejs_parallax_renderer.js`)
- **npm packages**: `three`, `pngjs`, `gl` (headless-gl)
- **System libraries** (for headless WebGL via `gl`):
  - `libxi-dev`
  - `libgl1-mesa-dev`
  - `libglew-dev`
  - `xvfb` (virtual framebuffer for headless rendering)

### 5.2 Runtime requirements

- **CPU-bound**: The Node.js renderer runs on CPU; no GPU required for rendering
- **Memory**: ~2-4 GB RAM per rendering process
- **Disk I/O**: High write throughput for PNG output (consider SSD storage)

### 5.3 Time estimates

Rendering time scales with: `samples × actors × views × frames`

| Samples | Actors | Views | Frames | Est. Time (4-core) |
|--------:|-------:|------:|-------:|-------------------:|
| 1k      | 1      | 250   | 4      | ~1-2 hours         |
| 10k     | 1      | 250   | 4      | ~10-20 hours       |
| 50k     | 1      | 50    | 2      | ~5-10 hours        |

**Tips:**
- Use `--max-samples N` for incremental/test runs
- Run in parallel across multiple CPU-only pods for large datasets
- Reduce `--views-per-motion` and `--frames-per-view` for faster iteration

### 5.4 CLI usage

```bash
# Generate parallax data for training
stick-gen generate-data \
  --config configs/medium.yaml \
  --augment-parallax \
  --views-per-motion 250 \
  --frames-per-view 4 \
  --output /runpod-volume/data/2.5d_parallax
```

See `README.md` and `docs/architecture/RENDERING.md` for full documentation.
