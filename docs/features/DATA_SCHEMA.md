# Data Schema & Ingestion Plan

This document defines the **canonical on-disk schema** for real motion datasets and how they flow into training.

## 1. Canonical sample format (on disk)

Each processed dataset writes a file

- `data/motions_processed/<dataset>/canonical.pt`

containing a **list of Python dicts**. Each dict is a *scene-level sample* with the following keys:

- `"description"` (str): Natural-language description of the motion/scene.
- `"motion"` (torch.FloatTensor):
  - Shape: `[T, 20]` for single-actor (flattened 5 line segments × (x1, y1, x2, y2)).
  - Optionally `[T, A, 20]` for multi-actor sources (e.g. InterHuman).
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

All tensors are **single-precision** (`torch.float32` for motion/physics/camera, `torch.int64` for actions).

## 2. Validation layer

All canonical samples must pass `src/data_gen/validator.py::DataValidator` before being written:

- `check_physics_consistency(physics)` now accepts `[T, 6]` and `[T, A, 6]`.
- `check_skeleton_consistency(motion)` now accepts `[T, 20]` and `[T, A, 20]`.

Converters are responsible for:

1. Building `motion` and (optionally) `physics` at the datasets native FPS.
2. Resampling/padding/truncating to the configured `target_frames` (e.g. 250) if needed.
3. Running `DataValidator` on each candidate sample and dropping or logging invalid ones.
4. Saving only valid samples into `canonical.pt`.

## 3. From canonical data to training datasets

Training uses `StickFigureDataset`, which expects per-sample dicts with **embeddings**:

- `"description"` (str)
- `"motion"` (FloatTensor `[T, 20]`)
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

