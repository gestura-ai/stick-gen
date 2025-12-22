# Datasets and Curation Policy

This document explains **which datasets feed Stick-Gen**, how they are mapped into the
canonical schema, and how the **curated pretraining and SFT splits** are constructed.

## 1. Source datasets and roles

All real motion datasets are first converted into the **canonical format**
(described in `docs/features/DATA_SCHEMA.md`) and stored as `.pt` files under
`data/motions_processed/<dataset>/canonical.pt` or a similar path.

### 1.1 AMASS
- **Type:** Large MoCap collection of everyday and locomotion motions.
- **Canonicalization:** Converted to single-actor sequences with physics and camera
  features added or inferred. The converter (`src/data_gen/convert_amass.py`) handles:
  - **SMPL+H format** (156 params) - most common
  - **SMPL-X format** (162/165 params) - including Stage II files
  - **Explicit pose component detection** - uses `root_orient`, `pose_body`, `pose_hand`
    fields when available for reliable extraction from Stage II files (CNRS, MoSh, etc.)
  - **Automatic file filtering** - skips `shape.npz` (body shape only) and `*_stagei.npz`
    (calibration data) files that don't contain motion sequences
- **Role:** Backbone of **pretraining**: diverse, physically plausible human motion.

### 1.2 NTU RGB+D
- **Type:** Action recognition dataset with a wide range of short actions.
- **Canonicalization:** 3D skeleton sequences mapped into the stick-figure layout.
- **Role:** Expands **action vocabulary**; valuable for SFT and evaluation.

### 1.3 100STYLE
- **Type:** Stylized motion clips with distinct motion styles.
- **Canonicalization:** Mapped from BVH; style labels become textual descriptions.
- **Role:** Adds **style diversity** (e.g., exaggerated, playful, cinematic motions).

### 1.4 InterHuman
- **Type:** Multi-person interaction dataset.
- **Canonicalization:** Mapped into one or more actors.
- **Role:** Provides **interaction-heavy** samples (talking, fighting, hugging).

### 1.5 KIT-ML
- **Type:** Text-to-motion dataset (processed similarly to HumanML3D).
- **Canonicalization:** Features mapped to canonical 20-dim pose; text descriptions retained.
- **Role:** High-quality **text-motion alignment** for core semantic understanding.

### 1.6 BABEL
- **Type:** Action-labeled AMASS sequences.
- **Canonicalization:** Uses AMASS poses but applies dense BABEL action labels (100+ categories).
- **Role:** Crucial for **fine-grained action conditioning** and specific motion retrieval.

### 1.7 BEAT
- **Type:** Large-scale high-quality semi-supervised dataset (Body Expression, Audio, Text).
- **Canonicalization:** Focuses on upper-body gestures and speech-motion alignment.
- **Role:** Enhances **conversational gestures** and emotional expression.

### 1.8 Synthetic data
- **Type:** Automatically generated text + motion + camera sequences.
- **Canonicalization:** Produced by the synthetic data generator.
- **Role:** Increases **text diversity** and covers rare combinations; max 40% of SFT.

## 2. Auto-annotation and quality metrics

`src/data_gen/auto_annotator.py` and `src/eval/metrics.py` attach robust quality metadata:

- **Quality Score**: Combined `[0, 1]` score blending physics, camera stability, and action diversity.
- **FID-based Realism**: Distributional distance to "gold standard" MoCap (lower is better).
- **Artifact Detection**: Flags specific issues:
  - `jitter`: High-frequency noise.
  - `static`: Period of no movement.
  - `sliding`: Foot skate.
  - `explosions`: Unrealistic velocity spikes.
- **Action Summary**: Dominant action labels derived from text or classifier.

## 3. Curation policy

Curation (`src/data_gen/curation.py`) applies strict filters to build the Pretrain and SFT splits.

### 3.1 Filtering Criteria
1.  **Physics Validity**: `DataValidator` rejects physically impossible poses.
2.  **Sequence Length**: Only sequences between **25 and 500 frames** are kept.
3.  **Artifact Threshold**: Samples with `max_artifact_score > 0.5` are dropped.

### 3.2 Quality Thresholds
- **Pretraining**: `quality_score >= 0.5`. Diverse, large scale.
- **SFT**: `quality_score >= 0.8`. High fidelity, instructional quality.
- **Camera Stability**: SFT samples must have stable camera movement (`stability >= 0.6`).

### 3.3 Source Balancing (New)
To prevent synthetic data from dominating the dataset (which can lead to hallucination):
- **Max Source Fraction**: No single source (e.g., Synthetic) can exceed **40%** of the SFT split.
- **Priority**: Real MoCap (AMASS, KIT-ML) is prioritized over Synthetic data during balancing.

### 3.4 Action Balancing
- **Max Action Fraction**: No single action class (e.g., "walk") can exceed **30%** of SFT.
- Ensures the model learns rare actions (e.g., "cartwheel") effectively.

## 4. Dataset Merging

Before curation, individual converted datasets must be merged into a unified training set.
`scripts/merge_datasets.py` handles this step with:

### 4.1 Source Balancing
- **Max Source Fraction**: Caps any single source at a configurable percentage (default: 30%).
- Prevents synthetic data or any single MoCap source from dominating.

### 4.2 Quality Filtering
- **Length Filter**: Removes sequences outside the 25-500 frame range.
- **Artifact Filter**: Optionally removes samples with high artifact scores.

### 4.3 Usage
```bash
# Merge all converted datasets with source balancing
python -m scripts.merge_datasets \
    --inputs data/humanml3d.pt data/kit_ml.pt data/babel.pt data/beat.pt data/synthetic.pt \
    --output data/merged_all.pt \
    --balance-sources \
    --max-source-fraction 0.3 \
    --filter-artifacts \
    --compute-stats
```

Output:
- `merged_all.pt`: Merged dataset ready for curation.
- `merged_all.stats.json`: Statistics on source distribution, action distribution, diversity scores.

## 5. Outputs

Running `scripts/prepare_curated_datasets.py` produces:
- `pretrain_data.pt` / `sft_data.pt`: Curated splits.
- `curation_stats.json`: Detailed report on dropped samples, artifact rates, and source distribution.

## 6. End-to-End Pipeline

The complete data preparation pipeline runs these steps in order:

```
1. Convert individual datasets (HumanML3D, KIT-ML, BABEL, BEAT, AMASS, etc.)
2. Merge datasets with source balancing (merge_datasets.py)
3. Curate into pretrain/SFT splits (prepare_curated_datasets.py)
4. Generate text embeddings (build_dataset_for_training.py)
5. Train model (train.py with appropriate config)
```

### 6.1 RunPod Automation

On RunPod, this is orchestrated by `runpod/data_prep_entrypoint.sh` with `USE_CURATED_DATA=true`.

**Merge Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MERGE_BALANCE_SOURCES` | `true` | Enable source balancing |
| `MERGE_MAX_SOURCE_FRACTION` | `0.3` | Max fraction per source |
| `MERGE_FILTER_ARTIFACTS` | `true` | Filter artifacts during merge |
| `MERGE_MAX_ARTIFACT_SCORE` | `0.4` | Max artifact score threshold |
| `MERGE_MIN_FRAMES` | `25` | Min sequence length |
| `MERGE_MAX_FRAMES` | `500` | Max sequence length |

**Curation Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `CURATION_MIN_QUALITY_PRETRAIN` | `0.5` | Min quality for pretrain |
| `CURATION_MIN_QUALITY_SFT` | `0.8` | Min quality for SFT |
| `CURATION_MIN_CAMERA_STABILITY_SFT` | `0.6` | Min camera stability for SFT |
| `CURATION_BALANCE_MAX_FRACTION` | `0.3` | Max fraction per action |

### 6.2 Model-Size Specific Settings

Each model size has tuned curation settings in its config file:

| Model | `max_source_fraction` | `max_artifact_score` | `min_quality_sft` |
|-------|----------------------|---------------------|-------------------|
| Small | 0.35 | 0.4 | 0.8 |
| Medium | 0.3 | 0.4 | 0.8 |
| Large | 0.25 | 0.35 | 0.85 |

Larger models use stricter thresholds to ensure higher-quality training data.
