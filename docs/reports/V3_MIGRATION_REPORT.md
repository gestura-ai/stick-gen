# Stick-Gen v3 Motion Schema Migration Report

**Author:** Gestura AI / Stick-Gen maintainers  
**Scope:** Migration from legacy v1 (20D, 5 segments) to canonical v3 (48D, 12 segments) across data, models, rendering, and docs.

---

## 1. Canonical v3 Schema Overview

- **Representation:**
  - 12 line segments per frame, each as `(x1, y1, x2, y2)` → **48D**.
  - Fixed segment ordering shared across converters, utils, and renderer.
- **Canonical joint set (2D):**
  - `pelvis_center`, `chest`, `neck`, `head_center`
  - `l_shoulder`, `l_elbow`, `l_wrist`, `r_shoulder`, `r_elbow`, `r_wrist`
  - `l_hip`, `l_knee`, `l_ankle`, `r_hip`, `r_knee`, `r_ankle`
- **Construction:**
  - All dataset skeletons are mapped to this joint set and then to v3 segments via
    `joints_to_v3_segments_2d`.
- **Connectivity guarantees:**
  - Enforced by `validate_v3_connectivity(segments)` in `src/data_gen/joint_utils.py`.
  - Examples:
    - `neck` is shared by head and upper‑torso segments.
    - `chest` links upper and lower torso.
    - `pelvis_center` is the midpoint of left/right hips and anchors both legs.

The v3 schema is **canonical** for all new data generation, training, and visualization.
Legacy v1 (20D, 5 segments) is retained only for backward‑compatible export paths.

---

## 2. Dataset Converters (Now v3-Native)

All primary dataset converters now emit **[T, 48]** v3 motion with strict connectivity
validation:

- `src/data_gen/convert_amass.py`
- `src/data_gen/convert_humanml3d.py`
- `src/data_gen/convert_kit_ml.py`
- `src/data_gen/convert_ntu_rgbd.py`
- `src/data_gen/convert_aist_plusplus.py`
- `src/data_gen/convert_100style.py`

Key changes:

- All converters map dataset joints → canonical joints → v3 segments.
- `validate_v3_connectivity` is called prior to writing motion.
- 100STYLE BVH converter was **rewritten** to be v3‑first rather than v1‑centric.

Unit tests in `tests/unit/test_converters.py` and related fixtures provide
small‑scale, deterministic examples covering each converter.

---

## 3. Joint Utilities and Inversion

**File:** `src/data_gen/joint_utils.py`

- `joints_to_v3_segments_2d(joints_2d)`
  - Builds `[T, 12, 4]` (or flattened `[T, 48]`) v3 segments from canonical joints.
- `validate_v3_connectivity(segments)`
  - Enforces exact equalities at shared joints (neck, chest, pelvis center,
    elbows, knees, etc.).
- `v3_segments_to_joints_2d(segments)`
  - Inverts v3 segments back to canonical 2D joints.

Tests:

- `tests/unit/test_joint_utils_v3.py`:
  - Validates segment construction.
  - Verifies round‑trip `joints → segments → joints` within tolerance.

---

## 4. v3 Renderer Integration

**File:** `src/data_gen/renderer.py`

- New method: `Renderer.render_v3_sequence(motion, output_path, ...)`.
- Accepts v3 motion as `[T, 48]` or `[T, 12, 4]`.
- Validates connectivity via `validate_v3_connectivity`.
- Reconstructs joints via `v3_segments_to_joints_2d`.
- Renders a **single high‑quality static frame** (mid‑sequence) with:
  - Rounded limb lines (`solid_capstyle="round"`, uniform thickness).
  - Circular joint markers (elbows, knees, wrists, ankles, shoulders, hips, neck).
  - Circular head sized from neck→head_center distance with clean outline.

Test:

- `tests/unit/test_v3_renderer.py` ensures that rendering a small canonical
  v3 motion clip produces a non‑empty PNG.

Legacy `render_raw_frames` remains for 20D export compatibility only.

---

## 5. Configuration and Training Alignment

- Training configs under `configs/` now assume **48D** v3 motion:
  - Model `input_dim`/`motion_dim` fields updated to 48.
  - Any old references to 20D inputs have been removed or marked legacy.
- Integration tests (e.g. `tests/integration/test_training_start.py`) verify that
  training can start end‑to‑end with v3 data and configs.

Users migrating from v1 must:

1. Update any custom model configs to use `input_dim=48` (or equivalent).
2. Regenerate datasets using the v3 converters listed above.

---

## 6. Remaining Legitimate v1 (20D) References

v1 (20D, 5‑segment) representations are intentionally retained only for
**backward‑compatible export and visualization**, not for training or new data
pipelines.

Examples (non‑exhaustive, but representative):

- Legacy web renderer or lightweight export scripts that expect 5 line segments.
- Documentation notes explaining how historic `.motion` files were structured.

In all such locations, comments/docs now:

- Explicitly label v1/20D as **legacy**.
- Clarify that new training and dataset generation should use v3 [T, 48].

---

## 7. Validation Summary

Focused v3 test sweep (examples):

- `tests/unit/test_joint_utils_v3.py`
- `tests/unit/test_converters.py`
- `tests/unit/test_metadata_extractors.py`
- `tests/unit/test_parallax_augmentation.py`
- `tests/unit/test_v3_renderer.py`

These suites collectively confirm that:

- v3 segments are constructed consistently across datasets.
- Connectivity invariants hold.
- Metadata and parallax augmentation remain compatible with the new schema.
- The renderer can consume v3 motion and produce high‑quality previews.

This report should be treated as the canonical reference for the v3 migration
and kept in sync with future schema or renderer changes.

