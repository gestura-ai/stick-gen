# Changelog

All notable changes to Stick Gen are documented in this file.

## [2.0.1] - 2025-12-23

### Fixed

#### Stick Figure Anatomical Alignment
- **Fixed segment ordering in all data converters** to match canonical format
  - Canonical order: `[torso, l_leg, r_leg, l_arm, r_arm]` (defined in `src/inference/exporter.py`)
  - Previously arms were at indices 1-2 and legs at 3-4 (incorrect)
  - Now legs are at indices 1-2 and arms at 3-4 (correct canonical format)
- **Fixed anatomical attachment points**:
  - Legs now attach at hip center (not torso center)
  - Arms attach at neck/shoulder center (not hip)
- **Updated converters**:
  - `convert_ntu_rgbd.py::joints_to_stick()` - NTU RGB+D skeleton conversion
  - `convert_amass.py::smpl_to_stick_figure()` - AMASS/SMPL skeleton conversion
  - `convert_humanml3d.py::_features_to_stick()` - HumanML3D feature extraction
  - `convert_aist_plusplus.py::keypoints3d_to_stick()` - AIST++ keypoint conversion
  - `convert_100style.py::extract_v3_segments_from_positions()` - 100STYLE BVH conversion
- **Added tests** for canonical segment ordering validation
- **Updated documentation** in `docs/architecture/RENDERING.md` with segment format specification

### Note
Data generated with previous versions may have incorrect anatomical alignment.
Regenerate training data after updating to this version.

## [2.0.0] - 2025-12-08

### Major Improvements - Spatial Movement & Realism

This release represents a complete overhaul of the system with focus on realistic spatial movement, physics-based motion, and state-of-the-art embeddings.

### Added

#### Spatial Movement System
- **ACTION_VELOCITIES mapping**: 57 actions with real-world speeds (m/s → units/s)
  - Walking: 1.9 units/s (1.3 m/s)
  - Running: 7.4 units/s (5.0 m/s)
  - Sprinting: 11.8 units/s (8.0 m/s)
  - All 57 actions from ActionType enum
- **OBJECT_SCALES mapping**: 25+ objects with realistic sizes
  - Human: 2.5 units (1.7m)
  - Table: 1.1 units (0.75m)
  - Tree: 15.0 units (10m)
- **Movement paths**: Waypoint-based trajectories for complex motion
  - Baseball players run around bases
  - Space explorers walk from ship to aliens
- **Position update system**: Actors traverse space realistically
  - `update_position()` method in StickFigure class
  - `_interpolate_path()` for waypoint interpolation
  - Frame-by-frame position updates

#### Model Architecture Enhancements
- **15.5M parameters** (upgraded from 5.2M)
  - d_model: 384 (from 256)
  - nhead: 12 (from 8)
  - num_layers: 8 (from 6)
- **Multi-task learning**: 3 decoder heads
  - Pose decoder (joint positions)
  - Position decoder (x, y coordinates)
  - Velocity decoder (vx, vy)
- **Enhanced text projection**: 1024 → 512 → 384 (deeper network)
- **Enhanced motion embedding**: 10 → 192 → 384

#### Training Improvements
- **100k training samples** (up from 5k)
- **500k total with 5x augmentation**:
  - Speed variation (0.8x - 1.2x)
  - Position jitter (±0.5 units)
  - Scale variation (0.9x - 1.1x)
  - Horizontal mirroring
- **10-second sequences** (250 frames @ 25fps, up from 5s)
- **Temporal consistency loss**: Penalizes jerky motion
- **80/10/10 train/val/test split** (from 90/10)
- **Gradient accumulation**: 4 steps for effective batch size of 16
- **Gradient clipping**: max_norm=1.0
- **Comprehensive evaluation metrics**:
  - Smoothness error
  - Position accuracy
  - Velocity accuracy

#### Embeddings Upgrade
- **BAAI/bge-large-en-v1.5**: Top-5 on MTEB leaderboard (Dec 2025)
- **1024-dim embeddings** (from 1536-dim Qwen2.5)
- **CPU-compatible**: No flash_attn dependency required
- **Higher quality**: Better semantic understanding

#### New Scripts & Tools
- `run_full_training_pipeline.sh`: Automated 3-phase training
- `test_improvements.py`: Spatial movement verification
- `quick_test.py`: Fast testing before full training

### Changed

- **Sequence duration**: 5s → 10s (250 frames)
- **Training samples**: 5k → 100k base (500k with augmentation)
- **Model size**: 5.2M → 15.5M parameters
- **Embedding model**: Qwen2.5-1.5B → BAAI/bge-large-en-v1.5
- **Embedding dimension**: 1536 → 1024
- **Train/val split**: 90/10 → 80/10/10
- **Batch size**: 8 → 4 (CPU-optimized with gradient accumulation)

### Fixed

- **Actors stuck in place**: Now move through space with realistic velocities
- **Unrealistic motion**: Added temporal consistency loss for smooth animations
- **Limited action vocabulary**: Expanded to 57 actions with proper velocities
- **Object scale inconsistencies**: Added OBJECT_SCALES mapping for realism
- **Flash attention dependency**: Switched to CPU-compatible embedding model
- **Missing SPRINT action**: Added to ActionType enum with animation
- **Incomplete ACTION_VELOCITIES**: All 57 actions now mapped
- **Incomplete OBJECT_SCALES**: All objects from ObjectType now mapped

### Performance

- **Training speed**: ~85 samples/sec on CPU
- **Inference speed**: ~2-3 seconds per 10-second animation
- **Model size**: 62MB (FP32)
- **Total training time**: 36-48 hours on CPU for 100k samples

## [1.0.0] - 2025-11-XX

### Initial Release

- Basic transformer model (5.2M parameters)
- Synthetic data generation
- Text-to-animation pipeline
- 5k training samples
- 5-second sequences
- Qwen2.5-1.5B embeddings
- Procedural animations
- CLI interface

