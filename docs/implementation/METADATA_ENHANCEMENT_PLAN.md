# Metadata Enhancement Implementation Plan

**Goal:** Enrich the canonical sample schema with additional metadata fields to improve motion generation quality by providing richer conditioning signals.

**Status:** Planning Complete  
**Estimated Total Effort:** 3-4 weeks  
**Priority:** High (impacts training data quality)

---

## Executive Summary

The current data pipeline captures motion, physics, actions, camera, and basic metadata. This enhancement adds:

| Category | Fields | Impact | Effort |
|----------|--------|--------|--------|
| **Motion Style** | tempo, energy_level, smoothness | High - Distinguishes athletic vs relaxed motion | Medium |
| **Temporal** | original_fps, original_duration_sec | Medium - Preserves source timing info | Low |
| **Quality** | reconstruction_confidence, marker_quality | Medium - Enables quality-weighted training | Medium |
| **Subject** | height_cm, gender, age_group | Medium - Body-aware generation | High |
| **Emotion** | emotion_label, valence, arousal | High - Expressive generation | Medium |
| **Music** | bpm, beat_frames, genre | High - Dance sync (AIST++ only) | Medium |
| **Interaction** | contact_frames, interaction_role | High - Multi-actor coordination | High |

---

## Phase 1: Schema & Foundation (Week 1, Days 1-3)

**Goal:** Define all Pydantic models and create utility module skeleton.

### Tasks

| ID | Task | Complexity | Dependencies | Files |
|----|------|------------|--------------|-------|
| 1.1 | Define MotionStyleMetadata | Low | None | schema.py |
| 1.2 | Define SubjectMetadata | Low | None | schema.py |
| 1.3 | Define MusicMetadata | Low | None | schema.py |
| 1.4 | Define InteractionMetadata | Low | None | schema.py |
| 1.5 | Define TemporalMetadata | Low | None | schema.py |
| 1.6 | Define QualityMetadata | Low | None | schema.py |
| 1.7 | Define EmotionMetadata | Low | None | schema.py |
| 1.8 | Define EnhancedSampleMetadata | Low | 1.1-1.7 | schema.py |
| 1.9 | Update DATA_SCHEMA.md | Low | 1.8 | docs/ |
| 1.10 | Create metadata_extractors.py skeleton | Low | 1.8 | metadata_extractors.py |

**Deliverables:**
- All Pydantic models in `src/data_gen/schema.py`
- Updated documentation in `docs/features/DATA_SCHEMA.md`
- Empty `src/data_gen/metadata_extractors.py` with function stubs

---

## Phase 2: Motion Style Metrics (Week 1, Days 4-5 + Week 2, Days 1-2)

**Goal:** Implement computed motion style metrics applicable to ALL datasets.

**Priority:** HIGH - These are universally applicable and provide immediate value.

### Algorithm Details

```python
# Tempo: Based on velocity oscillation frequency
def compute_tempo(motion: Tensor, fps: int) -> float:
    velocities = motion[1:] - motion[:-1]  # [T-1, 20]
    vel_magnitude = velocities.norm(dim=-1).mean(dim=-1)  # [T-1]
    # Use autocorrelation to find dominant period
    # Normalize: slow walking ~0.2, running ~0.6, sprinting ~0.9
    
# Energy: Based on velocity and acceleration magnitude
def compute_energy_level(motion: Tensor, fps: int) -> float:
    velocities = motion[1:] - motion[:-1]
    accelerations = velocities[1:] - velocities[:-1]
    energy = velocities.norm().mean() + accelerations.norm().mean()
    # Normalize to 0-1 based on empirical dataset statistics
    
# Smoothness: Based on jerk (derivative of acceleration)
def compute_smoothness(motion: Tensor, fps: int) -> float:
    # Lower jerk = higher smoothness
    # Robot motion: low smoothness (~0.3)
    # Tai chi: high smoothness (~0.9)
```

### Tasks

| ID | Task | Complexity | Dependencies | Files |
|----|------|------------|--------------|-------|
| 2.1 | Implement compute_tempo() | Medium | 1.10 | metadata_extractors.py |
| 2.2 | Implement compute_energy_level() | Medium | 1.10 | metadata_extractors.py |
| 2.3 | Implement compute_smoothness() | Medium | 1.10 | metadata_extractors.py |
| 2.4 | Implement compute_motion_style() | Low | 2.1-2.3 | metadata_extractors.py |
| 2.5 | Add to convert_amass.py | Low | 2.4 | convert_amass.py |
| 2.6 | Add to convert_humanml3d.py | Low | 2.4 | convert_humanml3d.py |
| 2.7 | Add to convert_babel.py | Low | 2.4 | convert_babel.py |
| 2.8 | Add to convert_aist_plusplus.py | Low | 2.4 | convert_aist_plusplus.py |
| 2.9 | Add to convert_interhuman.py | Low | 2.4 | convert_interhuman.py |
| 2.10 | Add to remaining converters | Medium | 2.4 | 5 files |
| 2.11 | Run regression tests | Low | 2.5-2.10 | - |

---

## Phase 3: Temporal & Quality Metadata (Week 2, Days 3-5)

**Goal:** Capture source timing info and quality metrics.

**Priority:** MEDIUM - Easy wins with moderate impact.

### Tasks

| ID | Task | Complexity | Dependencies | Files |
|----|------|------------|--------------|-------|
| 3.1 | Implement extract_temporal_metadata() | Low | 1.10 | metadata_extractors.py |
| 3.2 | Implement compute_marker_quality() | Medium | 1.10 | metadata_extractors.py |
| 3.3 | Add temporal to all converters | Medium | 3.1 | 10 files |
| 3.4 | Add quality to AMASS | Medium | 3.2 | convert_amass.py |
| 3.5 | Add quality to HumanML3D | Medium | 3.2 | convert_humanml3d.py |
| 3.6 | Run regression tests | Low | 3.3-3.5 | - |

---

## Phase 4: Dataset-Specific Enrichments (Week 3)

**Goal:** Add specialized metadata unique to specific datasets.

**Priority:** MEDIUM-HIGH - Higher complexity but significant value for specific use cases.

### 4.1 Demographics (AMASS, HumanML3D)

**Source:** SMPL beta parameters encode body shape. Beta[0] correlates with height, Beta[1] with weight.

```python
def estimate_height_from_betas(betas: np.ndarray) -> float:
    # SMPL neutral height ~1.7m, beta[0] adjusts ±0.2m typically
    base_height = 170.0  # cm
    height_adjustment = betas[0] * 10.0  # Approximate scaling
    return base_height + height_adjustment
```

### 4.2 Music Metadata (AIST++)

**Source:** AIST++ includes music files with BPM annotations.

```python
def extract_aist_music_metadata(seq_name: str, music_dir: str) -> MusicMetadata:
    # Parse sequence name: gBR_sBM_c01_d04_mBR0_ch01
    # mBR0 = music ID, extract from music annotation files
    # Beat frames available in official annotations
```

### 4.3 Interaction Metadata (InterHuman)

**Source:** Compute from multi-actor motion data.

```python
def compute_interaction_metadata(motion: Tensor) -> InteractionMetadata:
    # motion: [T, 2, 20]
    # Contact frames: frames where actors are within threshold distance
    # Role: leader has higher velocity variance, follower reacts
```

### 4.4 Emotion Inference

**Text-based (HumanML3D, KIT-ML, BABEL):**
```python
EMOTION_KEYWORDS = {
    "happy": (0.7, 0.6),   # (valence, arousal)
    "angry": (-0.5, 0.9),
    "sad": (-0.7, 0.3),
    "excited": (0.6, 0.9),
    "calm": (0.3, 0.2),
}
```

**Motion-based (all datasets):**
```python
def infer_emotion_from_motion(motion: Tensor, fps: int) -> EmotionMetadata:
    energy = compute_energy_level(motion, fps)
    # High energy = high arousal
    # Expansive motion (large limb spread) = positive valence
```

### Tasks

| ID | Task | Complexity | Dependencies | Files |
|----|------|------------|--------------|-------|
| 4.1 | AMASS demographics | High | Phase 1 | convert_amass.py |
| 4.2 | HumanML3D demographics | Medium | Phase 1 | convert_humanml3d.py |
| 4.3 | AIST++ music metadata | High | Phase 1 | convert_aist_plusplus.py |
| 4.4 | InterHuman interaction | High | Phase 1 | convert_interhuman.py |
| 4.5 | infer_emotion_from_text() | Medium | Phase 1 | metadata_extractors.py |
| 4.6 | infer_emotion_from_motion() | Medium | 2.2 | metadata_extractors.py |
| 4.7 | Add emotion to text datasets | Low | 4.5 | humanml3d, kit_ml |
| 4.8 | Add emotion to BABEL | Low | 4.5 | convert_babel.py |
| 4.9 | Add emotion to other datasets | Low | 4.6 | 5 files |
| 4.10 | Run regression tests | Low | 4.1-4.9 | - |

---

## Phase 5: Validation & Testing (Week 4)

**Goal:** Ensure quality, backward compatibility, and comprehensive test coverage.

### Tasks

| ID | Task | Complexity | Dependencies | Files |
|----|------|------------|--------------|-------|
| 5.1 | Extend DataValidator | Medium | Phase 1 | validator.py |
| 5.2 | Unit tests for extractors | Medium | Phase 2-4 | test_metadata_extractors.py |
| 5.3 | Integration tests | Medium | Phase 2-4 | test_enhanced_converters.py |
| 5.4 | Backward compatibility test | Low | Phase 2-4 | test_backward_compat.py |
| 5.5 | Full pipeline validation | Medium | 5.1-5.4 | - |
| 5.6 | Documentation update | Low | 5.5 | docs/ |

---

## Implementation Priority Matrix

**Prioritize by: Impact × (1 / Effort)**

| Feature | Impact | Effort | Priority Score | Recommended Order |
|---------|--------|--------|----------------|-------------------|
| Motion Style (all) | High | Medium | **1.5** | 1st - Universal value |
| Temporal Metadata | Medium | Low | **2.0** | 2nd - Quick win |
| Emotion (text-based) | High | Medium | **1.5** | 3rd - Rich datasets |
| Quality Metrics | Medium | Medium | **1.0** | 4th - Training weights |
| Music (AIST++) | High | Medium | **1.5** | 5th - Dance-specific |
| Emotion (motion-based) | Medium | Medium | **1.0** | 6th - Fallback |
| Interaction (InterHuman) | High | High | **1.0** | 7th - Multi-actor |
| Demographics | Medium | High | **0.67** | 8th - Complex parsing |

---

## Converter Impact Summary

| Converter | Motion Style | Temporal | Quality | Subject | Emotion | Music | Interaction |
|-----------|:------------:|:--------:|:-------:|:-------:|:-------:|:-----:|:-----------:|
| convert_amass.py | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| convert_humanml3d.py | ✓ | ✓ | ✓ | ○ | ✓ | - | - |
| convert_babel.py | ✓ | ✓ | - | - | ✓ | - | - |
| convert_aist_plusplus.py | ✓ | ✓ | - | - | ✓ | ✓ | - |
| convert_interhuman.py | ✓ | ✓ | - | - | ✓ | - | ✓ |
| convert_kit_ml.py | ✓ | ✓ | - | - | ✓ | - | - |
| convert_ntu_rgbd.py | ✓ | ✓ | - | - | ✓ | - | - |
| convert_lsmb19.py | ✓ | ✓ | - | - | ✓ | - | - |
| convert_100style.py | ✓ | ✓ | - | - | ✓ | - | - |
| convert_beat.py | ✓ | ✓ | - | - | ✓ | - | - |

Legend: ✓ = Full support, ○ = Partial/optional, - = Not applicable

---

## Risk Mitigation

1. **Backward Compatibility:** All new fields are `Optional[T] = None`. Existing samples load without changes.

2. **Regression Prevention:** Run `pytest tests/` after each phase before merging.

3. **Incremental Rollout:** Each phase is independently deployable. Stop after any phase if needed.

4. **Performance:** Metadata extraction adds ~5-10% overhead. Acceptable for offline preprocessing.

---

## Success Metrics

1. **Coverage:** >90% of samples have motion_style and temporal metadata populated
2. **Quality:** All metadata values within expected ranges (validated by DataValidator)
3. **Compatibility:** Existing training scripts run without modification
4. **Performance:** Data pipeline completes within 2x original time

---

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/enhanced-metadata`
3. Begin Phase 1 implementation
4. Weekly progress reviews after each phase

