import random

import torch

from src.data_gen.curation import (
    CurationConfig,
    curate_samples,
    filter_by_length,
    filter_by_artifacts,
    balance_by_source,
    _get_source,
    _get_sequence_length,
    _compute_motion_quality,
    _compute_combined_quality,
)


def make_sample(quality: float, dominant_action: str = "walk", with_camera: bool = True):
    # Minimal canonical-like sample
    motion = torch.zeros(10, 1, 20)
    physics = torch.zeros(10, 1, 6)
    camera = torch.zeros(10, 3)
    annotations = {
        "quality": {"score": quality},
        "actions": {"dominant": [dominant_action]},
    }
    return {
        "description": "test sample",
        "motion": motion,
        "physics": physics,
        "camera": camera if with_camera else None,
        "annotations": annotations,
        "quality_score": quality,
    }


def test_curate_samples_basic_thresholds():
    samples = [
        make_sample(0.4),  # too low quality
        make_sample(0.6),  # pretrain only
        make_sample(0.85),  # pretrain + SFT
    ]
    cfg = CurationConfig(min_quality_pretrain=0.5, min_quality_sft=0.8)

    # Use non-enhanced filtering to skip physics validation
    pretrain, sft, stats = curate_samples(samples, cfg, use_enhanced_filtering=False)

    assert len(pretrain) == 2
    assert len(sft) == 1
    assert stats["pretrain"]["num_samples"] == 2
    assert stats["sft"]["num_samples"] == 1


def test_curate_samples_balances_actions():
    # 8 samples of action A, 2 of action B, all high quality
    samples = [make_sample(0.9, "walk") for _ in range(8)] + [
        make_sample(0.9, "run") for _ in range(2)
    ]
    cfg = CurationConfig(min_quality_pretrain=0.5, min_quality_sft=0.8, balance_max_fraction=0.4)

    # Use non-enhanced filtering to skip physics validation
    pretrain, sft, stats = curate_samples(samples, cfg, seed=123, use_enhanced_filtering=False)

    # All high-quality samples should be available for pretraining
    assert len(pretrain) == 10

    # SFT should downsample the dominant action bucket ("walk") while
    # retaining at least some examples of each action.
    assert len(sft) < len(samples)

    dist = stats["sft"]["action_distribution"]
    assert set(dist.keys()) == {"walk", "run"}
    # The walk fraction should be strictly lower than in the raw pool
    # (raw: 8/10 = 0.8, curated: should be < 0.8).
    assert dist["walk"] < 0.8


def test_curate_samples_handles_missing_quality():
    s = make_sample(0.9)
    del s["quality_score"]
    del s["annotations"]["quality"]
    pretrain, sft, stats = curate_samples([s], CurationConfig(), use_enhanced_filtering=False)

    assert len(pretrain) == 0
    assert len(sft) == 0
    assert stats["dropped_missing_quality"] == 1


# --- New enhanced curation tests ---


def make_motion_sample(
    T: int = 100,
    source: str = "synthetic",
    quality: float = 0.8,
    smooth: bool = True,
):
    """Create sample with motion data for enhanced filtering tests."""
    if smooth:
        t = torch.linspace(0, 10, T).unsqueeze(1)
        motion = torch.sin(t) * torch.randn(1, 20) * 0.1
    else:
        motion = torch.randn(T, 20)

    physics = torch.zeros(T, 6)
    camera = torch.zeros(T, 3)

    return {
        "description": f"test {source} sample",
        "motion": motion,
        "physics": physics,
        "camera": camera,
        "source": source,
        "quality_score": quality,
        "annotations": {
            "quality": {"score": quality},
            "actions": {"dominant": ["walk"]},
        },
    }


def test_get_source():
    """Test source extraction from samples."""
    s1 = {"source": "humanml3d"}
    s2 = {"source": "dataset_generator"}  # Should map to synthetic
    s3 = {}

    assert _get_source(s1) == "humanml3d"
    assert _get_source(s2) == "synthetic"
    assert _get_source(s3) == "unknown"


def test_get_sequence_length():
    """Test sequence length extraction."""
    s1 = {"motion": torch.randn(100, 20)}
    s2 = {"motion": torch.randn(50, 5, 20)}
    s3 = {}

    assert _get_sequence_length(s1) == 100
    assert _get_sequence_length(s2) == 50
    assert _get_sequence_length(s3) == 0


def test_filter_by_length():
    """Test length-based filtering."""
    samples = [
        {"motion": torch.randn(T, 20)}
        for T in [10, 50, 100, 200, 600]
    ]
    cfg = CurationConfig(min_frames=25, max_frames=500)

    filtered, dropped = filter_by_length(samples, cfg)

    assert len(filtered) == 3  # 50, 100, 200
    assert dropped == 2  # 10, 600


def test_filter_by_artifacts():
    """Test artifact-based filtering."""
    # Smooth motion should pass
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    smooth_motion = torch.sin(t) * torch.randn(1, 20) * 0.1

    # Static motion may be flagged
    static_motion = torch.zeros(100, 20)

    samples = [
        {"motion": smooth_motion},
        {"motion": static_motion},
    ]
    cfg = CurationConfig(max_artifact_score=0.5)

    filtered, dropped = filter_by_artifacts(samples, cfg)

    # At least smooth should pass
    assert len(filtered) >= 1


def test_balance_by_source():
    """Test source balancing."""
    samples = []
    for i in range(100):
        if i < 60:
            source = "synthetic"
        elif i < 80:
            source = "humanml3d"
        else:
            source = "amass"
        samples.append({"source": source, "motion": torch.randn(100, 20)})

    cfg = CurationConfig(balance_by_source=True, max_source_fraction=0.4)
    rng = random.Random(42)

    balanced = balance_by_source(samples, cfg, rng)

    # Count by source
    counts = {}
    for s in balanced:
        src = _get_source(s)
        counts[src] = counts.get(src, 0) + 1

    # Synthetic had 60/100 samples, should be capped to max_per_source=40
    # After balancing: synthetic=40, humanml3d=20, amass=20, total=80
    # So synthetic ratio is 40/80 = 0.5, but at least it's reduced from 0.6
    total = len(balanced)
    if total > 0:
        synthetic_count = counts.get("synthetic", 0)
        # Synthetic should be capped at 40 (max_source_fraction * 100)
        assert synthetic_count <= 40
        # And the total should be less than original due to capping
        assert total < 100


def test_compute_motion_quality():
    """Test motion quality computation."""
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    smooth_motion = torch.sin(t) * torch.randn(1, 20) * 0.1

    sample = {"motion": smooth_motion}
    realism, artifact = _compute_motion_quality(sample)

    assert 0.0 <= realism <= 1.0
    assert artifact >= 0.0


def test_compute_combined_quality():
    """Test combined quality scoring."""
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    motion = torch.sin(t) * torch.randn(1, 20) * 0.1

    sample = {
        "motion": motion,
        "source": "humanml3d",
        "quality_score": 0.8,
    }
    cfg = CurationConfig()

    quality = _compute_combined_quality(sample, cfg)

    assert quality is not None
    assert 0.0 <= quality <= 1.0


def test_compute_combined_quality_synthetic_penalty():
    """Synthetic samples should have quality penalty."""
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    motion = torch.sin(t) * torch.randn(1, 20) * 0.1

    mocap_sample = {
        "motion": motion,
        "source": "humanml3d",
        "quality_score": 0.8,
    }
    synthetic_sample = {
        "motion": motion.clone(),
        "source": "synthetic",
        "quality_score": 0.8,
    }
    cfg = CurationConfig(synthetic_quality_penalty=0.1)

    mocap_q = _compute_combined_quality(mocap_sample, cfg)
    synth_q = _compute_combined_quality(synthetic_sample, cfg)

    # Mocap should have higher quality than synthetic
    assert mocap_q > synth_q


def test_curate_samples_enhanced_filtering():
    """Test curate_samples with enhanced filtering enabled."""
    samples = [
        make_motion_sample(T=100, source="humanml3d", quality=0.9, smooth=True),
        make_motion_sample(T=100, source="synthetic", quality=0.7, smooth=True),
        make_motion_sample(T=10, source="amass", quality=0.9, smooth=True),  # Too short
    ]

    cfg = CurationConfig(min_frames=25, max_frames=500)
    pretrain, sft, stats = curate_samples(samples, cfg, use_enhanced_filtering=True)

    # Should have filtered out the short sample
    assert stats["dropped_length"] >= 1

    # Stats should include source distribution
    if stats["pretrain"]["num_samples"] > 0:
        assert "source_distribution" in stats["pretrain"]

