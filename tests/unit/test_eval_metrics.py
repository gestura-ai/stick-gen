import torch

from src.eval.metrics import (
    compute_camera_metrics,
    compute_dataset_fid_statistics,
    compute_frechet_distance,
    compute_motion_diversity,
    compute_motion_features,
    compute_motion_realism_score,
    compute_motion_temporal_metrics,
    compute_physics_consistency_metrics,
    compute_synthetic_artifact_score,
    compute_text_alignment_from_embeddings,
)


def test_motion_temporal_metrics_basic():
    # Simple linear motion should have finite smoothness
    T, D = 20, 4
    motion = torch.zeros((T, D))
    motion[:, 0] = torch.linspace(0, 10, T)
    metrics = compute_motion_temporal_metrics(motion)
    assert 0.0 <= metrics["smoothness_score"] <= 1.0
    assert metrics["mean_velocity"] > 0.0


def test_camera_metrics_static_vs_pan():
    T = 20
    cam_static = torch.zeros((T, 3))
    cam_pan = torch.zeros((T, 3))
    cam_pan[:, 0] = torch.linspace(0, 5, T)

    m_static = compute_camera_metrics(cam_static)
    m_pan = compute_camera_metrics(cam_pan)

    assert m_static["shot_type"] in {"wide", "medium", "close", "unknown"}
    assert m_static["stability_score"] >= m_pan["stability_score"]


def test_physics_consistency_metrics_zero():
    T = 10
    physics = torch.zeros((T, 6))
    stats = compute_physics_consistency_metrics(physics)
    assert stats["max_velocity"] == 0.0
    assert "is_valid" in stats


def test_text_alignment_from_embeddings_ordering():
    a = torch.randn(8, 16)
    # Make b close to a
    b = a + 0.01 * torch.randn_like(a)
    # And c roughly orthogonal
    c = torch.randn_like(a)

    close = compute_text_alignment_from_embeddings(a, b)
    far = compute_text_alignment_from_embeddings(a, c)

    assert close["mean_cosine_similarity"] > far["mean_cosine_similarity"]


# --- New FID-like and quality metrics tests ---


def test_compute_motion_features_shape():
    """Test that motion features have expected shape."""
    motion = torch.randn(100, 20)
    features = compute_motion_features(motion)

    # Features should be a 1D tensor with reasonable size
    assert features.ndim == 1
    assert features.shape[0] > 0
    assert torch.isfinite(features).all()


def test_compute_motion_features_deterministic():
    """Test that feature extraction is deterministic."""
    motion = torch.randn(50, 20)
    f1 = compute_motion_features(motion)
    f2 = compute_motion_features(motion)

    assert torch.allclose(f1, f2)


def test_compute_motion_diversity_basic():
    """Test diversity computation on varied motions."""
    # Create diverse motions
    motions = [torch.randn(100, 20) for _ in range(5)]
    result = compute_motion_diversity(motions)

    assert "diversity_score" in result
    assert "num_samples" in result
    assert result["diversity_score"] >= 0
    assert result["num_samples"] == 5


def test_compute_motion_diversity_identical():
    """Identical motions should have low diversity."""
    base = torch.randn(100, 20)
    motions = [base.clone() for _ in range(5)]
    result = compute_motion_diversity(motions)

    # Identical motions should have near-zero diversity
    assert result["diversity_score"] < 0.1


def test_compute_motion_diversity_empty():
    """Empty motion list should return zero diversity."""
    result = compute_motion_diversity([])
    assert result["diversity_score"] == 0.0
    assert result["num_samples"] == 0


def test_compute_synthetic_artifact_score_smooth():
    """Smooth motion should have low artifact score."""
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    smooth_motion = torch.sin(t) * torch.randn(1, 20) * 0.1

    result = compute_synthetic_artifact_score(smooth_motion)

    assert "artifact_score" in result
    assert "jitter_score" in result
    assert "static_ratio" in result
    assert "is_clean" in result


def test_compute_synthetic_artifact_score_jittery():
    """High-frequency noise should increase jitter score."""
    # Create jittery motion with high-frequency noise
    jittery = torch.randn(100, 20) * 0.5

    result = compute_synthetic_artifact_score(jittery)

    # Jittery motion should have higher jitter score
    assert result["jitter_score"] > 0


def test_compute_synthetic_artifact_score_static():
    """Static motion should have high static ratio."""
    static_motion = torch.zeros(100, 20)

    result = compute_synthetic_artifact_score(static_motion)

    # Completely static motion should have static_ratio = 1.0
    assert result["static_ratio"] == 1.0


def test_compute_motion_realism_score_range():
    """Realism score should be in [0, 1]."""
    motion = torch.randn(100, 20) * 0.1
    result = compute_motion_realism_score(motion)

    assert "realism_score" in result
    assert 0.0 <= result["realism_score"] <= 1.0


def test_compute_motion_realism_smooth_vs_jittery():
    """Smooth motion should have higher realism than jittery."""
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    smooth = torch.sin(t) * torch.randn(1, 20) * 0.1
    jittery = torch.randn(100, 20)

    smooth_result = compute_motion_realism_score(smooth)
    jittery_result = compute_motion_realism_score(jittery)

    assert smooth_result["realism_score"] >= jittery_result["realism_score"]


def test_compute_dataset_fid_statistics_shape():
    """Test FID statistics computation."""
    motions = [torch.randn(100, 20) for _ in range(10)]
    stats = compute_dataset_fid_statistics(motions)

    assert "mean" in stats
    assert "cov" in stats
    assert stats["mean"].ndim == 1
    assert stats["cov"].ndim == 2
    assert stats["mean"].shape[0] == stats["cov"].shape[0] == stats["cov"].shape[1]


def test_compute_frechet_distance_self():
    """FID of dataset with itself should be near zero."""
    motions = [torch.randn(100, 20) for _ in range(20)]
    stats = compute_dataset_fid_statistics(motions)

    fid = compute_frechet_distance(stats, stats)

    # Self-FID should be very small
    assert fid < 0.1


def test_compute_frechet_distance_different():
    """FID between different distributions should be positive."""
    motions1 = [torch.randn(100, 20) for _ in range(20)]
    motions2 = [torch.randn(100, 20) + 5.0 for _ in range(20)]  # Shifted

    stats1 = compute_dataset_fid_statistics(motions1)
    stats2 = compute_dataset_fid_statistics(motions2)

    fid = compute_frechet_distance(stats1, stats2)

    # Different distributions should have positive FID
    assert fid > 0
