"""Unit tests for v3 joint utilities.

These tests cover the canonical joints -> v3 segments mapping and strict
connectivity validation for the 12-segment, 48-dim schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_gen.joint_utils import (
    V3_SEGMENT_NAMES,
    add_pose_noise_v3,
    joints_to_v3_segments_2d,
    mirror_v3,
    normalize_skeleton_height,
    smooth_joints_over_time,
    time_stretch_v3,
    v3_segments_to_joints_2d,
    validate_segment_lengths_v3,
    validate_v3_connectivity,
)


def _build_canonical_joints(T: int = 5) -> dict[str, np.ndarray]:
    """Build a simple, perfectly connected canonical joint set.

    The joints are constant over time and arranged in an upright pose so that
    all connectivity constraints are satisfied by construction.
    """
    dtype = np.float32

    # Axial joints
    pelvis_center = np.array([0.0, 0.0], dtype=dtype)
    chest = np.array([0.0, 1.0], dtype=dtype)
    neck = np.array([0.0, 1.5], dtype=dtype)
    head_center = np.array([0.0, 2.0], dtype=dtype)

    # Hips (symmetric around pelvis_center)
    l_hip = np.array([-0.5, 0.0], dtype=dtype)
    r_hip = np.array([0.5, 0.0], dtype=dtype)

    # Knees and ankles
    l_knee = np.array([-0.5, -1.0], dtype=dtype)
    r_knee = np.array([0.5, -1.0], dtype=dtype)
    l_ankle = np.array([-0.5, -2.0], dtype=dtype)
    r_ankle = np.array([0.5, -2.0], dtype=dtype)

    # Shoulders and arms
    l_shoulder = np.array([-0.5, 1.2], dtype=dtype)
    r_shoulder = np.array([0.5, 1.2], dtype=dtype)
    l_elbow = np.array([-0.9, 0.8], dtype=dtype)
    r_elbow = np.array([0.9, 0.8], dtype=dtype)
    l_wrist = np.array([-1.2, 0.4], dtype=dtype)
    r_wrist = np.array([1.2, 0.4], dtype=dtype)

    base = {
        "pelvis_center": pelvis_center,
        "chest": chest,
        "neck": neck,
        "head_center": head_center,
        "l_shoulder": l_shoulder,
        "r_shoulder": r_shoulder,
        "l_elbow": l_elbow,
        "r_elbow": r_elbow,
        "l_wrist": l_wrist,
        "r_wrist": r_wrist,
        "l_hip": l_hip,
        "r_hip": r_hip,
        "l_knee": l_knee,
        "r_knee": r_knee,
        "l_ankle": l_ankle,
        "r_ankle": r_ankle,
    }

    return {name: np.broadcast_to(coord, (T, 2)).copy() for name, coord in base.items()}


def test_joints_to_v3_segments_shape_and_dtype() -> None:
    """joints_to_v3_segments_2d should produce [T, 48] float32 arrays."""
    joints = _build_canonical_joints(T=10)
    segments = joints_to_v3_segments_2d(joints)

    assert segments.shape == (10, 48)
    assert segments.dtype == np.float32

    # Connectivity should hold for valid construction
    validate_v3_connectivity(segments)


def test_joints_to_v3_segments_preserves_segment_order() -> None:
    """Segment ordering must match V3_SEGMENT_NAMES length and count."""
    joints = _build_canonical_joints(T=3)
    segments = joints_to_v3_segments_2d(joints, flatten=False)

    assert segments.shape[1] == len(V3_SEGMENT_NAMES)


def test_validate_v3_connectivity_accepts_3d_shape() -> None:
    """Connectivity validation should accept [T, 12, 4] input as well."""
    joints = _build_canonical_joints(T=4)
    segments = joints_to_v3_segments_2d(joints, flatten=False)

    # Should not raise
    validate_v3_connectivity(segments)


def test_validate_v3_connectivity_raises_on_broken_elbow() -> None:
    """Breaking an elbow joint should trigger a connectivity error."""
    joints = _build_canonical_joints(T=4)
    segments = joints_to_v3_segments_2d(joints, flatten=False)

    # Perturb the left elbow connection between upper arm and forearm
    segments[:, 4, 0] += 0.1  # Move l_forearm.start away from l_upper_arm.end

    with pytest.raises(ValueError) as excinfo:
        validate_v3_connectivity(segments)

    assert "elbow" in str(excinfo.value).lower()


def test_v3_segments_to_joints_round_trip() -> None:
    """v3_segments_to_joints_2d should invert joints_to_v3_segments_2d."""
    joints = _build_canonical_joints(T=7)
    segments = joints_to_v3_segments_2d(joints)
    reconstructed = v3_segments_to_joints_2d(segments, validate=True)

    assert set(reconstructed.keys()) == set(joints.keys())
    for name, original in joints.items():
        assert np.allclose(reconstructed[name], original)


def test_normalize_skeleton_height_matches_target() -> None:
    """normalize_skeleton_height should approximately match target height."""
    joints = _build_canonical_joints(T=5)
    target_height = 2.0
    scaled = normalize_skeleton_height(joints, target_height=target_height)

    head = scaled["head_center"][0]
    l_ankle = scaled["l_ankle"][0]
    height = np.linalg.norm(head - l_ankle)
    assert np.isclose(height, target_height, atol=1e-3)


def test_smooth_joints_over_time_reduces_variance() -> None:
    """Temporal smoothing should reduce variance on noisy joints."""
    rng = np.random.default_rng(0)
    joints = _build_canonical_joints(T=32)
    noisy_head = joints["head_center"].copy()
    noisy_head += rng.normal(scale=0.05, size=noisy_head.shape).astype(np.float32)
    joints["head_center"] = noisy_head

    smoothed = smooth_joints_over_time(joints, window_size=5)
    var_before = float(noisy_head[:, 1].var())
    var_after = float(smoothed["head_center"][:, 1].var())
    assert var_after < var_before


def test_mirror_v3_flips_x_coordinates() -> None:
    """mirror_v3 should flip x coordinates while preserving connectivity."""
    joints = _build_canonical_joints(T=4)
    segments = joints_to_v3_segments_2d(joints)
    mirrored = mirror_v3(segments)

    # Original and mirrored x coordinates should be negatives of each other.
    orig = segments.reshape(4, 12, 4)
    mirr = mirrored.reshape(4, 12, 4)
    assert np.allclose(orig[:, :, 0] + mirr[:, :, 0], 0.0)
    assert np.allclose(orig[:, :, 2] + mirr[:, :, 2], 0.0)

    # Connectivity must still hold.
    validate_v3_connectivity(mirrored)


def test_time_stretch_v3_changes_length_and_preserves_connectivity() -> None:
    """time_stretch_v3 should resample while keeping segments connected."""
    joints = _build_canonical_joints(T=8)
    segments = joints_to_v3_segments_2d(joints)

    stretched = time_stretch_v3(segments, factor=0.5)
    assert stretched.shape[0] == 4
    assert stretched.shape[1] == 48
    validate_v3_connectivity(stretched)


def test_add_pose_noise_v3_preserves_connectivity_and_adds_noise() -> None:
    """add_pose_noise_v3 should modify poses but keep connectivity valid."""
    joints = _build_canonical_joints(T=6)
    segments = joints_to_v3_segments_2d(joints)

    rng = np.random.default_rng(123)
    noised = add_pose_noise_v3(segments, std=0.05, rng=rng)
    assert noised.shape == segments.shape
    validate_v3_connectivity(noised)
    # At least some coordinates should change.
    assert not np.allclose(noised, segments)


def test_validate_segment_lengths_v3_raises_on_degenerate_segment() -> None:
    """validate_segment_lengths_v3 should catch zero-length segments."""
    joints = _build_canonical_joints(T=3)
    segments = joints_to_v3_segments_2d(joints, flatten=False)
    segments[:, 0, :] = 0.0  # Collapse the head segment.

    with pytest.raises(ValueError):
        validate_segment_lengths_v3(segments)

