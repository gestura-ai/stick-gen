import numpy as np
import pytest
import torch

from src.data_gen.joint_utils import joints_to_v3_segments_2d
from src.data_gen.validator import DataValidator


def _build_canonical_joints_v3(T: int = 16) -> dict[str, np.ndarray]:
    """Build an upright canonical joint set compatible with v3 utilities.

    This mirrors the synthetic pose used in joint_utils v3 tests so that
    connectivity and basic limb geometry are well behaved.
    """

    dtype = np.float32

    pelvis_center = np.array([0.0, 0.0], dtype=dtype)
    chest = np.array([0.0, 1.0], dtype=dtype)
    neck = np.array([0.0, 1.5], dtype=dtype)
    head_center = np.array([0.0, 2.0], dtype=dtype)

    l_hip = np.array([-0.5, 0.0], dtype=dtype)
    r_hip = np.array([0.5, 0.0], dtype=dtype)

    l_knee = np.array([-0.5, -1.0], dtype=dtype)
    r_knee = np.array([0.5, -1.0], dtype=dtype)
    l_ankle = np.array([-0.5, -2.0], dtype=dtype)
    r_ankle = np.array([0.5, -2.0], dtype=dtype)

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


def test_check_joint_angles_v3_skips_non_v3_motion() -> None:
    """Non-v3 motion shapes should be skipped with a neutral score."""

    validator = DataValidator(fps=25)
    motion = torch.randn(40, 20)

    ok, score, reason = validator.check_joint_angles_v3(motion)

    assert ok
    assert score == 1.0
    assert "skipped" in reason.lower()


def test_check_joint_angles_v3_ok_on_upright_pose() -> None:
    """A simple upright v3 pose should pass the joint-angle check."""

    validator = DataValidator(fps=25)
    joints = _build_canonical_joints_v3(T=16)
    segments = joints_to_v3_segments_2d(joints)
    motion = torch.from_numpy(segments)

    ok, score, reason = validator.check_joint_angles_v3(motion)

    assert ok
    assert pytest.approx(score, rel=1e-6) == 1.0
    assert "joint angles" in reason.lower()


def test_check_joint_angles_v3_detects_collapsed_knee() -> None:
    """Collapsed knee angles should be reported as inconsistent."""

    validator = DataValidator(fps=25)
    joints = _build_canonical_joints_v3(T=16)

    # Collapse the left knee very close to the hip so the angle approaches 0°.
    joints["l_knee"][:, :] = joints["l_hip"][:, :] + np.array([0.0, 1e-4], dtype=np.float32)

    segments = joints_to_v3_segments_2d(joints)
    motion = torch.from_numpy(segments)

    ok, score, reason = validator.check_joint_angles_v3(motion)

    assert not ok
    assert score < 1.0
    assert "joint angle inconsistency" in reason.lower()

