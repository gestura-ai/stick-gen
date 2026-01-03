import hashlib
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.data_gen.joint_utils import joints_to_v3_segments_2d
from src.data_gen.renderer import Renderer


def _build_canonical_joints(T: int = 8) -> dict[str, np.ndarray]:
    """Build a simple upright canonical joint configuration for testing.

    The joints are constant over time and satisfy the v3 connectivity
    constraints by construction. This mirrors the layout used in the v3 joint
    utility tests.
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


def _build_static_v3_motion(T: int = 8) -> np.ndarray:
    """Create a simple static v3 motion clip with shape ``[T, 48]``."""

    joints = _build_canonical_joints(T=T)
    return joints_to_v3_segments_2d(joints)


def test_render_v3_sequence_writes_non_empty_png(tmp_path: Path) -> None:
    """Renderer.render_v3_sequence should generate a non-empty PNG file."""

    motion = _build_static_v3_motion(T=12)
    renderer = Renderer()

    output_path = tmp_path / "v3_preview.png"
    renderer.render_v3_sequence(motion, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_v3_sequence_visual_regression(tmp_path: Path) -> None:
    """Basic visual regression check for the v3 renderer.

    This uses a deterministic static v3 motion clip and asserts on the
    resulting PNG's dimensions and a SHA256 hash of its raw pixel buffer.
    This is intentionally strict so that even small visual regressions are
    detected. If this test starts failing after a Matplotlib or renderer
    refactor, recompute the hash via a small helper script and update the
    golden values here.
    """

    motion = _build_static_v3_motion(T=12)
    renderer = Renderer()

    output_path = tmp_path / "v3_preview_regression.png"
    renderer.render_v3_sequence(motion, str(output_path))

    assert output_path.exists()

    img = plt.imread(output_path)

    # Golden shape from a canonical run on the static test clip.
    assert img.shape == (389, 389, 4)

    # Golden SHA256 hash over the raw pixel buffer (RGBA float32 0-1).
    h = hashlib.sha256(img.tobytes()).hexdigest()
    assert h == "3a589d188f7b6a6e9156ef88d6190937aeba775965ee14c0387e41796d0d68d1"
