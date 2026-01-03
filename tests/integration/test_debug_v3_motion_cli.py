import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_gen.joint_utils import joints_to_v3_segments_2d  # noqa: E402
from src.inference.exporter import MotionExporter  # noqa: E402


def _build_canonical_joints(T: int = 8) -> dict[str, np.ndarray]:
    """Create a simple upright canonical joint dictionary for testing.

    The exact coordinates are not important; they are chosen to be simple,
    human-like proportions that satisfy v3 connectivity and angle checks.
    """

    dtype = np.float32
    l_shoulder = np.array([-0.4, 1.4], dtype=dtype)
    l_elbow = np.array([-0.7, 1.1], dtype=dtype)
    l_wrist = np.array([-0.9, 0.8], dtype=dtype)

    r_shoulder = np.array([0.4, 1.4], dtype=dtype)
    r_elbow = np.array([0.7, 1.1], dtype=dtype)
    r_wrist = np.array([0.9, 0.8], dtype=dtype)

    l_hip = np.array([-0.25, -0.1], dtype=dtype)
    l_knee = np.array([-0.25, -0.9], dtype=dtype)
    l_ankle = np.array([-0.25, -1.7], dtype=dtype)

    r_hip = np.array([0.25, -0.1], dtype=dtype)
    r_knee = np.array([0.25, -0.9], dtype=dtype)
    r_ankle = np.array([0.25, -1.7], dtype=dtype)

    pelvis_center = 0.5 * (l_hip + r_hip)
    chest = np.array([0.0, 1.0], dtype=dtype)
    neck = np.array([0.0, 1.5], dtype=dtype)
    head_center = np.array([0.0, 2.0], dtype=dtype)

    base = {
        "pelvis_center": pelvis_center,
        "chest": chest,
        "neck": neck,
        "head_center": head_center,
        "l_shoulder": l_shoulder,
        "l_elbow": l_elbow,
        "l_wrist": l_wrist,
        "r_shoulder": r_shoulder,
        "r_elbow": r_elbow,
        "r_wrist": r_wrist,
        "l_hip": l_hip,
        "l_knee": l_knee,
        "l_ankle": l_ankle,
        "r_hip": r_hip,
        "r_knee": r_knee,
        "r_ankle": r_ankle,
    }

    return {name: np.broadcast_to(coord, (T, 2)).copy() for name, coord in base.items()}


@pytest.mark.integration
def test_debug_v3_motion_cli_runs_without_gui(tmp_path: Path) -> None:
    """The debug_v3_motion CLI should run and exit successfully with --no-gui.

    This uses a small deterministic v3 motion clip exported via MotionExporter
    in .motion format so the CLI exercises its JSON loading path and summary
    diagnostics.
    """

    joints = _build_canonical_joints(T=6)
    motion = joints_to_v3_segments_2d(joints)  # [T, 48]

    exporter = MotionExporter(fps=25)
    motion_tensor = torch.from_numpy(motion)
    json_str = exporter.export_to_json(motion_tensor, description="debug_cli_test")

    motion_path = tmp_path / "debug_cli_test.motion"
    motion_path.write_text(json_str, encoding="utf-8")

    script_path = Path("scripts/dev/debug_v3_motion.py").resolve()
    assert script_path.exists()

    cmd = [sys.executable, str(script_path), "--input", str(motion_path), "--no-gui"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    assert (
        result.returncode == 0
    ), f"CLI exited with code {result.returncode}, stderr=\n{result.stderr}"
