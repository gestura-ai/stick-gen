import shutil
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
    """Create a simple upright canonical joint dictionary for v3 tests.

    Reuses the same basic proportions as the v3 renderer tests so that
    connectivity and joint-angle checks are satisfied.
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
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js runtime not available")
def test_threejs_parallax_v3_smoke_renders_pngs(tmp_path: Path) -> None:
    """End-to-end smoke test for the v3 -> Three.js parallax pipeline.

    This creates a tiny deterministic v3 motion clip, exports it as a .motion
    file via :class:`MotionExporter`, and invokes the real
    ``threejs_parallax_renderer.js`` script. The test only asserts that at
    least one PNG and a metadata JSON file are produced, providing a basic
    guardrail against breaking the 2.5D pipeline.
    """

    joints = _build_canonical_joints(T=8)
    motion = joints_to_v3_segments_2d(joints)  # [T, 48]

    exporter = MotionExporter(fps=25)
    motion_tensor = torch.from_numpy(motion)
    json_str = exporter.export_to_json(motion_tensor, description="threejs_smoke_test")

    motion_path = tmp_path / "threejs_smoke_test.motion"
    motion_path.write_text(json_str, encoding="utf-8")

    repo_root = Path(__file__).parent.parent.parent
    node_script = repo_root / "src" / "data_gen" / "renderers" / "threejs_parallax_renderer.js"
    assert node_script.exists()

    output_dir = tmp_path / "parallax_out"
    metadata_path = output_dir / "metadata.json"

    cmd = [
        "node",
        str(node_script),
        "--input",
        str(motion_path),
        "--output-dir",
        str(output_dir),
        "--views",
        "2",
        "--frames-per-view",
        "2",
        "--sample-id",
        "sample_000000",
        "--actor-id",
        "0",
        "--environment-type",
        "default",
        "--metadata",
        str(metadata_path),
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Node renderer failed with code {result.returncode}, stderr=\n{result.stderr}"

    assert output_dir.exists(), "Output directory was not created"
    pngs = sorted(output_dir.glob("*.png"))
    assert pngs, f"No PNG frames generated, stdout=\n{result.stdout}\nstderr=\n{result.stderr}"

    assert metadata_path.exists(), "Metadata JSON was not written"
