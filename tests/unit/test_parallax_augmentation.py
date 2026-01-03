import json
import sys
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_gen.parallax_augmentation import (  # noqa: E402
    _export_actor_motion_to_file,
    generate_parallax_for_dataset,
)
from src.data_gen.schema import ACTION_TO_IDX, ActionType  # noqa: E402


def test_export_actor_motion_to_file_writes_expected_motion(tmp_path: Path) -> None:
    """Exporter should write legacy 20D renderer motion correctly.

    This test intentionally uses the v1 20-dimensional schema to validate the
    JSON format consumed by the Three.js front-end. Canonical training uses
    the v3 48D schema; this is a backwards-compatibility check.
    """

    T, A, D = 4, 1, 20
    motion = torch.zeros(T, A, D, dtype=torch.float32)
    walk_idx = ACTION_TO_IDX[ActionType.WALK]
    actions = torch.full((T, A), walk_idx, dtype=torch.long)
    sample = {"motion": motion, "actions": actions, "description": "sample desc"}

    out_path = tmp_path / "sample_000000_actor0.motion"
    _export_actor_motion_to_file(sample, actor_idx=0, out_path=str(out_path), fps=25)

    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["meta"]["description"] == "sample desc"
    assert data["meta"]["total_frames"] == T
    assert data["meta"]["fps"] == 25
    assert len(data["motion"]) == T * D
    # Actions should be clip-level names corresponding to WALK
    assert data["actions"] == [ActionType.WALK.value] * T


@mock.patch("src.data_gen.parallax_augmentation._has_node_runtime", return_value=True)
@mock.patch("src.data_gen.parallax_augmentation.subprocess.run")
def test_generate_parallax_for_dataset_passes_expected_args(
    mock_run, _mock_has_node, tmp_path: Path
) -> None:
    """Node invocation should receive expected arguments (legacy 20D input).

    This test uses the v1 20-dimensional renderer format and only verifies
    that we call the Node.js script with the correct CLI arguments. It does
    not render real images.
    """

    # One sample, one actor in legacy 20D renderer format. Use non-zero motion
    # so the parallax pipeline does not treat it as corrupted/all-zero data.
    T, A, D = 3, 1, 20
    motion = torch.ones(T, A, D, dtype=torch.float32)
    actions = torch.zeros(T, A, dtype=torch.long)
    sample = {"motion": motion, "actions": actions, "description": "desc"}
    dataset_path = tmp_path / "dataset.pt"
    torch.save([sample], dataset_path)

    node_script = tmp_path / "threejs_parallax_renderer.js"
    node_script.write_text("// stub renderer", encoding="utf-8")

    output_dir = tmp_path / "out"
    generate_parallax_for_dataset(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        views_per_motion=123,
        node_script=str(node_script),
        max_samples=None,
        fps=25,
        frames_per_view=4,
    )

    # One call: 1 sample * 1 actor
    assert mock_run.call_count == 1
    cmd = mock_run.call_args[0][0]

    assert cmd[0] == "node"
    assert str(node_script) in cmd
    assert cmd[cmd.index("--views") + 1] == "123"
    assert cmd[cmd.index("--frames-per-view") + 1] == "4"
    assert cmd[cmd.index("--sample-id") + 1] == "sample_000000"
    assert cmd[cmd.index("--actor-id") + 1] == "0"
