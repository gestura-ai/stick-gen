import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.parallax_dataset import MultimodalParallaxDataset


def _make_motion_dataset(tmp_path: Path) -> Path:
    T, A, D = 3, 1, 20
    motion = torch.arange(T * A * D, dtype=torch.float32).view(T, A, D)
    actions = torch.zeros(T, A, dtype=torch.long)
    sample = {"motion": motion, "actions": actions, "description": "test sample"}
    path = tmp_path / "train.pt"
    torch.save([sample], path)
    return path


def _make_parallax_root(tmp_path: Path, with_view_id: bool = True) -> Path:
    root = tmp_path / "parallax"
    actor_dir = root / "sample_000000" / "actor_0"
    actor_dir.mkdir(parents=True, exist_ok=True)
    # Tiny 4x4 PNG so both backends can read it
    img_path = actor_dir / "view_00000.png"
    img = Image.fromarray(np.ones((4, 4, 3), dtype="uint8") * 255)
    img.save(img_path)

    frame_entry = {
        "file": "view_00000.png",
        "step_index": 0,
        "motion_frame_index": 1,
        "camera": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "target": {"x": 0.0, "y": 1.0, "z": 0.0},
            "fov": 45.0,
        },
    }
    if with_view_id:
        frame_entry["view_id"] = 7
    else:
        frame_entry["view_index"] = 7

    meta = {
        "sample_id": "sample_000000",
        "actor_id": "0",
        "views": 1,
        "frames_per_view": 1,
        "motion_total_frames": 3,
        "fps": 25,
        "frames": [frame_entry],
    }
    with open(actor_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return root


def test_dataset_init_and_getitem_pil(tmp_path: Path) -> None:
    motion_path = _make_motion_dataset(tmp_path)
    parallax_root = _make_parallax_root(tmp_path, with_view_id=True)

    ds = MultimodalParallaxDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        image_size=(16, 16),
        image_backend="pil",
    )

    assert len(ds) == 1
    image, motion_frame, camera_pose, text, action = ds[0]

    assert image.shape == (3, 16, 16)
    assert torch.is_floating_point(image)
    assert motion_frame.shape == (20,)
    assert camera_pose.shape == (7,)
    assert isinstance(text, str) and text == "test sample"
    assert isinstance(action, torch.Tensor) and action.dtype == torch.long
    assert ds.index[0]["view_id"] == 7


def test_dataset_image_backend_torchvision(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    motion_path = _make_motion_dataset(tmp_path)
    parallax_root = _make_parallax_root(tmp_path, with_view_id=True)

    ds = MultimodalParallaxDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        image_size=(8, 8),
        image_backend="torchvision",
    )

    image, _motion_frame, _camera_pose, _text, _action = ds[0]
    assert image.shape == (3, 8, 8)


def test_select_action_label_variants() -> None:
    ds = object.__new__(MultimodalParallaxDataset)  # bypass __init__

    # Per-frame [T]
    actions_1d = torch.tensor([1, 2, 3])
    assert ds._select_action_label(actions_1d, 1, 0).item() == 2

    # Per-frame [T, A]
    actions_2d = torch.tensor([[0, 1], [2, 3]])
    assert ds._select_action_label(actions_2d, 1, 1).item() == 3

    # Scalar / single-element tensors
    assert ds._select_action_label(torch.tensor(5), 0, 0).item() == 5
    assert ds._select_action_label(torch.tensor([7]), 2, 0).item() == 7

    # Non-tensor and None
    assert ds._select_action_label(4, 0, 0).item() == 4
    assert ds._select_action_label([9], 0, 0).item() == 9
    assert ds._select_action_label(None, 0, 0) is None


def test_view_id_fallback_to_view_index(tmp_path: Path) -> None:
    motion_path = _make_motion_dataset(tmp_path)
    parallax_root = _make_parallax_root(tmp_path, with_view_id=False)

    ds = MultimodalParallaxDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
    )

    assert ds.index[0]["view_id"] == 7
