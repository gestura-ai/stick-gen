import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.parallax_dataset import MultimodalParallaxDataset, MultimodalParallaxSequenceDataset


def _make_motion_dataset(tmp_path: Path) -> Path:
    T, A, D = 3, 1, 48
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
    image, motion_frame, camera_pose, text, action, env_type = ds[0]

    assert image.shape == (3, 16, 16)
    assert torch.is_floating_point(image)
    assert motion_frame.shape == (48,)
    assert camera_pose.shape == (7,)
    assert isinstance(text, str) and text == "test sample"
    assert isinstance(action, torch.Tensor) and action.dtype == torch.long
    assert ds.index[0]["view_id"] == 7
    assert isinstance(env_type, str)  # environment_type is now returned


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

    image, _motion_frame, _camera_pose, _text, _action, _env_type = ds[0]
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


# ============================================================================
# MultimodalParallaxSequenceDataset Tests
# ============================================================================


def _make_motion_dataset_long(tmp_path: Path, num_frames: int = 60) -> Path:
    """Create a motion dataset with T frames for sequence testing."""
    T, A, D = num_frames, 1, 48
    motion = torch.arange(T * A * D, dtype=torch.float32).view(T, A, D)
    actions = torch.zeros(T, A, dtype=torch.long)
    sample = {"motion": motion, "actions": actions, "description": "test sequence sample"}
    path = tmp_path / "train.pt"
    torch.save([sample], path)
    return path


def _make_parallax_root_multi_frame(
    tmp_path: Path, num_frames: int = 10, view_id: int = 0
) -> Path:
    """Create parallax data with multiple frames for sequence testing."""
    root = tmp_path / "parallax"
    actor_dir = root / "sample_000000" / "actor_0"
    actor_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(num_frames):
        img_path = actor_dir / f"frame_{i:05d}.png"
        img = Image.fromarray(np.ones((4, 4, 3), dtype="uint8") * (i * 10 + 50))
        img.save(img_path)

        frames.append({
            "file": f"frame_{i:05d}.png",
            "step_index": i,
            "motion_frame_index": i + 5,  # Offset to test extraction
            "view_id": view_id,
            "camera": {
                "position": {"x": 1.0 + i * 0.1, "y": 2.0, "z": 3.0},
                "target": {"x": 0.0, "y": 1.0, "z": 0.0},
                "fov": 45.0,
            },
        })

    meta = {
        "sample_id": "sample_000000",
        "actor_id": "0",
        "views": 1,
        "frames_per_view": num_frames,
        "motion_total_frames": 60,
        "fps": 25,
        "frames": frames,
    }
    with open(actor_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return root


def test_sequence_dataset_init_and_getitem(tmp_path: Path) -> None:
    """Test MultimodalParallaxSequenceDataset initialization and getitem."""
    motion_path = _make_motion_dataset_long(tmp_path, num_frames=60)
    parallax_root = _make_parallax_root_multi_frame(tmp_path, num_frames=10, view_id=0)

    ds = MultimodalParallaxSequenceDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        sequence_length=4,
        stride=2,
        image_size=(16, 16),
        image_backend="pil",
        conditioning_mode="first_frame",
    )

    # With 10 frames, seq_len=4, stride=2: (10-4)/2 + 1 = 4 sequences
    assert len(ds) == 4

    images, motion_seq, camera_seq, text, actions, env_type = ds[0]

    # Check shapes
    assert images.shape == (1, 3, 16, 16)  # first_frame mode: [1, 3, H, W]
    assert motion_seq.shape == (4, 48)  # [seq_len, D]
    assert camera_seq.shape == (4, 7)  # [seq_len, 7]
    assert isinstance(text, str)
    assert actions.shape == (4,)  # [seq_len]
    assert isinstance(env_type, str)


def test_sequence_dataset_contiguous_motion_indices(tmp_path: Path) -> None:
    """Verify that motion_frame_index values are contiguous within a sequence."""
    motion_path = _make_motion_dataset_long(tmp_path, num_frames=60)
    parallax_root = _make_parallax_root_multi_frame(tmp_path, num_frames=10, view_id=0)

    ds = MultimodalParallaxSequenceDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        sequence_length=4,
        stride=1,
        image_size=(8, 8),
        image_backend="pil",
    )

    # Get the first sequence and verify motion frame indices
    seq_meta = ds.sequence_index[0]
    frame_indices = [f["motion_frame_index"] for f in seq_meta["frames"]]

    # Should be contiguous: [5, 6, 7, 8]
    expected = list(range(5, 5 + 4))
    assert frame_indices == expected, f"Expected {expected}, got {frame_indices}"


def test_sequence_dataset_all_frames_mode(tmp_path: Path) -> None:
    """Test all_frames conditioning mode returns all sequence images."""
    motion_path = _make_motion_dataset_long(tmp_path, num_frames=60)
    parallax_root = _make_parallax_root_multi_frame(tmp_path, num_frames=10, view_id=0)

    ds = MultimodalParallaxSequenceDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        sequence_length=4,
        stride=2,
        image_size=(8, 8),
        image_backend="pil",
        conditioning_mode="all_frames",
    )

    images, motion_seq, camera_seq, text, actions, env_type = ds[0]

    # all_frames mode: [seq_len, 3, H, W]
    assert images.shape == (4, 3, 8, 8)
    assert motion_seq.shape == (4, 48)


def test_sequence_dataset_skips_short_views(tmp_path: Path) -> None:
    """Test that views with fewer frames than min_sequence_length are skipped."""
    motion_path = _make_motion_dataset_long(tmp_path, num_frames=60)
    parallax_root = _make_parallax_root_multi_frame(tmp_path, num_frames=3, view_id=0)

    # Requesting seq_len=4 but only 3 frames available - dataset should be empty
    ds = MultimodalParallaxSequenceDataset(
        parallax_root=str(parallax_root),
        motion_data_path=str(motion_path),
        sequence_length=4,
        stride=1,
    )
    # Views with insufficient frames should be skipped, resulting in empty dataset
    assert len(ds) == 0
