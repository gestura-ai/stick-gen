import json
import os
import sys
import unittest
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_gen.joint_utils import V3_SEGMENT_NAMES
from src.inference.exporter import MotionExporter


class TestMotionExporter(unittest.TestCase):
    """Unit tests for :class:`MotionExporter`.

    These tests cover both the legacy 5-segment, 20D schema and the canonical
    v3 12-segment, 48D schema used for exports.
    """

    def test_initialization(self) -> None:
        exporter = MotionExporter(fps=30)
        assert exporter.fps == 30
        # Verify topology is defined (legacy 5-segment layout)
        assert hasattr(exporter, "segment_names")
        assert len(exporter.segment_names) == 5
        assert "torso" in exporter.segment_names

    def test_export_schema_v1_20d(self) -> None:
        exporter = MotionExporter()
        seq_len = 10
        dim = 20
        motion = torch.randn(seq_len, dim)

        json_str = exporter.export_to_json(motion, description="Test Export")
        data = json.loads(json_str)

        # Meta
        assert data["meta"]["total_frames"] == seq_len
        assert data["meta"]["description"] == "Test Export"

        # Skeleton (legacy v1 5-segment schema)
        assert data["skeleton"]["type"] == "stick_figure_5_segment"
        assert data["skeleton"]["input_dim"] == dim
        assert data["skeleton"]["segments"] == [
            "torso",
            "l_leg",
            "r_leg",
            "l_arm",
            "r_arm",
        ]

        # Motion flattened length
        assert len(data["motion"]) == seq_len * dim

        # Camera export
        camera = torch.tensor([[0.0, 0.0, 1.0]] * seq_len, dtype=torch.float32)
        json_with_cam = exporter.export_to_json(motion, camera_data=camera)
        data_cam = json.loads(json_with_cam)
        assert "camera" in data_cam
        assert len(data_cam["camera"]) == seq_len * 3
        assert data_cam["camera"][0] == 0.0  # x
        assert data_cam["camera"][2] == 1.0  # zoom

    def test_export_schema_v3_48d(self) -> None:
        exporter = MotionExporter()
        seq_len = 8
        dim = 48
        motion = torch.randn(seq_len, dim)

        json_str = exporter.export_to_json(motion, description="v3 export")
        data = json.loads(json_str)

        # Meta
        assert data["meta"]["total_frames"] == seq_len
        assert data["meta"]["description"] == "v3 export"

        # Skeleton should describe the canonical v3 12-segment schema
        assert data["skeleton"]["type"] == "stick_figure_12_segment_v3"
        assert data["skeleton"]["input_dim"] == dim
        assert data["skeleton"]["segments"] == list(V3_SEGMENT_NAMES)

        # Motion flattened length should match T * D
        assert len(data["motion"]) == seq_len * dim

    def test_save_file(self) -> None:
        exporter = MotionExporter()
        motion = torch.randn(5, 20)
        json_str = exporter.export_to_json(motion)

        test_path = Path("test_motion.motion")
        try:
            exporter.save(json_str, str(test_path))
            assert test_path.exists()
            with open(test_path, encoding="utf-8") as f:
                content = f.read()
            assert content == json_str
        finally:
            if test_path.exists():
                test_path.unlink()


if __name__ == "__main__":
    unittest.main()
