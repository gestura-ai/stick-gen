
import sys
import unittest
import torch
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.exporter import MotionExporter

class TestMotionExporter(unittest.TestCase):
    def test_initialization(self):
        exporter = MotionExporter(fps=30)
        self.assertEqual(exporter.fps, 30)
        # Verify topology is defined (fix check)
        self.assertTrue(hasattr(exporter, 'segment_names'))
        self.assertEqual(len(exporter.segment_names), 5)
        self.assertIn("torso", exporter.segment_names)

    def test_export_schema(self):
        exporter = MotionExporter()
        seq_len = 10
        dim = 20
        motion = torch.randn(seq_len, dim)
        
        json_str = exporter.export_to_json(motion, description="Test Export")
        
        # Parse back
        data = json.loads(json_str)
        
        # Check Meta
        self.assertEqual(data['meta']['total_frames'], seq_len)
        self.assertEqual(data['meta']['description'], "Test Export")
        
        # Check Skeleton
        self.assertEqual(data['skeleton']['type'], "stick_figure_5_segment")
        self.assertEqual(data['skeleton']['input_dim'], dim)
        self.assertIn('segments', data['skeleton'])
        self.assertEqual(data['skeleton']['segments'], ["torso", "l_leg", "r_leg", "l_arm", "r_arm"])
        
        # Check Motion Data
        self.assertEqual(len(data['motion']), seq_len * dim)
        
        # Test Camera Export
        camera = torch.tensor([[0.0, 0.0, 1.0]] * seq_len) # Static camera
        json_with_cam = exporter.export_to_json(motion, camera_data=camera)
        data_cam = json.loads(json_with_cam)
        self.assertIn('camera', data_cam)
        self.assertEqual(len(data_cam['camera']), seq_len * 3)
        self.assertEqual(data_cam['camera'][0], 0.0) # x
        self.assertEqual(data_cam['camera'][2], 1.0) # zoom

    def test_save_file(self):
        exporter = MotionExporter()
        motion = torch.randn(5, 20)
        json_str = exporter.export_to_json(motion)
        
        test_path = "test_motion.motion"
        try:
            exporter.save(json_str, test_path)
            self.assertTrue(os.path.exists(test_path))
            with open(test_path, 'r') as f:
                content = f.read()
            self.assertEqual(content, json_str)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

if __name__ == '__main__':
    unittest.main()
