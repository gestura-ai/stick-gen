
import sys
import unittest
import torch
import logging
from unittest.mock import MagicMock, patch

# Add src to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.physics_layer import DifferentiablePhysicsLoss

class TestBraxPhysics(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.ERROR)
        
    def test_initialization(self):
        """Test that the layer initializes without crashing"""
        layer = DifferentiablePhysicsLoss()
        # Should default to False logic if JAX not found, or True if found
        # We just want to ensure it doesn't raise
        self.assertIsInstance(layer, torch.nn.Module)

    def test_forward_pass_dummy(self):
        """Test forward pass with dummy data"""
        layer = DifferentiablePhysicsLoss()
        
        # Explicitly disable for this test to check fallback logic first
        # (Or strict check if enabled depends on environment)
        # Use dummy tensors
        motion = torch.randn(10, 2, 20)
        physics = torch.randn(10, 2, 6)
        
        loss = layer(motion, physics)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        
        if not layer.enabled:
            self.assertEqual(loss.item(), 0.0)

    @patch('src.model.physics_layer.BRAX_AVAILABLE', False)
    def test_fallback_when_missing(self):
        """Test strict fallback behavior when Brax is reported missing"""
        # Re-import or cheat by patching the class instance logic if possible
        # Since logic is in __init__, we need to patch before init
        
        # Note: If we patch module level var, we might need reload. 
        # Easier to check the .enabled flag manually set.
        layer = DifferentiablePhysicsLoss()
        layer.enabled = False # Force false
        
        motion = torch.randn(5, 1, 20)
        physics = torch.randn(5, 1, 6)
        loss = layer(motion, physics)
        self.assertEqual(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
