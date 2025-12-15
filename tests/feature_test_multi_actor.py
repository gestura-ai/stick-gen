import torch
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model.transformer import StickFigureTransformer

def test_multi_actor_forward_pass():
    print("Testing Multi-Actor Forward Pass...")
    
    # Init small model
    model = StickFigureTransformer(
        input_dim=20, 
        d_model=64, 
        nhead=4, 
        num_layers=2
    )
    model.eval()
    
    # Dummy data
    bs = 2
    seq_len = 50
    motion = torch.randn(seq_len, bs, 20)
    embedding = torch.randn(bs, 1024)
    partner = torch.randn(seq_len, bs, 20)  # Partner motion
    
    print(f"Input shapes: Motion {motion.shape}, Partner {partner.shape}")
    
    # Forward pass WITH partner
    try:
        output = model(
            motion=motion,
            text_embedding=embedding,
            partner_motion=partner
        )
        print("✅ Forward pass with partner_motion successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise e

    # Forward pass WITHOUT partner (backward compatibility)
    try:
        output_solo = model(
            motion=motion,
            text_embedding=embedding
        )
        print("✅ Forward pass without partner_motion successful!")
    except Exception as e:
        print(f"❌ Solo forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_multi_actor_forward_pass()
