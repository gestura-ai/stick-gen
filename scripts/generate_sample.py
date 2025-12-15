import torch
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.model.transformer import StickFigureTransformer
from src.data_gen.renderer import Renderer

def generate():
    INPUT_DIM = 20
    D_MODEL = 128
    
    print("Loading Model...")
    model = StickFigureTransformer(input_dim=INPUT_DIM, d_model=D_MODEL, output_dim=INPUT_DIM)
    model.load_state_dict(torch.load("model_checkpoint.pth"))
    model.eval()
    
    # Start with a seed frame (e.g., zeros or random noise)
    # Ideally we'd take a real frame from the dataset, but let's try zeros
    current_seq = torch.zeros(1, 1, INPUT_DIM) # [seq_len=1, batch=1, dim]
    
    generated_frames = []
    
    print("Generating sequence...")
    with torch.no_grad():
        for i in range(50): # Generate 50 frames (2 seconds)
            output = model(current_seq)
            # Take the last predicted frame
            next_frame = output[-1, :, :].unsqueeze(0) # [1, 1, dim]
            
            # Append to sequence for autoregression
            current_seq = torch.cat([current_seq, next_frame], dim=0)
            
            # Store for rendering
            generated_frames.append(next_frame.squeeze().numpy().tolist())
            
    print("Rendering output...")
    renderer = Renderer()
    renderer.render_raw_frames(generated_frames, "generated_output.mp4")
    print("Done! Saved to generated_output.mp4")

if __name__ == "__main__":
    generate()
