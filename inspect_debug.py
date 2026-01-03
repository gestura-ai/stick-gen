import torch
import os

data_path = "data/processed/curated/pretrain_data.pt"
if not os.path.exists(data_path):
    print("File not found")
    exit(1)
    
data = torch.load(data_path)
indices = [0, 11]

for idx in indices:
    if idx >= len(data): continue
    sample = data[idx]
    source = sample.get("source", "unknown")
    motion = sample.get("motion")
    
    print(f"--- Sample {idx:06d} ---")
    print(f"Source: {source}")
    print(f"Shape: {motion.shape}")
    
    if motion is not None and len(motion) > 0:
        frame0 = motion[0].tolist()
        print("Frame 0 values:")
        # Print clearly with index
        for i in range(0, len(frame0), 2):
            val_x = frame0[i]
            val_y = frame0[i+1] if i+1 < len(frame0) else float('nan')
            print(f"  [{i:02d}, {i+1:02d}]: ({val_x:.4f}, {val_y:.4f})")
            
