import torch
import os

try:
    data_path = "data/processed/curated/pretrain_data.pt"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        exit(1)
        
    data = torch.load(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")
    
    indices_to_check = [0, 24, 10, 41]
    
    for idx in indices_to_check:
        if idx >= len(data):
            print(f"Sample {idx} out of range")
            continue
            
        sample = data[idx]
        source = sample.get("source", "unknown")
        meta = sample.get("meta", {})
        original_path = meta.get("source_path", "unknown")
        
        motion = sample.get("motion")
        shape = motion.shape if motion is not None else "None"
        
        # Check first frame values
        first_frame = motion[0] if motion is not None and len(motion) > 0 else None
        
        print(f"--- Sample {idx:06d} ---")
        print(f"  Source: {source}")
        print(f"  Original Path: {original_path}")
        print(f"  Shape: {shape}")
        if first_frame is not None:
             print(f"  First Frame Stats: min={first_frame.min():.3f}, max={first_frame.max():.3f}, mean={first_frame.mean():.3f}")
             print(f"  First Frame Data (partial): {first_frame[:10]}")

except Exception as e:
    print(f"Error: {e}")
