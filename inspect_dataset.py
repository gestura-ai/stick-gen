
import torch
import os

path = "data/processed/curated/pretrain_data.pt"
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        data = torch.load(path)
        print(f"Loaded {path}")
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            # Check first few samples for 'motion' key
            for i in range(min(5, len(data))):
                sample = data[i]
                motion = sample.get("motion")
                if motion is None:
                    print(f"Sample {i}: motion is None")
                else:
                    print(f"Sample {i}: motion shape {motion.shape}")
        else:
            print(f"Data is not a list, it is {type(data)}")
    except Exception as e:
        print(f"Error loading: {e}")
