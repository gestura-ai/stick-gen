import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")
sample = data[11]
frame0 = sample['motion'][0].tolist()

print("\nSample 000011 (Source: babel) - Raw V3 Frame 0:")
labels = {
    0: 'Neck', 10: 'Pelvis',
    12: 'L_Shoulder', 20: 'R_Shoulder',
    44: 'L_Hip', 46: 'R_Hip'
}

# Print key torso joints to check for structure
for idx, label in labels.items():
    print(f"{label:12s} [{idx:2d},{idx+1:2d}]: ({frame0[idx]:7.4f}, {frame0[idx+1]:7.4f})")

# Calculate Spine
neck = np.array([frame0[0], frame0[1]])
pelvis = np.array([frame0[10], frame0[11]])
spine_len = np.linalg.norm(neck - pelvis)
print(f"\nSpine Length: {spine_len:.4f}")

# Check distances from Neck to Shoulders
l_sh = np.array([frame0[12], frame0[13]])
dist_neck_lsh = np.linalg.norm(l_sh - neck)
print(f"Neck -> L_Shoulder: {dist_neck_lsh:.4f} (Ratio to Spine: {dist_neck_lsh/spine_len:.2f})")

