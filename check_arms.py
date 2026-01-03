import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")
sample = data[0]
frame0 = sample['motion'][0].tolist()

print("Sample 000000 - Raw Frame 0 (Legacy 20D):")
print(f"Neck:    [{0:2d},{1:2d}] = ({frame0[0]:7.4f}, {frame0[1]:7.4f})")
print(f"Pelvis:  [{2:2d},{3:2d}] = ({frame0[2]:7.4f}, {frame0[3]:7.4f})")
print(f"L_Wrist: [{14:2d},{15:2d}] = ({frame0[14]:7.4f}, {frame0[15]:7.4f})")
print(f"R_Wrist: [{18:2d},{19:2d}] = ({frame0[18]:7.4f}, {frame0[19]:7.4f})")

# After Y-flip (neck.y < pelvis.y, so flip)
scaleY = -1.0
neck = np.array([frame0[0], frame0[1] * scaleY])
pelvis = np.array([frame0[2], frame0[3] * scaleY])
l_wrist = np.array([frame0[14], frame0[15] * scaleY])
r_wrist = np.array([frame0[18], frame0[19] * scaleY])

print(f"\nAfter Y-flip:")
print(f"Neck:    ({neck[0]:7.4f}, {neck[1]:7.4f})")
print(f"Pelvis:  ({pelvis[0]:7.4f}, {pelvis[1]:7.4f})")
print(f"L_Wrist: ({l_wrist[0]:7.4f}, {l_wrist[1]:7.4f})")
print(f"R_Wrist: ({r_wrist[0]:7.4f}, {r_wrist[1]:7.4f})")

print(f"\nDistances from Pelvis:")
print(f"L_Wrist: {np.linalg.norm(l_wrist - pelvis):.4f}")
print(f"R_Wrist: {np.linalg.norm(r_wrist - pelvis):.4f}")

print(f"\nTorso Length: {np.linalg.norm(neck - pelvis):.4f}")
