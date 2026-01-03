import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")
sample = data[11]['motion'][0].tolist()

print("\n--- Sample 000011 Positions ---")
neck = np.array([sample[0], sample[1]])
head = np.array([sample[2], sample[3]])
pelvis = np.array([sample[10], sample[11]])
chest = np.array([sample[6], sample[7]]) # If used

print(f"Neck:   ({neck[0]:.4f}, {neck[1]:.4f})")
print(f"Head:   ({head[0]:.4f}, {head[1]:.4f})")
print(f"Pelvis: ({pelvis[0]:.4f}, {pelvis[1]:.4f})")

# Check ordering along the Spine Vector
spine_vec = neck - pelvis
spine_len = np.linalg.norm(spine_vec)
norm_spine = spine_vec / spine_len

def project(p, label):
    rel = p - pelvis
    dist = np.dot(rel, norm_spine)
    print(f"{label} projection on Spine: {dist:.4f} (0=Pelvis, {spine_len:.4f}=Neck)")

project(pelvis, "Pelvis")
project(head, "Head")
project(neck, "Neck")
