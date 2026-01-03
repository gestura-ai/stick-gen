import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")
# Check first few legacy samples
count = 0
for i in range(100):
    sample = data[i]
    frame0 = sample['motion'][0].tolist()
    if len(frame0) != 20: continue
    
    count += 1
    
    # Simulate current Legacy Extraction Logic LOCALLY to identify asymmetry
    neck = np.array([frame0[0], frame0[1]])
    pelvis = np.array([frame0[2], frame0[3]])
    
    # Raw Wrists
    l_wrist_raw = np.array([frame0[14], frame0[15]])
    r_wrist_raw = np.array([frame0[18], frame0[19]])
    
    print(f"\n--- Legacy Sample Index {i} ---")
    print(f"Neck: ({neck[0]:.4f}, {neck[1]:.4f})")
    print(f"Pelvis: ({pelvis[0]:.4f}, {pelvis[1]:.4f})")
    
    # Spine Calc
    spine = neck - pelvis
    spine_len = np.linalg.norm(spine)
    print(f"Spine Len: {spine_len:.4f}")
    
    # Check Distances of Raw Wrists from Body
    # Note: Logic uses Clamping if > 1.5x Torso etc.
    # But let's look at RAW asymmetry first.
    
    l_dist = np.linalg.norm(l_wrist_raw - pelvis)
    r_dist = np.linalg.norm(r_wrist_raw - pelvis)
    
    print(f"L_Wrist Dist: {l_dist:.4f} ({l_dist/spine_len:.2f}x Torso)")
    print(f"R_Wrist Dist: {r_dist:.4f} ({r_dist/spine_len:.2f}x Torso)")
    
    ratio = l_dist / r_dist if r_dist > 0 else 999
    if ratio < 1.0: ratio = 1.0 / ratio
    
    if ratio > 1.5:
        print("!! ASYMMETRY DETECTED !!")

    if count >= 5: break
