import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")
sample_good = data[1]['motion'][0].tolist()  # Sample 000001
sample_bad = data[11]['motion'][0].tolist()  # Sample 000011

def analyze_sample(frame, label):
    print(f"\n--- {label} ---")
    neck = np.array([frame[0], frame[1]])
    pelvis = np.array([frame[10], frame[11]])
    l_shoulder = np.array([frame[12], frame[13]])
    
    # Spine Vector
    spine = neck - pelvis
    spine_len = np.linalg.norm(spine)
    print(f"Spine Vector: ({spine[0]:.4f}, {spine[1]:.4f}) | Len: {spine_len:.4f}")
    
    # Shoulder Relative Position
    shoulder_rel = l_shoulder - neck
    print(f"Neck->L_Shoulder: ({shoulder_rel[0]:.4f}, {shoulder_rel[1]:.4f})")
    
    # Projection of Shoulder onto Spine (Ideally close to 0 or top of spine)
    # Projection onto Perpendicular (Width)
    spine_norm = spine / spine_len
    perp = np.array([-spine_norm[1], spine_norm[0]]) # Left perpendicular
    
    width = np.dot(shoulder_rel, perp)
    height = np.dot(shoulder_rel, spine_norm)
    
    print(f"Shoulder Offset: Width={width:.4f}, Height={height:.4f} (Height relative to Neck)")
    print(f"Ratio Width/Spine: {width/spine_len:.4f}")

analyze_sample(sample_good, "Sample 000001 (GOOD)")
analyze_sample(sample_bad, "Sample 000011 (BAD)")
