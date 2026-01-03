import torch
import numpy as np

data = torch.load("data/processed/curated/pretrain_data.pt")

# Check multiple samples including good and bad ones
samples_to_check = [0, 1, 11]

for idx in samples_to_check:
    if idx >= len(data): continue
    sample = data[idx]
    source = sample.get("source", "unknown")
    motion = sample.get("motion")
    
    print(f"\n{'='*60}")
    print(f"Sample {idx:06d} - Source: {source} - Shape: {motion.shape}")
    print('='*60)
    
    if motion is not None and len(motion) > 0:
        frame0 = motion[0] if hasattr(motion[0], 'tolist') else motion[0]
        frame0 = frame0.tolist() if hasattr(frame0, 'tolist') else list(frame0)
        
        is_legacy = len(frame0) == 20
        
        if is_legacy:
            print("LEGACY 20D FORMAT")
            print("  Neck:    [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(0, 1, frame0[0], frame0[1]))
            print("  Pelvis:  [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(2, 3, frame0[2], frame0[3]))
            print("  L_Knee?: [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(4, 5, frame0[4], frame0[5]))
            print("  L_Ankle: [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(6, 7, frame0[6], frame0[7]))
            print("  R_Knee?: [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(8, 9, frame0[8], frame0[9]))
            print("  R_Ankle: [{:2d},{:2d}] = ({:7.4f}, {:7.4f})".format(10, 11, frame0[10], frame0[11]))
            
            # Check for degeneracy
            l_knee = np.array([frame0[4], frame0[5]])
            r_knee = np.array([frame0[8], frame0[9]])
            knee_dist = np.linalg.norm(l_knee - r_knee)
            print(f"\n  L/R Knee Distance: {knee_dist:.4f} {'<-- DEGENERATE!' if knee_dist < 0.01 else ''}")
        else:
            print("V3 48D FORMAT")
            neck = np.array([frame0[0], frame0[1]])
            pelvis = np.array([frame0[10], frame0[11]])
            
            # Check all joints' distances from pelvis
            joint_names = ['Neck', 'Head', 'Chest', 'Pelvis', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                          'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Knee', 'L_Ankle', 'R_Knee', 'R_Ankle', 'L_Hip', 'R_Hip']
            indices = [0, 2, 6, 10, 12, 14, 18, 20, 22, 26, 30, 34, 38, 42, 44, 46]
            
            print(f"  Body Center (Pelvis): ({pelvis[0]:.4f}, {pelvis[1]:.4f})")
            print("\n  Joint Distances from Pelvis:")
            for name, idx in zip(joint_names, indices):
                joint = np.array([frame0[idx], frame0[idx+1]])
                dist = np.linalg.norm(joint - pelvis)
                flag = " <-- OUTLIER!" if dist > 2.0 else ""
                print(f"    {name:12s}: {dist:6.3f}{flag}")

