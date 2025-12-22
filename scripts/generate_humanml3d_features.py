#!/usr/bin/env python3
"""
Generate HumanML3D Preprocessed Motion Features

This script combines the three HumanML3D preprocessing notebooks into one executable:
1. raw_pose_processing.ipynb - Extract poses from AMASS
2. motion_representation.ipynb - Generate motion feature vectors
3. cal_mean_variance.ipynb - Calculate normalization statistics

Usage:
    python scripts/generate_humanml3d_features.py \
        --amass-root data/amass \
        --humanml3d-root data/HumanML3D \
        --smpl-root data/smpl_models

Author: Gestura AI
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add HumanML3D to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent


def setup_humanml3d_path(humanml3d_root: str):
    """Add HumanML3D directory to Python path."""
    humanml3d_path = Path(humanml3d_root).absolute()
    if str(humanml3d_path) not in sys.path:
        sys.path.insert(0, str(humanml3d_path))


def step1_extract_amass_poses(
    amass_root: str,
    humanml3d_root: str,
    smpl_root: str,
    device: str = "cpu",
    max_files: int = -1,
):
    """
    Step 1: Extract poses from AMASS dataset.

    Converts AMASS .npz files to joint positions using SMPL-H body model.
    Output: pose_data/<dataset>/<file>.npy
    """
    print("\n" + "=" * 60)
    print("Step 1: Extracting poses from AMASS")
    print("=" * 60)

    # Add HumanML3D to path for human_body_prior
    setup_humanml3d_path(humanml3d_root)

    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    
    humanml3d_path = Path(humanml3d_root)
    amass_path = Path(amass_root)
    smpl_path = Path(smpl_root)
    
    # Load body models
    print("Loading SMPL-H body models...")
    male_bm_path = smpl_path / "smplh" / "male" / "model.npz"
    female_bm_path = smpl_path / "smplh" / "female" / "model.npz"
    
    num_betas = 10
    comp_device = torch.device(device)
    
    male_bm = BodyModel(bm_fname=str(male_bm_path), num_betas=num_betas).to(comp_device)
    female_bm = BodyModel(bm_fname=str(female_bm_path), num_betas=num_betas).to(comp_device)
    
    # Coordinate transform (Y-up to Z-up)
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    ex_fps = 20  # Target FPS
    
    # Collect all npz files
    print(f"Scanning AMASS directory: {amass_path}")
    npz_files = list(amass_path.rglob("*.npz"))
    npz_files = [f for f in npz_files if "shape" not in f.name.lower()]  # Skip shape files
    print(f"Found {len(npz_files)} motion files")
    
    if max_files > 0:
        npz_files = npz_files[:max_files]
        print(f"Processing first {max_files} files")
    
    pose_data_dir = humanml3d_path / "pose_data"
    processed = 0
    skipped = 0
    
    for npz_path in tqdm(npz_files, desc="Extracting poses"):
        try:
            bdata = np.load(npz_path, allow_pickle=True)
            
            # Check required keys
            if "mocap_framerate" not in bdata or "trans" not in bdata:
                skipped += 1
                continue
            
            fps = float(bdata["mocap_framerate"])
            if fps < ex_fps:
                skipped += 1
                continue
            
            # Select body model based on gender
            gender = str(bdata.get("gender", "neutral"))
            if isinstance(gender, bytes):
                gender = gender.decode()
            bm = male_bm if gender == "male" else female_bm
            
            # Downsample to target FPS
            down_sample = max(1, int(fps / ex_fps))
            bdata_poses = bdata["poses"][::down_sample]
            bdata_trans = bdata["trans"][::down_sample]
            
            if len(bdata_trans) < 10:  # Skip very short sequences
                skipped += 1
                continue
            
            # Prepare body parameters
            body_parms = {
                "root_orient": torch.Tensor(bdata_poses[:, :3]).to(comp_device),
                "pose_body": torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
                "trans": torch.Tensor(bdata_trans).to(comp_device),
            }
            
            # Add betas if available
            if "betas" in bdata:
                betas = bdata["betas"][:num_betas]
                body_parms["betas"] = torch.Tensor(
                    np.repeat(betas[np.newaxis], len(bdata_trans), axis=0)
                ).to(comp_device)
            
            # Forward kinematics
            with torch.no_grad():
                body = bm(**body_parms)
            
            pose_seq = body.Jtr.detach().cpu().numpy()
            pose_seq = np.dot(pose_seq, trans_matrix)
            
            # Save to pose_data directory
            rel_path = npz_path.relative_to(amass_path)
            save_path = pose_data_dir / rel_path.with_suffix(".npy")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, pose_seq)
            processed += 1
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"Processed: {processed}, Skipped: {skipped}")
    return processed


def step2_segment_and_mirror(humanml3d_root: str):
    """
    Step 2: Segment and mirror motions based on index.csv.
    
    Uses index.csv to extract specific clips and creates mirrored versions.
    Output: joints/<clip_id>.npy and joints/M<clip_id>.npy
    """
    print("\n" + "=" * 60)
    print("Step 2: Segmenting and mirroring motions")
    print("=" * 60)
    
    import pandas as pd
    
    humanml3d_path = Path(humanml3d_root)
    index_path = humanml3d_path / "index.csv"
    joints_dir = humanml3d_path / "joints"
    joints_dir.mkdir(exist_ok=True)
    
    index_file = pd.read_csv(index_path)
    total = len(index_file)
    print(f"Processing {total} clips from index.csv")
    
    fps = 20
    processed = 0
    skipped = 0
    
    def swap_left_right(data):
        """Mirror motion by swapping left/right joints."""
        assert len(data.shape) == 3 and data.shape[-1] == 3
        data = data.copy()
        data[..., 0] *= -1
        right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
        left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
        tmp = data[:, right_chain]
        data[:, right_chain] = data[:, left_chain]
        data[:, left_chain] = tmp
        return data
    
    for i in tqdm(range(total), desc="Segmenting clips"):
        source_path = index_file.loc[i]["source_path"]
        new_name = index_file.loc[i]["new_name"]
        
        # Convert relative path to absolute
        full_path = humanml3d_path / source_path.lstrip("./")
        
        if not full_path.exists():
            skipped += 1
            continue
        
        try:
            data = np.load(full_path)
            start_frame = int(index_file.loc[i]["start_frame"])
            end_frame = int(index_file.loc[i]["end_frame"])
            
            # Apply dataset-specific offsets
            if "humanact12" not in source_path:
                if "Eyes_Japan_Dataset" in source_path:
                    data = data[3 * fps:]
                if "MPI_HDM05" in source_path:
                    data = data[3 * fps:]
                if "TotalCapture" in source_path:
                    data = data[1 * fps:]
                if "MPI_Limits" in source_path:
                    data = data[1 * fps:]
                if "Transitions_mocap" in source_path:
                    data = data[int(0.5 * fps):]
                
                if end_frame == -1:
                    data = data[start_frame:]
                else:
                    data = data[start_frame:end_frame]
                data[..., 0] *= -1  # Flip X axis
            
            if len(data) < 10:
                skipped += 1
                continue
            
            # Save original and mirrored versions
            data_m = swap_left_right(data)
            np.save(joints_dir / new_name, data)
            np.save(joints_dir / f"M{new_name}", data_m)
            processed += 1
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"Processed: {processed}, Skipped: {skipped}")
    return processed


def step3_generate_motion_features(humanml3d_root: str):
    """
    Step 3: Generate motion representation features.

    Converts joint positions to HumanML3D feature vectors.
    Output: HumanML3D/new_joints/*.npy and HumanML3D/new_joint_vecs/*.npy
    """
    print("\n" + "=" * 60)
    print("Step 3: Generating motion feature vectors")
    print("=" * 60)

    setup_humanml3d_path(humanml3d_root)

    from common.skeleton import Skeleton
    from common.quaternion import (
        qbetween_np, qfix, qinv, qinv_np, qmul_np, qrot, qrot_np,
        quaternion_to_cont6d_np,
    )
    from paramUtil import t2m_kinematic_chain, t2m_raw_offsets

    humanml3d_path = Path(humanml3d_root)
    joints_dir = humanml3d_path / "joints"
    save_dir1 = humanml3d_path / "HumanML3D" / "new_joints"
    save_dir2 = humanml3d_path / "HumanML3D" / "new_joint_vecs"

    save_dir1.mkdir(parents=True, exist_ok=True)
    save_dir2.mkdir(parents=True, exist_ok=True)

    # Configuration
    example_id = "000021"
    l_idx1, l_idx2 = 5, 8  # Lower legs
    fid_r, fid_l = [8, 11], [7, 10]  # Feet
    face_joint_indx = [2, 1, 17, 16]  # Face direction joints
    joints_num = 22

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    def load_joints_22(path):
        """Load joint file and extract only the first 22 joints (HumanML3D format)."""
        data = np.load(path)
        if data.ndim == 1:
            # Flat array - try to reshape
            if data.size % (22 * 3) == 0:
                return data.reshape(-1, 22, 3)
            elif data.size % (24 * 3) == 0:
                return data.reshape(-1, 24, 3)[:, :22, :]
            elif data.size % (52 * 3) == 0:
                return data.reshape(-1, 52, 3)[:, :22, :]
            else:
                raise ValueError(f"Cannot reshape array of size {data.size}")
        elif data.ndim == 2:
            # (frames, joints*3) format
            if data.shape[1] == 22 * 3:
                return data.reshape(-1, 22, 3)
            elif data.shape[1] >= 24 * 3:
                return data.reshape(-1, data.shape[1] // 3, 3)[:, :22, :]
            else:
                raise ValueError(f"Unexpected shape: {data.shape}")
        elif data.ndim == 3:
            # Already (frames, joints, 3) format
            if data.shape[1] >= 22:
                return data[:, :22, :]
            else:
                raise ValueError(f"Not enough joints: {data.shape[1]}")
        else:
            raise ValueError(f"Unexpected ndim: {data.ndim}")

    # Get target skeleton offsets
    example_path = joints_dir / f"{example_id}.npy"
    if not example_path.exists():
        print(f"ERROR: Example file {example_path} not found!")
        return 0

    example_data = load_joints_22(example_path)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def uniform_skeleton(positions, target_offset):
        src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = target_offset.numpy()

        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
        scale_rt = tgt_leg_len / src_leg_len

        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        src_skel.set_offset(target_offset)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints

    def process_file(positions, feet_thre):
        positions = uniform_skeleton(positions, tgt_offsets)
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height

        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz

        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]])
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
        positions = qrot_np(root_quat_init, positions)

        global_positions = positions.copy()

        def foot_detect(pos, thres):
            velfactor = np.array([thres, thres])
            feet_l_x = (pos[1:, fid_l, 0] - pos[:-1, fid_l, 0]) ** 2
            feet_l_y = (pos[1:, fid_l, 1] - pos[:-1, fid_l, 1]) ** 2
            feet_l_z = (pos[1:, fid_l, 2] - pos[:-1, fid_l, 2]) ** 2
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

            feet_r_x = (pos[1:, fid_r, 0] - pos[:-1, fid_r, 0]) ** 2
            feet_r_y = (pos[1:, fid_r, 1] - pos[:-1, fid_r, 1]) ** 2
            feet_r_z = (pos[1:, fid_r, 2] - pos[:-1, fid_r, 2]) ** 2
            feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
            return feet_l, feet_r

        feet_l, feet_r = foot_detect(positions, feet_thre)

        def get_cont6d_params(pos):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            quat_params = skel.inverse_kinematics_np(pos, face_joint_indx, smooth_forward=True)
            cont_6d_params = quaternion_to_cont6d_np(quat_params)
            r_rot = quat_params[:, 0].copy()
            velocity = (pos[1:, 0] - pos[:-1, 0]).copy()
            velocity = qrot_np(r_rot[1:], velocity)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            return cont_6d_params, r_velocity, velocity, r_rot

        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)

        def get_rifke(pos):
            pos = pos.copy()
            pos[..., 0] -= pos[:, 0:1, 0]
            pos[..., 2] -= pos[:, 0:1, 2]
            pos = qrot_np(np.repeat(r_rot[:, None], pos.shape[1], axis=1), pos)
            return pos

        positions = get_rifke(positions)

        root_y = positions[:, 0, 1:2]
        r_velocity_y = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)

        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
        ric_data = positions[:, 1:].reshape(len(positions), -1)

        local_vel = qrot_np(
            np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
            global_positions[1:] - global_positions[:-1]
        )
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)
        return data, global_positions, positions, l_velocity

    source_list = list(joints_dir.glob("*.npy"))
    print(f"Processing {len(source_list)} joint files")

    processed = 0
    skipped = 0
    frame_num = 0

    for source_file in tqdm(source_list, desc="Generating features"):
        try:
            source_data = load_joints_22(source_file)
            data, _, _, _ = process_file(source_data, 0.002)
            frame_num += data.shape[0]

            np.save(save_dir2 / source_file.name, data)
            processed += 1
        except Exception as e:
            skipped += 1

    print(f"Processed: {processed}, Skipped: {skipped}")
    print(f"Total frames: {frame_num}, Duration: {frame_num / 20 / 60:.1f} minutes")
    return processed


def step4_calculate_mean_variance(humanml3d_root: str):
    """
    Step 4: Calculate mean and variance for normalization.

    Output: HumanML3D/Mean.npy and HumanML3D/Std.npy
    """
    print("\n" + "=" * 60)
    print("Step 4: Calculating mean and variance")
    print("=" * 60)

    humanml3d_path = Path(humanml3d_root)
    data_dir = humanml3d_path / "HumanML3D" / "new_joint_vecs"

    file_list = list(data_dir.glob("*.npy"))
    print(f"Found {len(file_list)} feature files")

    if not file_list:
        print("ERROR: No feature files found!")
        return False

    # Collect all data
    all_data = []
    for file_path in tqdm(file_list, desc="Loading features"):
        try:
            data = np.load(file_path)
            all_data.append(data)
        except Exception:
            continue

    all_data = np.concatenate(all_data, axis=0)
    print(f"Total frames: {len(all_data)}")

    # Calculate statistics
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)

    # Avoid division by zero
    std = np.clip(std, 1e-8, None)

    # Save
    save_dir = humanml3d_path / "HumanML3D"
    np.save(save_dir / "Mean.npy", mean)
    np.save(save_dir / "Std.npy", std)

    print(f"Saved Mean.npy shape: {mean.shape}")
    print(f"Saved Std.npy shape: {std.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate HumanML3D preprocessed motion features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--amass-root",
        type=str,
        default="data/amass",
        help="Root directory of AMASS dataset",
    )
    parser.add_argument(
        "--humanml3d-root",
        type=str,
        default="data/HumanML3D",
        help="Root directory of HumanML3D repository",
    )
    parser.add_argument(
        "--smpl-root",
        type=str,
        default="data/smpl_models",
        help="Root directory of SMPL body models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for body model inference",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Maximum AMASS files to process (-1 for all)",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip step 1 (AMASS pose extraction)",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip step 2 (segmentation and mirroring)",
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Skip step 3 (motion feature generation)",
    )
    parser.add_argument(
        "--skip-step4",
        action="store_true",
        help="Skip step 4 (mean/variance calculation)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HumanML3D Feature Generation Pipeline")
    print("=" * 60)
    print(f"AMASS root: {args.amass_root}")
    print(f"HumanML3D root: {args.humanml3d_root}")
    print(f"SMPL root: {args.smpl_root}")
    print(f"Device: {args.device}")

    # Step 1: Extract poses from AMASS
    if not args.skip_step1:
        step1_extract_amass_poses(
            args.amass_root,
            args.humanml3d_root,
            args.smpl_root,
            device=args.device,
            max_files=args.max_files,
        )
    else:
        print("\nSkipping Step 1 (AMASS pose extraction)")

    # Step 2: Segment and mirror motions
    if not args.skip_step2:
        step2_segment_and_mirror(args.humanml3d_root)
    else:
        print("\nSkipping Step 2 (segmentation and mirroring)")

    # Step 3: Generate motion features
    if not args.skip_step3:
        step3_generate_motion_features(args.humanml3d_root)
    else:
        print("\nSkipping Step 3 (motion feature generation)")

    # Step 4: Calculate mean and variance
    if not args.skip_step4:
        step4_calculate_mean_variance(args.humanml3d_root)
    else:
        print("\nSkipping Step 4 (mean/variance calculation)")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    # Summary
    humanml3d_path = Path(args.humanml3d_root)
    new_joint_vecs = humanml3d_path / "HumanML3D" / "new_joint_vecs"
    if new_joint_vecs.exists():
        count = len(list(new_joint_vecs.glob("*.npy")))
        print(f"Generated {count} motion feature files")


if __name__ == "__main__":
    main()
