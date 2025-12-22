#!/usr/bin/env python3
"""
Stick-Gen Data Preparation Pipeline
Gestura AI - https://gestura.ai

Prepares training data by:
1. Converting 100STYLE BVH files to stick figure format
2. Generating synthetic data with camera trajectories
3. Merging all data sources
4. Creating text embeddings

Usage:
    python scripts/prepare_data.py --100style-dir data/100Style --output data/train_data_final.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer


def load_100style_data(style_dir: str, target_frames: int = 250) -> list:
    """Load and process 100STYLE dataset.

    Supports two formats:
    1. BVH files (original format) - requires bvh/scipy
    2. Pre-processed txt files (InputTrain.txt, etc.) - custom format
    """
    from src.data_gen.convert_100style import BVH_AVAILABLE, convert_100style

    output_path = "data/100style_processed.pt"

    # Check if already processed
    if os.path.exists(output_path):
        print(f"Loading pre-processed 100STYLE from {output_path}")
        data = torch.load(output_path)
        return data.get("sequences", [])

    # Check for pre-processed txt format (InputTrain.txt, etc.)
    txt_input = os.path.join(style_dir, "InputTrain.txt")
    txt_labels = os.path.join(style_dir, "InputLabels.txt")

    if os.path.exists(txt_input) and os.path.exists(txt_labels):
        print("Loading 100STYLE from pre-processed txt format...")
        return load_100style_txt(style_dir, target_frames)

    # Fall back to BVH processing
    if not BVH_AVAILABLE:
        print("BVH packages not available and no pre-processed txt found")
        return []

    print(f"Processing 100STYLE from {style_dir}...")
    convert_100style(input_dir=style_dir, output_path=output_path, target_fps=25)

    if os.path.exists(output_path):
        data = torch.load(output_path)
        return data.get("sequences", [])

    return []


def load_100style_txt(style_dir: str, target_frames: int = 250) -> list:
    """Load 100STYLE from pre-processed txt format.

    The 100STYLE dataset provides pre-processed motion data in txt files:
    - InputTrain.txt: Trajectory and bone positions
    - InputLabels.txt: Feature labels
    - Tr_Va_Te_Frames.txt: Frame ranges for each sequence
    """

    sequences = []
    input_file = os.path.join(style_dir, "InputTrain.txt")
    frames_file = os.path.join(style_dir, "Tr_Va_Te_Frames.txt")

    if not os.path.exists(input_file):
        print(f"InputTrain.txt not found in {style_dir}")
        return []

    print("Loading InputTrain.txt (this may take a while for large files)...")

    # Load frame ranges
    frame_ranges = []
    if os.path.exists(frames_file):
        with open(frames_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        start, end = int(parts[0]), int(parts[1])
                        frame_ranges.append((start, end))
                    except ValueError:
                        pass

    # Load motion data - sample every Nth sequence to limit data size
    # Full 100STYLE is ~21GB, we'll sample
    try:
        # Use memory mapping for large files
        data = np.loadtxt(
            input_file, dtype=np.float32, max_rows=250 * 1000
        )  # Limit to ~1000 sequences
    except Exception as e:
        print(f"Error loading InputTrain.txt: {e}")
        print("Falling back to partial load...")
        # Try loading just a portion
        with open(input_file) as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 250 * 100:  # Limit to 100 sequences
                    break
                lines.append([float(x) for x in line.strip().split()])
            data = np.array(lines, dtype=np.float32)

    print(f"Loaded {len(data)} frames from InputTrain.txt")

    # Each row has trajectory + bone data
    # We need to extract bone positions (typically columns 48+)
    # and reshape to [frames, 20] stick figure format

    # 100STYLE format: trajectory (48 cols) + bone positions (varies)
    if data.shape[1] > 48:
        # Extract bone positions (skip trajectory data)
        bone_data = data[:, 48:]

        # Split into sequences of target_frames
        num_sequences = len(bone_data) // target_frames
        print(f"Creating {num_sequences} sequences of {target_frames} frames each")

        for i in range(min(num_sequences, 500)):  # Limit to 500 sequences
            start_idx = i * target_frames
            end_idx = start_idx + target_frames
            sequence = bone_data[start_idx:end_idx]

            # Reshape to [frames, 20] - take first 20 bone features
            if sequence.shape[1] >= 20:
                motion = torch.from_numpy(sequence[:, :20].copy()).float()
            else:
                # Pad if needed
                motion_padded = np.zeros((target_frames, 20), dtype=np.float32)
                motion_padded[:, : sequence.shape[1]] = sequence
                motion = torch.from_numpy(motion_padded).float()

            sequences.append(
                {
                    "motion": motion,
                    "style": f"style_{i % 100}",  # Assign style labels
                    "source": "100style_txt",
                    "sequence_idx": i,
                }
            )

    print(f"Created {len(sequences)} sequences from 100STYLE txt")
    return sequences


def load_amass_data(
    amass_dir: str, smpl_dir: str, target_frames: int = 250, max_samples: int = None
) -> list:
    """Load and process AMASS dataset.

    Args:
        amass_dir: Path to AMASS dataset (contains subdirectories like CMU, BMLmovi, etc.)
        smpl_dir: Path to SMPL models directory
        target_frames: Target number of frames per sequence
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        List of processed motion sequences
    """
    from glob import glob

    try:
        from src.data_gen.convert_amass import (
            AMASSConverter,
            generate_description_from_action,
            infer_action_from_filename,
        )
    except ImportError as e:
        print(f"AMASS converter not available: {e}")
        return []

    if not os.path.exists(amass_dir):
        print(f"AMASS directory not found: {amass_dir}")
        return []

    if not os.path.exists(smpl_dir):
        print(f"SMPL models not found: {smpl_dir}")
        print("AMASS conversion requires SMPL models. Skipping AMASS.")
        return []

    # Find all .npz files in AMASS directory
    npz_files = glob(os.path.join(amass_dir, "**/*.npz"), recursive=True)
    # Filter out stagei files (we want stageii)
    npz_files = [f for f in npz_files if "stagei" not in f or "stageii" in f]

    print(f"Found {len(npz_files)} AMASS motion files")

    if max_samples:
        # Sample evenly from the dataset
        step = max(1, len(npz_files) // max_samples)
        npz_files = npz_files[::step][:max_samples]
        print(f"Sampling {len(npz_files)} files")

    sequences = []
    converter = AMASSConverter(smpl_model_path=smpl_dir)

    for npz_path in tqdm(npz_files, desc="Processing AMASS"):
        try:
            motion = converter.convert_sequence(
                npz_path, target_fps=25, target_duration=target_frames / 25.0
            )

            if motion is not None and motion.shape[0] > 0:
                # Infer action from filename
                action = infer_action_from_filename(npz_path)
                description = generate_description_from_action(action)

                sequences.append(
                    {
                        "motion": motion,
                        "action": action,
                        "description": description,
                        "source": "amass",
                        "file": os.path.basename(npz_path),
                    }
                )
        except Exception:
            # Skip problematic files
            continue

    print(f"Processed {len(sequences)} AMASS sequences")
    return sequences


def generate_synthetic_data(
    num_samples: int,
    target_frames: int = 250,
    output_dir: str = None,
    force: bool = False,
) -> list:
    """Generate synthetic training data with camera trajectories.

    Args:
        num_samples: Number of synthetic samples to generate
        target_frames: Target frames per sequence (unused - configured in base.yaml)
        output_dir: Output directory for intermediate files (default: "data")
        force: If True, regenerate data even if it already exists
    """
    # Note: target_frames is configured via configs/base.yaml data_generation.sequence
    # settings, so the parameter here is kept for backwards compatibility only.
    del target_frames  # Parameter kept for backwards compatibility
    from src.data_gen.dataset_generator import generate_dataset

    # Use provided output_dir or default to "data"
    if output_dir is None:
        output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_data.pt")

    # Check if already generated (skip if force=True)
    if os.path.exists(output_path) and not force:
        print(f"Loading pre-generated synthetic data from {output_path}")
        return torch.load(output_path)

    if force and os.path.exists(output_path):
        print(f"Force mode: removing existing {output_path}")
        os.remove(output_path)

    print(f"Generating {num_samples} synthetic samples...")
    generate_dataset(num_samples=num_samples, output_path=output_path, augment=True)

    if os.path.exists(output_path):
        return torch.load(output_path)

    return []


def create_100style_descriptions(style_data: list) -> list:
    """Create text descriptions for 100STYLE motion sequences."""
    style_descriptions = {
        "angry": "A person moving with angry, aggressive body language",
        "happy": "A person moving with happy, joyful energy",
        "sad": "A person moving with sad, dejected posture",
        "neutral": "A person walking with neutral body language",
        "proud": "A person walking with proud, confident posture",
        "tired": "A person moving with tired, exhausted movements",
        "old": "An elderly person walking slowly",
        "young": "A young person moving with energetic steps",
        "drunk": "A person stumbling with unsteady movements",
        "sneaky": "A person moving stealthily and cautiously",
    }

    processed = []
    for item in style_data:
        style = item.get("style", "neutral").lower()
        # Handle numeric style labels from txt format
        if style.startswith("style_"):
            style_num = int(style.split("_")[1]) % len(style_descriptions)
            style_keys = list(style_descriptions.keys())
            style = style_keys[style_num % len(style_keys)]

        description = style_descriptions.get(
            style, f"A person performing {style} movement"
        )

        motion = item.get("motion")
        if motion is None:
            continue

        # Ensure correct shape [frames, 20]
        if len(motion.shape) == 1:
            motion = motion.reshape(-1, 20)

        # Preserve original source (100style or 100style_txt)
        source = item.get("source", "100style")

        processed.append(
            {
                "description": description,
                "motion": motion,
                "source": source,
                "style": style,
            }
        )

    return processed


def add_embeddings(
    data: list, embedder: SentenceTransformer, batch_size: int = 32
) -> list:
    """Add text embeddings to all samples."""
    print(f"Creating embeddings for {len(data)} samples...")

    descriptions = [item["description"] for item in data]

    # Batch encode
    embeddings = []
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Embedding"):
        batch = descriptions[i : i + batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True)
        embeddings.extend(batch_embeddings)

    # Add embeddings to data
    for i, item in enumerate(data):
        item["embedding"] = embeddings[i]

    return data


def pad_or_truncate(
    tensor: torch.Tensor, target_frames: int, dim: int = 0
) -> torch.Tensor:
    """Pad or truncate tensor to target length."""
    current_len = tensor.shape[dim]

    if current_len >= target_frames:
        return tensor[:target_frames] if dim == 0 else tensor

    # Pad with last frame
    padding_shape = list(tensor.shape)
    padding_shape[dim] = target_frames - current_len

    if dim == 0:
        last_frame = tensor[-1:].expand(padding_shape[0], *tensor.shape[1:])
        return torch.cat([tensor, last_frame], dim=0)

    return tensor


def merge_datasets(
    style_data: list, synthetic_data: list, target_frames: int = 250
) -> list:
    """Merge motion capture (100STYLE + AMASS) and synthetic datasets."""
    print(
        f"Merging datasets: {len(style_data)} motion capture + {len(synthetic_data)} synthetic"
    )

    merged = []

    # Process motion capture data (100STYLE and AMASS)
    for item in tqdm(style_data, desc="Processing motion capture"):
        motion = item.get("motion")
        if motion is None:
            continue
        motion = pad_or_truncate(motion, target_frames)

        # Get actions tensor (AMASS may have this, 100STYLE won't)
        actions = item.get("actions", torch.zeros(target_frames, dtype=torch.long))
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)
        actions = pad_or_truncate(actions, target_frames)

        merged.append(
            {
                "description": item.get("description", "A person performing motion"),
                "motion": motion,
                "actions": actions,
                "physics": torch.zeros(target_frames, 6),  # Will compute later
                "camera": torch.zeros(target_frames, 3),  # Default static camera
                "source": item.get("source", "100style"),  # Preserve original source
            }
        )

    # Add synthetic data
    # Synthetic data may have multi-actor format [frames, actors, dims]
    # Training expects single-actor format [frames, dims]
    # Extract primary actor (index 0) for training
    for item in synthetic_data:
        motion = item["motion"]

        # Handle multi-actor motion: [frames, actors, 20] -> [frames, 20]
        if len(motion.shape) == 3:
            motion = motion[:, 0, :]  # Extract first actor

        motion = pad_or_truncate(motion, target_frames)

        # Handle actions: [frames, actors] -> [frames]
        actions = item.get("actions", torch.zeros(target_frames, dtype=torch.long))
        if len(actions.shape) == 2:
            actions = actions[:, 0]  # Extract first actor's actions
        actions = pad_or_truncate(actions, target_frames)

        # Handle physics: [frames, actors, 6] -> [frames, 6]
        physics = item.get("physics", torch.zeros(target_frames, 6))
        if len(physics.shape) == 3:
            physics = physics[:, 0, :]  # Extract first actor's physics
        physics = pad_or_truncate(physics, target_frames)

        # Handle camera: keep as-is or extract
        camera = item.get("camera", torch.zeros(target_frames, 3))
        if len(camera.shape) == 2 and camera.shape[1] == 3:
            pass  # Already [frames, 3]
        elif len(camera.shape) == 2:
            camera = camera[:, :3]  # Take first 3 dims
        camera = pad_or_truncate(camera, target_frames)

        merged.append(
            {
                "description": item["description"],
                "motion": motion,
                "actions": actions,
                "physics": physics,
                "camera": camera,
                "source": "synthetic",
            }
        )

    print(f"Total merged samples: {len(merged)}")
    return merged


def compute_physics(data: list) -> list:
    """Compute physics tensors for samples missing them."""
    dt = 1.0 / 25.0  # 25 FPS

    for item in tqdm(data, desc="Computing physics"):
        if item.get("source") == "100style":
            # Compute physics from motion
            motion = item["motion"]
            positions = motion[:, :2]  # First 2 coords as position

            velocities = torch.zeros_like(positions)
            velocities[:-1] = (positions[1:] - positions[:-1]) / dt
            velocities[-1] = velocities[-2]

            accelerations = torch.zeros_like(velocities)
            accelerations[:-1] = (velocities[1:] - velocities[:-1]) / dt
            accelerations[-1] = accelerations[-2]

            momentum = velocities.clone()

            item["physics"] = torch.cat([velocities, accelerations, momentum], dim=1)

    return data


def main():
    parser = argparse.ArgumentParser(description="Prepare Stick-Gen training data")
    parser.add_argument(
        "--100style-dir",
        type=str,
        default="data/100Style",
        help="Path to 100STYLE dataset directory",
    )
    parser.add_argument(
        "--amass-dir",
        type=str,
        default="data/amass",
        help="Path to AMASS dataset directory",
    )
    parser.add_argument(
        "--smpl-dir",
        type=str,
        default="data/smpl_models",
        help="Path to SMPL model files directory",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--max-amass-samples",
        type=int,
        default=500,
        help="Maximum AMASS samples to process (None for all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_data_final.pt",
        help="Output path for final dataset",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=250,
        help="Target number of frames per sequence (10s @ 25fps)",
    )
    parser.add_argument(
        "--skip-100style", action="store_true", help="Skip 100STYLE processing"
    )
    parser.add_argument(
        "--skip-amass", action="store_true", help="Skip AMASS processing"
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true", help="Skip synthetic data generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Text embedding model",
    )
    parser.add_argument(
        "--force",
        "--overwrite",
        action="store_true",
        dest="force",
        help="Force regeneration of all data, overwriting existing files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Stick-Gen Data Preparation Pipeline")
    print("by Gestura AI")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  100STYLE dir:  {args.__dict__['100style_dir']}")
    print(f"  AMASS dir:     {args.amass_dir}")
    print(f"  SMPL dir:      {args.smpl_dir}")
    print(f"  Synthetic:     {args.synthetic_samples} samples")
    print(f"  Max AMASS:     {args.max_amass_samples} samples")
    print(f"  Target frames: {args.target_frames}")

    # Load text embedder
    print(f"\nLoading text embedder: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)

    style_data = []
    amass_data = []
    synthetic_data = []

    # Step 1: Process 100STYLE
    if not args.skip_100style and os.path.exists(args.__dict__["100style_dir"]):
        print(f"\n[1/5] Processing 100STYLE from {args.__dict__['100style_dir']}")
        style_data = load_100style_data(
            args.__dict__["100style_dir"], args.target_frames
        )
        style_data = create_100style_descriptions(style_data)
        print(f"  Loaded {len(style_data)} 100STYLE sequences")
    else:
        print("\n[1/5] Skipping 100STYLE (not found or --skip-100style)")

    # Step 2: Process AMASS
    if not args.skip_amass and os.path.exists(args.amass_dir):
        print(f"\n[2/5] Processing AMASS from {args.amass_dir}")
        amass_data = load_amass_data(
            amass_dir=args.amass_dir,
            smpl_dir=args.smpl_dir,
            target_frames=args.target_frames,
            max_samples=args.max_amass_samples,
        )
        print(f"  Loaded {len(amass_data)} AMASS sequences")
    else:
        print("\n[2/5] Skipping AMASS (not found or --skip-amass)")

    # Step 3: Generate synthetic data
    if not args.skip_synthetic:
        print(f"\n[3/5] Generating {args.synthetic_samples} synthetic samples")
        if args.force:
            print("  Force mode: will overwrite existing data")
        # Use the same directory as the output file for intermediate files
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "data"
        synthetic_data = generate_synthetic_data(
            args.synthetic_samples, args.target_frames, output_dir=output_dir, force=args.force
        )
        print(f"  Generated {len(synthetic_data)} synthetic samples")
    else:
        print("\n[3/5] Skipping synthetic generation (--skip-synthetic)")

    # Step 4: Merge all datasets
    print("\n[4/5] Merging datasets")
    # Combine 100STYLE and AMASS as "motion capture" data
    mocap_data = style_data + amass_data
    all_data = merge_datasets(mocap_data, synthetic_data, args.target_frames)

    # Compute physics for samples that need it
    all_data = compute_physics(all_data)

    # Step 5: Add embeddings
    print("\n[5/5] Creating text embeddings")
    all_data = add_embeddings(all_data, embedder)

    # Save final dataset
    print(f"\nSaving to {args.output}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(all_data, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"  Total samples: {len(all_data)}")
    print(
        f"  100STYLE samples: {len([d for d in all_data if d.get('source') in ['100style', '100style_txt']])}"
    )
    print(
        f"  AMASS samples: {len([d for d in all_data if d.get('source') == 'amass'])}"
    )
    print(
        f"  Synthetic samples: {len([d for d in all_data if d.get('source') == 'synthetic'])}"
    )
    print(f"  Frames per sample: {args.target_frames}")
    print(f"  Output file: {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
