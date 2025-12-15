#!/usr/bin/env python3
"""
Merge AMASS Dataset with Existing Synthetic Dataset

This script:
1. Loads converted AMASS sequences
2. Loads AMASS descriptions
3. Generates text embeddings for AMASS descriptions
4. Computes physics ground truth for AMASS sequences
5. Merges with existing synthetic dataset
6. Saves combined dataset

Usage:
    python merge_amass_dataset.py --synthetic data/train_data.pt --output data/train_data_merged.pt
"""

import torch
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.schema import ACTION_TO_IDX, ActionType


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load configuration from YAML file."""
    import os
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_physics_ground_truth(motion_tensor):
    """
    Compute physics ground truth for a motion sequence
    
    Args:
        motion_tensor: [250, 20] tensor
    
    Returns:
        physics_tensor: [250, 6] tensor (vx, vy, ax, ay, mx, my)
    """
    # Extract head position (first line, first 2 coords)
    positions = motion_tensor[:, :2]  # [250, 2]
    
    # Compute velocity: v(t) = (p(t+1) - p(t)) / dt
    dt = 1.0 / 25.0  # 25 FPS
    velocities = torch.zeros_like(positions)
    velocities[:-1] = (positions[1:] - positions[:-1]) / dt
    velocities[-1] = velocities[-2]  # Copy last velocity
    
    # Compute acceleration: a(t) = (v(t+1) - v(t)) / dt
    accelerations = torch.zeros_like(velocities)
    accelerations[:-1] = (velocities[1:] - velocities[:-1]) / dt
    accelerations[-1] = accelerations[-2]  # Copy last acceleration
    
    # Compute momentum: p = m * v (assume unit mass)
    momentum = velocities.clone()
    
    # Combine into physics tensor: [250, 6]
    physics_tensor = torch.cat([
        velocities,      # [250, 2]
        accelerations,   # [250, 2]
        momentum         # [250, 2]
    ], dim=1)
    
    return physics_tensor

def generate_text_embedding(description, model_name="BAAI/bge-large-en-v1.5"):
    """
    Generate text embedding for a description

    Args:
        description: Text description
        model_name: Sentence transformer model name

    Returns:
        embedding: [1024] tensor for BAAI/bge-large-en-v1.5
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Load model (cached after first load)
        if not hasattr(generate_text_embedding, 'model'):
            print(f"Loading embedding model: {model_name}...")
            generate_text_embedding.model = SentenceTransformer(model_name)

        # Generate embedding
        embedding = generate_text_embedding.model.encode(description, convert_to_tensor=True)
        return embedding

    except ImportError:
        print("⚠️  sentence-transformers not installed. Using random embeddings.")
        print("   Install with: pip install sentence-transformers")
        # Return random embedding as fallback
        return torch.randn(1024)

def main():
    parser = argparse.ArgumentParser(description='Merge AMASS with synthetic dataset')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to YAML configuration file')
    parser.add_argument('--synthetic', type=str, default=None, help='Path to synthetic dataset (overrides config)')
    parser.add_argument('--amass_dir', type=str, default='data/amass_converted', help='Path to converted AMASS data')
    parser.add_argument('--descriptions', type=str, default='data/amass_descriptions.json', help='Path to AMASS descriptions')
    parser.add_argument('--output', type=str, default=None, help='Output path for merged dataset (overrides config)')
    parser.add_argument('--max_amass', type=int, default=None, help='Maximum AMASS samples to include')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    gen_config = config.get("data_generation", {})

    # Get paths from config with fallbacks
    synthetic_input = args.synthetic or gen_config.get("embedded_path", "data/train_data_embedded.pt")
    output_path_str = args.output or gen_config.get("merged_path", "data/train_data_merged.pt")

    print("="*70)
    print("AMASS DATASET MERGER")
    print("="*70)
    print(f"Config: {args.config}")

    # Load synthetic dataset
    synthetic_path = Path(synthetic_input)
    if synthetic_path.exists():
        print(f"\n📂 Loading synthetic dataset from {synthetic_path}...")
        synthetic_data = torch.load(synthetic_path)
        print(f"   Loaded {len(synthetic_data)} synthetic samples")
    else:
        print(f"\n⚠️  Synthetic dataset not found at {synthetic_path}")
        print("   Creating new dataset with AMASS data only")
        synthetic_data = []
    
    # Load AMASS descriptions
    descriptions_path = Path(args.descriptions)
    if not descriptions_path.exists():
        print(f"\n❌ Error: AMASS descriptions not found at {descriptions_path}")
        print("   Run generate_amass_descriptions.py first")
        return
    
    print(f"\n📂 Loading AMASS descriptions from {descriptions_path}...")
    with open(descriptions_path, 'r') as f:
        descriptions = json.load(f)
    print(f"   Loaded {len(descriptions)} AMASS descriptions")
    
    # Limit AMASS samples if requested
    amass_sequences = list(descriptions.items())
    if args.max_amass is not None:
        amass_sequences = amass_sequences[:args.max_amass]
        print(f"   Limiting to {len(amass_sequences)} AMASS samples")
    
    # Process AMASS sequences
    print(f"\n🔄 Processing AMASS sequences...")
    amass_data = []
    
    for seq_path, metadata in tqdm(amass_sequences, desc="Processing AMASS"):
        try:
            # Load motion tensor
            full_path = Path(args.amass_dir) / seq_path
            motion = torch.load(full_path)
            
            # Get description and action
            description = metadata['description']
            action_idx = metadata['action_idx']
            
            # Generate per-frame actions (all same action for AMASS)
            actions = torch.full((250,), action_idx, dtype=torch.long)
            
            # Compute physics ground truth
            physics = compute_physics_ground_truth(motion)
            
            # Generate text embedding
            embedding = generate_text_embedding(description)

            # Create AMASS sample in same format as synthetic data
            amass_sample = {
                "description": description,
                "motion": motion,
                "actions": actions,
                "physics": physics,
                "embedding": embedding,
                "source": "amass",
                "dataset": metadata['dataset'],
                "augmented": False
            }

            amass_data.append(amass_sample)

        except Exception as e:
            print(f"\n❌ Error processing {seq_path}: {e}")
            continue

    print(f"\n✅ Processed {len(amass_data)} AMASS samples")

    # Merge datasets
    print(f"\n🔗 Merging datasets...")
    merged_data = synthetic_data + amass_data

    print(f"\n📊 Merged Dataset Statistics:")
    print(f"  Synthetic samples: {len(synthetic_data)}")
    print(f"  AMASS samples: {len(amass_data)}")
    print(f"  Total samples: {len(merged_data)}")

    # Count sources
    source_counts = {}
    for sample in merged_data:
        source = sample.get('source', 'synthetic')
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n📁 Source Distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} samples ({count/len(merged_data)*100:.1f}%)")

    # Save merged dataset
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving merged dataset to {output_path}...")
    torch.save(merged_data, output_path)

    print(f"\n✅ Merge complete!")
    print(f"✅ Total samples: {len(merged_data)}")
    print(f"✅ Saved to: {output_path}")

    # Calculate dataset size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ File size: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    main()

