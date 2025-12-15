"""
AMASS Dataset Batch Processor

Processes AMASS dataset in batches and converts to stick figure format.

Usage:
    python process_amass.py --amass_root data/amass --output data/amass_stick_data.pt --max_samples 400000
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from convert_amass import AMASSConverter, infer_action_from_filename, generate_description_from_action
from schema import ACTION_TO_IDX


def process_amass_dataset(
    amass_root: str = 'data/amass',
    output_path: str = 'data/amass_stick_data.pt',
    max_samples: int = 400000,
    batch_size: int = 100
):
    """
    Process AMASS dataset to stick figure format
    
    Args:
        amass_root: Root directory of AMASS dataset
        output_path: Output path for processed data
        max_samples: Maximum number of samples to process
        batch_size: Number of samples to process before saving checkpoint
    """
    print(f"Processing AMASS dataset from: {amass_root}")
    print(f"Output path: {output_path}")
    print(f"Max samples: {max_samples}")
    
    converter = AMASSConverter()
    
    # Find all .npz files
    npz_files = list(Path(amass_root).rglob('*.npz'))
    print(f"Found {len(npz_files)} AMASS sequences")
    
    # Limit to max_samples
    if len(npz_files) > max_samples:
        npz_files = npz_files[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    amass_data = []
    failed_files = []
    
    # Process in batches with progress bar
    for i, npz_path in enumerate(tqdm(npz_files, desc="Converting AMASS")):
        try:
            # Convert to stick figure
            motion_tensor = converter.convert_sequence(str(npz_path))
            
            # Infer action from filename
            action = infer_action_from_filename(str(npz_path))
            
            # Generate per-frame actions (all same for AMASS single-action sequences)
            actions = [action] * 250
            action_indices = [ACTION_TO_IDX[a] for a in actions]
            
            # Generate description
            description = generate_description_from_action(action)
            
            # Store data
            amass_data.append({
                'motion': motion_tensor,
                'action': action,
                'actions': torch.tensor(action_indices, dtype=torch.long),
                'description': description,
                'source': 'amass',
                'original_file': str(npz_path)
            })
            
            # Save checkpoint every batch_size samples
            if (i + 1) % batch_size == 0:
                checkpoint_path = output_path.replace('.pt', f'_checkpoint_{i+1}.pt')
                torch.save(amass_data, checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path} ({len(amass_data)} samples)")
            
        except Exception as e:
            failed_files.append((str(npz_path), str(e)))
            if len(failed_files) <= 10:  # Only print first 10 errors
                print(f"\n✗ Error processing {npz_path}: {e}")
            continue
    
    # Save final dataset
    print(f"\n✓ Successfully converted {len(amass_data)} sequences")
    print(f"✗ Failed to convert {len(failed_files)} sequences")
    
    torch.save(amass_data, output_path)
    print(f"✓ Saved to: {output_path}")
    
    # Save error log
    if failed_files:
        error_log_path = output_path.replace('.pt', '_errors.txt')
        with open(error_log_path, 'w') as f:
            for file_path, error in failed_files:
                f.write(f"{file_path}: {error}\n")
        print(f"✓ Error log saved to: {error_log_path}")
    
    return amass_data


def merge_with_synthetic_dataset(
    amass_data_path: str = 'data/amass_stick_data.pt',
    synthetic_data_path: str = 'data/train_data_embedded.pt',
    output_path: str = 'data/train_data_merged.pt'
):
    """
    Merge AMASS data with synthetic data
    
    Args:
        amass_data_path: Path to processed AMASS data
        synthetic_data_path: Path to synthetic training data
        output_path: Output path for merged dataset
    """
    print(f"Merging datasets...")
    
    # Load datasets
    amass_data = torch.load(amass_data_path)
    synthetic_data = torch.load(synthetic_data_path)
    
    print(f"AMASS samples: {len(amass_data)}")
    print(f"Synthetic samples: {len(synthetic_data)}")
    
    # Merge
    merged_data = synthetic_data + amass_data
    print(f"Total samples: {len(merged_data)}")
    
    # Save merged dataset
    torch.save(merged_data, output_path)
    print(f"✓ Merged dataset saved to: {output_path}")
    
    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AMASS dataset')
    parser.add_argument('--amass_root', type=str, default='data/amass',
                        help='Root directory of AMASS dataset')
    parser.add_argument('--output', type=str, default='data/amass_stick_data.pt',
                        help='Output path for processed data')
    parser.add_argument('--max_samples', type=int, default=400000,
                        help='Maximum number of samples to process')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Checkpoint save frequency')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with synthetic dataset after processing')
    
    args = parser.parse_args()
    
    # Process AMASS dataset
    amass_data = process_amass_dataset(
        amass_root=args.amass_root,
        output_path=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Optionally merge with synthetic data
    if args.merge:
        merge_with_synthetic_dataset(
            amass_data_path=args.output,
            synthetic_data_path='data/train_data_embedded.pt',
            output_path='data/train_data_merged.pt'
        )

