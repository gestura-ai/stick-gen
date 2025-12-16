#!/usr/bin/env python3.9
"""
Quick test to verify training can start and process first batch
"""
import torch
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.insert(0, ".")

from src.model.transformer import StickFigureTransformer

print("=" * 60)
print("QUICK TRAINING START TEST")
print("=" * 60)

# Load or synthesize a small dataset
print("\n1. Preparing dataset...")


def _make_synthetic_sample(T: int = 32, embedding_dim: int = 1024):
    """Create a minimal sample matching the training schema.

    This avoids depending on a pre-generated train_data_final.pt file,
    while still exercising the same model input shapes.
    """
    motion = torch.zeros(T, 20)  # [T, 20] stick-figure motion
    embedding = torch.zeros(embedding_dim)  # [1024] text embedding
    actions = torch.zeros(T, dtype=torch.long)
    physics = torch.zeros(T, 6)
    return {
        "motion": motion,
        "embedding": embedding,
        "actions": actions,
        "physics": physics,
    }


try:
    data = torch.load("data/train_data_final.pt")
    print(f"   ✅ Loaded {len(data)} samples from data/train_data_final.pt")
except FileNotFoundError:
    # Fallback: synthetic dataset for CI / dev environments without data
    data = [_make_synthetic_sample() for _ in range(100)]
    print(
        f"   ⚠️ data/train_data_final.pt not found; using {len(data)} synthetic samples instead"
    )


# Create simple dataset
class StickFigureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item["motion"]
        embedding = item["embedding"]
        actions = item.get("actions", None)
        physics = item.get("physics", None)
        return (
            motion[:-1],
            embedding,
            motion[1:],
            actions[:-1] if actions is not None else None,
            physics[:-1] if physics is not None else None,
        )


# Create dataset and loader
print("\n2. Creating DataLoader...")
dataset = StickFigureDataset(data[:100])  # Just 100 samples
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
print(f"   ✅ DataLoader created with {len(dataset)} samples")

# Create model
print("\n3. Creating model...")
model = StickFigureTransformer(
    input_dim=20,  # 10 joints × 2 coords
    d_model=384,
    nhead=12,
    num_layers=8,
    output_dim=20,  # 10 joints × 2 coords
    embedding_dim=1024,
    dropout=0.1,
    num_actions=60,
)
print(f"   ✅ Model created")

# Try to process first batch
print("\n4. Processing first batch...")
for batch_idx, batch_data in enumerate(loader):
    print(f"   - Unpacking batch {batch_idx+1}...")
    data, embedding, target, actions, physics = batch_data

    print(f"   - Data shapes (batch-first):")
    print(f"     * data: {data.shape}")
    print(f"     * embedding: {embedding.shape}")
    print(f"     * target: {target.shape}")
    print(f"     * actions: {actions.shape if actions is not None else None}")
    print(f"     * physics: {physics.shape if physics is not None else None}")

    # Transpose to seq-first format [seq, batch, dim]
    data = data.transpose(0, 1)  # [batch, seq, dim] -> [seq, batch, dim]
    target = target.transpose(0, 1)
    if actions is not None:
        actions = actions.transpose(0, 1)  # [batch, seq] -> [seq, batch]
    if physics is not None:
        physics = physics.transpose(0, 1)  # [batch, seq, 6] -> [seq, batch, 6]

    print(f"   - Data shapes (seq-first):")
    print(f"     * data: {data.shape}")
    print(f"     * actions: {actions.shape if actions is not None else None}")

    print(f"   - Running forward pass...")
    with torch.no_grad():
        output = model(
            data, embedding, return_all_outputs=True, action_sequence=actions
        )

    print(f"   - Output keys: {output.keys()}")
    print(f"   - Output shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"     * {key}: {value.shape}")

    print(f"\n   ✅ First batch processed successfully!")
    break

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Training should work!")
print("=" * 60)
