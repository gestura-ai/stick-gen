"""
Stick-Gen Benchmark Suite
Gestura AI - https://gestura.ai

Runs a comprehensive evaluation of the model against a holdout set found in the data file.
Generates a "Scorecard" with:
- FID (Fréchet Inception Distance) vs Real Motion
- Physics Violation Rate
- Semantic Match Score
- Temporal Consistency (Smoothness)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_gen.schema import NUM_ACTIONS
from src.eval.metrics import (
    compute_dataset_fid_statistics,
    compute_frechet_distance,
    compute_motion_temporal_metrics,
    compute_physics_consistency_metrics,
    compute_synthetic_artifact_score,
)
from src.model.transformer import StickFigureTransformer


class BenchmarkDataset(Dataset):
    def __init__(self, data_path, split="test", limit=None):
        raw_data = torch.load(data_path)

        # Simple split logic (matches train.py 80/10/10)
        total = len(raw_data)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        total - train_size - val_size

        if split == "test":
            self.data = raw_data[train_size + val_size :]
        elif split == "val":
            self.data = raw_data[train_size : train_size + val_size]
        else:  # train
            self.data = raw_data[:train_size]

        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def run_benchmark(checkpoint_path, data_path, output_path, device="cpu", limit=100):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Init Model
    config = checkpoint.get(
        "config",
        {
            "input_dim": 20,
            "d_model": 384,
            "nhead": 12,
            "num_layers": 8,
            "output_dim": 20,
            "embedding_dim": 1024,
            "dropout": 0.1,
            "num_actions": NUM_ACTIONS,
        },
    )

    model = StickFigureTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading data from {data_path} (limit={limit})")
    dataset = BenchmarkDataset(data_path, split="test", limit=limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    generated_motions = []
    real_motions = []

    metrics = {
        "physics_violations": [],
        "semantic_match": [],
        "smoothness": [],
        "artifacts": [],
    }

    print("Running generation benchmark...")
    with torch.no_grad():
        for item in tqdm(loader):
            # item keys: motion, embedding, description, etc.
            # We predict using the embedding (or text if we had the text encoder loaded here,
            # but dataset has embeddings)

            embedding = item["embedding"].to(device)  # [1, 1024]
            real_motion = item["motion"].to(device)  # [1, T, 20]

            # Generate autoregressively (simplified for benchmark speed - using teacher forcing for now
            # OR ideally we should generate from scratch.
            # For comprehensive benchmark, we usually want open-loop generation.)
            # However, open-loop generation is slow.
            # Let's do open-loop generation for N steps.

            # Initialize with first frame of real motion
            curr_motion = real_motion[:, :1, :]
            gen_seq = [curr_motion]

            # Predict 50 frames (2 seconds) to save time, or full length?
            # Let's do 50 frames for speed in this demo benchmark
            target_len = min(50, real_motion.shape[1])

            for _ in range(target_len - 1):
                # Prepare input [seq, batch, dim]
                inp = torch.cat(gen_seq, dim=1).permute(1, 0, 2)
                out = model(inp, embedding)  # [seq, batch, dim]
                next_frame = out[-1:, :, :]  # [1, batch, dim]
                gen_seq.append(next_frame.permute(1, 0, 2))

            generated_motion = torch.cat(gen_seq, dim=1)  # [1, T, 20]

            # Store for FID
            generated_motions.append(generated_motion.cpu().squeeze(0))
            real_motions.append(real_motion[:, :target_len, :].cpu().squeeze(0))

            # 1. Physics Violation (Check on generated)
            phys_stats = compute_physics_consistency_metrics(generated_motion.cpu())
            metrics["physics_violations"].append(0 if phys_stats["is_valid"] else 1)

            # 2. Semantic Match (Cosine sim between embedding and motion-embedding?
            # Without a motion-to-text encoder (like CLIP), we can't do true Semantic Match
            # unless we trained a joint embedding.
            # Stick-Gen `metrics.py` has `compute_text_alignment_from_embeddings`
            # but that requires motion feature embedding in the same space.
            # We will interpret "Semantic Match" here as MSE against Ground Truth for now,
            # or skip if we lack the contrastive model.)
            # ... For this benchmark, we'll use MSE as a proxy for "Semantic Fidelity to Plan"
            semantic_proxy = torch.nn.functional.mse_loss(
                generated_motion, real_motion[:, :target_len, :]
            )
            metrics["semantic_match"].append(
                1.0 / (1.0 + semantic_proxy.item())
            )  # higher is better

            # 3. Smoothness
            temp_stats = compute_motion_temporal_metrics(generated_motion.cpu())
            metrics["smoothness"].append(temp_stats["smoothness_score"])

            # 4. Artifacts
            art_stats = compute_synthetic_artifact_score(generated_motion.cpu())
            metrics["artifacts"].append(art_stats["artifact_score"])

    # Compute FID
    print("Computing FID...")
    gen_stats = compute_dataset_fid_statistics(generated_motions)
    real_stats = compute_dataset_fid_statistics(real_motions)
    fid_score = compute_frechet_distance(gen_stats, real_stats)

    # Aggregate Scorecard
    scorecard = {
        "FID (Lower is better)": fid_score,
        "Physics Violation Rate (Lower is better)": np.mean(
            metrics["physics_violations"]
        ),
        "Semantic Match Score (Higher is better)": np.mean(metrics["semantic_match"]),
        "Smoothness Score (Higher is better)": np.mean(metrics["smoothness"]),
        "Artifact Score (Lower is better)": np.mean(metrics["artifacts"]),
    }

    print("\n" + "=" * 40)
    print("📊 STICK-GEN BENCHMARK SCORECARD")
    print("=" * 40)
    for k, v in scorecard.items():
        print(f"{k:45s}: {v:.4f}")
    print("=" * 40)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(scorecard, f, indent=2)
        print(f"Scorecard saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/train_data_final.pt")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    run_benchmark(args.checkpoint, args.data, args.output, device, args.limit)


if __name__ == "__main__":
    main()
