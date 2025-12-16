import argparse
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


def load_canonical(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Canonical file not found: {path}")
    data = torch.load(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts in {path}, got {type(data)}")
    return data


def add_embeddings(
    samples: list[dict[str, Any]], model_name: str, batch_size: int = 64
) -> None:
    """In-place addition of a 1024-d text embedding to each sample.

    This mirrors the behavior in scripts/prepare_data.py, but is generic over
    any canonical dataset that provides a "description" field.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    texts: list[str] = [s.get("description", "") for s in samples]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch, convert_to_tensor=True, device=device, show_progress_bar=False
            )
        all_embeddings.append(emb.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    if embeddings_tensor.ndim != 2:
        raise ValueError(
            f"Expected embeddings shape [N, D], got {embeddings_tensor.shape}"
        )

    for idx, sample in enumerate(samples):
        sample["embedding"] = embeddings_tensor[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training dataset from canonical motion data."
    )
    parser.add_argument(
        "--canonical", type=str, required=True, help="Path to canonical.pt file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output train_data.pt"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        # Must match the embedding dimension expected by training configs
        # and scripts/prepare_data.py (BAAI/bge-large-en-v1.5 → 1024-dim).
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model name (must match training config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size"
    )

    args = parser.parse_args()

    samples = load_canonical(args.canonical)
    if not samples:
        raise ValueError(f"No samples found in {args.canonical}")

    add_embeddings(samples, model_name=args.model_name, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(samples, args.output)
    print(f"Saved training dataset with {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
