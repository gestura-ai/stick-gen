import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_dataset(config_path: str = "configs/base.yaml",
                       input_path: str = None, output_path: str = None):
    """
    Add text embeddings to the training dataset.

    Args:
        config_path: Path to YAML configuration file
        input_path: Override input path (uses config if None)
        output_path: Override output path (uses config if None)
    """
    # Load configuration
    config = load_config(config_path)
    gen_config = config.get("data_generation", {})

    # Get paths from config with fallbacks
    if input_path is None:
        input_path = gen_config.get("output_path", "data/train_data.pt")
    if output_path is None:
        output_path = gen_config.get("embedded_path", "data/train_data_embedded.pt")
    print(f"Loading dataset from {input_path}...")
    data = torch.load(input_path)

    # Use BAAI/bge-large-en-v1.5 - Top-tier quality, CPU-compatible (1024-dim)
    # Ranked #5 on MTEB leaderboard (Dec 2025), no flash_attn required
    model_name = "BAAI/bge-large-en-v1.5"
    print(f"Loading High-Quality Embedding Model ({model_name})...")
    print("Note: Top-5 on MTEB leaderboard, CPU-compatible, 1024-dim embeddings")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu")
    model.eval()

    print("Computing embeddings for all samples...")
    print(f"Total samples: {len(data)}")
    descriptions = [item["description"] for item in data]

    embeddings = []
    batch_size = 16  # Can use larger batches with bge-large on CPU

    for i in tqdm(range(0, len(descriptions), batch_size), desc="Embedding batches"):
        batch_texts = descriptions[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

            # For BGE models, use CLS token embedding (first token)
            last_hidden_states = outputs.last_hidden_state
            batch_embeddings = []
            for j in range(len(batch_texts)):
                # Use CLS token (first token)
                emb = last_hidden_states[j, 0, :]
                # Normalize embedding (standard for BGE models)
                emb = F.normalize(emb, p=2, dim=0)
                batch_embeddings.append(emb)

            embeddings.append(torch.stack(batch_embeddings))

    # Concatenate all batches
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"Embedding dimension: {all_embeddings.shape[1]}")

    # Merge back into dataset
    new_data = []
    for i, item in enumerate(data):
        new_item = item.copy()
        new_item["embedding"] = all_embeddings[i]
        new_data.append(new_item)

    print(f"Saving to {output_path}...")
    torch.save(new_data, output_path)
    print("Done! Embeddings upgraded to state-of-the-art quality.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text embeddings for training dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file (default: configs/base.yaml)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path"
    )
    args = parser.parse_args()

    preprocess_dataset(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output
    )
