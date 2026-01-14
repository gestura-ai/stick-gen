#!/usr/bin/env python3
"""Sample unknown action labels from merged_canonical.pt for analysis."""
import torch
import random

path = 'data/processed/merged_canonical.pt'
samples = torch.load(path, weights_only=False)
print(f"Total samples: {len(samples)}")

# Collect unknown samples
unknown_samples = []
for i, s in enumerate(samples):
    if isinstance(s, dict):
        label = s.get("action_label", "unknown")
        if label == "unknown":
            unknown_samples.append((i, s))

print(f"Unknown samples: {len(unknown_samples)}")

# Sample 50 random unknown descriptions
random.seed(42)
sampled = random.sample(unknown_samples, min(50, len(unknown_samples)))

print("")
print("=== SAMPLE OF UNKNOWN DESCRIPTIONS ===")
print("")
for idx, (i, s) in enumerate(sampled):
    desc = s.get("description", s.get("text", "NO_DESC"))
    source = s.get("source", "?")
    if isinstance(desc, list):
        desc = desc[0] if desc else "EMPTY_LIST"
    desc_str = str(desc)[:180]
    print(f"[{idx+1}] src={source}: {desc_str}")

