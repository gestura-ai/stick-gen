import os
import sys

import torch

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def verify_dataset(path="data/test_data.pt"):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    data = torch.load(path)
    print(f"Loaded {len(data)} samples")

    # Check for camera data
    has_camera = False
    has_llm_content = False

    for sample in data:
        if "camera" in sample:
            has_camera = True
            if sample["camera"].shape[1] != 3:
                print(f"Error: Invalid camera shape {sample['camera'].shape}")

        if (
            "heist" in sample["description"].lower()
            or "dance" in sample["description"].lower()
        ):
            has_llm_content = True

    print(f"Has Camera Data: {has_camera}")
    print(f"Has LLM Content: {has_llm_content}")

    if has_camera and has_llm_content:
        print("SUCCESS: Dataset verification passed!")
    else:
        print("FAILURE: Dataset verification failed.")


if __name__ == "__main__":
    from src.data_gen.dataset_generator import generate_dataset

    # Generate small dataset
    generate_dataset(num_samples=20, output_path="data/test_data.pt", augment=False)

    # Verify
    verify_dataset("data/test_data.pt")
