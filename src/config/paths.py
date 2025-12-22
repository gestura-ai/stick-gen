"""
Centralized configuration for data generation paths.

This module defines a single source of truth for all data generation output paths,
making it easy to relocate generated data outside the repository.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GENERATION_DIR = PROJECT_ROOT / "generation"

# Environment variable override for generation directory
GENERATION_DIR = Path(os.getenv("STICK_GEN_GENERATION_DIR", DEFAULT_GENERATION_DIR))

# Data generation paths
PATHS = {
    # Motion capture conversions (canonical datasets)
    "motions_processed": GENERATION_DIR / "motions_processed",
    "amass_canonical": GENERATION_DIR / "motions_processed" / "amass" / "canonical.pt",
    "100style_canonical": GENERATION_DIR / "motions_processed" / "100style" / "canonical.pt",
    "humanml3d_canonical": GENERATION_DIR / "motions_processed" / "humanml3d" / "canonical.pt",
    "kit_ml_canonical": GENERATION_DIR / "motions_processed" / "kit_ml" / "canonical.pt",
    "interhuman_canonical": GENERATION_DIR / "motions_processed" / "interhuman" / "canonical.pt",
    "ntu_rgbd_canonical": GENERATION_DIR / "motions_processed" / "ntu_rgbd" / "canonical.pt",
    "babel_canonical": GENERATION_DIR / "motions_processed" / "babel" / "canonical.pt",
    "beat_canonical": GENERATION_DIR / "motions_processed" / "beat" / "canonical.pt",
    
    # Curated datasets
    "curated": GENERATION_DIR / "curated",
    "pretrain_data": GENERATION_DIR / "curated" / "pretrain_data.pt",
    "sft_data": GENERATION_DIR / "curated" / "sft_data.pt",
    "pretrain_data_embedded": GENERATION_DIR / "curated" / "pretrain_data_embedded.pt",
    "sft_data_embedded": GENERATION_DIR / "curated" / "sft_data_embedded.pt",
    "merged_canonical": GENERATION_DIR / "curated" / "merged_canonical.pt",
    
    # 2.5D parallax augmentation
    "parallax_2_5d": GENERATION_DIR / "2.5d_parallax",
    
    # Synthetic data
    "synthetic": GENERATION_DIR / "synthetic",
    
    # Legacy paths (deprecated but supported for backward compatibility)
    "legacy_data": PROJECT_ROOT / "data",
}


def get_path(key: str, create_parents: bool = False) -> Path:
    """
    Get a generation path by key.
    
    Args:
        key: Path key from PATHS dict
        create_parents: If True, create parent directories
        
    Returns:
        Path object
    """
    if key not in PATHS:
        raise KeyError(f"Unknown path key: {key}. Available: {list(PATHS.keys())}")
    
    path = PATHS[key]
    
    if create_parents and not path.exists():
        if path.suffix:  # It's a file
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
    
    return path


def get_generation_dir() -> Path:
    """Get the root generation directory."""
    return GENERATION_DIR


def set_generation_dir(path: str | Path) -> None:
    """
    Override the generation directory.
    
    Args:
        path: New generation directory path
    """
    global GENERATION_DIR, PATHS
    GENERATION_DIR = Path(path)
    
    # Regenerate all paths with new base
    PATHS.update({
        "motions_processed": GENERATION_DIR / "motions_processed",
        "amass_canonical": GENERATION_DIR / "motions_processed" / "amass" / "canonical.pt",
        "100style_canonical": GENERATION_DIR / "motions_processed" / "100style" / "canonical.pt",
        "humanml3d_canonical": GENERATION_DIR / "motions_processed" / "humanml3d" / "canonical.pt",
        "kit_ml_canonical": GENERATION_DIR / "motions_processed" / "kit_ml" / "canonical.pt",
        "interhuman_canonical": GENERATION_DIR / "motions_processed" / "interhuman" / "canonical.pt",
        "ntu_rgbd_canonical": GENERATION_DIR / "motions_processed" / "ntu_rgbd" / "canonical.pt",
        "babel_canonical": GENERATION_DIR / "motions_processed" / "babel" / "canonical.pt",
        "beat_canonical": GENERATION_DIR / "motions_processed" / "beat" / "canonical.pt",
        "curated": GENERATION_DIR / "curated",
        "pretrain_data": GENERATION_DIR / "curated" / "pretrain_data.pt",
        "sft_data": GENERATION_DIR / "curated" / "sft_data.pt",
        "pretrain_data_embedded": GENERATION_DIR / "curated" / "pretrain_data_embedded.pt",
        "sft_data_embedded": GENERATION_DIR / "curated" / "sft_data_embedded.pt",
        "merged_canonical": GENERATION_DIR / "curated" / "merged_canonical.pt",
        "parallax_2_5d": GENERATION_DIR / "2.5d_parallax",
        "synthetic": GENERATION_DIR / "synthetic",
    })
