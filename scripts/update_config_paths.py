#!/usr/bin/env python3
"""Update config files to use generation/ paths instead of data/"""

import re
from pathlib import Path

def update_config_file(filepath: Path):
    """Update a single config file's data paths."""
    content = filepath.read_text()
    original = content
    
    # Replace data/curated with generation/curated
    content = re.sub(
        r'"data/curated/',
        '"generation/curated/',
        content
    )
    
    # Replace data/motions_processed with generation/motions_processed
    content = re.sub(
        r'"data/motions_processed/',
        '"generation/motions_processed/',
        content
    )
    
    if content != original:
        filepath.write_text(content)
        print(f"✓ Updated {filepath.name}")
        return True
    else:
        print(f"  No changes needed for {filepath.name}")
        return False

def main():
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print(f"Error: {configs_dir} not found")
        return
    
    print("Updating config files to use generation/ paths...\n")
    
    updated_count = 0
    for config_file in sorted(configs_dir.glob("*.yaml")):
        if update_config_file(config_file):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} config files")

if __name__ == "__main__":
    main()
