#!/usr/bin/env python3
"""
Batch update converter scripts to use centralized paths.

This script updates all converter scripts in src/data_gen/ to use
the centralized path configuration from src/config/paths.py.
"""

import re
from pathlib import Path

# Map of converter file to path key
CONVERTER_PATHS = {
    "convert_humanml3d.py": "humanml3d_canonical",
    "convert_kit_ml.py": "kit_ml_canonical",
    "convert_interhuman.py": "interhuman_canonical",
    "convert_ntu_rgbd.py": "ntu_rgbd_canonical",
    "convert_babel.py": "babel_canonical",
    "convert_beat.py": "beat_canonical",
}

DATA_GEN_DIR = Path(__file__).parent.parent / "src" / "data_gen"


def add_path_import(content: str, path_key: str) -> str:
    """Add centralized path import after existing imports."""
    
    # Find the last import line
    import_lines = []
    other_lines = []
    in_imports = True
    
    for line in content.split("\n"):
        if in_imports and (line.startswith("import ") or line.startswith("from ") or line.strip() == ""):
            import_lines.append(line)
        else:
            in_imports = False
            other_lines.append(line)
    
    # Add our import block
    path_import = f"""
# Import centralized paths config
try:
    from ..config.paths import get_path
    DEFAULT_OUTPUT_PATH = str(get_path("{path_key}"))
except ImportError:
    DEFAULT_OUTPUT_PATH = "data/motions_processed/{path_key.replace('_canonical', '')}/canonical.pt"
"""
    
    return "\n".join(import_lines) + "\n" + path_import + "\n" + "\n".join(other_lines)


def update_function_signature(content: str) -> str:
    """Update function signature to use None as default for output_path."""
    
    # Find convert_* function and change output_path default
    pattern = r'(output_path:\s*str\s*=\s*)"data/motions_processed/[^"]+\.pt"'
    replacement = r'\1None'
    
    return re.sub(pattern, replacement, content)


def add_output_path_check(content: str) -> str:
    """Add output_path None check in function body."""
    
    # Find the function body and add the check
    # Look for the docstring end and add the check right after
    pattern = r'("""\s*\n\s*)(.*?)(\n\s+\w+\s*=)'
    
    def replacer(match):
        docstring_end = match.group(1)
        content = match.group(2)
        next_line = match.group(3)
        
        # Add the output_path check
        check = "\n    if output_path is None:\n        output_path = DEFAULT_OUTPUT_PATH\n"
        return docstring_end + content + check + next_line
    
    return re.sub(pattern, replacer, content, count=1)


def update_argparse_default(content: str) -> str:
    """Update argparse default to use DEFAULT_OUTPUT_PATH."""
    
    pattern = r'(parser\.add_argument\([^)]*--output[^)]*default=)"data/motions_processed/[^"]+\.pt"'
    replacement = r'\1DEFAULT_OUTPUT_PATH'
    
    return re.sub(pattern, replacement, content)


def update_converter(filepath: Path, path_key: str):
    """Update a single converter file."""
    
    print(f"Updating {filepath.name}...")
    
    content = filepath.read_text()
    
    # Check if already updated
    if "from ..config.paths import get_path" in content:
        print(f"  Already updated, skipping")
        return
    
    # Apply transformations
    content = add_path_import(content, path_key)
    content = update_function_signature(content)
    content = add_output_path_check(content)
    content = update_argparse_default(content)
    
    filepath.write_text(content)
    print(f"  ✓ Updated")


def main():
    print("Batch updating converter scripts...")
    print(f"Data gen directory: {DATA_GEN_DIR}")
    print()
    
    for converter_file, path_key in CONVERTER_PATHS.items():
        filepath = DATA_GEN_DIR / converter_file
        
        if not filepath.exists():
            print(f"⚠️  {converter_file} not found, skipping")
            continue
        
        try:
            update_converter(filepath, path_key)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
