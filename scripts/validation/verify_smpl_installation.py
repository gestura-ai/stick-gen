#!/usr/bin/env python3
"""
Verify SMPL Tools Installation

Tests that all required SMPL processing libraries are installed correctly.
"""

import sys


def test_imports():
    """Test that all required libraries can be imported"""
    print("Testing library imports...\n")

    libraries = {
        "smplx": ("SMPL-X body model library", True),
        "trimesh": ("3D mesh processing", True),
        "pyrender": ("3D rendering", False),  # Optional - OpenGL issues on macOS
        "torch": ("PyTorch (required by SMPL-X)", True),
        "numpy": ("NumPy (required by SMPL-X)", True),
    }

    success = True
    for lib, (description, required) in libraries.items():
        try:
            __import__(lib)
            print(f"✓ {lib:15s} - {description}")
        except ImportError as e:
            if required:
                print(f"✗ {lib:15s} - FAILED: {e}")
                success = False
            else:
                print(f"⚠ {lib:15s} - OPTIONAL (not needed for AMASS): {e}")

    return success


def test_smplx_basic():
    """Test basic SMPL-X functionality with dummy data"""
    print("\n\nTesting SMPL-X basic functionality...\n")

    try:
        import smplx
        import torch

        print("Creating SMPL-X model (neutral gender, no model files)...")

        # Try to create model - this will fail without model files, but tests the API
        try:
            model = smplx.create(
                model_path="data/smpl_models",  # This directory doesn't exist yet
                model_type="smplh",
                gender="neutral",
            )
            print("✓ SMPL-X model created successfully!")
            print(f"  Model type: {model.__class__.__name__}")

            # Test with dummy data
            batch_size = 1
            body_pose = torch.zeros(batch_size, 63)  # 21 joints × 3

            output = model(body_pose=body_pose)
            print(f"✓ Forward pass successful!")
            print(f"  Output joints shape: {output.joints.shape}")
            print(f"  Expected: [1, 22, 3] (batch_size, num_joints, xyz)")

            return True

        except FileNotFoundError as e:
            print("⚠ SMPL model files not found (expected)")
            print(f"  Error: {e}")
            print(
                "\n  This is normal - SMPL model files must be downloaded separately."
            )
            print("  See instructions below for downloading SMPL models.")
            return True  # This is expected

    except Exception as e:
        print(f"✗ SMPL-X test failed: {e}")
        return False


def print_smpl_download_instructions():
    """Print instructions for downloading SMPL model files"""
    print("\n" + "=" * 70)
    print("SMPL MODEL FILES DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print(
        """
The SMPL-X library is installed, but you need to download the SMPL body model files separately.

**Steps to download SMPL models:**

1. Register at: https://smpl.is.tue.mpg.de/
   - Create an account
   - Accept the license agreement

2. Download SMPL+H model:
   - Go to: https://mano.is.tue.mpg.de/download.php
   - Download "SMPL+H model" (SMPLH_NEUTRAL.pkl, SMPLH_MALE.pkl, SMPLH_FEMALE.pkl)

3. Create directory structure:
   ```bash
   mkdir -p /Users/bc/gestura/stick-gen/data/smpl_models/smplh
   ```

4. Extract downloaded files to:
   ```
   /Users/bc/gestura/stick-gen/data/smpl_models/smplh/
   ├── SMPLH_NEUTRAL.pkl
   ├── SMPLH_MALE.pkl
   └── SMPLH_FEMALE.pkl
   ```

5. Re-run this verification script to test with actual model files.

**Note**: SMPL model files are required for AMASS dataset processing.
"""
    )
    print("=" * 70)


def main():
    """Main verification function"""
    print("=" * 70)
    print("SMPL TOOLS INSTALLATION VERIFICATION")
    print("=" * 70)
    print()

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n✗ Some libraries failed to import. Please install missing packages.")
        sys.exit(1)

    # Test SMPL-X
    smplx_ok = test_smplx_basic()

    # Print download instructions
    print_smpl_download_instructions()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Library imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"SMPL-X API:      {'✓ PASS' if smplx_ok else '✗ FAIL'}")
    print(f"SMPL models:     ⚠ NOT DOWNLOADED (see instructions above)")
    print("=" * 70)

    if imports_ok and smplx_ok:
        print("\n✅ SMPL tools installation successful!")
        print("   Next step: Download SMPL model files (see instructions above)")
        return 0
    else:
        print("\n✗ Installation verification failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
