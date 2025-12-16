#!/usr/bin/env python3
"""Verify documented parameter counts match actual model instantiation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.transformer import StickFigureTransformer


def main():
    print("=" * 70)
    print("VERIFICATION: Documented vs Actual Parameter Counts")
    print("=" * 70)

    configs = [
        ("small", 256, 8, 6, "lightweight_cnn", "gated"),
        ("medium", 384, 12, 8, "lightweight_cnn", "gated"),
        ("large", 512, 16, 10, "resnet", "cross_attention"),
    ]

    # Actual parameter counts from model instantiation
    documented = {
        "small": {"motion_only": 7_248_122, "multimodal": 11_717_412},
        "medium": {"motion_only": 20_594_618, "multimodal": 25_063_908},
        "large": {"motion_only": 44_618_362, "multimodal": 71_282_874},
    }

    all_pass = True

    for name, d_model, nhead, num_layers, img_enc, fusion in configs:
        print(f"\n{name.upper()} VARIANT (d_model={d_model}, layers={num_layers})")
        print("-" * 50)

        # Motion-only
        model_mo = StickFigureTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            enable_image_conditioning=False,
        )
        actual_mo = sum(p.numel() for p in model_mo.parameters())
        doc_mo = documented[name]["motion_only"]
        match_mo = "✅" if actual_mo == doc_mo else "❌"
        print(f"  Motion-only:  {actual_mo:>12,} (doc: {doc_mo:>12,}) {match_mo}")
        if actual_mo != doc_mo:
            all_pass = False

        # Multimodal
        model_mm = StickFigureTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            enable_image_conditioning=True,
            image_encoder_arch=img_enc,
            fusion_strategy=fusion,
        )
        actual_mm = sum(p.numel() for p in model_mm.parameters())
        doc_mm = documented[name]["multimodal"]
        match_mm = "✅" if actual_mm == doc_mm else "❌"
        print(f"  Multimodal:   {actual_mm:>12,} (doc: {doc_mm:>12,}) {match_mm}")
        if actual_mm != doc_mm:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL DOCUMENTED FIGURES MATCH ACTUAL MODEL PARAMETERS")
    else:
        print("❌ SOME FIGURES DO NOT MATCH - DOCUMENTATION NEEDS UPDATE")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
