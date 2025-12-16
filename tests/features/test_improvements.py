#!/usr/bin/env python3.9
"""
Test script to verify all improvements work before full 100k training
Generates test videos with improved spatial movement
"""
from src.inference.generator import InferenceGenerator

print("=" * 60)
print("TESTING IMPROVED SPATIAL MOVEMENT SYSTEM")
print("=" * 60)
print()

# Test 1: Generate test videos with procedural animations
print("Test 1: Generating test videos with spatial movement...")

gen = InferenceGenerator("model_checkpoint.pth")

# Test case 1: Baseball
print("\n1. Baseball scenario (players running bases)...")
gen.generate(
    "Two teams playing against each other in a World Series playoff",
    "tests/outputs/test_baseball_improved.mp4",
)
print("✓ Baseball video generated: tests/outputs/test_baseball_improved.mp4")

# Test case 2: Space exploration
print("\n2. Space exploration scenario (walking through space)...")
gen.generate(
    "A man exploring space and meets an alien and eats a first meal with them",
    "tests/outputs/test_space_improved.mp4",
)
print("✓ Space video generated: tests/outputs/test_space_improved.mp4")

print()
print("=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
print()
print("Generated videos:")
print("  - tests/outputs/test_baseball_improved.mp4")
print("  - tests/outputs/test_space_improved.mp4")
print()
print("Please review these videos to verify:")
print("  ✓ Actors move through space (not stuck in place)")
print("  ✓ Baseball players run around bases")
print("  ✓ Space explorer walks from ship to aliens")
print("  ✓ Movements are smooth and realistic")
print()
