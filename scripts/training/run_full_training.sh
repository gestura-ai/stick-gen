#!/bin/bash

# Full Training Pipeline for 11M+ Parameter Model with 100k Samples
# This script runs all phases of the enhanced training pipeline

set -e  # Exit on error

echo "=========================================="
echo "STICK-GEN ENHANCED TRAINING PIPELINE"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Base samples: 100,000"
echo "  - Augmentation: 5x (total 500,000 samples)"
echo "  - Model size: 11M+ parameters"
echo "  - Embedding: Qwen3-Embedding-8B (4096-dim)"
echo "  - Sequence length: 10 seconds (250 frames)"
echo "  - Training device: CPU"
echo ""
echo "⚠️  WARNING: This will take significant time on CPU!"
echo "   Estimated time: 24-48 hours for full pipeline"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "=========================================="
echo "PHASE 1: Generate Training Data"
echo "=========================================="
echo "Generating 100k base samples (500k with augmentation)..."
python3.9 -m src.data_gen.dataset_generator

echo ""
echo "=========================================="
echo "PHASE 2: Generate Embeddings"
echo "=========================================="
echo "Computing state-of-the-art embeddings with Qwen3-Embedding-8B..."
echo "This will download the 8B parameter embedding model on first run..."
python3.9 -m src.data_gen.preprocess_embeddings

echo ""
echo "=========================================="
echo "PHASE 3: Train 11M+ Parameter Model"
echo "=========================================="
echo "Training with all improvements:"
echo "  - Temporal consistency loss"
echo "  - Multi-task learning"
echo "  - Gradient accumulation"
echo "  - 80/10/10 train/val/test split"
echo "  - Comprehensive metrics"
echo ""
python3.9 -m src.train.train

echo ""
echo "=========================================="
echo "TRAINING PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test the model:"
echo "     ./stick-gen \"Two teams playing against each other in a World Series playoff\" --output test_baseball.mp4"
echo "     ./stick-gen \"A man exploring space and meets an alien and eats a first meal with them\" --output test_space.mp4"
echo ""
echo "  2. Check model files:"
echo "     - model_checkpoint.pth (final model)"
echo "     - model_checkpoint_best.pth (best validation loss)"
echo "     - checkpoint_epoch_*.pth (periodic checkpoints)"
echo ""
echo "All done! 🎉"

