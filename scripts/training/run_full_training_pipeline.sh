#!/bin/bash
# Full training pipeline with all improvements
# Estimated time on CPU: 36-48 hours total

echo "============================================================"
echo "FULL TRAINING PIPELINE - 100K SAMPLES WITH ALL IMPROVEMENTS"
echo "============================================================"
echo ""
echo "This will run 3 phases:"
echo "  Phase 1: Generate 100k base samples (500k with augmentation)"
echo "  Phase 2: Generate embeddings with BAAI/bge-large-en-v1.5"
echo "  Phase 3: Train 15.5M parameter model for 50 epochs"
echo ""
echo "Estimated time: 36-48 hours on CPU"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Phase 1: Dataset generation
echo "============================================================"
echo "PHASE 1: Generating 100k training samples"
echo "============================================================"
echo "Estimated time: 16-20 hours"
echo ""
python3.9 -m src.data_gen.dataset_generator
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset generation failed!"
    exit 1
fi
echo ""

# Phase 2: Embedding generation
echo "============================================================"
echo "PHASE 2: Generating embeddings"
echo "============================================================"
echo "Estimated time: 4-8 hours"
echo ""
python3.9 -m src.data_gen.preprocess_embeddings
if [ $? -ne 0 ]; then
    echo "ERROR: Embedding generation failed!"
    exit 1
fi
echo ""

# Phase 3: Training
echo "============================================================"
echo "PHASE 3: Training 15.5M parameter model"
echo "============================================================"
echo "Estimated time: 12-24 hours"
echo ""
python3.9 -m src.train.train
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi
echo ""

echo "============================================================"
echo "TRAINING PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Model saved to: model_checkpoint.pth"
echo ""
echo "Next steps:"
echo "  1. Generate test videos with: ./stick-gen \"your prompt\" --output test.mp4"
echo "  2. Compare with previous version"
echo "  3. Analyze improvements"
echo ""

