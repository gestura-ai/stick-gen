#!/bin/bash
# Monitor Training Pipeline Progress
# Usage: ./monitor_training_pipeline.sh

echo "=========================================="
echo "STICK-GEN TRAINING PIPELINE MONITOR"
echo "=========================================="
echo ""

# Phase 1: Dataset Generation
echo "📊 PHASE 1: SYNTHETIC DATASET GENERATION"
echo "------------------------------------------"
if [ -f "dataset_generation_phase1.log" ]; then
    echo "✅ Log file exists"
    
    # Get latest progress
    latest_progress=$(tail -1 dataset_generation_phase1.log 2>/dev/null | grep -o '[0-9]*%' | head -1)
    if [ -n "$latest_progress" ]; then
        echo "   Progress: $latest_progress"
    fi
    
    # Get current sample count
    current_samples=$(tail -1 dataset_generation_phase1.log 2>/dev/null | grep -o '[0-9]*/10000' | head -1)
    if [ -n "$current_samples" ]; then
        echo "   Samples: $current_samples"
    fi
    
    # Get speed
    speed=$(tail -1 dataset_generation_phase1.log 2>/dev/null | grep -o '[0-9.]*it/s' | head -1)
    if [ -n "$speed" ]; then
        echo "   Speed: $speed"
    fi
    
    # Check if complete
    if grep -q "Done! Generated" dataset_generation_phase1.log 2>/dev/null; then
        echo "   Status: ✅ COMPLETE"
        total=$(grep "Done! Generated" dataset_generation_phase1.log | grep -o '[0-9]* total' | grep -o '[0-9]*')
        echo "   Total samples: $total"
    else
        echo "   Status: 🔄 IN PROGRESS"
    fi
else
    echo "❌ Not started (log file missing)"
fi

# Check if train_data.pt exists
if [ -f "data/train_data.pt" ]; then
    echo "   Output: ✅ data/train_data.pt exists"
    size=$(du -h data/train_data.pt | cut -f1)
    echo "   Size: $size"
else
    echo "   Output: ⏳ data/train_data.pt not created yet"
fi

echo ""

# Phase 2: Embedding Generation
echo "🔤 PHASE 2: TEXT EMBEDDING GENERATION"
echo "------------------------------------------"
if [ -f "embedding_generation_phase2.log" ]; then
    echo "✅ Log file exists"
    
    # Check progress
    if grep -q "Done! Embeddings upgraded" embedding_generation_phase2.log 2>/dev/null; then
        echo "   Status: ✅ COMPLETE"
    else
        echo "   Status: 🔄 IN PROGRESS"
        latest=$(tail -5 embedding_generation_phase2.log 2>/dev/null | grep -o '[0-9]*%' | tail -1)
        if [ -n "$latest" ]; then
            echo "   Progress: $latest"
        fi
    fi
else
    echo "⏳ Not started yet"
fi

if [ -f "data/train_data_embedded.pt" ]; then
    echo "   Output: ✅ data/train_data_embedded.pt exists"
    size=$(du -h data/train_data_embedded.pt | cut -f1)
    echo "   Size: $size"
else
    echo "   Output: ⏳ data/train_data_embedded.pt not created yet"
fi

echo ""

# Phase 3: AMASS Merge
echo "🔗 PHASE 3: AMASS DATA MERGE"
echo "------------------------------------------"
if [ -f "amass_merge_phase3.log" ]; then
    echo "✅ Log file exists"
    
    if grep -q "Merge complete" amass_merge_phase3.log 2>/dev/null; then
        echo "   Status: ✅ COMPLETE"
        total=$(grep "Total samples:" amass_merge_phase3.log | tail -1 | grep -o '[0-9]*')
        if [ -n "$total" ]; then
            echo "   Total samples: $total"
        fi
    else
        echo "   Status: 🔄 IN PROGRESS"
    fi
else
    echo "⏳ Not started yet"
fi

if [ -f "data/train_data_final.pt" ]; then
    echo "   Output: ✅ data/train_data_final.pt exists"
    size=$(du -h data/train_data_final.pt | cut -f1)
    echo "   Size: $size"
else
    echo "   Output: ⏳ data/train_data_final.pt not created yet"
fi

echo ""

# Phase 4: Model Training
echo "🎓 PHASE 4: MODEL TRAINING"
echo "------------------------------------------"
if [ -f "training_phase4.log" ]; then
    echo "✅ Log file exists"
    
    # Get latest epoch
    latest_epoch=$(tail -50 training_phase4.log 2>/dev/null | grep "Epoch" | tail -1 | grep -o 'Epoch [0-9]*/[0-9]*' | head -1)
    if [ -n "$latest_epoch" ]; then
        echo "   Progress: $latest_epoch"
    fi
    
    # Get latest loss
    latest_loss=$(tail -50 training_phase4.log 2>/dev/null | grep "Loss:" | tail -1 | grep -o 'Loss: [0-9.]*' | head -1)
    if [ -n "$latest_loss" ]; then
        echo "   $latest_loss"
    fi
    
    # Check for completion
    if grep -q "Training complete" training_phase4.log 2>/dev/null; then
        echo "   Status: ✅ COMPLETE"
    else
        echo "   Status: 🔄 IN PROGRESS"
    fi
else
    echo "⏳ Not started yet"
fi

# Check for checkpoints
if ls checkpoints/model_checkpoint_*.pth 1> /dev/null 2>&1; then
    echo "   Checkpoints: ✅ Found"
    count=$(ls checkpoints/model_checkpoint_*.pth 2>/dev/null | wc -l | tr -d ' ')
    echo "   Count: $count"
else
    echo "   Checkpoints: ⏳ None yet"
fi

echo ""
echo "=========================================="
echo "PROCESS STATUS"
echo "=========================================="
ps aux | grep -E "dataset_generator|preprocess_embeddings|merge_amass|train.py" | grep -v grep | awk '{print $2, $11, $12, $13, $14}'

echo ""
echo "=========================================="
echo "To monitor live progress:"
echo "  Phase 1: tail -f dataset_generation_phase1.log"
echo "  Phase 2: tail -f embedding_generation_phase2.log"
echo "  Phase 3: tail -f amass_merge_phase3.log"
echo "  Phase 4: tail -f training_phase4.log"
echo "=========================================="

