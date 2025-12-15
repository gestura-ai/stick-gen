#!/bin/bash
# Live training monitor - updates every 30 seconds

echo "=========================================="
echo "TRAINING MONITOR - Press Ctrl+C to stop"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if training process is running
    TRAIN_PID=$(ps aux | grep "src.train.train" | grep -v grep | awk '{print $2}')
    
    if [ -z "$TRAIN_PID" ]; then
        echo "❌ Training process NOT running!"
        echo ""
        echo "Last checkpoint files:"
        ls -lht *.pth 2>/dev/null | head -3
        break
    else
        echo "✅ Training process RUNNING (PID: $TRAIN_PID)"
        echo ""
        
        # CPU and Memory usage
        echo "Resource Usage:"
        ps -p $TRAIN_PID -o %cpu,%mem,etime,rss | tail -1 | awk '{printf "  CPU: %s%%  Memory: %s%%  Runtime: %s  RAM: %.1fGB\n", $1, $2, $3, $4/1024/1024}'
        echo ""
        
        # Check for new checkpoint files
        echo "Recent Checkpoints:"
        ls -lht *.pth 2>/dev/null | head -5 | while read line; do
            echo "  $line"
        done
        echo ""
        
        # Estimate progress based on file modification times
        LATEST_CHECKPOINT=$(ls -t checkpoint_epoch_*.pth 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            EPOCH_NUM=$(echo $LATEST_CHECKPOINT | grep -o '[0-9]\+')
            echo "Last Saved Epoch: $EPOCH_NUM / 50"
            
            # Calculate time since last checkpoint
            LAST_MOD=$(stat -f %m "$LATEST_CHECKPOINT" 2>/dev/null || stat -c %Y "$LATEST_CHECKPOINT" 2>/dev/null)
            NOW=$(date +%s)
            DIFF=$((NOW - LAST_MOD))
            HOURS=$((DIFF / 3600))
            MINS=$(((DIFF % 3600) / 60))
            echo "Time since last checkpoint: ${HOURS}h ${MINS}m"
        else
            echo "No epoch checkpoints yet - still in Epoch 1"
        fi
        echo ""
        
        # System overall
        echo "System CPU: $(ps -A -o %cpu | awk '{s+=$1} END {print s}')%"
        echo ""
        echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
    fi
    
    sleep 30
done

