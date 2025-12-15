# Start of file update
HF_DATASET_REPO="${HF_DATASET_REPO:-GesturaAI/stick-gen-dataset}"
UPLOAD_DATASET="${UPLOAD_DATASET:-false}"

# ... existing validate_environment ...

# Run evaluation
run_evaluation() {
    local variant="$1"
    local checkpoint="${CHECKPOINT_DIR}/model_checkpoint_best.pth"
    local output_file="${CHECKPOINT_DIR}/evaluation_results.json"
    
    echo ""
    echo -e "${YELLOW}[2.5/5] Running comprehensive evaluation for ${variant}...${NC}"
    
    if [ ! -f "${checkpoint}" ]; then
        echo -e "${YELLOW}WARNING: Checkpoint not found for evaluation.${NC}"
        return 1
    fi
    
    python scripts/run_comprehensive_eval.py \
        --checkpoint "${checkpoint}" \
        --output "${output_file}" \
        --num_samples 50  # Run enough samples for valid metrics
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Evaluation complete. Report saved.${NC}"
        cat "${output_file}"
    else
        echo -e "${RED}  ❌ Evaluation failed.${NC}"
        # detailed error handling if needed
    fi
}

# Upload dataset
upload_dataset() {
    if [ "${UPLOAD_DATASET}" != "true" ]; then
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}[Optional] Uploading dataset to HuggingFace...${NC}"
    
    # Assuming TRAIN_DATA_PATH is the dataset we want to share
    python scripts/push_dataset_to_hub.py \
        --dataset-path "${TRAIN_DATA_PATH}" \
        --repo-id "${HF_DATASET_REPO}" \
        --token "${HF_TOKEN}"
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Dataset uploaded: ${HF_DATASET_REPO}${NC}"
    else
        echo -e "${RED}  ❌ Dataset upload failed.${NC}"
    fi
}

# Update push_to_huggingface to use the enhanced script
push_to_huggingface() {
    local variant="$1"
    local checkpoint="${CHECKPOINT_DIR}/model_checkpoint_best.pth"
    local metrics_file="${CHECKPOINT_DIR}/evaluation_results.json"

    # ... checkpoint checks ...
    if [ ! -f "${checkpoint}" ]; then
         # (Check logic same as before, see context)
         checkpoint="${CHECKPOINT_DIR}/best_model.pth" # Simplification for brevity in this replace block, assume logic matches
    fi
    # ...

    local push_exit_code=0
    # Use the enhanced push_to_hub.py which handles metrics!
    python scripts/push_to_hub.py \
        --checkpoint "${checkpoint}" \
        --variant "${variant}" \
        --version "${VERSION}" \
        --token "${HF_TOKEN}" \
        --metrics "${metrics_file}" \
        --repo-name "${HF_REPO_NAME}-${variant}" || push_exit_code=$?

    # ... Success/Fail handling ...
    return $push_exit_code
}

# ... existing code ...

# In run_training_workflow:
run_training_workflow() {
    local variant="$1"
    # ...
    # Train
    # ...
    
    # EVALUATE
    run_evaluation "${variant}"
    
    # Push Model
    # ... (calls updated push_to_huggingface)
    
    # Cleanup
    # ...
}

# In main:
main() {
    validate_environment
    
    # Upload Dataset (once)
    upload_dataset
    
    # ... Training loop ...
}

