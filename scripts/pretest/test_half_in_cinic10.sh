#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Script started at $(date)"

# Project configuration
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../../data/cinic/"
SCRIPT_PATH="${PROJECT_ROOT}/../../pretest/permuted_training.py"

DATASET="CINIC10L"

declare -a HALF=(1 4 8 12 16 24)
declare -a SEED=(1 0 1 2 2 1)

# Training parameters
EPOCHS=30
LEARNING_RATE=2e-5
BATCH_SIZE=256
LOG_FILE="test_in_cinic10_bottom_only.log"
UPDATE_MODE='bottom_only'

# 验证数组长度是否匹配
if [ ${#HALF[@]} -ne ${#SEED[@]} ]; then
    echo "Error: HALF and SEED arrays must have the same length"
    exit 1
fi

# 遍历所有配置并执行
for i in "${!HALF[@]}"; do
    current_half=${HALF[$i]}
    current_seed=${SEED[$i]}
    
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/TECB/cinic/half_${current_half}/${current_seed}_saved_models"
    
    echo "Processing: HALF=${current_half}, SEED=${current_seed}, MODE=${UPDATE_MODE}"

    python "${SCRIPT_PATH}" \
        --dataset "${DATASET}" \
        --data_dir "${DATA_DIR}" \
        --model_dir "${MODEL_DIR}" \
        --half "${current_half}" \
        --epochs "${EPOCHS}" \
        --lr "${LEARNING_RATE}" \
        --batch_size "${BATCH_SIZE}" \
        --update_mode "${UPDATE_MODE}" \
        --log_file_name "${LOG_FILE}"
    
    echo "Completed processing for HALF=${current_half}"
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"