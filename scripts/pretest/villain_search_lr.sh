#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Script started at $(date)"

# Project configuration
PROJECT_ROOT="$(pwd)"
DATASET="CIFAR10"
DATA_DIR="${PROJECT_ROOT}/../../data/"
SCRIPT_PATH="${PROJECT_ROOT}/../../pretest/permuted_train_for_villain.py"
SAVE_PATH="${PROJECT_ROOT}/../../results/pretest/permuted_training/Villain"

declare -a HALF=(16)
declare -a SEED=(1)
declare -a LRS=(1e-3 1e-4 5e-5 2e-5 1e-5)

# Training parameters
ATTACK_METHOD='Villain'
EPOCHS=30
# LEARNING_RATE=1e-4
BATCH_SIZE=256
UPDATE_MODE='both'
UPDATE_TOP_LAYER='all'

# 验证数组长度是否匹配
if [ ${#HALF[@]} -ne ${#SEED[@]} ]; then
    echo "Error: HALF and SEED arrays must have the same length"
    exit 1
fi

# 遍历所有配置并执行
for i in "${!HALF[@]}"; do
    current_half=${HALF[$i]}
    current_seed=${SEED[$i]}
    
    LOG_FILE="villain_search_lr.log"
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/Villain/cifar10/half_${current_half}/${current_seed}_saved_models"
    
    for lr in "${LRS[@]}"; do
        echo "Processing: lr: ${lr}"

        python "${SCRIPT_PATH}" \
            --attack_method "${ATTACK_METHOD}" \
            --dataset "${DATASET}" \
            --data_dir "${DATA_DIR}" \
            --model_dir "${MODEL_DIR}" \
            --half "${current_half}" \
            --epochs "${EPOCHS}" \
            --lr "${lr}" \
            --batch_size "${BATCH_SIZE}" \
            --update_mode "${UPDATE_MODE}" \
            --update_top_layers ${UPDATE_TOP_LAYER} \
            --save "${SAVE_PATH}" \
            --log_file_name "${LOG_FILE}"
    
        echo "Completed processing for lr: ${lr}"
    done
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"