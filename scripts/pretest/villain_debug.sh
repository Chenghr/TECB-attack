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
# 定义要更新的层配置
declare -a UPDATE_TOP_LAYERS=(
    "all"                                                   # 更新所有层
)

# Training parameters
ATTACK_METHOD='Villain'
EPOCHS=5
LEARNING_RATE=1e-4
BATCH_SIZE=256
UPDATE_MODE='both'

# 验证数组长度是否匹配
if [ ${#HALF[@]} -ne ${#SEED[@]} ]; then
    echo "Error: HALF and SEED arrays must have the same length"
    exit 1
fi

# 遍历所有配置并执行
for i in "${!HALF[@]}"; do
    current_half=${HALF[$i]}
    current_seed=${SEED[$i]}
    
    LOG_FILE="villain_debug.log"
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/Villain/cifar10/half_${current_half}/${current_seed}_saved_models"
    
    for layers in "${UPDATE_TOP_LAYERS[@]}"; do
        echo "Processing: half: ${current_half}, seed: ${current_seed}, mode: ${UPDATE_MODE}, layers: ${layers}"

        python "${SCRIPT_PATH}" \
            --attack_method "${ATTACK_METHOD}" \
            --dataset "${DATASET}" \
            --data_dir "${DATA_DIR}" \
            --model_dir "${MODEL_DIR}" \
            --half "${current_half}" \
            --epochs "${EPOCHS}" \
            --lr "${LEARNING_RATE}" \
            --batch_size "${BATCH_SIZE}" \
            --update_mode "${UPDATE_MODE}" \
            --update_top_layers ${layers} \
            --save "${SAVE_PATH}" \
            --log_file_name "${LOG_FILE}"
    
        echo "Completed processing for half: ${current_half}, seed: ${current_seed}, mode: ${UPDATE_MODE}, layers: ${layers}"
    done
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"