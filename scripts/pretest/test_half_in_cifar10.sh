#!/bin/bash

# Project configuration
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../../data"
SCRIPT_PATH="${PROJECT_ROOT}/../../pretest/permuted_training.py"  # 请替换为实际的Python脚本路径

# Dataset configuration
DATASET="CIFAR10"

# Model configurations
# 定义要测试的half值和对应的seed值
# declare -a HALF=(1 4 8 12 14 16 18 20 26)
# declare -a SEED=(1 1 1 1 3  1  1  1  3)
declare -a HALF=(1 4 8 12 14 16 18 20 26)
declare -a SEED=(1 1 1 1 2  1  1  1  2)

# Training parameters
EPOCHS=30
LEARNING_RATE=1e-3
BATCH_SIZE=256
LOG_FILE="test_half_mode=bottom_only.log"
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
    
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/TECB/cifar10/half_${current_half}/${current_seed}_saved_models"
    
    # 如果seed不为2，则跳过此次循环
    if [ "${current_seed}" != "2" ]; then
        echo "Skipping HALF=${current_half} with SEED=${current_seed} (only processing SEED=2)"
        continue
    fi
    
    echo "Processing: HALF=${current_half}, SEED=${current_seed}"
    
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