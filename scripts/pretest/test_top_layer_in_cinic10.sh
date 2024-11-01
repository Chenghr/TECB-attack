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
# 定义要更新的层配置
declare -a UPDATE_TOP_LAYERS=(
    "fc1top bn0top"                                         # 更新第一层(fc1+bn0)
    "fc2top bn1top"                                         # 更新第二层(fc2+bn1)
    "fc3top bn2top"                                         # 更新第三层(fc3+bn2)
    "fc4top bn3top"                                         # 更新第四层(fc3+bn2)
    "fc1top bn0top fc2top bn1top"                           # 更新前两层(fc1+bn0, fc2+bn1)
    "fc1top bn0top fc2top bn1top fc3top bn2top"             # 更新前三层(fc1+bn0, fc2+bn1, fc3+bn2)
    "all"                                                   # 更新所有层
)

# Training parameters
EPOCHS=30
LEARNING_RATE=2e-5
BATCH_SIZE=256
UPDATE_MODE='top_only'

# 验证数组长度是否匹配
if [ ${#HALF[@]} -ne ${#SEED[@]} ]; then
    echo "Error: HALF and SEED arrays must have the same length"
    exit 1
fi

# 遍历所有配置并执行
for i in "${!HALF[@]}"; do
    current_half=${HALF[$i]}
    current_seed=${SEED[$i]}
    
    LOG_FILE="test_cinic10_top_layer_AC_$((32-HALF)).log"
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/TECB/cinic/half_${current_half}/${current_seed}_saved_models"
    
    for layers in "${UPDATE_TOP_LAYERS[@]}"; do
        echo "Processing: half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: ${layers}"

        python "${SCRIPT_PATH}" \
            --dataset "${DATASET}" \
            --data_dir "${DATA_DIR}" \
            --model_dir "${MODEL_DIR}" \
            --half "${current_half}" \
            --epochs "${EPOCHS}" \
            --lr "${LEARNING_RATE}" \
            --batch_size "${BATCH_SIZE}" \
            --update_mode "${UPDATE_MODE}" \
            --update_top_layers ${layers} \
            --log_file_name "${LOG_FILE}"
    
        echo "Completed processing for half: ${current_half}, seed: ${current_seed}, mode: ${update_mode}, layers: ${layers}"
    done
done

echo "All configurations processed successfully"
echo "Script completed at $(date)"