#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

echo "Script started at $(date)"

# Project configuration
PROJECT_ROOT="$(pwd)"
DATA_DIR="${PROJECT_ROOT}/../../data"
SCRIPT_PATH="${PROJECT_ROOT}/../../pretest/permuted_training.py"

DATASET="CIFAR10"

# 定义要更新的层配置
declare -a UPDATE_TOP_LAYERS=(
    "all"                                                   # 更新所有层
    "fc1top bn0top"                                         # 更新第一层(fc1+bn0)
    "fc2top bn1top"                                         # 更新第二层(fc2+bn1)
    "fc3top bn2top"                                         # 更新第三层(fc3+bn2)
    "fc4top bn3top"                                         # 更新第四层(fc3+bn2)
    "fc1top bn0top fc2top bn1top"                           # 更新前两层(fc1+bn0, fc2+bn1)
    "fc1top bn0top fc2top bn1top fc3top bn2top"             # 更新前三层(fc1+bn0, fc2+bn1, fc3+bn2)
)
# 定义更新模式
declare -a UPDATE_MODES=("top_only" "both")
declare -a HALFS=(4 8)

# 其他基础参数
EPOCHS=30
BATCH_SIZE=256
# HALF=4
SEED=1
LEARNING_RATE=2e-5


for HALF in "${HALFS[@]}"; do
    LOG_FILE="test_top_layer_AC_$((32-HALF)).log"
    MODEL_DIR="${PROJECT_ROOT}/../../results/models/TECB/cifar10/half_${HALF}/${SEED}_saved_models"

    # 遍历所有更新模式
    for update_mode in "${UPDATE_MODES[@]}"; do
        # 遍历所有层配置
        for layers in "${UPDATE_TOP_LAYERS[@]}"; do
            echo "Processing with mode: ${update_mode}, layers: ${layers}"
            
            python "${SCRIPT_PATH}" \
                --dataset "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --model_dir "${MODEL_DIR}" \
                --half "${HALF}" \
                --epochs "${EPOCHS}" \
                --lr "${LEARNING_RATE}" \
                --batch_size "${BATCH_SIZE}" \
                --update_mode "${update_mode}" \
                --update_top_layers ${layers} \
                --log_file_name "${LOG_FILE}"
                
            echo "Completed processing for mode: ${update_mode}, layers: ${layers}"
        done
    done
done
echo "All configurations processed successfully"
echo "Script completed at $(date)"