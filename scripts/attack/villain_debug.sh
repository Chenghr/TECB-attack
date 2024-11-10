#!/bin/bash
set -e

# 基础路径
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR10"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar10/debug"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain.py"
YAML_PATH="${PROJECT_ROOT}/../attack/villain/best_configs/cifar10_bestattack.yml"
HALF=16
EPS=$(echo "scale=3; 16/255" | bc)

python ${SCRIPT_PATH} \
    --dataset ${DATASET} \
    --data_dir ${DATA_PATH} \
    --half ${HALF} \
    --save ${BASE_SAVE_PATH} \
    --yaml_path ${YAML_PATH} \
    --load_yaml \
    --seed_num 1 \
    --backdoor_start_epoch 2 \
    --epochs 10 \

# 基础路径
# PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
# DATASET="CIFAR100"
# DATA_PATH="${PROJECT_ROOT}/../data"
# BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar100/debug"
# SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain.py"
# YAML_PATH="${PROJECT_ROOT}/../attack/villain/best_configs/cifar10_bestattack.yml"

# REPORT_FREQ=5
# WORKERS=8
# POISON_BUDGET=0.1
# DROPOUT_RATIO=0.25
# BETA=0.4
# GAMMA_UP=1.2
# GAMMA_LOW=0.6
# TARGET_CLASS="apple"
# BATCH_SIZE=512
# LOCAL_LR=0.5
# EPS=0.06
# HALF=16

# python ${SCRIPT_PATH} \
#     --dataset ${DATASET} \
#     --data_dir ${DATA_PATH} \
#     --half ${HALF} \
#     --save ${BASE_SAVE_PATH} \
#     --seed_num 1 \
#     --report_freq ${REPORT_FREQ} \
#     --workers ${WORKERS} \
#     --poison_budget ${POISON_BUDGET} \
#     --dropout_ratio ${DROPOUT_RATIO} \
#     --beta ${BETA} \
#     --gamma_up ${GAMMA_UP} \
#     --gamma_low ${GAMMA_LOW} \
#     --target_class ${TARGET_CLASS} \
#     --batch_size ${BATCH_SIZE} \
#     --local_lr ${LOCAL_LR} \
#     --eps ${EPS} \
#     --backdoor_start_epoch 2 \
#     --epochs 10 \

# PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
# DATASET="CINIC10L"
# DATA_PATH="${PROJECT_ROOT}/../data/cinic/"
# BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cinic10/debug"
# SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain.py"
# YAML_PATH="${PROJECT_ROOT}/../attack/villain/best_configs/cifar10_bestattack.yml"

# REPORT_FREQ=5
# WORKERS=8
# POISON_BUDGET=0.1
# DROPOUT_RATIO=0.25
# BETA=0.4
# GAMMA_UP=1.2
# GAMMA_LOW=0.6
# TARGET_CLASS="cat"
# BATCH_SIZE=512
# LOCAL_LR=0.5
# EPS=0.06
# HALF=16

# python ${SCRIPT_PATH} \
#     --dataset ${DATASET} \
#     --data_dir ${DATA_PATH} \
#     --half ${HALF} \
#     --save ${BASE_SAVE_PATH} \
#     --seed_num 1 \
#     --report_freq ${REPORT_FREQ} \
#     --workers ${WORKERS} \
#     --poison_budget ${POISON_BUDGET} \
#     --dropout_ratio ${DROPOUT_RATIO} \
#     --beta ${BETA} \
#     --gamma_up ${GAMMA_UP} \
#     --gamma_low ${GAMMA_LOW} \
#     --target_class ${TARGET_CLASS} \
#     --batch_size ${BATCH_SIZE} \
#     --local_lr ${LOCAL_LR} \
#     --eps ${EPS} \
#     --backdoor_start_epoch 2 \
#     --epochs 10 \