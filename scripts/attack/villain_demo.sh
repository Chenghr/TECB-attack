
set -e

# 基础路径
# PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
# DATASET="CIFAR10"
# DATA_PATH="${PROJECT_ROOT}/../data"
# BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar10/debug/demo"
# SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain_cifar10_training.py"
# YAML_PATH="${PROJECT_ROOT}/../attack/villain/best_configs/cifar10_bestattack.yml"
# HALF=16

# python ${SCRIPT_PATH} \
#     --dataset ${DATASET} \
#     --data_dir ${DATA_PATH} \
#     --half ${HALF} \
#     --save ${BASE_SAVE_PATH} \
#     --c ${YAML_PATH}

PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
DATASET="CIFAR100"
DATA_PATH="${PROJECT_ROOT}/../data"
BASE_SAVE_PATH="${PROJECT_ROOT}/../results/models/Villain/cifar100/debug/demo"
SCRIPT_PATH="${PROJECT_ROOT}/../attack/villain/villain_cifar100_training.py"
YAML_PATH="${PROJECT_ROOT}/../attack/villain/best_configs/cifar100_bestattack.yml"
HALF=16

python ${SCRIPT_PATH} \
    --dataset ${DATASET} \
    --data_dir ${DATA_PATH} \
    --half ${HALF} \
    --save ${BASE_SAVE_PATH} \
    --c ${YAML_PATH} \
    --epochs 10 \


