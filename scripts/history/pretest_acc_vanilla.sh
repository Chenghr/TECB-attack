#!/bin/bash

# 定义参数组合
base_save=".//results/logs/CIFAR10/pretest_shuffle/vanilla"

load_model_values=(1 2)
train_bottom_model_b_values=(False)
shuffle_label_way_values=("class_to_class")

lr_values=(0.001 0.0001)
shuffle_epochs_values=(50 100 150)
batch_size_values=(128)

log_file_name="pretest.log"
for shuffle_label_way in "${shuffle_label_way_values[@]}"; do
    save="${base_save}/${shuffle_label_way}_shuffle"
    for load_model in "${load_model_values[@]}"; do
        # 如果 shuffle_label_way 是 random，且 load_model 不为 2，则跳过
        if [ "$shuffle_label_way" = "random" ] && [ "$load_model" -ne 2 ]; then
            continue
        fi
        for train_bottom_model_b in "${train_bottom_model_b_values[@]}"; do
            base_log_name="load_model_mode=${load_model}-train_b=${train_bottom_model_b}"
            for lr in "${lr_values[@]}"; do
                for shuffle_epochs in "${shuffle_epochs_values[@]}"; do
                    for batch_size in "${batch_size_values[@]}"; do
                        # 生成 log 文件名
                        # log_file_name="${base_log_name}-lr=${lr}-shuffle_epochs=${shuffle_epochs}-batch_size=${batch_size}.log"

                        # 准备参数列表
                        params=(
                            --save "$save"
                            --log_file_name "$log_file_name"
                            --load_model "$load_model"
                            --shuffle_label_way "$shuffle_label_way"
                            --lr "$lr"
                            --shuffle_epochs "$shuffle_epochs"
                            --batch_size "$batch_size"
                        )
                        # 根据 train_bottom_model_b 值添加参数
                        if [ "$train_bottom_model_b" = true ]; then
                            params+=(--train_bottom_model_b)
                        fi

                        # 运行 Python 脚本并传递参数
                        python vanilla_cifar10_pretest_shuffle.py "${params[@]}"

                        # 等待 Python 脚本执行完毕
                        wait
                    done
                done
            done
        done
    done
done


        