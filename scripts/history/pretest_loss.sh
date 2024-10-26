python vfl_cifar10_pretest_loss.py \
        --save ./results/CIFAR10/pretest_loss \
        --log_file_name debug.log \
        --load_model 0 \
        --lr 0.001 --epochs 100 --batch_size 64 \
        --poison_num 6000