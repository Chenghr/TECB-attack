import copy
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.data_preprocessing.cifar10.dataset import IndexedCIFAR10
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.baseline.vfl_models import (
    BottomModelForCifar10,
    TopModelForCifar10,
    BottomModelForCinic10,
    TopModelForCinic10,
    BottomModelForCifar100,
    TopModelForCifar100,
)
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import (
    AverageMeter,
    keep_predict_loss,
    over_write_args_from_file,
    image_format_2_rgb,
)
from fedml_core.utils.logger import setup_logger
from fedml_core.utils.metrics import ShuffleTrainMetric

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import shutil
import torchvision.transforms as transforms
from torch.utils.data import Subset
import copy
import random


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_shuffle_dataset(args, trainset, logger):
    shuffle_trainset = copy.deepcopy(trainset)
    shuffle_testset = copy.deepcopy(trainset)

    if args.shuffle_label_way == "class_to_class":
        logger.info(
            "Randomly replace the label of a certain class with the label of another class"
        )

        original_trainset_labels = shuffle_trainset.targets
        original_testset_labels = shuffle_testset.targets
        # 获取所有唯一的标签
        unique_labels = list(set(original_trainset_labels))
        num_labels = len(unique_labels)

        # 创建一个新的标签列表，确保新标签与原始标签不同
        new_labels = unique_labels[:]
        random.shuffle(new_labels)

        # 保证新标签和原标签不同
        for i in range(num_labels):
            if unique_labels[i] == new_labels[i]:
                swap_index = (i + 1) % num_labels
                new_labels[i], new_labels[swap_index] = (
                    new_labels[swap_index],
                    new_labels[i],
                )

        # 创建新的标签字典，将每个原始标签映射到一个新的唯一标签
        new_labels_mapping = {
            original: new for original, new in zip(unique_labels, new_labels)
        }
        # 记录扰乱前后标签的对应关系
        # logger.info("Original to new label mapping:")
        # for original_label, new_label in new_labels_mapping.items():
        #     logger.info(f"{original_label} -> {new_label}")
        logger.info(f"Original label: {new_labels_mapping.keys()}")
        logger.info(f"New label: {new_labels_mapping.values()}")

        # 将所有标签替换为新的标签
        shuffle_trainset.targets = [
            new_labels_mapping[label] for label in original_trainset_labels
        ]
        shuffle_testset.targets = [
            new_labels_mapping[label] for label in original_testset_labels
        ]

    elif args.shuffle_label_way == "random":
        logger.info(
            "Randomly replace the label of each sample with the label of other samples"
        )
        shuffle_labels = shuffle_trainset.targets
        random.shuffle(shuffle_labels)
        shuffle_trainset.targets = shuffle_labels

    else:
        logger.info("Do not shuffle labels.")

    return shuffle_trainset, shuffle_testset


def set_dataset(args):
    # load data
    if args.dataset == "CINIC10L":
        transform = transforms.Compose(
            [
                transforms.Lambda(image_format_2_rgb),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47889522, 0.47227842, 0.43047404),
                    (0.24205776, 0.23828046, 0.25874835),
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # Load dataset
    if args.dataset == "CIFAR10":
        # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
        trainset = IndexedCIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = IndexedCIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        # CIFAR-10 类别标签（以类别名称的列表形式给出）
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        target_class = args.target_class
        target_label = classes.index(target_class)
    elif args.dataset == "CIFAR100":
        trainset = IndexedCIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        testset = IndexedCIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
        target_class = args.target_class
        target_label = trainset.class_to_idx[target_class]
    elif args.dataset == "CINIC10L":
        trainset = CINIC10L(root=args.data_dir, split="train", transform=transform)
        testset = CINIC10L(root=args.data_dir, split="test", transform=transform)
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        target_class = args.target_class
        target_label = classes.index(target_class)
    else:
        raise ValueError("Not support dataset.")
    
    # 构建扰乱标签后的数据集
    shuffle_trainset, shuffle_testset = set_shuffle_dataset(args, trainset, logger)
    # 找出所有属于这个类别的样本的索引
    non_target_indices = np.where(np.array(testset.targets) != target_label)[0]
    non_target_set = Subset(testset, non_target_indices)

    train_queue = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size, num_workers=args.workers
    )
    non_target_queue = torch.utils.data.DataLoader(
        dataset=non_target_set, batch_size=args.batch_size, num_workers=args.workers
    )
    shuffle_train_queue = torch.utils.data.DataLoader(
        dataset=shuffle_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    shuffle_test_queue = torch.utils.data.DataLoader(
        dataset=shuffle_testset, batch_size=args.batch_size, num_workers=args.workers
    )

    return (
        train_queue,
        test_queue,
        non_target_queue,
        shuffle_train_queue,
        shuffle_test_queue,
        target_label,
    )


def set_model_releated(args, logger):
    # build model
    model_list = []

    if args.dataset == "CIFAR10":
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
    elif args.dataset == "CIFAR100":
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())
    elif args.dataset == "CINIC10L":
        model_list.append(BottomModelForCinic10())
        model_list.append(BottomModelForCinic10())
        model_list.append(TopModelForCinic10())
    else:
        raise ValueError("Not support dataset.")

    # 加载预训练模型以及 delta
    checkpoint_model_path = args.resume + "/model_best.pth.tar"
    checkpoint_delta_path = args.resume + "/delta.pth"
    checkpoint = torch.load(checkpoint_model_path)
    delta = torch.load(checkpoint_delta_path)

    # 加载每个模型的参数
    if args.load_model == 1:
        logger.info("Set_Model_Releated_Info: Only load bottom_model_b.")
        specific_models = [1]
        for i in specific_models:
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
    elif args.load_model == 2:
        logger.info("Set_Model_Releated_Info: Load all model.")
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
    else:
        logger.info("Set_Model_Releated_Info: No model loading operation performed.")

    # 复制一份 model_list
    copied_model_list = copy.deepcopy(model_list)

    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for model in copied_model_list
    ]

    stone1 = args.stone1  # 50 int(args.epochs * 0.5) 学习率衰减的epoch数
    stone2 = args.stone2  # 85 int(args.epochs * 0.8) 学习率衰减的epoch数
    lr_scheduler_list = [
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[stone1, stone2], 
            gamma=args.step_gamma
        )
        for optimizer in optimizer_list
    ]

    return model_list, copied_model_list, delta, optimizer_list, lr_scheduler_list


def main(logger, device, args):
    for seed in range(3):
        logger.info(f"seed: {seed}")
        set_seed(seed=seed)

        (
            train_queue,
            test_queue,
            non_target_queue,
            shuffle_train_queue,
            shuffle_test_queue,
            target_label,
        ) = set_dataset(args)

        (
            model_list,
            copied_model_list,
            delta,
            optimizer_list,
            lr_scheduler_list,
        ) = set_model_releated(args, logger)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        vfl_trainer = VFLTrainer(model_list)
        vfl_shuffle_trainer = VFLTrainer(copied_model_list)

        if args.train_bottom_model_b:
            logger.info("Training_info: Train bottom_model_b.")
        else:
            logger.info("Training_info: Not training bottom_model_b.")

        metrics = ShuffleTrainMetric()
        for epoch in range(0, args.epochs):
            shuffle_train_loss = vfl_shuffle_trainer.train_shuffle(
                shuffle_train_queue,
                criterion,
                bottom_criterion,
                optimizer_list,
                device,
                args,
            )

            lr_scheduler_list[0].step()
            if args.train_bottom_model_b:  # 训练 bottom model b
                lr_scheduler_list[1].step()
            lr_scheduler_list[2].step()
            
            shuffle_test_loss, _, _ = vfl_shuffle_trainer.test_mul(
                test_queue, criterion, device, args
            )
            # test clean loss
            main_task_loss, main_task_acc, main_task_top5_acc = vfl_shuffle_trainer.test_mul(
                train_queue, criterion, device, args
            )
            # test backdoor attack loss
            (
                backdoor_task_loss,
                backdoor_task_acc,
                _,
            ) = vfl_shuffle_trainer.test_backdoor_mul(
                train_queue, criterion, device, args, delta, target_label
            )

            metrics.update(
                shuffle_train_loss=shuffle_train_loss,
                shuffle_test_loss=shuffle_test_loss,
                main_task_loss=main_task_loss,
                main_task_top5_acc=main_task_top5_acc,
                backdoor_task_loss=backdoor_task_loss,
                main_task_acc=main_task_acc,
                backdoor_task_acc=backdoor_task_acc,
            )

            # logger.info(f"epoch: {epoch+1}, shuffle_train_loss: {shuffle_train_loss:.4f}, main_task_loss: {main_task_loss:.4f}, backdoor_task_loss: {backdoor_task_loss:.4f}")
        
        # 记录实验关键参数
        logger.info("########## Key Args ##########")
        logger.info(
            f"load_model={args.load_model}, train_bottom_model_b={args.train_bottom_model_b}, "
            f"shuffle_label_way={args.shuffle_label_way}, lr={args.lr}, shuffle_epochs={args.shuffle_epochs}, batch_size={args.batch_size}"
        )

        # 将列表结构记录到日志中
        logger.info("########## Record Results ##########")
        logger.info(f"Shuffle Train Loss: {metrics.shuffle_train_loss}")
        logger.info(f"Shuffle Test Loss: {metrics.shuffle_test_loss}")
        logger.info(f"Main Task Loss: {metrics.main_task_loss}")
        logger.info(f"Backdoor Attack Loss: {metrics.backdoor_attack_loss}")
        logger.info(f"Main Task Acc: {metrics.main_task_acc}")
        logger.info(f"Main Task Top5 Acc: {metrics.main_task_top5_acc}")
        logger.info(f"Backdoor Attack Acc: {metrics.backdoor_task_acc}")

        # 对比 shuffle model 和 normal 的输出差异
        logger.info("########## Compare acc in Main Task ##########")
        _, top1_acc_origin, top5_acc_origin = vfl_trainer.test_mul(
            test_queue, criterion, device, args
        )
        _, top1_acc_shuffle, top5_acc_shuffle = vfl_shuffle_trainer.test_mul(
            test_queue, criterion, device, args
        )
        logger.info(
            f"Origin Model in Main Task:  top1_acc:{top1_acc_origin:.4f}, top5_acc:{top5_acc_origin:.4f}"
        )
        logger.info(
            f"Shuffle Model in Main Task:  top1_acc:{top1_acc_shuffle:.4f}, top5_acc:{top5_acc_shuffle:.4f}"
        )

        logger.info("########## Compare acc in Backdoor Task ##########")
        _, test_asr_acc_origin, _ = vfl_trainer.test_backdoor_mul(
            non_target_queue, criterion, device, args, delta, target_label
        )
        _, test_asr_acc_shuffle, _ = vfl_shuffle_trainer.test_backdoor_mul(
            non_target_queue, criterion, device, args, delta, target_label
        )
        logger.info(
            f"Origin Model in Backdoor Task:  test_asr_acc:{test_asr_acc_origin:.4f}"
        )
        logger.info(
            f"Shuffle Model in Backdoor Task:  test_asr_acc:{test_asr_acc_shuffle:.4f}"
        )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("vfl_modelnet")

    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--data_dir", default="./data/CIFAR10/", help="location of the data corpus")
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=[
        "CIFAR10", "CIFAR100", "TinyImageNet", "CINIC10L", "Yahoo", "Criteo", "BCW"
    ], help="name of dataset")
    data_group.add_argument('--half', type=int, default=16,
        help='half number of features, generally seen as the adversary\'s feature num. '
            'You can change this para (lower that party_num) to evaluate the sensitivity '
            'of our attack -- pls make sure that the model to be resumed is '
            'correspondingly trained.',
    )  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32

    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
    training_group.add_argument("--epochs", type=int, default=100, help="num of training epochs")
    training_group.add_argument("--batch_size", type=int, default=256, help="batch size")
    training_group.add_argument("--lr", type=float, default=0.1, help="init learning rate")
    training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--report_freq", type=float, default=10, help="report frequency")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--stone1", default=50, type=int, metavar="s1", help="stone1 for step scheduler")
    training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")
    training_group.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--workers", type=int, default=0, help="num of workers")

    # 后门相关参数
    backdoor_group = parser.add_argument_group('Backdoor')
    backdoor_group.add_argument("--backdoor", type=float, default=20, help="backdoor frequency")
    backdoor_group.add_argument("--poison_epochs", type=float, default=20, help="backdoor frequency")
    backdoor_group.add_argument("--target_class", type=str, default="cat", help="backdoor target class")
    backdoor_group.add_argument("--alpha", type=float, default=0.01, help="uap learning rate decay")
    backdoor_group.add_argument("--eps", type=float, default=16 / 255, help="uap clamp bound")
    backdoor_group.add_argument("--poison_num", type=int, default=100, help="num of poison data")
    backdoor_group.add_argument("--corruption_amp", type=float, default=10, help="amplification of corruption")
    backdoor_group.add_argument("--backdoor_start", action="store_true", default=False, help="backdoor")
    backdoor_group.add_argument("--trigger_lr", type=float, default=0.001, help="init learning rate for trigger")

    # 防御相关参数
    # defense_group = parser.add_argument_group('Defense')
    # defense_group.add_argument("--marvell", action="store_true", default=False, help="marvell defense")
    # defense_group.add_argument("--max_norm", action="store_true", default=False, help="maxnorm defense")
    # defense_group.add_argument("--iso", action="store_true", default=False, help="iso defense")
    # defense_group.add_argument("--gc", action="store_true", default=False, help="gc defense")
    # defense_group.add_argument("--lap_noise", action="store_true", default=False, help="lap_noise defense")
    # defense_group.add_argument("--signSGD", action="store_true", default=False, help="sign_SGD defense")
    # defense_group.add_argument("--iso_ratio", type=float, default=0.01, help="iso defense ratio")
    # defense_group.add_argument("--gc_ratio", type=float, default=0.01, help="gc defense ratio")
    # defense_group.add_argument("--lap_noise_ratio", type=float, default=0.01, help="lap_noise defense ratio")

    # 扰动训练参数
    shuffle_group = parser.add_argument_group('Shuffle parameters')
    shuffle_group.add_argument('--shuffle_label_way', type=str, default="class_to_class", 
                        choices=['random', 'class_to_class', 'none'], help='how to shuffle label')
    shuffle_group.add_argument('--load_model', type=int, default=2, 
                        help='1: load bottom_model_b; 2: load all; else: not load.')
    shuffle_group.add_argument('--train_bottom_model_b', action='store_true', help='whether training bottom_model_b (default: False)')
    shuffle_group.add_argument('--shuffle_epochs', type=int, default=50, help='epochs of shuffle training')

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument('--name', type=str, default='vanilla_cifar10', help='experiment name')
    experiment_group.add_argument('--save', default='./results/logs/CIFAR10/pretest_shuffle/vanilla', 
                        type=str, metavar='PATH', help='path to save checkpoint (default: none)')
    experiment_group.add_argument('--log_file_name', type=str, default="experiment.log", help='log name')
    experiment_group.add_argument('--c', type=str, default='', help='config file')

    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)
    
    main(logger=logger, device=device, args=args)

