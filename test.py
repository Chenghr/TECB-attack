import copy
import os
import random
import sys
import json

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
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
from fedml_core.trainer.defense_trainer import DefenseTrainer
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


def load_dataset(args, logger):
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
        transform = transforms.Compose([transforms.ToTensor(),])

    # Load dataset
    if args.dataset == "CIFAR10":
        # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
        trainset = IndexedCIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = IndexedCIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "CIFAR100":
        trainset = IndexedCIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        testset = IndexedCIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "CINIC10L":
        trainset = CINIC10L(root=args.data_dir, split="train", transform=transform)
        testset = CINIC10L(root=args.data_dir, split="test", transform=transform)
    else:
        raise ValueError("Not support dataset.")

    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size, num_workers=args.workers
    )

    return train_dataloader, test_dataloader
    
def load_model_and_backdoor_data(args, logger):
    """加载模型和后门攻击的相关信息, backdoor_data: dict
    """
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
    
    checkpoint_model_path = args.resume + "/model_best.pth.tar"
    checkpoint_backdoor_path = args.resume + "/backdoor.pth"
    checkpoint = torch.load(checkpoint_model_path)
    backdoor_data = torch.load(checkpoint_backdoor_path)
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])
    
    return model_list, backdoor_data


def main(args, logger, device):
    train_dataloader, test_dataloader = load_dataset(args, logger)
    model_list, backdoor_data = load_model_and_backdoor_data(args, logger)
    
    defense_trainer = DefenseTrainer(model_list)
    
    # 1. 选择子数据集
    print(f"Original train_dataloader length: {len(train_dataloader.dataset)}")
    top_confidence_subset = defense_trainer.select_top_confidence_subset_by_class(
        train_dataloader, device, args, top_fraction=args.top_fraction
    )
    print(f"Top confidence subset length: {len(top_confidence_subset)}")
    selected_dataloader = torch.utils.data.DataLoader(
        dataset=top_confidence_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    # 2. 标签扰动
    perturbed_dataset = defense_trainer.create_label_perturbed_subset(
        selected_dataloader, device, args, 
    )
    print(f"Label perturbed subset length: {len(perturbed_dataset)}")
    # for i in range(10):
    #     _, origin_label, _ = top_confidence_subset[i]
    #     _, new_label, _ = perturbed_dataset[i]
    #     mappint_result = perturbed_dataset.label_mapping[origin_label]
    #     print(origin_label, new_label, mappint_result)
        
    perturbed_train_dataloader = torch.utils.data.DataLoader(
        dataset=perturbed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    
    # 3. 训练扰动模型
    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss
    defense_trainer.create_perturbed_models(args)
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            args.perturbed_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for model in defense_trainer.perturbed_models
    ]
    lr_scheduler_list = [
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[args.stone1, args.stone2], 
            gamma=args.step_gamma
        )
        for optimizer in optimizer_list
    ]
    
    _, base_main_task_acc, _ = defense_trainer.test_perturbed(test_dataloader, criterion, device, args)
    
    cur_main_task_acc, epoch = base_main_task_acc, 0
    perturbed_loss, main_task_acc = [], []
    
    while cur_main_task_acc > base_main_task_acc * args.acc_fraction:
        loss = defense_trainer.train_perturbed(
            perturbed_train_dataloader, criterion, bottom_criterion, optimizer_list, device, args
        )
        
        lr_scheduler_list[0].step()
        lr_scheduler_list[2].step()
        epoch += 1
        
        _, cur_main_task_acc, _ = defense_trainer.test_perturbed(test_dataloader, criterion, device, args)
        
        perturbed_loss.append(loss)
        main_task_acc.append(cur_main_task_acc)
        
    logger.info(f"totol epochs: {epoch}")
    logger.info(f"perturbed_loss: {perturbed_loss}")
    logger.info(f"main_task_acc: {main_task_acc}")
        
    # 4. evaluate defense method
    print(backdoor_data)
    target_class = backdoor_data["target_class"]
    delta = backdoor_data["delta"]
    # target_class = 3
    # checkpoint_backdoor_path = args.resume + "/delta.pth"
    # delta = torch.load(checkpoint_backdoor_path)
    
    poison_indices = defense_trainer.create_poison_index(
        test_dataloader, target_class, args.poison_ratio
    )
    # print(len(poison_indices))
    
    predictions, is_clean = defense_trainer.predict_with_defense(
        test_dataloader, criterion, device, args, delta, poison_indices, gap_threshold=0.3
    )

    metrics = defense_trainer.evaluate_interception_metrics(predictions, is_clean)
    
    metrics_str = ""
    for key, value in metrics.items():
        metrics_str += f"{key}: {value}, "
    
    logger.info(metrics_str)



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("test_defense")

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument('--name', type=str, default='test_defense', help='experiment name')
    experiment_group.add_argument('--save', default='./results/models/CIFAR10/Defense/test', 
                        type=str, metavar='PATH', help='path to save checkpoint (default: none)')
    experiment_group.add_argument('--log_file_name', type=str, default="test.log", help='log name')
    experiment_group.add_argument('--resume', type=str, default="./results/models/TECB/cifar10/", help='log name')
    experiment_group.add_argument('--c', type=str, default='', help='config file')
    
    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=[
        "CIFAR10", "CIFAR100","CINIC10L",], help="name of dataset")
    data_group.add_argument("--data_dir", default="./data/CIFAR10/", help="location of the data corpus")
    data_group.add_argument('--half', type=int, default=16,)  

    # 防御参数
    defense_group = parser.add_argument_group('Defense')
    defense_group.add_argument("--top_fraction", type=float, default=0.3,)
    defense_group.add_argument("--acc_fraction", type=float, default=0.6,)
    defense_group.add_argument("--perturbed_lr", type=float, default=0.0001,)
    defense_group.add_argument("--poison_ratio", type=float, default=0.6, help="poison sample ratio in testset.")
    
    # defense_group.add_argument("--marvell", action="store_true", default=False, )

    training_group = parser.add_argument_group('Training')
    # training_group.add_argument("--epochs", type=int, default=1, help="num of training epochs")
    training_group.add_argument("--batch_size", type=int, default=256, help="batch size")
    training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--report_freq", type=float, default=10, help="report frequency")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--stone1", default=50, type=int, metavar="s1", help="stone1 for step scheduler")
    training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")
    training_group.add_argument("--workers", type=int, default=0, help="num of workers")
    
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)
    
    main(logger=logger, device=device, args=args)

