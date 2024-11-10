import os
import sys
import random
import shutil

import numpy as np
from sklearn.utils import shuffle
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import copy

import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.vfl_models import (BottomModelForCifar10,
                                                  BottomModelForCifar100,
                                                  BottomModelForCinic10,
                                                  TopModelForCifar10,
                                                  TopModelForCifar100,
                                                  TopModelForCinic10)
from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.utils.utils import (AverageMeter, image_format_2_rgb,
                                    keep_predict_loss,
                                    over_write_args_from_file)
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_recommended_num_workers():
    """获取推荐的num_workers值"""
    import multiprocessing as mp
    
    # 获取CPU核心数
    cpu_count = mp.cpu_count()
    
    # 一般建议设置为CPU核心数的2-4倍
    recommended = min(cpu_count * 4, 8)  # 设置上限为16
    
    return recommended
    
def init_dataloader(dataset, data_dir="./data", batch_size=128):
    """加载数据集
    
    Parameters:
    -----------
    dataset_name : str
        数据集名称 ('CIFAR10', 'CIFAR100', 或 'CINIC10L')
    data_dir : str, optional
        数据存储路径，默认为'./data'
    batch_size : int, optional
        批次大小，默认为128
    num_workers : int, optional
        数据加载的工作进程数，默认为4
        
    Returns:
    --------
    tuple
        (train_dataloader, test_dataloader)
        
    Raises:
    -------
    ValueError
        当数据集不支持时抛出
    """
    transform = _init_transform(dataset)
    trainset, testset = _load_dataset(dataset, transform, data_dir)
    
    # 创建数据加载器
    num_workers = get_recommended_num_workers()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=False,   # set false for villian attack
        num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader

def _init_transform(dataset):
    # 定义数据转换
    if dataset == "CINIC10L":
        transform = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.47889522, 0.47227842, 0.43047404),
                std=(0.24205776, 0.23828046, 0.25874835)
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform

def _load_dataset(dataset, transform, data_dir):
    # 加载数据集
    if dataset == "CIFAR10":
        trainset = IndexedCIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        testset = IndexedCIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "CIFAR100":
        trainset = IndexedCIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        testset = IndexedCIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset == "CINIC10L":
        trainset = CINIC10L(
            root=data_dir,
            split="train",
            transform=transform
        )
        testset = CINIC10L(
            root=data_dir,
            split="test",
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return trainset, testset
    

def init_model_releated(dataset, lr, local_lr, momentum, weight_decay, stone1, stone2, step_gamma):
    model_list = []
    if dataset == "CIFAR10":
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
    elif dataset == "CIFAR100":
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())
    elif dataset == "CINIC10L":
        model_list.append(BottomModelForCinic10())
        model_list.append(BottomModelForCinic10())
        model_list.append(TopModelForCinic10())
    else:
        raise ValueError
    
    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        for model in model_list
    ]
    optimizer_list[1].param_groups[0]['lr'] = local_lr
    
    stone1, stone2 = stone1, stone2 
    lr_scheduler_list = [
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[stone1, stone2], 
            gamma=step_gamma
        )
        for optimizer in optimizer_list
    ]
    
    return model_list, optimizer_list, lr_scheduler_list


def load_model_and_backdoor_data(dataset, save_dir):
    model_dir = os.path.join(save_dir, "model_best.pth.tar")
    backdoor_data_dir = os.path.join(save_dir, "backdoor.pth")
    # print(model_dir, backdoor_data_dir)
    
    model_list = []
    if dataset == "CIFAR10":
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
        model_list.append(BottomModelForCifar10())  # set for pretrain
    elif dataset == "CIFAR100":
        model_list.append(BottomModelForCifar100())
        model_list.append(BottomModelForCifar100())
        model_list.append(TopModelForCifar100())
        model_list.append(BottomModelForCifar100())  # set for pretrain
    elif dataset == "CINIC10L":
        model_list.append(BottomModelForCinic10())
        model_list.append(BottomModelForCinic10())
        model_list.append(TopModelForCinic10())
        model_list.append(BottomModelForCinic10())  # set for pretrain
    else:
        raise ValueError
    
    saved_model_list = torch.load(model_dir)
    for i in range(len(model_list)):
        model_list[i].load_state_dict(saved_model_list["state_dict"][i])

    backdoor_data = torch.load(backdoor_data_dir)
    
    return model_list, backdoor_data


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)

