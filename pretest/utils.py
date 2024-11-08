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


def split_data(dataset, data, half=16):
    if dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
        x_a = data[:, :, :, 0 : half]
        x_b = data[:, :, :, half : 32]
    else:
        raise Exception("Unknown dataset name!")
    return x_a, x_b


def load_tecb_cifar10(batch_size=32, workers=1):
    data_dir = "../data/"
    save_model_dir = "../results/models/TECB/cifar10/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCifar10())
    model_list.append(BottomModelForCifar10())
    model_list.append(TopModelForCifar10())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = IndexedCIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = IndexedCIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


def load_tecb_cifar100(batch_size=32, workers=1):
    data_dir = "../data/"
    save_model_dir = "../results/models/TECB/cifar100/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCifar100())
    model_list.append(BottomModelForCifar100())
    model_list.append(TopModelForCifar100())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = IndexedCIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = IndexedCIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


def load_tecb_cinic10(batch_size=32, workers=1):
    data_dir = "../data/cinic/"
    save_model_dir = "../results/models/TECB/cinic10/"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCinic10())
    model_list.append(BottomModelForCinic10())
    model_list.append(TopModelForCinic10())
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

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
    trainset = CINIC10L(root=data_dir, split="/train", transform=transform)
    testset = CINIC10L(root=data_dir, split="/test", transform=transform)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, num_workers=workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=workers
    )
    
    return model_list, train_dataloader, test_dataloader, delta, target_label


def load_model_and_backdoor_data(dataset_name, file_path):
    """加载模型和后门攻击的相关信息
    
    Parameters:
    -----------
    dataset : str
        数据集名称 ('CIFAR10', 'CIFAR100', 或 'CINIC10L')
    file_path : str
        模型和后门数据的文件路径
        
    Returns:
    --------
    tuple
        (model_list, backdoor_data)
        
    Raises:
    -------
    ValueError
        当数据集不支持或文件不存在时抛出
    """
    
    # 检查文件路径
    model_path = os.path.join(file_path, "model_best.pth.tar")
    backdoor_path = os.path.join(file_path, "backdoor.pth")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(backdoor_path):
        raise FileNotFoundError(f"Backdoor file not found: {backdoor_path}")
        
    # 检查文件大小
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    if os.path.getsize(backdoor_path) == 0:
        raise ValueError(f"Backdoor file is empty: {backdoor_path}")

    model_list = []

    # 根据数据集选择模型
    if dataset_name == "CIFAR10":
        model_list.extend([
            BottomModelForCifar10(),
            BottomModelForCifar10(),
            TopModelForCifar10()
        ])
    elif dataset_name == "CIFAR100":
        model_list.extend([
            BottomModelForCifar100(),
            BottomModelForCifar100(),
            TopModelForCifar100()
        ])
    elif dataset_name == "CINIC10L":
        model_list.extend([
            BottomModelForCinic10(),
            BottomModelForCinic10(),
            TopModelForCinic10()
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    try:
        # 加载模型和后门数据
        checkpoint = torch.load(model_path)
        backdoor_data = torch.load(backdoor_path)
        
        # 验证checkpoint格式
        if "state_dict" not in checkpoint:
            raise ValueError("Invalid model checkpoint: 'state_dict' not found")
        if len(checkpoint["state_dict"]) != len(model_list):
            raise ValueError(f"Model state dict count mismatch: expected {len(model_list)}, got {len(checkpoint['state_dict'])}")
        
        # 加载模型参数
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
            
    except Exception as e:
        raise RuntimeError(f"Error loading model or backdoor data: {str(e)}")
    
    return model_list, backdoor_data


def load_dataset(dataset_name, data_dir="./data", batch_size=128):
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
    def get_recommended_num_workers():
        """获取推荐的num_workers值"""
        import multiprocessing as mp
        
        # 获取CPU核心数
        cpu_count = mp.cpu_count()
        
        # 一般建议设置为CPU核心数的2-4倍
        recommended = min(cpu_count * 4, 16)  # 设置上限为16
        
        return recommended
    
    # 定义数据转换
    if dataset_name == "CINIC10L":
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

    # 加载数据集
    if dataset_name == "CIFAR10":
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
    elif dataset_name == "CIFAR100":
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
    elif dataset_name == "CINIC10L":
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
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 创建数据加载器
    num_workers = get_recommended_num_workers()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    model_list, train_dataloader, test_dataloader, delta, target_label = load_tecb_cinic10()
    print(target_label)