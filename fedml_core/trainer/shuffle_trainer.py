import argparse
import copy
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fedml_core.data_preprocessing.cifar10.dataset import IndexedCIFAR10
from fedml_core.data_preprocessing.cifar100.dataset import IndexedCIFAR100
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.baseline.vfl_models import (
    BottomModelForCifar10,
    BottomModelForCifar100,
    BottomModelForCinic10,
    TopModelForCifar10,
    TopModelForCifar100,
    TopModelForCinic10,
)
from fedml_core.utils.utils import (
    AverageMeter,
    backdoor_truepostive_rate,
    gradient_compression,
    gradient_gaussian_noise_masking,
    gradient_masking,
    laplacian_noise_masking,
    marvell_g,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import Subset

from .vfl_trainer import VFLTrainer


class ShuffleDefense(object):
    def __init__(self, logger, args):
        self.logger = logger
        self.args = args

        self.vfl_model = self.load_vfl_model(args)

    def select_high_confidence_subset(self, confidence_threshold=0.8):
        """
        从训练集中选择预测准确率高且预测概率接近标签的样本，构建高置信度子集
        Args:
            model: 训练好的模型，用于预测
            dataloader: 训练数据加载器
            confidence_threshold: 置信度阈值，只有预测概率高于该值且预测正确的样本才会被选入子集

        Returns:
            high_confidence_subset: 经过筛选后的高置信度子集
        """
        dataloader = self.load_train_dataset(self.args)
        device = self.args.device
        model_list = [model.to(device) for model in self.vfl_model]
        model_list = [model.eval() for model in model_list]

        high_confidence_indices = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(dataloader):
                if self.args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    raise Exception("Unknown dataset name!")

                output_tensor_bottom_model_a = model_list[0](Xa)
                output_tensor_bottom_model_b = model_list[1](Xb)
                output = model_list[2](
                    output_tensor_bottom_model_a, output_tensor_bottom_model_b
                )

                probabilities = torch.softmax(outputs, dim=1)
                predicted_probs, predicted_labels = torch.max(probabilities, dim=1)

                # 筛选预测正确且预测概率高的样本
                correct_predictions = predicted_labels == labels
                high_confidence = predicted_probs > confidence_threshold

                # 两个条件同时满足时的样本索引
                selected_indices = (
                    indices[correct_predictions & high_confidence].cpu().numpy()
                )
                high_confidence_indices.extend(selected_indices)
                all_labels.extend(
                    labels[correct_predictions & high_confidence].cpu().numpy()
                )

        # 使用选定的样本索引构建子集
        high_confidence_subset = Subset(dataloader.dataset, high_confidence_indices)

        return high_confidence_subset

    @staticmethod
    def load_vfl_model(args):
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

        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint["state_dict"][i])

        return model_list

    @staticmethod
    def load_train_dataset(args):
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
            trainset = IndexedCIFAR10(
                root=args.data_dir, train=True, download=True, transform=transform
            )
        elif args.dataset == "CIFAR100":
            trainset = IndexedCIFAR100(
                root=args.data_dir, train=True, download=True, transform=transform
            )
        elif args.dataset == "CINIC10L":
            trainset = CINIC10L(root=args.data_dir, split="train", transform=transform)
        else:
            raise ValueError("Not support dataset.")

        train_dataloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        return train_dataloader

    @staticmethod
    def split_data(data, args):
        if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
            x_a = data[:, :, :, 0 : args.half]
            x_b = data[:, :, :, args.half : 32]
        else:
            raise Exception("Unknown dataset name!")

        return x_a, x_b

    @staticmethod
    def shuffle_subdataset():
        pass

    @staticmethod
    def train_shuffle_model():
        pass

    @staticmethod
    def evaluate_defense_method():
        pass
