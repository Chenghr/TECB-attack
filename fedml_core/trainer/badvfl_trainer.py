import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import numpy as np

import torch.nn.functional as F

from fedml_core.utils.utils import (
    AverageMeter,
    gradient_masking,
    gradient_gaussian_noise_masking,
    marvell_g,
    backdoor_truepostive_rate,
    gradient_compression,
    laplacian_noise_masking,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

from .vfl_trainer import VFLTrainer, split_data
from itertools import combinations


class BadVFLTrainer(VFLTrainer):
    def pre_train(self, train_data, criterion, optimizer_list, args):
        """假设本地已有标签,先得到本地的特征嵌入以及classifier的梯度,便于后续操作
            这里用的模型与后续训练的不同,操作与train_mlu一致
        """
        model_list = [model.to(args.device).train() for model in self.model]
        
        # train and update
        batch_loss = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(args.device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(args.device)
            else:
                Xa = trn_X[0].float().to(args.device)
                Xb = trn_X[1].float().to(args.device)
                target = trn_y.long().to(args.device)

            output = model_list[-1](Xb)
            
            # --top model backward/update--
            loss = self.update_model_one_batch(
                optimizer=optimizer_list[-1],
                model=model_list[-1],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )
            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss
    
    def select_closest_class_pair(self, dataloader, args):
        """对 labels 中的值进行组队,计算每一对的平均成对距离,选择最小距离的对,作为 source label 和target label 返回。"""
        features, labels = self._extract_features(dataloader, args.device, args)
        
        # 获取所有独特的标签值
        unique_labels = torch.unique(labels)
        
        # 初始化一个很大的值作为最小距离
        min_distance = float('inf')
        source_label = None
        target_label = None
        
        # 遍历每一对独特的标签值
        for label_a in unique_labels:
            for label_b in unique_labels:
                if label_a != label_b:
                    # 获取每个标签对应的特征向量
                    features_a = features[labels == label_a]
                    features_b = features[labels == label_b]
                    
                    # 计算两个特征向量集合之间的平均成对距离
                    pairwise_distances = torch.cdist(features_a, features_b)
                    average_distance = pairwise_distances.mean()
                    
                    # 更新最小距离和对应的标签
                    if average_distance < min_distance:
                        min_distance = average_distance
                        source_label = label_a
                        target_label = label_b
        
        return source_label, target_label
        
    def _extract_features(self, dataloader, args):
        """提取样本的 embedding 和 label"""
        model = self.model[-1].eval().to(args.device)

        features = []
        labels = []
        with torch.no_grad():
            for step, (trn_X, trn_y, indices) in enumerate(dataloader):
                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(args.device)
                    Xa, Xb = self.split_data(trn_X, args)
                    target = trn_y.long().to(args.device)
                else:
                    Xb = trn_X.float().to(args.device)
                    target = torch.argmax(trn_y, dim=1).long().to(args.device)

                output = model(Xb)
                
                features.append(output.cpu().detach())
                labels.append(target.cpu().detach())
                
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels
    
    def find_optimal_trigger_position(self, train_dataloader, source_label, criterion, optimizer_list, args):
        """
        计算基于雅可比矩阵的显著性图,并使用滑动窗口找到具有最大梯度值的位置,作为注入触发器的最佳位置。

        Args:
            train_dataloader (torch.utils.data.DataLoader): 训练数据加载器。
            source_label (int): 源标签,用于筛选出目标样本。
            criterion (torch.nn.Module): 损失函数。
            optimizer_list (List[torch.optim.Optimizer]): 优化器列表。
            args.device (torch.args.device): 设备(CPU或GPU)。
            args (argparse.Namespace): 包含其他超参数和配置的对象。

        Returns:
            best_position (Tuple[int, int]): 注入触发器的滑动窗口的左上角(行索引,列索引)。
        """
        model = self.model[-1].to(args.device).eval()  # 将模型移动到设备上并设置为评估模式

        # 初始化累计梯度
        accumulated_grad = None

        for step, (trn_X, trn_y, indices) in enumerate(train_dataloader):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(args.device)
                Xa, Xb = self.split_data(trn_X, args)  # 将输入分割为Xa和Xb两部分
                target = trn_y.long().to(args.device)
            else:
                Xa = trn_X[0].float().to(args.device)
                Xb = trn_X[1].float().to(args.device)
                target = trn_y.long().to(args.device)

            # 筛选出 label 为 source_label 的样本
            target_mask = (target == source_label)
            Xb_target = Xb[target_mask]
            if Xb_target.size(0) == 0:
                continue  # 如果当前批次没有目标标签样本,则跳过

            output = model(Xb_target)
            loss = criterion(output, target[target_mask])

            # 计算基于雅可比矩阵的显著性图
            loss.backward(retain_graph=True)
            saliency = Xb_target.grad.abs().sum(dim=1, keepdim=True)

            # 累加梯度
            if accumulated_grad is None:
                accumulated_grad = saliency.detach().clone()
            else:
                accumulated_grad += saliency.detach().clone()

            # 重置梯度
            optimizer_list.zero_grad()

        # 使用滑动窗口找到具有最大梯度值的位置
        window_size = args.window_size
        saliency_windows = [accumulated_grad[:, :, i:i+window_size, j:j+window_size] for i in range(accumulated_grad.size(2)-window_size+1) for j in range(accumulated_grad.size(3)-window_size+1)]

        # 计算每个窗口的平均梯度幅值
        avg_saliency_windows = [window.abs().mean() for window in saliency_windows]

        # 找到最大平均梯度幅值及其对应的索引
        _, max_index = torch.max(torch.stack(avg_saliency_windows), dim=0)

        # 根据索引计算最佳位置(左上角)
        max_y, max_x = max_index // (accumulated_grad.size(3)-window_size+1), max_index % (accumulated_grad.size(3)-window_size+1)
        best_position = (max_y, max_x)

        return best_position
    
    def train_trigger(self, selected_dataloader, best_position, delta, trigger_optimizer, args):
        def frobenius_norm(x, y):
            return torch.norm(x - y, p='fro', dim=(1, 2, 3))
        
        device = args.device
        by = best_position[0]
        bx = best_position[1]
        
        model = self.model[-1].train().to(device)
        for param in model.parameters():
            param.requires_grad = False
        delta.requires_grad_(True)
        
        batch_loss = []
        for _, (trn_X, trn_y, indices) in enumerate(selected_dataloader):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            
            # 筛选出 target 为源类别标签的样本
            source_mask = (target == args.source_label)
            Xb_source = Xb[source_mask]
            Xb_target = Xb[~source_mask]
            
            # 构建包含触发器的源样本
            batch_delta = torch.zeros_like(Xb_source).to(device)
            batch_delta[:, :, by:by+args.window_size, bx:bx+args.window_size] = delta.expand(Xb_source.size(0), -1, -1, -1)
            Xb_source_poisoned = Xb_source + batch_delta
            
            # 前向传播
            source_output = model(Xb_source_poisoned)
            target_output = model(Xb_target)
            
            # 计算损失并反向传播
            trigger_optimizer.zero_grad()
            loss = frobenius_norm(source_output, target_output)
            loss = torch.mean(loss)
            loss.backward()
            trigger_optimizer.step()
            
            # 将 delta 限制在指定范围内
            with torch.no_grad():
                delta = torch.clamp(delta, -args.eps, args.eps)
            
            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)
        
        return delta, epoch_loss
    
    def convert_delta(self, dataloader, best_position, delta, args):
        """
        将触发器模式 delta 转换为与输入数据 Xb 形状相同的张量,以便直接相加。

        Args:
            dataloader (torch.utils.data.DataLoader): 用于获取输入数据形状的数据加载器。
            best_position (tuple): 一个包含两个元素的元组,表示触发器模式delta在输入图像中的位置(by, bx)。
            delta (torch.Tensor): 形状为(1, channels, args.window_size, args.window_size)的触发器模式张量。
            args (object): 包含必要参数的对象,如device、dataset和window_size等。

        Returns:
            converted_delta (torch.Tensor): 与输入数据 Xb 形状相同的触发器模式张量。
        """
        device = args.device
        by = best_position[0]
        bx = best_position[1]
        
        # 获取一个批次的数据,以确定输入张量的形状
        batch = next(iter(dataloader))
        if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
            trn_X = batch[0].float().to(device)
            _, Xb = split_data(trn_X, args)
        else:
            Xb = batch[1].float().to(device)
        
        # 创建与 Xb 形状相同的零张量
        converted_delta = torch.zeros_like(Xb).to(device)
        
        # 将 delta 复制到 converted_delta 的指定位置
        batch_size, channels, height, width = Xb.size()
        converted_delta[:, :, by:by+args.window_size, bx:bx+args.window_size] = delta.expand(batch_size, -1, -1, -1)
        
        return converted_delta
    
    def train(self, train_dataloader, criterion, bottom_criterion, optimizer_list, args):
        return super().train(train_dataloader, criterion, bottom_criterion, optimizer_list, args.device, args)
    
    def train_poisoning(self, train_dataloader, criterion, bottom_criterion, optimizer_list, 
                        selected_source_indices, selected_target_indices, delta, args):
        """
        """
        device = args.device
        model_list = [model.train().to(device) for model in self.model]
        
        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_dataloader):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            
            # 如果 indices 中存在 下标在 selected_target_indices, 将对应的 xb 替换为 selected_source_indices 中对应下标的 xb,并执行 xb + delta
            ...
            
            output_tensor_bottom_model_a = model_list[0](Xa)
            output_tensor_bottom_model_b = model_list[1](Xb)
            
            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)
            
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            
            loss = self.update_model_one_batch(
                optimizer=optimizer_list[2],
                model=model_list[2],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )
            
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad
            
            _ = self.update_model_one_batch(
                optimizer=optimizer_list[1],
                model=model_list[1],
                output=output_tensor_bottom_model_b,
                batch_target=grad_output_bottom_model_b,
                loss_func=bottom_criterion,
                args=args,
            )

            # -- bottom model a backward/update--
            _ = self.update_model_one_batch(
                optimizer=optimizer_list[0],
                model=model_list[0],
                output=output_tensor_bottom_model_a,
                batch_target=grad_output_bottom_model_a,
                loss_func=bottom_criterion,
                args=args,
            )

            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss
        
    def test(self, dataloader, criterion, args):
        return super().test(dataloader, criterion, args.device, args)
    
    def test_backdoor(self,):
        ...
        
    def poison_dataset(self, dataloader, selected_source_indices, selected_target_indices, delta, args):
        device = args.device
        dataset = dataloader.dataset
        poisoned_dataset = []

        for i in range(len(dataset)):
            data, target, _ = dataset[i]
            data = data.float().to(device)

            if i in selected_target_indices:
                source_idx = selected_source_indices[selected_target_indices.tolist().index(i)]
                source_data, _, _ = dataset[source_idx]
                source_data = source_data.float().to(device)

                source_data = source_data.unsqueeze(0)  # 添加批次维度
                batch_delta = torch.zeros_like(source_data).to(device)
                batch_delta[:, :, args.by:args.by+args.window_size, args.bx:args.bx+args.window_size] = delta.expand(source_data.size(0), -1, -1, -1)
                poisoned_data = source_data + batch_delta
                poisoned_data = poisoned_data.squeeze(0)

                poisoned_dataset.append((poisoned_data, target))
            else:
                poisoned_dataset.append((data, target))

        return poisoned_dataset
        