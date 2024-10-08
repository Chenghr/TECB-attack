import copy
import random

import numpy as np
import torch
from fedml_core.data_preprocessing.cifar10 import SelectedIndexedCIFAR10
from fedml_core.trainer.vfl_trainer import VFLTrainer
from torch.utils.data import Subset


class DefenseTrainer(VFLTrainer):
    def select_top_confidence_subset_by_class(
        self, dataloader, device, args, top_fraction=0.2, verbose=False
    ):
        """
        从训练集中每类标签中选择预测概率最接近真实标签的前top_fraction比例的样本
        Args:
            dataloader: 训练数据加载器
            top_fraction: 每类标签中选择的样本比例

        Returns:
            top_confidence_subset: 经过筛选后的高置信度子集
        """
        model_list = self.model
        model_list = [model.to(device).eval() for model in model_list]

        label_confidences = {}
        label_indices = {}

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(dataloader):
                if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = self.split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    raise Exception("Unknown dataset name!")

                output_tensor_bottom_model_a = model_list[0](Xa)
                output_tensor_bottom_model_b = model_list[1](Xb)
                output = model_list[2](
                    output_tensor_bottom_model_a, output_tensor_bottom_model_b
                )

                probabilities = torch.softmax(output, dim=1)
                predicted_probs, predicted_labels = torch.max(probabilities, dim=1)

                # 遍历每个样本，记录预测概率和真实标签
                for i in range(len(target)):
                    label = target[i].item()
                    confidence = predicted_probs[i].item()
                    index = indices[i].item()

                    if label not in label_confidences:
                        label_confidences[label] = []
                        label_indices[label] = []

                    label_confidences[label].append(confidence)
                    label_indices[label].append(index)

        # 选择每类标签中预测概率最高的前top_fraction比例的样本
        selected_indices = []
        for label in label_confidences:
            confidences = label_confidences[label]
            indices = label_indices[label]

            # 按照置信度排序
            sorted_confidences_indices = sorted(
                range(len(confidences)), key=lambda i: confidences[i], reverse=True
            )

            # 选取前top_fraction比例的样本
            num_to_select = max(1, int(len(sorted_confidences_indices) * top_fraction))
            selected_indices.extend(
                [indices[i] for i in sorted_confidences_indices[:num_to_select]]
            )

            if verbose:
                # 打印每类前5个预测概率最高的样本及其对应的概率
                print(f"Label {label}:")
                for i in sorted_confidences_indices[:5]:  # 只输出前5个样本
                    print(f"  Sample index: {indices[i]}, Confidence: {confidences[i]}")
                print("......")
                for i in sorted_confidences_indices[-5:]:  # 只输出最后5个样本
                    print(f"  Sample index: {indices[i]}, Confidence: {confidences[i]}")

        # 使用选定的样本索引构建子集
        # top_confidence_subset = Subset(dataloader.dataset, selected_indices)
        top_confidence_subset = SelectedIndexedCIFAR10(
            dataloader.dataset, selected_indices
        )

        return top_confidence_subset

    def create_label_perturbed_subset(self, selected_dataloader, device, args):
        """
        基于高置信度子集构建标签扰乱后的子集，将每个类的标签替换为与其最近的且未被使用的类，并输出替换信息。

        Args:
            origin_dataloader: 原始数据集
            selected_dataloader: 高置信度子集
            model: 训练好的模型，用于预测
            device: 模型运行的设备
            args: 额外参数，例如数据集名称

        Returns:
            perturbed_dataset: 标签扰乱后的子集
        """
        model_list = [model.to(device).eval() for model in self.model]

        output_by_class = {}  # 存储每个类的输出结果
        perturbed_data = []
        used_classes = set()  # 记录已经使用的类

        # 计算每个样本的输出并分类
        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(selected_dataloader):
                if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = self.split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    raise Exception("Unknown dataset name!")

                # 获取每个模型的输出
                output_tensor_bottom_model_a = model_list[0](Xa)
                output_tensor_bottom_model_b = model_list[1](Xb)
                output = model_list[2](
                    output_tensor_bottom_model_a, output_tensor_bottom_model_b
                )

                probabilities = torch.softmax(output, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)

                for i in range(predicted_labels.shape[0]):
                    predicted_label = predicted_labels[i].item()
                    if predicted_label not in output_by_class:
                        output_by_class[predicted_label] = []
                    output_by_class[predicted_label].append(output[i].squeeze())

                perturbed_data.extend(
                    [
                        (trn_X[i].cpu(), trn_y[i].cpu(), indices[i].cpu())
                        for i in range(len(trn_X))
                    ]
                )  # 保存原始数据和标签

        # 计算类别之间的距离并找到每个类最近的不同类
        nearest_classes_map, class_distances = self.calculate_class_distances(
            output_by_class
        )

        # 构建标签映射关系
        label_mapping = {}
        for origin_label in list(nearest_classes_map.keys()):
            for nearest_class, distance in nearest_classes_map[origin_label]:
                if nearest_class not in used_classes:
                    new_label = nearest_class
                    used_classes.add(new_label)  # 标记该类已使用
                    # 输出替换信息
                    print(
                        f"Class {origin_label} is replaced by class {new_label} (distance: {distance:.4f})"
                    )
                    break
            label_mapping[origin_label] = new_label  # 使用距离最近的未被使用过的类替换标签

        # selected_dataloader.dataset 是 SelectedIndexedCIFAR10 类，支持标签映射
        perturbed_dataset = copy.deepcopy(selected_dataloader.dataset)
        perturbed_dataset.set_label_mapping(label_mapping)

        return perturbed_dataset

    def train_perturbed(
        self, train_data, criterion, bottom_criterion, optimizer_list, device, args
    ):
        """仅更新主动方的模型，即第一个和最后一个模型"""
        model_list = [model.to(device).train() for model in self.perturbed_models]

        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)

            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(False)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss = self.update_model_one_batch(
                optimizer=optimizer_list[2],
                model=model_list[2],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_a = input_tensor_top_model_a.grad

            # -- bottom model a (active party) backward/update--
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

    def test_perturbed(self, test_data, criterion, device, args):
        self.perturbed_models_trainer.update_model(self.perturbed_models)
        test_loss, top1_acc, top5_acc = self.perturbed_models_trainer.test(
            test_data, criterion, device, args
        )
        return test_loss, top1_acc, top5_acc

    def predict_with_defense(
        self,
        test_data,
        criterion,
        device,
        args,
        delta,
        poison_indices,
        gap_threshold=0.1,
    ):
        """
        对dataloader中的样本进行批量预测。每个样本使用model和perturbed_models进行预测。
        如果两个模型的置信度分数之差大于指定阈值，则该样本预测为-1，否则为model的预测结果。

        参数:
        test_data -- 传入的DataLoader，包含待预测样本
        criterion -- 损失函数（未使用，但保留作为接口参数）
        device -- 用于计算的设备（'cpu' 或 'cuda'）
        args -- 额外的参数信息
        delta -- trigger 值, tensor, 形状和 x_b 一致
        poison_indices -- 随机投毒样本下标
        gap_threshold -- 置信度差异的阈值，默认为0.1

        返回:
        predictions -- 每个样本的预测结果列表
        is_clean -- 每个样本是否干净的标签列表, 内部元素为 bool
        """
        predictions, is_clean = [], []

        # 将模型放入指定设备
        model_list = [model.to(device).eval() for model in self.model]
        perturbed_model_list = [
            model.to(device).eval() for model in self.perturbed_models
        ]

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):
                if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = self.split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                # 对投毒数据添加 trigger
                indices = indices.cpu().numpy()
                mask = torch.from_numpy(np.isin(indices, poison_indices))
                batch_delta = torch.zeros_like(Xb).to(device)
                batch_delta[mask] = delta.detach().clone()
                Xb += batch_delta

                batch_is_clean = ~mask  # 如果 mask 为 True，表示是 poison 样本，取反即可得到清洁样本

                # 获取原始模型和扰动模型的预测结果
                original_model_output = model_list[2](
                    model_list[0](Xa), model_list[1](Xb)
                )
                perturbed_model_output = perturbed_model_list[2](
                    perturbed_model_list[0](Xa), perturbed_model_list[1](Xb)
                )

                # 将输出转为softmax概率
                original_probs = torch.softmax(original_model_output, dim=1)
                perturbed_probs = torch.softmax(perturbed_model_output, dim=1)

                # 获取每个样本的最大置信度及其对应类别
                original_confidences, original_preds = torch.max(original_probs, dim=1)
                perturbed_confidences, _ = torch.max(perturbed_probs, dim=1)

                # 计算置信度差值
                confidence_diff = torch.abs(
                    original_confidences - perturbed_confidences
                )

                # 根据置信度差值与阈值的比较，生成最终预测结果
                batch_predictions = torch.where(
                    confidence_diff > gap_threshold, -1, original_preds
                )

                predictions.extend(batch_predictions.cpu().numpy())  # 将预测结果放入列表
                is_clean.extend(batch_is_clean.cpu().numpy())  # 将clean标签放入列表

        return predictions, is_clean

    def create_perturbed_models(self, args):
        model_list = copy.deepcopy(self.model)
        self.perturbed_models = model_list
        self.perturbed_models_trainer = VFLTrainer(model_list)

    @staticmethod
    def calculate_class_distances(output_by_class):
        """
        计算类别之间的距离，并输出结果。

        Args:
            output_by_class: 一个字典，键是类别标签，值是对应类别的输出张量列表

        Returns:
            nearest_classes_map: 每个类到其最近类的映射字典
            class_distances: 每对类之间的距离字典
        """
        class_centroids = {}
        for label, outputs in output_by_class.items():
            class_centroids[label] = torch.mean(
                torch.stack(outputs), dim=0
            )  # 计算每个类别的输出均值

        # 初始化类别最近邻映射
        nearest_classes_map = {}
        class_distances = {}
        labels = list(class_centroids.keys())

        # 计算类别之间的距离
        for i in range(len(labels)):
            nearest_classes = []
            distances = []
            for j in range(len(labels)):
                if i != j:
                    dist = torch.dist(
                        class_centroids[labels[i]], class_centroids[labels[j]]
                    )  # 欧氏距离
                    class_distances[(labels[i], labels[j])] = dist
                    nearest_classes.append((labels[j], dist))

            # 按距离排序
            nearest_classes.sort(key=lambda x: x[1])
            nearest_classes_map[labels[i]] = nearest_classes  # 储存所有距离最小的类

        return nearest_classes_map, class_distances

    @staticmethod
    def create_poison_index(test_dataloader, target_class, poison_ratio):
        """
        根据 poison_ratio，在非 target_class 的样本中生成对应比例的随机样本的下标，作为后续 poison sample。

        Args:
            test_data: DataLoader，返回的数据为 (img, target, index)。
            target_class: int，目标类别，不对其进行中毒。
            poison_ratio: float，poison样本的比例 (0.0 - 1.0)。

        Returns:
            List[int]，被选中用于 poison 的样本下标。
        """
        # 存储所有非 target_class 的样本下标
        non_target_indices = []

        # 遍历 DataLoader，找出所有非 target_class 的样本下标
        for trn_X, trn_y, indices in test_dataloader:
            for target, index in zip(trn_y, indices):
                if target.item() != target_class:
                    non_target_indices.append(index.item())

        # 根据 poison_ratio 随机选择要 poison 的样本数量
        poison_sample_count = int(len(non_target_indices) * poison_ratio)

        # 随机选择 poison_sample_count 个样本下标
        poison_indices = (
            random.sample(non_target_indices, poison_sample_count)
            if poison_sample_count > 0
            else []
        )

        return poison_indices

    @staticmethod
    def evaluate_interception_metrics(predictions, is_clean):
        """
        评估拦截的各类指标。计算精确率、召回率、F1-score 和特异性。

        参数:
        predictions -- 预测结果列表，其中值为 -1 表示被拦截的样本
        is_clean -- 对应样本的干净标签列表，True 表示干净样本，False 表示带有后门的样本

        返回:
        metrics -- 一个包含各类指标的字典
        """
        assert len(predictions) == len(is_clean), "Predictions和is_clean长度不匹配"

        # 初始化计数
        true_positives = 0  # 实际带有后门，且成功拦截
        false_positives = 0  # 实际干净，但被拦截
        false_negatives = 0  # 实际带有后门，但没有拦截
        true_negatives = 0  # 实际干净，且没有拦截

        for pred, clean_label in zip(predictions, is_clean):
            if pred == -1:  # 预测为拦截
                if not clean_label:  # 实际为带有后门的样本 (clean_label 为 False)
                    true_positives += 1
                else:  # 实际为干净样本 (clean_label 为 True)
                    false_positives += 1
            else:  # 预测为正常
                if not clean_label:  # 实际为带有后门的样本
                    false_negatives += 1
                else:  # 实际为干净样本
                    true_negatives += 1

        # 计算指标
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0.0
        )
        acc = (
            (true_positives + true_negatives) / len(predictions)
            if len(predictions) > 0
            else 0.0
        )

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Specificity": specificity,
            "Accuracy": acc,
        }

        return metrics
