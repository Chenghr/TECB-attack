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

from .vfl_trainer import VFLTrainer


def update_model_one_batch(optimizer, model, output, batch_target, loss_func, args):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    # 裁剪梯度,防止梯度爆炸
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss


def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    y_hat_lbls = []
    pred_pos_count = 0
    pred_neg_count = 0
    correct_count = 0
    for y_prob, y_t in zip(y_prob_preds, y_targets):
        if y_prob <= threshold:
            pred_neg_count += 1
            y_hat_lbl = 0
        else:
            pred_pos_count += 1
            y_hat_lbl = 1
        y_hat_lbls.append(y_hat_lbl)
        if y_hat_lbl == y_t:
            correct_count += 1

    return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]


class VanillaTrainer(VFLTrainer):
    def train_mul(
        self, train_data, criterion, bottom_criterion, optimizer_list, device, args
    ):
        """
        正常 VFL 训练，原始训练数据的特征未拆分，在训练函数中进行拆分

        返回：
        - epoch_loss：一个浮点数，表示当前 epoch 的平均损失。
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            X_b = trn_X.float().to(device)
            target = trn_y.long().to(device)

            # bottom model
            output_tensor_bottom_model_b = model_list[0](X_b)
            input_tensor_top_model = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model.requires_grad_(True)

            # top model
            output = model_list[1](input_tensor_top_model)

            # --top model backward/update--
            loss = update_model_one_batch(
                optimizer=optimizer_list[1],
                model=model_list[1],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_b = input_tensor_top_model.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(
                optimizer=optimizer_list[0],
                model=model_list[0],
                output=output_tensor_bottom_model_b,
                batch_target=grad_output_bottom_model_b,
                loss_func=bottom_criterion,
                args=args,
            )

            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss

    def train_narcissus(
        self,
        train_data,
        criterion,
        bottom_criterion,
        optimizer_list,
        device,
        args,
        delta,
        poisoned_indices,
    ):
        """
        对于指定的 poisoned_indices，生成用于添加后门的 delta。

        逻辑：
        1. 从 batch_delta.grad 中提取需要投毒样本的梯度 batch_delta_grad。
        2. 将这些梯度加入 poisoned_grads 列表中。
        3. 计算这些梯度的平均值的符号（sign），得到 grad_sign。
        4. 更新 delta：delta = delta - grad_sign * args.alpha，其中 args.alpha 是学习率。
        5. 将 delta 限制在 [-args.eps, args.eps] 的范围内。

        返回：
        - epoch_loss：一个浮点数，表示当前 epoch 的平均损失。
        - delta：一个 tesnor，形状和输入图像 Xb 一致，表示优化后的后门扰动。
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        batch_loss = []
        poisoned_grads = []

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            X_b = trn_X.float().to(device)
            target = trn_y.long().to(device)
            indices = indices.cpu().numpy()

            batch_delta = torch.zeros_like(X_b).to(device)
            # mask标记哪些是要毒害的
            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            batch_delta[mask] = delta.detach().clone()
            batch_delta.requires_grad_()

            # bottom model
            output_tensor_bottom_model_b = model_list[0](X_b + batch_delta)
            input_tensor_top_model = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model.requires_grad_(True)

            # top model
            output = model_list[1](input_tensor_top_model)

            # --top model backward/update--
            loss = update_model_one_batch(
                optimizer=optimizer_list[1],
                model=model_list[1],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_b = input_tensor_top_model.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(
                optimizer=optimizer_list[0],
                model=model_list[0],
                output=output_tensor_bottom_model_b,
                batch_target=grad_output_bottom_model_b,
                loss_func=bottom_criterion,
                args=args,
            )

            batch_loss.append(loss.item())

            batch_delta_grad = batch_delta.grad[mask]
            poisoned_grads.append(batch_delta_grad)
            grad_sign = batch_delta_grad.detach().mean(dim=0).sign()

            delta = delta - grad_sign * args.alpha
            delta = torch.clamp(delta, -args.eps, args.eps)

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss, delta

    def train_poisoning(
        self,
        train_data,
        criterion,
        bottom_criterion,
        optimizer_list,
        device,
        args,
        delta,
        poisoned_indices,
    ):
        """
        基于 delta 对模型植入后门。原始训练数据的特征未拆分，在训练函数中进行拆分。

        参数：
        - train_data：训练数据加载器，包含特征、标签和索引。
        - criterion：损失函数，用于顶部模型。
        - bottom_criterion：损失函数，用于底部模型。
        - optimizer_list：优化器列表，对应每个模型的优化器。
        - device：设备（CPU 或 GPU），用于训练。
        - args：其他训练参数。
        - delta：用于投毒的扰动。
        - poisoned_indices：投毒样本的索引。

        返回：
        - epoch_loss：一个浮点数，表示当前 epoch 的平均损失。
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            X_b = trn_X.float().to(device)
            target = trn_y.long().to(device)

            indices = indices.cpu().numpy()
            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            num_masked = mask.sum().item()

            non_mask_indices = torch.where(~mask)[0]  # 找到非mask的索引
            random_indices = torch.randperm(non_mask_indices.size(0))[
                :num_masked
            ]  # 从非mask的索引中随机选择5个

            X_b[
                non_mask_indices[random_indices]
            ] += delta.detach().clone()  # 将delta添加到选定的Xb值

            # bottom model
            output_tensor_bottom_model_b = model_list[0](X_b)
            input_tensor_top_model = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model.requires_grad_(True)

            # top model
            output = model_list[1](input_tensor_top_model)

            # --top model backward/update--
            loss = update_model_one_batch(
                optimizer=optimizer_list[1],
                model=model_list[1],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_b = input_tensor_top_model.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(
                optimizer=optimizer_list[0],
                model=model_list[0],
                output=output_tensor_bottom_model_b,
                batch_target=grad_output_bottom_model_b,
                loss_func=bottom_criterion,
                args=args,
            )

            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss

    def train_shuffle(
        self, train_data, criterion, bottom_criterion, optimizer_list, device, args
    ):
        model_list = [model.to(device).train() for model in self.model]

        # train and update
        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            X_b = trn_X.float().to(device)
            target = trn_y.long().to(device)

            # bottom model
            output_tensor_bottom_model_b = model_list[0](X_b)
            input_tensor_top_model = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model.requires_grad_(True)

            # top model
            output = model_list[1](input_tensor_top_model)

            # --top model backward/update--
            loss = update_model_one_batch(
                optimizer=optimizer_list[1],
                model=model_list[1],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_b = input_tensor_top_model.grad

            # -- bottom model b backward/update--
            if args.train_bottom_model_b:
                _ = update_model_one_batch(
                    optimizer=optimizer_list[0],
                    model=model_list[0],
                    output=output_tensor_bottom_model_b,
                    batch_target=grad_output_bottom_model_b,
                    loss_func=bottom_criterion,
                    args=args,
                )

            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss

    def test_mul(self, test_data, criterion, device, args):
        model_list = [model.to(device).eval() for model in self.model]

        batch_loss, top5_correct, correct = [], 0, 0
        total = len(test_data.dataset)
        with torch.no_grad():
            for trn_X, trn_y, _ in test_data:
                X_b = trn_X.float().to(device)
                target = trn_y.long().to(device)

                output = model_list[1](model_list[0](X_b))
                loss = criterion(output, target)
                batch_loss.append(loss.item())

                probs = F.softmax(output, dim=1)

                correct += (probs.argmax(dim=1) == target).sum().item()
                top5_correct += (
                    (probs.topk(5, dim=1)[1] == target.view(-1, 1)).sum().item()
                )

        test_loss = sum(batch_loss) / len(batch_loss)
        top1_acc = 100.0 * correct / total
        top5_acc = 100.0 * top5_correct / total

        return test_loss, top1_acc, top5_acc

    def test_backdoor_mul(
        self, test_data, criterion, device, args, delta, poison_target_label
    ):
        """
        测试模型在添加后门攻击样本情况下的表现（ASR）。
        """
        model_list = [model.to(device).eval() for model in self.model]

        batch_loss, top5_correct, correct = [], 0, 0
        total = len(test_data.dataset)
        with torch.no_grad():
            for trn_X, trn_y, _ in test_data:
                X_b = trn_X.float().to(device)
                target = trn_y.long().to(device)
                target = (
                    torch.tensor([poison_target_label])
                    .repeat(target.shape[0])
                    .to(device)
                )

                output = model_list[1](model_list[0](X_b + delta))
                loss = criterion(output, target)
                batch_loss.append(loss.item())

                probs = F.softmax(output, dim=1)

                correct += (probs.argmax(dim=1) == target).sum().float().item()
                top5_correct += (
                    (probs.topk(5, dim=1)[1] == target.view(-1, 1)).sum().float().item()
                )

        test_loss = sum(batch_loss) / len(batch_loss)
        top1_acc = 100.0 * correct / total
        top5_acc = 100.0 * top5_correct / total

        return test_loss, top1_acc, top5_acc
