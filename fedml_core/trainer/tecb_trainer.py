import numpy as np
import torch

# import wandb
import torch.nn.functional as F
from fedml_core.trainer.vfl_trainer import VFLTrainer

# from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, backdoor_truepostive_rate, apply_noise_patch, gradient_compression, laplacian_noise_masking
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


class TECBTrainer(VFLTrainer):
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
        batch_poisoned_count = 0

        selected_set = set(poisoned_indices)

        # delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)

            indices = indices.cpu().numpy()

            batch_delta = torch.zeros_like(Xb).to(device)
            # mask标记哪些是要毒害的
            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            batch_delta[mask] = delta.detach().clone()
            batch_delta.requires_grad_()

            # batch_delta.grad.zero_()
            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb + batch_delta)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            # corruption_amp
            # output_tensor_bottom_model_a[mask]*=args.corruption_amp

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

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

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(
                    grad_output_bottom_model_b
                )
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(
                    grad_output_bottom_model_b, ratio=args.iso_ratio
                )
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(
                    grad_output_bottom_model_b, target
                )

            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(
                    grad_output_bottom_model_b, preserved_perc=args.gc_ratio
                )

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(
                    grad_output_bottom_model_b, beta=args.lap_noise_ratio
                )

            # sign SGD
            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
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

            # UAB attack
            batch_delta_grad = batch_delta.grad[mask]
            poisoned_grads.append(batch_delta_grad)
            grad_sign = batch_delta_grad.detach().mean(dim=0).sign()

            delta = delta - grad_sign * args.alpha

            delta = torch.clamp(delta, -args.eps, args.eps)

            batch_loss.append(loss.item())

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
        poisoned_grads = []
        batch_poisoned_count = 0

        selected_set = set(poisoned_indices)

        # delta = torch.clamp(delta, -args.eps*2, args.eps*2)

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                # trn_X = [x.float().to(device) for x in trn_X]
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # target = trn_y.float().to(device)

            indices = indices.cpu().numpy()

            mask = torch.from_numpy(np.isin(indices, poisoned_indices))
            num_masked = mask.sum().item()

            # 假设Xb是你的数据张量，mask是你的掩码张量
            non_mask_indices = torch.where(~mask)[0]  # 找到非mask的索引
            random_indices = torch.randperm(non_mask_indices.size(0))[
                :num_masked
            ]  # 从非mask的索引中随机选择5个

            Xb[
                non_mask_indices[random_indices]
            ] += delta.detach().clone()  # 将delta添加到选定的Xb值

            # batch_delta.grad.zero_()
            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()

            # random-valued vector to replace the gradient of poisoned data
            random_grad = torch.randn(num_masked, input_tensor_top_model_b.shape[1]).to(
                device
            )
            input_tensor_top_model_b[non_mask_indices[random_indices]] = random_grad

            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

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

            # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # Target Label replacement
            # 将grad_output_bottom_model_b中按mask中为True的元素索引赋值给non_mask_indices[random_indices]中的元素
            grad_output_bottom_model_b[non_mask_indices[random_indices]] = (
                args.corruption_amp * grad_output_bottom_model_b[mask]
            )

            if args.max_norm:
                grad_output_bottom_model_b = gradient_masking(
                    grad_output_bottom_model_b
                )
            # add iso gaussian noise
            if args.iso:
                grad_output_bottom_model_b = gradient_gaussian_noise_masking(
                    grad_output_bottom_model_b, ratio=args.iso_ratio
                )
            # add marvell noise
            if args.marvell:
                grad_output_bottom_model_b = marvell_g(
                    grad_output_bottom_model_b, target
                )

            # gradient compression
            if args.gc:
                grad_output_bottom_model_b = gradient_compression(
                    grad_output_bottom_model_b, preserved_perc=args.gc_ratio
                )

            # differential privacy
            if args.lap_noise:
                grad_output_bottom_model_b = laplacian_noise_masking(
                    grad_output_bottom_model_b, beta=args.lap_noise_ratio
                )

            if args.signSGD:
                torch.sign(grad_output_bottom_model_b, out=grad_output_bottom_model_b)

            # -- bottom model b backward/update--
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

    def test_backdoor_mul(
        self, test_data, criterion, device, args, delta, poison_target_label
    ):
        """
        测试模型在添加后门攻击样本情况下的表现（ASR）。
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

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

                target_class = (
                    torch.tensor([poison_target_label])
                    .repeat(target.shape[0])
                    .to(device)
                )

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb + delta)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](
                    output_tensor_bottom_model_a, output_tensor_bottom_model_b
                )

                # update here.
                loss = criterion(output, target_class)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += (
                    torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()
                )

        test_loss = test_loss / total
        asr_top1_acc = 100.0 * correct / total
        asr_top5_acc = 100.0 * top5_correct / total

        return test_loss, asr_top1_acc, asr_top5_acc
