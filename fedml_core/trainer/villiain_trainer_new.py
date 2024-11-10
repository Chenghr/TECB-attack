import copy
import torch
from torch.utils.data import Subset
from fedml_core.trainer.vfl_trainer import VFLTrainer
import torch.nn.functional as F
import numpy as np
import random

class VillainTrainer(VFLTrainer):
    def trigger_gen_pre(self, train_data, selected_target_indices, device, args):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]
        mask_list = {}
        delta_avg = 0
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            #确定delta值
            indices = indices.cpu().numpy()
            target_mask = torch.from_numpy(np.isin(indices, selected_target_indices))
            embedding_vector = model_list[1](Xb)
            target_embedding_vector = embedding_vector[target_mask].detach()
            std_vector = torch.std(target_embedding_vector, dim=1)
            avg_std = torch.mean(std_vector)
            delta = avg_std
            #确定delta插入位置
            poison_num = int(std_vector.shape[0] * args.poison_budget)
            if poison_num == 0:
                poison_num = 1
            _, indices = torch.topk(std_vector, poison_num)
            mask = torch.zeros(std_vector.shape[0], dtype=torch.bool)
            mask[indices] = True
            target_indices = torch.where(target_mask)[0]
            filtered_indices = target_indices[mask]
            poison_mask = torch.zeros_like(target_mask, dtype=torch.bool)
            poison_mask[filtered_indices] = True
            mask_list[step] = poison_mask
            delta_avg += delta
        delta_avg = delta_avg / step
        return delta_avg, mask_list
    
    def trigger_gen(self, train_data, selected_target_indices, device, args):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]
        mask_list = {}
        delta_avg = 0
        total_steps = 0

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            # 数据预处理
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)

            # 确定目标样本
            indices = indices.cpu().numpy()
            target_mask = torch.from_numpy(np.isin(indices, selected_target_indices))
            
            # 如果这个批次没有目标样本，跳过
            if not torch.any(target_mask):
                mask_list[step] = torch.zeros_like(target_mask, dtype=torch.bool)
                continue

            # 获取嵌入向量
            embedding_vector = model_list[1](Xb)
            target_embedding_vector = embedding_vector[target_mask].detach()

            # 确保有足够的样本来计算标准差
            if target_embedding_vector.shape[0] < 2:
                mask_list[step] = torch.zeros_like(target_mask, dtype=torch.bool)
                continue

            # 计算标准差
            std_vector = torch.std(target_embedding_vector, dim=1, unbiased=True)
            avg_std = torch.mean(std_vector)
            delta = avg_std

            # 确定毒化样本数量
            num_available_samples = std_vector.shape[0]
            poison_num = min(int(num_available_samples * args.poison_budget), num_available_samples)
            poison_num = max(1, poison_num)  # 至少选择1个样本

            # 选择具有最高标准差的样本
            _, indices = torch.topk(std_vector, min(poison_num, num_available_samples))
            
            # 创建毒化掩码
            mask = torch.zeros(std_vector.shape[0], dtype=torch.bool)
            mask[indices] = True
            target_indices = torch.where(target_mask)[0]
            filtered_indices = target_indices[mask]
            poison_mask = torch.zeros_like(target_mask, dtype=torch.bool)
            poison_mask[filtered_indices] = True
            
            mask_list[step] = poison_mask
            delta_avg += delta
            total_steps += 1

        # 避免除以零
        if total_steps > 0:
            delta_avg = delta_avg / total_steps
        else:
            delta_avg = torch.tensor(0.0).to(device)

        return delta_avg, mask_list
    
    def train_with_trigger(self, train_data, delta, mask_list, criterion, bottom_criterion, optimizer_list, device, args):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []
        gamma = random.uniform(args.gamma_up, args.gamma_low)
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
                
            batch_loss = []
            # bottom model B
            output_tensor_bottom_model_b = model_list[1](Xb)
            # bottom model A
            output_tensor_bottom_model_a = model_list[0](Xa)

            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(True)

            # m_vector = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
            dim = output_tensor_bottom_model_b.shape[1]
            m_vector = self.create_m_vector(dim)
            # print(m_vector)
                
            delta_vector = m_vector * delta.item() * args.beta
            delta_vector.requires_grad = False
            mask = mask_list[step]
            mask.requires_grad = False
            true_indices = torch.nonzero(mask).squeeze()
            if true_indices.dim() == 0:
                input_tensor_top_model_b_poison = input_tensor_top_model_b
            else:
                dropout_num = int(args.dropout_ratio * len(true_indices))
                dropout_indices = true_indices[torch.randperm(len(true_indices))[:dropout_num]]
                dropout_mask = np.copy(mask)
                dropout_mask[dropout_indices] = False
                delta_expanded = torch.zeros(dropout_mask.shape[0], 10)
                delta_expanded[dropout_mask] = delta_vector * gamma
                delta_expanded.requires_grad = False
                input_tensor_top_model_b_poison = input_tensor_top_model_b + delta_expanded.to(device)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b_poison)
            # --top model backward/update--
            loss = self.update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = self.update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = self.update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            batch_loss.append(loss.item())
            
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]
    
    def test_backdoor(self, test_data, criterion, delta, target_label, device, args):
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

        # batch_delta = delta.unsqueeze(0).repeat([poison_target_label.shape[0], 1, 1, 1]).detach().clone()

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = self.split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)
                
                # m_vector = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
                dim = output_tensor_bottom_model_b.shape[1]
                m_vector = self.create_m_vector(dim)
                
                delta_vector = m_vector * delta.item() * args.beta
                delta_expended = delta_vector.unsqueeze(0).expand(target.shape[0], -1).to(device)
                output_tensor_bottom_model_b_backdoor = output_tensor_bottom_model_b + delta_expended

                target_class = torch.tensor([target_label]).repeat(target.shape[0]).to(device)
                
                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b_backdoor)

                loss = criterion(output, target_class)
                test_loss += loss.item()
                
                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        return test_loss, top1_acc, top5_acc
    
    @staticmethod
    def create_m_vector(pattern_length=10):
        """创建交替模式的tensor: [1, 1, -1, -1, 1, 1, -1, -1, ...]
    
        Parameters:
        -----------
        length : int
            tensor的长度
            
        Returns:
        --------
        torch.Tensor
            按照模式生成的tensor
        """
        pattern = torch.ones(pattern_length)
        pattern[2::4] = -1  # 每4个数中的第3个设为-1
        pattern[3::4] = -1  # 每4个数中的第4个设为-1
        return pattern