import copy
import torch
from torch.utils.data import Subset
from fedml_core.trainer.vfl_trainer_old import VFLTrainer, split_data, update_model_one_batch
import torch.nn.functional as F
import numpy as np
import random

class VillainTrainer(VFLTrainer):
    def trigger_gen(self, train_data, selected_target_indices, device, args):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]
        mask_list = {}
        delta_avg = 0
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
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
                Xa, Xb = split_data(trn_X, args)
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

            m_vector = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
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
            loss = update_model_one_batch(optimizer=optimizer_list[2],
                                          model=model_list[2],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            grad_output_bottom_model_b = input_tensor_top_model_b.grad

            # -- bottom model b backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[1],
                                       model=model_list[1],
                                       output=output_tensor_bottom_model_b,
                                       batch_target=grad_output_bottom_model_b,
                                       loss_func=bottom_criterion,
                                       args=args)

            # -- bottom model a backward/update--
            _ = update_model_one_batch(optimizer=optimizer_list[0],
                                       model=model_list[0],
                                       output=output_tensor_bottom_model_a,
                                       batch_target=grad_output_bottom_model_a,
                                       loss_func=bottom_criterion,
                                       args=args)

            batch_loss.append(loss.item())
            
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]
    
    def test_backdoor(self, test_data, delta, target_label, device, args):
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
                    Xa, Xb = split_data(trn_X, args)
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
                m_vector = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
                delta_vector = m_vector * delta.item() * args.beta
                delta_expended = delta_vector.unsqueeze(0).expand(target.shape[0], -1).to(device)
                output_tensor_bottom_model_b_backdoor = output_tensor_bottom_model_b + delta_expended

                target_class = torch.tensor([target_label]).repeat(target.shape[0]).to(device)
                
                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b_backdoor)

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target_class.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        # print(
        #     'Backdoor Test set: Average loss: {:.4f}, ASR Top-1 Accuracy: {}/{} ({:.4f}%, ASR Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
        #         test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return top1_acc, top5_acc
