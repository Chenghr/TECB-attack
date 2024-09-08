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
    def extract_features(self, train_data, device, args):
        """
        从训练数据集中提取特征和标签。
        
        返回:
        - features: 从被动方模型中提取的样本的embedding,形状为 (num_samples, feature_dim)。
        - labels: 所有样本对应的标签，形状为 (num_samples,)。
        """
        victim_model = self.model[-1].eval().to(device)

        features = []
        labels = []
        with torch.no_grad():
            for step, (trn_X, trn_y, indices) in enumerate(train_data):
                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    # trn_X = [x.float().to(device) for x in trn_X]
                    Xb = trn_X.float().to(device)
                    target = torch.argmax(trn_y, dim=1).long().to(device)
                # target = trn_y.float().to(device)
                batch_loss = []

                # bottom model B
                output_tensor_bottom_model_b = victim_model(Xb)
                features.append(output_tensor_bottom_model_b.detach().cpu().numpy())
                labels.append(target.detach().cpu().numpy())

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def pre_train(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):
        """
        假设本地已有标签,先得到本地的特征嵌入以及classifier的梯度,便于后续操作
        这里用的模型与后续训练的不同,操作与train_mlu一致
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

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

            # top model
            output = model_list[-1](Xb)
            # --top model backward/update--
            loss = update_model_one_batch(optimizer=optimizer_list[-1],
                                          model=model_list[-1],
                                          output=output,
                                          batch_target=target,
                                          loss_func=criterion,
                                          args=args)

            batch_loss.append(loss.item())
            
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return epoch_loss[0]

    def pairwise_distance_min(self, train_data, device, args):
        features, labels = self.extract_features(train_data, device, args)
        #cifar-10
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_pairs = list(combinations(classes, 2))
        pairwise_distances_dict = {}
        for (class_a, class_b) in class_pairs:
            class_a_embeddings = features[labels == classes.index(class_a)]
            class_b_embeddings = features[labels == classes.index(class_b)]
            distances = np.abs(class_a_embeddings - class_b_embeddings)
    
            avg_distance = distances.mean()
            pairwise_distances_dict[(class_a, class_b)] = avg_distance
        min_distance_pair = min(pairwise_distances_dict, key=pairwise_distances_dict.get)
        return min_distance_pair[0], min_distance_pair[1]

    def optimal_trigger_injection(self, train_data, selected_source_indices, criterion, optimizer_list, device, args):
        '''
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        target_num = np.sum(labels == classes.index(target_class))
        source_num = np.sum(labels == classes.index(source_class))
        poison_num = args.poison_budget * target_num
        source_indices = np.where(labels == classes.index(source_class))[0]
        selected_indices = np.random.choice(source_indices, poison_num, replace=False)
        '''
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]
        saliency_map_dict = {}
        for y in range(32 - args.window_size + 1):
                for x in range(16 - args.window_size + 1):
                    saliency_map_dict[(y, x)] = 0

        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            # top model
            indices = indices.cpu().numpy()
            source_mask = torch.from_numpy(np.isin(indices, selected_source_indices))
            Xb_source = Xb[source_mask].detach().clone()
            target_source = target[source_mask].detach().clone()
            best_pos_dict = {}
            for i in range(500):
                Xb_eval = Xb_source[i].detach().clone()
                Xb_eval.requires_grad_(True)
                output = model_list[-1](Xb_eval.unsqueeze(0))
                # --top model backward/update--
                loss = criterion(output, target_source[i].unsqueeze(0))
                optimizer_list[-1].zero_grad()
                loss.backward()

                saliency_map = Xb_eval.grad.data.abs()
                _, H, W = saliency_map.size()
                max_avg_grad = 0
                best_position = (0, 0)
        
                for y in range(H - args.window_size + 1):
                    for x in range(W - args.window_size + 1):
                        window_grad = saliency_map[:, y:y + args.window_size, x:x + args.window_size]
                        saliency_map_dict[(y, x)] += window_grad.mean()

                        if saliency_map_dict[(y, x)] > max_avg_grad:
                            max_avg_grad = saliency_map_dict[(y, x)]
                            best_position = (y, x)

                best_pos_dict[i] = best_position
        return best_pos_dict

    def train_trigger(self, train_data, device, selected_source_indices, selected_target_indices, delta, best_pos_dict, optimizer, args):
        victim_model = self.model[-1].train().to(device)
        modules = list(victim_model.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        for param in feature_extractor.parameters():
            param.requires_grad = False
        delta.requires_grad_(True)
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                trn_X = trn_X.float().to(device)
                Xa, Xb = split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            indices = indices.cpu().numpy()
            source_mask = torch.from_numpy(np.isin(indices, selected_source_indices))
            target_mask = torch.from_numpy(np.isin(indices, selected_target_indices))
            Xb_source = Xb[source_mask].detach()            
            Xb_target = Xb[target_mask].detach()
            total_loss = 0
            for i in range(500):
                by = best_pos_dict[i][0]
                bx = best_pos_dict[i][1]
                batch_delta = torch.zeros_like(Xb_source[i]).to(device)
                batch_delta[:, by : by + args.window_size, bx : bx + args.window_size] = delta[i].clone()
                batch_delta.requires_grad_(True)
                source_features = feature_extractor((Xb_source[i] + batch_delta).unsqueeze(0))
                target_features = feature_extractor((Xb_target[i]).unsqueeze(0))
                optimizer.zero_grad()
                loss = torch.norm(source_features - target_features, p = 'fro') ** 2
                total_loss += loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    delta[i] = torch.clamp(delta[i], -args.eps, args.eps)
            print(total_loss/500)
        return delta

    def train_mul(self, train_data, criterion, bottom_criterion, optimizer_list, device, args):
        """
        正常 VFL 训练，原始训练数据的特征未拆分，在训练函数中进行拆分

        返回：
        - epoch_loss：一个浮点数，表示当前 epoch 的平均损失。
        """
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.train() for model in model_list]

        # train and update
        epoch_loss = []

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

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
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
    
    def test_backdoor_mul(self, test_data, criterion, device, args, delta, poison_target_label):
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

                target_class = torch.tensor([poison_target_label]).repeat(target.shape[0]).to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

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
                top5_correct += torch.eq(top5_preds, target_class.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        # print(
        #     'Backdoor Test set: Average loss: {:.4f}, ASR Top-1 Accuracy: {}/{} ({:.4f}%, ASR Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
        #         test_loss, correct, total, top1_acc, top5_correct, total, top5_acc))

        return test_loss, top1_acc, top5_acc
