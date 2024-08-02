import torch
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report, \
    confusion_matrix, roc_curve
import numpy as np
# import wandb
import torch.nn.functional as F
# from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, backdoor_truepostive_rate, apply_noise_patch, gradient_compression, laplacian_noise_masking
from fedml_core.utils.utils import AverageMeter, gradient_masking, gradient_gaussian_noise_masking, marvell_g, \
    backdoor_truepostive_rate, gradient_compression, laplacian_noise_masking
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize

from .model_trainer import ModelTrainer


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


def split_data(data, args):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b

def split_data_and_add_trigger(data, trigger_value=1, trigger_size=5, args=None):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
        x_b[:, :trigger_size, :trigger_size] = trigger_value
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b


class VFLTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def update_model(self, new_model):
        # 更新self.model
        self.model = new_model

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
    
    def test_mul(self, test_data, criterion, device, args):
        model_list = self.model
        model_list = [model.to(device) for model in model_list]
        model_list = [model.eval() for model in model_list]

        test_loss = 0
        top5_correct = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (trn_X, trn_y, indices) in enumerate(test_data):

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    Xa, Xb = split_data(trn_X, args)
                    target = trn_y.long().to(device)
                else:
                    Xa = trn_X[0].float().to(device)
                    Xb = trn_X[1].float().to(device)
                    target = trn_y.long().to(device)

                # bottom model B
                output_tensor_bottom_model_b = model_list[1](Xb)
                # bottom model A
                output_tensor_bottom_model_a = model_list[0](Xa)

                # top model
                output = model_list[2](output_tensor_bottom_model_a, output_tensor_bottom_model_b)

                loss = criterion(output, target)
                test_loss += loss.item()  # sum up batch loss

                probs = F.softmax(output, dim=1)
                # Top-1 accuracy
                total += target.size(0)
                _, pred = probs.topk(1, 1, True, True)
                correct += torch.eq(pred, target.view(-1, 1)).sum().float().item()

                # Top-5 accuracy
                _, top5_preds = probs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_preds, target.view(-1, 1)).sum().float().item()

        test_loss = test_loss / total
        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total

        return test_loss, top1_acc, top5_acc

    def test_backdoor_mul(self, test_data, criterion, device, args, poison_target_label):
        """测试后门数据上的 data 和 trigger. 
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

                if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
                    trn_X = trn_X.float().to(device)
                    # add trigger here.
                    Xa, Xb = split_data_and_add_trigger(data=trn_X, args=args)
                    target = trn_y.long().to(device)
                else:
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
        top1_acc = 1.0 * correct / total
        top5_acc = 1.0 * top5_correct / total

        return test_loss, top1_acc, top5_acc

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
    
    def train(self, train_data, device, args=None):
        return False

    def test(self, test_data, device, args=None):
        return False 
