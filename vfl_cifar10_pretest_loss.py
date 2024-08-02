import copy
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# from fedml_core.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_two_party_data
from fedml_core.data_preprocessing.cifar10.dataset import IndexedCIFAR10
from fedml_core.model.baseline.vfl_models import BottomModelForCifar10, TopModelForCifar10
from fedml_core.trainer.vfl_pretest_trainer import VFLTrainer
from fedml_core.utils.utils import AverageMeter, keep_predict_loss, over_write_args_from_file
from fedml_core.utils.logger import setup_logger
from fedml_core.utils.metrics import ShuffleTrainMetric

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import logging
import copy
import random


class Metric(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_loss = []
        self.clean_loss = []
        self.poison_loss = []
        self.asr = []

    def update(self, train_loss=None, clean_loss=None, poison_loss=None, asr=None,
        ):
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if clean_loss is not None:
            self.clean_loss.append(clean_loss)
        if poison_loss is not None:
            self.poison_loss.append(poison_loss)
        if asr is not None:
            self.asr.append(asr)   


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_trigger_to_xb(data, indices, trigger_value=1, trigger_size=5, args=None):
    for idx in indices:
        x_b = data[idx, :, :, args.half:32]
        x_b[:, :trigger_size, :trigger_size] = trigger_value
        data[idx, :, :, args.half:32] = x_b
    return data


def set_dataset(args):
    # load data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 dataset, IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
    trainset = IndexedCIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = IndexedCIFAR10(root='./data', train=False, download=True, transform=train_transform)

    # CIFAR-10 类别标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    target_class = args.target_class
    target_label = classes.index(target_class)

    # target_indices = np.where(np.array(trainset.targets) == target_label)[0]
    non_target_indices = np.where(np.array(testset.targets) != target_label)[0]
    selected_indices = np.random.choice(non_target_indices, args.poison_num, replace=False)
    
    trainset.data = add_trigger_to_xb(trainset.data, selected_indices, args=args)
    for idx in selected_indices:
        trainset.targets[idx] = target_label
    
    poison_set = Subset(trainset, selected_indices)
    clean_set = Subset(trainset, non_target_indices)
    non_target_set = Subset(testset, non_target_indices)

    train_queue = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    clean_queue = torch.utils.data.DataLoader(
        dataset=clean_set,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    poison_queue = torch.utils.data.DataLoader(
        dataset=poison_set,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    non_target_queue = torch.utils.data.DataLoader(
        dataset=non_target_set,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    return train_queue, test_queue, clean_queue, poison_queue, non_target_queue, target_label


def set_model_releated(args, logger):
    # build model
    model_list = []
    model_list.append(BottomModelForCifar10())
    model_list.append(BottomModelForCifar10())
    model_list.append(TopModelForCifar10())

    # 加载预训练模型
    save_model_dir = args.resume + "/0_saved_models"
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)

    # 加载每个模型的参数
    if args.load_model == 1:
        logger.info("Set_Model_Releated_Info: Only load bottom_model_b.")
        specific_models = [1]
        for i in specific_models:
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
    elif args.load_model == 2:
        logger.info("Set_Model_Releated_Info: Load all model.")
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
    else:
        logger.info("Set_Model_Releated_Info: No model loading operation performed.")

    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for model in model_list
    ]

    stone1 = args.stone1  # 50 int(args.epochs * 0.5) 学习率衰减的epoch数
    stone2 = args.stone2  # 85 int(args.epochs * 0.8) 学习率衰减的epoch数
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_list[2], milestones=[stone1, stone2], gamma=args.step_gamma
    )
    lr_scheduler_a = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_list[0], milestones=[stone1, stone2], gamma=args.step_gamma
    )
    lr_scheduler_b = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_list[1], milestones=[stone1, stone2], gamma=args.step_gamma
    )
    # change the lr_scheduler to the one you want to use
    lr_scheduler_list = [lr_scheduler_a, lr_scheduler_b, lr_scheduler_top_model]

    return model_list, optimizer_list, lr_scheduler_list


def main(logger, device, args):
    for seed in range(1):
        logger.info(f"seed: {seed}")
        set_seed(seed=seed)

        (
            train_queue, 
            test_queue, 
            clean_queue, 
            poison_queue,
            non_target_queue, 
            target_label
        ) = set_dataset(args)

        (
            model_list,
            optimizer_list,
            lr_scheduler_list,
        ) = set_model_releated(args, logger)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        vfl_trainer = VFLTrainer(model_list)

        metrics = Metric()
        for epoch in range(0, args.epochs):
            train_loss = vfl_trainer.train_mul(
                train_queue, criterion, bottom_criterion, optimizer_list, device, args
            )

            lr_scheduler_list[0].step()
            lr_scheduler_list[1].step()
            lr_scheduler_list[2].step()

            clean_loss, _, _ = vfl_trainer.test_mul(
                clean_queue, criterion, device, args
            )
            poison_loss, _, _ = vfl_trainer.test_backdoor_mul(
                poison_queue, criterion, device, args, target_label
            )
            _, asr, _ = vfl_trainer.test_backdoor_mul(
                non_target_queue, criterion, device, args, target_label
            )

            metrics.update(train_loss, clean_loss, poison_loss, asr)
            logger.info(f"epoch: {epoch}, train_loss: {train_loss:.4f}, clean_loss: {clean_loss:.4f}, poison_loss: {poison_loss:.4f}")

        # 将列表结构记录到日志中
        logger.info(f"Train Loss: {metrics.train_loss}")
        logger.info(f"Clean Loss: {metrics.clean_loss}")
        logger.info(f"Poison Loss: {metrics.poison_loss}")
        logger.info(f"Asr: {metrics.asr}")


       
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("vflmodelnet")

    parser.add_argument('--data_dir', default="./data/CIFAR10/", help='location of the data corpus')
    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--resume', default='./model/CIFAR10/base', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', type=str, default='vfl_cifar10', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.2, help='init learning rate')
    parser.add_argument('--trigger_lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
    parser.add_argument('--workers', type=int, default=8, help='num of workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
    parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    parser.add_argument('--backdoor', type=float, default=50, help='backdoor frequency')
    parser.add_argument('--poison_epochs', type=int, default=80, help='backdoor frequency')
    parser.add_argument('--target_class', type=str, default='plane', help='backdoor target class')
    parser.add_argument('--alpha', type=float, default=0.05, help='uap learning rate decay')
    parser.add_argument('--eps', type=float, default=1.0, help='uap clamp bound')

    parser.add_argument('--poison_num', type=int, default=4, help='num of poison data')
    parser.add_argument('--corruption_amp', type=float, default=5, help='amplication of corruption')
    parser.add_argument('--backdoor_start', action='store_true', default=True, help='backdoor')

    # shuffle pretest
    parser.add_argument('--shuffle_epochs', type=int, default=50, help='epochs of shuffle training')
    parser.add_argument('--shuffle_label_way', type=str, default="class_to_class", 
                        choices=['random', 'class_to_class', 'none'], help='how to shuffle label')
    parser.add_argument('--load_model', type=int, default=2, 
                        help='1: load bottom_model_b; 2: load all; else: not load.')
    parser.add_argument('--train_bottom_model_b', action='store_true', help='whether training bottom_model_b (default: False)')
    
    # dir 配置
    parser.add_argument('--save', default='./model/CIFAR10/baseline', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--log_file_name', type=str, default="experiment.log", help='log name')

    # config file
    parser.add_argument('--c', type=str, default='configs/shuffle/cifar10_test.yml', help='config file')

    args = parser.parse_args()
    # over_write_args_from_file(args, args.c)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)

    logger.info("")
    shuffle_args_str = f"Running experiment with args: " + \
        f"load_model={args.load_model}, train_bottom_model_b={args.train_bottom_model_b}, " + \
        f"shuffle_label_way={args.shuffle_label_way}, lr={args.lr}, shuffle_epochs={args.shuffle_epochs}, batch_size={args.batch_size}"
    logger.info(shuffle_args_str)
    
    main(logger=logger, device=device, args=args)


