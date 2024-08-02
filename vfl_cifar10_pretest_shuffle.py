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
from fedml_core.trainer.vfl_trainer import VFLTrainer
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

def set_shuffle_dataset(args, trainset, logger):
    shuffle_trainset = copy.deepcopy(trainset)
    shuffle_testset = copy.deepcopy(trainset)

    if args.shuffle_label_way == "class_to_class":
        logger.info("Randomly replace the label of a certain class with the label of another class")
        
        original_trainset_labels = shuffle_trainset.targets
        original_testset_labels = shuffle_testset.targets
        # 获取所有唯一的标签
        unique_labels = list(set(original_trainset_labels))
        num_labels = len(unique_labels)
        
        # 创建一个新的标签列表，确保新标签与原始标签不同
        new_labels = unique_labels[:]
        random.shuffle(new_labels)
        
        # 保证新标签和原标签不同
        for i in range(num_labels):
            if unique_labels[i] == new_labels[i]:
                swap_index = (i + 1) % num_labels
                new_labels[i], new_labels[swap_index] = new_labels[swap_index], new_labels[i]
        
        # 创建新的标签字典，将每个原始标签映射到一个新的唯一标签
        new_labels_mapping = {original: new for original, new in zip(unique_labels, new_labels)}
        # 记录扰乱前后标签的对应关系
        logger.info("Original to new label mapping:")
        for original_label, new_label in new_labels_mapping.items():
            logger.info(f"{original_label} -> {new_label}")
        
        # 将所有标签替换为新的标签
        shuffle_trainset.targets = [new_labels_mapping[label] for label in original_trainset_labels]
        shuffle_testset.targets = [new_labels_mapping[label] for label in original_testset_labels]
    
    elif args.shuffle_label_way == "random":
        logger.info("Randomly replace the label of each sample with the label of other samples")
        shuffle_labels = shuffle_trainset.targets  
        random.shuffle(shuffle_labels)
        shuffle_trainset.targets = shuffle_labels
    
    else:
        logger.info("Do not shuffle labels.")

    return shuffle_trainset, shuffle_testset


def set_dataset(args):
    # load data
    # Data normalization and augmentation (optional)
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # 基于 CIFAR-10 数据集计算出来的每个通道的均值和标准差。
        # 确保每个通道的像素值有一个均值为 0 和标准差为 1 的分布。
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        # transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2471, 0.2435, 0.2616))
    ])

    # Load CIFAR-10 dataset
    # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
    trainset = IndexedCIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = IndexedCIFAR10(root='./data', train=False, download=True, transform=train_transform)
    shuffle_trainset, shuffle_testset = set_shuffle_dataset(args, trainset, logger)

    # CIFAR-10 类别标签（以类别名称的列表形式给出）
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    target_class = args.target_class
    target_label = classes.index(target_class)

    # 找出所有属于这个类别的样本的索引
    non_target_indices = np.where(np.array(testset.targets) != target_label)[0]
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
    non_target_queue = torch.utils.data.DataLoader(
        dataset=non_target_set,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    shuffle_train_queue = torch.utils.data.DataLoader(
        dataset=shuffle_trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers
    )
    shuffle_test_queue = torch.utils.data.DataLoader(
        dataset=shuffle_testset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    return train_queue, test_queue, non_target_queue, shuffle_train_queue, shuffle_test_queue, target_label


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

    # 复制一份 model_list
    copied_model_list = copy.deepcopy(model_list)

    # 加载delta,用来生成有毒样本
    delta = torch.load(save_model_dir + "/delta.pth")

    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        for model in copied_model_list
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

    return model_list, copied_model_list, delta, optimizer_list, lr_scheduler_list


def main(logger, device, args):
    
    for seed in range(1):
        # set_seed(seed=seed)
        logger.info(f"seed: {seed}")

        (
            train_queue, 
            test_queue, 
            non_target_queue, 
            shuffle_train_queue, 
            shuffle_test_queue,
            target_label
        ) = set_dataset(args)

        (
            model_list,
            copied_model_list,
            delta,
            optimizer_list,
            lr_scheduler_list,
        ) = set_model_releated(args, logger)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        vfl_trainer = VFLTrainer(model_list)
        vfl_shuffle_trainer = VFLTrainer(copied_model_list)

        if args.train_bottom_model_b:
            logger.info("Training_info: Train bottom_model_b.")
        else:
            logger.info("Training_info: Not training bottom_model_b.")

        metrics = ShuffleTrainMetric()
        for epoch in range(0, args.shuffle_epochs):
            shuffle_train_loss = vfl_shuffle_trainer.train_shuffle(
                shuffle_train_queue, criterion, bottom_criterion, optimizer_list, device, args
            )

            lr_scheduler_list[0].step()
            lr_scheduler_list[2].step()
            if args.train_bottom_model_b:   # 训练 bottom model b
                lr_scheduler_list[1].step()

            shuffle_test_loss, _, _ = vfl_shuffle_trainer.test_mul(
                test_queue, criterion, device, args
            )
            # test clean loss            
            main_task_loss, main_task_acc, _ = vfl_shuffle_trainer.test_mul(
                train_queue, criterion, device, args
            )
            # test backdoor attack loss
            backdoor_task_loss, backdoor_task_acc, _ = vfl_shuffle_trainer.test_backdoor_mul(
                train_queue, criterion, device, args, delta, target_label
            )

            metrics.update(
                shuffle_train_loss=shuffle_train_loss, 
                shuffle_test_loss=shuffle_test_loss,
                main_task_loss=main_task_loss, 
                backdoor_task_loss=backdoor_task_loss,
                main_task_acc=main_task_acc,
                backdoor_task_acc=backdoor_task_acc,
            )

            # logger.info(f"epoch: {epoch+1}, shuffle_train_loss: {shuffle_train_loss:.4f}, main_task_loss: {main_task_loss:.4f}, backdoor_task_loss: {backdoor_task_loss:.4f}")

        # 将列表结构记录到日志中
        logger.info(f"Shuffle Train Loss: {metrics.shuffle_train_loss}")
        logger.info(f"Shuffle Test Loss: {metrics.shuffle_test_loss}")
        logger.info(f"Main Task Loss: {metrics.main_task_loss}")
        logger.info(f"Backdoor Attack Loss: {metrics.backdoor_attack_loss}")
        logger.info(f"Main Task Acc: {metrics.main_task_acc}")
        logger.info(f"Backdoor Attack Acc: {metrics.backdoor_task_acc}")

        # 对比 shuffle model 和 normal 的输出差异
        logger.info("########## Compare acc in Main Task ##########")
        _, top1_acc_origin, top5_acc_origin = vfl_trainer.test_mul(
            test_queue, criterion, device, args
        )
        _, top1_acc_shuffle, top5_acc_shuffle = vfl_shuffle_trainer.test_mul(
            test_queue, criterion, device, args
        )
        logger.info(f"Origin Model in Main Task:  top1_acc:{top1_acc_origin:.4f}, top5_acc:{top5_acc_origin:.4f}")
        logger.info(f"Shuffle Model in Main Task:  top1_acc:{top1_acc_shuffle:.4f}, top5_acc:{top5_acc_shuffle:.4f}")

        logger.info("########## Compare acc in Backdoor Task ##########")
        _, test_asr_acc_origin, _ = vfl_trainer.test_backdoor_mul(
            non_target_queue, criterion, device, args, delta, target_label
        )
        _, test_asr_acc_shuffle, _ = vfl_shuffle_trainer.test_backdoor_mul(
            non_target_queue, criterion, device, args, delta, target_label
        )
        logger.info(f"Origin Model in Backdoor Task:  test_asr_acc:{test_asr_acc_origin:.4f}")
        logger.info(f"Shuffle Model in Backdoor Task:  test_asr_acc:{test_asr_acc_shuffle:.4f}")

       
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

    parser.add_argument('--marvell', action='store_true', default=False, help='marvell defense')
    parser.add_argument('--max_norm', action='store_true', default=False, help='maxnorm defense')
    parser.add_argument('--iso', action='store_true', default=False, help='iso defense')
    parser.add_argument('--gc', action='store_true', default=False, help='gc defense')
    parser.add_argument('--lap_noise', action='store_true', default=False, help='lap_noise defense')
    parser.add_argument('--signSGD', action='store_true', default=False, help='sign_SGD defense')

    parser.add_argument('--iso_ratio', type=float, default=0.01, help='iso defense ratio')
    parser.add_argument('--gc_ratio', type=float, default=0.01, help='gc defense ratio')
    parser.add_argument('--lap_noise_ratio', type=float, default=0.01, help='lap_noise defense ratio')

    parser.add_argument('--poison_num', type=int, default=4, help='num of poison data')
    parser.add_argument('--corruption_amp', type=float, default=5, help='amplication of corruption')
    parser.add_argument('--backdoor_start', action='store_true', default=True, help='backdoor')

    parser.add_argument('--shuffle_epochs', type=int, default=50, help='epochs of shuffle training')
    parser.add_argument('--shuffle_label_way', type=str, default="class_to_class", 
                        choices=['random', 'class_to_class', 'none'], help='how to shuffle label')
    parser.add_argument('--load_model', type=int, default=2, 
                        help='1: load bottom_model_b; 2: load all; else: not load.')
    # argparse 解析布尔参数时，如果直接传递字符串 “True” 或 “False”，会被解释为字符串类型而不是布尔类型。
    # 通常，处理布尔参数的推荐方法是使用 store_true 和 store_false 动作。
    parser.add_argument('--train_bottom_model_b', action='store_true', help='whether training bottom_model_b (default: False)')
    
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


