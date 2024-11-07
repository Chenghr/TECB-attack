import logging
import os
import random
import sys
import datetime
import copy
import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# from fedml_core.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_two_party_data
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
from fedml_core.model.vfl_models import (
    BottomModelForCifar10,
    TopModelForCifar10,
)
# from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.trainer.villain_trainer import VillainTrainer
from fedml_core.utils.utils import (
    AverageMeter,
    keep_predict_loss,
    over_write_args_from_file,
)
from fedml_core.utils.logger import setup_logger

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import glob
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import logging


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model_list, load_path, logger):
    if os.path.isfile(load_path):
        logger.info(f"=> loading checkpoint {load_path}")

        checkpoint = torch.load(load_path, map_location=device)
        # args.start_epoch = checkpoint["epoch"]
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint["state_dict"][i])
        
        epoch = checkpoint["epoch"]
        logger.info(f"=> loaded checkpoint '{load_path} (epoch {epoch})")

    else:
        logger.info(f"=> no checkpoint found at '{load_path}'")
    
    return model_list

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_dataset_basic(args):
    # load data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 dataset
    # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
    trainset = IndexedCIFAR10(root='../../data', train=True, download=True, transform=train_transform)
    testset = IndexedCIFAR10(root='../../data', train=False, download=True, transform=train_transform)
    
    # CIFAR-10 类别标签（以类别名称的列表形式给出）
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    target_class = args.target_class
    target_label = classes.index(target_class)
    train_queue = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    train_queue_nobatch = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=50000,
        num_workers=args.workers,
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size, num_workers=args.workers
    )
    # 找出所有属于这个类别的样本的索引
    # 从目标索引中随机选择 poison_num 个索引, 作为毒样本
    return trainset, train_queue, train_queue_nobatch, test_queue, target_label

def set_villain_model_releated(args, logger):
    # build model
    model_list = []
    model_list.append(BottomModelForCifar10())
    model_list.append(BottomModelForCifar10())
    model_list.append(TopModelForCifar10())

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
    optimizer_list[1].param_groups[0]['lr'] = args.local_lr
    stone1 = args.stone1  # 50 int(args.epochs * 0.5) 学习率衰减的epoch数
    lr_scheduler_list = [
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[stone1], 
            gamma=args.step_gamma
        )
        for optimizer in optimizer_list
    ]

    # optionally resume from a checkpoint
    if args.resume:
        model_list = load_checkpoint(model_list, args.resume, logger)

    return model_list, optimizer_list, lr_scheduler_list
    


def main(device, args, logger):
    ASR_Top1 = AverageMeter()
    Main_Top1_acc = AverageMeter()
    Main_Top5_acc = AverageMeter()

    for seed in range(5):
        set_seed(seed)
        
        save_model_dir = args.save + f"/seed={seed}_saved_models"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
            
        # Load data
        trainset, train_queue, train_queue_nobatch, test_queue, target_label = set_dataset_basic(args)
        target_indices = np.where(np.array(trainset.targets) == target_label)[0]

        # Build model
        model_list, optimizer_list, lr_scheduler_list= set_villain_model_releated(args, logger)
        villain_trainer = VillainTrainer(model_list)
        
        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        # Set trigger
        logger.info("###### Generate Trigger ######") 
        mask_list = {}
        delta, mask_list = villain_trainer.trigger_gen(train_queue, target_indices, device, args)
        print(f"delta: {delta}")
        # Trian VFL
        logger.info("###### Train Federated Models ######") 
        best_score, best_top1_acc, best_asr = 0.0, 0.0, 0.0
        
        for epoch in range(args.start_epoch, args.epochs):
            logging.info("epoch %d args.lr %e ", epoch, args.lr)

            # train_loss, delta = vfltrainer.train_narcissus(train_queue, criterion, bottom_criterion,optimizer_list, device, args, delta, selected_indices, trigger_optimizer)
            if args.backdoor_start:
                train_loss = villain_trainer.train_with_trigger(
                    train_queue, delta, mask_list, criterion, bottom_criterion, optimizer_list, device, args
                )
            else:
                train_loss = villain_trainer.train(
                    train_queue, criterion, bottom_criterion, optimizer_list, device, args
                )

            lr_scheduler_list[0].step()
            lr_scheduler_list[1].step()
            lr_scheduler_list[2].step()

            test_loss, top1_acc, top5_acc = villain_trainer.test_mul(
                test_queue, criterion, device, args
            )
            test_asr_acc, _ =villain_trainer.test_backdoor(
                test_queue, delta, target_label, device, args
            )

            logger.info(
                    f"epoch: {epoch + 1}, train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}"
                f"top1_acc: {top1_acc:.5f}, top5_acc: {top5_acc:.5f}, test_asr_acc: {test_asr_acc:.5f}"
            )

            if not args.backdoor_start:
                is_best = top1_acc >= best_score
                if is_best:
                    best_score, best_top1_acc = top1_acc, top1_acc
            else:
                # Dynamically adjust the weight over epochs
                epoch_ratio = epoch / args.epochs
                weight_asr = min(0.5 + 0.5 * epoch_ratio, 1.0)  # Example: gradually increase ASR importance
                weight_top1 = 1.0 - weight_asr
                total_value = weight_asr * test_asr_acc + weight_top1 * top1_acc
                
                is_best = total_value >= best_score
                if is_best:
                    best_score, best_top1_acc, best_asr = total_value, top1_acc, test_asr_acc

            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "top1_acc": top1_acc,
                        "test_asr_acc": test_asr_acc,
                        "state_dict": [
                            model_list[i].state_dict() for i in range(len(model_list))
                        ],
                        "optimizer": [
                            optimizer_list[i].state_dict() for i in range(len(optimizer_list))
                        ],
                    },
                    is_best,
                    save_model_dir,
                    "checkpoint_{:04d}.pth.tar".format(epoch),
                )
        
        Main_Top1_acc.update(best_top1_acc)
        ASR_Top1.update(best_asr)

        saved_stdout = sys.stdout
        with open(os.path.join(args.save, f"saved_result.txt"), "a") as file:
            sys.stdout = file
            
            print(f"### Seed: {seed} ###")
            print(f"Best Top1 Acc: {best_top1_acc}")
            print(f"Best Asr Acc: {best_asr}")
            
            if seed == 4:
                print("### Final Result ###")
                print(f"Main AVG Top1 acc: {Main_Top1_acc.avg}, Main STD Top1 acc: {Main_Top1_acc.std_dev()}")
                print(f"ASR AVG Top1 acc: {ASR_Top1.avg}, ASR STD Top1 acc: {ASR_Top1.std_dev()}")
            
            sys.stdout = saved_stdout


if __name__ == "__main__":
    default_config_path = os.path.abspath("../../attack/villain/best_configs/cifar10_bestattack.yml")
    default_data_path = os.path.abspath("../../data/")
    default_save_path = os.path.abspath("../../results/models/Villain/cifar10")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser("villain_cifar10")

    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--data_dir", default="../../data/CIFAR10/", help="location of the data corpus")
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=[
        "CIFAR10", "CIFAR100", "TinyImageNet", "CINIC10L", "Yahoo", "Criteo", "BCW"
    ], help="name of dataset")

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument("--name", type=str, default="villain_cifar10", help="experiment name")
    experiment_group.add_argument("--save", default=default_save_path, type=str, metavar="PATH", help="path to save checkpoint")
    experiment_group.add_argument("--log_file_name", type=str, default="experiment.log", help="log name")

    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--batch_size", type=int, default=1024, help="batch size")
    training_group.add_argument("--lr", type=float, default=0.1, help="init learning rate")
    training_group.add_argument("--trigger_lr", type=float, default=0.001, help="init learning rate for trigger")
    training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--report_freq", type=float, default=10, help="report frequency")
    training_group.add_argument("--workers", type=int, default=0, help="num of workers")
    training_group.add_argument("--epochs", type=int, default=80, help="num of training epochs")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
    training_group.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--stone1", default=30, type=int, metavar="s1", help="stone1 for step scheduler")
    # training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")

    # 模型相关参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--layers", type=int, default=18, help="total number of layers")
    model_group.add_argument("--u_dim", type=int, default=64, help="u layer dimensions")
    model_group.add_argument("--k", type=int, default=2, help="num of clients")
    model_group.add_argument("--parallel", action="store_true", default=False, help="data parallelism")
    model_group.add_argument('--load_model', type=int, default=0, help='1: load bottom_model_b; 2: load all; else: not load.')

    # 后门相关参数
    backdoor_group = parser.add_argument_group('Backdoor')
    backdoor_group.add_argument("--alpha", type=float, default=0.02, help="uap learning rate decay")
    backdoor_group.add_argument("--eps", type=float, default=16 / 255, help="uap clamp bound")
    backdoor_group.add_argument("--corruption_amp", type=float, default=5.0, help="amplification of corruption")
    backdoor_group.add_argument("--backdoor_start", action="store_true", default=True, help="backdoor")
    backdoor_group.add_argument("--pre_train_epoch", type=int, default=10, help="pre train epoch")
    backdoor_group.add_argument("--poison_budget", type=float, default=0.1, help="poison sample fraction")
    backdoor_group.add_argument("--beta", type=float, default=0.4, help="controling trigger magnitude")
    backdoor_group.add_argument("--gamma_up", type=float, default=1.2, help="upper bound of shifting")
    backdoor_group.add_argument("--gamma_low", type=float, default=0.6, help="lower bound of shifting")
    backdoor_group.add_argument("--dropout_ratio", type=float, default=0.25, help="dropout ratio")
    backdoor_group.add_argument("--target_class", type=str, default="cat", help="target class index")
    backdoor_group.add_argument("--local_lr", type=float, default=0.5, help="local learning rate")
    
    # 防御相关参数
    # defense_group = parser.add_argument_group('Defense')
    # defense_group.add_argument("--marvell", action="store_true", default=False, help="marvell defense")
    # defense_group.add_argument("--max_norm", action="store_true", default=False, help="maxnorm defense")
    # defense_group.add_argument("--iso", action="store_true", default=False, help="iso defense")
    # defense_group.add_argument("--gc", action="store_true", default=False, help="gc defense")
    # defense_group.add_argument("--lap_noise", action="store_true", default=False, help="lap_noise defense")
    # defense_group.add_argument("--signSGD", action="store_true", default=False, help="sign_SGD defense")
    # defense_group.add_argument("--iso_ratio", type=float, default=0.01, help="iso defense ratio")
    # defense_group.add_argument("--gc_ratio", type=float, default=0.01, help="gc defense ratio")
    # defense_group.add_argument("--lap_noise_ratio", type=float, default=0.01, help="lap_noise defense ratio")

    # 配置文件相关参数
    # config_group = parser.add_argument_group('Config')
    # config_group.add_argument("--c", type=str, default="./configs/BadVFL/cifar10_test.yml", help="config file")

    # 半特征相关参数
    parser.add_argument(
        "--c",
        type=str,
        default=default_config_path,
        help="config file",
    )
    feature_group = parser.add_argument_group('Feature')
    feature_group.add_argument("--half", type=int, default=16, help="half number of features")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    args.save = os.path.join(args.save, timestamp)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)

    # 记录所有的参数信息
    logger.info(f"Experiment arguments: {args}")
    
    main(logger=logger, device=device, args=args)
