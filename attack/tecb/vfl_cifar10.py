import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse
import glob
import logging
import shutil
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fedml_core.data_preprocessing.cifar10 import IndexedCIFAR10
from fedml_core.model.vfl_models import (BottomModelForCifar10,
                                                  TopModelForCifar10)
from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.utils.utils import (AverageMeter, keep_predict_loss,
                                    over_write_args_from_file)
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def train(device, args):
    ASR_Top1 = AverageMeter()
    ASR_Top5 = AverageMeter()
    Main_Top1_acc = AverageMeter()
    Main_Top5_acc = AverageMeter()

    for seed in [0, 1, 2]:
        # random seed for 10 runs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # load data
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load CIFAR-10 dataset
        # 唯一的区别是，IndexedCIFAR10 类返回的图片的第三个元素是图片的索引
        trainset = IndexedCIFAR10(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        testset = IndexedCIFAR10(
            root=args.data_dir, train=False, download=True, transform=train_transform
        )

        # 找出所有属于这个类别的样本的索引
        target_label =trainset.class_to_idx[args.target_class]
        target_indices = np.where(np.array(trainset.targets) == target_label)[0]
        non_target_indices = np.where(np.array(testset.targets) != target_label)[0]
        non_target_set = Subset(testset, non_target_indices)

        # 从目标索引中随机选择 poison_num 个索引, 作为毒样本
        selected_indices = np.random.choice(
            target_indices, args.poison_num, replace=False
        )

        train_queue = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        test_queue = torch.utils.data.DataLoader(
            dataset=testset, batch_size=args.batch_size, num_workers=args.workers
        )
        non_target_queue = torch.utils.data.DataLoader(
            dataset=non_target_set, batch_size=args.batch_size, num_workers=args.workers
        )

        # build model
        model_list = []
        model_list.append(BottomModelForCifar10(model_name=args.bottom_model_name))
        model_list.append(BottomModelForCifar10(model_name=args.bottom_model_name))
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

        # vfltrainer = VFLTrainer(model_list)
        vfltrainer = TECBTrainer(model_list)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=device)
                args.start_epoch = checkpoint["epoch"]
                for i in range(len(model_list)):
                    model_list[i].load_state_dict(checkpoint["state_dict"][i])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        print(
            "################################ Train Federated Models ############################"
        )
        best_asr = 0.0

        _, (x_val, y_val, index) = next(enumerate(train_queue))

        # delta = torch.zeros_like(x_val[1][1]).float().to(device)
        if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
            delta = torch.zeros((1, 3, 32, 32-args.half), device=device)
        else:
            raise ValueError
            
        delta.requires_grad_(True)

        # Set a 9-pixel pattern to 1
        # delta[:, 0:3, 0:3] = 1
        for epoch in range(args.start_epoch, args.epochs):
            logging.info("epoch %d args.lr %e ", epoch, args.lr)

            # train_loss, delta = vfltrainer.train_narcissus(train_queue, criterion, bottom_criterion,optimizer_list, device, args, delta, selected_indices, trigger_optimizer)
            if args.backdoor_start:
                if (epoch + 1) < args.backdoor:
                    # acc = vfltrainer.pesudo_label_predict(train_queue, device)
                    # if epoch <20:
                    # if args.backdoor:
                    train_loss, delta = vfltrainer.train_narcissus(
                        train_queue,
                        criterion,
                        bottom_criterion,
                        optimizer_list,
                        device,
                        args,
                        delta,
                        selected_indices,
                    )

                elif (epoch + 1) >= args.backdoor and (epoch + 1) < args.poison_epochs:
                    train_loss = vfltrainer.train(
                        train_queue,
                        criterion,
                        bottom_criterion,
                        optimizer_list,
                        device,
                        args,
                    )
                else:
                    train_loss = vfltrainer.train_poisoning(
                        train_queue,
                        criterion,
                        bottom_criterion,
                        optimizer_list,
                        device,
                        args,
                        delta,
                        selected_indices,
                    )
            else:
                train_loss = vfltrainer.train(
                    train_queue,
                    criterion,
                    bottom_criterion,
                    optimizer_list,
                    device,
                    args,
                )

            for scheduler in lr_scheduler_list:
                scheduler.step()

            test_loss, top1_acc, top5_acc = vfltrainer.test(
                test_queue, criterion, device, args
            )
            _, test_asr_acc, test_asr_acc5 = vfltrainer.test_backdoor(
                non_target_queue, criterion, device, args, delta, target_label
            )

            print(f"Epoch: {epoch + 1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Top1 Acc: {top1_acc:.2f}% | "
                  f"Top5 Acc: {top5_acc:.2f}% | "
                  f"ASR Top1: {test_asr_acc:.2f}% | "
                  f"ASR Top5: {test_asr_acc5:.2f}%")

            ## save partyA and partyB model parameters
            if not args.backdoor_start:
                is_best = top1_acc >= best_asr
                best_asr = max(top1_acc, best_asr)
            else:
                total_value = test_asr_acc + top1_acc
                is_best = total_value >= best_asr
                best_asr = max(total_value, best_asr)

            save_model_dir = args.save + f"/{seed}_saved_models"
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "best_auc": best_asr,
                        "state_dict": [
                            model_list[i].state_dict() for i in range(len(model_list))
                        ],
                        "optimizer": [
                            optimizer_list[i].state_dict()
                            for i in range(len(optimizer_list))
                        ],
                    },
                    is_best,
                    save_model_dir,
                    "checkpoint_{:04d}.pth.tar".format(epoch),
                )

                backdoor_data = {
                    "delta": delta,
                    "source_label": -1,  #  tecb 是 source 无关的攻击方法
                    "target_label": target_label,
                }
                torch.save(backdoor_data, os.path.join(save_model_dir, "backdoor.pth"))

        print("Testing Best Model")
        checkpoint_path = os.path.join(args.save, f"{seed}_saved_models", "model_best.pth.tar")
        
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            for i, model in enumerate(model_list):
                model.load_state_dict(checkpoint["state_dict"][i])

        vfltrainer.update_model(model_list)

        with open(os.path.join(args.save, "saved_result.txt"), "a") as file:
            sys.stdout = file
            test_loss, top1_acc, top5_acc = vfltrainer.test(
                test_queue, criterion, device, args
            )
            _, asr_top1_acc, asr_top5_acc = vfltrainer.test_backdoor(
                non_target_queue, criterion, device, args, delta, target_label
            )

            print("\nTest Results (Seed {})".format(seed))
            print("Main Task Metrics:")
            print(f"Loss: {test_loss:.4f} | "
                  f"Top1: {top1_acc:.2f}% | "
                  f"Top5: {top5_acc:.2f}%")

            print("Backdoor Task Metrics:")
            print(f"ASR Top1: {asr_top1_acc:.2f}% | "
                  f"ASR Top5: {asr_top5_acc:.2f}%\n")

            Main_Top1_acc.update(top1_acc)
            Main_Top5_acc.update(top5_acc)
            ASR_Top1.update(asr_top1_acc)
            ASR_Top5.update(asr_top5_acc)

            if seed == 4:
                print("Final Results Summary")
                print("Main Model Performance:")
                print(f"Top1: {Main_Top1_acc.avg:.2f}% ± {Main_Top1_acc.std_dev():.2f}%")
                print(f"Top5: {Main_Top5_acc.avg:.2f}% ± {Main_Top5_acc.std_dev():.2f}%")
                
                print("Backdoor Performance:")
                print(f"ASR Top1: {ASR_Top1.avg:.2f}% ± {ASR_Top1.std_dev():.2f}%")
                print(f"ASR Top5: {ASR_Top5.avg:.2f}% ± {ASR_Top5.std_dev():.2f}%")

            sys.stdout = sys.__stdout__

        print(f"Results for seed {seed} saved to file")


def test(device, args):
    # 加载预训练模型
    save_model_dir = args.save
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)

    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)
    
    # load model
    model_list = []
    model_list.append(BottomModelForCifar10(model_name=args.bottom_model_name))
    model_list.append(BottomModelForCifar10(model_name=args.bottom_model_name))
    model_list.append(TopModelForCifar10())

    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    # load data
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = IndexedCIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    testset = IndexedCIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    # 找出所有属于这个类别的样本的索引
    non_target_indices = np.where(np.array(testset.targets) != target_label)[0]
    non_target_set = Subset(testset, non_target_indices)

    train_queue = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size, num_workers=args.workers
    )
    non_target_queue = torch.utils.data.DataLoader(
        dataset=non_target_set, batch_size=args.batch_size, num_workers=args.workers
    )

    print(
        "################################ Test Backdoor Models ############################"
    )
    vfltrainer = TECBTrainer(model_list)
    test_loss, top1_acc, top5_acc = vfltrainer.test(test_queue, criterion, device, args)

    test_loss, test_asr_acc, _ = vfltrainer.test_backdoor(
        non_target_queue, criterion, device, args, delta, target_label
    )
    print(
        "test_loss: ",
        test_loss,
        "top1_acc: ",
        top1_acc,
        "top5_acc: ",
        top5_acc,
        "test_asr_acc: ",
        test_asr_acc,
    )


if __name__ == "__main__":
    print("################################ Prepare Data ############################")

    default_config_path = os.path.abspath("./best_configs/cifar10_bestattack.yml")
    default_data_path = os.path.abspath("../../data/")
    default_save_path = os.path.abspath("../../results/models/TECB/cifar10")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser("tecb_cifar10")

    parser.add_argument(
        "--data_dir", default=default_data_path, help="location of the data corpus"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="CIFAR10",
        type=str,
        help="name of dataset",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "TinyImageNet",
            "CINIC10L",
            "Yahoo",
            "Criteo",
            "BCW",
        ],
    )
    parser.add_argument(
        "--name", type=str, default="vfl_cifar10", help="experiment name"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="init learning rate")
    parser.add_argument(
        "--trigger_lr", type=float, default=0.001, help="init learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--report_freq", type=float, default=10, help="report frequency"
    )
    parser.add_argument("--workers", type=int, default=0, help="num of workers")
    parser.add_argument(
        "--epochs", type=int, default=100, help="num of training epochs"
    )
    parser.add_argument("--layers", type=int, default=18, help="total number of layers")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--grad_clip", type=float, default=5.0, help="gradient clipping"
    )
    parser.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    parser.add_argument(
        "--decay_period",
        type=int,
        default=1,
        help="epochs between two learning rate decays",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="data parallelism"
    )
    parser.add_argument("--u_dim", type=int, default=64, help="u layer dimensions")
    parser.add_argument("--k", type=int, default=2, help="num of client")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--save",
        # default="./model/CIFAR10/baseline",
        default=default_save_path,
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: none)",
    )
    parser.add_argument(
        "--step_gamma",
        default=0.1,
        type=float,
        metavar="S",
        help="gamma for step scheduler",
    )
    parser.add_argument(
        "--stone1", default=50, type=int, metavar="s1", help="stone1 for step scheduler"
    )
    parser.add_argument(
        "--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler"
    )
    parser.add_argument(
        "--half",
        help="half number of features, generally seen as the adversary's feature num. "
        "You can change this para (lower that party_num) to evaluate the sensitivity "
        "of our attack -- pls make sure that the model to be resumed is "
        "correspondingly trained.",
        type=int,
        default=16,
    )  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    parser.add_argument("--backdoor", type=float, default=20, help="backdoor frequency")
    parser.add_argument(
        "--poison_epochs", type=float, default=20, help="backdoor frequency"
    )
    parser.add_argument(
        "--target_class", type=str, default="cat", help="backdoor target class"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="uap learning rate decay"
    )
    parser.add_argument("--eps", type=float, default=16 / 255, help="uap clamp bound")

    parser.add_argument(
        "--marvell", action="store_true", default=False, help="marvell defense"
    )
    parser.add_argument(
        "--max_norm", action="store_true", default=False, help="maxnorm defense"
    )
    parser.add_argument("--iso", action="store_true", default=False, help="iso defense")
    parser.add_argument("--gc", action="store_true", default=False, help="gc defense")
    parser.add_argument(
        "--lap_noise", action="store_true", default=False, help="lap_noise defense"
    )
    parser.add_argument(
        "--signSGD", action="store_true", default=False, help="sign_SGD defense"
    )

    parser.add_argument(
        "--iso_ratio", type=float, default=0.01, help="iso defense ratio"
    )
    parser.add_argument("--gc_ratio", type=float, default=0.01, help="gc defense ratio")
    parser.add_argument(
        "--lap_noise_ratio", type=float, default=0.01, help="lap_noise defense ratio"
    )

    parser.add_argument(
        "--poison_num", type=int, default=100, help="num of poison data"
    )
    parser.add_argument(
        "--corruption_amp", type=float, default=10, help="amplication of corruption"
    )
    parser.add_argument(
        "--backdoor_start", action="store_true", default=False, help="backdoor"
    )
    parser.add_argument(
        "--bottom_model_name", type=str, default="resnet20", 
        choices=["resnet20", "vgg16", "LeNet"], 
    )

    # config file
    parser.add_argument(
        "--c",
        type=str,
        default=default_config_path,
        help="config file",
    )

    args = parser.parse_args()
    # over_write_args_from_file(args, args.c)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.save + "/experiment.log")
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    logger.info(args)
    logger.info(device)
    
    train(device=device, args=args)
    # test(device=device, args=args)

    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)
