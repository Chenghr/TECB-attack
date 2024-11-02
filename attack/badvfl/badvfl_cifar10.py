import os
import sys
import random

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from attack.badvfl.utils import (
    set_seed, 
    init_model_releated, init_dataloader, 
    sample_poisoned_source_target_data, 
    construct_poison_train_dataloader, get_source_label_dataloader,
    save_checkpoint
)
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L
from fedml_core.model.baseline.vfl_models import (
    BottomModelForCinic10,
    TopModelForCinic10,
)
from fedml_core.trainer.badvfl_trainer import BadVFLTrainer
from fedml_core.utils.utils import (
    AverageMeter,
    keep_predict_loss,
)
from torch.utils.data import Subset


def train(args, logger):
    ASR_Top1 = AverageMeter()
    ASR_Top5 = AverageMeter()
    Main_Top1_acc = AverageMeter()
    Main_Top5_acc = AverageMeter()
    
    for seed in range(args.seed_num):
        set_seed(seed)
        
        save_model_dir = args.save + f"/seed={seed}_saved_models"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        # 加载基础数据
        train_dataloader, test_dataloader = init_dataloader(
            args.dataset, args.data_dir, args.batch_size
        )
        # 加载模型
        model_list, optimizer_list, lr_scheduler_list = init_model_releated(
            args.dataset, args.lr, args.momentum, args.weight_decay, args.stone1, args.stone2, args.step_gamma
        )
        # 设置训练函数
        criterion = nn.CrossEntropyLoss().to(args.device)
        bottom_criterion = keep_predict_loss

        trainer = BadVFLTrainer(model_list)
        
        # Pretrain
        logger.info("###### Pre-Trained ######")
        pre_train_loss = []
        for _ in range(args.pre_train_epochs):
            loss = trainer.pre_train(
                train_dataloader, criterion, bottom_criterion, optimizer_list, args, 
            )
            pre_train_loss.append(loss)
        logger.info(f"Pre-Train Loss: [{', '.join([f'{l:.4f}' for l in pre_train_loss])}]")
        
        # Optimal select label
        source_label, target_label = trainer.select_closest_class_pair(
            train_dataloader, args
        )
        selected_source_indices, selected_target_indices, selected_dataloader = sample_poisoned_source_target_data(
            train_dataloader.dataset, source_label, target_label, args.poison_budget
        )
        logger.info(f"source_label: {source_label}, target_label: {target_label}")
        
        # Set trigger
        logger.info("###### Train Trigger ######") 
       
        best_position = trainer.find_optimal_trigger_position(
            train_dataloader, source_label, criterion, optimizer_list, args
        )
        
        delta = torch.full((1, 3, args.window_size, args.window_size), 0.0).to(args.device)
        
        trigger_optimizer = torch.optim.SGD([delta], 0.25)
        trigger_loss= []
        for _ in range(args.trigger_train_epochs):
            delta, loss = trainer.train_trigger(
                selected_dataloader, best_position, delta, trigger_optimizer, args
            )
            trigger_loss.append(loss)
        logger.info(f"Trigger Train Loss: [{', '.join([f'{l:.4f}' for l in trigger_loss])}]")
        
        # Set poison data
        poison_train_dataloader = construct_poison_train_dataloader(
            train_dataloader, args.dataset, selected_source_indices, selected_target_indices, delta, best_position, args
        )
        convert_delta = trainer.convert_delta(
            train_dataloader, best_position, delta, args
        )
        source_label_dataloader = get_source_label_dataloader(
            test_dataloader, args.dataset, source_label, args
        )
        
        # Trian VFL
        logger.info("###### Train Federated Models ######") 
        best_score, best_top1_acc, best_asr = 0.0, 0.0, 0.0
        
        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.backdoor_start_epoch:
                train_loss = trainer.train(
                    train_dataloader, criterion, bottom_criterion, optimizer_list, args
                )
            else:
                train_loss = trainer.train_poisoning(
                    poison_train_dataloader, criterion, bottom_criterion, optimizer_list, args
                )
            
            for i in range(3):
                lr_scheduler_list[i].step()

            test_loss, top1_acc, top5_acc = trainer.test(
                test_dataloader, criterion, args
            )
            _, test_asr_acc, test_asr_acc5 = trainer.test_backdoor(
                source_label_dataloader, criterion, convert_delta, target_label, args
            )

            print(f"Epoch: {epoch + 1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Top1 Acc: {top1_acc:.2f}% | "
                  f"Top5 Acc: {top5_acc:.2f}% | "
                  f"ASR Top1: {test_asr_acc:.2f}% | "
                  f"ASR Top5: {test_asr_acc5:.2f}%")

            if not args.backdoor_start:
                is_best = top1_acc >= best_asr
                best_asr = max(top1_acc, best_asr)
            else:
                total_value = test_asr_acc + top1_acc
                is_best = total_value >= best_asr
                best_asr = max(total_value, best_asr)

            save_model_dir = os.path.join(args.save, f"{seed}_saved_models")
            os.makedirs(save_model_dir, exist_ok=True)

            if is_best:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "best_auc": best_asr,
                    "state_dict": [model.state_dict() for model in model_list],
                    "optimizer": [opt.state_dict() for opt in optimizer_list]
                }, is_best, save_model_dir, f"checkpoint_{epoch:04d}.pth.tar")

                backdoor_data = {
                    "delta": convert_delta,
                    "source_label": source_label,
                    "target_label": target_label
                }
                torch.save(backdoor_data, os.path.join(save_model_dir, "backdoor.pth"))

        logger.info("Testing Best Model")
        checkpoint_path = os.path.join(args.save, f"{seed}_saved_models", "model_best.pth.tar")
        
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            for i, model in enumerate(model_list):
                model.load_state_dict(checkpoint["state_dict"][i])
        
        trainer.update_model(model_list)
        
        with open(os.path.join(args.save, "saved_result.txt"), "a") as file:
            sys.stdout = file
            test_loss, top1_acc, top5_acc = trainer.test(
                test_dataloader, criterion, args
            )
            _, asr_top1_acc, asr_top5_acc = trainer.test_backdoor(
                source_label_dataloader, criterion, convert_delta, target_label, args
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

            if seed == args.seed_num-1:
                print("Final Results Summary")
                print("Main Model Performance:")
                print(f"Top1: {Main_Top1_acc.avg:.2f}% ± {Main_Top1_acc.std_dev():.2f}%")
                print(f"Top5: {Main_Top5_acc.avg:.2f}% ± {Main_Top5_acc.std_dev():.2f}%")
                
                print("Backdoor Performance:")
                print(f"ASR Top1: {ASR_Top1.avg:.2f}% ± {ASR_Top1.std_dev():.2f}%")
                print(f"ASR Top5: {ASR_Top5.avg:.2f}% ± {ASR_Top5.std_dev():.2f}%")

            sys.stdout = sys.__stdout__

        print(f"Results for seed {seed} saved to file")

def test():
    """需要修改
    """
    # 加载模型
    save_model_dir = args.save
    checkpoint_path = save_model_dir + "/model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path)

    # 加载后门攻击信息
    backdoor_data = torch.load(save_model_dir + "/backdoor.pth")
    delta = backdoor_data.get("delta", None)
    target_label = backdoor_data.get("target_label", None)

    # load model
    model_list = []
    model_list.append(BottomModelForCinic10())
    model_list.append(BottomModelForCinic10())
    model_list.append(TopModelForCinic10())

    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss

    # 加载每个模型的参数
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint["state_dict"][i])

    # load data
    # Data normalization and augmentation (optional)
    transform = transforms.Compose(
        [
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )

    # Load CIFAR-10 dataset
    trainset = CINIC10L(root=args.data_dir, split="/train", transform=transform)
    testset = CINIC10L(root=args.data_dir, split="/test", transform=transform)

    target_indices = np.where(np.array(trainset.targets) == target_label)[0]
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
    vfltrainer = BadVFLTrainer(model_list)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    default_data_path = os.path.abspath("../../data/cinic/")
    default_save_path = os.path.abspath("../../results/models/BadVFL/cifar10")

    parser = argparse.ArgumentParser("badvfl_cifar10")

    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--data_dir", default=default_data_path, help="location of the data corpus")
    data_group.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "CINIC10L"], help="name of dataset")

    # 实验相关参数
    experiment_group = parser.add_argument_group('Experiment')
    experiment_group.add_argument("--name", type=str, default="badvfl_cifar10", help="experiment name")
    experiment_group.add_argument("--save", default=default_save_path, type=str, metavar="PATH", help="path to save checkpoint")
    experiment_group.add_argument("--log_file_name", type=str, default="experiment.log", help="log name")
    experiment_group.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
    experiment_group.add_argument("--seed_num", type=int, default=3, help="repeat num.")
    
    # 模型相关参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--layers", type=int, default=18, help="total number of layers")
    model_group.add_argument("--u_dim", type=int, default=64, help="u layer dimensions")
    model_group.add_argument("--k", type=int, default=2, help="num of clients")
    model_group.add_argument("--parallel", action="store_true", default=False, help="data parallelism")
    model_group.add_argument("--half", type=int, default=16, help="half number of features")
    
    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--epochs", type=int, default=80, help="num of training epochs")
    training_group.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--batch_size", type=int, default=256, help="batch size")
    training_group.add_argument("--lr", type=float, default=0.1, help="init learning rate")
    training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--stone1", default=30, type=int, metavar="s1", help="stone1 for step scheduler")
    training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--workers", type=int, default=0, help="num of workers")
    training_group.add_argument("--report_freq", type=float, default=10, help="report frequency")

    # 后门相关参数
    backdoor_group = parser.add_argument_group('Backdoor')
    training_group.add_argument("--trigger_lr", type=float, default=0.001, help="init learning rate for trigger")
    backdoor_group.add_argument("--alpha", type=float, default=0.02, help="uap learning rate decay")
    backdoor_group.add_argument("--eps", type=float, default=16 / 255, help="uap clamp bound")
    backdoor_group.add_argument("--corruption_amp", type=float, default=5.0, help="amplification of corruption")
    backdoor_group.add_argument("--backdoor_start", action="store_true", default=False, help="backdoor")
    backdoor_group.add_argument("--poison_budget", type=float, default=0.1, help="poison sample fraction")
    backdoor_group.add_argument("--optimal_sel", action="store_true", default=True, help="optimal select tartget class")
    backdoor_group.add_argument("--saliency_map_injection", action="store_true", default=True, help="optimal select trigger loaction")
    backdoor_group.add_argument("--pre_train_epochs", default=20, type=int, metavar="N", help="")
    backdoor_group.add_argument("--trigger_train_epochs", default=40, type=int, metavar="N", help="")
    backdoor_group.add_argument("--window_size", default=3, type=int, metavar="N", help="")
    
    
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


    args = parser.parse_args()
    args.device = device
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    args.save = os.path.join(args.save, timestamp)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)

    # 记录所有的参数信息
    logger.info(f"Experiment arguments: {args}")
    
    main(logger=logger, args=args)
