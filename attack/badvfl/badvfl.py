import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from attack.badvfl.utils import (get_delta_exten, get_poison_test_dataloader,
                                 get_poison_train_dataloader, init_dataloader,
                                 init_model_releated, load_model_and_backdoor_data,
                                 sample_poisoned_source_target_data,
                                 save_checkpoint, set_seed)
from fedml_core.trainer.badvfl_trainer import BadVFLTrainer
from fedml_core.utils.logger import setup_logger
from fedml_core.utils.utils import (AverageMeter, keep_predict_loss,
                                    over_write_args_from_file)


def train(args, logger):
    ASR_Top1 = AverageMeter()
    Main_Top1_acc = AverageMeter()
    Main_Top5_acc = AverageMeter()
    
    for seed in range(args.seed_num):
        set_seed(seed)
        
        save_model_dir = args.save + f"/seed={seed}_saved_models"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        # 加载基础数据
        train_dataloader, train_dataloader_nobatch, test_dataloader = init_dataloader(
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
                train_dataloader, criterion, optimizer_list, args, 
            )
            pre_train_loss.append(loss)
        logger.info(f"Pre-Train Loss: [{', '.join([f'{l:.4f}' for l in pre_train_loss])}]")
        
        # Optimal select label
        source_label, target_label = trainer.select_closest_class_pair(
            train_dataloader_nobatch, args
        )
        selected_source_indices, selected_target_indices = sample_poisoned_source_target_data(
            train_dataloader.dataset, source_label, target_label, args.poison_budget
        )
        logger.info(f"source_label: {source_label}, target_label: {target_label}")
        
        # Set trigger
        logger.info("###### Train Trigger ######") 
        print(len(selected_source_indices))
        best_position = trainer.find_optimal_trigger_position(
            train_dataloader_nobatch, selected_source_indices, criterion, optimizer_list, args
        )
        
        delta = torch.full((1, 3, args.window_size, args.window_size), 0.0).to(args.device)
        
        trigger_optimizer = torch.optim.SGD([delta], 0.25)
        trigger_loss= []
        for _ in range(args.trigger_train_epochs):
            delta, loss = trainer.train_trigger(
                train_dataloader_nobatch, selected_source_indices, selected_target_indices,best_position, delta, trigger_optimizer, args
            )
            trigger_loss.append(loss)
        logger.info(f"Trigger Train Loss: [{', '.join([f'{l:.4f}' for l in trigger_loss])}]")
        logger.info(f"best_position: {best_position}, delta: {delta}")
        
        # Set poison data
        dataset = copy.deepcopy(train_dataloader.dataset)
        delta_exten = get_delta_exten(
            args.dataset, dataset, delta, best_position, selected_target_indices, args.window_size, args.half, args.device)
        poison_train_dataloader = get_poison_train_dataloader(
            args.dataset, args.data_dir, args.batch_size, selected_source_indices, selected_target_indices, delta_exten
        )
        poison_source_test_dataloader = get_poison_test_dataloader(
            args.dataset, args.data_dir, args.batch_size, source_label
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
            _, test_asr_acc, _ = trainer.test_backdoor(
                poison_source_test_dataloader, criterion, delta, best_position, target_label, args
            )

            print(f"Epoch: {epoch + 1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Top1 Acc: {top1_acc:.2f}% | "
                  f"Top5 Acc: {top5_acc:.2f}% | "
                  f"ASR Top1: {test_asr_acc:.2f}% ")

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
                    "source_label": source_label,
                    "target_label": target_label,
                    "delta": delta,
                    "best_position": best_position,
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
            _, asr_top1_acc, _ = trainer.test_backdoor(
                poison_source_test_dataloader, criterion, delta, best_position, target_label, args
            )

            print("\nTest Results (Seed {})".format(seed))
            print("Main Task Metrics:")
            print(f"Loss: {test_loss:.4f} | "
                  f"Top1: {top1_acc:.2f}% | "
                  f"Top5: {top5_acc:.2f}%")

            print("Backdoor Task Metrics:")
            print(f"ASR Top1: {asr_top1_acc:.2f}% \n")

            Main_Top1_acc.update(top1_acc)
            Main_Top5_acc.update(top5_acc)
            ASR_Top1.update(asr_top1_acc)

            if seed == args.seed_num-1:
                print("Final Results Summary")
                print("Main Model Performance:")
                print(f"Top1: {Main_Top1_acc.avg:.2f}% ± {Main_Top1_acc.std_dev():.2f}%")
                print(f"Top5: {Main_Top5_acc.avg:.2f}% ± {Main_Top5_acc.std_dev():.2f}%")
                
                print("Backdoor Performance:")
                print(f"ASR Top1: {ASR_Top1.avg:.2f}% ± {ASR_Top1.std_dev():.2f}%")

            sys.stdout = sys.__stdout__

        print(f"Results for seed {seed} saved to file")

def test(args):
    model_list, backdoor_data = load_model_and_backdoor_data(args.dataset, args.save)

    delta = backdoor_data.get("delta", None)
    source_label = backdoor_data.get("source_label", None)
    target_label = backdoor_data.get("target_label", None)
    delta = backdoor_data.get("delta", None)
    best_position = backdoor_data.get("best_position", None)
    
    _, _, test_dataloader = init_dataloader(
        args.dataset, args.data_dir, args.batch_size
    )
    poison_source_test_dataloader = get_poison_test_dataloader(
        args.dataset, args.data_dir, args.batch_size, source_label
    )
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    trainer = BadVFLTrainer(model_list)
    
    test_loss, top1_acc, top5_acc = trainer.test(
        test_dataloader, criterion, args
    )
    _, asr_top1_acc, _ = trainer.test_backdoor(
        poison_source_test_dataloader, criterion, delta, best_position, target_label, args
    )
    
    print("Main Task Metrics:")
    print(f"Loss: {test_loss:.4f} | "
            f"Top1: {top1_acc:.2f}% | "
            f"Top5: {top5_acc:.2f}%")

    print("Backdoor Task Metrics:")
    print(f"ASR Top1: {asr_top1_acc:.2f}% \n")
    

if __name__ == "__main__":
    
    default_data_path = os.path.abspath("../../data/")
    default_yaml_path = os.path.abspath("../badvfl/best_configs/cifar10_bestattack.yml")
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
    experiment_group.add_argument("--yaml_path", type=str, default=default_yaml_path, help="attack yaml file")
    experiment_group.add_argument("--load_yaml", action="store_true", default=False, help="backdoor")
    experiment_group.add_argument("--device", type=str, default="cuda:0")
    
    # 模型相关参数
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--layers", type=int, default=18, help="total number of layers")
    model_group.add_argument("--u_dim", type=int, default=64, help="u layer dimensions")
    model_group.add_argument("--k", type=int, default=2, help="num of clients")
    model_group.add_argument("--parallel", action="store_true", default=False, help="data parallelism")
    model_group.add_argument("--half", type=int, default=16, help="half number of features")
    
    # 训练相关参数
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--epochs", type=int, default=60, help="num of training epochs")
    training_group.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--backdoor_start_epoch", default=20, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    training_group.add_argument("--batch_size", type=int, default=64, help="batch size")
    training_group.add_argument("--lr", type=float, default=0.02, help="init learning rate")
    training_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
    training_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    training_group.add_argument("--decay_period", type=int, default=1, help="epochs between two learning rate decays")
    training_group.add_argument("--stone1", default=30, type=int, metavar="s1", help="stone1 for step scheduler")
    training_group.add_argument("--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler")
    training_group.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    training_group.add_argument("--gamma", type=float, default=0.97, help="learning rate decay")
    training_group.add_argument("--step_gamma", default=0.1, type=float, metavar="S", help="gamma for step scheduler")
    training_group.add_argument("--workers", type=int, default=8, help="num of workers")
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
    backdoor_group.add_argument("--window_size", default=5, type=int, metavar="N", help="")
    
    
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
    if args.load_yaml:
        over_write_args_from_file(args, args.yaml_path)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device
    
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # args.timestamp = timestamp
    # args.save = os.path.join(args.save, timestamp)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = setup_logger(args)

    # 记录所有的参数信息
    logger.info(f"Experiment arguments: {args}")
    
    train(logger=logger, args=args)
