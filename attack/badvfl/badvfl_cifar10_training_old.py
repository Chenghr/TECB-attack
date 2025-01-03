import logging
import os
import random
import sys
import copy
import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# from fedml_core.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_two_party_data
from fedml_core.data_preprocessing.cifar10.dataset import IndexedCIFAR10
from fedml_core.model.vfl_models import (
    BottomModelForCifar10,
    TopModelForCifar10,
    LocalClassifierForCifar10,
)
from fedml_core.trainer.badvfl_trainer_old import BadVFLTrainer
from fedml_core.utils.utils import (
    AverageMeter,
    keep_predict_loss,
    over_write_args_from_file,
)

# from fedml_api.utils.utils import save_checkpoint
import torch
import torch.nn as nn
import argparse
import time
import glob
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torch.utils.data import Subset
import logging


def save_checkpoint(state, is_best, save, checkpoint):
    filename = os.path.join(save, checkpoint)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def main(device, args):
    ASR_Top1 = AverageMeter()
    ASR_Top5 = AverageMeter()
    Main_Top1_acc = AverageMeter()
    Main_Top5_acc = AverageMeter()

    for seed in range(5):
        # random seed for 10 runs
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # load data
        # Data normalization and augmentation (optional)
        train_transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2471, 0.2435, 0.2616))
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
            root="./data", train=True, download=True, transform=train_transform
        )
        testset = IndexedCIFAR10(
            root="./data", train=False, download=True, transform=train_transform
        )

        # CIFAR-10 类别标签（以类别名称的列表形式给出）
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        train_queue = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
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

        # build model
        model_list = []
        model_list.append(BottomModelForCifar10())
        model_list.append(BottomModelForCifar10())
        model_list.append(TopModelForCifar10())
        model_list.append(BottomModelForCifar10())

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

        stone1 = args.stone1

        lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_list[2], milestones=[stone1], gamma=args.step_gamma
        )
        lr_scheduler_a = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_list[0], milestones=[stone1], gamma=args.step_gamma
        )
        lr_scheduler_b = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_list[1], milestones=[stone1], gamma=args.step_gamma
        )
        # change the lr_scheduler to the one you want to use
        lr_scheduler_list = [lr_scheduler_a, lr_scheduler_b, lr_scheduler_top_model]

        badvfltrainer = BadVFLTrainer(model_list)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss
        print(
            "################################ Pre-Trained ############################"
        )
        for epoch in range(args.pre_train_epochs):
            print("pre-train epoch %d args.lr %e ", epoch, args.lr)
            train_loss = badvfltrainer.pre_train(
                train_queue,
                criterion,
                bottom_criterion,
                optimizer_list,
                device,
                args,
            )
            print(train_loss)
        source_class, target_class = badvfltrainer.pairwise_distance_min(train_queue_nobatch, device, args)
        print(source_class, target_class)
        source_label = classes.index(source_class)
        target_label = classes.index(target_class)
        source_indices = np.where(np.array(trainset.targets) == source_label)[0]
        target_indices = np.where(np.array(trainset.targets) == target_label)[0]
        target_num = np.sum(np.array(trainset.targets) == target_label)
        poison_num = int(args.poison_budget * target_num)
        selected_source_indices = np.sort(np.random.choice(source_indices, poison_num, replace=False))
        selected_target_indices = np.sort(np.random.choice(target_indices, poison_num, replace=False))
        selected_source_set = Subset(trainset, selected_source_indices)
        selected_source_queue = torch.utils.data.DataLoader(
            dataset=selected_source_set,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        print(
            "################################ Train Trigger ############################"
        ) 
        best_position = badvfltrainer.optimal_trigger_injection(train_queue_nobatch, selected_source_indices, criterion, optimizer_list, device, args)
        _, (x_val, y_val, index) = next(enumerate(train_queue))
        print(best_position)
        delta = torch.full((1, 3, args.window_size, args.window_size), 0.0).to(device)
        for epoch in range(args.trigger_train_epochs):
            trigger_optimizer = torch.optim.SGD([delta], 0.25)
            delta = badvfltrainer.train_trigger(train_queue_nobatch, device, selected_source_indices, 
                                            selected_target_indices, delta, best_position, trigger_optimizer, args)
            print(delta) 
        poison_train_set = copy.deepcopy(trainset)
        delta_upd = delta.permute(0, 2, 3, 1)
        delta_upd = delta_upd * 255.0
        delta_upd = delta_upd.to(torch.uint8)
        by = best_position[0]
        bx = best_position[1]
        delta_exten = torch.zeros_like(torch.from_numpy(poison_train_set.data[selected_target_indices])).to(device)
        delta_exten[:, by : by + args.window_size, bx + args.half : bx + args.window_size + args.half, :] = delta_upd.expand(500, -1, -1, -1).detach().clone()
        poison_train_set.data[selected_target_indices] = np.clip(trainset.data[selected_source_indices] + delta_exten.cpu().numpy(), 0, 255)
        poison_train_queue = torch.utils.data.DataLoader(
            dataset=poison_train_set,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        poison_test_set = copy.deepcopy(testset)
        poison_test_queue = torch.utils.data.DataLoader(
            dataset=poison_test_set,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        non_target_indices = np.where(np.array(testset.targets) == source_label)[0]
        source_poison_testset = Subset(poison_test_set, non_target_indices)
        poison_source_test_queue = torch.utils.data.DataLoader(
            dataset=source_poison_testset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=device)
                args.start_epoch = checkpoint["epoch"]
                for i in range(len(model_list)):
                    model_list[i].load_state_dict(checkpoint["state_dict"][i])
                    # optimizer_list[i].load_state_dict(checkpoint['optimizer'][i])
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

        # Set a 9-pixel pattern to 1
        # delta[:, 0:3, 0:3] = 1
        for epoch in range(args.start_epoch, args.epochs):
            logging.info("epoch %d args.lr %e ", epoch, args.lr)

            # train_loss, delta = vfltrainer.train_narcissus(train_queue, criterion, bottom_criterion,optimizer_list, device, args, delta, selected_indices, trigger_optimizer)
            if args.backdoor_start:
                train_loss = badvfltrainer.train_mul(
                        poison_train_queue,
                        criterion,
                        bottom_criterion,
                        optimizer_list,
                        device,
                        args,
                )
            else:
                train_loss = badvfltrainer.train_mul(
                    train_queue,
                    criterion,
                    bottom_criterion,
                    optimizer_list,
                    device,
                    args,
                )

            lr_scheduler_list[0].step()
            lr_scheduler_list[1].step()
            lr_scheduler_list[2].step()

            test_loss, top1_acc, top5_acc = badvfltrainer.test_mul(
                test_queue, criterion, device, args
            )
            _, test_asr_acc, _ = badvfltrainer.test_backdoor_mul(
                poison_source_test_queue, criterion, device, args, delta, best_position, target_label
            )

            print(
                "epoch:",
                epoch + 1,
                "train_loss:",
                train_loss,
                "test_loss: ",
                test_loss,
                "top1_acc: ",
                top1_acc,
                "top5_acc: ",
                top5_acc,
                "test_asr_acc: ",
                test_asr_acc,
            )

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

                torch.save(delta, os.path.join(save_model_dir, "delta.pth"))

        # test
        print(
            "##################################test############################################"
        )

        # load best model and test on test set
        checkpoint_path = args.save + f"/{seed}_saved_models" + "/model_best.pth.tar"
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            for i in range(len(model_list)):
                model_list[i].load_state_dict(checkpoint["state_dict"][i])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

        badvfltrainer.update_model(model_list)

        txt_name = f"saved_result"
        savedStdout = sys.stdout
        with open(args.save + "/" + txt_name + ".txt", "a") as file:
            sys.stdout = file
            test_loss, top1_acc, top5_acc = badvfltrainer.test_mul(
                test_queue, criterion, device, args
            )

            _, asr_top1_acc, _ = badvfltrainer.test_backdoor_mul(
                poison_source_test_queue, criterion, device, args, delta, best_position, target_label
            )
            print(
                "################################ Test each seed ############################"
            )
            print(
                "################################ Main Task ############################"
            )
            print(
                "--- epoch: {0}, seed: {1},test_loss: {2}, test_top1_acc: {3}, test_top5_acc: {4} ---".format(
                    epoch, seed, test_loss, top1_acc, top5_acc
                )
            )

            print(
                "################################ Backdoor Task ############################"
            )
            print(
                "--- epoch: {0}, seed: {1}, test_loss: {2}, test_asr_top1_acc: {3} ---".format(
                    epoch, seed, test_loss, asr_top1_acc
                )
            )

            print(
                "################################ End Task ############################"
            )
            print(
                "######################################################################"
            )

            Main_Top1_acc.update(top1_acc)
            Main_Top5_acc.update(top5_acc)
            ASR_Top1.update(asr_top1_acc)
            # ASR_Top5.update(asr_top5_acc)

            if seed == 4:
                print(
                    "################################ Final Result ############################"
                )

                print(
                    "################################ Main Model ############################"
                )
                print(
                    "Main AVG Top1 acc: ",
                    Main_Top1_acc.avg,
                    "Main STD Top1 acc: ",
                    Main_Top1_acc.std_dev(),
                    "Main AVG Top5 acc: ",
                    Main_Top5_acc.avg,
                    "Main STD Top5 acc: ",
                    Main_Top5_acc.std_dev(),
                )

                print(
                    "################################ Backdoor Model ############################"
                )
                print(
                    "ASR AVG Top1 acc: ",
                    ASR_Top1.avg,
                    "ASR STD Top1 acc: ",
                    ASR_Top1.std_dev(),
                )

            sys.stdout = savedStdout

        print("Last epoch evaluation saved to txt!")


if __name__ == "__main__":
    print("################################ Prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser("vflmodelnet")

    parser.add_argument(
        "--data_dir", default="./data/CIFAR10/", help="location of the data corpus"
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
        default="./model/CIFAR10/baseline",
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
        "--stone1", default=30, type=int, metavar="s1", help="stone1 for step scheduler"
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
    # parser.add_argument(
    #     "--target_class", type=str, default="cat", help="backdoor target class"
    # )
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

    # config file
    parser.add_argument(
        "--c",
        type=str,
        default="configs/BadVFL/cifar10_test.yml",
        help="config file",
    )

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

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

    main(device=device, args=args)

    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)
import logging
import os
import logging
import os
