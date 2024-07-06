def main(device, args):
    for seed in range(1):
        set_seed(seed=seed)

        train_queue, test_queue, non_target_queue = set_dataset(args)

        (
            model_list,
            copied_model_list,
            delta,
            optimizer_list,
            lr_scheduler_list,
        ) = set_model_and_delta(args)

        criterion = nn.CrossEntropyLoss().to(device)
        bottom_criterion = keep_predict_loss

        vfl_trainer = VFLTrainer(model_list)
        vfl_shuffle_trainer = VFLTrainer(copied_model_list)

        print(
            "\n################################ Test Backdoor Models with TrainData ############################"
        )

        training_loss, top1_acc, top5_acc = vfl_trainer.test_mul(
            train_queue, criterion, device, args
        )
        print(
            "training_loss: ",
            training_loss,
            "top1_acc: ",
            top1_acc,
            "top5_acc: ",
            top5_acc,
        )

        print(
            "\n################################ Test Backdoor Models with TestData ############################"
        )

        test_loss, top1_acc, top5_acc = vfl_trainer.test_mul(
            test_queue, criterion, device, args
        )
        test_loss, test_asr_acc, _ = vfl_trainer.test_backdoor_mul(
            non_target_queue, criterion, device, args, delta, args.target_label
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

        print(
            "\n################################ Shuffle Training with TrainData ############################"
        )
        best_asr = 0.0
        for epoch in range(0, args.shuffle_epochs):
            logging.info("epoch %d args.lr %e ", epoch, args.lr)

            train_loss = vfl_shuffle_trainer.train_shuffle(
                train_queue, criterion, bottom_criterion, optimizer_list, device, args
            )

            lr_scheduler_list[0].step()
            lr_scheduler_list[1].step()
            lr_scheduler_list[2].step()

            test_loss, top1_acc, top5_acc = vfl_shuffle_trainer.test_mul(
                test_queue, criterion, device, args
            )
            _, test_asr_acc, _ = vfl_shuffle_trainer.test_backdoor_mul(
                non_target_queue, criterion, device, args, delta, target_label
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
