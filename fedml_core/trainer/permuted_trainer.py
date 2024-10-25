import copy

import torch
from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.trainer.vfl_trainer import VFLTrainer


class PermutedTrainer(VFLTrainer):
    def __init__(self, model, args=None, attack_method="TECB"):
        super().__init__(model, args)
        
        # 创建模型副本用于扰动
        self.perturbed_model = copy.deepcopy(model)
        
        if attack_method == "TECB":
            self.baseline_trainer = TECBTrainer(model)  # 基准训练器
            self.modified_trainer = TECBTrainer(self.perturbed_model)  # 扰动训练器
        elif attack_method == "BadVFL":
            ...
        else:
            raise ValueError(f"Unsupported attack method: {attack_method}")
    
    def train_perturbed(
        self, train_data, criterion, bottom_criterion, optimizer_list, device, args
    ):
        """仅更新主动方的模型，即第一个和最后一个模型"""
        model_list = [model.to(device).train() for model in self.perturbed_model]
        num_classes = 10 if args.dataset in ["CIFAR10", "CINIC10L"] else 100
        
        batch_loss = []
        for step, (trn_X, trn_y, indices) in enumerate(train_data):
            if args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
                trn_X = trn_X.float().to(device)
                Xa, Xb = self.split_data(trn_X, args)
                target = trn_y.long().to(device)
            else:
                Xa = trn_X[0].float().to(device)
                Xb = trn_X[1].float().to(device)
                target = trn_y.long().to(device)
            
            # 使用随机扰动后的数据集进行训练
            target = self.randomly_perturb_labels(target, num_classes)
            
            output_tensor_bottom_model_a = model_list[0](Xa)
            output_tensor_bottom_model_b = model_list[1](Xb)
            
            input_tensor_top_model_a = output_tensor_bottom_model_a.detach().clone()
            input_tensor_top_model_b = output_tensor_bottom_model_b.detach().clone()
            input_tensor_top_model_a.requires_grad_(True)
            input_tensor_top_model_b.requires_grad_(False)

            # top model
            output = model_list[2](input_tensor_top_model_a, input_tensor_top_model_b)
            
            # --top model backward/update--
            loss = self.update_model_one_batch(
                optimizer=optimizer_list[2],
                model=model_list[2],
                output=output,
                batch_target=target,
                loss_func=criterion,
                args=args,
            )

            grad_output_bottom_model_a = input_tensor_top_model_a.grad

            # -- bottom model a (active party) backward/update--
            _ = self.update_model_one_batch(
                optimizer=optimizer_list[0],
                model=model_list[0],
                output=output_tensor_bottom_model_a,
                batch_target=grad_output_bottom_model_a,
                loss_func=bottom_criterion,
                args=args,
            )

            batch_loss.append(loss.item())

        epoch_loss = sum(batch_loss) / len(batch_loss)

        return epoch_loss
    
    def test_baseline_model(self, test_data, criterion, device, args, delta, target_label):
        _, baseline_clean_top1, baseline_clean_top5 = self.baseline_trainer.test(
            test_data, criterion, device, args
        )
        _, baseline_asr_top1, baseline_asr_top5 = self.baseline_trainer.test_backdoor(
            test_data, criterion, device, args, delta, target_label
        )
        return baseline_clean_top1, baseline_clean_top5, baseline_asr_top1, baseline_asr_top5
        
    def test_modified_model(self, test_data, criterion, device, args, delta, target_label):
        self.modified_trainer.update_model(self.perturbed_model)
        _, modified_clean_top1, modified_clean_top5 = self.modified_trainer.test(
            test_data, criterion, device, args
        )
        _, modified_asr_top1, modified_asr_top5 = self.modified_trainer.test_backdoor(
            test_data, criterion, device, args, delta, target_label
        )
        return modified_clean_top1, modified_clean_top5, modified_asr_top1, modified_asr_top5
    
    @staticmethod
    def randomly_perturb_labels(target, num_classes=10):
        """随机扰动标签，将原始标签替换为不同的标签
        
        Parameters:
        -----------
        target : torch.Tensor
            原始标签张量
        num_classes : int, optional
            类别总数，默认为10
            
        Returns:
        --------
        torch.Tensor
            扰动后的标签张量
        """
        # 获取与每个标签不同的随机标签
        mask = torch.randint(0, num_classes-1, target.shape, device=target.device)
        # 将大于等于原始标签的值+1，确保新标签与原始标签不同
        perturbed_target = mask + (mask >= target).long()
        
        return perturbed_target