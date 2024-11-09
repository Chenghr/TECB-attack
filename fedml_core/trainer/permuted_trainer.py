import copy

import torch
from fedml_core.trainer.tecb_trainer import TECBTrainer
from fedml_core.trainer.badvfl_trainer import BadVFLTrainer
from fedml_core.trainer.vfl_trainer import VFLTrainer


class PermutedTrainer(VFLTrainer):
    def __init__(self, model, args=None, attack_method="TECB"):
        super().__init__(model, args)
        
        # 创建模型副本用于扰动
        self.perturbed_model = copy.deepcopy(model)
        self.attack_method = attack_method
        
        if attack_method == "TECB":
            self.baseline_trainer = TECBTrainer(model)  # 基准训练器
            self.modified_trainer = TECBTrainer(self.perturbed_model)  # 扰动训练器
        elif attack_method == "BadVFL":
            self.baseline_trainer = BadVFLTrainer(model)
            self.modified_trainer = BadVFLTrainer(self.perturbed_model)
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
            
            # Update based on mode
            if args.update_mode in ['top_only', 'both']:
                # Update top model
                loss = self.update_model_one_batch(
                    optimizer=optimizer_list[2],
                    model=model_list[2],
                    output=output,
                    batch_target=target,
                    loss_func=criterion,
                    args=args,
                )
                grad_output_bottom_model_a = input_tensor_top_model_a.grad

            if args.update_mode in ['bottom_only', 'both']:
                # Update bottom model
                if args.update_mode == 'bottom_only':
                    # 如果只更新bottom，需要单独计算loss和梯度
                    loss = criterion(output, target)
                    loss.backward()
                    grad_output_bottom_model_a = input_tensor_top_model_a.grad
                
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
    
    def test_baseline_model(self, backdoor_data, test_dataloader, criterion, args, poison_test_dataloader=None):
        device = args.device
        
        if self.attack_method == "TECB":
            delta = backdoor_data.get("delta", None)
            target_label = backdoor_data.get("target_label", None) 
            
            _, baseline_clean_top1, baseline_clean_top5 = self.baseline_trainer.test(
                test_dataloader, criterion, device, args
            )
            _, baseline_asr_top1, baseline_asr_top5 = self.baseline_trainer.test_backdoor(
                test_dataloader, criterion, device, args, delta, target_label
            )
        elif self.attack_method == "BadVFL":
            delta = backdoor_data.get("delta", None)
            target_label = backdoor_data.get("target_label", None)
            best_position = backdoor_data.get("best_position", None)
            
            _, baseline_clean_top1, baseline_clean_top5 = self.baseline_trainer.test(
                test_dataloader, criterion, args
            )
            _, baseline_asr_top1, baseline_asr_top5 = self.baseline_trainer.test_backdoor(
                poison_test_dataloader, criterion, delta, best_position, target_label, args
            )
        else:
            raise ValueError
        
        return baseline_clean_top1, baseline_clean_top5, baseline_asr_top1, baseline_asr_top5
        
    def test_modified_model(self, backdoor_data, test_dataloader, criterion, args, poison_test_dataloader=None):
        device = args.device
        self.modified_trainer.update_model(self.perturbed_model)
        
        if self.attack_method == "TECB":
            delta = backdoor_data.get("delta", None)
            target_label = backdoor_data.get("target_label", None) 
            
            _, modified_clean_top1, modified_clean_top5 = self.modified_trainer.test(
                test_dataloader, criterion, device, args
            )
            _, modified_asr_top1, modified_asr_top5 = self.modified_trainer.test_backdoor(
                test_dataloader, criterion, device, args, delta, target_label
            )
        elif self.attack_method == "BadVFL":
            delta = backdoor_data.get("delta", None)
            target_label = backdoor_data.get("target_label", None)
            best_position = backdoor_data.get("best_position", None)
            
            _, modified_clean_top1, modified_clean_top5 = self.modified_trainer.test(
                test_dataloader, criterion, args
            )
            _, modified_asr_top1, modified_asr_top5 = self.modified_trainer.test_backdoor(
                poison_test_dataloader, criterion, delta, best_position, target_label, args
            )
        else:
            raise ValueError

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
    
    @staticmethod
    def update_optimizer_for_layers(model_list, optimizer_list, args):
        """
        为指定的层更新优化器
        
        Args:
            model_list: 模型集合
            optimizer_list: 优化器列表
            args: 包含更新层设置的参数
        
        Returns:
            optimizer_list: 更新后的优化器列表
        """
        params_to_update = []
        
        # 确保参数存在且有效
        if not hasattr(args, 'update_top_layers'):
            print("Warning: update_top_layers not found in args, using default 'all'")
            args.update_top_layers = ['all']
        
        for name, param in model_list[2].named_parameters():
            should_update = False
            
            if args.update_top_layers == ['all'] or args.update_top_layers == 'all':
                should_update = True
            elif isinstance(args.update_top_layers, str):
                should_update = args.update_top_layers in name
            elif isinstance(args.update_top_layers, (list, tuple)):
                should_update = any(layer in name for layer in args.update_top_layers)
            
            if should_update:
                param.requires_grad = True
                params_to_update.append(param)
                print(f"Layer {name} will be updated")
            else:
                param.requires_grad = False
                print(f"Layer {name} will be frozen")
        
        # 检查是否有参数要更新
        if not params_to_update:
            print("Warning: No parameters selected for update!")
        
        # 重新定义优化器
        if hasattr(args, 'lr'):
            optimizer_list[2] = torch.optim.Adam(params_to_update, lr=args.lr)
        else:
            print("Warning: Learning rate not found in args!")
            optimizer_list[2] = torch.optim.Adam(params_to_update)
        
        return optimizer_list