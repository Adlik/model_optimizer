# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
autoslim algorithm
"""
import copy
import time
from abc import abstractmethod
import yaml
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from ...core.utils import AverageMeter, ProgressMeter, accuracy


def _lr_scheduler_per_iteration(optimizer, lr, num_epochs, iters_per_epoch, epoch, batch_idx=1):
    """ function for learning rate scheuling per iteration
        epoch: from 0
    """
    current_iter = epoch * iters_per_epoch + batch_idx
    max_iter = iters_per_epoch * num_epochs

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (1 - current_iter / max_iter)


class AutoSlim(nn.Module):
    """
    AutoSlim algorithm
    """
    def __init__(self, model, pruner, distiller,
                 num_sample_training=4, input_shape=None, bn_training_mode=False,
                 retraining=False, channel_config_path=None, distributed=True):
        """

        Args:
            model: model processed by distributed_model
            pruner:
            distiller:
            num_sample_training:
            input_shape:
            bn_training_mode:
            retraining:
            channel_config_path:
            distributed:
        """
        super().__init__()
        assert num_sample_training >= 2, \
            'num_sample_training should be no less than 2'
        self.num_sample_training = num_sample_training
        self.bn_training_mode = bn_training_mode
        self.model = model
        if distributed:
            self.model_without_ddp = model.module
        else:
            self.model_without_ddp = model
        self.distributed = distributed
        self.retraining = retraining
        if channel_config_path:
            self.channel_config = self.load_subnet(channel_config_path)
        self._init_pruner(pruner)
        self.distiller = distiller
        if input_shape is not None:
            self.input_shape = input_shape

    @staticmethod
    def load_subnet(config_path):
        """
        load subnet
        Args:
            config_path:

        Returns:

        """
        file_format = config_path.split('.')[-1]
        if file_format in ["yaml", "yml"]:
            print(f"loading subnet config from {config_path}")
            with open(config_path, encoding='utf-8') as subnet_file:
                channel_config = yaml.load(subnet_file.read(), Loader=yaml.FullLoader)
                return channel_config
        else:
            raise NotImplementedError("Only yaml or yml file format of channel_config_path is supportted")

    def _init_pruner(self, pruner):
        """Build registered pruners and make preparations.

               Args:
                   pruner (dict): The registered pruner to be used
                       in the algorithm.
        """
        if pruner is None:
            self.pruner = None
            return
        else:
            self.pruner = pruner

        # judge whether our StructurePruner can prune the architecture
        try:
            pseudo_pruner = pruner
            pseudo_model = copy.deepcopy(self.model)
            if self.distributed:
                pseudo_without_ddp = pseudo_model.module
            else:
                pseudo_without_ddp = pseudo_model
            pseudo_pruner.prepare_from_supernet(pseudo_without_ddp)
            subnet_dict = pseudo_pruner.sample_subnet()
            pseudo_pruner.set_subnet(subnet_dict)
            subnet_dict = pseudo_pruner.export_subnet()

            # pseudo_pruner.deploy_subnet(pseudo_model, subnet_dict)
            pseudo_img = torch.randn(1, 3, 224, 224)
            pseudo_model(pseudo_img)
        except RuntimeError as runtime_error:
            raise NotImplementedError('Our current StructurePruner does not '
                                      'support pruning this architecture. '
                                      'StructurePruner is not perfect enough '
                                      'to handle all the corner cases. We will'
                                      ' appreciate it if you create a issue.') from runtime_error

        self.pruner.prepare_from_supernet(self.model_without_ddp)

    # pylint: disable=too-many-branches,too-many-statements
    def train_epoch(self, train_loader, criterion,
                    optimizer, epoch, args, writer, scaler):
        """
        train one epoch
        Args:
            train_loader:
            criterion:
            optimizer:
            epoch:
            args:
            writer:
            scaler:

        Returns:

        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        max_model_losses = AverageMeter('Max_model_loss', ':.4e')
        min_model_losses = AverageMeter('Min_model_loss', ':.4e')
        lr = AverageMeter('lr', ':.4e')

        progress = ProgressMeter(args.batch_num, batch_time, data_time, lr, max_model_losses, min_model_losses,
                                 prefix=f"Epoch: [{epoch + 1}]")
        if args.hp.multi_gpu.rank in [-1, 0]:
            print(f'gpu id: {args.gpu}')
        # switch to train mode
        self.train()

        end = time.time()
        base_step = epoch * args.batch_num
        for batch_index, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            batch_index = batch_index + 1
            optimizer.zero_grad()
            inputs = data[0]
            targets = data[1]
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            # compute supernet output
            self.pruner.set_max_channel()
            with torch.cuda.amp.autocast(enabled=args.hp.amp):
                soft_target = self.model(inputs)
                loss = criterion(soft_target, targets)
            max_model_losses.update(loss.item(), inputs.size(0))
            lr.update(optimizer.param_groups[0]['lr'])
            if writer is not None:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + batch_index)
            # compute gradient and do SGD step
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            soft_target = soft_target.detach()
            # compute min model output
            self.pruner.set_min_channel()
            with torch.cuda.amp.autocast(enabled=args.hp.amp):
                output_min = self.model(inputs)
                min_loss = self.distiller.compute_distill_loss(output_min, soft_target)
            min_model_losses.update(min_loss.item(), inputs.size(0))
            if scaler is not None:
                scaler.scale(min_loss).backward()
            else:
                min_loss.backward()

            for _ in range(self.num_sample_training - 2):
                subnet_dict = self.pruner.sample_subnet()
                self.pruner.set_subnet(subnet_dict)
                with torch.cuda.amp.autocast(enabled=args.hp.amp):
                    output = self.model(inputs)
                    model_loss = self.distiller.compute_distill_loss(output, soft_target)
                if scaler is not None:
                    scaler.scale(model_loss).backward()
                else:
                    model_loss.backward()

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # warning 1. backward 2. step 3. zero_grad
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % args.hp.print_freq == 0 and args.hp.multi_gpu.rank in [-1, 0]:
                progress.print(batch_index)

    # pylint: enable=too-many-branches,too-many-statements
    def retrain_epoch(self, train_loader, criterion, optimizer, epoch, args, writer):
        """
        retrain
        Args:
            train_loader:
            criterion:
            optimizer:
            epoch:
            args:
            writer:

        Returns:

        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        model_losses = AverageMeter('Model_loss', ':.4e')
        lr = AverageMeter('lr', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(args.batch_num, batch_time, data_time, lr, model_losses, top1, top5,
                                 prefix=f"Epoch: [{epoch + 1}]")
        print(f'gpu id: {args.gpu}')
        # switch to train mode
        self.train()

        end = time.time()
        base_step = epoch * args.batch_num
        for batch_index, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            batch_index = batch_index + 1
            _lr_scheduler_per_iteration(optimizer, args.hp.lr, args.hp.epochs, args.batch_num, epoch, batch_index)
            inputs = data[0]
            targets = data[1]
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = self.model(inputs)

            loss = criterion(outputs, targets)
            model_losses.update(loss.item(), inputs.size(0))
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            lr.update(optimizer.param_groups[0]['lr'])
            if writer is not None:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + batch_index)
                writer.add_scalar('train/acc1', top1.avg, base_step + batch_index)
                writer.add_scalar('train/acc5', top5.avg, base_step + batch_index)
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # warning 1. backward 2. step 3. zero_grad
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % args.hp.print_freq == 0:
                progress.print(batch_index)

    def train(self, mode=True):
        """Overwrite the train method in ``nn.Module`` to set ``nn.BatchNorm``
        to training mode when model is set to eval mode when
        ``self.bn_training_mode`` is ``True``.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.
        """
        super().train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True

    def get_subnet_flops(self):
        """A hacky way to get flops information of a subnet."""
        flops = 0
        last_out_mask_ratio = None
        for _, module in self.model_without_ddp.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                in_mask_ratio = float(module.in_mask.sum() / module.in_mask.numel())
                out_mask_ratio = float(module.out_mask.sum() / module.out_mask.numel())
                flops += module.__flops__ * in_mask_ratio * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif isinstance(module, nn.BatchNorm2d):
                out_mask_ratio = float(module.out_mask.sum() / module.out_mask.numel())
                flops += module.__flops__ * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                assert last_out_mask_ratio, 'An activate module can not be ' \
                                            'the first module of a network.'
                flops += module.__flops__ * last_out_mask_ratio

        return round(flops)

    @abstractmethod
    def forward(self, x):  # pylint: disable=missing-function-docstring
        pass
