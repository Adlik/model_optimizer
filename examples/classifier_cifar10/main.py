# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cifar10 train
"""
import time
import datetime
import os
import torchvision
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.backends import cudnn
import torch.distributed as dist
from torch import nn
import model_optimizer.models.cifar10 as cifar10_extra_models
from model_optimizer.core import (get_base_parser, get_hyperparam, get_freer_gpu, main_s1_set_seed,
                                  main_s2_start_worker, display_model, process_model,
                                  distributed_model, get_optimizer, get_summary_writer,
                                  get_model_info, get_lr_scheduler, validate, train, save_checkpoint)
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.datasets import DataloaderFactory

best_acc1 = 0


def main():
    """
    main entry
    Returns:

    """
    parser = get_base_parser()
    args = parser.parse_args()
    hp = get_hyperparam(args)
    if hp.gpu_id == eppb.GPU.ANY:
        args.gpu = get_freer_gpu()
    elif hp.gpu_id == eppb.GPU.NONE:
        args.gpu = None  # TODO: test

    print("Start training")
    start_time = time.time()
    main_s1_set_seed(hp)
    main_s2_start_worker(main_worker, args, hp)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


def _get_model_from_source(args):
    # create model
    if args.hp.pretrained:
        print(f"=> using pre-trained model '{args.hp.arch}'")
    else:
        print(f"=> creating model '{args.hp.arch}'")
    if args.hp.model_source == eppb.HyperParam.ModelSource.TorchVision:
        model = torchvision.models.__dict__[args.hp.arch](pretrained=args.hp.pretrained, num_classes=10)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.PyTorchCV:
        model = ptcv_get_model(args.hp.arch, pretrained=args.hp.pretrained, num_classes=10)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.Local:
        model = cifar10_extra_models.__dict__[args.hp.arch](pretrained=args.hp.pretrained, num_classes=10)
    else:
        raise NotImplementedError
    return model


def main_worker(gpu, ngpus_per_node, args):  # pylint: disable=too-many-branches,too-many-statements
    """
    main worker one gpu
    Args:
        gpu:
        ngpus_per_node:
        args:

    Returns:

    """
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    args.hp = get_hyperparam(args)
    if args.distributed:
        if args.hp.multi_gpu.dist_url == "env://" and args.hp.multi_gpu.rank == -1:
            args.hp.multi_gpu.rank = int(os.environ["RANK"])
        if args.hp.multi_gpu.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.hp.multi_gpu.rank = args.hp.multi_gpu.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.hp.multi_gpu.dist_backend, init_method=args.hp.multi_gpu.dist_url,
                                world_size=args.world_size, rank=args.hp.multi_gpu.rank)

    model = _get_model_from_source(args)

    if args.gpu == 0:
        print(args)
    print('model:\n=========\n')
    display_model(model)

    process_model(model, args)
    global best_acc1  # pylint: disable=global-statement
    best_acc1 = args.best_acc1
    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.hp.label_smoothing).cuda(args.gpu)
    optimizer = get_optimizer(model, args)
    cudnn.benchmark = True

    dataload_factory = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = dataload_factory.product_train_val_loader(dataload_factory.cifar10)
    writer = get_summary_writer(args, ngpus_per_node, model)
    if args.hp.evaluate:
        if writer is not None:
            get_model_info(model, args, val_loader)
    args.batch_num = len(train_loader)

    scheduler_lr = get_lr_scheduler(optimizer, args)

    if args.hp.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(0, args.start_epoch):
        scheduler_lr.step()
        pass
    for epoch in range(args.start_epoch, args.hp.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        epoch_start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        scheduler_lr.step()
        epoch_total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print(f'Epoch[{epoch}] total time {total_time_str}')
        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if writer is not None:
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)
            writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
        # remember best acc@1 and save checkpoint

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if writer is not None:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, prefix=f'{args.log_name}/{args.arch}')
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
