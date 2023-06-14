# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
quantization train with PTQ or QAT
"""
import time
import copy
import datetime
import os
from torch.backends import cudnn
import torch.distributed as dist
from torch import nn
import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx, fuse_fx
from model_optimizer.core import (get_base_parser, get_hyperparam, get_freer_gpu, main_s1_set_seed,
                                  main_s2_start_worker, display_model, process_model, resume_model,
                                  distributed_model, get_optimizer, get_summary_writer,
                                  get_model_info, get_lr_scheduler, validate, train, save_checkpoint,
                                  save_torchscript_model, save_file_on_master, save_yaml_file_on_master,
                                  print_on_master)
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.quantizer import (get_qconfig, get_lsq_qconfig, get_model_quantize_bit_config,
                                       clip_model_weight_in_quant_min_max)
from model_optimizer.losses import KLDivergence
from model_optimizer.datasets import DataloaderFactory
from model_optimizer.models import get_model_from_source, get_teacher_model

best_acc1 = 0


def main():
    """
    main process
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


def main_worker(gpu, ngpus_per_node, args):  # pylint: disable=too-many-branches,too-many-statements
    """
    main worker on per gpu
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

    if args.hp.quantization.post_training_quantize and args.distributed:
        raise RuntimeError("Post training quantization should not be performed "
                           "on distributed mode")

    if args.distributed:
        if args.hp.multi_gpu.dist_url == "env://" and args.hp.multi_gpu.rank == -1:
            args.hp.multi_gpu.rank = int(os.environ["RANK"])
        if args.hp.multi_gpu.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.hp.multi_gpu.rank = args.hp.multi_gpu.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.hp.multi_gpu.dist_backend, init_method=args.hp.multi_gpu.dist_url,
                                world_size=args.world_size, rank=args.hp.multi_gpu.rank)
    # create model
    if args.hp.pretrained:
        print(f"=> using pre-trained model '{args.hp.arch}'")
    else:
        print(f"=> creating model '{args.hp.arch}'")
    load_quantize_model = args.hp.quantization.quantize and (not args.hp.quantization.quantize_fx)
    model = get_model_from_source(args.hp.arch, args.hp.model_source, args.hp.pretrained, args.hp.width_mult,
                                  args.hp.depth_mult, load_quantize_model, args.hp.is_subnet,
                                  args.hp.auto_slim.channel_config_path if args.hp.HasField('auto_slim') else None)
    # Teacher model
    if args.hp.distill.teacher_model.arch:
        teacher_model = get_teacher_model(args.hp.distill.teacher_model.arch, args.hp.distill.teacher_model.source)
    else:
        teacher_model = None
    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        print(args)
        print('model:\n=========\n')
        display_model(model)

    process_model(model, args)

    if args.hp.quantization.quantize:
        torch.backends.quantized.engine = args.hp.quantization.backend

    if args.hp.quantization.quantize and not args.hp.quantization.post_training_quantize:
        if args.hp.quantization.quantize_fx:
            qconfig_dict = get_lsq_qconfig(model, args.hp.quantization, args.hp.quantization.backend)
            model = prepare_qat_fx(model, qconfig_dict)
        else:
            qconfig = get_qconfig(args.hp.quantization, args.hp.quantization.backend)
            model.fuse_model()
            model.qconfig = qconfig
            torch.quantization.prepare_qat(model, inplace=True)

        df_calibrate = DataloaderFactory(args)
        calibrate_train_loader, _, _ = df_calibrate.product_train_val_loader(
            df_calibrate.imagenet2012,
            num_batches=1,
            use_val_trans=True,
            batch_same_data=True)
        # simple_train_loader = df_simple.product_train_loader(df_simple.imagenet2012)
        model.apply(torch.quantization.enable_observer)
        model.apply(torch.quantization.disable_fake_quant)

        print('init scale for lsq')
        model.apply(torch.quantization.enable_observer)
        with torch.no_grad():
            for images, _ in calibrate_train_loader:
                model(images)
        del calibrate_train_loader
        del df_calibrate
        torch.distributed.barrier()
    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.hp.distill.teacher_model.arch:
        teacher_model = distributed_model(teacher_model, ngpus_per_node, args)
        # define distillation loss function (distill_criterion)
        distill_criterion = KLDivergence(args.hp.distill.kl_divergence.temperature,
                                         loss_weight=args.hp.distill.kl_divergence.loss_weight)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.hp.label_smoothing).cuda(args.gpu)

    optimizer = get_optimizer(model, args)
    cudnn.benchmark = True

    if args.hp.quantization.quantize and args.hp.quantization.post_training_quantize:

        dataload_factory = DataloaderFactory(args)
        data_loader_calibration, val_loader, train_sampler = dataload_factory.product_train_val_loader(
            dataload_factory.imagenet2012,
            args.hp.quantization.num_calibration_batches, use_val_trans=True)
        model.eval()
        qconfig = get_qconfig(args.hp.quantization, args.hp.quantization.backend)
        if args.hp.quantization.quantize_fx:
            qconfig_dict = {"": qconfig}
            model = prepare_fx(model, qconfig_dict)
        else:
            model.fuse_model()
            model.qconfig = qconfig
            torch.quantization.prepare(model, inplace=True)
        validate(data_loader_calibration, model, criterion, args)
        model.to(torch.device('cpu'))
        writer = get_summary_writer(args, ngpus_per_node, model)
        if args.hp.quantization.quantize_fx:
            if args.hp.save_jit_trace:
                bit_config = get_model_quantize_bit_config(model)
                save_yaml_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_bit_config.yaml', bit_config)
            model = convert_fx(model)
            model = fuse_fx(model)
            if args.hp.save_jit_trace:
                clip_model_weight_in_quant_min_max(model, bit_config)
        else:
            torch.quantization.convert(model, inplace=True)
        if args.hp.save_jit_trace:
            filename = 'ptq_quant.jit'
            save_torchscript_model(model, val_loader, prefix=f'{args.log_name}/{args.arch}',
                                   filename=filename)
        save_file_on_master(args, f'{args.log_name}/{args.arch}_quantized_by_ptq.txt', str(model))
        acc1, acc5 = validate(val_loader, model, criterion, args, device='cpu')
        if writer is not None:
            writer.add_scalar('val/acc1', acc1, 0)
            writer.add_scalar('val/acc5', acc5, 0)
            writer.close()
        return

    dataload_factory = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = dataload_factory.product_train_val_loader(dataload_factory.imagenet2012)
    writer = get_summary_writer(args, ngpus_per_node, model)
    if args.hp.evaluate:
        if writer is not None:
            get_model_info(copy.deepcopy(model_without_ddp), args, val_loader)
    args.batch_num = len(train_loader)

    lr_scheduler = get_lr_scheduler(optimizer, args)

    resume_model(model_without_ddp, args, optimizer, lr_scheduler)

    global best_acc1  # pylint: disable=global-statement
    best_acc1 = args.best_acc1
    if args.hp.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.hp.amp:
        # Automatic mixed precision training
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.hp.quantization.quantize:
        model.apply(torch.quantization.disable_observer)
        model.apply(torch.quantization.disable_fake_quant)
        acc1, acc5 = validate(val_loader, model, criterion, args)
        print_on_master(args, f'validate after calibration: acc1: {acc1}, acc5: {acc5}')
        model.apply(torch.quantization.enable_fake_quant)

    for epoch in range(args.start_epoch, args.hp.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        epoch_start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler,
              teacher_model=teacher_model, distill_criterion=distill_criterion)
        lr_scheduler.step()
        epoch_total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))

        if args.hp.multi_gpu.rank in [-1, 0]:
            print(f'Epoch[{epoch + 1}/{args.hp.epochs}] total time {total_time_str}')
        with torch.no_grad():
            if args.hp.quantization.quantize:
                if epoch >= args.hp.quantization.num_observer_update_epochs:
                    print('Disabling observer for subseq epochs, epoch = ', epoch)
                    model.apply(torch.quantization.disable_observer)
                if epoch >= args.hp.quantization.num_batch_norm_update_epochs:
                    print('Freezing BN for subseq epochs, epoch = ', epoch)
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # evaluate on validation set
        if args.hp.quantization.quantize:
            print('Evaluate QAT model')
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if writer is not None:
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)
            writer.add_scalar('val/lr', optimizer.param_groups[0]['lr'], epoch)
        # remember best acc@1 and save checkpoint

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print_on_master(args, f'** best at epoch: {epoch + 1}, acc1: {acc1}, acc5: {acc5}')
        if writer is not None:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_without_ddp.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, prefix=f'{args.log_name}/{args.arch}')
        with torch.no_grad():
            if args.hp.quantization.quantize:
                save_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_by_qat_prepared.txt',
                                    str(model_without_ddp))
                quantized_eval_model = copy.deepcopy(model_without_ddp)
                quantized_eval_model.eval()
                quantized_eval_model.to(torch.device('cpu'))
                if args.hp.quantization.quantize_fx:
                    if args.hp.save_jit_trace:
                        bit_config = get_model_quantize_bit_config(quantized_eval_model)
                        save_yaml_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_bit_config.yaml',
                                                 bit_config)
                    quantized_eval_model = convert_fx(quantized_eval_model)
                    if args.hp.save_jit_trace:
                        clip_model_weight_in_quant_min_max(quantized_eval_model, bit_config)
                else:
                    torch.quantization.convert(quantized_eval_model, inplace=True)
                save_file_on_master(args, f'{args.log_name}/{args.arch}_quantized_by_qat.txt',
                                    str(quantized_eval_model))
                print('Evaluate Quantized model')
                validate(val_loader, quantized_eval_model, criterion, args, device='cpu')
        if is_best and args.hp.save_jit_trace:
            save_file_on_master(args, f'{args.log_name}/{args.arch}_save_jit_log.txt',
                                f'best at epoch: {epoch+1}, acc1: {acc1}, acc5: {acc5}')
            if args.hp.quantization.quantize:
                save_jit_model = quantized_eval_model
                filename = 'best_qat_quant.jit'
            else:
                save_jit_model = model
                filename = 'best.jit'
            save_torchscript_model(save_jit_model, val_loader, prefix=f'{args.log_name}/{args.arch}',
                                   filename=filename)
        del quantized_eval_model
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
