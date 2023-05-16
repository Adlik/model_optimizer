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
from model_optimizer.core import (get_base_parser, get_hyperparam, get_freer_gpu, main_s1_set_seed,
                                  main_s2_start_worker, display_model, process_model, resume_model,
                                  distributed_model, get_optimizer, get_summary_writer,
                                  get_model_info, get_lr_scheduler, validate, train, save_checkpoint,
                                  save_file_on_master, save_yaml_file_on_master, print_on_master)
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.quantizer import get_qconfig_dict, get_model_quantize_bit_config, prepare_fx_with_backend, \
    get_qconfig, convert_model_by_backend, get_convert_fx_model
from model_optimizer.losses import KLDivergence
from model_optimizer.datasets import DataloaderFactory
from model_optimizer.models import get_model_from_source, get_teacher_model
from model_optimizer.utils.mp.sensitivity_analysis import layer_sensitivity_profiling, get_skip_layers, \
    get_sensitivity_qconfig_dict


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
    channel_config_path = args.hp.auto_slim.channel_config_path if args.hp.HasField('auto_slim') else None
    model = get_model_from_source(args.hp.arch, args.hp.model_source, args.hp.pretrained, args.hp.width_mult,
                                  args.hp.depth_mult, is_subnet=args.hp.is_subnet,
                                  channel_config_path=channel_config_path)

    if args.hp.distill.teacher_model.arch:
        teacher_model = get_teacher_model(args.hp.distill.teacher_model.arch, args.hp.distill.teacher_model.source)
    else:
        teacher_model = None

    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        print(args)
        print('model:\n=========\n')
        display_model(model)

    process_model(model, args)

    dataload_factory = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = dataload_factory.product_train_val_loader(dataload_factory.imagenet2012)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.hp.label_smoothing).cuda(args.gpu)

    # sensitivity analysis
    sensitivity_qconfig_dict = None
    if args.hp.quantization.HasField('sensitivity_analysis'):
        data_loader_calibration, _, _ = dataload_factory.product_train_val_loader(
            dataload_factory.imagenet2012,
            args.hp.quantization.num_calibration_batches,
            use_val_trans=True, batch_same_data=True)
        qconfig = get_qconfig(args.hp.quantization, args.hp.quantization.backend)
        sensitivity_analysis = args.hp.quantization.sensitivity_analysis
        target_metric = sensitivity_analysis.target_metric
        sensitivity_type = sensitivity_analysis.sensitivity_type
        metric_big_best = sensitivity_analysis.metric_big_best

        layer_sensitivity = layer_sensitivity_profiling(model, val_loader, data_loader_calibration, validate,
                                                        criterion, target_metric, sensitivity_type, qconfig,
                                                        metric_big_best, args.hp.quantization.backend, args)
        print(f'layer_sensitivity:{layer_sensitivity}')

        skip_layers = get_skip_layers(model, val_loader, data_loader_calibration, validate, criterion,
                                      target_metric, metric_big_best, qconfig, layer_sensitivity,
                                      args.hp.quantization.backend, args)
        print(f'skip_layers:{skip_layers}')

        sensitivity_qconfig_dict = get_sensitivity_qconfig_dict(skip_layers)

    if args.hp.quantization.quantize:
        qconfig_dict = get_qconfig_dict(model, args.hp.quantization, args.hp.quantization.backend,
                                        extra_qconfig_dict=sensitivity_qconfig_dict)
        model = prepare_fx_with_backend(model, qconfig_dict, args.hp.quantization.backend,
                                        not args.hp.quantization.post_training_quantize)

        model.eval()
        df_calibrate = DataloaderFactory(args)
        calibrate_train_loader, _, _ = df_calibrate.product_train_val_loader(
            df_calibrate.imagenet2012,
            args.hp.quantization.num_calibration_batches,
            use_val_trans=True,
            batch_same_data=True)
        model.apply(torch.quantization.enable_observer)
        model.apply(torch.quantization.disable_fake_quant)

        # Calibrate
        with torch.no_grad():
            for images, _ in calibrate_train_loader:
                model(images)
        del calibrate_train_loader
        del df_calibrate
        print('finish calibration')

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

    optimizer = get_optimizer(model, args)
    cudnn.benchmark = True

    if args.hp.quantization.quantize and args.hp.quantization.post_training_quantize:
        model.eval()
        model.apply(torch.quantization.disable_observer)
        model.apply(torch.quantization.enable_fake_quant)

        writer = get_summary_writer(args, ngpus_per_node, model)
        save_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_by_ptq_calibrate.txt', str(model))

        if args.hp.save_jit_trace:
            bit_config = get_model_quantize_bit_config(model)
            save_yaml_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_bit_config.yaml', bit_config)
        converted_model = get_convert_fx_model(model, evaluate=True, backend=args.hp.quantization.backend)
        convert_model_by_backend(model, args.log_name, args.hp.quantization.backend,
                                 onnx_model_name=f'{args.arch}_ptq_fakequant',
                                 converted_model_name=f'{args.arch}_ptq_quant')

        save_file_on_master(args, f'{args.log_name}/{args.arch}_quantized_by_ptq.txt', str(converted_model))
        acc1, acc5 = validate(val_loader, converted_model, criterion, args, device='cpu')
        if writer is not None:
            writer.add_scalar('val/acc1', acc1, 0)
            writer.add_scalar('val/acc5', acc5, 0)
            writer.close()
        return

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
        model.train()
        model.apply(torch.quantization.disable_observer)
        model.apply(torch.quantization.enable_fake_quant)

    for epoch in range(args.start_epoch, args.hp.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        epoch_start_time = time.time()
        if args.hp.distill.teacher_model.arch:
            train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler,
                  teacher_model=teacher_model, distill_criterion=distill_criterion)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler)
        lr_scheduler.step()
        epoch_total_time = time.time() - epoch_start_time
        total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))

        if args.hp.multi_gpu.rank in [-1, 0]:
            print(f'Epoch[{epoch + 1}/{args.hp.epochs}] total time {total_time_str}')
        with torch.no_grad():
            if args.hp.quantization.quantize:
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
            if args.hp.save_jit_trace:
                bit_config = get_model_quantize_bit_config(model_without_ddp)
                save_yaml_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_bit_config.yaml', bit_config)
            save_file_on_master(args, f'{args.log_name}/{args.arch}_quantize_by_qat_prepared.txt',
                                str(model_without_ddp))
            quantized_eval_model = copy.deepcopy(model_without_ddp)
            converted_model = get_convert_fx_model(quantized_eval_model, evaluate=True,
                                                   backend=args.hp.quantization.backend)
            convert_model_by_backend(quantized_eval_model, args.log_name, args.hp.quantization.backend)
            save_file_on_master(args, f'{args.log_name}/{args.arch}_quantized_by_qat.txt',
                                str(converted_model))
            print('Evaluate Quantized model')
            validate(val_loader, converted_model, criterion, args, device='cpu')

    if is_best and args.hp.save_jit_trace and writer is not None:
        if args.hp.quantization.quantize:
            save_jit_model = quantized_eval_model
            filename = 'best_qat_quant'
        else:
            save_jit_model = model
            filename = 'best'
        convert_model_by_backend(save_jit_model, args.log_name, args.hp.quantization.backend,
                                 onnx_model_name='best_fakequant', converted_model_name=filename)
    del quantized_eval_model
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
