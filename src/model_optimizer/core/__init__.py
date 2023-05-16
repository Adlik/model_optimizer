# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
model optimizer core function/class
"""
import argparse
import datetime
import hashlib
import os
import random
import shutil
import time
import warnings
from pathlib import Path
import google.protobuf as pb
import google.protobuf.text_format  # noqa # pylint: disable=unused-import
import numpy as np
import yaml
import torch
from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from ..proto import model_optimizer_torch_pb2 as eppb
from .utils import AverageMeter, ProgressMeter, accuracy
from ..utils.ptflops import get_model_complexity_info


def get_base_parser():
    """
        Default values should keep stable.
    """

    print('Please do not import ipdb when using distributed training')

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--hp', type=str,
                        help='File path to save hyperparameter configuration')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    return parser


def main_s1_set_seed(hyper_param):
    """
    main set seed
    Args:
        hyper_param:

    Returns:

    """
    if hyper_param.HasField('seed'):
        random.seed(hyper_param.seed)
        torch.manual_seed(hyper_param.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def main_s2_start_worker(main_worker, args, hyper_param):
    """
    main_s2_start_worker
    Args:
        main_worker:
        args:
        hyper_param:

    Returns:

    """
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.world_size = hyper_param.multi_gpu.world_size
    args.lr = hyper_param.lr
    if hyper_param.HasField('multi_gpu') and hyper_param.multi_gpu.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or (hyper_param.HasField(
        'multi_gpu') and hyper_param.multi_gpu.multiprocessing_distributed)

    ngpus_per_node = torch.cuda.device_count()
    if hyper_param.HasField('multi_gpu') and hyper_param.multi_gpu.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_hyperparam(args):
    """
    get hyperparam
    Args:
        args:

    Returns:

    """
    assert os.path.exists(args.hp)
    hyper_param = eppb.HyperParam()
    with open(args.hp, 'r', encoding='UTF-8') as file:
        pb.text_format.Merge(file.read(), hyper_param)
    return hyper_param


def get_freer_gpu():
    """
    get free gpu
    Returns:

    """
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    with open('tmp', 'r', encoding='UTF-8') as file:
        memory_available = [int(x.split()[2]) for x in file.readlines()]
    os.system('rm tmp')
    # TODO; if no gpu, return None
    try:
        visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        memory_visible = []
        for i in visible_gpu.split(','):
            memory_visible.append(memory_available[int(i)])
        return np.argmax(memory_visible)
    except KeyError:
        return np.argmax(memory_available)


def reduce_mean(tensor, world_size):
    """
    reduce mean
    Args:
        tensor:
        world_size:

    Returns:

    """
    reduce_tensor = tensor.clone()
    dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
    reduce_tensor /= world_size
    return reduce_tensor


def get_lr_scheduler(optimizer, lr_domain):  # pylint: disable=too-many-branches
    """
    Args:
        optimizer:
        lr_domain ([proto]): [lr configuration domain] e.g. args.hp args.hp.bit_pruner
    """

    if isinstance(lr_domain, argparse.Namespace) and hasattr(lr_domain, 'hp'):
        lr_domain = lr_domain.hp
    if not lr_domain.HasField('warmup'):
        warmup_epochs = 0
    else:
        warmup_epochs = lr_domain.warmup.lr_warmup_epochs
    if lr_domain.lr_scheduler == eppb.LRScheduleType.CosineAnnealingLR:
        print('Use cosine scheduler')
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr_domain.epochs-warmup_epochs)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.ExponentialLR:
        scheduler_next = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_domain.exponential_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.StepLR:
        print(f'Use step scheduler, step size: {lr_domain.step_lr.step_size}, gamma: {lr_domain.step_lr.gamma}')
        scheduler_next = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_domain.step_lr.step_size, gamma=lr_domain.step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.MultiStepLR:
        print(f'Use MultiStepLR scheduler, milestones: {lr_domain.multi_step_lr.milestones},'
              f' gamma: {lr_domain.multi_step_lr.gamma}')
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_domain.multi_step_lr.milestones, gamma=lr_domain.multi_step_lr.gamma)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.CyclicLR:
        print('Use CyclicLR scheduler')
        if not lr_domain.cyclic_lr.HasField('step_size_down'):
            step_size_down = None
        else:
            step_size_down = lr_domain.cyclic_lr.step_size_down

        cyclic_mode_map = {eppb.CyclicLRParam.Mode.triangular: 'triangular',
                           eppb.CyclicLRParam.Mode.triangular2: 'triangular2',
                           eppb.CyclicLRParam.Mode.exp_range: 'exp_range', }

        scheduler_next = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr_domain.cyclic_lr.base_lr, max_lr=lr_domain.cyclic_lr.max_lr,
            step_size_up=lr_domain.cyclic_lr.step_size_up, step_size_down=step_size_down,
            mode=cyclic_mode_map[lr_domain.cyclic_lr.mode], gamma=lr_domain.cyclic_lr.gamma)
    else:
        raise NotImplementedError
    if not lr_domain.HasField('warmup'):
        lr_scheduler = scheduler_next
    else:
        print('Use warmup scheduler')
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=lr_domain.warmup.lr_warmup_decay,
            total_iters=lr_domain.warmup.lr_warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, scheduler_next],
            milestones=[lr_domain.warmup.lr_warmup_epochs]
        )
    return lr_scheduler


def get_optimizer(model, args):
    """
    get optimizer
    Args:
        model:
        args:

    Returns:

    """
    # define optimizer after process model
    print('define optimizer')
    if args.hp.optimizer == eppb.OptimizerType.SGD:
        params = add_weight_decay(model, weight_decay=args.hp.sgd.weight_decay,
                                  skip_keys=['expand_', 'running_scale'])
        optimizer = torch.optim.SGD(params, args.hp.lr,
                                    momentum=args.hp.sgd.momentum)
        print('Use SGD')
    elif args.hp.optimizer == eppb.OptimizerType.Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.hp.lr, weight_decay=args.hp.adam.weight_decay)
        print('Use Adam')
    elif args.hp.optimizer == eppb.OptimizerType.RMSprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.hp.lr, momentum=args.hp.rmsprop.momentum,
                                        weight_decay=args.hp.rmsprop.weight_decay, eps=0.0316, alpha=0.9)
        print('Use RMSprop')
    else:
        raise NotImplementedError
    return optimizer


def add_weight_decay(model, weight_decay, skip_keys):
    """
    add weight decay
    Args:
        model:
        weight_decay:
        skip_keys:

    Returns:

    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def set_bn_eval(m):
    """
    [summary]
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
    https://github.com/pytorch/pytorch/issues/16149
    requires_grad does not change the train/eval mode,
    but will avoid calculating the gradients for the affine parameters (weight and bias).
    bn.train() and bn.eval() will change the usage of the running stats (running_mean and running_var).
    For detailed computation of Batch Normalization, please refer to the source code here.
    https://github.com/pytorch/pytorch/blob/83c054de481d4f65a8a73a903edd6beaac18e8bc/torch/csrc/jit/passes/graph_fuser.cpp#L232
    The input is normalized by the calculated mean and variance first.
    Then the transformation of w*x+b is applied on it by adding the operations to the computational graph.
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_grad_false(module):
    """
    freeze gamma and beta in BatchNorm
    model.apply(set_bn_grad_false)
    optimizer = SGD(model.parameters())
    """
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if module.affine:
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)


def set_param_grad_false(model):
    """

    Args:
        model:

    Returns:

    """
    for _, param in model.named_parameters():  # same to set bn val? No
        if param.requires_grad:
            param.requires_grad_(False)
            print(f'frozen weights. shape:{param.shape}')


# pylint: disable=too-many-branches,too-many-statements
def validate(val_loader, model, criterion, args, algorithm=None, device=None):
    """

    :param val_loader:
    :param model: if algorithm is not None, model can be None
    :param criterion: if don't neet criterion, set None
    :param args:
    :param algorithm:
    :param device:
    :return:
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')
    if algorithm is not None:
        # switch to evaluate mode
        algorithm.eval()
        model = algorithm.model
    else:
        model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (_input, target) in enumerate(val_loader):
            if args.gpu is not None and device is None:
                _input = _input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            if device == 'cpu':
                _input = _input.cpu()
                target = target.cpu()
            # compute output
            output = model(_input)
            if criterion is not None:
                loss = criterion(output, target)
            else:
                loss = torch.zeros(1)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0 and args.hp.multi_gpu.rank in [-1, 0]:
                progress.print(i)
            if args.hp.overfit_test:
                break
            if args.distributed:
                torch.distributed.barrier()  # Without adding this line, different GPU validate results may be different

        if args.hp.multi_gpu.rank in [-1, 0]:
            print(f' *Time {batch_time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
        if (not args.hp.validate_data_full) and args.distributed:
            progress.synchronize_between_processes()
            top1_avg = top1.sum / top1.count
            top5_avg = top5.sum / top5.count
            if args.hp.multi_gpu.rank in [-1, 0]:
                print(f' *Average Acc@1 {top1_avg:.5f} Acc@5 {top5_avg:.5f}')
            return top1_avg, top5_avg

    return top1.avg, top5.avg


def train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler=None, no_backward=False,
          teacher_model=None, distill_criterion=None):
    """
    train
    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:
        args:
        writer:
        scaler:
        no_backward:
        teacher_model:
        distill_criterion:

    Returns:

    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    if teacher_model is not None:
        distill_losses = AverageMeter('D_Loss', ':.4e')
        student_losses = AverageMeter('Stu_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if teacher_model is not None:
        progress = ProgressMeter(args.batch_num, batch_time, data_time, distill_losses, student_losses, top1,
                                 top5, prefix=f"Epoch: [{epoch + 1}]")
        teacher_model.eval()

        loss_weight = distill_criterion.loss_weight
    else:
        progress = ProgressMeter(args.batch_num, batch_time, data_time, losses, top1,
                                 top5, prefix=f"Epoch: [{epoch + 1}]")
    if args.hp.multi_gpu.rank in [-1, 0]:
        print(f'gpu id: {args.gpu}')
    # switch to train mode
    model.train()

    end = time.time()
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.hp.amp and scaler is not None):
            output = model(inputs)
            if no_backward:
                return
            if teacher_model is not None:
                teacher_output = teacher_model(inputs)
                student_loss = (1 - loss_weight) * criterion(output, targets)
                distill_loss = distill_criterion(output, teacher_output)
                loss = student_loss + distill_loss
            else:
                loss = criterion(output, targets)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        if args.distributed:
            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.world_size)
            reduced_acc1 = reduce_mean(acc1, args.world_size)
            reduced_acc5 = reduce_mean(acc5, args.world_size)

            losses.update(reduced_loss.item(), inputs.size(0))
            top1.update(reduced_acc1[0], inputs.size(0))
            top5.update(reduced_acc5[0], inputs.size(0))

            if teacher_model is not None:
                reduced_distill_loss = reduce_mean(distill_loss, args.world_size)
                reduced_student_loss = reduce_mean(student_loss, args.world_size)
                distill_losses.update(reduced_distill_loss.item(), inputs.size(0))
                student_losses.update(reduced_student_loss.item(), inputs.size(0))
        else:
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            if teacher_model is not None:
                distill_losses.update(distill_loss.item(), inputs.size(0))
                student_losses.update(student_loss.item(), inputs.size(0))

        if writer is not None:
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
            writer.add_scalar('train/acc1', top1.avg, base_step + i)
            writer.add_scalar('train/acc5', top5.avg, base_step + i)
            if teacher_model is not None:
                writer.add_scalar('train/student_loss', student_losses.avg, base_step + i)
                writer.add_scalar('train/distill_loss', distill_losses.avg, base_step + i)

        if args.hp.amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        # warning 1. backward 2. step 3. zero_grad
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0 and args.hp.multi_gpu.rank in [-1, 0]:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
    return


# pylint: enable=too-many-branches,too-many-statements
def get_summary_writer(args, ngpus_per_node, model):
    """
    get summary writer
    Args:
        args:
        ngpus_per_node:
        model:

    Returns:

    """
    args.log_name = f'logger/{args.hp.arch}_{args.hp.log_name}_{get_current_time()}'
    if not args.hp.multi_gpu.multiprocessing_distributed \
            or (args.hp.multi_gpu.multiprocessing_distributed and args.hp.multi_gpu.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_name)
        with open(f'{args.log_name}/{args.arch}.prototxt', 'w', encoding='utf-8') as write_file:
            write_file.write(str(args.hp))
        with open(f'{args.log_name}/{args.arch}.txt', 'w', encoding='utf-8') as write_file:
            write_file.write(str(model))
        return writer
    return None


def get_model_info(model, args, input_size=(3, 224, 224)):
    """
    get model info
    Args:
        model:
        args:
        input_size:

    Returns:

    """
    print('Inference for complexity summary')
    if isinstance(input_size, torch.utils.data.DataLoader):
        input_size = input_size.dataset[0][0].shape
        input_size = (input_size[0], input_size[1], input_size[2])

    with open(f'{args.log_name}/{args.arch}_flops.txt', 'w', encoding='utf-8') as out_file:
        flops, params = get_model_complexity_info(
            model, input_size, as_strings=True, print_per_layer_stat=True, ost=out_file)
    print(f'Computational complexity: {flops:<8}')
    print(f'Number of parameters:  {params:<8}')

    if args.hp.export_onnx:
        import onnx  # pylint: disable=import-error
        import onnxruntime  # pylint: disable=import-error
        if args.hp.is_subnet:
            save_name = Path(args.hp.auto_slim.channel_config_path).with_suffix('.onnx').name
            save_path = str(Path(args.log_name) / save_name)
        else:
            save_path = f"{args.log_name}/{args.arch}.onnx"
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], requires_grad=True).cuda(args.gpu)
        torch_out = model(dummy_input)
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          # where to save the model (can be a file or file-like object)
                          save_path,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=13,  # the ONNX version to export the model to
                          input_names=['input'],  # the model's input names
                          output_names=['output']  # the model's output names
                          )

        # Checks
        model_onnx = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        dynamic = False

        if args.hp.onnx_simplify:
            import onnxsim  # pylint: disable=import-error

            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': list(dummy_input.shape)} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, save_path)

        print(f'export success, saved as {save_path}')

        # compute ONNX Runtime output prediction
        print('Compute ONNX Runtime output prediction')
        ort_session = onnxruntime.InferenceSession(save_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-04)
        mse = np.sqrt(np.mean(torch_out.detach().cpu().numpy() - ort_outs[0]) ** 2)
        print(f'MSE Error = {mse}')
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return flops, params


def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    """
    save checkpoint
    Args:
        state:
        is_best:
        prefix:
        filename:

    Returns:

    """
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'best.pth.tar')


def save_torchscript_model(model, val_loader, prefix, filename='best.jit'):
    """
    save torchscript model
    Args:
        model:
        val_loader:
        prefix:
        filename:

    Returns:

    """
    for _input, _ in val_loader:
        torch.jit.save(torch.jit.trace(model, _input), prefix + filename)
        return


def process_model(model, args):
    """

    Args:
        model: not distributed model
        args:

    Returns:

    """
    if not hasattr(args, 'arch'):
        args.arch = args.hp.arch

    if args.hp.HasField('weight'):
        if os.path.isfile(args.hp.weight):
            load_weight(model, args.hp.weight)
        else:
            print(f"=> no weight found at '{args.hp.weight}'")

    if args.hp.auto_slim.search_config.HasField('weight_path'):
        if os.path.isfile(args.hp.auto_slim.search_config.weight_path):
            load_weight(model, args.hp.auto_slim.search_config.weight_path)
        else:
            print(f"=> no weight found at '{args.hp.auto_slim.search_config.weight_path}'")


def resume_model(model, args, optimizer, lr_scheduler):
    """
    resume model from checkpoint
    Args:
        model:
        args:
        optimizer:
        lr_scheduler:

    Returns:

    """
    if args.hp.HasField('resume'):
        if os.path.isfile(args.hp.resume):
            print(f"=> resume from checkpoint '{args.hp.resume}'")
            checkpoint = torch.load(args.hp.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            args.best_acc1 = checkpoint['best_acc1']
            optimizer.load_state_dict(checkpoint["optimizer"])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            else:
                for _ in range(0, args.start_epoch):
                    lr_scheduler.step()
        else:
            print(f"=> no checkpoint found at '{args.hp.resume}'")
            args.optimizer_state = None
            args.best_acc1 = 0
    else:
        args.optimizer_state = None
        args.best_acc1 = 0


def load_weight_from_ddp_ckpt(model, weight_path):
    """

    Args:
        model:
        weight_path:

    Returns:

    """
    print(f"=> loading weight '{weight_path}'")
    checkpoint = torch.load(weight_path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)


def load_weight(model, weight_path):
    """
    load model weight
    Args:
        model:
        weight_path:

    Returns:

    """
    print(f"=> loading weight '{weight_path}'")
    checkpoint = torch.load(weight_path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']

    model.load_state_dict(pretrained_dict, strict=False)


def fast_collate(batch, memory_format):
    """

    Args:
        batch:
        memory_format:

    Returns:

    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    width = imgs[0].size[0]
    height = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, width, height), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def distributed_model(model, ngpus_per_node, args):
    """
    distribute model
    Args:
        model:
        ngpus_per_node:
        args:

    Returns:

    """
    if not torch.cuda.is_available() or args.gpu is None:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.hp.batch_size = int(args.hp.batch_size)
            args.hp.val_batch_size = int(args.hp.val_batch_size)
            args.hp.lr = args.lr * ngpus_per_node * args.hp.multi_gpu.world_size
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            raise RuntimeError('In distributed mode, but gpu is None.')
    else:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    return model


def distributed_model_not_ddp(model, args):
    """
    distribute model not ddp
    Args:
        model:
        ngpus_per_node:
        args:

    Returns:

    """
    if not torch.cuda.is_available() or args.gpu is None:
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    return model


def get_hash_code(message):
    """
    get hash code
    Args:
        message:

    Returns:

    """
    hash_code = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash_code[:8]


def get_current_time():
    """
    get current time
    Returns:

    """
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%Y%m%d%H%M")


def display_model(model):
    """
    display model info
    Args:
        model:

    Returns:

    """
    str_list = str(model).split('\n')
    if len(str_list) < 30:
        print(model)
        return
    begin = 10
    end = 5
    middle = len(str_list) - begin - end
    abbr_middle = ['...', f'{middle} lines', '...']
    abbr_str = '\n'.join(str_list[:10] + abbr_middle + str_list[-5:])
    print(abbr_str)


def save_file_on_master(args, filename, file_content):
    """

    Args:
        args:
        filename:
        file_content:

    Returns:

    """
    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(file_content)


def save_file_append_on_master(args, filename, file_content):
    """

    Args:
        args:
        filename:
        file_content:

    Returns:

    """
    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(file_content)


def save_yaml_file_on_master(args, filename, file_content):
    """

    Args:
        args:
        filename:
        file_content:

    Returns:

    """
    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        with open(filename, 'w', encoding='utf-8') as file:
            yaml.dump(file_content, file)


def print_on_master(args, content):
    """

    Args:
        args:
        content:

    Returns:

    """
    if args.gpu == 0 and args.hp.multi_gpu.rank in [-1, 0]:
        print(content)
