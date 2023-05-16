# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
run inference on tvm
"""
import os
import time
import argparse
import torch
import torchvision
from torchvision import transforms
import numpy as np
import tvm  # pylint: disable=import-error
from tvm import relay  # pylint: disable=import-error


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, count=1):
        """

        Args:
            val:
            count:

        Returns:

        """
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter:
    """
    progress meter
    """
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        """
        print entries
        Args:
            batch:

        Returns:
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


def accuracy(output, target, topk=(1,)):  # pylint: disable=missing-function-docstring
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_val_loader(data_dir):  # pylint: disable=missing-function-docstring
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    test_sampler = None

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True, sampler=test_sampler, persistent_workers=True)
    return val_loader


def validate_onnx_tvm(val_loader, onnx_model_path):  # pylint: disable=missing-function-docstring
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5,
                             prefix='Test: ')

    import onnx  # pylint: disable=import-error
    onnx_model = onnx.load(onnx_model_path)
    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    input_shapes = {input_name: [16, 3, 224, 224]}
    model, params = relay.frontend.from_onnx(onnx_model, input_shapes)
    tvm_target = "llvm -mcpu=core-avx2"  # like "llvm -mcpu=skylake-avx512" or "llvm -mcpu=core-avx2"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, tvm_target, params=params)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(tvm_target, 0)))
    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        runtime.set_input(input_name, _input.numpy())
        runtime.run()
        output = torch.from_numpy(runtime.get_output(0).numpy())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        top1.update(acc1[0], _input.size(0))
        top5.update(acc5[0], _input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.print(i)
    print(f' *Time {batch_time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg


def validate_onnx(val_loader, model):  # pylint: disable=missing-function-docstring
    import onnxruntime  # pylint: disable=import-error
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("input name", input_name)
    input_shape = session.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = session.get_inputs()[0].type
    print("input type", input_type)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5,
                             prefix='Test: ')

    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        if _input.size(0) < input_shape[0]:
            break
        with torch.no_grad():
            result = session.run([output_name], {input_name: _input.numpy()})
            output = torch.tensor(np.array(result).squeeze())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        top1.update(acc1[0], _input.size(0))
        top5.update(acc5[0], _input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.print(i)
    print(f' *Time {batch_time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg


def validate_jit(val_loader, jit_model_path):  # pylint: disable=missing-function-docstring
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5,
                             prefix='Test: ')

    jit_model = torch.jit.load(jit_model_path, map_location='cpu')
    jit_model.eval()
    print(f'jit_model: {jit_model}')
    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        output = jit_model(_input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        top1.update(acc1[0], _input.size(0))
        top5.update(acc5[0], _input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.print(i)
    print(f' *Time {batch_time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg


def validate_jit_tvm(val_loader, jit_model_path):  # pylint: disable=missing-function-docstring
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, top1, top5,
                             prefix='Test: ')

    jit_model = torch.jit.load(jit_model_path, map_location='cpu')
    pt_inp = torch.rand(1, 3, 224, 224)
    print(f'jit_model: {jit_model}')
    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    input_shapes = [(input_name, pt_inp.shape)]
    model, params = relay.frontend.from_pytorch(jit_model, input_shapes)
    tvm_target = "llvm -mcpu=skylake-avx512"  # like "llvm -mcpu=skylake-avx512" or "llvm -mcpu=core-avx2"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, tvm_target, params=params)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(tvm_target, 0)))
    end = time.time()
    for i, (_input, target) in enumerate(val_loader):
        runtime.set_input(input_name, _input.numpy())
        runtime.run()
        output = torch.from_numpy(runtime.get_output(0).numpy())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        top1.update(acc1[0], _input.size(0))
        top5.update(acc5[0], _input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.print(i)
    print(f' *Time {batch_time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='resnet50_quantized.jit', help='model path')
    parser.add_argument('--data-dir', type=str, default='/data/imagenet/imagenet-torch', help='dataset path')
    parser.add_argument('--no-tvm', action='store_true', help='not run with tvm engine')
    args = parser.parse_args()
    _model = args.weights
    _data_dir = args.data_dir
    _val_loader = get_val_loader(_data_dir)
    postfix = os.path.splitext(_model)[-1][1:]
    if postfix == "jit":
        if args.no_tvm:
            validate_jit(_val_loader, _model)
        else:
            validate_jit_tvm(_val_loader, _model)
    elif postfix == "onnx":
        if args.no_tvm:
            validate_onnx_tvm(_val_loader, _model)
        else:
            validate_onnx(_val_loader, _model)
