"""
Takes an ONNX file and creates a TensorRT engine.

python onnx2trt.py --onnx-path resnet18_ptq_quant_deploy_model.onnx --trt-path resnet18_ptq_quant_deploy_model.engine
--mode int8 --clip-range-file resnet18_ptq_quant_clip_ranges.json --evaluate --data-path /data/imagenet-torch
"""
import os
import json
import time
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import tensorrt as trt  # pylint: disable=import-error
from calibrator import ImageEntropyCalibrator2, ImageMixMaxCalibrator, ImageLegacyCalibrator
from tensorrt_infer import allocate_buffers, do_inference
from model_optimizer.core import AverageMeter, accuracy


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def onnx2trt(onnx_path, trt_path, max_batch_size=256, log_level=trt.Logger.ERROR, mode='fp32', is_explicit=False,
             batch_size=1, cali_batch=10, cali_dataset_path=None, calibration_method=None, dynamic_range_file=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with NVIDIA GPU.

    Args:
        onnx_path:  path of onnx file
        trt_path: path to save the engine
        max_batch_size: max_batch_size
        log_level: trt.Logger.ERROR or trt.Logger.VERBOSE
        mode: fp32, fp16, int8
        is_explicit: network containing Q/DQ layers
        cali_batch: the number of calibration datasets is equal to batch_size * cali_batch
        cali_dataset_path: the directory format is similar to the ImageNet datasets
        calibration_method: 'entropy', 'minmax', 'percentile'
        dynamic_range_file: quantification parameters for each layer

    Returns:
        tensorrt engine
    """
    if os.path.exists(trt_path):
        print(f'The "{trt_path}" exists. Remove it and continue.')
        os.remove(trt_path)

    trt_logger = trt.Logger(log_level)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(trt_logger) as builder, builder.create_network(explicit_batch) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network, trt_logger) as onnx_parser:

        print(f'Loading ONNX file from path {onnx_path}...')
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not onnx_parser.parse(model.read()):
                for e in range(onnx_parser.num_errors):
                    print(onnx_parser.get_error(e))
                raise TypeError('onnx_parser parse failed.')
        print('Completed parsing of ONNX file')

        if mode == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        if mode == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if dynamic_range_file:
                with open(dynamic_range_file, 'r', encoding="utf-8") as f:
                    dynamic_range = json.load(f)['tensorrt']['blob_range']

                for input_index in range(network.num_inputs):
                    input_tensor = network.get_input(input_index)
                    if input_tensor.name in dynamic_range:
                        amax = dynamic_range[input_tensor.name]
                        input_tensor.dynamic_range = (-amax, amax)
                        print(f'Set dynamic range of {input_tensor.name} as [{-amax}, {amax}]')

                for layer_index in range(network.num_layers):
                    layer = network[layer_index]
                    output_tensor = layer.get_output(0)
                    if output_tensor.name in dynamic_range:
                        amax = dynamic_range[output_tensor.name]
                        output_tensor.dynamic_range = (-amax, amax)
                        print(f'Set dynamic range of {output_tensor.name} as [{-amax}, {amax}]')
            elif is_explicit:
                # explicit mode do not need calibrator
                pass
            else:
                dataset = datasets.ImageFolder(cali_dataset_path, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]))
                cali_num = min(len(dataset), batch_size * cali_batch)
                cali_dataset = torch.utils.data.Subset(dataset, indices=torch.arange(cali_num))
                cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=batch_size, shuffle=False,
                                                          num_workers=1, pin_memory=False)

                cali_cache_path = os.path.basename(onnx_path).split('.')[0] + '_cali.cache'

                if calibration_method == 'entropy':
                    calibrator = ImageEntropyCalibrator2(cali_loader, cache_file=cali_cache_path)
                elif calibration_method == 'minmax':
                    calibrator = ImageMixMaxCalibrator(cali_loader, cache_file=cali_cache_path)
                elif calibration_method == 'percentile':
                    calibrator = ImageLegacyCalibrator(cali_loader, cache_file=cali_cache_path)
                else:
                    raise RuntimeError('Please select the correct calibration method!')
                config.int8_calibrator = calibrator
                print('Calibration Set!')

        builder.max_batch_size = max_batch_size
        config.max_workspace_size = 1 << 30  # 1GB

        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)

        engine = builder.build_engine(network, config)

        with open(trt_path, mode='wb') as f:
            f.write(bytearray(engine.serialize()))
    return engine


def validate(trt_path, batch_size=64, dataset_path=None):
    """Verifying model inference accuracy on the validation datasets.

    Args:
        trt_file: tensorrt engine file
        batch_size: keep the same batch size as the onnx model input
        dataset_path: the path to the validation datasets

    Returns:
        None
    """
    trt_logger = trt.Logger(trt.Logger.INFO)
    with open(trt_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    val_dataset_path = os.path.join(dataset_path, 'val')
    val_dataset = datasets.ImageFolder(val_dataset_path, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=4, pin_memory=False)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total_time = 0

    for index, (images, target) in enumerate(val_loader):
        images = images.detach().numpy()
        inputs[0].host = np.ascontiguousarray(images)

        start_time = time.time()
        trt_output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end_time = time.time()

        output = torch.from_numpy(np.array(trt_output))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        top1.update(acc1[0], images.shape[0])
        top5.update(acc5[0], images.shape[0])
        total_time += (end_time - start_time)
        batch_time = total_time / ((index + 1)*batch_size)

        if index % 100 == 0:
            print(f' {index} ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  time {batch_time * 1000:.3f}ms')
    print(f'Final ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  time {batch_time * 1000:.3f}ms ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Onnx to tensorrt')
    parser.add_argument('--onnx-path', type=str, default=None,
                        help="The ONNX model file to convert to TensorRT")
    parser.add_argument('--trt-path', type=str, default=None,
                        help="The path at which to write the engine")
    parser.add_argument('--mode', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'],
                        help="Supports three formats, fp32, fp16, int8")
    parser.add_argument('--clip-range-file', type=str, default=None,
                        help="File of saved calibration range")
    parser.add_argument('--cali-type', type=str, default='percentile', choices=['entropy', 'minmax', 'percentile'],
                        help="Calibration method")
    parser.add_argument('--max-bs', type=int, default=256,
                        help="max batch size")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="batch size")
    parser.add_argument('--cali-batch', type=int, default=10,
                        help="Control the number of calibration datasets, cali_batch * batch_size")
    parser.add_argument('--evaluate', action='store_true',
                        help="Evaluate the accuracy of the TensorRT model on the validation datasets")
    parser.add_argument('--verbose', action='store_true',
                        help="indicates the severity of a message")
    parser.add_argument('--explicit', action='store_true',
                        help="Explicit Quantization that the Q/DQ layer exists in the network")
    parser.add_argument('--data-path', type=str, required=True,
                        help="The path to the validation dataset")
    parser.add_argument('--cali-data-path', type=str, default=None,
                        help="The path of calibration dataset")
    args = parser.parse_args()

    if args.onnx_path:
        onnx2trt(args.onnx_path,
                 trt_path=args.trt_path,
                 max_batch_size=args.max_bs,
                 log_level=trt.Logger.VERBOSE if args.verbose else trt.Logger.ERROR,
                 mode=args.mode,
                 is_explicit=args.explicit,
                 batch_size=args.batch_size,
                 cali_batch=args.cali_batch,
                 cali_dataset_path=args.cali_data_path,
                 calibration_method=args.cali_type,
                 dynamic_range_file=args.clip_range_file)

    if args.evaluate:
        validate(args.trt_path, batch_size=args.batch_size, dataset_path=args.data_path)
