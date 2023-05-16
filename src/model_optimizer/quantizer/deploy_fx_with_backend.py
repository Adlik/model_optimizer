# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"
"""
Deploy model by backends.
"""
import os
import copy
import torch
from torch.quantization.quantize_fx import convert_fx
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.utils.registry import register_deploy_function, DEPLOY_FUNCTION_BY_BACKEND
from model_optimizer.quantizer.fusion_convert import fuse_prepared_model
from model_optimizer.quantizer.deploy import remove_fake_quant_and_collect_scale
from .onnx_utils import ONNXQNNPass
from .quant_utils import get_specified_not_quant_module


# pylint: disable=unused-argument
@register_deploy_function(eppb.InferenceBackend.TORCH_FBGEMM)
@register_deploy_function(eppb.InferenceBackend.TORCH_QNNPACK)
@register_deploy_function(eppb.InferenceBackend.TVM)
def deploy_model_to_tvm(onnx_model_path, fused_model, convert_model_path, **extra_kwargs):
    """
    Convert prepared onnx model to quantized onnx model for inference on TVM engine.
    """
    qconfig_module_with_none = get_specified_not_quant_module(fused_model, onnx_model_path)
    onnx_pass = ONNXQNNPass(onnx_model_path, qconfig_module_with_none)
    onnx_pass.run(convert_model_path)


# pylint: disable=unused-argument
@register_deploy_function(eppb.InferenceBackend.TORCH_FBGEMM)
@register_deploy_function(eppb.InferenceBackend.TORCH_QNNPACK)
@register_deploy_function(eppb.InferenceBackend.TVM)
def deploy_model_to_jit(prepared_model, input_data, convert_model_path, **extra_kwargs):
    """
    Convert torch_model to jit script.
    """
    converted_model = get_convert_fx_model(prepared_model)
    jit_input = torch.rand(input_data)
    # pylint: disable=too-many-function-args
    convert_model_to_jit(converted_model, jit_input, convert_model_path)


@register_deploy_function(eppb.InferenceBackend.TENSORRT)
def deploy_model_to_tensorrt(onnx_model_path, convert_model_path, **kwargs):
    """
    Convert prepared onnx model to quantized onnx model for inference on TensorRT engine.
    """
    print("Extract qparams for TensorRT.")
    remove_fake_quant_and_collect_scale(onnx_model_path, convert_model_path, backend=eppb.InferenceBackend.TENSORRT)


def get_convert_fx_model(prepared_model, evaluate=False, backend=None):
    """
    Warp the convert_fx interface and use to evaluate the converted model.
    When backend is tensorrt, the model will not be converted.
    Args:
        prepared_model: pytorch prepared_fx model
        evaluate: whether to use alone
        backend: inference backend

    Returns:
        converted_model: after convert_fx
    """
    need_convert_model = prepared_model
    if evaluate:
        need_convert_model = copy.deepcopy(prepared_model)
        need_convert_model.eval()
        need_convert_model.to(torch.device('cpu'))

    if backend == eppb.InferenceBackend.TENSORRT:
        return fuse_prepared_model(need_convert_model)

    converted_model = convert_fx(need_convert_model)
    return converted_model


def convert_model_to_jit(model, jit_input, filename):
    """
    Save torchscript model
    Args:
        model:
        jit_input:
        filename:

    Returns:

    """
    filename = f'{filename}.jit'
    torch.jit.save(torch.jit.trace(model, jit_input), filename)


def export_onnx_model(prepared_model, base_path, onnx_model_name, input_data, opset_version=11, **extra_kwargs):
    """
    Save torchscript model
    Args:
        prepared_model:
        base_path
        onnx_model_name:
        input_data:
        opset_version:

    Returns:

    """
    input_shape_dict = {"input": input_data}
    dummy_input = {name: torch.rand(shape) for name, shape in input_shape_dict.items()}
    input_names = list(dummy_input.keys())
    dummy_input = tuple(dummy_input.values())
    output_names = ["output"]

    onnx_model_path = os.path.join(base_path, onnx_model_name)

    torch.onnx.export(prepared_model, dummy_input, onnx_model_path, input_names=input_names,
                      output_names=output_names, opset_version=opset_version, **extra_kwargs)


def check_input_data(input_data):
    """
    Verify the input_data.
    """
    if input_data is None:
        input_data = [16, 3, 224, 224]

    assert isinstance(input_data, list) and len(input_data) == 4, f'input_data should be a list and length=4, \
            but input type: {type(input_data)}, len: {len(input_data)}.'
    return input_data


def convert_model_by_backend(prepared_model, base_path, backend, onnx_model_name='fake_quant_model',
                             converted_model_name='deploy_model', input_data=None,
                             opset_version=11, **extra_kwargs):
    """
    Save torchscript model.
    Args:
        prepared_model:
        onnx_model_path:
        backend:
        converted_model_name:
        input_data:
        opset_version:

    Returns:

    """
    input_data = check_input_data(input_data)
    prepared_model.eval()
    prepared_model.to(torch.device('cpu'))
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    fused_model = fuse_prepared_model(prepared_model)

    onnx_model_name = onnx_model_name + '.onnx'
    export_onnx_model(fused_model, base_path, onnx_model_name, input_data, opset_version, **extra_kwargs)

    onnx_model_path = os.path.join(base_path, onnx_model_name)
    convert_model_path = os.path.join(base_path, converted_model_name)

    kwargs = {
        'onnx_model_path': onnx_model_path,
        'fused_model': fused_model,
        'prepared_model': prepared_model,
        'base_path': base_path,
        'convert_model_path': convert_model_path,
        'input_data': input_data,
        **extra_kwargs
    }

    assert backend in DEPLOY_FUNCTION_BY_BACKEND, \
           f'You should firstly need to register deploy func for backend: {backend}.'

    for convert_func in DEPLOY_FUNCTION_BY_BACKEND[backend]:
        convert_func(**kwargs)
