# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
layer sensitivity analysis of quantization
"""
import copy
from collections import OrderedDict
import torch
from torch.quantization import QConfig
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from model_optimizer.quantizer import prepare_fx_with_backend
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.core import distributed_model_not_ddp


def layer_sensitivity_profiling(model: Module, val_dataloader: DataLoader, cali_dataloader: DataLoader, val_func,
                                criterion: _Loss, target_metric: float, sensitivity_type: int, qconfig: QConfig,
                                metric_big_best: bool, backend: str, args):
    """Profiling the layer sensitivity of quantization

    Args:
        model: the model to be profiled
        val_dataloader: the validation dataloader
        cali_dataloader: the calibration dataloader
        val_func: the validation function
        criterion: the loss criterion
        target_metric: the expected metric of the quantized model
        sensitivity_type: specify the sensitivity algorithm, as follows
                          [eppb.SensitivityType.ONE_AT_A_TIME_ACC, eppb.SensitivityType.ONE_AT_A_TIME_LOSS]
        qconfig: the QConfig of quantization
        metric_big_best: sepcify if the metric(e.g. accuracy, loss, mAP) is the bigger, the better
        backend:
        args:

    Returns: The layer sensitivity dict[layer_name, sensitivity], sorted by reverse order.
             The greater the sensitivity, the more sensitive

    """
    t_model = copy.deepcopy(model)
    t_model.eval()
    distributed_model_not_ddp(t_model, args)
    device = next(t_model.parameters()).device
    print(f'layer_sensitivity_profiling use device:{device}')
    if sensitivity_type == eppb.SensitivityType.ONE_AT_A_TIME_ACC:  # type: ignore[attr-defined]
        layer_sensitivity = _profiling_by_acc(t_model, val_dataloader, cali_dataloader, val_func, criterion,
                                              qconfig, device, target_metric, metric_big_best, backend, args)
    elif sensitivity_type == eppb.SensitivityType.ONE_AT_A_TIME_LOSS:  # type: ignore[attr-defined]
        layer_sensitivity = _profiling_by_loss(t_model, val_dataloader, cali_dataloader, val_func,
                                               criterion, qconfig, device, backend, args)
    else:
        raise NotImplementedError(f'SensitivityMethod: {sensitivity_type} not implemented')

    layer_sensitivity = OrderedDict(sorted(layer_sensitivity.items(), key=lambda x: x[1], reverse=True))

    return layer_sensitivity


def get_skip_layers(model: Module, val_dataloader: DataLoader, cali_dataloader: DataLoader, val_func,
                    criterion: _Loss, target_metric: float, metric_big_best: bool, qconfig: QConfig,
                    layer_sensitivity: dict, backend: str, args):
    """Get the layers which are sensitive to quantization

    Args:
        model:
        val_dataloader:
        cali_dataloader:
        val_func:
        criterion:
        target_metric:
        metric_big_best:
        qconfig:
        layer_sensitivity: the layer sensitivity dict[layer_name, sensitivity], sorted by reverse order
        backend:
        args:

    Returns: the list of layers not to be quantized

    """
    # quantized all layers first, then skip the most sensitive layer one by one until accuracy meet target_result
    t_model = copy.deepcopy(model)
    t_model.eval()
    distributed_model_not_ddp(t_model, args)
    device = next(t_model.parameters()).device
    print(f'get_skip_layers use device: {device}')
    qconfig_dict = {"": qconfig}
    result = _cali_eval_model(t_model, qconfig_dict, device, cali_dataloader,
                              val_dataloader, val_func, criterion, backend, args)

    metric = (target_metric - result[0]) if metric_big_best else (result[0] - target_metric)
    module_name_qconfig = []
    skip_layers = []

    for layer_name, _ in layer_sensitivity.items():
        print(f'layer {layer_name} sensitivity metric: {metric}')
        if metric >= 0:
            module_name_qconfig.append((layer_name, None))
            qconfig_dict.update({'module_name': module_name_qconfig})  # type: ignore[dict-item]
            print(f'qconfig_dict:{qconfig_dict}')
            result = _cali_eval_model(t_model, qconfig_dict, device, cali_dataloader,
                                      val_dataloader, val_func, criterion, backend, args)

            metric = (target_metric - result[0]) if metric_big_best else (result[0] - target_metric)
            skip_layers.append(layer_name)
        else:
            break
    return skip_layers


def _profiling_by_acc(model: Module, val_dataloader: DataLoader, cali_dataloader: DataLoader, val_func,
                      criterion: _Loss, qconfig: QConfig, device: torch.device, target_metric: float,
                      metric_big_best: bool, backend: str, args):
    layer_sensitivity = {}
    for quantized_layer, module in model.named_modules():
        if module.__module__.startswith('torch.nn') and not isinstance(module, torch.nn.Sequential):
            if isinstance(module, torch.nn.ReLU):
                continue
            if isinstance(module, torch.nn.MaxPool2d):
                continue
            print(f'sensitivity profiling layer:{quantized_layer}')
            qconfig_dict = {
                "": None,
                "module_name": [(quantized_layer, qconfig)]
            }
            result = _cali_eval_model(model, qconfig_dict, device, cali_dataloader,
                                      val_dataloader, val_func, criterion, backend, args)
            print(f'layer {quantized_layer} acc:{result[0]}')
            sensitivity = (target_metric - result[0]) if metric_big_best else (result[0] - target_metric)
            layer_sensitivity[quantized_layer] = sensitivity
    return layer_sensitivity


def _profiling_by_loss(model: Module, val_dataloader: DataLoader, cali_dataloader: DataLoader, val_func,
                       criterion: _Loss, qconfig: QConfig, device: torch.device, backend: str, args):
    layer_sensitivity = {}
    model.to(device)
    with torch.no_grad():
        val_images, val_labels = next(iter(val_dataloader))
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)
        output = model(val_images)
        loss = criterion(output, val_labels)

    if str(device) != 'cpu':
        device = None  # type: ignore[assignment]
    for quantized_layer, module in model.named_modules():
        if module.__module__.startswith('torch.nn') and not isinstance(module, torch.nn.Sequential):
            if isinstance(module, torch.nn.ReLU):
                continue
            if isinstance(module, torch.nn.MaxPool2d):
                continue
            print(f'sensitivity profiling layer:{quantized_layer}')
            qconfig_dict = {
                "": None,
                "module_name": [(quantized_layer, qconfig)]
            }

            prepared_model = prepare_fx_with_backend(model, qconfig_dict, backend, False)
            prepared_model.apply(torch.quantization.enable_observer)
            prepared_model.apply(torch.quantization.disable_fake_quant)
            # calibrate model
            val_func(cali_dataloader, prepared_model, criterion, args, device=device)
            # evaluate model
            with torch.no_grad():
                quantized_output = prepared_model(val_images)
                quantized_loss = criterion(quantized_output, val_labels)
            sensitivity = quantized_loss - loss
            print(f'quantized_loss:{quantized_loss}，loss:{loss}, sensitivity:{sensitivity}')
            layer_sensitivity[quantized_layer] = sensitivity
    return layer_sensitivity


def _cali_eval_model(model: Module, qconfig_dict: dict, device: torch.device, cali_dataloader: DataLoader,
                     val_dataloader: DataLoader, val_func, criterion: _Loss, backend: str, args):
    model.to(device)
    if str(device) != 'cpu':
        device = None  # type: ignore[assignment]

    prepared_model = prepare_fx_with_backend(model, qconfig_dict, backend, False)
    prepared_model.apply(torch.quantization.enable_observer)
    prepared_model.apply(torch.quantization.disable_fake_quant)
    # calibrate model
    val_func(cali_dataloader, prepared_model, criterion, args, device=device)
    # evaluate model
    prepared_model.apply(torch.quantization.disable_observer)
    prepared_model.apply(torch.quantization.enable_fake_quant)
    result = val_func(val_dataloader, prepared_model, criterion, args, device=device)
    return result


def get_sensitivity_qconfig_dict(skip_layers: list):
    """Packaging the dict for extra_qconfig_dict in get_qconfig_dict

    Args:
        skip_layers: the layers which are sensitive to quantization

    Returns: None or the dict for extra_qconfig_dict

    """
    qconfig_dict = {}
    module_name_qconfig = []
    for not_quantized_layer in skip_layers:
        module_name_qconfig.append((not_quantized_layer, None))
    qconfig_dict.update({'module_name': module_name_qconfig})
    if len(skip_layers) == 0:
        qconfig_dict = None  # type: ignore[assignment]
    return qconfig_dict
