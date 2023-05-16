# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Warp the prepare_qat_fx API with backend
"""
from torch.quantization.quantize_fx import prepare_qat_fx
from torch.quantization.fake_quantize import FakeQuantizeBase
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.utils.registry import BACKEND_QUANTIZER_FUNCTION
from model_optimizer.quantizer.fusion_convert import fuse_prepared_model


# mypy: disable-error-code="attr-defined"
export_onnx_backend = [eppb.InferenceBackend.TENSORRT]


def prepare_fx_with_backend(model, qconfig_dict, backend, is_qat, **kwargs):
    """Prepare to insert the FakeQuantize model based on different backend

    Args:
        model (torch.nn.Module): torch.nn.Module model

        qconfig_dict (dict): qconfig_dict is a dictionary with the following configurations:
            qconfig_dict = {
                # optional, global config
                "": qconfig?,
                # optional, used for module and function types
                # could also be split into module_types and function_types if we prefer
                "object_type": [
                  (torch.nn.Conv2d, qconfig?),
                  (torch.nn.functional.add, qconfig?),
                  ...,
                 ],
                # optional, used for module names
                "module_name": [
                  ("foo.bar", qconfig?)
                  ...,
                ],

        backend: different backend that specifies how operators are quantized, this includes how
        fake quant ops are inserted etc. Currently supported backends include:
        [TORCH_FBGEMM, TORCH_QNNPACK, TVM, TENSORRT]

        is_qat: bool

    Returns:
        A GraphModule with fake quant modules (configured by qconfig_mapping and backend), ready for
      calibration and quantization aware training
    """
    model = prepare_qat_fx(model, qconfig_dict, **kwargs)

    if not is_qat:
        model.eval()
        model = fuse_prepared_model(model)

    if backend in export_onnx_backend:
        print(f"Prepare to insert the FakeQuantize model based on {backend}")
        node_name_not_quantize = get_not_quantize_node_name(model)
        model = BACKEND_QUANTIZER_FUNCTION[backend](model, node_name_not_quantize)

    return model


def get_not_quantize_node_name(model):
    """Gets the name of the node that is not quantified

    Args:
        model (torch.nn.Module): torch.nn.Module model with qconfig map

    Returns:
        List of node names that is not quantified
    """
    node_name_not_quantize = []
    qconfig_map = model._qconfig_map  # pylint: disable=protected-access

    modules = dict(model.named_modules(remove_duplicate=False))

    for node in model.graph.nodes:
        if node.op == "call_module" and isinstance(modules[node.target], FakeQuantizeBase) or node.op == "placeholder":
            continue

        if qconfig_map[node.name] is None:
            node_name_not_quantize.append(node.name)

    if not node_name_not_quantize:
        node_name_not_quantize = None

    return node_name_not_quantize
