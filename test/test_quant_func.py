'''
test quantization functionality
'''
import copy
import torch
from torch.quantization.quantize_fx import prepare_qat_fx
from torchvision import models
from model_optimizer.quantizer.deploy_fx_with_backend import check_input_data, export_onnx_model, convert_model_to_jit
from model_optimizer.quantizer.fusion_convert import fuse_prepared_model
from model_optimizer.utils import _parent_name, deepcopy_graphmodule, replace_module
from model_optimizer.quantizer.quant_utils import get_specified_not_quant_module


def test_input_data():
    '''
    test check_input_data
    Returns:
    '''
    input_data = [1, 3, 224, 224]

    input_data = check_input_data(input_data)
    print(f'input_data will be {input_data}')

    input_data = check_input_data(None)
    print(f'input is None, input_data will be {input_data}')


def test_get_module_name():
    '''
    test _parent_name
    Returns:
    '''
    input_name = 'layer1.block0.2.conv'
    parent_name, _ = _parent_name(input_name)

    print(f'the module {input_name} of parent name is {parent_name}')


def test_export_onnx_model():
    '''
    test export_onnx_model
    Returns:
    '''
    quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantization_dict = {"": quantization_config}

    model = models.resnet18()
    prepared_model = prepare_qat_fx(model, quantization_dict)
    prepared_model.eval()
    export_onnx_model(prepared_model, './', 'test_onnx.onnx', [1, 3, 224, 224])


def test_copy_graph():
    '''
    test deepcopy_graphmodule
    Return:
    '''
    quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantization_dict = {"": quantization_config}

    model = models.resnet18()
    prepared_model = prepare_qat_fx(model, quantization_dict)
    copy_model = deepcopy_graphmodule(prepared_model)
    print(f'deep copy the model {copy_model}')


def test_replace_module():
    '''
    test replace_module
    Return:
    '''
    model = models.resnet18()
    new_model = copy.deepcopy(model)
    sigmoid = torch.nn.Sigmoid()
    replace_module(new_model, model, torch.nn.ReLU, sigmoid)
    print(f'before replaced model: {model}, after replaced model: {new_model}')


def test_fused_prepared_model():
    '''
    test fuse_prepared_model
    Return:
    '''
    quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantization_dict = {"": quantization_config}

    model = models.resnet18()
    prepared_model = prepare_qat_fx(model, quantization_dict)
    prepared_model.eval()
    fused_model = fuse_prepared_model(prepared_model)
    print(f'before fused prepared model: {prepared_model}, after fused prepared model: {fused_model}')


def test_convert_jit_model():
    '''
    test convert_model_to_jit
    Return:
    '''
    quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantization_dict = {"": quantization_config}

    model = models.resnet18()
    prepared_model = prepare_qat_fx(model, quantization_dict)
    prepared_model.eval()
    input_shape = [16, 3, 224, 224]
    input_data = torch.rand(input_shape)
    convert_model_to_jit(prepared_model, input_data, 'test_model')


def test_get_match_pair():
    '''
    test get_specified_not_quant_module
    '''
    quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantization_dict = {"": quantization_config}

    model = models.resnet18()
    prepared_model = prepare_qat_fx(model, quantization_dict)
    prepared_model.eval()
    fused_model = fuse_prepared_model(prepared_model)
    export_onnx_model(fused_model, './', 'test_onnx_1.onnx', [1, 3, 224, 224])
    onnx_model_path = './test_onnx_1.onnx'
    qconfig_list = get_specified_not_quant_module(fused_model, onnx_model_path)
    print(f'The matched node pairs: {qconfig_list}')


if __name__ == '__main__':
    test_get_module_name()
    test_input_data()
    test_export_onnx_model()
    test_copy_graph()
    test_replace_module()
    test_fused_prepared_model()
    test_convert_jit_model()
    test_get_match_pair()
