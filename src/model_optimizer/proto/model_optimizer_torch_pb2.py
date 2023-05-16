# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model_optimizer_torch.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1bmodel_optimizer_torch.proto\x12\x15model_optimizer_torch\"\xfb\r\n\nHyperParam\x12\x37\n\tmain_file\x18\x01 \x01(\t:$examples/classifier_imagenet/main.py\x12\x15\n\x04\x61rch\x18\x02 \x01(\t:\x07\x61lexnet\x12\x15\n\nwidth_mult\x18\t \x01(\x02:\x01\x31\x12\x15\n\ndepth_mult\x18\x1d \x01(\x02:\x01\x31\x12\x43\n\x0cmodel_source\x18\x03 \x01(\x0e\x32-.model_optimizer_torch.HyperParam.ModelSource\x12\x1a\n\x08log_name\x18\x04 \x01(\t:\x08template\x12\x0c\n\x04\x64\x61ta\x18\x05 \x02(\t\x12\r\n\x05\x64\x65\x62ug\x18\x06 \x01(\x08\x12\x14\n\x0coverfit_test\x18\x07 \x01(\x08\x12\x1a\n\x12validate_data_full\x18\x08 \x01(\x08\x12\x0f\n\x02lr\x18\n \x01(\x02:\x03\x30.1\x12\x12\n\x06\x65pochs\x18\x0b \x01(\x05:\x02\x39\x30\x12\x17\n\nbatch_size\x18\x0c \x01(\x05:\x03\x32\x35\x36\x12\x1b\n\x0eval_batch_size\x18\x18 \x01(\x05:\x03\x32\x35\x36\x12\x12\n\x07workers\x18\r \x01(\x05:\x01\x34\x12\x16\n\nprint_freq\x18\x0e \x01(\x05:\x02\x35\x30\x12\x10\n\x08\x65valuate\x18\x0f \x01(\x08\x12\x12\n\npretrained\x18\x10 \x01(\x08\x12\x0c\n\x04seed\x18\x11 \x01(\x05\x12\x13\n\x0b\x65xport_onnx\x18\x12 \x01(\x08\x12\x15\n\ronnx_simplify\x18\x1e \x01(\x08\x12\x0e\n\x06resume\x18\x13 \x01(\t\x12\x0e\n\x06weight\x18\x16 \x01(\t\x12*\n\x06gpu_id\x18\x14 \x01(\x0e\x32\x1a.model_optimizer_torch.GPU\x12\x32\n\tmulti_gpu\x18\x15 \x01(\x0b\x32\x1f.model_optimizer_torch.MultiGPU\x12\x32\n\tauto_slim\x18\x17 \x01(\x0b\x32\x1f.model_optimizer_torch.AutoSlim\x12\x18\n\tis_subnet\x18\x19 \x01(\x08:\x05\x66\x61lse\x12\x1c\n\x0fval_resize_size\x18\x1a \x01(\x05:\x03\x32\x35\x36\x12\x1a\n\rval_crop_size\x18\x1b \x01(\x05:\x03\x32\x32\x34\x12\x1c\n\x0ftrain_crop_size\x18\x1c \x01(\x05:\x03\x32\x32\x34\x12\x0f\n\x07\x65xcepts\x18\x35 \x01(\t\x12-\n\x06warmup\x18\x63 \x01(\x0b\x32\x1d.model_optimizer_torch.Warmup\x12;\n\x0clr_scheduler\x18\x64 \x01(\x0e\x32%.model_optimizer_torch.LRScheduleType\x12\x33\n\x07step_lr\x18\x65 \x01(\x0b\x32\".model_optimizer_torch.StepLRParam\x12>\n\rmulti_step_lr\x18\x66 \x01(\x0b\x32\'.model_optimizer_torch.MultiStepLRParam\x12\x37\n\tcyclic_lr\x18g \x01(\x0b\x32$.model_optimizer_torch.CyclicLRParam\x12\x41\n\x0e\x65xponential_lr\x18h \x01(\x0b\x32).model_optimizer_torch.ExponentialLRParam\x12\x38\n\toptimizer\x18\xc8\x01 \x01(\x0e\x32$.model_optimizer_torch.OptimizerType\x12-\n\x03sgd\x18\xc9\x01 \x01(\x0b\x32\x1f.model_optimizer_torch.SGDParam\x12/\n\x04\x61\x64\x61m\x18\xca\x01 \x01(\x0b\x32 .model_optimizer_torch.AdamParam\x12\x35\n\x07rmsprop\x18\xcb\x01 \x01(\x0b\x32#.model_optimizer_torch.RMSpropParam\x12\x1b\n\x0flabel_smoothing\x18\xac\x02 \x01(\x02:\x01\x30\x12<\n\x0b\x61utoaugment\x18\xb6\x02 \x01(\x0e\x32&.model_optimizer_torch.AutoAugmentType\x12\x17\n\x0bmixup_alpha\x18\xc0\x02 \x01(\x02:\x01\x30\x12\x18\n\x0c\x63utmix_alpha\x18\xca\x02 \x01(\x02:\x01\x30\x12\x13\n\x03\x61mp\x18\x90\x03 \x01(\x08:\x05\x66\x61lse\x12\x30\n\x07\x64istill\x18\x9a\x03 \x01(\x0b\x32\x1e.model_optimizer_torch.Distill\x12:\n\x0cquantization\x18\xc2\x03 \x01(\x0b\x32#.model_optimizer_torch.Quantization\x12\x1e\n\x0esave_jit_trace\x18\xcc\x03 \x01(\x08:\x05\x66\x61lse\"B\n\x0bModelSource\x12\x0f\n\x0bTorchVision\x10\x01\x12\r\n\tPyTorchCV\x10\x02\x12\t\n\x05Local\x10\x03\x12\x08\n\x04Timm\x10\x04\"\x9d\x01\n\x08MultiGPU\x12\x16\n\nworld_size\x18\x01 \x01(\x05:\x02-1\x12\x0f\n\x04rank\x18\x02 \x01(\x05:\x01\x30\x12\'\n\x08\x64ist_url\x18\x03 \x01(\t:\x15tcp://127.0.0.1:23456\x12\x1a\n\x0c\x64ist_backend\x18\x04 \x01(\t:\x04nccl\x12#\n\x1bmultiprocessing_distributed\x18\x05 \x01(\x08\"\xd2\x01\n\x08\x41utoSlim\x12\x38\n\x0cratio_pruner\x18\x01 \x01(\x0b\x32\".model_optimizer_torch.RatioPruner\x12\x1e\n\x10\x62n_training_mode\x18\x02 \x01(\x08:\x04true\x12\x34\n\rsearch_config\x18\x03 \x01(\x0b\x32\x1d.model_optimizer_torch.Search\x12\x19\n\nretraining\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x13\x63hannel_config_path\x18\x05 \x01(\t\"8\n\x0bRatioPruner\x12\x0e\n\x06ratios\x18\x01 \x03(\x02\x12\x19\n\x11\x65xcept_start_keys\x18\x02 \x03(\t\"\\\n\x0cKLDivergence\x12\x16\n\x0btemperature\x18\x01 \x01(\x02:\x01\x31\x12\x1c\n\treduction\x18\x02 \x01(\t:\tbatchmean\x12\x16\n\x0bloss_weight\x18\x03 \x01(\x02:\x01\x31\"\x81\x01\n\x07\x44istill\x12:\n\rteacher_model\x18\x01 \x01(\x0b\x32#.model_optimizer_torch.TeacherModel\x12:\n\rkl_divergence\x18\x02 \x01(\x0b\x32#.model_optimizer_torch.KLDivergence\"P\n\x0cTeacherModel\x12\x0c\n\x04\x61rch\x18\x01 \x03(\t\x12\x32\n\x06source\x18\x02 \x03(\x0e\x32\".model_optimizer_torch.ModelSource\"\x8e\x02\n\x14QuantizationObserver\x12#\n\x13quantization_method\x18\x01 \x01(\t:\x06minmax\x12\x19\n\x0bper_channel\x18\x02 \x01(\x08:\x04true\x12\x18\n\tsymmetric\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x0creduce_range\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x19\n\npercentile\x18\x05 \x01(\x02:\x05\x39\x39.99\x12\x15\n\x05\x64type\x18\x06 \x01(\t:\x06quint8\x12\x10\n\x05nbits\x18\x07 \x01(\x05:\x01\x38\x12\x1a\n\x0b\x66\x61ke_method\x18\x08 \x01(\t:\x05\x62\x61sic\x12\x1f\n\x17layers_restrict_to_8bit\x18\t \x01(\t\"\xa0\x01\n\x13SensitivityAnalysis\x12S\n\x10sensitivity_type\x18\x01 \x01(\x0e\x32&.model_optimizer_torch.SensitivityType:\x11ONE_AT_A_TIME_ACC\x12\x15\n\rtarget_metric\x18\x02 \x01(\x02\x12\x1d\n\x0fmetric_big_best\x18\x03 \x01(\x08:\x04true\"\x9c\x04\n\x0cQuantization\x12\x16\n\x08quantize\x18\x01 \x01(\x08:\x04true\x12\x19\n\x0bquantize_fx\x18\x02 \x01(\x08:\x04true\x12$\n\x16post_training_quantize\x18\x03 \x01(\x08:\x04true\x12\x46\n\x07\x62\x61\x63kend\x18\x04 \x01(\x0e\x32\'.model_optimizer_torch.InferenceBackend:\x0cTORCH_FBGEMM\x12#\n\x17num_calibration_batches\x18\x05 \x01(\x05:\x02\x33\x32\x12%\n\x1anum_observer_update_epochs\x18\x06 \x01(\x05:\x01\x34\x12+\n\x1cnum_batch_norm_update_epochs\x18\x07 \x01(\x05:\x05\x39\x39\x39\x39\x39\x12U\n activation_quantization_observer\x18\x08 \x01(\x0b\x32+.model_optimizer_torch.QuantizationObserver\x12Q\n\x1cweight_quantization_observer\x18\t \x01(\x0b\x32+.model_optimizer_torch.QuantizationObserver\x12H\n\x14sensitivity_analysis\x18\n \x01(\x0b\x32*.model_optimizer_torch.SensitivityAnalysis\"\x8c\x01\n\x06Search\x12\x13\n\x0bweight_path\x18\x01 \x01(\t\x12\x13\n\x0binput_shape\x18\x02 \x03(\x05\x12>\n\x0fgreedy_searcher\x18\x03 \x01(\x0b\x32%.model_optimizer_torch.GreedySearcher\x12\x18\n\ralign_channel\x18\x04 \x01(\x05:\x01\x38\"Y\n\x0eGreedySearcher\x12\x14\n\x0ctarget_flops\x18\x01 \x03(\x03\x12\x1c\n\x10max_channel_bins\x18\x02 \x01(\x05:\x02\x31\x32\x12\x13\n\x0bresume_from\x18\x04 \x01(\t\"?\n\x08SGDParam\x12\x1c\n\x0cweight_decay\x18\x01 \x01(\x02:\x06\x30.0001\x12\x15\n\x08momentum\x18\x02 \x01(\x02:\x03\x30.9\")\n\tAdamParam\x12\x1c\n\x0cweight_decay\x18\x01 \x01(\x02:\x06\x30.0001\"C\n\x0cRMSpropParam\x12\x1c\n\x0cweight_decay\x18\x01 \x01(\x02:\x06\x30.0001\x12\x15\n\x08momentum\x18\x02 \x01(\x02:\x03\x30.9\"D\n\x06Warmup\x12\x1b\n\x10lr_warmup_epochs\x18\x01 \x01(\x05:\x01\x35\x12\x1d\n\x0flr_warmup_decay\x18\x02 \x01(\x02:\x04\x30.01\"8\n\x0bStepLRParam\x12\x15\n\tstep_size\x18\x03 \x01(\x05:\x02\x32\x30\x12\x12\n\x05gamma\x18\x04 \x01(\x02:\x03\x30.1\":\n\x10MultiStepLRParam\x12\x12\n\nmilestones\x18\x03 \x03(\x05\x12\x12\n\x05gamma\x18\x04 \x01(\x02:\x03\x30.1\")\n\x12\x45xponentialLRParam\x12\x13\n\x05gamma\x18\x04 \x01(\x02:\x04\x30.97\"\xe7\x01\n\rCyclicLRParam\x12\x0f\n\x07\x62\x61se_lr\x18\x01 \x01(\x02\x12\x0e\n\x06max_lr\x18\x02 \x01(\x02\x12\x1a\n\x0cstep_size_up\x18\x03 \x01(\x05:\x04\x32\x30\x30\x30\x12\x16\n\x0estep_size_down\x18\x04 \x01(\x05\x12\x37\n\x04mode\x18\x05 \x01(\x0e\x32).model_optimizer_torch.CyclicLRParam.Mode\x12\x10\n\x05gamma\x18\x06 \x01(\x02:\x01\x31\"6\n\x04Mode\x12\x0e\n\ntriangular\x10\x01\x12\x0f\n\x0btriangular2\x10\x02\x12\r\n\texp_range\x10\x03*B\n\x0bModelSource\x12\x0f\n\x0bTorchVision\x10\x01\x12\r\n\tPyTorchCV\x10\x02\x12\t\n\x05Local\x10\x03\x12\x08\n\x04Timm\x10\x04*7\n\x0f\x41utoAugmentType\x12\x06\n\x02RA\x10\x01\x12\x0b\n\x07TA_WIDE\x10\x02\x12\x0f\n\x0b\x41UTOAUGMENT\x10\x03*\x18\n\x03GPU\x12\x07\n\x03\x41NY\x10\x01\x12\x08\n\x04NONE\x10\x02*J\n\x0fSensitivityType\x12\x15\n\x11ONE_AT_A_TIME_ACC\x10\x00\x12\x16\n\x12ONE_AT_A_TIME_LOSS\x10\x01\x12\x08\n\x04SQNR\x10\x02*\\\n\x10InferenceBackend\x12\x10\n\x0cTORCH_FBGEMM\x10\x01\x12\x11\n\rTORCH_QNNPACK\x10\x02\x12\x07\n\x03TVM\x10\x03\x12\x0c\n\x08TENSORRT\x10\x04\x12\x0c\n\x08OPENVINO\x10\x05*/\n\rOptimizerType\x12\x07\n\x03SGD\x10\x01\x12\x08\n\x04\x41\x64\x61m\x10\x02\x12\x0b\n\x07RMSprop\x10\x03*e\n\x0eLRScheduleType\x12\n\n\x06StepLR\x10\x01\x12\x0f\n\x0bMultiStepLR\x10\x02\x12\x15\n\x11\x43osineAnnealingLR\x10\x03\x12\x0c\n\x08\x43yclicLR\x10\x04\x12\x11\n\rExponentialLR\x10\x05')

_MODELSOURCE = DESCRIPTOR.enum_types_by_name['ModelSource']
ModelSource = enum_type_wrapper.EnumTypeWrapper(_MODELSOURCE)
_AUTOAUGMENTTYPE = DESCRIPTOR.enum_types_by_name['AutoAugmentType']
AutoAugmentType = enum_type_wrapper.EnumTypeWrapper(_AUTOAUGMENTTYPE)
_GPU = DESCRIPTOR.enum_types_by_name['GPU']
GPU = enum_type_wrapper.EnumTypeWrapper(_GPU)
_SENSITIVITYTYPE = DESCRIPTOR.enum_types_by_name['SensitivityType']
SensitivityType = enum_type_wrapper.EnumTypeWrapper(_SENSITIVITYTYPE)
_INFERENCEBACKEND = DESCRIPTOR.enum_types_by_name['InferenceBackend']
InferenceBackend = enum_type_wrapper.EnumTypeWrapper(_INFERENCEBACKEND)
_OPTIMIZERTYPE = DESCRIPTOR.enum_types_by_name['OptimizerType']
OptimizerType = enum_type_wrapper.EnumTypeWrapper(_OPTIMIZERTYPE)
_LRSCHEDULETYPE = DESCRIPTOR.enum_types_by_name['LRScheduleType']
LRScheduleType = enum_type_wrapper.EnumTypeWrapper(_LRSCHEDULETYPE)
TorchVision = 1
PyTorchCV = 2
Local = 3
Timm = 4
RA = 1
TA_WIDE = 2
AUTOAUGMENT = 3
ANY = 1
NONE = 2
ONE_AT_A_TIME_ACC = 0
ONE_AT_A_TIME_LOSS = 1
SQNR = 2
TORCH_FBGEMM = 1
TORCH_QNNPACK = 2
TVM = 3
TENSORRT = 4
OPENVINO = 5
SGD = 1
Adam = 2
RMSprop = 3
StepLR = 1
MultiStepLR = 2
CosineAnnealingLR = 3
CyclicLR = 4
ExponentialLR = 5


_HYPERPARAM = DESCRIPTOR.message_types_by_name['HyperParam']
_MULTIGPU = DESCRIPTOR.message_types_by_name['MultiGPU']
_AUTOSLIM = DESCRIPTOR.message_types_by_name['AutoSlim']
_RATIOPRUNER = DESCRIPTOR.message_types_by_name['RatioPruner']
_KLDIVERGENCE = DESCRIPTOR.message_types_by_name['KLDivergence']
_DISTILL = DESCRIPTOR.message_types_by_name['Distill']
_TEACHERMODEL = DESCRIPTOR.message_types_by_name['TeacherModel']
_QUANTIZATIONOBSERVER = DESCRIPTOR.message_types_by_name['QuantizationObserver']
_SENSITIVITYANALYSIS = DESCRIPTOR.message_types_by_name['SensitivityAnalysis']
_QUANTIZATION = DESCRIPTOR.message_types_by_name['Quantization']
_SEARCH = DESCRIPTOR.message_types_by_name['Search']
_GREEDYSEARCHER = DESCRIPTOR.message_types_by_name['GreedySearcher']
_SGDPARAM = DESCRIPTOR.message_types_by_name['SGDParam']
_ADAMPARAM = DESCRIPTOR.message_types_by_name['AdamParam']
_RMSPROPPARAM = DESCRIPTOR.message_types_by_name['RMSpropParam']
_WARMUP = DESCRIPTOR.message_types_by_name['Warmup']
_STEPLRPARAM = DESCRIPTOR.message_types_by_name['StepLRParam']
_MULTISTEPLRPARAM = DESCRIPTOR.message_types_by_name['MultiStepLRParam']
_EXPONENTIALLRPARAM = DESCRIPTOR.message_types_by_name['ExponentialLRParam']
_CYCLICLRPARAM = DESCRIPTOR.message_types_by_name['CyclicLRParam']
_HYPERPARAM_MODELSOURCE = _HYPERPARAM.enum_types_by_name['ModelSource']
_CYCLICLRPARAM_MODE = _CYCLICLRPARAM.enum_types_by_name['Mode']
HyperParam = _reflection.GeneratedProtocolMessageType('HyperParam', (_message.Message,), {
  'DESCRIPTOR' : _HYPERPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.HyperParam)
  })
_sym_db.RegisterMessage(HyperParam)

MultiGPU = _reflection.GeneratedProtocolMessageType('MultiGPU', (_message.Message,), {
  'DESCRIPTOR' : _MULTIGPU,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.MultiGPU)
  })
_sym_db.RegisterMessage(MultiGPU)

AutoSlim = _reflection.GeneratedProtocolMessageType('AutoSlim', (_message.Message,), {
  'DESCRIPTOR' : _AUTOSLIM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.AutoSlim)
  })
_sym_db.RegisterMessage(AutoSlim)

RatioPruner = _reflection.GeneratedProtocolMessageType('RatioPruner', (_message.Message,), {
  'DESCRIPTOR' : _RATIOPRUNER,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.RatioPruner)
  })
_sym_db.RegisterMessage(RatioPruner)

KLDivergence = _reflection.GeneratedProtocolMessageType('KLDivergence', (_message.Message,), {
  'DESCRIPTOR' : _KLDIVERGENCE,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.KLDivergence)
  })
_sym_db.RegisterMessage(KLDivergence)

Distill = _reflection.GeneratedProtocolMessageType('Distill', (_message.Message,), {
  'DESCRIPTOR' : _DISTILL,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.Distill)
  })
_sym_db.RegisterMessage(Distill)

TeacherModel = _reflection.GeneratedProtocolMessageType('TeacherModel', (_message.Message,), {
  'DESCRIPTOR' : _TEACHERMODEL,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.TeacherModel)
  })
_sym_db.RegisterMessage(TeacherModel)

QuantizationObserver = _reflection.GeneratedProtocolMessageType('QuantizationObserver', (_message.Message,), {
  'DESCRIPTOR' : _QUANTIZATIONOBSERVER,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.QuantizationObserver)
  })
_sym_db.RegisterMessage(QuantizationObserver)

SensitivityAnalysis = _reflection.GeneratedProtocolMessageType('SensitivityAnalysis', (_message.Message,), {
  'DESCRIPTOR' : _SENSITIVITYANALYSIS,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.SensitivityAnalysis)
  })
_sym_db.RegisterMessage(SensitivityAnalysis)

Quantization = _reflection.GeneratedProtocolMessageType('Quantization', (_message.Message,), {
  'DESCRIPTOR' : _QUANTIZATION,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.Quantization)
  })
_sym_db.RegisterMessage(Quantization)

Search = _reflection.GeneratedProtocolMessageType('Search', (_message.Message,), {
  'DESCRIPTOR' : _SEARCH,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.Search)
  })
_sym_db.RegisterMessage(Search)

GreedySearcher = _reflection.GeneratedProtocolMessageType('GreedySearcher', (_message.Message,), {
  'DESCRIPTOR' : _GREEDYSEARCHER,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.GreedySearcher)
  })
_sym_db.RegisterMessage(GreedySearcher)

SGDParam = _reflection.GeneratedProtocolMessageType('SGDParam', (_message.Message,), {
  'DESCRIPTOR' : _SGDPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.SGDParam)
  })
_sym_db.RegisterMessage(SGDParam)

AdamParam = _reflection.GeneratedProtocolMessageType('AdamParam', (_message.Message,), {
  'DESCRIPTOR' : _ADAMPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.AdamParam)
  })
_sym_db.RegisterMessage(AdamParam)

RMSpropParam = _reflection.GeneratedProtocolMessageType('RMSpropParam', (_message.Message,), {
  'DESCRIPTOR' : _RMSPROPPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.RMSpropParam)
  })
_sym_db.RegisterMessage(RMSpropParam)

Warmup = _reflection.GeneratedProtocolMessageType('Warmup', (_message.Message,), {
  'DESCRIPTOR' : _WARMUP,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.Warmup)
  })
_sym_db.RegisterMessage(Warmup)

StepLRParam = _reflection.GeneratedProtocolMessageType('StepLRParam', (_message.Message,), {
  'DESCRIPTOR' : _STEPLRPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.StepLRParam)
  })
_sym_db.RegisterMessage(StepLRParam)

MultiStepLRParam = _reflection.GeneratedProtocolMessageType('MultiStepLRParam', (_message.Message,), {
  'DESCRIPTOR' : _MULTISTEPLRPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.MultiStepLRParam)
  })
_sym_db.RegisterMessage(MultiStepLRParam)

ExponentialLRParam = _reflection.GeneratedProtocolMessageType('ExponentialLRParam', (_message.Message,), {
  'DESCRIPTOR' : _EXPONENTIALLRPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.ExponentialLRParam)
  })
_sym_db.RegisterMessage(ExponentialLRParam)

CyclicLRParam = _reflection.GeneratedProtocolMessageType('CyclicLRParam', (_message.Message,), {
  'DESCRIPTOR' : _CYCLICLRPARAM,
  '__module__' : 'model_optimizer_torch_pb2'
  # @@protoc_insertion_point(class_scope:model_optimizer_torch.CyclicLRParam)
  })
_sym_db.RegisterMessage(CyclicLRParam)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MODELSOURCE._serialized_start=1776
  _MODELSOURCE._serialized_end=1842
  _AUTOAUGMENTTYPE._serialized_start=4506
  _AUTOAUGMENTTYPE._serialized_end=4561
  _GPU._serialized_start=4563
  _GPU._serialized_end=4587
  _SENSITIVITYTYPE._serialized_start=4589
  _SENSITIVITYTYPE._serialized_end=4663
  _INFERENCEBACKEND._serialized_start=4665
  _INFERENCEBACKEND._serialized_end=4757
  _OPTIMIZERTYPE._serialized_start=4759
  _OPTIMIZERTYPE._serialized_end=4806
  _LRSCHEDULETYPE._serialized_start=4808
  _LRSCHEDULETYPE._serialized_end=4909
  _HYPERPARAM._serialized_start=55
  _HYPERPARAM._serialized_end=1842
  _HYPERPARAM_MODELSOURCE._serialized_start=1776
  _HYPERPARAM_MODELSOURCE._serialized_end=1842
  _MULTIGPU._serialized_start=1845
  _MULTIGPU._serialized_end=2002
  _AUTOSLIM._serialized_start=2005
  _AUTOSLIM._serialized_end=2215
  _RATIOPRUNER._serialized_start=2217
  _RATIOPRUNER._serialized_end=2273
  _KLDIVERGENCE._serialized_start=2275
  _KLDIVERGENCE._serialized_end=2367
  _DISTILL._serialized_start=2370
  _DISTILL._serialized_end=2499
  _TEACHERMODEL._serialized_start=2501
  _TEACHERMODEL._serialized_end=2581
  _QUANTIZATIONOBSERVER._serialized_start=2584
  _QUANTIZATIONOBSERVER._serialized_end=2854
  _SENSITIVITYANALYSIS._serialized_start=2857
  _SENSITIVITYANALYSIS._serialized_end=3017
  _QUANTIZATION._serialized_start=3020
  _QUANTIZATION._serialized_end=3560
  _SEARCH._serialized_start=3563
  _SEARCH._serialized_end=3703
  _GREEDYSEARCHER._serialized_start=3705
  _GREEDYSEARCHER._serialized_end=3794
  _SGDPARAM._serialized_start=3796
  _SGDPARAM._serialized_end=3859
  _ADAMPARAM._serialized_start=3861
  _ADAMPARAM._serialized_end=3902
  _RMSPROPPARAM._serialized_start=3904
  _RMSPROPPARAM._serialized_end=3971
  _WARMUP._serialized_start=3973
  _WARMUP._serialized_end=4041
  _STEPLRPARAM._serialized_start=4043
  _STEPLRPARAM._serialized_end=4099
  _MULTISTEPLRPARAM._serialized_start=4101
  _MULTISTEPLRPARAM._serialized_end=4159
  _EXPONENTIALLRPARAM._serialized_start=4161
  _EXPONENTIALLRPARAM._serialized_end=4202
  _CYCLICLRPARAM._serialized_start=4205
  _CYCLICLRPARAM._serialized_end=4436
  _CYCLICLRPARAM_MODE._serialized_start=4382
  _CYCLICLRPARAM_MODE._serialized_end=4436
# @@protoc_insertion_point(module_scope)