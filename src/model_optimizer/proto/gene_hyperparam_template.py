import os

import google.protobuf as pb
import google.protobuf.text_format

from model_optimizer.proto import model_optimizer_torch_pb2 as eppb


def main():
    gene_base_template()


def gene_base_template():
    # default values
    hyper = eppb.HyperParam()
    hyper.main_file = hyper.main_file
    hyper.arch = hyper.arch
    hyper.model_source = hyper.model_source
    hyper.debug = hyper.debug
    hyper.overfit_test = hyper.overfit_test
    hyper.log_name = hyper.log_name
    hyper.data = f"/home/{os.getenv('USER')}/datasets/data.cifar10"
    hyper.workers = hyper.workers
    hyper.batch_size = hyper.batch_size
    hyper.val_batch_size = hyper.val_batch_size
    hyper.print_freq = hyper.print_freq
    hyper.evaluate = hyper.evaluate
    hyper.pretrained = hyper.pretrained
    hyper.lr = hyper.lr
    hyper.width_mult = hyper.width_mult
    hyper.depth_mult = hyper.depth_mult
    hyper.epochs = hyper.epochs
    hyper.resume = hyper.resume
    hyper.weight = hyper.weight

    hyper.amp = hyper.amp
    hyper.seed = hyper.seed
    hyper.export_onnx = hyper.export_onnx

    hyper.gpu_id = hyper.gpu_id
    hyper.multi_gpu.world_size = hyper.multi_gpu.world_size
    hyper.multi_gpu.rank = hyper.multi_gpu.rank
    hyper.multi_gpu.dist_url = hyper.multi_gpu.dist_url
    hyper.multi_gpu.dist_backend = hyper.multi_gpu.dist_backend
    hyper.multi_gpu.multiprocessing_distributed = hyper.multi_gpu.multiprocessing_distributed

    hyper.warmup.lr_warmup_epochs = hyper.warmup.lr_warmup_epochs
    hyper.warmup.lr_warmup_decay = hyper.warmup.lr_warmup_decay
    # print(hyper.HasField("warmup"))

    hyper.optimizer = hyper.optimizer
    hyper.sgd.weight_decay = hyper.sgd.weight_decay
    hyper.sgd.momentum = hyper.sgd.momentum
    hyper.adam.weight_decay = hyper.adam.weight_decay

    hyper.lr_scheduler = hyper.lr_scheduler
    # StepLR
    hyper.step_lr.step_size = hyper.step_lr.step_size
    hyper.step_lr.gamma = hyper.step_lr.gamma
    # MultiStepLR
    hyper.multi_step_lr.milestones.extend([20, 30, 50])
    # list(hyper.multi_step_lr.milestones) convert it to list
    hyper.multi_step_lr.gamma = hyper.multi_step_lr.gamma
    # CyclicLR
    hyper.cyclic_lr.base_lr = hyper.cyclic_lr.base_lr
    hyper.cyclic_lr.max_lr = hyper.cyclic_lr.max_lr
    hyper.cyclic_lr.step_size_up = hyper.cyclic_lr.step_size_up
    hyper.cyclic_lr.mode = hyper.cyclic_lr.mode
    hyper.cyclic_lr.gamma = hyper.cyclic_lr.gamma
    # AutoSlim
    hyper.auto_slim.bn_training_mode = hyper.auto_slim.bn_training_mode
    hyper.auto_slim.retraining = hyper.auto_slim.retraining
    hyper.auto_slim.channel_config_path = hyper.auto_slim.channel_config_path
    hyper.auto_slim.ratio_pruner.ratios.extend([2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0])
    hyper.auto_slim.ratio_pruner.except_start_keys.extend(["classifier.1"])
    hyper.auto_slim.search_config.weight_path = hyper.auto_slim.search_config.weight_path
    hyper.auto_slim.search_config.align_channel = hyper.auto_slim.search_config.align_channel
    hyper.auto_slim.search_config.input_shape.extend([3, 224, 224])
    hyper.auto_slim.search_config.greedy_searcher.target_flops.extend([500000000, 300000000, 200000000])
    hyper.auto_slim.search_config.greedy_searcher.max_channel_bins = \
        hyper.auto_slim.search_config.greedy_searcher.max_channel_bins
    hyper.auto_slim.search_config.greedy_searcher.resume_from = \
        hyper.auto_slim.search_config.greedy_searcher.resume_from
    hyper.label_smoothing = hyper.label_smoothing
    hyper.autoaugment = hyper.autoaugment
    hyper.mixup_alpha = hyper.mixup_alpha
    hyper.cutmix_alpha = hyper.cutmix_alpha
    hyper.is_subnet = hyper.is_subnet
    hyper.val_resize_size = hyper.val_resize_size
    hyper.val_crop_size = hyper.val_crop_size
    hyper.train_crop_size = hyper.train_crop_size

    # distillation
    # hyper.distill.teacher_model.arch
    # hyper.distill.teacher_model.source

    with open(os.path.join('proto', 'template.prototxt'), 'w') as wf:
        print(hyper)
        print('Writing hyper parameter at ./proto/template.prototxt')
        wf.write(str(hyper))


"""
Load hyperparameter from prototxt.
```python
person2 = addressbook_pb2.Person()
with open('person.prototxt', 'r') as rf:
    pb.text_format.Merge(rf.read(), person2)
```
"""


def dump_object(obj):
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(dump_object, value)
            else:
                dump_object(value)
        else:
            print("%s: %s" % (descriptor.full_name, value))


if __name__ == '__main__':
    main()
