"""
test functions
"""

import argparse
from torchvision import models
import torch

import google.protobuf as pb
from model_optimizer.core import get_lr_scheduler
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb


def get_hyperparam_from_content(args, proto_content):
    """
    get hyperparam from proto
    Args:
        args:
        proto_content:

    Returns:

    """
    hyper_param = eppb.HyperParam()
    pb.text_format.Merge(proto_content, hyper_param)
    args.hp = hyper_param


def test_get_step_lr_scheduler():
    """
    test get_step_lr_scheduler
    Returns:

    """
    args = argparse.Namespace()
    proto_content = """
                    warmup {
                      lr_warmup_epochs: 5
                      lr_warmup_decay: 0.01
                    }
                    lr: 0.1
                    lr_scheduler: StepLR
                    step_lr {
                      step_size: 30
                      gamma: 0.1
                    }
                    epochs: 100
                    """
    get_hyperparam_from_content(args, proto_content)
    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.hp.lr,
                                momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, args)
    for i in range(args.hp.epochs):
        optimizer.step()
        scheduler.step()
        print(f'{i}: {optimizer.param_groups[0]["lr"]}')


def test_get_multi_step_lr_scheduler():
    """
    test get_multi_step_lr_scheduler
    Returns:

    """
    args = argparse.Namespace()
    proto_content = """
                    warmup {
                      lr_warmup_epochs: 5
                      lr_warmup_decay: 0.01
                    }
                    lr: 0.1
                    lr_scheduler: MultiStepLR
                    multi_step_lr {
                      milestones: [30, 60, 80]
                      gamma: 0.1
                    }
                    epochs: 100

                    """
    get_hyperparam_from_content(args, proto_content)
    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.hp.lr,
                                momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, args)
    for i in range(args.hp.epochs):
        optimizer.step()
        scheduler.step()
        print(f'{i}: {optimizer.param_groups[0]["lr"]}')


def test_get_cosine_lr_scheduler():
    """
    test get_cosine_lr_scheduler
    Returns:

    """
    args = argparse.Namespace()
    proto_content = """
                    warmup {
                      lr_warmup_epochs: 5
                      lr_warmup_decay: 0.01
                    }
                    lr: 0.1
                    lr_scheduler: CosineAnnealingLR
                    epochs: 100
                    """
    get_hyperparam_from_content(args, proto_content)
    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.hp.lr,
                                momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, args)
    for i in range(args.hp.epochs):
        optimizer.step()
        scheduler.step()
        print(f'{i}: {optimizer.param_groups[0]["lr"]}')


if __name__ == '__main__':
    test_get_step_lr_scheduler()
    test_get_multi_step_lr_scheduler()
    test_get_cosine_lr_scheduler()
