# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
dataset
"""
import os
from functools import partial
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from ..proto import model_optimizer_torch_pb2 as eppb
from ..datasets import transforms as custom_transforms


class DataloaderFactory:
    """
    data load factory
    """
    # MNIST
    mnist = 0
    # CIFAR10
    cifar10 = 10
    # ImageNet2012
    imagenet2012 = 40

    def __init__(self, args):
        self.args = args

    def set_args(self, args):
        """
        set args
        Args:
            args:

        Returns:

        """
        self.args = args

    def _get_dataset(self, data_type, data_dir, train_tansform, val_transform):
        if data_type == self.mnist:
            train_dir = data_dir
            val_dir = data_dir
            train_dataset = torchvision.datasets.MNIST(train_dir, train=True, download=True,
                                                       transform=train_tansform)
            test_dataset = torchvision.datasets.MNIST(val_dir, train=False, transform=val_transform)
        elif data_type == self.cifar10:
            train_dir = data_dir
            val_dir = data_dir
            train_dataset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True,
                                                         transform=train_tansform)
            test_dataset = torchvision.datasets.CIFAR10(root=val_dir, train=False, download=True,
                                                        transform=val_transform)
        elif data_type == self.imagenet2012:
            train_dir = os.path.join(data_dir, 'train')
            val_dir = os.path.join(data_dir, 'val')
            train_dataset = torchvision.datasets.ImageFolder(
                train_dir,
                train_tansform)

            test_dataset = torchvision.datasets.ImageFolder(val_dir, val_transform)
        return train_dataset, test_dataset

    def _get_transform(self, data_type, train_crop_size, val_resize_size, val_crop_size, autoaugment):
        train_transform = []
        val_transform = []
        # MNIST
        if data_type == self.mnist:
            train_transform.extend([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            val_transform = train_transform

        # CIFAR10
        elif data_type == self.cifar10:
            train_transform.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            val_transform.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # ImageNet
        elif data_type == self.imagenet2012:
            # Data loading code

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            auto_augment_policy = autoaugment
            train_transform.extend([
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip()
                ])
            if auto_augment_policy is not None:
                if auto_augment_policy == eppb.AutoAugmentType.RA:
                    train_transform.append(transforms.autoaugment.RandAugment())
                elif auto_augment_policy == eppb.AutoAugmentType.TA_WIDE:
                    train_transform.append(transforms.autoaugment.TrivialAugmentWide())
                else:
                    aa_policy = transforms.autoaugment.AutoAugmentPolicy(auto_augment_policy)
                    train_transform.append(transforms.autoaugment.AutoAugment(policy=aa_policy))
            train_transform.extend([
                transforms.ToTensor(),
                normalize,
                ])
            val_transform.extend([
                transforms.Resize(val_resize_size),
                transforms.CenterCrop(val_crop_size),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            raise NotImplementedError

        return train_transform, val_transform

    # pylint: disable=too-many-branches,too-many-statements
    def product_train_val_loader(self,
                                 data_type,
                                 num_batches=0,
                                 use_val_trans=False,
                                 batch_same_data=False):
        """
        product train validate loader
        Args:
            data_type:
            num_batches:
            use_val_trans:
            batch_same_data:

        Returns:

        """
        args = self.args
        noverfit = not args.hp.overfit_test

        train_transform, val_transform = self._get_transform(
            data_type, args.hp.train_crop_size, args.hp.val_resize_size, args.hp.val_crop_size, args.hp.autoaugment)
        if use_val_trans:
            train_transform = val_transform
        train_transform_compose = transforms.Compose(train_transform)
        val_transform_compose = transforms.Compose(val_transform)
        train_dataset, val_dataset = self._get_dataset(
            data_type, args.hp.data, train_transform_compose, val_transform_compose)

        collate_fn = None
        num_classes = len(train_dataset.classes)

        if num_batches > 0:
            train_dataset = torch.utils.data.Subset(
                train_dataset,
                indices=list(range(self.args.hp.batch_size * num_batches)))
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            if not args.hp.validate_data_full:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            else:
                val_sampler = None
        else:
            train_sampler = None
            val_sampler = None

        mixup_transforms = []
        if args.hp.mixup_alpha > 0.0:
            mixup_transforms.append(custom_transforms.RandomMixup(num_classes, p=1.0, alpha=args.hp.mixup_alpha))
        if args.hp.cutmix_alpha > 0.0:
            mixup_transforms.append(custom_transforms.RandomCutmix(num_classes, p=1.0, alpha=args.hp.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = self.data_augment_collate_fn

        if batch_same_data:
            train_sampler = None
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.hp.batch_size, shuffle=False,
                num_workers=0, pin_memory=False, sampler=None)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.hp.batch_size, shuffle=(train_sampler is None) and noverfit,
                num_workers=args.hp.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True,
                collate_fn=partial(collate_fn, mixupcutmix) if collate_fn else default_collate)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.hp.val_batch_size, shuffle=False,
            num_workers=args.hp.workers, pin_memory=True, sampler=val_sampler, persistent_workers=True)
        return train_loader, val_loader, train_sampler

    # pylint: enable=too-many-branches,too-many-statements
    @staticmethod
    def data_augment_collate_fn(mixupcutmix, batch):
        """ collate function
        Args:
            mixupcutmix:
            batch:
        data augment

        Returns:

        """
        return mixupcutmix(*default_collate(batch))
