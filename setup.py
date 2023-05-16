#!/usr/bin/env python3

# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model optimizer.
"""

from setuptools import find_packages, setup
from pkg_resources import DistributionNotFound, get_distribution
_VERSION = '0.0.1'

_REQUIRED_PACKAGES = [
    'torch==1.12.1',
    'torchvision==0.13.1',
    'timm==0.6.5',
    'ordered-set==4.0.2',
    'plotly==5.7.0',
    'pytorchcv==0.0.67',
    'tensorboard==2.6.0',
    'tensorboardX==2.5',
    'pyyaml==6.0',
    'protobuf==3.19.4',
    'pandas',
    'onnx'
]

_TEST_REQUIRES = [
    'bandit',
    'pytest==7.0.1',
    'pytest-cov',
    'flake8==4.0.1',
    'pytest-flake8-v2',
    'pytest-mypy',
    'pytest-pylint',
    'pytest-xdist',
    'types-PyYAML',
    'types-setuptools',
    'types-protobuf'
]


def get_dist(pkgname):
    """
    Get distribution
    :param pkgname: str, package name
    :return:
    """
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


if get_dist('torch') is not None and get_dist('torchvision') is not None:
    _REQUIRED_PACKAGES.remove('torch==1.12.1')
    _REQUIRED_PACKAGES.remove('torchvision==0.13.1')

setup(
    name="model_optimizer",
    version=_VERSION.replace('-', ''),
    author='ZTE',
    author_email='ai@zte.com.cn',
    packages=find_packages('src'),
    # packages=find_packages(exclude=('test',)),
    package_dir={'': 'src'},
    description=__doc__,
    license='Apache 2.0',
    keywords='optimizer model',
    install_requires=_REQUIRED_PACKAGES,
    extras_require={'test': _TEST_REQUIRES},
    package_data={
        'model_optimizer': ['**/*.prototxt',
                            'pruner/scheduler/uniform_auto/*.yaml',
                            'pruner/scheduler/uniform_specified_layer/*.yaml',
                            'pruner/scheduler/distill/*.yaml',
                            'models/imagenet/yolo/*.yaml']
    },

)
