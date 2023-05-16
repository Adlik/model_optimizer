# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration mechanism
"""
from collections import OrderedDict


BACKEND_QUANTIZER_FUNCTION = OrderedDict()


def register_backend_quantizer(backend):  # pylint: disable=missing-function-docstring
    def update(quantizer_function):
        BACKEND_QUANTIZER_FUNCTION[backend] = quantizer_function
        return quantizer_function
    return update


FUSED_MODULE_CONVERT_FUNCTION = OrderedDict()


def register_convert_fucntion(module_type):  # pylint: disable=missing-function-docstring
    def insert(fn):
        FUSED_MODULE_CONVERT_FUNCTION[module_type] = fn
        return fn
    return insert


DEPLOY_FUNCTION_BY_BACKEND = OrderedDict()  # type: ignore[var-annotated]


def register_deploy_function(backend):  # pylint: disable=missing-function-docstring
    def insert(fn):
        if backend in DEPLOY_FUNCTION_BY_BACKEND:
            DEPLOY_FUNCTION_BY_BACKEND[backend].append(fn)
        else:
            DEPLOY_FUNCTION_BY_BACKEND[backend] = [fn]
        return fn
    return insert
