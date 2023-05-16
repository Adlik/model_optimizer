# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
run entry
"""
import argparse
import os

import google.protobuf as pb
# pylint: disable=unused-import
import google.protobuf.text_format  # noqa: F401
# pylint: enable=unused-import
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb

parser = argparse.ArgumentParser(description='ModelOptimizerTorch Begin')
parser.add_argument('--hp', type=str, help='File path to save hyperparameter configuration')
args = parser.parse_args()
assert os.path.exists(args.hp)
hyper_param = eppb.HyperParam()  # type: ignore[attr-defined]
with open(args.hp, 'r', encoding='utf-8') as file:
    pb.text_format.Merge(file.read(), hyper_param)
command = f'python {hyper_param.main_file} --hp {args.hp}'
os.system(command)
