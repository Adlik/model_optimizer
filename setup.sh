#!/usr/bin/env bash

export CURRENT_DIR=`pwd`
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}/src

protoc -I=${CURRENT_DIR}/src/model_optimizer/proto --python_out=${CURRENT_DIR}/src/model_optimizer/proto ${CURRENT_DIR}/src/model_optimizer/proto/model_optimizer_torch.proto
cd src/model_optimizer
python proto/gene_hyperparam_template.py