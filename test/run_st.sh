#!/usr/bin/env bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
echo $SHELL_FOLDER
path=$(dirname "$SHELL_FOLDER")
cd $path
export PYTHONPATH=${path}/src
echo $PYTHONPATH

./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_train_multi_gpu_consine_b64_lr0.025.prototxt
./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_train_multi_gpu_steplr_b64_lr0.025.prototxt



./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_autoslim_supernet_20epochs.prototxt
logger_path=`ls -lrt logger | tail -n 1 | awk -v P="$(pwd)" '{print P"/logger/"$NF}'`

sed "s#{{logger_path}}#${logger_path}/" test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_autoslim_search_param.prototxt > test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_autoslim_search.prototxt
./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_autoslim_search.prototxt
./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_quantization_ptq.prototxt
./run_cli.sh test/ci_test/classifier_imagenet/prototxt/resnet/resnet50_quantization_qat.prototxt
