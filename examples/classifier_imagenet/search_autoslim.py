# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
autoslim search on multi gpu
"""
import time
import datetime
import os
import torch.distributed as dist

from model_optimizer.core import (get_base_parser, get_hyperparam, get_freer_gpu, main_s1_set_seed,
                                  main_s2_start_worker, display_model, process_model,
                                  distributed_model, get_summary_writer)
from model_optimizer.proto import model_optimizer_torch_pb2 as eppb
from model_optimizer.algorithms.autoslim.autoslim import AutoSlim
from model_optimizer.pruners.ratio_pruning import RatioPruner
from model_optimizer.searcher.greedy_search import GreedySearcher
from model_optimizer.datasets import DataloaderFactory
from model_optimizer.models import get_model_from_source

best_acc1 = 0


def main():
    """
    main process
    Returns:

    """
    parser = get_base_parser()
    args = parser.parse_args()
    hp = get_hyperparam(args)
    if hp.gpu_id == eppb.GPU.ANY:
        args.gpu = get_freer_gpu()
    elif hp.gpu_id == eppb.GPU.NONE:
        args.gpu = None  # TODO: test

    print("Start training")
    start_time = time.time()
    main_s1_set_seed(hp)
    main_s2_start_worker(main_worker, args, hp)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Searching time {total_time_str}')


# TODO: in mmrazor, samples_per_gpu is 1024, here is 256
def main_worker(gpu, ngpus_per_node, args):  # pylint: disable=too-many-branches,too-many-statements
    """
    main worker on per gpu
    Args:
        gpu:
        ngpus_per_node:
        args:

    Returns:

    """
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {args.gpu} for training")
    args.hp = get_hyperparam(args)
    if args.distributed:
        if args.hp.multi_gpu.dist_url == "env://" and args.hp.multi_gpu.rank == -1:
            args.hp.multi_gpu.rank = int(os.environ["RANK"])
        if args.hp.multi_gpu.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.hp.multi_gpu.rank = args.hp.multi_gpu.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.hp.multi_gpu.dist_backend, init_method=args.hp.multi_gpu.dist_url,
                                world_size=args.world_size, rank=args.hp.multi_gpu.rank)
    # create model
    if args.hp.pretrained:
        print(f"=> using pre-trained model '{args.hp.arch}'")
    else:
        print(f"=> creating model '{args.hp.arch}'")
    model = get_model_from_source(args.hp.arch, args.hp.model_source, args.hp.pretrained, args.hp.width_mult,
                                  args.hp.depth_mult)

    if args.gpu == 0:
        print(args)
        print('model:\n=========\n')
        display_model(model)
        print('\n=========\n')

    process_model(model, args)
    global best_acc1  # pylint: disable=global-statement
    best_acc1 = args.best_acc1

    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)

    dataload_factory = DataloaderFactory(args)
    train_loader, val_loader, _ = dataload_factory.product_train_val_loader(dataload_factory.imagenet2012)
    writer = get_summary_writer(args, ngpus_per_node, model)

    args.batch_num = len(train_loader)

    pruner = RatioPruner(args.hp.auto_slim.ratio_pruner.ratios,
                         except_start_keys=args.hp.auto_slim.ratio_pruner.except_start_keys)
    algorithm = AutoSlim(model, pruner, None, input_shape=tuple(args.hp.auto_slim.search_config.input_shape),
                         bn_training_mode=args.hp.auto_slim.bn_training_mode, distributed=args.distributed)

    searcher = GreedySearcher(algorithm, val_loader, args.hp.auto_slim.search_config.greedy_searcher.target_flops,
                              args, args.hp.auto_slim.search_config.greedy_searcher.max_channel_bins,
                              resume_from=args.hp.auto_slim.search_config.greedy_searcher.resume_from,
                              distributed=args.distributed)
    searcher.search()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
