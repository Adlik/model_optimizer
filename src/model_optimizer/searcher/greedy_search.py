# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
  Search with the greedy algorithm.
"""
import sys
import copy
import pickle
import os
import torch
import yaml
from ..utils.ptflops import get_model_complexity_info
from ..core import validate


class GreedySearcher:  # pylint: disable=too-many-branches,too-many-statements
    """Search with the greedy algorithm.

    We start with the largest model and compare the network accuracy among
    the architectures where each layer is slimmed by one channel bin. We then
    greedily slim the layer with minimal performance drop. During the iterative
    slimming, we obtain optimized channel configurations under different
    resource constraints. We stop until reaching the strictest constraint
    (e.g., 200M FLOPs).

    Args:
        algorithm (:obj:`torch.nn.Module`): Specific implemented algorithm
         based specific task, eg: AutoSlim.
        dataloader (:obj:`torch.nn.Dataloader`): Pytorch data loader.
        target_flops (list): The target flops of the searched models.
        args:
        max_channel_bins (int): The maximum number of channel bins in each
            layer. Note that each layer is slimmed by one channel bin.
        min_channel_bins (int): The minimum number of channel bins in each
            layer. Default to 1.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching. Defaults to "".
        distributed:
    """

    def __init__(self,
                 algorithm,
                 val_dataloader,
                 target_flops,
                 args,
                 max_channel_bins,
                 min_channel_bins=1,
                 resume_from="",
                 distributed=True):
        super().__init__()
        if not hasattr(algorithm.model, 'module'):
            raise NotImplementedError('Do not support searching with cpu.')

        self.algorithm = algorithm
        self.model = algorithm.model  # model processed by distributed_model
        if distributed:
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model
        self.val_dataloader = val_dataloader
        self.target_flops = sorted(target_flops, reverse=True)
        self.args = args
        self.max_channel_bins = max_channel_bins
        self.min_channel_bins = min_channel_bins
        self.resume_from = resume_from

    def search(self):
        """Greedy Slimming."""
        print("start to search")
        macs, params = get_model_complexity_info(self.model_without_ddp, self.algorithm.input_shape,
                                                 as_strings=True,
                                                 print_per_layer_stat=False,
                                                 ost=sys.stdout)
        print(f"########## macs:{macs}, params: {params}")
        work_dir = os.path.join(self.args.log_name, "search")
        is_exist = os.path.exists(work_dir)
        align_channel = self.args.hp.auto_slim.search_config.align_channel
        if self.args.hp.multi_gpu.rank == 0:
            if not is_exist:
                os.makedirs(work_dir)
        if self.resume_from != "":
            with open(self.resume_from, "rb") as resume_file:
                searcher_resume = pickle.load(resume_file)
            result_subnet = searcher_resume['result_subnet']
            result_flops = searcher_resume['result_flops']
            subnet = searcher_resume['subnet']
            flops = searcher_resume['flops']
            print(f'Resume from subnet: {subnet}, flops: {flops}')
        else:
            result_subnet, result_flops = [], []
            # We start with the largest model
            self.algorithm.pruner.set_max_channel()
            max_subnet = self.algorithm.pruner.get_max_channel_bins(
                self.max_channel_bins)
            # channel_cfg
            subnet = max_subnet

            # add for test
            self.algorithm.pruner.set_channel_bins(subnet, self.max_channel_bins, True, align_channel)
            flops = self.algorithm.get_subnet_flops()
            print("########## max flops:", flops)

        for target in self.target_flops:
            if self.resume_from != "" and flops <= target:
                continue

            if flops <= target:
                self.algorithm.pruner.set_channel_bins(subnet, self.max_channel_bins,
                                                       True, align_channel)
                channel_cfg = self.algorithm.pruner.export_subnet()
                result_subnet.append(channel_cfg)
                result_flops.append(flops)
                print(f'Find model flops {flops} <= {target}')
                continue

            while flops > target:
                # search which layer needs to shrink
                best_score = None
                best_subnet = None

                # During distributed training, the order of ``subnet.keys()``
                # on different ranks may be different. So we need to sort it
                # first.
                for _, name in enumerate(sorted(subnet.keys())):
                    new_subnet = copy.deepcopy(subnet)
                    # we prune the very last channel bin
                    last_bin_ind = torch.where(new_subnet[name] == 1)[0][-1]
                    # The ``new_subnet`` on different ranks are the same,
                    # so we do not need to broadcast here.
                    new_subnet[name][last_bin_ind] = 0  # set last bin of one layer to 0
                    if torch.sum(new_subnet[name]) < self.min_channel_bins:
                        # subnet is invalid
                        continue

                    self.algorithm.pruner.set_channel_bins(new_subnet,
                                                           self.max_channel_bins, True, align_channel)

                    top1, _ = validate(self.val_dataloader, None, None, self.args, self.algorithm)

                    score = top1
                    print(f'Slimming group {name}, Top-1: {score}')
                    if best_score is None or score > best_score:
                        best_score = score
                        best_subnet = new_subnet

                    torch.distributed.barrier()

                if best_subnet is None:
                    raise RuntimeError(
                        'Cannot find any valid model, check your '
                        'configurations.')

                subnet = best_subnet
                self.algorithm.pruner.set_channel_bins(subnet, self.max_channel_bins, True, align_channel)
                flops = self.algorithm.get_subnet_flops()
                print(
                    f'Greedy find model, score: {best_score}, FLOPS: {flops}')

                if self.args.hp.multi_gpu.rank == 0:
                    save_for_resume = {}
                    save_for_resume['result_subnet'] = result_subnet
                    save_for_resume['result_flops'] = result_flops
                    save_for_resume['subnet'] = subnet
                    save_for_resume['flops'] = flops
                    print('########## save_for_resume:', save_for_resume)
                    with open(os.path.join(work_dir, 'latest.pkl'), "wb+") as resume_file:
                        pickle.dump(save_for_resume, resume_file)

            self.algorithm.pruner.set_channel_bins(subnet, self.max_channel_bins, True,
                                                   self.args.hp.auto_slim.search_config.align_channel)
            channel_cfg = self.algorithm.pruner.export_subnet()
            result_subnet.append(channel_cfg)
            result_flops.append(flops)

        print('Search models done.')

        if self.args.hp.multi_gpu.rank == 0:
            for flops, subnet in zip(result_flops, result_subnet):
                with open(os.path.join(work_dir, f'subnet_{flops}.yaml'), 'w+', encoding='utf-8') as outfile:
                    yaml.dump(subnet, outfile, default_flow_style=False)
            print(f'Save searched results to {work_dir}')
