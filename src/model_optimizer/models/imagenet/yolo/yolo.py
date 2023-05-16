# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import math
import pkg_resources as pkg
import yaml
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .common import (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, Concat, Contract, Expand, check_yaml)
from .experimental import MixConv2d, CrossConv


PREFIX = 'AutoAnchor: '


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        print(s)
    return result


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def parse_model_backbone(d, ch):  # model_dict, input_channels(3)
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yolov5_backbone(pretrained=False, width_multiple=1.0, depth_multiple=1.0):
    """
    yolov5 backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5l.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    cfg_copy = deepcopy(cfg)
    cfg_copy['width_multiple'] = width_multiple
    cfg_copy['depth_multiple'] = depth_multiple

    backbone, _ = parse_model_backbone(cfg_copy, [3])

    yolov5temp_list = list(backbone.children())
    backbone_last = yolov5temp_list[-1]
    # print(f"backbone_last:{backbone_last}")
    # print("out_channels:", backbone_last.cv2.conv.out_channels)

    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(backbone_last.cv2.conv.out_channels, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    return backbone_classification


def yolov5(pretrained=False, width_multiple=1.0, depth_multiple=1.0):
    cfg_file = check_yaml("yolov5l.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    cfg_copy = deepcopy(cfg)
    cfg_copy['width_multiple'] = width_multiple
    cfg_copy['depth_multiple'] = depth_multiple
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


def yolov5l_backbone(pretrained=False):
    """
    yolov5l backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5l.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    backbone, _ = parse_model_backbone(deepcopy(cfg), [3])
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(1024, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    return backbone_classification


def yolov5m_backbone(pretrained=False):
    """
    yolov5 backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5m.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    backbone, _ = parse_model_backbone(deepcopy(cfg), [3])
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(768, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    return backbone_classification


def yolov5n_backbone(pretrained=False):
    """
    yolov5 backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5n.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    backbone, _ = parse_model_backbone(deepcopy(cfg), [3])
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(256, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    # print("backbone:", backbone_classification)
    return backbone_classification


def yolov5s_backbone(pretrained=False):
    """
    yolov5 backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5s.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    backbone, _ = parse_model_backbone(deepcopy(cfg), [3])
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(512, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    return backbone_classification


def yolov5x_backbone(pretrained=False):
    """
    yolov5 backbone + linear, no softmax
    Returns:

    """
    cfg_file = check_yaml("yolov5x.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    backbone, _ = parse_model_backbone(deepcopy(cfg), [3])
    global_avg_pool = nn.AdaptiveAvgPool2d(1)
    flatten = nn.Flatten(1)
    fc = nn.Linear(1280, 1000)
    backbone_classification = nn.Sequential(backbone, global_avg_pool, flatten, fc)
    return backbone_classification


def parse_model(d, ch):  # model_dict, input_channels(3)
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]   # input_num, output_num, n_repeats, ..
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def parse_model_custom(d, ch, backbone_layers):  # model_dict, input_channels(3)
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m
        if i < 10:  # backbone
            m_ = backbone_layers[i]
            if m in [C3]:
                c1, c2 = m_.cv1.conv.in_channels, m_.cv3.conv.out_channels
            elif m in [Conv]:
                c1, c2 = m_.conv.in_channels, m_.conv.out_channels
            elif m in [SPPF]:
                c1, c2 = m_.cv1.conv.in_channels, m_.cv2.conv.out_channels
            else:
                raise NotImplementedError(f'Our current parse_model_custom only support yolov5 v6.1.'
                                          f' {m} is not supported')
            n = n_ = max(round(n * gd), 1) if n > 1 else n
            args = [c1, c2, *args[1:]]
        else:
            # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                     BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2
            elif m is Expand:
                c2 = ch[f] // args[0] ** 2
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yolov5x(pretrained=False):
    cfg_file = check_yaml("yolov5x.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


def yolov5l(pretrained=False):
    cfg_file = check_yaml("yolov5l.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


def yolov5m(pretrained=False):
    cfg_file = check_yaml("yolov5m.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


def yolov5s(pretrained=False):
    cfg_file = check_yaml("yolov5s.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


def yolov5n(pretrained=False):
    cfg_file = check_yaml("yolov5n.yaml")
    with open(cfg_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict
    model, _ = parse_model(deepcopy(cfg), [3])
    return model


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, parse_custom=False, backbone_layers=None):
        # model, input channels, number of classes
        """

        :param cfg:
        :param ch:
        :param nc:
        :param anchors:
        :param parse_custom: if parse custom yolo model
        :param backbone_layers: used only when parse_custom=True
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        if parse_custom:
            self.model, self.save = parse_model_custom(deepcopy(self.yaml), [ch], backbone_layers)
        else:
            self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y = []  # outputs

        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


class Width(float, Enum):
    yolov5x = 1.25
    yolov5l = 1.0
    yolov5m = 0.75
    yolov5s = 0.5
    yolov5n = 0.25


class Depth(float, Enum):
    yolov5x = 1.33
    yolov5l = 1.0
    yolov5m = 0.67
    yolov5s = 0.33
    yolov5n = 0.33


if __name__ == "__main__":
    width_multiple = Width.yolov5m
    depth_multiple = Depth.yolov5s
    backbone = yolov5_backbone(pretrained=False, width_multiple=width_multiple, depth_multiple=depth_multiple)
    print(f'########## yolov5backbone--width_multiple:{width_multiple},depth_multiple:{depth_multiple} ')
    print(backbone)

    yolo = yolov5(pretrained=False, width_multiple=width_multiple, depth_multiple=depth_multiple)
    print(f'########## yolov5--width_multiple:{width_multiple},depth_multiple:{depth_multiple} ')
    print(yolo)
