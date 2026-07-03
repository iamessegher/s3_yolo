# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""Experimental modules."""
import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download
from collections import OrderedDict


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
    
# def attempt_load(weights, device=None, inplace=True, fuse=True):
#     # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
#     from models.yolo import Detect, Model

#     model = Ensemble()
#     for w in weights if isinstance(weights, list) else [weights]:
#         ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
#         ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

#         # Model compatibility updates
#         if not hasattr(ckpt, "stride"):
#             ckpt.stride = torch.tensor([32.0])
#         if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
#             ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

#         model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

#     # Module compatibility updates
#     for m in model.modules():
#         t = type(m)
#         if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
#             m.inplace = inplace  # torch 1.7.0 compatibility
#             if t is Detect and not isinstance(m.anchor_grid, list):
#                 delattr(m, "anchor_grid")
#                 setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
#         elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
#             m.recompute_scale_factor = None  # torch 1.11.0 compatibility

#     # Return model
#     if len(model) == 1:
#         return model[-1]

#     # Return detection ensemble
#     print(f"Ensemble created with {weights}\n")
#     for k in "names", "nc", "yaml":
#         setattr(model, k, getattr(model[0], k))
#     model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
#     assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
#     return model


def attempt_load(cfg,weights,hyp,f, device=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model
    import os

    model = Ensemble()   
    ckpt = torch.load(str(f), map_location=device) # loadt

    # Extract necessary components from your new checkpoint
    epoch = ckpt["epoch"]
    best_fitness = ckpt["best_fitness"]
    model_state_dict = ckpt["model_state_dict"]
    ema_state_dict = ckpt["ema_state_dict"]
    updates = ckpt["updates"]
    optimizer_state_dict = ckpt["optimizer_state_dict"]
    # opt = ckpt["opt"]

    # Create your DetectionModel instance
    detection_model = Model(cfg=cfg, ch=3, nc=3, anchors=hyp.get('anchors'))

    # Load the state_dict into your DetectionModel instance
    detection_model.load_state_dict(model_state_dict)
    detection_model.to(
        device
    ).eval()  # Move the model to the appropriate device and set to eval mode

    model.append(detection_model)

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f"Models have different class counts: {[m.nc for m in model]}"
    return model

# def inference_load(cfg,weights,device, inplace=True, fuse=True):
#     from models.yolo import Detect, Model
#     import os

#     model = Ensemble()   
#     ckpt = torch.load(weights[0], map_location=device) # loadt

#     # Extract necessary components from your new checkpoint
#     model_state_dict = ckpt["model_state_dict"]
#     # Create your DetectionModel instance
#     detection_model = Model(cfg=cfg, ch=5, nc=3)

#     # Load the state_dict into your DetectionModel instance
#     detection_model.load_state_dict(model_state_dict)
#     detection_model.to(device).eval()  # Move the model to the appropriate device and set to eval mode
#     model.append(detection_model)

#     # Module updates
#     for m in model.modules():
#         t = type(m)
#         if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
#             m.inplace = inplace
#             if t is Detect and not isinstance(m.anchor_grid, list):
#                 delattr(m, "anchor_grid")
#                 setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
#         elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
#             m.recompute_scale_factor = None  # torch 1.11.0 compatibility

#     # Return model
#     if len(model) == 1:
#         return model[-1]

#     # Return detection ensemble
#     print(f"Ensemble created with {weights}\n")
#     for k in "names", "nc", "yaml":
#         setattr(model, k, getattr(model[0], k))
#     model.stride = model[
#         torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
#     ].stride  # max stride
#     assert all(
#         model[0].nc == m.nc for m in model
#     ), f"Models have different class counts: {[m.nc for m in model]}"
#     return model
def _strip_prefix_if_present(state_dict, prefixes=("module.",)):
    clean = OrderedDict()
    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    return clean


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
            return ckpt["ema"].float().state_dict()
        if "model" in ckpt:
            model_obj = ckpt["model"]
            if hasattr(model_obj, "state_dict"):
                return model_obj.float().state_dict()
            return model_obj
    if hasattr(ckpt, "state_dict"):
        return ckpt.float().state_dict()
    if isinstance(ckpt, (dict, OrderedDict)):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _is_quantized_model(model):
    quant_names = {
        "QuantConv",
        "SQuantConv",
        "QuantSimpleConv",
        "QuantConv2d",
        "QuantLinear",
        "QuantReLU",
        "QuantIdentity",
        "QuantMaxPool2d",
    }
    return any(type(m).__name__ in quant_names for m in model.modules())

def load_trusted_checkpoint(weight_path, device):
    """
    Load a training checkpoint created by this project.

    Use weights_only=False only for trusted local checkpoints because PyTorch
    full checkpoint loading can execute pickled Python objects.
    """
    try:
        return torch.load(weight_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def inference_load(cfg, weights, device, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    model = Ensemble()

    weight_path = weights[0] if isinstance(weights, list) else weights
    ckpt = load_trusted_checkpoint(weight_path, device)

    model_state_dict = ckpt["model_state_dict"]

    detection_model = Model(cfg=cfg, ch=5, nc=3)
    detection_model.load_state_dict(model_state_dict)
    detection_model.to(device).eval()

    model.append(detection_model)

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]

    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))

    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride

    assert all(model[0].nc == m.nc for m in model), (
        f"Models have different class counts: {[m.nc for m in model]}"
    )

    return model