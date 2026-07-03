import re
import torch
import torch.nn as nn

def _expand_weight_tensor(w_rgb: torch.Tensor, in_channels: int = 5, strategy: str = "avg") -> torch.Tensor:
    """Expand first conv weights [out,3,k,k] -> [out,in_channels,k,k]. Keeps RGB; fills extras."""
    if not (w_rgb.ndim == 4 and w_rgb.size(1) == 3):
        return w_rgb  # no-op if not 3-ch
    oc, _, kh, kw = w_rgb.shape
    w_new = torch.zeros((oc, in_channels, kh, kw), device=w_rgb.device, dtype=w_rgb.dtype)
    w_new[:, :3] = w_rgb
    if strategy in ("avg", "copy_rgb"):
        filler = w_rgb.mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            w_new[:, c:c+1] = filler
    elif strategy == "random":
        for c in range(3, in_channels):
            nn.init.kaiming_normal_(w_new[:, c:c+1], nonlinearity="leaky_relu")
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return w_new

def remap_state_dict_first_conv(sd: dict, in_channels: int = 5, strategy: str = "avg") -> dict:
    """If checkpoint's first conv expects 3 channels, expand to `in_channels`."""
    # Copy to avoid mutating the caller's dict
    out = dict(sd)
    # Try common first-conv key patterns; fallback to first *.weight
    conv_candidates = [k for k in out.keys() if k.endswith(".weight") and (
        ".conv.weight" in k or
        re.search(r"\bconv\d*\.weight$", k) or
        "cv1.weight" in k
    )]
    key = conv_candidates[0] if conv_candidates else next((k for k in out if k.endswith(".weight")), None)
    if key is None:
        return out
    w = out[key]
    if isinstance(w, torch.Tensor) and w.ndim == 4 and w.size(1) == 3 and in_channels != 3:
        out[key] = _expand_weight_tensor(w, in_channels=in_channels, strategy=strategy)
    return out