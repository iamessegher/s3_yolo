# path: tools/export_qonnx.py
import argparse
import torch
import numpy as np

from models.yolo import Detect, Model
# from utils.weight_adapt import remap_state_dict_first_conv  # optional if loading RGB->5ch
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup


def load_checkpoint_state(weights_path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(weights_path, map_location=device)
    if "ema_state_dict" in ckpt:              # common
        return ckpt["ema_state_dict"]
    if "model_state_dict" in ckpt:            # your train.py
        return ckpt["model_state_dict"]
    if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        return ckpt["model"].float().state_dict()
    # last resort: assume whole file is a raw state_dict
    return ckpt


def mark_detect_export(m: torch.nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, Detect):
            mod.export = True  # many YOLO forks check this to run export-friendly forward


def main():
    p = argparse.ArgumentParser(description="Export YOLO to (Q)ONNX for FINN")
    p.add_argument("--cfg", type=str, required=True, help="model yaml")
    p.add_argument("--weights", type=str, default=None, help="checkpoint path (.pt)")
    p.add_argument("--output", type=str, required=True, help="output onnx path")
    p.add_argument("--channels", type=int, default=5, help="input channels (event slices)")
    p.add_argument("--nc", type=int, default=3, help="num classes")
    p.add_argument("--imgsz", type=int, default=320, help="square input H=W")
    p.add_argument("--imgszh", type=int, default=None, help="optional non-square H (if set, imgsz is W)")
    p.add_argument("--quantized", action="store_true", help="use Brevitas QONNX export (quantized model)")
    args = p.parse_args()

    device = "cpu"  # FINN/QONNX flow runs on CPU

    # Build model with correct input channels/classes
    model = Model(cfg=args.cfg, ch=args.channels, nc=args.nc).to(device)

    # Load weights if provided
    if args.weights:
        sd = load_checkpoint_state(args.weights, device)
        # sd = remap_state_dict_first_conv(sd, in_channels=args.channels)  # only if starting from RGB ckpt
        model.load_state_dict(sd, strict=False)

    model.eval()
    mark_detect_export(model)

    # Dummy input (B, C, H, W)
    H = int(args.imgszh) if args.imgszh is not None else int(args.imgsz)
    W = int(args.imgsz)
    dummy = torch.randn(1, args.channels, H, W, device=device)

    # Export
    if args.quantized:
        # QONNX for Brevitas-quantized models
        export_qonnx(model, dummy, export_path=args.output, export_params=True)
        # Optional graph cleanups
        qonnx_cleanup(args.output, out_file=args.output)
    else:
        # Standard ONNX for float models
        torch.onnx.export(
            model,
            dummy,
            args.output,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["outputs"],
            dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}},
        )

    print(f"Exported to {args.output}  (shape: 1x{args.channels}x{H}x{W})")


if __name__ == "__main__":
    main()
