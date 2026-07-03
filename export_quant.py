# path: tools/export_finn_qonnx_quant.py
"""
Export an already QAT-trained Brevitas quantized YOLO model to QONNX for FINN.

Quantized-only exporter.

Recommended FINN boundary:
    --export-mode raw
        Export Detect head tensors before sigmoid/grid/anchor decode.
        Decode + NMS should run outside FINN.

Optional:
    --export-mode decoded
        Export decoded YOLO predictions. This includes sigmoid/grid/anchor logic.
"""

import argparse
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from brevitas.export import export_qonnx

from models.yolo import Detect, Model


def install_onnx_mapping_compat() -> None:
    """
    Compatibility shim for old QONNX versions with newer ONNX.

    Some QONNX releases import `from onnx import mapping`, while newer ONNX
    exposes the internal table as `onnx._mapping`.
    """
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        return

    if hasattr(onnx, "mapping"):
        return

    if not hasattr(onnx, "_mapping"):
        return

    tensor_type_map = getattr(onnx._mapping, "TENSOR_TYPE_MAP", None)
    if tensor_type_map is None:
        return

    mapping_module = types.ModuleType("onnx.mapping")

    tensor_type_to_np_type = {}
    tensor_type_to_storage_tensor_type = {}
    np_type_to_tensor_type = {}

    for tensor_type, dtype_info in tensor_type_map.items():
        np_dtype = getattr(dtype_info, "np_dtype", None)
        storage_dtype = getattr(dtype_info, "storage_dtype", tensor_type)

        if np_dtype is None:
            continue

        np_dtype = np.dtype(np_dtype)
        tensor_type_to_np_type[tensor_type] = np_dtype
        tensor_type_to_storage_tensor_type[tensor_type] = storage_dtype
        np_type_to_tensor_type[np_dtype] = tensor_type
        np_type_to_tensor_type[np_dtype.type] = tensor_type

    mapping_module.TENSOR_TYPE_TO_NP_TYPE = tensor_type_to_np_type
    mapping_module.NP_TYPE_TO_TENSOR_TYPE = np_type_to_tensor_type
    mapping_module.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = tensor_type_to_storage_tensor_type

    mapping_module.STORAGE_TENSOR_TYPE_TO_FIELD = {
        TensorProto.FLOAT: "float_data",
        TensorProto.DOUBLE: "double_data",
        TensorProto.INT64: "int64_data",
        TensorProto.UINT64: "uint64_data",
        TensorProto.INT32: "int32_data",
        TensorProto.UINT32: "uint64_data",
        TensorProto.INT16: "int32_data",
        TensorProto.UINT16: "int32_data",
        TensorProto.INT8: "int32_data",
        TensorProto.UINT8: "int32_data",
        TensorProto.BOOL: "int32_data",
        TensorProto.FLOAT16: "int32_data",
        TensorProto.BFLOAT16: "int32_data",
    }

    onnx.mapping = mapping_module
    sys.modules["onnx.mapping"] = mapping_module


def cleanup_qonnx_inplace(output_path: Path, preserve_qnt_ops: bool = True) -> None:
    """
    Run QONNX cleanup lazily.

    Importing qonnx.cleanup at module import time can fail on Python 3.12
    with newer ONNX, so cleanup is imported only after the compatibility shim.
    """
    install_onnx_mapping_compat()

    try:
        from qonnx.util.cleanup import cleanup as qonnx_cleanup
    except Exception as exc:
        raise RuntimeError(
            "QONNX export succeeded, but qonnx cleanup could not be imported. "
            "Use --skip-cleanup to keep the raw exported QONNX file, or upgrade qonnx."
        ) from exc

    try:
        qonnx_cleanup(
            str(output_path),
            out_file=str(output_path),
            preserve_qnt_ops=preserve_qnt_ops,
        )
    except TypeError:
        qonnx_cleanup(str(output_path), out_file=str(output_path))


def load_trusted_checkpoint(weights_path: str, device: str = "cpu") -> dict[str, Any]:
    """
    Load a trusted local checkpoint created by this project.

    PyTorch 2.6+ defaults to weights_only=True, which can reject older project
    checkpoints containing numpy/python objects.
    """
    try:
        return torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def extract_state_dict(checkpoint: Any, prefer_ema: bool = True) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if prefer_ema and "ema_state_dict" in checkpoint:
            return checkpoint["ema_state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "ema_state_dict" in checkpoint:
            return checkpoint["ema_state_dict"]
        if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
            return checkpoint["model"].float().state_dict()

    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "No model weights found. Expected 'ema_state_dict', 'model_state_dict', "
        "'model', or a raw state_dict checkpoint."
    )


def strip_prefixes(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...] = ("module.",),
) -> dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    return cleaned


def model_has_brevitas_quant(model: nn.Module) -> bool:
    for module in model.modules():
        module_name = type(module).__name__.lower()
        module_path = type(module).__module__.lower()

        if module_path.startswith("brevitas."):
            return True
        if "quant" in module_name:
            return True

    return False


def set_detect_export_flag(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, Detect):
            module.export = enabled


class FinnYoloExportWrapper(nn.Module):
    """
    Wrap YOLO output into a QONNX-friendly export boundary.

    raw:
        Returns raw Detect tensors, one per scale. Recommended for FINN.

    decoded:
        Returns decoded YOLO predictions. This includes sigmoid/grid/anchor logic.
    """

    def __init__(self, model: nn.Module, export_mode: str):
        super().__init__()
        self.model = model
        self.export_mode = export_mode

        if export_mode == "raw":
            set_detect_export_flag(self.model, False)
        elif export_mode == "decoded":
            set_detect_export_flag(self.model, True)
        else:
            raise ValueError(f"Unsupported export_mode={export_mode!r}")

    def forward(self, x: torch.Tensor):
        output = self.model(x)

        if self.export_mode == "decoded":
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(
                "Raw export expected YOLO eval output as (decoded_predictions, raw_heads). "
                "Check Detect.forward() and model.eval()."
            )

        raw_heads = output[1]
        if not isinstance(raw_heads, (tuple, list)):
            return raw_heads

        return tuple(raw_heads)


def load_model(args: argparse.Namespace, device: str) -> nn.Module:
    model = Model(cfg=args.cfg, ch=args.channels, nc=args.nc).to(device)

    checkpoint = load_trusted_checkpoint(args.weights, device)
    state_dict = extract_state_dict(checkpoint, prefer_ema=not args.no_ema)
    state_dict = strip_prefixes(state_dict)

    strict = not args.allow_partial_load
    load_result = model.load_state_dict(state_dict, strict=strict)

    missing = [k for k in load_result.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in load_result.unexpected_keys if not k.endswith("num_batches_tracked")]

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint/model mismatch during strict load.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    if args.allow_partial_load and (missing or unexpected):
        print("WARNING: partial checkpoint load")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model.eval()
    return model


def output_names_from_example(outputs) -> list[str]:
    if isinstance(outputs, torch.Tensor):
        return ["output"]

    if isinstance(outputs, (tuple, list)):
        return [f"output_{idx}" for idx in range(len(outputs))]

    raise TypeError(f"Unsupported output type for export: {type(outputs)}")


def export_quantized_qonnx(
    wrapper: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    skip_cleanup: bool,
) -> None:
    export_qonnx(
        wrapper,
        dummy,
        export_path=str(output_path),
        export_params=True,
    )

    if not skip_cleanup:
        cleanup_qonnx_inplace(output_path, preserve_qnt_ops=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an already QAT-trained Brevitas quantized YOLO model to QONNX for FINN."
    )

    parser.add_argument("--cfg", type=str, required=True, help="quantized model YAML")
    parser.add_argument("--weights", type=str, required=True, help="QAT checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="output .onnx/.qonnx path")
    parser.add_argument("--channels", type=int, default=5, help="input channels")
    parser.add_argument("--nc", type=int, default=3, help="number of classes")
    parser.add_argument("--imgsz", type=int, default=640, help="input width, or square size if --imgszh is unset")
    parser.add_argument("--imgszh", type=int, default=None, help="optional input height")
    parser.add_argument(
        "--export-mode",
        choices=("raw", "decoded"),
        default="raw",
        help="raw exports Detect heads; decoded exports YOLO decoded predictions",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="use model_state_dict instead of ema_state_dict when both exist",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="allow missing/unexpected checkpoint keys; not recommended for final FPGA export",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="skip qonnx.cleanup; useful if old qonnx is incompatible with Python 3.12 ONNX packages",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cpu"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height = int(args.imgszh) if args.imgszh is not None else int(args.imgsz)
    width = int(args.imgsz)
    dummy = torch.zeros(1, args.channels, height, width, device=device)

    model = load_model(args, device)

    if not model_has_brevitas_quant(model):
        raise RuntimeError(
            "The built model does not appear to contain Brevitas quantized layers. "
            "Use the quantized/QAT YAML that exactly matches the checkpoint."
        )

    wrapper = FinnYoloExportWrapper(model, export_mode=args.export_mode).eval()

    with torch.no_grad():
        example_outputs = wrapper(dummy)

    output_names = output_names_from_example(example_outputs)
    export_quantized_qonnx(wrapper, dummy, output_path, skip_cleanup=args.skip_cleanup)

    print(
        f"Exported QONNX: {output_path}\n"
        f"Input shape: 1x{args.channels}x{height}x{width}\n"
        f"Export mode: {args.export_mode}\n"
        f"Outputs: {output_names}\n"
        f"Cleanup: {'skipped' if args.skip_cleanup else 'applied'}"
    )


if __name__ == "__main__":
    main()