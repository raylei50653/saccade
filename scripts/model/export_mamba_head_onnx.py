#!/usr/bin/env python3
"""Export PyTorch MambaHead to ONNX with custom SelectiveScan op.

The `selective_scan_fwd` CUDA kernel is replaced with a torch.autograd.Function
whose `symbolic()` method emits a `saccade::SelectiveScan` ONNX node.  This
node is later compiled by TensorRT via the plugin in `libsaccade_scan_plugin.so`.

Usage:
    uv run scripts/model/export_mamba_head_onnx.py

Output:
    models/yolo/mamba_head.onnx
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import torch
import torch.nn as nn
from torch import Tensor

from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead
from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)

# ---------------------------------------------------------------------------
# Custom autograd Function — forward calls CUDA kernel, symbolic emits ONNX node
# ---------------------------------------------------------------------------


class SelectiveScanFn(torch.autograd.Function):
    """Wraps `selective_scan_fwd` so ONNX export produces a single plugin node.

    During ONNX tracing, forward() returns a shaped placeholder (no CUDA call);
    symbolic() provides the actual ONNX node that TensorRT will later compile via
    the SelectiveScan plugin in libsaccade_scan_plugin.so.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
    ) -> Tensor:
        if torch.jit.is_tracing():
            return torch.empty_like(u)

        try:
            import saccade_tracking_ext  # noqa: F811
        except ImportError:
            return torch.empty_like(u)

        N = A.shape[-1]
        a_per_channel = 1 if (A.dim() == 2 and A.shape[0] == u.shape[2]) else 0
        is_half = u.dtype == torch.float16

        if C.shape[-1] < N:
            C = C.expand(*C.shape[:-1], N).contiguous()
        else:
            C = C.contiguous()

        u = u.contiguous()
        delta = delta.contiguous()
        A = A.to(u.dtype).contiguous()
        B = B.contiguous()
        D_tensor = (
            D if D.numel() > 0 else torch.empty(0, dtype=u.dtype, device=u.device)
        )
        has_D = 1 if D.numel() > 0 else 0

        y = torch.empty_like(u)
        stream = torch.cuda.current_stream(u.device).cuda_stream

        saccade_tracking_ext.selective_scan_fwd(
            u.data_ptr(),
            delta.data_ptr(),
            A.data_ptr(),
            B.data_ptr(),
            C.data_ptr(),
            D_tensor.data_ptr() if has_D else 0,
            y.data_ptr(),
            u.shape[0],
            u.shape[1],
            u.shape[2],
            N,
            has_D,
            a_per_channel,
            is_half,
            stream,
        )

        return y

    @staticmethod
    def symbolic(
        g: torch.onnx.GraphContext,
        u: torch.Value,
        delta: torch.Value,
        A: torch.Value,
        B: torch.Value,
        C: torch.Value,
        D: torch.Value,
    ) -> torch.Value:
        # C may be rank-1 in the last dim (c_rank=1 from legacy checkpoints).
        # Broadcast to (B, L, N) before the plugin, since the CUDA kernel
        # indexes C with N-stride and expects the full dimension.
        import torch.onnx

        N = A.type().dim()  # A is (1, N) or (d_inner, N)
        g.op("Tile", C, g.op("Constant", value_t=torch.tensor([1, 1, N])))
        # Actually Tile doesn't work easily with dynamic shapes. Instead,
        # the plugin handles the C broadcast internally.
        return g.op(
            "saccade::SelectiveScan",
            u,
            delta,
            A,
            B,
            C,
            D,
            domain_s="saccade",
            outputs=1,
        ).setType(u.type())


# ---------------------------------------------------------------------------
# ONNX export wrapper — fixed I/O
# ---------------------------------------------------------------------------

_STRIDES = (8, 16, 32)
_IN_CHANNELS = (128, 256, 512)
_NUM_CLASSES = 80
_REG_MAX = 1
_IMG_SIZE = 640


class ONNXMambaHead(nn.Module):
    """Wraps MambaDetectionHead so forward() has 3 inputs + 6 outputs."""

    def __init__(self, head: MambaDetectionHead):
        super().__init__()
        self.head = head

    def forward(
        self,
        p3: Tensor,
        p4: Tensor,
        p5: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        feats = [p3, p4, p5]
        cls_preds, reg_preds = self.head(feats, return_embeddings=False)
        return (
            cls_preds[0],
            cls_preds[1],
            cls_preds[2],
            reg_preds[0],
            reg_preds[1],
            reg_preds[2],
        )


def _patch_selective_scan(module: nn.Module) -> None:
    """Recursively replace _selective_scan calls with SelectiveScanFn.apply."""
    import saccade.perception.temporal_yolo.mamba_head as mh

    mh._selective_scan = lambda u, delta, A, B, C, D=None: SelectiveScanFn.apply(
        u,
        delta,
        A,
        B,
        C,
        D if D is not None else torch.empty(0, dtype=u.dtype, device=u.device),
    )


def export() -> None:
    device = torch.device("cuda")

    print("Building detector (v14 mamba head)...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str(project_root / "models/yolo/yolo26s.pt"),
        teacher_ckpt=str(project_root / "runs/gated_det_v1/best.ckpt"),
        mamba_ckpt=str(project_root / "runs/mamba_gt_vgt_mamba_v14/best.ckpt"),
        img_size=_IMG_SIZE,
        device=device,
        emb_dim=0,
    )
    detector.eval()
    head = detector.mamba_head

    _patch_selective_scan(head)
    wrapper = ONNXMambaHead(head).eval()

    # Dummy FPN features matching backbone output shapes
    H, W = _IMG_SIZE, _IMG_SIZE
    p3_dummy = torch.zeros(1, 128, H // 8, W // 8, device=device)
    p4_dummy = torch.zeros(1, 256, H // 16, W // 16, device=device)
    p5_dummy = torch.zeros(1, 512, H // 32, W // 32, device=device)

    print("Warm-up forward...")
    with torch.no_grad():
        wrapper(p3_dummy, p4_dummy, p5_dummy)
    torch.cuda.synchronize()

    output_path = project_root / "models/yolo/mamba_head.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (p3_dummy, p4_dummy, p5_dummy),
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["p3", "p4", "p5"],
            output_names=[
                "cls_p3",
                "cls_p4",
                "cls_p5",
                "reg_p3",
                "reg_p4",
                "reg_p5",
            ],
            dynamic_axes={
                "p3": {0: "batch"},
                "p4": {0: "batch"},
                "p5": {0: "batch"},
                "cls_p3": {0: "batch"},
                "cls_p4": {0: "batch"},
                "cls_p5": {0: "batch"},
                "reg_p3": {0: "batch"},
                "reg_p4": {0: "batch"},
                "reg_p5": {0: "batch"},
            },
            dynamo=False,
            verbose=False,
        )

    print(f"ONNX saved: {output_path}")

    print("\nNext — build TRT engine:")
    print(
        f"trtexec --onnx={output_path} \\\n"
        f"  --plugins=build/libsaccade_scan_plugin.so \\\n"
        f"  --saveEngine=models/yolo/mamba_head.engine \\\n"
        f"  --fp16"
    )


if __name__ == "__main__":
    export()
