"""TRT construction skips Ultralytics YOLO(); import-gate and fail-closed fallbacks."""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    MambaGatedDetector,
    _postprocess_mamba_fixed,
    _postprocess_mamba_fixed_eager,
    _precompute_anchor_grid,
    build_mamba_gated_detector,
)
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead
from saccade.perception.temporal_yolo.yolo_conditioned import TrackSpatialGate

ULTRALYTICS = "ultralytics"

_TINY_CHANNELS = (8, 16, 32)
_TINY_ARGS = {
    "d_model": 8,
    "d_state": 4,
    "num_blocks": 1,
    "num_classes": 1,
    "spatial_reduction": 4,
    "in_channels": list(_TINY_CHANNELS),
    "reg_max": 1,
    "head_depth": 1,
}

HEADLINE_YOLO_PT = Path("models/yolo/yolo26s.pt")
HEADLINE_ENGINE = Path("models/yolo/yolo26s_backbone_640_best.engine")
HEADLINE_CKPT = Path("runs/mamba_gt_v14replica_t3_t1/best.ckpt")


class _StubTRTBackbone(nn.Module):
    """Stand-in for TRTYoloBackbone so the TRT branch runs without TensorRT."""

    def __init__(self, engine_path: str):
        super().__init__()
        self.engine_path = engine_path
        self.output_channels = _TINY_CHANNELS

    def infer(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = images.shape[0]
        device = images.device
        dtype = images.dtype
        return (
            torch.zeros(batch, 8, 80, 80, device=device, dtype=dtype),
            torch.zeros(batch, 16, 40, 40, device=device, dtype=dtype),
            torch.zeros(batch, 32, 20, 20, device=device, dtype=dtype),
        )


def _purge_ultralytics() -> None:
    stale = [
        name
        for name in sys.modules
        if name == ULTRALYTICS or name.startswith(ULTRALYTICS + ".")
    ]
    for name in stale:
        del sys.modules[name]


def _assert_no_ultralytics() -> None:
    assert ULTRALYTICS not in sys.modules
    assert not any(name.startswith(ULTRALYTICS + ".") for name in sys.modules)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tiny_ckpt(path: Path, *, yolo_sha: str | None = None) -> None:
    head = MambaDetectionHead(
        in_channels=_TINY_CHANNELS,
        d_model=int(_TINY_ARGS["d_model"]),
        d_state=int(_TINY_ARGS["d_state"]),
        num_blocks=int(_TINY_ARGS["num_blocks"]),
        num_classes=int(_TINY_ARGS["num_classes"]),
        reg_max=int(_TINY_ARGS["reg_max"]),
        head_depth=int(_TINY_ARGS["head_depth"]),
        spatial_reduction=int(_TINY_ARGS["spatial_reduction"]),
        emb_dim=0,
    )
    mamba_args = dict(_TINY_ARGS)
    if yolo_sha is not None:
        mamba_args["base_yolo_sha256"] = yolo_sha
    torch.save({"student": head.state_dict(), "mamba_args": mamba_args}, path)


def _build_tiny_trt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trt_backbone_engine: str | None = "stub.engine",
    yolo_payload: bytes = b"not-a-real-yolo-checkpoint",
    yolo_sha: str | None = "auto",
) -> MambaGatedDetector:
    monkeypatch.setattr(
        "saccade.perception.temporal_yolo.mamba_gated_detector.TRTYoloBackbone",
        _StubTRTBackbone,
    )
    yolo_pt = tmp_path / "yolo26s.pt"
    yolo_pt.write_bytes(yolo_payload)
    ckpt = tmp_path / "mamba.ckpt"
    if yolo_sha is None:
        sha = None
    elif yolo_sha == "auto":
        sha = _sha256_bytes(yolo_payload)
    else:
        sha = yolo_sha
    _write_tiny_ckpt(ckpt, yolo_sha=sha)
    engine = "" if trt_backbone_engine is None else str(tmp_path / trt_backbone_engine)
    return build_mamba_gated_detector(
        yolo_pt_path=str(yolo_pt),
        teacher_ckpt="",
        mamba_ckpt=str(ckpt),
        img_size=640,
        device="cpu",
        conf_thr=0.0,
        max_det=16,
        trt_backbone_engine=engine,
        use_cuda_graph=False,
        use_whole_graph=False,
    )


def test_trt_construction_does_not_import_ultralytics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _purge_ultralytics()
    monkeypatch.setattr(
        "saccade.perception.temporal_yolo.mamba_gated_detector.build_gated_yolo_detector",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("YOLO teacher must not be built on the TRT path")
        ),
    )
    det = _build_tiny_trt(tmp_path, monkeypatch)

    assert det.teacher is None
    assert isinstance(det.gate_module, TrackSpatialGate)
    _assert_no_ultralytics()

    out = det.detect_raw(torch.zeros(1, 3, 640, 640))
    assert out.shape == (1, 16, 6)
    _assert_no_ultralytics()

    cls_preds = [
        torch.zeros(1, 1, 80, 80),
        torch.zeros(1, 1, 40, 40),
        torch.zeros(1, 1, 20, 20),
    ]
    reg_preds = [
        torch.zeros(1, 4, 80, 80),
        torch.zeros(1, 4, 40, 40),
        torch.zeros(1, 4, 20, 20),
    ]
    strides = torch.tensor([8.0, 16.0, 32.0])
    feat_shapes = [tuple(pred.shape) for pred in cls_preds]
    anchors, anchor_strides = _precompute_anchor_grid(strides, feat_shapes)
    _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=16,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    _postprocess_mamba_fixed(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=16,
        anchors=anchors,
        anchor_strides=anchor_strides,
        _compile=False,
    )
    _assert_no_ultralytics()


def test_trt_path_still_checks_yolo_pt_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "saccade.perception.temporal_yolo.mamba_gated_detector.TRTYoloBackbone",
        _StubTRTBackbone,
    )
    yolo_pt = tmp_path / "yolo26s.pt"
    yolo_pt.write_bytes(b"not-a-real-yolo-checkpoint")
    ckpt = tmp_path / "mamba.ckpt"
    _write_tiny_ckpt(ckpt, yolo_sha="0" * 64)
    with pytest.raises(ValueError, match="Base YOLO weights do not match"):
        build_mamba_gated_detector(
            yolo_pt_path=str(yolo_pt),
            teacher_ckpt="",
            mamba_ckpt=str(ckpt),
            img_size=640,
            device="cpu",
            trt_backbone_engine=str(tmp_path / "stub.engine"),
        )


def test_trt_path_does_not_unpickle_yolo_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dummy .pt bytes are not a pickle; construction must not torch.load them."""
    det = _build_tiny_trt(tmp_path, monkeypatch)
    assert det.teacher is None


def test_pytorch_backbone_fail_closed_without_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    det = _build_tiny_trt(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="did not build a teacher"):
        det._forward_pytorch_backbone(torch.zeros(1, 3, 64, 64))


def test_fpn_embedding_fallback_fail_closed_without_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    det = _build_tiny_trt(tmp_path, monkeypatch)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    with pytest.raises(RuntimeError, match="did not build a teacher"):
        det.extract_fpn_embeddings(torch.zeros(1, 3, 64, 64), boxes)


def test_no_trt_path_still_builds_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _fake_build(
        yolo_pt_path: str,
        cfg: object = None,
        device: str = "cuda",
        weights_path: str = "",
    ) -> nn.Module:
        calls.append(yolo_pt_path)
        teacher = nn.Module()
        teacher.gate = TrackSpatialGate(scales=("p3", "p4", "p5"))
        teacher._gate_layers = {}
        return teacher

    monkeypatch.setattr(
        "saccade.perception.temporal_yolo.mamba_gated_detector.build_gated_yolo_detector",
        _fake_build,
    )
    det = _build_tiny_trt(tmp_path, monkeypatch, trt_backbone_engine=None)
    assert calls
    assert det.teacher is not None
    assert det.gate_module is det.teacher.gate


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    not (
        HEADLINE_YOLO_PT.exists()
        and HEADLINE_ENGINE.exists()
        and HEADLINE_CKPT.exists()
    ),
    reason="headline artifacts missing",
)
def test_headline_trt_build_does_not_import_ultralytics() -> None:
    _purge_ultralytics()
    det = build_mamba_gated_detector(
        yolo_pt_path=str(HEADLINE_YOLO_PT),
        teacher_ckpt="",
        mamba_ckpt=str(HEADLINE_CKPT),
        img_size=640,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine=str(HEADLINE_ENGINE),
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    assert det.teacher is None
    _assert_no_ultralytics()
    out = det.detect_raw(torch.zeros(1, 3, 640, 640, device="cuda"))
    assert out.shape[0] == 1 and out.shape[-1] == 6
    _assert_no_ultralytics()
    del det
    torch.cuda.empty_cache()
