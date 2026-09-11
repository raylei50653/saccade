"""TAL-free headline decode: import gate, TAL bit-compare, C++ numeric contract."""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    _dist2bbox_xywh,
    _dfl_decode,
    _make_anchor_grid,
    _postprocess_mamba,
    _postprocess_mamba_fixed,
    _postprocess_mamba_fixed_eager,
    _precompute_anchor_grid,
)

TAL_MODULE = "ultralytics.utils.tal"
DETECTOR_SRC = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "saccade"
    / "perception"
    / "temporal_yolo"
    / "mamba_gated_detector.py"
)

# Frozen 640-input P3/P4/P5 layout used by headline native_640 decode.
_FEAT_HW = ((80, 80), (40, 40), (20, 20))


def _purge_tal() -> None:
    stale = [
        name
        for name in sys.modules
        if name == TAL_MODULE or name.startswith(TAL_MODULE + ".")
    ]
    for name in stale:
        del sys.modules[name]


def _frozen_head_outputs(
    *,
    device: torch.device | str = "cpu",
    batch: int = 1,
    num_classes: int = 1,
    seed: int = 393,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cls_preds: list[Tensor] = []
    reg_preds: list[Tensor] = []
    for h, w in _FEAT_HW:
        cls_preds.append(torch.randn(batch, num_classes, h, w, generator=generator))
        reg_preds.append(torch.randn(batch, 4, h, w, generator=generator))
    device_t = torch.device(device)
    cls_preds = [pred.to(device_t) for pred in cls_preds]
    reg_preds = [pred.to(device_t) for pred in reg_preds]
    strides = torch.tensor([8.0, 16.0, 32.0], device=device_t)
    return cls_preds, reg_preds, strides


def _cpp_ltrb_to_xywh(distance: Tensor, anchor_points: Tensor, dim: int = 1) -> Tensor:
    """C++ semantic baseline: c_xy = anchor + (rb - lt) / 2, wh = lt + rb."""
    lt, rb = distance.chunk(2, dim)
    c_xy = anchor_points + (rb - lt) / 2
    h_w = lt + rb
    return torch.cat([c_xy, h_w], dim)


def _tal_postprocess_mamba(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
    small_p3_max_threshold: float = 0.0,
) -> Tensor:
    from ultralytics.utils.tal import dist2bbox, make_anchors

    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        _fuse_small_p3_scores,
    )

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    B, _, _N = cls_all.shape

    anchors, anchor_strides = make_anchors(cls_preds, strides, 0.5)
    anchors = anchors.to(device=cls_all.device, dtype=cls_all.dtype)
    anchor_strides = anchor_strides.to(device=cls_all.device, dtype=cls_all.dtype)

    bboxes = dist2bbox(_dfl_decode(reg_all), anchors.T.unsqueeze(0), xywh=True, dim=1)
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)
    if small_p3_max_threshold > 0:
        scores_max = _fuse_small_p3_scores(
            scores_max, boxes_xyxy, cls_preds, small_p3_max_threshold
        )

    results = boxes_xyxy.new_zeros(B, max_det, 6)
    for b in range(B):
        mask = scores_max[b] >= conf_thr
        s = scores_max[b][mask]
        c = class_ids[b][mask].float()
        bx = boxes_xyxy[b][mask]

        n = min(s.shape[0], max_det)
        if n > 0:
            if s.shape[0] > max_det:
                _, topk = s.topk(max_det)
                s = s[topk]
                c = c[topk]
                bx = bx[topk]
            results[b, :n, :4] = bx[:n]
            results[b, :n, 4] = s[:n]
            results[b, :n, 5] = c[:n]
    return results


def _tal_postprocess_mamba_fixed_eager(
    cls_preds: list[Tensor],
    reg_preds: list[Tensor],
    strides: Tensor,
    conf_thr: float,
    max_det: int,
    *,
    anchors: Tensor | None = None,
    anchor_strides: Tensor | None = None,
) -> Tensor:
    from ultralytics.utils.tal import dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)

    if anchors is None or anchor_strides is None:
        anchors, anchor_strides = _precompute_anchor_grid(
            strides, [tuple(c.shape) for c in cls_preds]
        )

    bboxes = dist2bbox(_dfl_decode(reg_all), anchors.T.unsqueeze(0), xywh=True, dim=1)
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)
    results = boxes_xyxy.new_zeros(cls_all.shape[0], max_det, 6)
    for b in range(cls_all.shape[0]):
        topk_scores, topk_idx = scores_max[b].topk(max_det)
        results[b, :, :4] = boxes_xyxy[b][topk_idx]
        results[b, :, 4] = topk_scores
        results[b, :, 5] = class_ids[b][topk_idx].float()
    return results


def test_mamba_gated_detector_source_has_no_tal_import() -> None:
    tree = ast.parse(DETECTOR_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith(TAL_MODULE)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(TAL_MODULE)


def test_headline_postprocess_does_not_import_ultralytics_tal() -> None:
    _purge_tal()
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    feat_shapes = [tuple(pred.shape) for pred in cls_preds]
    anchors, anchor_strides = _precompute_anchor_grid(strides, feat_shapes)

    _postprocess_mamba(cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300)
    _postprocess_mamba_fixed_eager(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300
    )
    _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    _postprocess_mamba_fixed(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
        _compile=False,
    )

    assert TAL_MODULE not in sys.modules
    assert not any(name.startswith(TAL_MODULE + ".") for name in sys.modules)


def test_make_anchor_grid_matches_tal_bit_exact() -> None:
    tal = pytest.importorskip("ultralytics.utils.tal")
    cls_preds, _reg_preds, strides = _frozen_head_outputs()
    ours, our_strides = _make_anchor_grid(cls_preds, strides, 0.5)
    theirs, their_strides = tal.make_anchors(cls_preds, strides, 0.5)
    assert torch.equal(ours, theirs)
    assert torch.equal(our_strides, their_strides)


def test_dist2bbox_xywh_matches_tal_bit_exact() -> None:
    tal = pytest.importorskip("ultralytics.utils.tal")
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    anchors, _anchor_strides = _make_anchor_grid(cls_preds, strides, 0.5)
    distance = _dfl_decode(torch.cat([r.flatten(2) for r in reg_preds], dim=2))
    anchor_points = anchors.T.unsqueeze(0)
    ours = _dist2bbox_xywh(distance, anchor_points, dim=1)
    theirs = tal.dist2bbox(distance, anchor_points, xywh=True, dim=1)
    assert torch.equal(ours, theirs)


def test_postprocess_mamba_matches_tal_bit_exact() -> None:
    pytest.importorskip("ultralytics.utils.tal")
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    ours = _postprocess_mamba(cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300)
    theirs = _tal_postprocess_mamba(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300
    )
    assert torch.equal(ours, theirs)


def test_postprocess_mamba_fixed_eager_matches_tal_bit_exact() -> None:
    pytest.importorskip("ultralytics.utils.tal")
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    feat_shapes = [tuple(pred.shape) for pred in cls_preds]
    anchors, anchor_strides = _precompute_anchor_grid(strides, feat_shapes)

    ours = _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    theirs = _tal_postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    assert torch.equal(ours, theirs)

    ours_fallback = _postprocess_mamba_fixed_eager(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300
    )
    theirs_fallback = _tal_postprocess_mamba_fixed_eager(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300
    )
    assert torch.equal(ours_fallback, theirs_fallback)


def test_postprocess_mamba_fixed_dispatcher_matches_tal_bit_exact() -> None:
    pytest.importorskip("ultralytics.utils.tal")
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    feat_shapes = [tuple(pred.shape) for pred in cls_preds]
    anchors, anchor_strides = _precompute_anchor_grid(strides, feat_shapes)
    ours = _postprocess_mamba_fixed(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
        _compile=False,
    )
    theirs = _tal_postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    assert torch.equal(ours, theirs)


def test_dist2bbox_xywh_is_numerically_equivalent_to_cpp_formula() -> None:
    cls_preds, reg_preds, strides = _frozen_head_outputs()
    anchors, _anchor_strides = _make_anchor_grid(cls_preds, strides, 0.5)
    distance = _dfl_decode(torch.cat([r.flatten(2) for r in reg_preds], dim=2))
    anchor_points = anchors.T.unsqueeze(0)
    ours = _dist2bbox_xywh(distance, anchor_points, dim=1)
    cpp = _cpp_ltrb_to_xywh(distance, anchor_points, dim=1)
    assert torch.allclose(ours, cpp, rtol=1e-5, atol=1e-5)

    ours64 = _dist2bbox_xywh(distance.double(), anchor_points.double(), dim=1)
    cpp64 = _cpp_ltrb_to_xywh(distance.double(), anchor_points.double(), dim=1)
    assert torch.allclose(ours64, cpp64, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_headline_decode_matches_tal_bit_exact_on_cuda() -> None:
    pytest.importorskip("ultralytics.utils.tal")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    cls_preds, reg_preds, strides = _frozen_head_outputs(device="cuda")
    feat_shapes = [tuple(pred.shape) for pred in cls_preds]
    anchors, anchor_strides = _precompute_anchor_grid(strides, feat_shapes)

    eager = _postprocess_mamba(cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300)
    tal_eager = _tal_postprocess_mamba(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=300
    )
    assert torch.equal(eager, tal_eager)

    fixed = _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    tal_fixed = _tal_postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=300,
        anchors=anchors,
        anchor_strides=anchor_strides,
    )
    assert torch.equal(fixed, tal_fixed)
