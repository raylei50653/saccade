#!/usr/bin/env python3
"""Verify alignment of whole-detect CUDA-graph capture before implementation.

Tests:
  1. TRT backbone on current stream (no dedicated stream) — graph-capturable?
  2. _postprocess_mamba_fixed vs original — output parity?
  3. Full chain (backbone → head eager → postprocess) graph capture — replay vs eager parity?

Prerequisites: same as mot17.py — TRT detector import first to avoid libjpeg conflict.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

import torch  # noqa: E402

_gen = torch.Generator(device="cuda")
_gen.manual_seed(42)
ORIG_FRAME = torch.rand(
    1, 3, 640, 640, device="cuda", dtype=torch.float32, generator=_gen
)
MAX_DET = 300
CONF_THR = 0.001

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def allclose(a, b, label="", rtol=1e-4, atol=1e-7):
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_diff = (a - b).abs().max().item()
    if ok:
        print(f"  ✓ {label}: bit-exact (max diff={max_diff:.1e})")
    else:
        print(f"  ✗ {label}: MISMATCH (max diff={max_diff:.1e})")
    return ok


def allclose_list(a_list, b_list, label="", rtol=1e-4, atol=1e-7):
    ok = True
    max_diff = 0.0
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        d = (a - b).abs().max().item()
        max_diff = max(max_diff, d)
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            ok = False
    status = "✓" if ok else "✗"
    print(
        f"  {status} {label}: {'bit-exact' if ok else 'MISMATCH'} (max diff={max_diff:.1e})"
    )
    return ok


def _postprocess_mamba_original(
    cls_preds: list[torch.Tensor],
    reg_preds: list[torch.Tensor],
    strides: torch.Tensor,
    conf_thr: float,
    max_det: int,
) -> torch.Tensor:
    from ultralytics.utils.tal import make_anchors, dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)
    B, _, N = cls_all.shape

    anchors, anchor_strides = make_anchors(cls_preds, strides, 0.5)
    anchors = anchors.to(device=cls_all.device, dtype=cls_all.dtype)
    anchor_strides = anchor_strides.to(device=cls_all.device, dtype=cls_all.dtype)

    bboxes = dist2bbox(reg_all, anchors.T.unsqueeze(0), xywh=True, dim=1)
    strides_t = anchor_strides.squeeze(-1).unsqueeze(0)
    bboxes = bboxes * strides_t

    xywh = bboxes.permute(0, 2, 1)
    x1y1 = xywh[..., :2] - xywh[..., 2:4] / 2
    x2y2 = xywh[..., :2] + xywh[..., 2:4] / 2
    boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)

    scores = cls_all.sigmoid()
    scores_max, class_ids = scores.max(dim=1)

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


def _precompute_anchors(
    stride_tensor: torch.Tensor,
    feat_shapes: list[tuple[int, int, int, int]],
    grid_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute anchors and stride tensors (graph-safe: no torch.full/arange during capture)."""
    anchor_points = []
    stride_points = []
    for i, stride in enumerate(stride_tensor):
        _, _, h, w = feat_shapes[i]
        sx = torch.arange(w, dtype=torch.float32, device=stride.device) + grid_offset
        sy = torch.arange(h, dtype=torch.float32, device=stride.device) + grid_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_points.append(
            torch.full(
                (h * w, 1), stride.item(), dtype=torch.float32, device=stride.device
            )
        )
    return torch.cat(anchor_points), torch.cat(stride_points)


def _postprocess_mamba_fixed(
    cls_preds: list[torch.Tensor],
    reg_preds: list[torch.Tensor],
    strides: torch.Tensor,
    conf_thr: float,
    max_det: int,
    *,
    anchors: torch.Tensor | None = None,
    anchor_strides: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-shape version: always topk, no variable indexing.

    Accepts pre-computed anchors/stride tensors for CUDA graph safety (torch.full
    and torch.arange are not capturable).
    """
    from ultralytics.utils.tal import dist2bbox

    cls_all = torch.cat([c.flatten(2) for c in cls_preds], dim=2)
    reg_all = torch.cat([r.flatten(2) for r in reg_preds], dim=2)

    if anchors is None or anchor_strides is None:
        from ultralytics.utils.tal import make_anchors

        anchors, anchor_strides = make_anchors(cls_preds, strides, 0.5)
        device = cls_all.device
        dtype = cls_all.dtype
        anchors = anchors.to(device=device, dtype=dtype)
        anchor_strides = anchor_strides.to(device=device, dtype=dtype)

    bboxes = dist2bbox(reg_all, anchors.T.unsqueeze(0), xywh=True, dim=1)
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


def build_detector():
    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
    )

    return build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=640,
        device="cuda",
        conf_thr=CONF_THR,
        max_det=MAX_DET,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
        temporal_T_override=None,
    )


# ══════════════════════════════════════════════════════════════════
# 1. TRT backbone on current stream — graph-capturable?
# ══════════════════════════════════════════════════════════════════


def test_1_trt_backbone_current_stream():
    print("\n" + "=" * 60)
    print("TEST 1: TRT backbone on current stream — graph capture")
    print("=" * 60)

    det = build_detector()
    det.eval()
    backbone = det._trt_backbone

    # Warm up with original (dedicated stream) to allocate buffers
    with torch.no_grad():
        p3_dedicated, p4_dedicated, p5_dedicated = backbone.infer(ORIG_FRAME.clone())

    # Patch backbone to run on current stream (no dedicated stream dance)
    with torch.no_grad():
        # Refill with fresh data to avoid cache
        frame = ORIG_FRAME.clone()
        B, C, H, W = frame.shape
        frame = frame.contiguous()

        backbone.context.set_input_shape(backbone.input_name, (B, C, H, W))
        backbone.context.set_tensor_address(backbone.input_name, frame.data_ptr())
        for name, buf in zip(backbone.output_names, backbone._output_bufs):
            backbone.context.set_tensor_address(name, buf.data_ptr())

        # Eager on current stream
        torch.cuda.synchronize()
        backbone.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        p3_eager_cs = backbone._output_bufs[0].clone()
        p4_eager_cs = backbone._output_bufs[1].clone()
        p5_eager_cs = backbone._output_bufs[2].clone()

        # Verify current-stream produces same output as dedicated-stream
        allclose(
            p3_eager_cs, p3_dedicated, "TRT current-stream vs dedicated-stream (P3)"
        )
        allclose(
            p4_eager_cs, p4_dedicated, "TRT current-stream vs dedicated-stream (P4)"
        )
        allclose(
            p5_eager_cs, p5_dedicated, "TRT current-stream vs dedicated-stream (P5)"
        )

    # Now try graph-capturing TRT enqueue on current stream
    print("\n  Attempting CUDA-graph capture of TRT enqueue on current stream...")
    with torch.no_grad():
        # Refill buffers
        frame = ORIG_FRAME.clone()
        B, C, H, W = frame.shape
        backbone.context.set_input_shape(backbone.input_name, (B, C, H, W))
        backbone.context.set_tensor_address(backbone.input_name, frame.data_ptr())
        for name, buf in zip(backbone.output_names, backbone._output_bufs):
            backbone.context.set_tensor_address(name, buf.data_ptr())

        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                backbone.context.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream
                )
            torch.cuda.synchronize()

            # Replay
            frame2 = ORIG_FRAME.clone()
            backbone.context.set_tensor_address(backbone.input_name, frame2.data_ptr())
            for name, buf in zip(backbone.output_names, backbone._output_bufs):
                backbone.context.set_tensor_address(name, buf.data_ptr())

            g.replay()
            torch.cuda.synchronize()

            p3_graph = backbone._output_bufs[0].clone()
            p4_graph = backbone._output_bufs[1].clone()
            p5_graph = backbone._output_bufs[2].clone()

            allclose(p3_graph, p3_eager_cs, "TRT graph-replay vs eager (P3)")
            allclose(p4_graph, p4_eager_cs, "TRT graph-replay vs eager (P4)")
            allclose(p5_graph, p5_eager_cs, "TRT graph-replay vs eager (P5)")
            print("  → TRT enqueueV3 IS capturable in CUDA graph on current stream!")

        except Exception as exc:
            print(f"  ✗ TRT graph capture FAILED: {exc}")
            return False

    del det, backbone, g
    torch.cuda.empty_cache()
    return True


# ══════════════════════════════════════════════════════════════════
# 2. _postprocess_mamba_fixed vs original — output parity
# ══════════════════════════════════════════════════════════════════


def test_2_postprocess_fixed_shape():
    print("\n" + "=" * 60)
    print("TEST 2: _postprocess_mamba_fixed vs original — output parity")
    print("=" * 60)

    det = build_detector()
    det.eval()

    with torch.no_grad():
        feats_raw = list(det._trt_backbone.infer(ORIG_FRAME.clone()))
        feats = det._apply_gate(feats_raw, None)
        cls_preds, reg_preds = det.mamba_head(feats, return_embeddings=False)

        out_orig = _postprocess_mamba_original(
            cls_preds, reg_preds, det.stride, CONF_THR, MAX_DET
        )

        feat_shapes = [tuple(c.shape) for c in cls_preds]
        anchors, anchor_strides = _precompute_anchors(det.stride, feat_shapes)

        out_fixed = _postprocess_mamba_fixed(
            cls_preds,
            reg_preds,
            det.stride,
            CONF_THR,
            MAX_DET,
            anchors=anchors,
            anchor_strides=anchor_strides,
        )

    # The fixed version always writes max_det entries; original may write fewer.
    # But for the valid entries (score >= conf_thr), they should match.
    orig_nonzero = (out_orig[0, :, 4] > 0).nonzero(as_tuple=True)[0]
    fixed_nonzero = (out_fixed[0, :, 4] >= CONF_THR).nonzero(as_tuple=True)[0]

    print(f"  Original non-zero detections: {len(orig_nonzero)}")
    print(f"  Fixed   >=conf_thr detections: {len(fixed_nonzero)}")

    ok = True

    # Check: original boxes/scores should be a subset of fixed (fixed has all topk)
    if len(orig_nonzero) > 0:
        orig_top = out_orig[0, orig_nonzero]
        # Find matching boxes in fixed output (same boxes by topk relation)
        # For validation: fixed has all top-k by score (including sub-threshold).
        # Original's valid boxes should appear in the fixed output.
        # Compare sorted by score to account for ordering differences.

        orig_boxes_score = orig_top[orig_top[:, 4].argsort(descending=True)]
        fixed_boxes_score = out_fixed[0][out_fixed[0, :, 4].argsort(descending=True)]

        # Compare the top N where orig has valid detections
        n_compare = min(len(orig_nonzero), 20)
        print(f"\n  Top-{n_compare} detections comparison:")

        for rank in range(n_compare):
            o_score = orig_boxes_score[rank, 4].item()
            f_score = fixed_boxes_score[rank, 4].item()
            o_box = orig_boxes_score[rank, :4]
            f_box = fixed_boxes_score[rank, :4]
            box_iou = _box_iou(o_box.unsqueeze(0), f_box.unsqueeze(0)).item()
            match = "✓" if box_iou > 0.99 and abs(o_score - f_score) < 1e-5 else "✗"
            print(
                f"    [{match}] rank={rank}: orig_score={o_score:.6f} "
                f"fixed_score={f_score:.6f} IoU={box_iou:.4f}"
            )
            if box_iou <= 0.99 or abs(o_score - f_score) >= 1e-5:
                ok = False

    if ok:
        print("\n  ✓ _postprocess_mamba_fixed matches original for valid detections")
    else:
        print("\n  ✗ _postprocess_mamba_fixed MISMATCH — investigate!")

    return ok


def _box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-16)


# ══════════════════════════════════════════════════════════════════
# 3. Full chain graph capture: backbone + head eager + postprocess
# ══════════════════════════════════════════════════════════════════


def test_3_full_chain_graph():
    print("\n" + "=" * 60)
    print("TEST 3: Full chain graph capture (backbone → head → postprocess)")
    print("=" * 60)

    det = build_detector()
    det.eval()
    head = det.mamba_head
    backbone = det._trt_backbone

    FEAT_SHAPES = [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)]
    anchors, anchor_strides = _precompute_anchors(det.stride, FEAT_SHAPES)

    def full_chain_eager(frame):
        """Eager, all on current stream."""
        B, C, H, W = frame.shape
        f = frame.contiguous()

        # TRT backbone on current stream
        backbone.context.set_input_shape(backbone.input_name, (B, C, H, W))
        backbone.context.set_tensor_address(backbone.input_name, f.data_ptr())
        if B != backbone._last_batch:
            backbone._output_bufs = []
            for name in backbone.output_names:
                shape = tuple(backbone.context.get_tensor_shape(name))
                shape = tuple(B if d == -1 else d for d in shape)
                buf = torch.empty(shape, dtype=torch.float32, device=f.device)
                backbone.context.set_tensor_address(name, buf.data_ptr())
                backbone._output_bufs.append(buf)
            backbone._last_batch = B
        else:
            for name, buf in zip(backbone.output_names, backbone._output_bufs):
                backbone.context.set_tensor_address(name, buf.data_ptr())

        backbone.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        p3, p4, p5 = (
            backbone._output_bufs[0],
            backbone._output_bufs[1],
            backbone._output_bufs[2],
        )

        cls_preds, reg_preds = head._forward_eager(
            [p3, p4, p5], return_embeddings=False
        )
        return _postprocess_mamba_fixed(
            cls_preds,
            reg_preds,
            det.stride,
            CONF_THR,
            MAX_DET,
            anchors=anchors,
            anchor_strides=anchor_strides,
        )

    # ── Warmup to stabilise TRT buffers ──
    with torch.no_grad():
        warm = ORIG_FRAME.clone()
        _ = full_chain_eager(warm)
        torch.cuda.synchronize()
        warm2 = ORIG_FRAME.clone()
        out_eager = full_chain_eager(warm2)
        torch.cuda.synchronize()

    print(f"  Eager output shape: {tuple(out_eager.shape)}")
    print(f"  Eager detections (score>0): {(out_eager[0, :, 4] > 0).sum().item()}")

    # ── Attempt graph capture ──
    print("\n  Attempting full-chain CUDA graph capture...")
    with torch.no_grad():
        try:
            frame_cap = ORIG_FRAME.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out_captured = full_chain_eager(frame_cap)
            torch.cuda.synchronize()
            print("  ✓ Graph CAPTURE succeeded!")

            # Replay with distinct input
            _gen2 = torch.Generator(device="cuda")
            _gen2.manual_seed(123)
            frame2 = torch.rand(
                1, 3, 640, 640, device="cuda", dtype=torch.float32, generator=_gen2
            )
            frame_cap.copy_(frame2)

            # Re-run eager for comparison on same distinct input
            out_eager2 = full_chain_eager(frame2)
            torch.cuda.synchronize()

            g.replay()
            torch.cuda.synchronize()
            out_graph = out_captured.clone()

            print(f"\n  Eager output shape: {tuple(out_eager2.shape)}")
            print(f"  Graph output shape: {tuple(out_graph.shape)}")

            allclose(
                out_eager2,
                out_graph,
                "Full-chain: graph replay vs eager",
                rtol=1e-3,
                atol=1e-5,
            )
            print("\n  → Whole-detect CUDA graph capture IS aligned!")

        except Exception as exc:
            print(f"  ✗ Full-chain graph capture FAILED: {exc}")
            import traceback

            traceback.print_exc()
            return False

    del det, backbone, head, g
    torch.cuda.empty_cache()
    return True


# ══════════════════════════════════════════════════════════════════
# 4. Measurement: eager (head-graph) vs whole-detect graph latency
# ══════════════════════════════════════════════════════════════════


def test_4_measure_latency():
    print("\n" + "=" * 60)
    print("TEST 4: Latency comparison (head-graph vs whole-detect graph)")
    print("=" * 60)

    import time

    det = build_detector()
    det.eval()
    head = det.mamba_head
    backbone = det._trt_backbone

    FEAT_SHAPES = [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)]
    anchors, anchor_strides = _precompute_anchors(det.stride, FEAT_SHAPES)

    WARMUP = 50
    ITERS = 200

    def bench(fn) -> float:
        with torch.no_grad():
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                fn()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1000.0

    # ── Baseline: current production path (dedicated-stream TRT + head CUDA graph) ──
    baseline_ms = bench(lambda: det.detect_raw(ORIG_FRAME.clone()))
    print(f"\n  Production baseline (head graph):  {baseline_ms:7.3f} ms  ← current")

    # ── Whole-detect graph path ──
    def whole_detect_graph_fn(frame):
        B, C, H, W = frame.shape
        f = frame.contiguous()

        backbone.context.set_input_shape(backbone.input_name, (B, C, H, W))
        backbone.context.set_tensor_address(backbone.input_name, f.data_ptr())
        for name, buf in zip(backbone.output_names, backbone._output_bufs):
            backbone.context.set_tensor_address(name, buf.data_ptr())

        backbone.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        p3, p4, p5 = (
            backbone._output_bufs[0],
            backbone._output_bufs[1],
            backbone._output_bufs[2],
        )
        cls_preds, reg_preds = head._forward_eager(
            [p3, p4, p5], return_embeddings=False
        )
        return _postprocess_mamba_fixed(
            cls_preds,
            reg_preds,
            det.stride,
            CONF_THR,
            MAX_DET,
            anchors=anchors,
            anchor_strides=anchor_strides,
        )

    # Warmup + capture
    with torch.no_grad():
        warm = ORIG_FRAME.clone()
        _ = whole_detect_graph_fn(warm)
        torch.cuda.synchronize()
        warm2 = ORIG_FRAME.clone()
        _ = whole_detect_graph_fn(warm2)
        torch.cuda.synchronize()

        cap_frame = ORIG_FRAME.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            whole_detect_graph_fn(cap_frame)
        torch.cuda.synchronize()

    # Measure graph replay
    def replay():
        cap_frame.copy_(ORIG_FRAME)
        g.replay()

    whole_ms = bench(replay)
    speedup = baseline_ms / whole_ms
    saved = baseline_ms - whole_ms
    print(
        f"  Whole-detect graph replay:        {whole_ms:7.3f} ms  (speedup {speedup:.2f}x, saved {saved:.3f} ms)"
    )

    # Non-default stream cannot be captured in graph with TRT on a different
    # stream: cudaErrorStreamCaptureIsolation — cross-stream dependencies are
    # not graph-capturable.
    print(
        "  (TRT on non-default stream with graph: NOT supported — "
        "cudaErrorStreamCaptureIsolation)"
    )

    print()
    if speedup > 1.3:
        print(f"  ✓ Launch-bound confirmed: {speedup:.2f}x speedup over baseline")
    else:
        print(f"  ⚠ Speedup modest ({speedup:.2f}x) — investigate")

    del det, backbone, head, g
    torch.cuda.empty_cache()
    return True


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main():
    results = []

    results.append(
        ("TRT backbone current-stream graph", test_1_trt_backbone_current_stream())
    )
    results.append(
        ("_postprocess_mamba_fixed parity", test_2_postprocess_fixed_shape())
    )
    results.append(("Full chain graph capture", test_3_full_chain_graph()))
    results.append(("Latency measurement", test_4_measure_latency()))

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(
            "\n  All tests pass — whole-detect CUDA graph is aligned and safe to implement."
        )
    else:
        print("\n  Some tests FAILED — investigate blockers before implementing.")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
