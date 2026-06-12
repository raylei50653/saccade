"""Integration tests for the multi-stream batched Mamba detector.

Covers the correctness contract of cross-stream batching:
  * _detect_batch at N=1 is bit-equal to single-stream _detect_from_feats
    (the batch-major head layout matches the single-stream path).
  * Identical inputs across batch slots produce identical output rows
    (per-stream routing / indexing is correct).
  * The MultiStreamMambaServer yields per-stream detections that are invariant
    to concurrent batching, and actually coalesces across streams.

All tests are skipped without CUDA / the C++ extension / model artifacts.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

import torch

# The TorchScript C++ head dispatches selective_scan_fwd back to the
# saccade_tracking_ext CUDA op; load it (RTLD_GLOBAL) before torchvision/PIL to
# avoid a libtiff symbol clash on that lazy import. Restore the dlopen flags
# afterwards so libraries loaded later (e.g. cuBLAS) initialise normally.
_old_dlflags = sys.getdlopenflags()
sys.setdlopenflags(_old_dlflags | ctypes.RTLD_GLOBAL)
try:
    import saccade_tracking_ext  # noqa: F401
except ImportError:
    pass
finally:
    sys.setdlopenflags(_old_dlflags)

import pytest  # noqa: E402

CKPT = Path("runs/mamba_gt_vgt_mamba_v14/best.ckpt")
YOLO = Path("models/yolo/yolo26s.pt")
BACKBONE_B4 = Path("models/yolo/yolo26s_backbone_640_batch4.engine")
try:
    import saccade_perception_ext as _spe  # noqa: F401

    _HAS_EXT = True
except Exception:
    _HAS_EXT = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(not _HAS_EXT, reason="saccade_perception_ext not built"),
    pytest.mark.skipif(
        not (CKPT.exists() and YOLO.exists()), reason="model artifacts missing"
    ),
]


def _make_feats(seed: int, n_frames: int = 4):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        [
            torch.rand(1, 128, 80, 80, generator=g).cuda(),
            torch.rand(1, 256, 40, 40, generator=g).cuda(),
            torch.rand(1, 512, 20, 20, generator=g).cuda(),
        ]
        for _ in range(n_frames)
    ]


@pytest.fixture(scope="module")
def model():
    from saccade.perception.temporal_yolo.mamba_gated_detector import (
        build_mamba_gated_detector,
    )

    return build_mamba_gated_detector(
        yolo_pt_path=str(YOLO),
        teacher_ckpt="",
        mamba_ckpt=str(CKPT),
        img_size=640,
        device="cuda",
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine="",
        use_cuda_graph=False,
    )


def test_detect_batch_n1_bit_equal(model):
    """N=1 _detect_batch must be bit-equal to single-stream _detect_from_feats."""
    from saccade.perception.temporal_yolo.mamba_gated_detector import StreamState

    feats_seq = _make_feats(seed=0)
    st_ref = StreamState.create(model._temporal_T)
    st_batch = StreamState.create(model._temporal_T)
    with torch.no_grad():
        for feats in feats_seq:
            w = torch.eye(2, 3).cuda()
            st_ref.push_gmc(w)
            st_batch.push_gmc(w)
            ref, _ = model._detect_from_feats([f.clone() for f in feats], st_ref)
            out = model._detect_batch([[f.clone() for f in feats]], [st_batch])
            d = out[0][0]
            assert d.shape == ref.shape
            assert torch.equal(d, ref), "N=1 batched diverged from single-stream"


def test_detect_batch_intra_consistency(model):
    """Identical inputs in all batch slots -> identical output rows."""
    from saccade.perception.temporal_yolo.mamba_gated_detector import StreamState

    feats_seq = _make_feats(seed=7)
    states = [StreamState.create(model._temporal_T) for _ in range(4)]
    with torch.no_grad():
        for feats in feats_seq:
            for st in states:
                st.push_gmc(torch.eye(2, 3).cuda())
            outs = model._detect_batch(
                [[f.clone() for f in feats] for _ in range(4)], states
            )
            base = outs[0][0]
            for s in range(1, 4):
                assert torch.equal(outs[s][0], base), (
                    "identical rows differ (indexing bug)"
                )


@pytest.mark.skipif(not BACKBONE_B4.exists(), reason="batch-4 backbone engine missing")
def test_server_concurrent_invariance():
    """A stream's detections are invariant to concurrent batching, and the
    server actually coalesces (batch > 1)."""
    from saccade.perception.multistream_mamba_server import MultiStreamMambaServer

    server = MultiStreamMambaServer(
        str(BACKBONE_B4), str(CKPT), max_batch=4, flush_timeout_s=0.02
    )
    try:
        n_frames = 6
        target = [
            torch.rand(1, 3, 640, 640).cuda().contiguous() for _ in range(n_frames)
        ]

        # Solo reference (batch always 1).
        server.register_stream("solo")
        server.reset_stream("solo")
        ref = []
        for f in target:
            server.set_gmc_warp("solo", torch.eye(2, 3).cuda())
            ref.append(server.submit("solo", f).cpu())

        # Concurrent: target as s0 plus 3 noise streams, submitted in lockstep.
        fps = {"s0": target}
        for i in range(1, 4):
            fps[f"s{i}"] = [
                torch.rand(1, 3, 640, 640).cuda().contiguous() for _ in range(n_frames)
            ]
        for sid in fps:
            server.register_stream(sid)
            server.reset_stream(sid)
        results = {sid: [] for sid in fps}
        barrier = threading.Barrier(len(fps))

        def worker(sid, frames):
            for f in frames:
                barrier.wait()
                server.set_gmc_warp(sid, torch.eye(2, 3).cuda())
                results[sid].append(server.submit(sid, f).cpu())

        threads = [
            threading.Thread(target=worker, args=(s, fr)) for s, fr in fps.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # NOTE: solo runs the backbone at batch-1, concurrent at batch-4 —
        # crossing the TRT batch-size boundary differs at fp level. A real
        # routing/state bug would differ by hundreds of px, so a small
        # tolerance distinguishes correctness from batch-fp noise.
        for a, b in zip(ref, results["s0"]):
            assert a.shape == b.shape, "detection count changed under batching"
            if a.numel():
                assert (a - b).abs().max().item() < 1.0, "stream output diverged"
        assert sum(server.batch_hist[2:]) > 0, "no coalescing occurred"
    finally:
        server.shutdown()
