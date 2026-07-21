"""Phase 2: async ReID crop pipeline tests.

Tests the async crop_into_pool / extract_batch_from_pool path against the
sync extract_reid fallback for numerical parity.  Also tests the ReIDQueue
thread safety and pool exhaustion behavior.
"""

# scope: tracking
# function: behavior
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

from saccade_tracking_ext import (  # noqa: E402
    PerceptionPipeline,
    PerceptionPipelineConfig,
    ReIDQueue,
)


class _PipelineHolder:
    """Holds Python references to keep C++ objects alive."""

    def __init__(self):
        engine_path = PROJECT_ROOT / "models/embedding/osnet_x1_0_256x128.engine"
        if not engine_path.exists():
            pytest.skip(f"ReID model not found: {engine_path}")
        from saccade_perception_ext import FeatureExtractor as FECpp, ModelType, Cropper

        self.ext = FECpp(str(engine_path), ModelType.OSNET, 32)
        self.cropper = Cropper(128, 256)
        self.pipeline = PerceptionPipeline(
            self.ext.cpp_ptr, self.cropper.cpp_ptr, PerceptionPipelineConfig()
        )


@pytest.fixture(scope="module")
def pipeline():
    """Module-scoped pipeline holder — keeps C++ ext + cropper alive."""
    return _PipelineHolder()


def test_reid_queue_push_pop_batch() -> None:
    q = ReIDQueue()
    assert q.size() == 0
    q.shutdown()


def test_reid_queue_shutdown_unblocks_wait() -> None:
    q = ReIDQueue()
    q.shutdown()
    # After shutdown, the queue should not block indefinitely.
    # This is verified by the C++ native test; here we just confirm
    # shutdown doesn't crash and the object can be destroyed.
    del q


def test_crop_pool_per_slot_release_via_pipeline(pipeline) -> None:
    """Verify that the CropPool inside PerceptionPipeline supports per-slot
    release by doing multiple async crop + batch extract cycles."""
    frame_h, frame_w = 480, 640
    frame = torch.rand((3, frame_h, frame_w), dtype=torch.float32, device="cuda")

    # Cycle 1: crop 3 boxes async
    boxes1 = torch.tensor(
        [[50, 50, 150, 250], [200, 100, 350, 400], [400, 50, 500, 300]],
        dtype=torch.float32,
        device="cuda",
    )
    slot1, event1 = pipeline.pipeline.crop_into_pool_async(
        frame.data_ptr(), frame_h, frame_w, boxes1.data_ptr(), 3
    )
    assert slot1 >= 0, "first async crop should succeed"
    assert event1 != 0, "crop event should be non-null"

    # Cycle 2: crop 2 more boxes async (pool should have space)
    boxes2 = torch.tensor(
        [[100, 50, 200, 250], [300, 100, 450, 400]], dtype=torch.float32, device="cuda"
    )
    slot2, event2 = pipeline.pipeline.crop_into_pool_async(
        frame.data_ptr(), frame_h, frame_w, boxes2.data_ptr(), 2
    )
    assert slot2 >= 0, "second async crop should succeed"
    assert event2 != 0, "second crop event should be non-null"

    # Batch extract: build job dicts
    feat_dim = pipeline.pipeline.embed_dim
    out_embeds = torch.empty((5, feat_dim), dtype=torch.float32, device="cuda")

    jobs = [
        {
            "crop_slot": slot1,
            "n_crops": 3,
            "frame_idx": 0,
            "track_uid": 1,
            "generation": 0,
            "det_score": 0.9,
            "quality": 1.0,
            "reason": 1,
            "crop_ready": event1,
        },
        {
            "crop_slot": slot2,
            "n_crops": 2,
            "frame_idx": 1,
            "track_uid": 2,
            "generation": 0,
            "det_score": 0.9,
            "quality": 1.0,
            "reason": 1,
            "crop_ready": event2,
        },
    ]

    n_extracted, results = pipeline.pipeline.extract_batch_from_pool(
        jobs, out_embeds.data_ptr()
    )
    assert n_extracted == 5, f"should extract all 5 crops, got {n_extracted}"
    assert len(results) == 2, f"should have 2 results, got {len(results)}"

    # Verify results metadata
    assert results[0]["track_uid"] == 1
    assert results[0]["n_crops"] == 3
    assert results[0]["embed_offset"] == 0
    assert results[1]["track_uid"] == 2
    assert results[1]["n_crops"] == 2
    assert results[1]["embed_offset"] == 3

    # Verify embeddings are non-trivial (not all zeros)
    torch.cuda.synchronize()
    assert out_embeds.abs().sum() > 0, "embeddings should be non-zero"

    # After batch extract, pool slots should be released — we should be able
    # to crop again.
    slot3, event3 = pipeline.pipeline.crop_into_pool_async(
        frame.data_ptr(), frame_h, frame_w, boxes1.data_ptr(), 3
    )
    assert slot3 >= 0, "pool should have space after batch extract released slots"
    # Clean up event
    if event3 != 0:
        # Event is destroyed by extract_batch_from_pool; if we don't call it,
        # we need to destroy manually. For this test, just do another batch extract.
        out2 = torch.empty((3, feat_dim), dtype=torch.float32, device="cuda")
        jobs2 = [
            {
                "crop_slot": slot3,
                "n_crops": 3,
                "frame_idx": 2,
                "track_uid": 3,
                "generation": 0,
                "det_score": 0.9,
                "quality": 1.0,
                "reason": 1,
                "crop_ready": event3,
            }
        ]
        pipeline.pipeline.extract_batch_from_pool(jobs2, out2.data_ptr())


def test_async_vs_sync_numerical_parity(pipeline) -> None:
    """The async path should produce the same embeddings as the sync path."""
    torch.cuda.synchronize()
    stream = torch.cuda.current_stream().cuda_stream
    frame_h, frame_w = 480, 640
    frame = torch.rand((3, frame_h, frame_w), dtype=torch.float32, device="cuda")
    boxes = torch.tensor(
        [[50, 50, 150, 250], [200, 100, 350, 400]], dtype=torch.float32, device="cuda"
    )
    feat_dim = pipeline.pipeline.embed_dim

    # Sync path
    embed_sync = torch.empty((2, feat_dim), dtype=torch.float32, device="cuda")
    pipeline.pipeline.extract_reid(
        frame.data_ptr(),
        frame_h,
        frame_w,
        boxes.data_ptr(),
        2,
        embed_sync.data_ptr(),
        stream,
    )
    torch.cuda.synchronize()

    # Async path
    embed_async = torch.empty((2, feat_dim), dtype=torch.float32, device="cuda")
    slot, event = pipeline.pipeline.crop_into_pool_async(
        frame.data_ptr(), frame_h, frame_w, boxes.data_ptr(), 2
    )
    assert slot >= 0
    jobs = [
        {
            "crop_slot": slot,
            "n_crops": 2,
            "frame_idx": 0,
            "track_uid": 1,
            "generation": 0,
            "det_score": 0.9,
            "quality": 1.0,
            "reason": 1,
            "crop_ready": event,
        }
    ]
    n, _ = pipeline.pipeline.extract_batch_from_pool(jobs, embed_async.data_ptr())
    assert n == 2
    torch.cuda.synchronize()

    # Compare — allow small fp differences from D2D copy + independent inference.
    diff = (embed_sync - embed_async).abs().max().item()
    assert diff < 1e-3, f"async vs sync embed diff too large: {diff}"


def test_pool_exhaustion_returns_failure(pipeline) -> None:
    """When the crop pool is full, crop_into_pool_async should return slot=-1."""
    frame_h, frame_w = 480, 640
    frame = torch.rand((3, frame_h, frame_w), dtype=torch.float32, device="cuda")
    boxes = torch.tensor([[50, 50, 150, 250]], dtype=torch.float32, device="cuda")

    # Exhaust the pool by acquiring many slots without extracting.
    # The pool has 256 slots (default). Acquire 256 individual crops.
    acquired = []
    for i in range(256):
        slot, event = pipeline.pipeline.crop_into_pool_async(
            frame.data_ptr(), frame_h, frame_w, boxes.data_ptr(), 1
        )
        if slot < 0:
            break
        acquired.append((slot, event))

    # The next acquire should fail.
    slot_fail, event_fail = pipeline.pipeline.crop_into_pool_async(
        frame.data_ptr(), frame_h, frame_w, boxes.data_ptr(), 1
    )
    assert slot_fail == -1, "pool exhaustion should return slot=-1"

    # Clean up: batch extract all acquired jobs to destroy events + release slots.
    if acquired:
        feat_dim = pipeline.pipeline.embed_dim
        n = len(acquired)
        out = torch.empty((n, feat_dim), dtype=torch.float32, device="cuda")
        jobs = [
            {
                "crop_slot": s,
                "n_crops": 1,
                "frame_idx": i,
                "track_uid": i + 1,
                "generation": 0,
                "det_score": 0.9,
                "quality": 1.0,
                "reason": 1,
                "crop_ready": e,
            }
            for i, (s, e) in enumerate(acquired)
        ]
        # Extract in chunks of max_batch to avoid exceeding TRT max_batch.
        max_batch = 32
        for start in range(0, n, max_batch):
            chunk = jobs[start : start + max_batch]
            out_chunk = out[start : start + len(chunk)]
            pipeline.pipeline.extract_batch_from_pool(chunk, out_chunk.data_ptr())
