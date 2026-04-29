import pytest
import torch
import numpy as np
import os
import shutil
import time
from unittest.mock import patch, AsyncMock
from gi.repository import GLib
from media.mediamtx_client import MediaMTXClient
from storage.redis_cache import RedisCache
from storage.chroma_store import ChromaStore
from perception.tracking import SmartTracker, GPUByteTracker
from perception.tracking.tracker_gpu import (
    DynamicReIDController,
    ReIDTrackObservation,
    TrackAppearanceBank,
    need_reid_frame,
)
from perception.zero_copy import GstZeroCopyDecoder

# --- Media Tests ---


def test_media_client_init():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live", use_local=False)
    assert "test" in client.rtsp_url
    assert client.pipeline is None


def test_media_client_grab_frame():
    client = MediaMTXClient(dummy_video="non_existent.mp4")
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_tensor = torch.zeros((100, 100, 3), dtype=torch.float32)
    client._ret = True
    client._last_frame = fake_frame
    client._last_tensor = fake_tensor

    ret_frame, frame = client.grab_frame()
    assert ret_frame is True
    assert np.array_equal(frame, fake_frame)


# --- Storage Tests ---


def test_chroma_store_operations():
    test_db = "./storage/test_pytest_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)
    try:
        store = ChromaStore(path=test_db, collection_name="test_coll")
        mid = store.add_memory(
            content="Person detected.", metadata={"timestamp": time.time()}
        )
        assert mid is not None
        results = store.hybrid_query("Who is there?")
        assert len(results["ids"]) > 0
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db)


# --- Tracking Tests ---


def test_tracker_initialization():
    try:
        tracker = SmartTracker()
        gpu_tracker = GPUByteTracker()
    except RuntimeError as exc:
        message = str(exc).lower()
        if (
            "cuda driver version is insufficient" in message
            or "no cuda-capable device" in message
        ):
            pytest.skip(f"GPU tracker unavailable in test environment: {exc}")
        raise

    assert tracker is not None
    assert gpu_tracker is not None


def test_need_reid_frame_uses_count_heuristic():
    prev_track_ids = {1, 2, 3}

    assert need_reid_frame(prev_track_ids, 0) is False
    assert need_reid_frame(set(), 5) is False
    assert need_reid_frame(prev_track_ids, 3) is False
    assert need_reid_frame(prev_track_ids, 5) is True
    assert need_reid_frame(set(range(8)), 8) is True


def test_track_appearance_bank_caches_consistency_and_prunes():
    bank = TrackAppearanceBank(k=3)
    emb = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    bank.update(7, emb, det_score=0.9, iou=0.9, frame_id=1)
    bank.update(7, emb, det_score=0.8, iou=0.8, frame_id=2)

    rep = bank.representative(7)
    assert rep is not None
    assert bank.has_clean_embedding(7) is True
    assert bank.is_consistent(7) is True
    assert bank.consistency(7) == pytest.approx(1.0)
    assert 7 in bank.clean_ids()

    bank.prune(set())
    assert bank.representative(7) is None
    assert 7 not in bank.clean_ids()


def test_track_appearance_bank_consistency_threshold_is_configurable():
    emb_a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    emb_b = torch.tensor([0.75, (1.0 - 0.75**2) ** 0.5, 0.0], dtype=torch.float32)

    strict_bank = TrackAppearanceBank(k=3, consistency_threshold=0.82)
    relaxed_bank = TrackAppearanceBank(k=3, consistency_threshold=0.75)

    for bank in (strict_bank, relaxed_bank):
        bank.update(9, emb_a, det_score=0.9, iou=0.9, frame_id=1)
        bank.update(9, emb_b, det_score=0.9, iou=0.9, frame_id=2)

    assert strict_bank.consistency(9) == pytest.approx(0.75, abs=1e-4)
    assert strict_bank.is_consistent(9) is False
    assert 9 not in strict_bank.clean_ids()

    assert relaxed_bank.is_consistent(9) is True
    assert 9 in relaxed_bank.clean_ids()


def test_dynamic_reid_controller_uses_recent_bbox_events():
    controller = DynamicReIDController(history_size=5)

    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(2) is False

    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9),
            3: ReIDTrackObservation((40.0, 0.0, 50.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(3) is True

    controller = DynamicReIDController(history_size=5)
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9)})
    controller.observe({1: ReIDTrackObservation((60.0, 0.0, 70.0, 20.0), 0.9)})
    assert controller.should_reid(1) is True


def test_dynamic_reid_controller_persist_mode_is_more_conservative():
    controller = DynamicReIDController(history_size=5, mode="event_persist")
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9)})
    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(2) is True

    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            3: ReIDTrackObservation((40.0, 0.0, 50.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(2) is True


def test_dynamic_reid_controller_count_jump_mode_ignores_motion_only_events():
    controller = DynamicReIDController(history_size=5, mode="count_jump")
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9)})
    controller.observe({1: ReIDTrackObservation((60.0, 0.0, 70.0, 20.0), 0.9)})
    assert controller.should_reid(1) is False


def test_dynamic_reid_controller_event_memory_keeps_past_signal():
    controller = DynamicReIDController(
        history_size=5,
        mode="event_memory",
        long_memory_decay=0.9,
        long_memory_trigger=1.0,
    )
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9)})
    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(2) is True

    controller.observe(
        {
            1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.9),
            2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9),
        }
    )
    assert controller.should_reid(2) is True


def test_dynamic_reid_controller_score_ema_weights_mature_loss():
    controller = DynamicReIDController(
        history_size=5,
        mode="score_ema",
        score_decay=0.8,
        score_threshold=1.4,
        weight_new=1.0,
        weight_lost=1.4,
        weight_geom=0.5,
        weight_conf=0.5,
        birth_death_boost=0.0,
        lost_age_cap=4,
    )
    for _ in range(4):
        controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.95)})
    controller.observe({2: ReIDTrackObservation((20.0, 0.0, 30.0, 20.0), 0.9)})
    assert controller.should_reid(1) is True


def test_dynamic_reid_controller_score_ema_conf_detects_confidence_jitter():
    controller = DynamicReIDController(
        history_size=5,
        mode="score_ema_conf",
        score_decay=0.8,
        score_threshold=0.2,
        weight_new=0.0,
        weight_lost=0.0,
        weight_conf=1.0,
        birth_death_boost=0.0,
        conf_jitter_gate=0.05,
    )
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.95)})
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.40)})
    assert controller.should_reid(1) is True


def test_dynamic_reid_controller_score_ema_respects_persist_frames():
    controller = DynamicReIDController(
        history_size=5,
        mode="score_ema_conf",
        score_decay=0.8,
        score_threshold=0.2,
        weight_new=0.0,
        weight_lost=0.0,
        weight_conf=1.0,
        birth_death_boost=0.0,
        conf_jitter_gate=0.05,
        trigger_persist_frames=2,
    )
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.95)})
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.40)})
    assert controller.should_reid(1) is False
    controller.observe({1: ReIDTrackObservation((0.0, 0.0, 10.0, 20.0), 0.35)})
    assert controller.should_reid(1) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_tracker_update_basic():
    tracker = GPUByteTracker()
    # Set confirm_streak to 1 for easier testing, or run it multiple times.
    # Here we run it multiple times to match default behavior.
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9], dtype=torch.float32, device="cuda")
    classes = torch.tensor([0], dtype=torch.int32, device="cuda")

    # Frame 1: Tentative
    results = tracker.update(boxes, scores, classes)
    assert len(results) == 0

    # Frame 2: Tentative
    results = tracker.update(boxes, scores, classes)
    assert len(results) == 0

    # Frame 3: Confirmed
    results = tracker.update(boxes, scores, classes)
    assert len(results) == 1
    assert results[0].obj_id == 1  # Updated to obj_id as per C++ struct definition


# --- Redis Cache Tests ---


@pytest.mark.anyio
async def test_redis_cache_operations():
    mock_client = AsyncMock()
    # 這裡 patch 的路徑需與 RedisCache 內部使用的 redis 模組一致
    with patch("storage.redis_cache.redis.from_url", return_value=mock_client):
        cache = RedisCache()
        await cache.connect()
        # 設定 Mock 的 get 方法回傳字串
        mock_client.get.return_value = '{"id": 123, "label": "person", "box": [100,100,200,200], "timestamp": 123456}'
        state = await cache.get_object_history(123)
        assert state is not None
        assert state["label"] == "person"


def test_gst_decoder_initialization():
    try:
        decoder = GstZeroCopyDecoder("rtsp://localhost:8554/test")
    except GLib.GError as exc:
        pytest.skip(
            f"GStreamer decoder pipeline unavailable in test environment: {exc}"
        )

    assert decoder.source_url == "rtsp://localhost:8554/test"
    assert decoder.decoder_name in ["nvh264dec", "avdec_h264"]
