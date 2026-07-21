"""Cross-package unit tests for core media/storage/tracking helpers."""

# scope: system
# function: behavior
# lifecycle: active

import pytest  # noqa: E402
import torch
import numpy as np
import os
import shutil
import time
from unittest.mock import patch, AsyncMock, MagicMock
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402
from saccade.media.mediamtx_client import MediaMTXClient  # noqa: E402
from saccade.media.rtsp import (  # noqa: E402
    DEFAULT_RTSP_HOST,
    DEFAULT_RTSP_PORT,
    DEFAULT_RTSP_READ_PASSWORD,
    DEFAULT_RTSP_READ_USER,
    DEFAULT_RTSP_STREAM_PREFIX,
    RTSPEndpoint,
    build_reader_url,
    build_rtsp_url,
    build_stream_path,
)
from saccade.storage.redis_cache import RedisCache  # noqa: E402
from saccade.storage.chroma_store import ChromaStore  # noqa: E402
from saccade.perception.tracking import GPUByteTracker  # noqa: E402
from saccade.perception.tracking.dynamic_reid import DynamicReIDController  # noqa: E402
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    ReIDTrackObservation,
    TrackAppearanceBank,
    need_reid_frame,
)
from saccade.perception.zero_copy import GstZeroCopyDecoder  # noqa: E402

# --- Media Tests ---


def test_rtsp_helper_builds_reader_url_with_defaults():
    assert build_reader_url("live") == (
        f"rtsp://{DEFAULT_RTSP_READ_USER}:{DEFAULT_RTSP_READ_PASSWORD}"
        f"@{DEFAULT_RTSP_HOST}:{DEFAULT_RTSP_PORT}/live"
    )


def test_rtsp_helper_builds_stream_path():
    assert build_stream_path(7) == f"{DEFAULT_RTSP_STREAM_PREFIX}7"


def test_rtsp_endpoint_round_trips():
    url = build_rtsp_url(
        "stream_3",
        host="10.0.0.8",
        port=9554,
        username="reader",
        password="secret",
    )
    endpoint = RTSPEndpoint.from_url(url)

    assert endpoint.host == "10.0.0.8"
    assert endpoint.port == 9554
    assert endpoint.path == "stream_3"
    assert endpoint.username == "reader"
    assert endpoint.password == "secret"
    assert endpoint.url() == url


def test_media_client_default_rtsp_url_is_canonical():
    client = MediaMTXClient()
    assert client.rtsp_url == build_rtsp_url("live")


def test_media_client_init():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live", use_local=False)
    assert "test" in client.rtsp_url
    assert client.pipeline is None


def test_media_client_rtsp_pipeline_prefers_tcp_and_cpu_decode():
    client = MediaMTXClient(rtsp_url="rtsp://reader:pass@test:8554/live")
    pipeline = client._get_pipeline_str()

    assert "protocols=tcp" in pipeline
    assert "avdec_h264" in pipeline
    assert "video/x-raw,format=RGB" in pipeline
    assert "sync=false" in pipeline
    assert "nvh264dec" not in pipeline


def test_media_client_cpp_path_is_opt_in_for_rtsp():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live", use_local=False)
    assert client.use_cpp is False


def test_media_client_waits_for_first_frame_success():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live")
    client._ret = True
    client._last_tensor = torch.zeros((1, 1, 3), dtype=torch.uint8)
    assert client._await_first_frame(timeout_sec=0.01) is True


def test_media_client_waits_for_first_frame_fails_on_bus_error():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live")
    client._bus_error = "Not found"
    assert client._await_first_frame(timeout_sec=0.01) is False


def test_media_client_grab_frame():
    client = MediaMTXClient(dummy_video="non_existent.mp4")
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_tensor = torch.zeros((100, 100, 3), dtype=torch.float32)
    client._ret = True
    client._last_frame = fake_frame
    client._last_tensor = fake_tensor
    client._frame_generation = 1

    ret_frame, frame = client.grab_frame()
    assert ret_frame is True
    assert np.array_equal(frame, fake_frame)


def test_media_client_grab_tensor_only_returns_new_frame_once():
    client = MediaMTXClient(dummy_video="non_existent.mp4")
    fake_tensor = torch.zeros((100, 100, 3), dtype=torch.float32)
    client._ret = True
    client._last_tensor = fake_tensor
    client._frame_generation = 1

    first_ret, first_tensor = client.grab_tensor()
    second_ret, second_tensor = client.grab_tensor()

    assert first_ret is True
    assert first_tensor is fake_tensor
    assert second_ret is False
    assert second_tensor is None


# --- Storage Tests ---


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="ChromaDB default embedding function loads onnxruntime which SIGABRTs in CI",
)
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


@pytest.mark.gpu
def test_tracker_initialization():
    try:
        gpu_tracker = GPUByteTracker()
    except RuntimeError as exc:
        message = str(exc).lower()
        if (
            "cuda driver version is insufficient" in message
            or "no cuda-capable device" in message
        ):
            pytest.skip(f"GPU tracker unavailable in test environment: {exc}")
        raise

    assert gpu_tracker is not None


def test_need_reid_frame_uses_count_heuristic():
    prev_track_ids = {1, 2, 3}

    assert need_reid_frame(prev_track_ids, 0) is False
    assert need_reid_frame(set(), 5) is False
    assert need_reid_frame(prev_track_ids, 3) is False
    assert need_reid_frame(prev_track_ids, 5) is True
    assert need_reid_frame(set(range(8)), 8) is True


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_tracker_update_basic():
    tracker = GPUByteTracker()
    tracker.set_params(confirm_streak=3)
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9], dtype=torch.float32, device="cuda")
    classes = torch.tensor([0], dtype=torch.int32, device="cuda")

    for _f in range(10):
        results = tracker.update(boxes, scores, classes)

    assert isinstance(results, list)


# --- Redis Cache Tests ---


@pytest.mark.anyio
async def test_redis_cache_operations():
    mock_client = AsyncMock()
    # 這裡 patch 的路徑需與 RedisCache 內部使用的 redis 模組一致
    with patch("saccade.storage.redis_cache.redis.from_url", return_value=mock_client):
        cache = RedisCache()
        await cache.connect()
        # 設定 Mock 的 get 方法回傳字串
        mock_client.get.return_value = '{"id": 123, "label": "person", "box": [100,100,200,200], "timestamp": 123456}'
        state = await cache.get_object_history(123)
        assert state is not None
        assert state["label"] == "person"


def test_gst_decoder_initialization():
    source_url = build_rtsp_url("test")
    try:
        decoder = GstZeroCopyDecoder(source_url)
    except GLib.GError as exc:
        pytest.skip(
            f"GStreamer decoder pipeline unavailable in test environment: {exc}"
        )

    assert decoder.source_url == source_url
    assert decoder.decoder_name in ["nvh264dec", "avdec_h264"]


def test_gst_decoder_rtsp_pipeline_prefers_tcp_and_cpu_decode():
    decoder = GstZeroCopyDecoder.__new__(GstZeroCopyDecoder)
    decoder.source_url = build_rtsp_url("test")
    decoder.decoder_name = "nvh264dec"

    pipeline = decoder._build_pipeline_str()

    assert "protocols=tcp" in pipeline
    assert "avdec_h264" in pipeline
    assert "video/x-raw,format=RGB" in pipeline
    assert "sync=false" in pipeline
    assert "nvh264dec" not in pipeline


def test_gst_decoder_rtsp_pipeline_quotes_special_url_characters():
    decoder = GstZeroCopyDecoder.__new__(GstZeroCopyDecoder)
    decoder.source_url = "rtsp://reader:pa!ss word@127.0.0.1:8554/live"
    decoder.decoder_name = "nvh264dec"

    pipeline = decoder._build_pipeline_str()

    assert f"location='{decoder.source_url}'" in pipeline
    assert "protocols=tcp" in pipeline


def test_gst_decoder_file_pipeline_quotes_paths_with_spaces():
    decoder = GstZeroCopyDecoder.__new__(GstZeroCopyDecoder)
    decoder.source_url = "file:///tmp/sample clip!.mp4"
    decoder.decoder_name = "avdec_h264"

    pipeline = decoder._build_pipeline_str()

    assert "filesrc location='/tmp/sample clip!.mp4'" in pipeline
    assert "qtdemux" in pipeline


def test_gst_decoder_on_new_sample_without_caps_returns_error():
    decoder = GstZeroCopyDecoder.__new__(GstZeroCopyDecoder)

    class _Sample:
        def get_buffer(self):
            return object()

        def get_caps(self):
            return None

    class _Sink:
        def emit(self, name):
            assert name == "pull-sample"
            return _Sample()

    assert decoder._on_new_sample(_Sink()) == Gst.FlowReturn.ERROR


def test_gst_decoder_start_sets_pipeline_playing():
    decoder = GstZeroCopyDecoder.__new__(GstZeroCopyDecoder)
    decoder._nv12_buffer = False
    decoder._running = False
    decoder.source_url = build_rtsp_url("test")
    decoder.pipeline = MagicMock()

    decoder.start()

    assert decoder._running is True
    decoder.pipeline.set_state.assert_called_once()
