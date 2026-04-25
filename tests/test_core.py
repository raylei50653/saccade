import pytest
import torch
import numpy as np
import os
import shutil
import time
from unittest.mock import MagicMock, patch, AsyncMock
from media.mediamtx_client import MediaMTXClient
from storage.redis_cache import RedisCache
from storage.chroma_store import ChromaStore
from perception.tracking import SmartTracker, GPUByteTracker
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

@pytest.mark.anyio
async def test_redis_cache_operations():
    mock_client = AsyncMock()
    with patch("redis.asyncio.from_url", return_value=mock_client):
        cache = RedisCache()
        await cache.connect()
        mock_client.get.return_value = '{"id": 123, "label": "person"}'
        state = await cache.get_object_history(123)
        assert state["label"] == "person"

def test_chroma_store_operations():
    test_db = "./storage/test_pytest_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)
    try:
        store = ChromaStore(path=test_db, collection_name="test_coll")
        mid = store.add_memory(content="Person detected.", metadata={"timestamp": time.time()})
        assert mid is not None
        results = store.hybrid_query("Who is there?")
        assert len(results["ids"]) > 0
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db)

# --- Tracking Tests ---

def test_tracker_initialization():
    tracker = SmartTracker()
    assert tracker is not None
    
    # 測試 GPUByteTracker 初始化
    gpu_tracker = GPUByteTracker()
    assert gpu_tracker is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_tracker_update_basic():
    tracker = GPUByteTracker()
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9], dtype=torch.float32, device="cuda")
    classes = torch.tensor([0], dtype=torch.int32, device="cuda")
    
    results = tracker.update(boxes, scores, classes)
    assert len(results) == 1
    assert results[0].obj_id == 1

# --- Zero-Copy Tests ---

def test_gst_decoder_initialization():
    decoder = GstZeroCopyDecoder("rtsp://localhost:8554/test")
    assert decoder.source_url == "rtsp://localhost:8554/test"
    assert decoder.decoder_name in ["nvh264dec", "avdec_h264"]
