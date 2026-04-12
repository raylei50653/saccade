import numpy as np
import os
import torch
from unittest.mock import MagicMock, patch
from media.mediamtx_client import MediaMTXClient

def test_media_client_init():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live", use_local=False)
    assert "test" in client.rtsp_url
    assert client.use_local is False
    assert client.pipeline is None

@patch("media.mediamtx_client.Gst.parse_launch")
@patch("media.mediamtx_client.threading.Thread")
def test_media_client_connect_dummy_video(mock_thread, mock_parse_launch):
    mock_pipeline = MagicMock()
    mock_sink = MagicMock()
    mock_pipeline.get_by_name.return_value = mock_sink
    mock_parse_launch.return_value = mock_pipeline
    
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance
    
    dummy_path = "test_dummy.mp4"
    with open(dummy_path, "w") as f:
        f.write("fake video data")
    
    client = MediaMTXClient(dummy_video=dummy_path)
    try:
        success = client.connect()
        assert success is True
        assert client._running is True
        assert client._loop_thread is not None
        
        client.release()
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

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
    
    ret_tensor, tensor = client.grab_tensor()
    assert ret_tensor is True
    assert torch.equal(tensor, fake_tensor)
