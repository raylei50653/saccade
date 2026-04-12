import numpy as np
import os
from unittest.mock import MagicMock, patch
from media.mediamtx_client import MediaMTXClient

def test_media_client_init():
    client = MediaMTXClient(rtsp_url="rtsp://test:8554/live", use_local=False)
    assert "test" in client.rtsp_url
    assert client.use_local is False
    assert client.cap is None

@patch("media.mediamtx_client.cv2.VideoCapture")
def test_media_client_connect_dummy_video(mock_video_capture):
    # Mock VideoCapture instance
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap
    
    # Create dummy file
    dummy_path = "test_dummy.mp4"
    with open(dummy_path, "w") as f:
        f.write("fake video data")
    
    client = MediaMTXClient(dummy_video=dummy_path)
    try:
        success = client.connect()
        assert success is True
        assert client._running is True
        assert client._thread is not None
        assert client._thread.is_alive()
        
        # Cleanup
        client.release()
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

@patch("media.mediamtx_client.cv2.VideoCapture")
def test_media_client_grab_frame(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Simulate first frame
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    mock_video_capture.return_value = mock_cap
    
    client = MediaMTXClient(dummy_video="non_existent.mp4")
    # Manually set cap to skip connect logic if needed, but let's try connect
    with patch("os.path.exists", return_value=True):
        client.connect()
        
    # Give some time for background thread to run or manually call _update_loop once
    # For testing, we can just manually set _last_frame and _ret
    client._ret = True
    client._last_frame = fake_frame
    
    ret, frame = client.grab_frame()
    assert ret is True
    assert np.array_equal(frame, fake_frame)
    
    client.release()
