import cv2
import os
import threading
import time
from typing import Optional, Tuple, Any
import numpy as np

class MediaMTXClient:
    """
    MediaMTX RTSP 或 本地攝像頭影格抓取客戶端
    
    支援背景線程持續抓取，確保低延遲。
    """
    def __init__(self, rtsp_url: str = "rtsp://localhost:8554/live", use_local: bool = False, dummy_video: Optional[str] = None):
        self.rtsp_url = rtsp_url.replace("localhost", "127.0.0.1")
        self.use_local = use_local
        self.dummy_video = dummy_video
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _get_gst_pipeline(self) -> str:
        """構建高效能 GStreamer 管道"""
        return (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! decodebin ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )

    def _update_loop(self):
        """背景持續抓取影格"""
        print(f"🔄 [MediaClient] Background frame reader started.")
        while self._running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                with self._lock:
                    self._ret = ret
                    if ret:
                        self._last_frame = frame
                    else:
                        print("⚠️ [MediaClient] Failed to read frame, stream might be disconnected.")
                        # 如果斷線，這裡不主動重連，交給 grab_frame 判斷
            
            if not self._ret:
                time.sleep(1) # 斷線時稍微休息
            else:
                time.sleep(0.01) # 正常時維持高頻率抓取

    def connect(self) -> bool:
        """建立連線"""
        # 1. 模擬影片優先
        if self.dummy_video and os.path.exists(self.dummy_video):
            self.cap = cv2.VideoCapture(self.dummy_video)
        else:
            # 2. 嘗試 GStreamer -> FFMPEG
            pipeline = self._get_gst_pipeline() if not self.use_local else "v4l2src ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                # 關鍵修正：限制 FFMPEG 執行緒數量並關閉 async 選項以防衝突
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|latency;0|threads;1"
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self.cap.isOpened():
            self._running = True
            # 確保舊線程已結束
            if self._thread and self._thread.is_alive():
                self._running = False
                self._thread.join(timeout=1)
                self._running = True
            
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            return True
        
        return False

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """獲取最新影格"""
        with self._lock:
            if not self._ret and self.cap:
                # 嘗試重新連接
                print("🔄 [MediaClient] Attempting to reconnect...")
                self.cap.release()
                self.connect()
                
            return self._ret, self._last_frame

    def release(self):
        """關閉連線"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    client = MediaMTXClient()
    if client.connect():
        print("Connected successfully!")
        client.release()
