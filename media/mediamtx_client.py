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
                
                # 針對本地檔案的循環播放邏輯
                if not ret and self.dummy_video:
                    print("🔄 [MediaClient] Video end reached, looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                with self._lock:
                    self._ret = ret
                    if ret:
                        self._last_frame = frame
                    else:
                        if not self.dummy_video: # 只有串流才報錯
                            print("⚠️ [MediaClient] Failed to read frame, stream might be disconnected.")
            
            if not self._ret and not self.dummy_video:
                time.sleep(1) # 串流斷線時稍微休息
            else:
                time.sleep(0.01) # 正常時維持高頻率抓取

    def connect(self) -> bool:
        """建立連線"""
        # 關鍵修正：環境變數必須在所有 cv2 調用前設定，並包含更多限制
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|latency;0|threads;1"
        os.environ["OPENCV_FFMPEG_THREADS"] = "1"
        
        # 確保舊線程已徹底結束並釋放資源
        if self._thread and self._thread.is_alive():
            self._running = False
            self._thread.join(timeout=2)
            
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        # 1. 模擬影片優先 (移除強制 CAP_FFMPEG，讓 OpenCV 自動選擇最穩定的後端)
        if self.dummy_video and os.path.exists(self.dummy_video):
            self.cap = cv2.VideoCapture(self.dummy_video)
        else:
            # 2. 嘗試 GStreamer -> FFMPEG
            pipeline = self._get_gst_pipeline() if not self.use_local else "v4l2src ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if self.cap.isOpened():
            self._running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            return True
        
        return False

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """獲取最新影格"""
        with self._lock:
            # 只有在「非本地影片」且「確定斷線」時才嘗試重連
            if not self._ret and self.cap and not self.dummy_video:
                print("🔄 [MediaClient] Stream seems down, attempting to reconnect...")
                # 避免頻繁重連，這裡可以增加時間戳判斷，但目前先簡單處理
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
