# perception/zero_copy.py
# 此檔案記錄 Saccade 的 Zero-Copy 研發進度。

"""
當前感知路徑：OpenCV + GStreamer (nvh264dec)
狀態：穩定生產路徑 (Stable)
效能：1080p @ 30 FPS

研究紀錄：
1. NVIDIA DALI (Experimental): 
   - 優點：真正 Zero-Copy (GPU-to-GPU)，CPU 負載極低。
   - 缺點：單路串流同步開銷較大 (13-17 FPS)，且 RTSP 支援度視版本而定。
   - 適用場景：未來 4+ 路串流高並發處理。

2. OpenCV + NVDEC (Current):
   - 優點：FPS 穩定 (30+)，與現有 YOLO Pipeline 完美整合。
   - 缺點：解碼後需經過一次 CPU 記憶體拷貝回傳給 OpenCV。
"""

class PerceptionPathInfo:
    def __init__(self):
        self.current_path = "OpenCV + NVDEC"
        self.experimental_path = "NVIDIA DALI"
