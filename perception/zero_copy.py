import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import torch
import numpy as np
from typing import Optional, Tuple, Callable
import threading
import os

# 初始化 GStreamer
Gst.init(None)

class GstZeroCopyDecoder:
    """
    Saccade GStreamer 零拷貝解碼器 (Pillar 4)
    
    支援 RTSP 與本地檔案，強制使用 NVDEC (nvh264dec) 並嘗試保留在 CUDA 記憶體中。
    """
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.pipeline_str = self._build_pipeline_str()
        
        print(f"🛠️  Generated Pipeline: {self.pipeline_str}")
        
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_sample)
        
        # 錯誤監聽
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)
        
        self.last_tensor: Optional[torch.Tensor] = None
        self._lock = threading.Lock()
        self._running = False

    def _build_pipeline_str(self) -> str:
        """根據輸入源自動構建管線"""
        # 基礎處理路徑 (解碼 -> CUDA 轉換 -> RGB)
        decoder_path = "nvh264dec ! cudaconvert ! video/x-raw(memory:CUDAMemory),format=RGB"
        sink_path = "appsink name=sink emit-signals=true max-buffers=1 drop=true"

        if self.source_url.startswith("rtsp://"):
            return (
                f"rtspsrc location={self.source_url} latency=0 ! "
                f"rtph264depay ! h264parse ! {decoder_path} ! {sink_path}"
            )
        else:
            # 處理本地檔案 (支援 file:// 或 絕對路徑)
            path = self.source_url.replace("file://", "")
            return (
                f"filesrc location={path} ! qtdemux ! h264parse ! "
                f"{decoder_path} ! {sink_path}"
            )

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ GStreamer Error: {err.message}")
            if debug: print(f"🔍 Debug Info: {debug}")
        elif t == Gst.MessageType.EOS:
            print("🏁 GStreamer: End of stream")

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        # 處理記憶體對齊 (Stride/Pitch)
        # 1080p 下 NVDEC 可能會補齊到 2048 像素寬 (stride = 6144 bytes)
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                # 計算實際步長
                actual_size = len(map_info.data)
                stride = actual_size // height
                
                # 先以步長建立原始陣列，再裁剪掉 Padding 部分
                raw_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, stride // 3, 3))
                frame_data = raw_array[:, :width, :].copy() # 裁切並建立副本以釋放原始 Buffer
                
                tensor = torch.from_numpy(frame_data).to("cuda")
                
                with self._lock:
                    self.last_tensor = tensor
            finally:
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def start(self):
        self._running = True
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"🚀 GStreamer Pipeline started: {self.source_url}")

    def grab_frame_tensor(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self.last_tensor

    def stop(self):
        self._running = False
        self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    decoder = GstZeroCopyDecoder("rtsp://localhost:8554/live")
    print("GStreamer Decoder Ready.")
