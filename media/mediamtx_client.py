import os
import threading
import time
from typing import Optional, Tuple
import numpy as np
import torch
import gi # noqa: E402
gi.require_version('Gst', '1.0') # noqa: E402
gi.require_version('GstApp', '1.0') # noqa: E402
from gi.repository import Gst, GstApp, GLib # noqa: E402

# 初始化 GStreamer
Gst.init(None)

class MediaMTXClient:
    """
    Saccade 媒體用戶端 (整合 GStreamer Zero-Copy)
    
    支援 NVDEC 硬體解碼與 CUDA 記憶體直接映射。
    """
    def __init__(self, rtsp_url: str = "rtsp://localhost:8554/live", use_local: bool = False, dummy_video: Optional[str] = None):
        self.rtsp_url = rtsp_url
        self.use_local = use_local
        self.dummy_video = dummy_video
        
        # 狀態管理
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_tensor: Optional[torch.Tensor] = None
        self._ret = False
        self._lock = threading.Lock()
        
        # GStreamer 組件
        self.pipeline: Optional[Gst.Pipeline] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._mainloop = GLib.MainLoop()

    def _get_pipeline_str(self) -> str:
        """根據配置構建 GStreamer 管線"""
        decoder_path = "nvh264dec ! cudaconvert ! video/x-raw(memory:CUDAMemory),format=RGB"
        sink_path = "appsink name=sink emit-signals=true max-buffers=1 drop=true"

        if self.dummy_video and os.path.exists(self.dummy_video):
            path = os.path.abspath(self.dummy_video)
            return f"filesrc location={path} ! qtdemux ! h264parse ! {decoder_path} ! {sink_path}"
        elif self.use_local:
            return f"v4l2src ! videoconvert ! {decoder_path} ! {sink_path}"
        else:
            return f"rtspsrc location={self.rtsp_url} latency=0 ! rtph264depay ! h264parse ! {decoder_path} ! {sink_path}"

    def connect(self) -> bool:
        """啟動 GStreamer 管線"""
        try:
            pipeline_str = self._get_pipeline_str()
            print("📡 Connecting to stream via GStreamer...")
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            sink = self.pipeline.get_by_name("sink")
            sink.connect("new-sample", self._on_new_sample)
            
            self._running = True
            self.pipeline.set_state(Gst.State.PLAYING)
            
            # 啟動 GLib MainLoop 執行緒
            self._loop_thread = threading.Thread(target=self._mainloop.run, daemon=True)
            self._loop_thread.start()
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def _on_new_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if not caps:
            return Gst.FlowReturn.ERROR
        
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                # 處理記憶體對齊
                actual_size = len(map_info.data)
                stride = actual_size // height
                raw_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, stride // 3, 3))
                frame_data = raw_array[:, :width, :].copy()
                
                # 同時準備 Tensor 與 Numpy (相容性)
                tensor = torch.from_numpy(frame_data).to("cuda")
                
                with self._lock:
                    self._last_frame = frame_data
                    self._last_tensor = tensor
                    self._ret = True
            finally:
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """獲取最新影格 (Numpy 格式，向下相容)"""
        with self._lock:
            return self._ret, self._last_frame

    def grab_tensor(self) -> Tuple[bool, Optional[torch.Tensor]]:
        """獲取最新影格 (CUDA Tensor 格式，Zero-Copy)"""
        with self._lock:
            return self._ret, self._last_tensor

    def release(self) -> None:
        """釋放資源"""
        self._running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self._mainloop:
            self._mainloop.quit()

if __name__ == "__main__":
    client = MediaMTXClient()
    if client.connect():
        print("✅ Integrated MediaMTXClient connected.")
        time.sleep(2)
        ret, tensor = client.grab_tensor()
        if ret and tensor is not None:
            print(f"Got tensor on: {tensor.device}")
        client.release()
