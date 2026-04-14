import os
import threading
import time
from typing import Optional, Tuple, Any
import numpy as np
import torch
import gi # noqa: E402
gi.require_version('Gst', '1.0') # noqa: E402
gi.require_version('GstApp', '1.0') # noqa: E402
from gi.repository import Gst, GstApp, GLib # noqa: E402

try:
    import saccade_media_ext
    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

# 初始化 GStreamer
Gst.init(None)

class MediaMTXClient:
    """
    Saccade 媒體用戶端 (整合 C++ GStreamer 與 GPU Buffer Pool)
    
    支援 C++ 層級的高效能 5-Buffer 循環緩衝池。
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
        
        # 偵測是否使用 C++ 擴展
        self.use_cpp = HAS_CPP_EXT
        self.cpp_client: Optional[saccade_media_ext.GstClient] = None
        
        # GStreamer 組件 (Python 備援用)
        self.pipeline: Optional[Gst.Pipeline] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._mainloop = GLib.MainLoop()
        
        self.decoder_name = self._get_best_decoder()

    def _get_best_decoder(self) -> str:
        """偵測 GStreamer 註冊表，回傳最佳解碼器名稱"""
        registry = Gst.Registry.get()
        if registry.find_feature("nvh264dec", Gst.ElementFactory.__gtype__):
            return "nvh264dec"
        else:
            print("⚠️ Warning: nvh264dec not found, falling back to CPU decoder (avdec_h264).")
            return "avdec_h264"

    def _get_pipeline_str(self) -> str:
        """根據配置構建 GStreamer 管線"""
        if self.decoder_name == "nvh264dec":
            decoder_path = "nvh264dec ! cudaconvert ! video/x-raw(memory:CUDAMemory),format=RGB"
        else:
            decoder_path = "avdec_h264 ! videoconvert ! video/x-raw,format=RGB"
            
        sink_path = "appsink name=sink emit-signals=true max-buffers=1 drop=true"

        if self.dummy_video and os.path.exists(self.dummy_video):
            path = os.path.abspath(self.dummy_video)
            return f"filesrc location={path} ! qtdemux ! h264parse ! {decoder_path} ! {sink_path}"
        elif self.use_local:
            return f"v4l2src ! videoconvert ! {decoder_path} ! {sink_path}"
        else:
            return f"rtspsrc location={self.rtsp_url} latency=0 ! rtph264depay ! h264parse ! {decoder_path} ! {sink_path}"

    def connect(self) -> bool:
        """啟動媒體管線"""
        if self.use_cpp:
            try:
                pipeline_str = self._get_pipeline_str()
                print(f"🚀 [MediaClient] Connecting via C++ Pipeline (with GPU Pool)...")
                self.cpp_client = saccade_media_ext.GstClient(pipeline_str)
                self.cpp_client.set_frame_callback(self._on_cpp_frame)
                
                if self.cpp_client.connect():
                    self._running = True
                    return True
            except Exception as e:
                print(f"⚠️ [MediaClient] C++ Connection failed: {e}. Falling back to Python...")
                self.use_cpp = False

        # Python 備援模式
        try:
            pipeline_str = self._get_pipeline_str()
            print("📡 [MediaClient] Connecting via Python Pipeline...")
            self.pipeline = Gst.parse_launch(pipeline_str)
            sink = self.pipeline.get_by_name("sink")
            sink.connect("new-sample", self._on_new_sample)
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            self._running = True
            self.pipeline.set_state(Gst.State.PLAYING)
            self._loop_thread = threading.Thread(target=self._mainloop.run, daemon=True)
            self._loop_thread.start()
            return True
        except Exception as e:
            print(f"❌ [MediaClient] Python Connection failed: {e}")
            return False

    def _on_cpp_frame(self, frame_data: Any) -> None:
        """C++ 擴展的回調函式：處理 GPU 指標"""
        try:
            class CudaPointerHolder:
                def __init__(self, ptr: int, size: int, shape: Tuple[int, int, int], dtype: str) -> None:
                    self.__cuda_array_interface__ = {
                        "shape": shape,
                        "typestr": dtype,
                        "data": (ptr, False),
                        "version": 3
                    }
            
            holder = CudaPointerHolder(
                ptr=frame_data.cuda_ptr,
                size=frame_data.width * frame_data.height * 3,
                shape=(frame_data.height, frame_data.width, 3),
                dtype="|u1" # uint8
            )
            
            tensor = torch.as_tensor(holder, device="cuda")
            
            with self._lock:
                self._last_tensor = tensor
                self._ret = True
                
        except Exception as e:
            print(f"❌ [MediaClient] Error processing C++ frame: {e}")

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ [MediaClient] GStreamer Bus Error: {err.message}")
        elif t == Gst.MessageType.EOS:
            print("🏁 [MediaClient] GStreamer: End of stream")

    def _on_new_sample(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if not sample: return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if not caps: return Gst.FlowReturn.ERROR
        struct = caps.get_structure(0)
        width, height = struct.get_value("width"), struct.get_value("height")
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                stride = len(map_info.data) // height
                raw_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, stride // 3, 3))
                frame_data = raw_array[:, :width, :].copy()
                tensor = torch.from_numpy(frame_data).to("cuda")
                with self._lock:
                    self._last_frame, self._last_tensor, self._ret = frame_data, tensor, True
            finally:
                buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock: return self._ret, self._last_frame

    def grab_tensor(self) -> Tuple[bool, Optional[torch.Tensor]]:
        with self._lock: return self._ret, self._last_tensor

    def release(self) -> None:
        self._running = False
        if self.use_cpp and self.cpp_client:
            self.cpp_client.release()
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
