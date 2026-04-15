import os
import threading
import time
from typing import Optional, Tuple, Any
import numpy as np
import torch
import gi  # noqa: E402

gi.require_version("Gst", "1.0")  # noqa: E402
gi.require_version("GstApp", "1.0")  # noqa: E402
from gi.repository import Gst, GstApp, GLib  # noqa: E402

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

    def __init__(
        self,
        rtsp_url: str = "rtsp://localhost:8554/live",
        use_local: bool = False,
        dummy_video: Optional[str] = None,
    ):
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
            print(
                "⚠️ Warning: nvh264dec not found, falling back to CPU decoder (avdec_h264)."
            )
            return "avdec_h264"

    def _get_pipeline_str(self) -> str:
        """根據配置構建 GStreamer 管線"""
        if self.decoder_name == "nvh264dec":
            # 硬體路徑：輸出 NV12 以達成零拷貝轉換
            decoder_path = "nvh264dec ! video/x-raw,format=NV12"
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
                print("🚀 [MediaClient] Connecting via C++ Pipeline (with GPU Pool)...")
                self.cpp_client = saccade_media_ext.GstClient(pipeline_str)
                self.cpp_client.set_frame_callback(self._on_cpp_frame)

                if self.cpp_client.connect():
                    self._running = True
                    return True
            except Exception as e:
                print(
                    f"⚠️ [MediaClient] C++ Connection failed: {e}. Falling back to Python..."
                )
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

    def _nv12_to_rgb_gpu(self, raw_nv12: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        在 GPU 內將 NV12 (YUV420) 轉換為 RGB (Zero-Copy)
        使用 ITU-R BT.601 標準
        """
        # 分離 Y 與 UV 平面
        y_plane = raw_nv12[: h * w].view(1, 1, h, w).float()
        uv_plane = raw_nv12[h * w :].view(1, h // 2, w // 2, 2).float()

        # 提取 U 與 V (NV12 是 interleaved UV)
        u_plane = uv_plane[:, :, :, 0].unsqueeze(1)
        v_plane = uv_plane[:, :, :, 1].unsqueeze(1)

        # 縮放 U/V 平面至與 Y 一致
        u_up = torch.nn.functional.interpolate(
            u_plane, size=(h, w), mode="bilinear", align_corners=False
        )
        v_up = torch.nn.functional.interpolate(
            v_plane, size=(h, w), mode="bilinear", align_corners=False
        )

        # 轉換公式 (BT.601)
        y = (y_plane - 16.0) * 1.164
        u = u_up - 128.0
        v = v_up - 128.0

        r = y + 1.596 * v
        g = y - 0.391 * u - 0.813 * v
        b = y + 2.018 * u

        rgb = torch.cat([r, g, b], dim=1).clamp(0, 255).byte()
        return rgb.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

    def _on_cpp_frame(self, frame_data: Any) -> None:
        """C++ 擴展的回調函式：處理 GPU 指標"""
        try:
            h, w = frame_data.height, frame_data.width
            # 根據 channels 判斷格式 (C++ 層目前寫死 3，未來若支援 NV12 可透過 channels=0 或其他約定辨識)
            channels = getattr(frame_data, "channels", 3)
            is_nv12 = channels == 0  # 假設 0 代表 YUV/NV12，3 代表 RGB

            class CudaPointerHolder:
                def __init__(
                    self, ptr: int, shape: Tuple[int, ...], dtype: str
                ) -> None:
                    self.__cuda_array_interface__ = {
                        "shape": shape,
                        "typestr": dtype,
                        "data": (ptr, False),
                        "version": 3,
                    }

            if is_nv12:
                # NV12 佔用空間為 1.5 * H * W
                holder = CudaPointerHolder(
                    ptr=frame_data.cuda_ptr,
                    shape=(int(h * 1.5), w),
                    dtype="|u1",
                )
                raw_tensor = torch.as_tensor(holder, device="cuda")
                tensor = self._nv12_to_rgb_gpu(raw_tensor.flatten(), h, w)
            else:
                # 預設為 RGB [H, W, 3]
                holder = CudaPointerHolder(
                    ptr=frame_data.cuda_ptr,
                    shape=(h, w, 3),
                    dtype="|u1",
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
        if not sample:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if not caps:
            return Gst.FlowReturn.ERROR
        struct = caps.get_structure(0)
        width, height = struct.get_value("width"), struct.get_value("height")
        fmt = struct.get_value("format")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                # 將原始資料載入 GPU
                raw_data = torch.from_numpy(
                    np.frombuffer(map_info.data, dtype=np.uint8)
                ).to("cuda")

                if fmt == "NV12":
                    rgb_tensor = self._nv12_to_rgb_gpu(raw_data, height, width)
                else:
                    stride = len(map_info.data) // height
                    rgb_tensor = raw_data.view(height, stride // 3, 3)[:, :width, :]

                with self._lock:
                    self._last_frame = None  # 延遲解碼 np.ndarray 以節省效能
                    self._last_tensor = rgb_tensor
                    self._ret = True
            finally:
                buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is None and self._last_tensor is not None:
                # 延遲轉換：僅在真正需要視覺化時才從 GPU 搬回 CPU
                self._last_frame = self._last_tensor.cpu().numpy()
            return self._ret, self._last_frame

    def grab_tensor(self) -> Tuple[bool, Optional[torch.Tensor]]:
        with self._lock:
            return self._ret, self._last_tensor

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
