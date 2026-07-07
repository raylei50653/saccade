import os
import shlex
import threading
import time
import asyncio
from typing import Optional, Tuple, Any
import numpy as np
import torch
import gi  # noqa: E402

gi.require_version("Gst", "1.0")  # noqa: E402
gi.require_version("GstApp", "1.0")  # noqa: E402
from gi.repository import Gst, GstApp, GLib  # noqa: E402
from saccade.media.rtsp import build_rtsp_url, DEFAULT_RTSP_SINGLE_STREAM_PATH  # noqa: E402

try:
    import saccade_media_ext

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

# 初始化 GStreamer
Gst.init([])


def _gst_property_value(value: str) -> str:
    """Quote values embedded into a Gst.parse_launch pipeline string."""
    return shlex.quote(value)


class MediaMTXClient:
    """
    Saccade 媒體用戶端 (整合 C++ GStreamer 與 GPU Buffer Pool)

    支援 C++ 層級的高效能 5-Buffer 循環緩衝池。
    """

    def __init__(
        self,
        rtsp_url: str = build_rtsp_url(DEFAULT_RTSP_SINGLE_STREAM_PATH),
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
        self._last_frame_time = time.time()
        self._ret = False
        self._bus_error: Optional[str] = None
        self._frame_generation = 0
        self._last_grab_generation = 0
        self._lock = threading.Lock()

        # 舊的 C++ GstClient 路徑仍屬早期實驗功能，對 RTSP 場景預設關閉。
        self.use_cpp = HAS_CPP_EXT and os.getenv("SACCADE_MEDIA_USE_CPP", "0") == "1"
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
        sink_path = (
            "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        if self.dummy_video and os.path.exists(self.dummy_video):
            path = _gst_property_value(os.path.abspath(self.dummy_video))
            return (
                f"filesrc location={path} ! decodebin ! videoconvert ! "
                f"video/x-raw,format=RGB ! {sink_path}"
            )
        elif self.use_local:
            return f"v4l2src ! videoconvert ! video/x-raw,format=RGB ! {sink_path}"
        else:
            # RTSP 讀流路徑優先追求穩定拿到 frame，而不是實驗性的 GPU decode。
            # MediaMTX + test publisher 在 nvh264dec 路徑下會報 internal data stream error；
            # 使用 TCP + avdec_h264/decodebin 則能穩定輸出到 appsink。
            rtsp_url = _gst_property_value(self.rtsp_url)
            return (
                f"rtspsrc location={rtsp_url} latency=0 protocols=tcp ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                f"video/x-raw,format=RGB ! {sink_path}"
            )

    def connect(self) -> bool:
        """啟動媒體管線"""
        self._last_frame_time = time.time()
        self._ret = False
        self._bus_error = None
        if self.use_cpp:
            try:
                pipeline_str = self._get_pipeline_str()
                print("🚀 [MediaClient] Connecting via C++ Pipeline (with GPU Pool)...")
                self.cpp_client = saccade_media_ext.GstClient(pipeline_str)
                self.cpp_client.set_frame_callback(self._on_cpp_frame)

                if self.cpp_client.connect():
                    if not self._await_first_frame():
                        print(
                            "❌ [MediaClient] C++ pipeline connected but no frame arrived."
                        )
                        self.release()
                        return False
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
            if not self._await_first_frame():
                reason = self._bus_error or "Timed out waiting for first frame"
                print(
                    f"❌ [MediaClient] Python pipeline failed before first frame: {reason}"
                )
                self.release()
                return False
            return True
        except Exception as e:
            print(f"❌ [MediaClient] Python Connection failed: {e}")
            return False

    def _await_first_frame(self, timeout_sec: float = 3.0) -> bool:
        """阻塞直到收到第一幀或發生 bus error。"""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._lock:
                if self._ret and self._last_tensor is not None:
                    return True
                if self._bus_error:
                    return False
            time.sleep(0.05)
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
        """C++ 擴展的回調函式:處理 GPU 指標 (ADR-009 race-fixed 契約)。

        契約流程 (修 S1/S4/S10):
          1. ``with frame_data`` 觸發 ``__enter__`` (READY→PROCESSING CAS),
             取得 buffer 所有權。
          2. ``sync_buffer`` 等 H2D 完成 (修 S1: 讀 cuda_ptr 前必須 sync)。
          3. ``ExternalStream`` 把後續算子排入 buffer 專屬 stream (修 S10:
             stream-ordered 接軌,推理算子與 H2D 同 stream 不需額外 sync)。
          4. ``tensor.clone()`` D2D 複製成自有顯存,脫離 pool buffer 生命週期,
             pool slot 立即回收 (修 grab_tensor 返回 live view 的覆寫 race)。
          5. ``with`` 結束自動 ``__exit__`` (PROCESSING→EMPTY CAS),歸還 buffer。
             例外路徑也會 release,避免 buffer leak (修 S4/S11)。
        """
        try:
            if self.cpp_client is None:
                return
            idx = frame_data.buffer_index
            h, w = frame_data.height, frame_data.width
            channels = getattr(frame_data, "channels", 3)
            is_nv12 = channels == 0

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

            with frame_data:
                # 等 H2D 完成 (修 S1):讀 cuda_ptr 前必須 sync 該 buffer 專屬 stream。
                self.cpp_client.sync_buffer(idx)
                # 在 buffer 專屬 stream 上建 tensor (修 S10):
                # 後續算子 (NV12→RGB / clone) 排入同 stream,stream-ordered 接軌。
                with torch.cuda.stream(
                    torch.cuda.ExternalStream(frame_data.stream_ptr)  # type: ignore[no-untyped-call]
                ):
                    if is_nv12:
                        holder = CudaPointerHolder(
                            ptr=frame_data.cuda_ptr,
                            shape=(int(h * 1.5), w),
                            dtype="|u1",
                        )
                        raw_tensor = torch.as_tensor(holder, device="cuda")
                        rgb_tensor = self._nv12_to_rgb_gpu(raw_tensor.flatten(), h, w)
                    else:
                        holder = CudaPointerHolder(
                            ptr=frame_data.cuda_ptr,
                            shape=(h, w, 3),
                            dtype="|u1",
                        )
                        raw_tensor = torch.as_tensor(holder, device="cuda")
                        rgb_tensor = raw_tensor
                    # D2D clone 成自有顯存,脫離 pool buffer 生命週期。
                    # pool slot 隨即由 __exit__ 回收為 EMPTY;grab_tensor
                    # 返回的 tensor 不會被下一幀覆寫。
                    tensor = rgb_tensor.clone()
            # __exit__ 已自動 release_buffer (PROCESSING→EMPTY)。

            with self._lock:
                # 智慧抽樣 (Smart Sampling):若像素差異過小則丟棄 (降低低資訊幀)
                if self._last_tensor is not None:
                    diff = torch.mean(
                        torch.abs(tensor.float() - self._last_tensor.float())
                    ).item()
                    if diff < 2.0:
                        return  # 忽略低資訊幀

                self._last_tensor = tensor
                self._last_frame_time = time.time()
                self._frame_generation += 1
                self._ret = True

        except Exception as e:
            print(f"❌ [MediaClient] Error processing C++ frame: {e}")

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            debug_msg = f" ({debug})" if debug else ""
            self._bus_error = err.message
            print(f"❌ [MediaClient] GStreamer Bus Error: {err.message}{debug_msg}")
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
                    # 智慧抽樣 (Smart Sampling)
                    if self._last_tensor is not None:
                        diff = torch.mean(
                            torch.abs(rgb_tensor.float() - self._last_tensor.float())
                        ).item()
                        if diff < 2.0:
                            return Gst.FlowReturn.OK

                    self._last_frame = None  # 延遲解碼 np.ndarray 以節省效能
                    self._last_tensor = rgb_tensor
                    self._last_frame_time = time.time()
                    self._frame_generation += 1
                    self._ret = True
            finally:
                buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def _is_alive(self) -> bool:
        """檢查管線是否仍然活躍 (5秒內有新幀)"""
        return time.time() - self._last_frame_time < 5.0

    def _restart_pipeline(self) -> bool:
        """重啟 GStreamer 管線"""
        print("🔄 [MediaClient] Restarting pipeline...")
        self.release()
        time.sleep(1)  # 等待釋放
        return self.connect()

    async def watchdog_loop(self) -> None:
        """非同步監控循環，負責自動重連"""
        retry_delay = 1
        while self._running:
            if not self._is_alive() and not self.dummy_video:
                print("⚠️ [MediaClient] Stream timeout detected.")
                if self._restart_pipeline():
                    print("✅ [MediaClient] Reconnected successfully.")
                    retry_delay = 1
                else:
                    print(
                        f"❌ [MediaClient] Reconnection failed. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue

            await asyncio.sleep(5)

    def grab_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if (
                not self._ret
                or self._last_tensor is None
                or self._frame_generation == self._last_grab_generation
            ):
                return False, None
            if self._last_frame is None and self._last_tensor is not None:
                # 延遲轉換：僅在真正需要視覺化時才從 GPU 搬回 CPU
                self._last_frame = self._last_tensor.cpu().numpy()
            self._last_grab_generation = self._frame_generation
            return True, self._last_frame

    def grab_tensor(self) -> Tuple[bool, Optional[torch.Tensor]]:
        with self._lock:
            if (
                not self._ret
                or self._last_tensor is None
                or self._frame_generation == self._last_grab_generation
            ):
                return False, None
            self._last_grab_generation = self._frame_generation
            return True, self._last_tensor

    def release(self) -> None:
        self._running = False
        if self.use_cpp and self.cpp_client:
            self.cpp_client.release()
            self.cpp_client = None
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        if self._mainloop:
            self._mainloop.quit()
        if (
            self._loop_thread is not None
            and self._loop_thread.is_alive()
            and self._loop_thread is not threading.current_thread()
        ):
            self._loop_thread.join(timeout=1.0)
        self._loop_thread = None


if __name__ == "__main__":
    client = MediaMTXClient()
    if client.connect():
        print("✅ Integrated MediaMTXClient connected.")
        time.sleep(2)
        ret, tensor = client.grab_tensor()
        if ret and tensor is not None:
            print(f"Got tensor on: {tensor.device}")
        client.release()
