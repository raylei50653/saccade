import os
import torch
import tensorrt as trt
import threading
from typing import Any, Dict, List, Optional, Tuple

from saccade.perception.tracking import GPUByteTracker  # noqa: E402

try:
    from saccade_perception_ext import TRTEngine as CppTRTEngine

    HAS_CPP_EXT = True
except ImportError as e:
    print(f"❌ [TRT] Failed to import C++ extension: {e}")
    HAS_CPP_EXT = False


class TRTYoloDetector:
    """
    YOLO TensorRT 偵測器，支援純偵測 (6-col) 與 Pose (57-col) 兩種輸出格式。
    優先使用 C++ 核心引擎以獲得最佳效能與最低抖動。

    支援併發模式（透過 context_factory）：
    - 共享 engine 例項（只讀）
    - 每個序列建立獨立 context
    """

    def __init__(
        self,
        engine_path: str = "models/yolo/yolo11n_pose_batch6.engine",
        device: str = "cuda:0",
        enable_concurrent: bool = False,
    ):
        self.device = device
        self.enable_concurrent = enable_concurrent
        backend = os.environ.get("SACCADE_TRT_BACKEND", "auto").strip().lower()
        if backend not in {"auto", "cpp", "python"}:
            raise ValueError("SACCADE_TRT_BACKEND must be one of: auto, cpp, python")

        self.use_cpp = backend != "python" and HAS_CPP_EXT
        if backend == "cpp" and not HAS_CPP_EXT:
            raise RuntimeError(
                "SACCADE_TRT_BACKEND=cpp requested, but saccade_perception_ext is unavailable"
            )
        self.input_name: str = "images"
        self.output_name: str = "output0"
        self.output_names: List[str] = []
        self.output_tensors: Dict[str, torch.Tensor] = {}
        self.is_dynamic: bool = False
        self.input_shape: List[int] = []
        self.output_shape: Tuple[int, ...] = (0,)

        if self.use_cpp:
            print(f"🚀 [TRT] Loading C++ Optimized Engine from {engine_path}...")
            self.cpp_engine = CppTRTEngine(engine_path)
            # Use get_tensor_shape instead of get_input_shape as seen in previous grep
            self.input_shape = self.cpp_engine.get_tensor_shape(self.input_name)
            self.output_shape = tuple(
                self.cpp_engine.get_tensor_shape(self.output_name)
            )
            self.output_names = [self.output_name]
            self.is_dynamic = self.output_shape[0] == -1
            self.tracker = GPUByteTracker(max_objects=2048)

            # 併發模式：建立 context factory
            if enable_concurrent:
                self._concurrent_contexts: Dict[str, "_TRTContext"] = {}
                self._context_lock = threading.Lock()
                print("✅ C++ YOLO Detector Ready (concurrent mode).")
            else:
                print("✅ C++ YOLO Detector & Tracker Ready.")
            return
        else:
            print(
                f"⚠️ [TRT] Using Python Native API for {engine_path} "
                f"(backend={backend}, cpp_available={HAS_CPP_EXT})"
            )
            self.logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

            self.context = self.engine.create_execution_context()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name

            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # 💡 偵測模型是否支援動態 Batch
        self.is_dynamic = self.output_shape[0] == -1

        # 取得輸入輸出名稱與形狀 (支援 detection-only 與 YOLOE segmentation 多輸出)
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors.")

        self.output_name = self.output_names[0]
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            self.output_tensors[name] = torch.empty(
                self._resolve_output_shape(shape, batch_size=4),
                device=self.device,
                dtype=torch.float32,
            )

        # 初始化 GPU Tracker (包含 Sinkhorn + Kalman 邏輯)
        self.tracker = GPUByteTracker(max_objects=2048)

        print(
            f"✅ Native YOLO Detector Ready. Input: {self.input_name}, Outputs: {self._format_outputs()}"
        )

    def _format_outputs(self) -> str:
        return ", ".join(
            f"{name} {self.engine.get_tensor_shape(name)}" for name in self.output_names
        )

    def _resolve_output_shape(
        self, shape: Tuple[int, ...], batch_size: int
    ) -> Tuple[int, ...]:
        dims = []
        for idx, dim in enumerate(shape):
            if dim == -1:
                dims.append(batch_size if idx == 0 else 1)
            else:
                dims.append(dim)
        return tuple(dims)

    def reset_tracker(self) -> None:
        """重置追蹤器狀態，用於切換影片序列時。"""
        self.tracker = GPUByteTracker(max_objects=2048)

    def _empty_result(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return (
            torch.empty((0, 4), device=self.device),
            torch.empty((0,), device=self.device),
            torch.empty((0,), device=self.device),
            None,
        )

    def infer_raw_batch(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        執行 TensorRT 推理並回傳所有輸出張量。
        """
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.contiguous()
        stream = torch.cuda.current_stream().cuda_stream

        if self.use_cpp:
            # 1. 準備所有輸出空間
            for name in self.output_names:
                shape = self.output_shape
                if self.is_dynamic:
                    # Resolve -1 to actual batch_size
                    shape = tuple([batch_size if d == -1 else d for d in shape])

                current = self.output_tensors.get(name)
                if current is None or tuple(current.shape) != shape:
                    self.output_tensors[name] = torch.empty(
                        shape, device=self.device, dtype=torch.float32
                    )

            # 2. 準備 bindings 並執行
            if self.is_dynamic:
                self.cpp_engine.set_input_shape(
                    self.input_name, list(input_tensor.shape)
                )

            binding_ptrs = [input_tensor.data_ptr()]
            for name in self.output_names:
                binding_ptrs.append(self.output_tensors[name].data_ptr())

            self.cpp_engine.infer(binding_ptrs, stream)
            return self.output_tensors

        # Python Native Path (Fallback)
        # 1. 設定動態輸入 Shape
        self.context.set_input_shape(self.input_name, input_tensor.shape)

        # 2. 準備所有輸出空間
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                shape = self._resolve_output_shape(
                    self.engine.get_tensor_shape(name), batch_size
                )

            current = self.output_tensors.get(name)
            if current is None or tuple(current.shape) != shape:
                # print(f"DEBUG: creating output tensor {name} with shape {shape}")
                self.output_tensors[name] = torch.empty(
                    shape, device=self.device, dtype=torch.float32
                )

        # 3. 綁定並執行所有輸入/輸出
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        bound_names = []
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
            bound_names.append(name)

        self.context.execute_async_v3(stream)
        return self.output_tensors

    def detect_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行 TensorRT 推理並直接回傳原始輸出張量 [Batch, 300, 6]。
        """
        outputs = self.infer_raw_batch(input_tensor)
        return outputs[self.output_name]

    def detect_batch(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        執行批次偵測與追蹤
        """
        batch_size = input_tensor.size(0)
        outputs = self.infer_raw_batch(input_tensor)
        output_tensor = outputs[self.output_name]

        batch_results = []
        for i in range(batch_size):
            results = output_tensor[i]
            mask = results[:, 4] > conf_threshold
            valid_results = results[mask]

            if valid_results.size(0) == 0:
                batch_results.append(self._empty_result())
                continue

            boxes = valid_results[:, :4].contiguous()
            scores = valid_results[:, 4].contiguous()
            classes = valid_results[:, 5].to(torch.int32).contiguous()

            extra: Optional[torch.Tensor] = None
            if "embeddings" in outputs:
                extra = outputs["embeddings"][i][mask].contiguous()
            elif valid_results.size(1) > 6:
                extra = valid_results[:, 6:].contiguous()
                if extra.size(1) == 51:  # COCO 17-keypoint pose: reshape to [N, 17, 3]
                    extra = extra.reshape(-1, 17, 3)
            else:
                extra = None

            batch_results.append((boxes, scores, classes, extra))

        return batch_results

    def detect(
        self, input_tensor: torch.Tensor, conf_threshold: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """單路相容性接口"""
        results = self.detect_batch(input_tensor, conf_threshold)
        return results[0] if results else self._empty_result()


def _match_pose_to_dets(
    det_boxes: torch.Tensor,
    pose_boxes: torch.Tensor,
    pose_kpts: torch.Tensor,
    max_center_dist: float = 0.5,
) -> torch.Tensor:
    """Match pose keypoints [M, 17, 3] to detection boxes [N, 4] by nearest center.

    Returns keypoints [N, 17, 3]. Unmatched detections (no pose box within
    max_center_dist * sqrt(det_area)) receive NaN keypoints.
    """
    det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) * 0.5
    det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5
    pose_cx = (pose_boxes[:, 0] + pose_boxes[:, 2]) * 0.5
    pose_cy = (pose_boxes[:, 1] + pose_boxes[:, 3]) * 0.5

    dx = det_cx.unsqueeze(1) - pose_cx.unsqueeze(0)  # [N, M]
    dy = det_cy.unsqueeze(1) - pose_cy.unsqueeze(0)  # [N, M]
    dist2 = dx * dx + dy * dy  # [N, M]

    nearest_idx = dist2.argmin(dim=1)  # [N]
    nearest_dist2 = dist2[torch.arange(det_boxes.size(0)), nearest_idx]

    # Threshold: max_center_dist * sqrt(det_area)
    det_w = (det_boxes[:, 2] - det_boxes[:, 0]).clamp(min=1)
    det_h = (det_boxes[:, 3] - det_boxes[:, 1]).clamp(min=1)
    thresh2 = (max_center_dist * (det_w * det_h).sqrt()) ** 2  # [N]

    matched_kpts = pose_kpts[nearest_idx]  # [N, 17, 3]
    nan_kpts = torch.full_like(matched_kpts, float("nan"))
    valid = nearest_dist2 <= thresh2  # [N]
    return torch.where(valid[:, None, None], matched_kpts, nan_kpts)


class TwostageDetector:
    """Two-stage detector: yolo (detection) + pose model (keypoints).

    Stage 1 – yolo engine: detection boxes, scores, classes.
    Stage 2 – pose engine: pose boxes + keypoints matched back to Stage 1 boxes.

    detect_raw() delegates to the detection engine (used by tiled detection paths).
    detect_batch() runs both engines and returns (boxes, scores, classes, kpts [N,17,3]).
    """

    def __init__(
        self,
        det_engine: str,
        pose_engine: str,
        pose_conf_threshold: float = 0.001,
    ) -> None:
        self.det = TRTYoloDetector(engine_path=det_engine)
        self.pose = TRTYoloDetector(engine_path=pose_engine)
        self.pose_conf_threshold = pose_conf_threshold

    # ── proxy attributes / methods to the detection engine ──────────────────

    @property
    def tracker(self):
        return self.det.tracker

    def reset_tracker(self):
        return self.det.reset_tracker()

    def detect_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.det.detect_raw(input_tensor)

    def infer_raw_batch(self, input_tensor: torch.Tensor):
        return self.det.infer_raw_batch(input_tensor)

    # ── two-stage interface ──────────────────────────────────────────────────

    def detect_batch(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        det_results = self.det.detect_batch(input_tensor, conf_threshold)
        pose_results = self.pose.detect_batch(input_tensor, self.pose_conf_threshold)

        merged: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = []
        for (det_boxes, det_scores, det_classes, _), pose_item in zip(
            det_results, pose_results
        ):
            pose_boxes, _, _, pose_kpts = pose_item
            if pose_kpts is None or pose_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
                merged.append((det_boxes, det_scores, det_classes, None))
            else:
                kpts = _match_pose_to_dets(det_boxes, pose_boxes, pose_kpts)
                merged.append((det_boxes, det_scores, det_classes, kpts))
        return merged

    def detect(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        results = self.detect_batch(input_tensor, conf_threshold)
        return results[0] if results else self.det._empty_result()


# ============================================================================
# Concurrent Inference Support
# ============================================================================


class _TRTContext:
    """
    Per-sequence TRT context with independent stream + output buffers.

    This class wraps a raw IExecutionContext pointer and manages:
    - CUDA stream for async execution
    - Output tensor buffers
    - Reference counting (auto-cleanup on release)
    """

    def __init__(self, engine: "CppTRTEngine", stream: torch.cuda.Stream):
        self.engine = engine
        self.stream = stream
        self.context_ptr: Optional[int] = None
        self.output_tensors: Dict[str, torch.Tensor] = {}
        self._released = False

    def allocate_output_buffers(self, output_shape: Tuple[int, ...]) -> None:
        """預先分配輸出 buffer"""
        self.output_tensors["output0"] = torch.empty(
            output_shape, device="cuda", dtype=torch.float32
        )

    def infer(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        執行單幀推理（使用獨立 context + stream）
        """
        if self.context_ptr is None:
            # 建立 context
            self.context_ptr = self.engine.create_context()

        input_tensor = input_tensor.contiguous().unsqueeze(0)

        binding_ptrs = [input_tensor.data_ptr()]
        if "output0" in self.output_tensors:
            binding_ptrs.append(self.output_tensors["output0"].data_ptr())

        self.engine.infer_with_context(
            self.context_ptr, binding_ptrs, int(self.stream.cuda_stream)
        )
        self.stream.synchronize()
        output_tensor = self.output_tensors["output0"]

        # 後處理
        return self._postprocess(output_tensor, conf_threshold)

    def _postprocess(
        self,
        output_tensor: torch.Tensor,
        conf_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """標準 YOLO 後處理"""
        results = output_tensor[0]
        mask = results[:, 4] > conf_threshold
        valid_results = results[mask]

        if valid_results.size(0) == 0:
            return (
                torch.empty((0, 4), device="cuda"),
                torch.empty((0,), device="cuda"),
                torch.empty((0,), dtype=torch.int32, device="cuda"),
                None,
            )

        boxes = valid_results[:, :4].contiguous()
        scores = valid_results[:, 4].contiguous()
        classes = valid_results[:, 5].to(torch.int32).contiguous()

        return boxes, scores, classes, None

    def release(self) -> None:
        if not self._released and self.context_ptr is not None:
            CppTRTEngine.delete_context(self.context_ptr)
            self._released = True

    def __del__(self) -> None:
        self.release()


class ConcurrentDetectorProxy:
    """
    併發偵測器代理類，提供與 TRTYoloDetector 相容的 API。

    每個序列獲得獨立 context，共享同一 engine。
    """

    def __init__(
        self,
        detector: TRTYoloDetector,
        stream_id: str,
    ):
        self.detector = detector
        self.stream_id = stream_id
        self.context: Optional[_TRTContext] = None
        self.tracker = GPUByteTracker(max_objects=2048)

    def reset_tracker(self) -> None:
        """重置 tracker 狀態"""
        self.tracker = GPUByteTracker(max_objects=2048)

    def detect(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """單幀偵測"""
        if self.context is None:
            stream = torch.cuda.Stream(device=self.detector.device)
            self.context = _TRTContext(self.detector.cpp_engine, stream)
            self.context.allocate_output_buffers(self.detector.output_shape)

        return self.context.infer(input_tensor, conf_threshold)

    def detect_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Raw output for tiled detection paths (mirrors TRTYoloDetector.detect_raw)."""
        if self.context is None:
            stream = torch.cuda.Stream(device=self.detector.device)
            self.context = _TRTContext(self.detector.cpp_engine, stream)
            self.context.allocate_output_buffers(self.detector.output_shape)
        if self.context.context_ptr is None:
            self.context.context_ptr = self.detector.cpp_engine.create_context()
        batch = input_tensor.contiguous()
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        binding_ptrs = [
            batch.data_ptr(),
            self.context.output_tensors["output0"].data_ptr(),
        ]
        self.detector.cpp_engine.infer_with_context(
            self.context.context_ptr,
            binding_ptrs,
            int(self.context.stream.cuda_stream),
        )
        self.context.stream.synchronize()
        return self.context.output_tensors["output0"]

    def detect_batch(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """批次偵測"""
        batch_size = input_tensor.size(0)
        results = []
        for i in range(batch_size):
            result = self.detect(input_tensor[i], conf_threshold)
            results.append(result)
        return results

    @property
    def tracker(self) -> GPUByteTracker:
        return self._tracker

    @tracker.setter
    def tracker(self, value: GPUByteTracker) -> None:
        self._tracker = value


# ============================================================================
# Batch-fused concurrent detection
# ============================================================================


class BatchingTRTDetector:
    """Fuses per-sequence detect_raw() calls into one batch-N TRT inference.

    Workers (threads) call BatchedDetectorProxy.detect_raw(frame_1x3xHxW).
    This class collects up to batch_size frames, fires one TRT call, then
    distributes result slices back to the waiting callers.

    Requires a dynamic-batch engine (min=1, opt=batch_size, max=batch_size).

    Usage:
        batcher = BatchingTRTDetector("models/yolo/yolo26s_960_batch4.engine",
                                      batch_size=4)
        # per sequence — each thread gets its own proxy with independent tracker
        proxy = batcher.make_proxy()

        # normal concurrent eval:
        run_eval(..., detector=proxy, ...)
    """

    def __init__(
        self,
        engine_path: str,
        batch_size: int = 4,
        drain_ms: float = 2.0,
        device: str = "cuda:0",
    ) -> None:
        self._base = TRTYoloDetector(
            engine_path=engine_path, device=device, enable_concurrent=False
        )
        self._batch_size = batch_size
        self._drain_s = drain_ms / 1000.0
        self._device = device

        # Dedicated server stream — isolates batch inference from worker
        # postprocess streams. The server waits on each frame_event before
        # launching inference, then records infer_event so workers can chain
        # their postprocess kernels via stream.wait_event (no host sync).
        self._server_stream = torch.cuda.Stream(device=device)

        # Pending items: (frame [1,C,H,W], frame_event, result_holder, done_event)
        self._pending: List[
            Tuple[torch.Tensor, torch.cuda.Event, List, threading.Event]
        ] = []
        self._cv = threading.Condition()
        self._shutdown = False

        self._server = threading.Thread(
            target=self._loop, name="BatchTRT-server", daemon=True
        )
        self._server.start()
        print(
            f"🔀 [BatchingTRTDetector] batch_size={batch_size} drain={drain_ms}ms  {engine_path}"
        )

    # ── public ────────────────────────────────────────────────────────────────

    def make_proxy(self) -> "BatchedDetectorProxy":
        """Return a per-sequence proxy with its own tracker."""
        return BatchedDetectorProxy(self, device=self._device)

    def submit(
        self, frame: torch.Tensor, worker_stream: torch.cuda.Stream
    ) -> torch.Tensor:
        """Submit one [1,C,H,W] frame; block until the server has dispatched
        the batch inference. The returned tensor is a view into a server-owned
        clone; ``worker_stream`` is made to wait on the inference event so any
        kernels launched on it afterwards safely see the data.
        """
        frame_event = torch.cuda.Event()
        frame_event.record(worker_stream)
        holder: List[Optional[Tuple[torch.Tensor, torch.cuda.Event]]] = [None]
        done = threading.Event()
        with self._cv:
            self._pending.append((frame, frame_event, holder, done))
            if len(self._pending) >= self._batch_size:
                self._cv.notify()
        done.wait()
        slice_tensor, infer_event = holder[0]  # type: ignore[misc]
        worker_stream.wait_event(infer_event)
        return slice_tensor

    def shutdown(self) -> None:
        self._shutdown = True
        with self._cv:
            self._cv.notify_all()
        self._server.join(timeout=2.0)

    # ── server loop ───────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._shutdown:
            with self._cv:
                self._cv.wait_for(
                    lambda: len(self._pending) >= self._batch_size or self._shutdown,
                    timeout=self._drain_s,
                )
                if not self._pending:
                    continue
                # Drain: take up to batch_size items regardless of fill level
                batch = self._pending[: self._batch_size]
                self._pending = self._pending[self._batch_size :]

            # All GPU work runs on the server stream. Wait on each worker's
            # frame_event so their preproc writes are visible before we read
            # the input tensors. Record infer_event after clone so workers can
            # chain their own postprocess kernels via wait_event.
            with torch.cuda.stream(self._server_stream):
                for _, frame_event, _, _ in batch:
                    self._server_stream.wait_event(frame_event)
                frames = torch.cat([item[0] for item in batch], dim=0)  # [B,C,H,W]
                raw_outputs = self._base.infer_raw_batch(frames)  # [B,300,6]
                output = raw_outputs[self._base.output_name].clone()  # own the buffer
                infer_event = torch.cuda.Event()
                infer_event.record(self._server_stream)

            for i, (_, _, holder, done) in enumerate(batch):
                holder[0] = (output[i : i + 1], infer_event)  # [1,300,6]
                done.set()


class BatchedDetectorProxy:
    """Per-sequence proxy that routes detect_raw() through BatchingTRTDetector.

    Owns an independent GPUByteTracker so sequence state never mixes.
    All other detector attributes are forwarded to the underlying base detector.
    """

    def __init__(self, batcher: BatchingTRTDetector, device: str = "cuda:0") -> None:
        self._batcher = batcher
        self._base = batcher._base
        self.tracker = GPUByteTracker(max_objects=2048)

    # ── detection API ─────────────────────────────────────────────────────────

    def detect_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Submit one frame to the batch queue and wait for the raw [1,300,6] result.

        The submission is ordered against the **calling thread's current
        stream**: the server waits for that stream to finish writing the input
        frame, and the same stream is made to wait on the batch inference
        before this function returns. Wrap caller work in
        ``with torch.cuda.stream(s):`` to isolate it from sibling workers.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        worker_stream = torch.cuda.current_stream(input_tensor.device)
        return self._batcher.submit(input_tensor.contiguous(), worker_stream)

    def detect_batch(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> List[Any]:
        """Batch detection — each frame submitted individually (for API compatibility)."""
        results = []
        for i in range(input_tensor.size(0)):
            raw = self.detect_raw(input_tensor[i : i + 1])
            results.append(self._base.detect_batch(raw, conf_threshold)[0])
        return results

    def reset_tracker(self) -> None:
        self.tracker = GPUByteTracker(max_objects=2048)

    # ── attribute forwarding ──────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attributes to base detector (input_shape, output_shape,
        # is_dynamic, device, use_cpp, cpp_engine, etc.)
        return getattr(self._base, name)
