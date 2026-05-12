import os
import torch
import tensorrt as trt
from typing import Dict, Tuple, Optional, List

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
    """

    def __init__(
        self,
        engine_path: str = "models/yolo/yolo11n_pose_batch6.engine",
        device: str = "cuda:0",
    ):
        self.device = device
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
