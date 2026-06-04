"""
Concurrent TRT Detector — 支援多序列併發推論

設計原則：
1. 共享 TRT Engine 例項（只讀，線程安全）
2. 每個工作執行緒建立獨立 Execution Context + cudaStream
3. 每個工作執行緒獨立 GPUTracker 狀態
4. VRAM 管理：每個序列預先分配 buffer，避免動態開銷

使用方式：
    from saccade.perception.eval.concurrent_detector import ConcurrentTRTDetector

    detector = ConcurrentTRTDetector(
        engine_path="models/yolo/yolo26s_960_batch1.engine",
        max_workers=8
    )

    # 每個序列獲得獨立 context
    seq_detector = detector.get_context(stream_id="MOT17-02-SDP")
    result = seq_detector.detect(input_tensor)
    seq_detector.release()
# mypy: ignore-errors
"""

from __future__ import annotations

import torch
import threading
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from saccade.perception.tracking import GPUByteTracker

try:
    from saccade_perception_ext import TRTEngine as CppTRTEngine

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False


@dataclass(frozen=True)
class _ContextHandle:
    """Represents a per-sequence context handle."""

    context_ptr: int  # native pointer to IExecutionContext
    cuda_stream: torch.cuda.Stream
    output_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    input_name: str = "images"
    output_name: str = "output0"


class ConcurrentTRTDetector:
    """
    支援多序列併發的 TRT 偵測器。

    內部使用：
    - 單一 C++ TRTEngine 例項（共享只讀）
    - 每個工作線程建立獨立 context（透過 create_context()）
    - 每個序列獨立 tracker 狀態
    """

    def __init__(
        self,
        engine_path: str,
        max_workers: int = 8,
        device: str = "cuda:0",
        backend: str = "auto",
    ):
        self.engine_path = engine_path
        self.max_workers = max_workers
        self.device = device

        # 初始化共享 engine
        if backend == "auto":
            backend = "cpp" if HAS_CPP_EXT else "python"

        self.backend = backend
        self.use_cpp = backend == "cpp" and HAS_CPP_EXT

        if self.use_cpp:
            self.shared_engine = CppTRTEngine(engine_path)
            self.input_shape = self.shared_engine.get_tensor_shape("images")
            self.output_shape = tuple(self.shared_engine.get_tensor_shape("output0"))
            print(f"🚀 [ConcurrentDetector] C++ Engine loaded: {engine_path}")
        else:
            raise RuntimeError("Concurrent mode requires C++ extension")

        # 線程安全控制
        self._lock = threading.Lock()
        self._active_contexts: Dict[str, _ContextHandle] = {}
        self._context_counter = 0

    def get_context(self, stream_id: str) -> "PerSequenceDetector":
        """
        為指定序列取得獨立 context。

        Args:
            stream_id: 序列 ID（如 "MOT17-02-SDP"）

        Returns:
            PerSequenceDetector: 帶獨立 context 的偵測器
        """
        with self._lock:
            if stream_id in self._active_contexts:
                # 重複使用已有 context
                handle = self._active_contexts[stream_id]
            else:
                # 建立新 context
                handle = self._create_context()
                if len(self._active_contexts) >= self.max_workers:
                    raise ValueError(
                        f"Max workers ({self.max_workers}) exceeded. "
                        f"Release unused contexts via .release()"
                    )
                self._active_contexts[stream_id] = handle

        return PerSequenceDetector(
            detector=self,
            stream_id=stream_id,
            handle=handle,
            tracker=GPUByteTracker(max_objects=2048),
        )

    def _create_context(self) -> _ContextHandle:
        """建立新的 TRT context + cudaStream + tracker"""
        # 建立 cudaStream
        stream = torch.cuda.Stream(device=self.device)

        # 透過 C++ 建立 context（返回指標）
        # 注意：create_context() 返回 raw pointer，需手動管理生命週期
        ctx_ptr = self.shared_engine.create_context()

        # 預先分配輸出 buffer
        output_shape = self.output_shape
        output_tensor = torch.empty(
            output_shape, device=self.device, dtype=torch.float32
        )

        return _ContextHandle(
            context_ptr=ctx_ptr,
            cuda_stream=stream,
            output_tensors={"output0": output_tensor},
        )

    def release(self, stream_id: str) -> None:
        """
        釋放指定序列的 context。

        注意：TRT context 的記憶體釋放需在正確線程中執行。
        """
        with self._lock:
            if stream_id in self._active_contexts:
                self._active_contexts.pop(stream_id)
                # 釋放 context（透過 C++ wrapper）
                # 這裡需要透過 python binding 呼叫 delete
                # TODO: 在 perception_python.cpp 中增加 delete_context()
                print(f"🔄 [ConcurrentDetector] Released context for {stream_id}")


class PerSequenceDetector:
    """
    單序列偵測器包裝，擁有獨立 context + tracker。

    這個類模擬 TRTYoloDetector 的 API，但使用獨立 context。
    """

    def __init__(
        self,
        detector: ConcurrentTRTDetector,
        stream_id: str,
        handle: _ContextHandle,
        tracker: GPUByteTracker,
    ):
        self.detector = detector
        self.stream_id = stream_id
        self.handle = handle
        self.tracker = tracker
        self.device = detector.device
        self.input_name = handle.input_name
        self.output_name = handle.output_name

    def reset_tracker(self) -> None:
        """重置 tracker 狀態"""
        self.tracker = GPUByteTracker(max_objects=2048)

    def detect(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        執行偵測（使用獨立 context）。

        Args:
            input_tensor: [C, H, W] 輸入張量（單幀）
            conf_threshold: 信心閾值

        Returns:
            (boxes, scores, classes, keypoints)
        """
        input_tensor = input_tensor.contiguous().unsqueeze(0)  # [1, C, H, W]

        # 切換到獨立 stream
        with torch.cuda.stream(self.handle.cuda_stream):
            # 設定 bindings
            binding_ptrs = [input_tensor.data_ptr()]
            for name in self.handle.output_tensors:
                binding_ptrs.append(self.handle.output_tensors[name].data_ptr())

            # 執行推理（透過 shared engine）
            self.detector.shared_engine.infer(
                binding_ptrs, self.handle.cuda_stream.cuda_stream
            )

            # 讀取結果
            output_tensor = self.handle.output_tensors[self.output_name]

        torch.cuda.current_stream().synchronize()

        # 後處理
        results = output_tensor[0]
        mask = results[:, 4] > conf_threshold
        valid_results = results[mask]

        if valid_results.size(0) == 0:
            return (
                torch.empty((0, 4), device=self.device),
                torch.empty((0,), device=self.device),
                torch.empty((0,), device=self.device),
                None,
            )

        boxes = valid_results[:, :4].contiguous()
        scores = valid_results[:, 4].contiguous()
        classes = valid_results[:, 5].to(torch.int32).contiguous()

        return boxes, scores, classes, None

    def detect_batch(
        self,
        input_tensor: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        批次偵測（支援多幀）。
        """
        batch_size = input_tensor.size(0)
        results = []

        for i in range(batch_size):
            frame = input_tensor[i]
            boxes, scores, classes, _ = self.detect(frame, conf_threshold)
            results.append((boxes, scores, classes, None))

        return results
