import tensorrt as trt
import torch
import sys
from typing import Tuple, Optional, List

# 🚀 嘗試導入 C++ 加速擴展
try:
    if "/app/build" not in sys.path:
        sys.path.append("/app/build")
    from saccade_perception_ext import TRTEngine as CppTRTEngine

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False

from saccade_tracking_ext import GPUByteTracker


class TRTYoloDetector:
    """
    YOLO26 極速 TensorRT 偵測器
    優先使用 C++ 核心引擎以獲得最佳效能與最低抖動。
    """

    def __init__(
        self,
        engine_path: str = "models/yolo/yolo26n.engine",
        device: str = "cuda:0",
    ):
        self.device = device
        self.use_cpp = HAS_CPP_EXT

        if self.use_cpp:
            print(f"🚀 [TRT] Loading C++ Optimized Engine from {engine_path}...")
            self.cpp_engine = CppTRTEngine(engine_path)
            self.input_shape = self.cpp_engine.get_tensor_shape("images")
            self.output_shape = self.cpp_engine.get_tensor_shape("output0")
            self.input_name = "images"
            self.output_name = "output0"
        else:
            print(
                f"⚠️ [TRT] C++ Extension not found, using Python Native API for {engine_path}"
            )
            self.logger = trt.Logger(trt.Logger.ERROR)
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name
            self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # 💡 偵測模型是否支援動態 Batch
        self.is_dynamic = self.output_shape[0] == -1

        # 建立專用的輸出 Tensor 緩衝區
        self.output_tensor = torch.empty(
            tuple(abs(s) if s != -1 else 1 for s in self.output_shape),
            device=self.device,
            dtype=torch.float32,
        )

        # 初始化 GPU Tracker (Zero-Sync)
        self.tracker = GPUByteTracker(max_objects=2048)

        print(
            f"✅ Native YOLO Detector Ready. Engine: {'C++' if self.use_cpp else 'Python'}, Dynamic: {self.is_dynamic}"
        )

    def _empty_result(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return (
            torch.empty((0, 4), device=self.device),
            torch.empty((0,), device=self.device),
            torch.empty((0,), device=self.device),
            None,
        )

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
        current_stream = stream if stream is not None else torch.cuda.current_stream()

        if self.is_dynamic:
            # 🚀 動態模式：一次性推論
            input_tensor = input_tensor.contiguous()

            if self.output_tensor.size(0) < batch_size:
                output_shape = list(self.output_shape)
                output_shape[0] = batch_size
                self.output_tensor = torch.empty(
                    tuple(output_shape), device=self.device, dtype=torch.float32
                )

            if self.use_cpp:
                # 使用 C++ 推論 (直接傳遞資料指標與 Stream 指標)
                self.cpp_engine.infer(
                    [input_tensor.data_ptr(), self.output_tensor.data_ptr()],
                    current_stream.cuda_stream,
                )
            else:
                self.context.set_input_shape(self.input_name, input_tensor.shape)
                self.context.set_tensor_address(
                    self.input_name, input_tensor.data_ptr()
                )
                self.context.set_tensor_address(
                    self.output_name, self.output_tensor.data_ptr()
                )
                self.context.execute_async_v3(current_stream.cuda_stream)

            # 解包
            return [
                self._postprocess(self.output_tensor[i], conf_threshold)
                for i in range(batch_size)
            ]
        else:
            # 🐢 靜態模式
            results = []
            for i in range(batch_size):
                single_input = input_tensor[i].unsqueeze(0).contiguous()
                if self.use_cpp:
                    self.cpp_engine.infer(
                        [single_input.data_ptr(), self.output_tensor.data_ptr()],
                        current_stream.cuda_stream,
                    )
                else:
                    self.context.set_tensor_address(
                        self.input_name, single_input.data_ptr()
                    )
                    self.context.set_tensor_address(
                        self.output_name, self.output_tensor.data_ptr()
                    )
                    self.context.execute_async_v3(current_stream.cuda_stream)
                results.append(
                    self._postprocess(self.output_tensor.clone(), conf_threshold)
                )
            return results

    def _postprocess(
        self, results: torch.Tensor, conf_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """結果過濾邏輯"""
        # 如果是 [1, 300, 6] 或 [300, 6]
        if results.dim() == 3:
            results = results.squeeze(0)

        mask = results[:, 4] > conf_threshold
        valid_results = results[mask]

        if valid_results.size(0) == 0:
            return self._empty_result()

        boxes = valid_results[:, :4].contiguous()
        scores = valid_results[:, 4].contiguous()
        classes = valid_results[:, 5].to(torch.int32).contiguous()
        return (boxes, scores, classes, None)

    def detect(
        self, input_tensor: torch.Tensor, conf_threshold: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """單路相容性接口"""
        results = self.detect_batch(input_tensor, conf_threshold)
        return results[0] if results else self._empty_result()


if __name__ == "__main__":
    # 簡單測試
    print("🚀 Testing TRTYoloDetector...")
    detector = TRTYoloDetector()
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float32)

    # 預熱
    _b, _s, _c, _i = detector.detect(dummy_input)
    torch.cuda.synchronize()

    import time

    start = time.perf_counter()
    for i in range(100):
        boxes, scores, classes, ids = detector.detect(dummy_input)
    torch.cuda.synchronize()

    print(
        f"⚡ Average Native TRT Latency: {(time.perf_counter() - start):.2f} ms (for 100 iterations)"
    )
    print(f"✅ Found {boxes.size(0)} objects in dummy frame.")
