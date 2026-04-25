import tensorrt as trt
import torch
from typing import Dict, Tuple, Optional, cast, List
from perception.tracking import GPUByteTracker


class TRTYoloDetector:
    """
    YOLO26 極速 TensorRT 偵測器
    優先使用 C++ 核心引擎以獲得最佳效能與最低抖動。
    """

    def __init__(
        self,
        engine_path: str = "models/yolo/yolo26s_batch4.engine",
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
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        self.output_tensors: Dict[str, torch.Tensor] = {}
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

    def _empty_result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return (
            torch.empty((0, 4), device=self.device),
            torch.empty((0,), device=self.device),
            torch.empty((0,), device=self.device),
            None,
        )

    def infer_raw_batch(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        執行 TensorRT 推理並回傳所有輸出張量。

        Detection-only YOLO26 engines expose only ``output0``. YOLOE segmentation
        engines expose ``output0`` detections plus ``output1`` mask prototypes.
        """
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.contiguous()
        
        # 1. 設定動態輸入 Shape
        self.context.set_input_shape(self.input_name, input_tensor.shape)

        # 2. 準備所有輸出空間；動態 batch engine 會根據 input shape 解析輸出 shape。
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                shape = self._resolve_output_shape(self.engine.get_tensor_shape(name), batch_size)

            current = self.output_tensors.get(name)
            if current is None or tuple(current.shape) != shape:
                self.output_tensors[name] = torch.empty(
                    shape, device=self.device, dtype=torch.float32
                )

        # 3. 綁定並執行所有輸入/輸出
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Launch on the caller's current CUDA stream and let downstream GPU ops
        # establish ordering naturally. Callers that need wall-clock timings or
        # CPU-visible results should synchronize explicitly at their boundary.
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return self.output_tensors

    def detect_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行 TensorRT 推理並直接回傳原始輸出張量 [Batch, 300, 6]。
        避免 Python 列表解包與迴圈開銷。
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

        # 4. 解包結果 (Scattering)
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
            if "embeddings" in outputs:
                extra = outputs["embeddings"][i][mask].contiguous()
            else:
                extra = valid_results[:, 6:].contiguous() if valid_results.size(1) > 6 else None

            # 這裡注意：多路模式下 Tracker 應該是按路數實例化的，
            # 但目前為了 Phase 1 展示，我們暫用全域 Tracker 或預留擴展。
            # 生產環境下，此處應調用對應 stream_id 的 tracker.update
            batch_results.append((boxes, scores, classes, extra))
            
        return batch_results

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
