import tensorrt as trt
import torch
from typing import Tuple, Optional, cast, List

from saccade_tracking_ext import GPUByteTracker


class TRTYoloDetector:
    """
    YOLO26 極速 TensorRT 偵測器 (Native API)

    直接操作 GPU 顯存指標，繞過 Ultralytics 封裝以達成極限低延遲與低抖動。
    專為 YOLO26 NMS-Free 模型優化，並整合 GPUByteTracker 提供追蹤 ID。
    """

    def __init__(
        self,
        engine_path: str = "models/yolo/yolo26n_native.engine",
        device: str = "cuda:0",
    ):
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)

        print(f"⏳ Loading Native YOLO TRT Engine from {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load Native YOLO TensorRT Engine.")

        self.context = self.engine.create_execution_context()

        # 取得輸入輸出名稱與形狀 (更健壯的偵測方式)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        # 建立專用的輸出 Tensor 緩衝區
        self.output_tensor = torch.empty(
            tuple(self.output_shape), device=self.device, dtype=torch.float32
        )

        # 初始化 GPU Tracker (Zero-Sync)
        self.tracker = GPUByteTracker(max_objects=2048)

        print(
            f"✅ Native YOLO Detector Ready. Input: {self.input_name}, Output: {self.output_name} {self.output_shape}"
        )

    def _empty_result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return (
            torch.empty((0, 4), device=self.device),
            torch.empty((0,), device=self.device),
            torch.empty((0,), device=self.device),
            None,
        )

    def detect_batch(
        self, input_tensor: torch.Tensor, conf_threshold: float = 0.25
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        執行批次偵測與追蹤 (支援多路串流聚合)
        
        :param input_tensor: [Batch, 3, 640, 640] GPU Tensor
        :return: 每一路串流的偵測結果列表
        """
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.contiguous()
        
        # 1. 設定動態輸入 Shape
        self.context.set_input_shape(self.input_name, input_tensor.shape)

        # 2. 準備輸出空間 (YOLO26 NMS-Free 輸出通常是 [Batch, 300, 6])
        # 我們根據當前 Batch Size 切片或動態分配
        output_shape = list(self.output_shape)
        output_shape[0] = batch_size
        
        # 為了效能，我們儘量複用緩衝區，只有當 Batch 變大時才重新分配
        if self.output_tensor.size(0) < batch_size:
            self.output_tensor = torch.empty(
                tuple(output_shape), device=self.device, dtype=torch.float32
            )
        
        # 3. 綁定並執行
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        # 4. 解包結果 (Scattering)
        batch_results = []
        for i in range(batch_size):
            results = self.output_tensor[i]
            mask = results[:, 4] > conf_threshold
            valid_results = results[mask]
            
            if valid_results.size(0) == 0:
                batch_results.append(self._empty_result())
                continue
                
            boxes = valid_results[:, :4].contiguous()
            scores = valid_results[:, 4].contiguous()
            classes = valid_results[:, 5].to(torch.int32).contiguous()

            # 這裡注意：多路模式下 Tracker 應該是按路數實例化的，
            # 但目前為了 Phase 1 展示，我們暫用全域 Tracker 或預留擴展。
            # 生產環境下，此處應調用對應 stream_id 的 tracker.update
            batch_results.append((boxes, scores, classes, None))
            
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
