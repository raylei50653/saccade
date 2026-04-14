import os
import tensorrt as trt
import torch
import numpy as np
from typing import List, Tuple, Optional, cast

from saccade_tracking_ext import GPUByteTracker

class TRTYoloDetector:
    """
    YOLO26 極速 TensorRT 偵測器 (Native API)
    
    直接操作 GPU 顯存指標，繞過 Ultralytics 封裝以達成極限低延遲與低抖動。
    專為 YOLO26 NMS-Free 模型優化，並整合 GPUByteTracker 提供追蹤 ID。
    """
    def __init__(self, engine_path: str = "models/yolo/yolo26n_native.engine", device: str = "cuda:0"):
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        print(f"⏳ Loading Native YOLO TRT Engine from {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError("Failed to load Native YOLO TensorRT Engine.")
            
        self.context = self.engine.create_execution_context()
        
        # 取得輸入輸出名稱與形狀
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # 預先分配輸出 Tensor [1, 300, 6]
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        self.output_tensor = torch.empty(tuple(self.output_shape), device=self.device, dtype=torch.float32)
        
        # 初始化 GPU Tracker (Zero-Sync)
        self.tracker = GPUByteTracker(max_objects=2048)
        
        print(f"✅ Native YOLO Detector Ready with GPUByteTracker. Output Shape: {self.output_shape}")

    def detect(self, input_tensor: torch.Tensor, conf_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        執行偵測與追蹤 (零拷貝，零同步)
        """
        # 1. 確保連續
        input_tensor = input_tensor.contiguous()
        
        # 2. 綁定記憶體指標
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
        
        # 3. 異步觸發推理 (使用當前 Stream)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        
        # 4. YOLO26 NMS-Free 結果處理
        results = self.output_tensor[0] # [300, 6]
        
        # 過濾置信度
        mask = results[:, 4] > conf_threshold
        valid_results = results[mask]
        
        if valid_results.size(0) == 0:
            return (torch.empty((0, 4), device=self.device), 
                    torch.empty((0,), device=self.device), 
                    torch.empty((0,), device=self.device),
                    cast(Optional[torch.Tensor], None))
        
        # 5. 更新追蹤 (GPUByteTracker - Zero Sync)
        boxes = valid_results[:, :4].contiguous()
        scores = valid_results[:, 4].contiguous()
        classes = valid_results[:, 5].to(torch.int32).contiguous()
        
        tracks = self.tracker.update(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes.data_ptr(),
            boxes.size(0),
            stream
        )
        
        if tracks:
            t_data = [[t.x1, t.y1, t.x2, t.y2, t.score, t.class_id, t.obj_id] for t in tracks]
            tracks_tensor = torch.tensor(t_data, device=self.device, dtype=torch.float32)
            return tracks_tensor[:, :4], tracks_tensor[:, 4], tracks_tensor[:, 5], tracks_tensor[:, 6]
        else:
            return boxes, scores, valid_results[:, 5], None

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
    
    print(f"⚡ Average Native TRT Latency: {(time.perf_counter()-start):.2f} ms (for 100 iterations)")
    print(f"✅ Found {boxes.size(0)} objects in dummy frame.")

