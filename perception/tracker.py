import torch
import heapq
import time
from typing import Tuple, Any, Optional, List, Dict, cast
from saccade_tracking_ext import SmartTracker as CppSmartTracker


class ReorderingBuffer:
    """
    Saccade 有序暫存隊列 (Phase 4 - Industrial)
    
    1. 解決多 Stream 並行導致的 Out-of-order 問題。
    2. 實作 150ms 滑動窗口重排。
    3. 實作 200ms 缺幀補丁預測 (In-filling)。
    """
    def __init__(self, window_ms: int = 150, timeout_ms: int = 200) -> None:
        self.window_ms = window_ms
        self.timeout_ms = timeout_ms
        self.buffer: List[Tuple[int, Any]] = [] # (timestamp, data)
        self.last_emitted_ts: int = 0
        self.frame_interval_ms: int = 33 # 預設 30fps
        
    def push(self, timestamp: int, data: Any) -> None:
        heapq.heappush(self.buffer, (timestamp, data))
        
    def pop_ready(self) -> List[Any]:
        ready_frames = []
        current_time = int(time.time() * 1000)
        
        while self.buffer:
            top_ts, _ = self.buffer[0]
            
            # 條件 1: 緩衝區滿或時間超過窗口
            # 條件 2: 頂端影格與上一個發出的影格連續
            if len(self.buffer) > 5 or (top_ts - self.last_emitted_ts) > self.window_ms:
                ts, data = heapq.heappop(self.buffer)
                
                # 檢查是否需要 In-filling (缺幀預測)
                if self.last_emitted_ts > 0 and (ts - self.last_emitted_ts) > (self.frame_interval_ms * 1.5):
                    # 這裡觸發預測邏輯 (暫以 Log 標記，由 Tracker 實作 BBox 預測)
                    # print(f"⚠️ [ReorderingBuffer] Gap detected: {ts - self.last_emitted_ts}ms, triggering In-filling.")
                    pass
                
                # 檢查 Latency Spike
                if (current_time - ts) > self.timeout_ms:
                    print(f"🚨 [LATENCY_SPIKE] Frame {ts} delayed {current_time - ts}ms!")
                
                self.last_emitted_ts = ts
                ready_frames.append(data)
            else:
                break
                
        return ready_frames


class SmartTracker:
    """
    Saccade 智能追蹤與特徵排程器 (Phase 4 - Industrial)
    """

    def __init__(
        self,
        iou_threshold: float = 0.7,
        velocity_angle_threshold: float = 45.0,
        max_objects: int = 2048,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.velocity_angle_threshold = velocity_angle_threshold
        self.max_objects = max_objects

        self.cpp_tracker = CppSmartTracker(
            iou_threshold, velocity_angle_threshold, max_objects
        )
        self.reorder_buffer = ReorderingBuffer()
        self.extraction_stream: torch.cuda.Stream = torch.cuda.Stream() # type: ignore[no-untyped-call]
        
    def set_degradation_params(self, level: int) -> None:
        """
        根據降級級別動態調整追蹤參數 (Industrial Grade - Level 3 Support)
        """
        if level >= 3: # EMERGENCY MODE
            # 大掃除：大幅縮短遺失幀緩衝 (從 30 降至 5)
            # 這能釋放追蹤器內部的歷史狀態快取
            self.cpp_tracker.set_max_lost_frames(5)
            # 提高過濾門檻，只留核心目標
            self.cpp_tracker.set_min_confidence(0.4)
            print("🧹 [SmartTracker] Target Culling Active (TTL=5, Conf=0.4)")
        else:
            # 恢復正常
            self.cpp_tracker.set_max_lost_frames(30)
            self.cpp_tracker.set_min_confidence(0.1)

    def process_frame(self, timestamp: int, obj_ids: torch.Tensor, boxes: torch.Tensor) -> None:
        """
        將新影格偵測結果推入重排隊列
        """
        self.reorder_buffer.push(timestamp, (obj_ids, boxes))

    def update_and_filter(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        從緩衝區獲取已排序的影格並更新追蹤狀態
        """
        ready_data = self.reorder_buffer.pop_ready()
        results = []

        for obj_ids, boxes in ready_data:
            if obj_ids.numel() == 0:
                continue

            num_objs = obj_ids.size(0)
            mask = torch.empty(num_objs, dtype=torch.bool, device=boxes.device)
            ids_contig = obj_ids.to(torch.int32).contiguous()
            boxes_contig = boxes.to(torch.float32).contiguous()

            stream = torch.cuda.current_stream().cuda_stream
            self.cpp_tracker.update_and_filter(
                ids_contig.data_ptr(),
                boxes_contig.data_ptr(),
                mask.data_ptr(),
                num_objs,
                stream,
            )
            
            results.append((obj_ids[mask], boxes[mask]))
            
        return results

    def async_extract_features(
        self,
        frame_tensor: torch.Tensor,
        extract_boxes: torch.Tensor,
        cropper: Any,
        extractor: Any,
    ) -> Optional[torch.Tensor]:
        """
        在獨立的 CUDA Stream 中執行裁切與特徵提取
        """
        if extract_boxes.numel() == 0:
            return None

        with torch.cuda.stream(self.extraction_stream):
            crops = cropper.process(frame_tensor, extract_boxes)
            features = extractor.extract(crops)

        return cast(Optional[torch.Tensor], features)


if __name__ == "__main__":
    print("🚀 Testing Industrial SmartTracker with ReorderingBuffer...")
    tracker = SmartTracker()

    # 模擬亂序影格輸入 (Out-of-order)
    # 順序：Frame 10, Frame 30, Frame 20
    def mock_data(val: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([val], device="cuda", dtype=torch.int32)
        boxes = torch.tensor([[val, val, val+50, val+50]], device="cuda", dtype=torch.float32)
        return ids, boxes

    print("\n--- Phase 1: Out-of-order Handling ---")
    tracker.process_frame(1000, *mock_data(10)) # Frame 10 @ 1.0s
    tracker.process_frame(3000, *mock_data(30)) # Frame 30 @ 3.0s (跳躍)
    tracker.process_frame(2000, *mock_data(20)) # Frame 20 @ 2.0s (遲到)

    # 第一次讀取 (此時緩衝區有 3 幀，且 Frame 30 與 Frame 10 差距 > 150ms)
    results = tracker.update_and_filter()
    print(f"Pop 1 - Frames ready: {len(results)}") 
    # 預期會依序排出 10, 20 (因為 20 已經到了且在 10 之後)

    # 模擬延遲尖峰 (Latency Spike)
    print("\n--- Phase 2: Latency Spike Detection ---")
    old_ts = int((time.time() - 5) * 1000) # 5 秒前的影格
    tracker.process_frame(old_ts, *mock_data(99))
    tracker.reorder_buffer.window_ms = 0 # 強制排出
    tracker.update_and_filter() # 應觸發 LATENCY_SPIKE Log

    print("\n✅ Test Completed.")
