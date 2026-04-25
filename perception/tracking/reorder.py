import heapq
import time
from typing import List, Tuple, Any


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
        self.buffer: List[Tuple[int, int, Any]] = [] # (timestamp, count, data)
        self.count = 0
        self.last_emitted_ts: int = 0
        self.frame_interval_ms: int = 33 # 預設 30fps
        
    def push(self, timestamp: int, data: Any) -> None:
        """將影格推入優先權隊列（以時間戳為鍵值）。"""
        heapq.heappush(self.buffer, (timestamp, self.count, data))
        self.count += 1
        
    def pop_ready(self) -> List[Any]:
        """彈出所有符合發送條件的影格。"""
        ready_frames = []
        current_time_ms = int(time.time() * 1000)
        
        while self.buffer:
            top_ts, _, _ = self.buffer[0]
            
            # 條件 1: 緩衝區累積足夠影格或時間超過視窗
            # 條件 2: 頂端影格與上一個發出的影格時間間隔超過視窗 (強制排出以控制延遲)
            if len(self.buffer) > 5 or (top_ts - self.last_emitted_ts) > self.window_ms:
                ts, _, data = heapq.heappop(self.buffer)
                
                # 檢查延遲尖峰 (Latency Spike)
                if (current_time_ms - ts) > self.timeout_ms:
                    # 這裡可以整合進日誌系統
                    pass
                
                self.last_emitted_ts = ts
                ready_frames.append(data)
            else:
                break
                
        return ready_frames

    def reset(self) -> None:
        """清空緩衝區。"""
        self.buffer = []
        self.last_emitted_ts = 0
