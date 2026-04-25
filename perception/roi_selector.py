import torch
from typing import List, Tuple, Optional

class ROISelector:
    """
    Saccade 動態關注區選擇器 (Phase 3)
    基於現有軌跡密度，決定下一影格的「高解析度注視區 (Fixation)」。
    """
    def __init__(self, grid_size: int = 4, frame_width: int = 1920, frame_height: int = 1080) -> None:
        self.grid_size = grid_size
        self.w = frame_width
        self.h = frame_height
        self.last_roi_center = (frame_width // 2, frame_height // 2)
        
    def select_best_roi(self, tracks: torch.Tensor, roi_size: int = 640, smoothing: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
        """
        根據軌跡熱度選擇一個最佳的 ROI 區域 (x1, y1, x2, y2)
        :param tracks: 目前軌跡的 BBoxes [N, 4] (x1, y1, x2, y2)
        :param roi_size: 希望獲取的 ROI 解析度
        :param smoothing: ROI 移動平滑度 (0.0 ~ 1.0, 越小移動越快)
        """
        if tracks is None or tracks.numel() == 0:
            return None
        
        # 1. 計算所有軌跡的中心點
        cx = (tracks[:, 0] + tracks[:, 2]) / 2.0
        cy = (tracks[:, 1] + tracks[:, 3]) / 2.0
        
        # 2. 找出重心
        target_cx = torch.mean(cx).item()
        target_cy = torch.mean(cy).item()
        
        # 3. 緩動平滑 (Exponential Moving Average)
        curr_cx = self.last_roi_center[0] * smoothing + target_cx * (1.0 - smoothing)
        curr_cy = self.last_roi_center[1] * smoothing + target_cy * (1.0 - smoothing)
        self.last_roi_center = (curr_cx, curr_cy)
        
        # 4. 計算邊界，確保不超出範圍
        half_roi = roi_size / 2.0
        x1 = int(max(0, min(self.w - roi_size, curr_cx - half_roi)))
        y1 = int(max(0, min(self.h - roi_size, curr_cy - half_roi)))
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        return (x1, y1, x2, y2)

if __name__ == "__main__":
    # 簡單測試
    selector = ROISelector()
    dummy_tracks = torch.tensor([
        [100, 100, 200, 200],
        [150, 150, 250, 250],
        [50, 50, 150, 150]
    ], device="cuda", dtype=torch.float32)
    
    roi = selector.select_best_roi(dummy_tracks)
    print(f"🎯 Recommended ROI: {roi}")
