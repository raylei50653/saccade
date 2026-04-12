import torch
import torchvision.ops as ops
from typing import Tuple

class ZeroCopyCropper:
    """
    Saccade 零拷貝裁切器 (Phase 1)
    
    直接在 GPU 顯存中接收原始高解析度 Frame Tensor 與 YOLO Bounding Boxes，
    使用 torchvision.ops.roi_align 進行批次裁切與縮放，產出 CLIP/SigLIP 相容的 Tensor。
    """
    def __init__(self, output_size: Tuple[int, int] = (512, 512)):
        self.output_size = output_size

    def process(self, frame_tensor: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        執行零拷貝裁切與縮放
        
        :param frame_tensor: 原始或縮放後的 GPU 影格，形狀 [1, C, H, W]，型別 Float (0~1)
        :param boxes: YOLO 輸出的 Bounding Boxes，形狀 [N, 4]，格式為 (x1, y1, x2, y2)，相對於 frame_tensor 的絕對座標
        :return: 裁切並縮放後的 Tensor，形狀 [N, C, output_size[0], output_size[1]]
        """
        # 若無目標，回傳空 Tensor 以免報錯
        if boxes is None or boxes.numel() == 0:
            return torch.empty((0, frame_tensor.shape[1], *self.output_size), device=frame_tensor.device)
            
        # 確保 boxes 與 frame_tensor 在相同的裝置上
        boxes = boxes.to(frame_tensor.device)
        
        # roi_align 需要的 boxes 格式為 [N, 5]，第一欄是 batch_index。
        # 由於我們每次只處理單張圖片 (Batch=1)，所以 index 全為 0。
        batch_indices = torch.zeros((boxes.shape[0], 1), device=boxes.device, dtype=boxes.dtype)
        rois = torch.cat([batch_indices, boxes], dim=1)
        
        # 使用 RoI Align 進行硬體加速的裁切與對齊
        # spatial_scale=1.0 表示 boxes 的座標比例與 frame_tensor 的像素比例為 1:1
        # aligned=True 確保像素採樣中心對齊，提升邊緣精準度
        crops = ops.roi_align(
            input=frame_tensor, 
            boxes=rois, 
            output_size=self.output_size,
            spatial_scale=1.0, 
            aligned=True
        )
        
        return crops

if __name__ == "__main__":
    # 微秒級延遲測試 (Dry Run)
    import time
    
    print("🚀 Testing ZeroCopyCropper Initialization...")
    cropper = ZeroCopyCropper(output_size=(224, 224))
    
    # 模擬 1080p 影格 (1, 3, 1080, 1920) 在 GPU 上
    dummy_frame = torch.rand((1, 3, 1080, 1920), device="cuda", dtype=torch.float32)
    
    # 模擬 YOLO 抓到 5 個目標的 BBox (x1, y1, x2, y2)
    dummy_boxes = torch.tensor([
        [100, 150, 300, 450],
        [500, 200, 800, 900],
        [50, 50, 100, 100],
        [1000, 500, 1200, 800],
        [1500, 100, 1800, 600]
    ], device="cuda", dtype=torch.float32)
    
    # 預熱 CUDA
    _ = cropper.process(dummy_frame, dummy_boxes)
    torch.cuda.synchronize()
    
    # 測速
    start = time.perf_counter()
    for _ in range(100):
        out_crops = cropper.process(dummy_frame, dummy_boxes)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / 100 * 1000 * 1000 # 轉換為微秒 (µs)
    
    print(f"✅ Cropped Tensor Shape: {out_crops.shape} (N, C, H, W)")
    print(f"⚡ Average Latency for 5 objects: {latency:.2f} µs")
