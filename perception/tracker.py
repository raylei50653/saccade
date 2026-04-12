import torch
import torchvision.ops as ops
from typing import Dict, Any, List, Optional, Tuple
import math

class SmartTracker:
    """
    Saccade 智能追蹤與特徵排程器 (Phase 3)
    
    1. 維護物件的歷史狀態 (BBox, Velocity)。
    2. 基於規則 (New ID, IOU Change, Velocity Change) 篩選需要更新特徵的物件。
    3. 在獨立的 CUDA Stream 中非同步觸發特徵提取，不阻塞主 YOLO 推理。
    """
    def __init__(self, iou_threshold: float = 0.7, velocity_angle_threshold: float = 45.0):
        self.iou_threshold = iou_threshold
        # 將角度閾值轉為 Cosine 相似度閾值
        self.cos_threshold = math.cos(math.radians(velocity_angle_threshold))
        
        # 狀態儲存：obj_id -> { "last_box": tensor, "velocity": tensor, "last_extracted_box": tensor }
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # 建立獨立的 CUDA Stream 供特徵提取使用
        self.extraction_stream = torch.cuda.Stream()
        
    def _calculate_center(self, box: torch.Tensor) -> torch.Tensor:
        """計算 BBox 中心點 (x, y)"""
        return torch.tensor([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], device=box.device)

    def _should_extract_features(self, obj_id: int, current_box: torch.Tensor) -> bool:
        """
        評估是否需要觸發特徵提取 (Event Hooks)
        """
        # Condition A: 全新的 Object ID 首度出現
        if obj_id not in self.states:
            return True
            
        state = self.states[obj_id]
        
        # Condition C: Bounding Box IOU 變化率超過設定閾值
        last_extracted_box = state.get("last_extracted_box")
        if last_extracted_box is not None:
            # 計算 IoU
            iou = ops.box_iou(current_box.unsqueeze(0), last_extracted_box.unsqueeze(0))[0, 0].item()
            if iou < self.iou_threshold:
                return True
                
        # Condition B: 移動向量 (Velocity Vector) 發生劇烈改變
        last_box = state["last_box"]
        current_center = self._calculate_center(current_box)
        last_center = self._calculate_center(last_box)
        
        current_velocity = current_center - last_center
        last_velocity = state.get("velocity")
        
        if last_velocity is not None:
            # 計算速度向量的 Cosine 相似度 (判斷方向改變)
            norm_curr = torch.norm(current_velocity)
            norm_last = torch.norm(last_velocity)
            
            # 若移動距離極小 (雜訊)，不視為方向改變
            if norm_curr > 2.0 and norm_last > 2.0:
                cos_sim = torch.dot(current_velocity, last_velocity) / (norm_curr * norm_last)
                if cos_sim.item() < self.cos_threshold:
                    return True
                    
        return False

    def update_and_filter(self, obj_ids: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新追蹤狀態，並過濾出需要提取特徵的物件。
        
        :param obj_ids: YOLO 輸出的追蹤 ID [N]
        :param boxes: YOLO 輸出的 Bounding Boxes [N, 4]
        :return: (需要提取的 obj_ids, 對應的 boxes)
        """
        extract_indices = []
        
        for i, (obj_id_tensor, box) in enumerate(zip(obj_ids, boxes)):
            obj_id = int(obj_id_tensor.item())
            
            # 判斷是否觸發提取
            should_extract = self._should_extract_features(obj_id, box)
            
            if should_extract:
                extract_indices.append(i)
                
            # 更新狀態
            if obj_id not in self.states:
                self.states[obj_id] = {}
                
            current_center = self._calculate_center(box)
            if "last_box" in self.states[obj_id]:
                last_center = self._calculate_center(self.states[obj_id]["last_box"])
                self.states[obj_id]["velocity"] = current_center - last_center
                
            self.states[obj_id]["last_box"] = box
            
            if should_extract:
                self.states[obj_id]["last_extracted_box"] = box
                
        if not extract_indices:
            return torch.empty(0, device=boxes.device), torch.empty((0, 4), device=boxes.device)
            
        indices_tensor = torch.tensor(extract_indices, device=boxes.device, dtype=torch.long)
        return obj_ids[indices_tensor], boxes[indices_tensor]

    def async_extract_features(self, frame_tensor: torch.Tensor, extract_boxes: torch.Tensor, 
                               cropper, extractor) -> Optional[torch.Tensor]:
        """
        在獨立的 CUDA Stream 中執行裁切與特徵提取
        """
        if extract_boxes.numel() == 0:
            return None
            
        # 切換到背景 Stream
        with torch.cuda.stream(self.extraction_stream):
            # 1. Zero-Copy 裁切
            crops = cropper.process(frame_tensor, extract_boxes)
            
            # 2. TensorRT 特徵提取
            features = extractor.extract(crops)
            
        # 注意：我們沒有在這裡呼叫 torch.cuda.synchronize()。
        # 因此主迴圈的 YOLO 推理不會被阻塞。
        # 當後續需要使用這批特徵寫入 DB 時，才需要確保這個 Stream 執行完畢。
        return features

if __name__ == "__main__":
    print("🚀 Testing SmartTracker Logic...")
    tracker = SmartTracker()
    
    # 模擬第 1 幀：兩個新物件 (應觸發 A)
    ids1 = torch.tensor([1, 2], device="cuda")
    boxes1 = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], device="cuda", dtype=torch.float32)
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes1)
    print(f"Frame 1 - Extracted IDs (Expect [1, 2]): {ext_ids.tolist()}")
    
    # 模擬第 2 幀：物件輕微移動 (不應觸發)
    boxes2 = torch.tensor([[12, 12, 52, 52], [102, 102, 152, 152]], device="cuda", dtype=torch.float32)
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes2)
    print(f"Frame 2 - Extracted IDs (Expect []): {ext_ids.tolist()}")
    
    # 模擬第 3 幀：物件 1 大幅形變 (觸發 C)，物件 2 輕微移動
    boxes3 = torch.tensor([[10, 10, 100, 100], [105, 105, 155, 155]], device="cuda", dtype=torch.float32)
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes3)
    print(f"Frame 3 - Extracted IDs (Expect [1]): {ext_ids.tolist()}")
    
    # 模擬第 4 幀：物件 2 突然改變方向 (折返跑，觸發 B)
    boxes4 = torch.tensor([[15, 15, 105, 105], [90, 90, 140, 140]], device="cuda", dtype=torch.float32)
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes4)
    print(f"Frame 4 - Extracted IDs (Expect [2]): {ext_ids.tolist()}")
