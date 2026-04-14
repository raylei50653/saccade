import torch
from typing import Tuple, Any, Optional, cast

from saccade_tracking_ext import SmartTracker as CppSmartTracker


class SmartTracker:
    """
    Saccade 智能追蹤與特徵排程器 (Phase 3 - C++ Native)

    1. 維護物件的歷史狀態 (BBox, Velocity)，使用 CUDA Kernel 加速處理。
    2. 基於規則 (New ID, IOU Change, Velocity Change) 篩選需要更新特徵的物件。
    3. 在獨立的 CUDA Stream 中非同步觸發特徵提取，不阻塞主 YOLO 推理。
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
        self.extraction_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]

    def update_and_filter(
        self, obj_ids: torch.Tensor, boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根據物件 ID 和 BBox 判斷是否需要提取特徵
        """
        if obj_ids.numel() == 0:
            return torch.empty(
                0, device=boxes.device, dtype=obj_ids.dtype
            ), torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype)

        num_objs = obj_ids.size(0)
        mask = torch.empty(num_objs, dtype=torch.bool, device=boxes.device)

        # 確保型別和連續性 (必須與 C++ Kernel 期望的型別一致)
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

        # 使用 PyTorch 切片返回結果
        return obj_ids[mask], boxes[mask]

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

        # 切換到背景 Stream
        with torch.cuda.stream(self.extraction_stream):
            # 1. Zero-Copy 裁切
            crops = cropper.process(frame_tensor, extract_boxes)

            # 2. TensorRT 特徵提取
            features = extractor.extract(crops)

        return cast(Optional[torch.Tensor], features)


if __name__ == "__main__":
    print("🚀 Testing SmartTracker Logic...")
    tracker = SmartTracker()

    # 模擬第 1 幀：兩個新物件 (應觸發 A)
    ids1 = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    boxes1 = torch.tensor(
        [[10, 10, 50, 50], [100, 100, 150, 150]], device="cuda", dtype=torch.float32
    )
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes1)
    print(f"Frame 1 - Extracted IDs (Expect [1, 2]): {ext_ids.tolist()}")

    # 模擬第 2 幀：物件輕微移動 (不應觸發)
    boxes2 = torch.tensor(
        [[12, 12, 52, 52], [102, 102, 152, 152]], device="cuda", dtype=torch.float32
    )
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes2)
    print(f"Frame 2 - Extracted IDs (Expect []): {ext_ids.tolist()}")

    # 模擬第 3 幀：物件 1 大幅形變 (觸發 C)，物件 2 輕微移動
    boxes3 = torch.tensor(
        [[10, 10, 100, 100], [105, 105, 155, 155]], device="cuda", dtype=torch.float32
    )
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes3)
    print(f"Frame 3 - Extracted IDs (Expect [1]): {ext_ids.tolist()}")

    # 模擬第 4 幀：物件 2 突然改變方向 (折返跑，觸發 B)
    boxes4 = torch.tensor(
        [[15, 15, 105, 105], [90, 90, 140, 140]], device="cuda", dtype=torch.float32
    )
    ext_ids, ext_boxes = tracker.update_and_filter(ids1, boxes4)
    print(f"Frame 4 - Extracted IDs (Expect [2]): {ext_ids.tolist()}")
