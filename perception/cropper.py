import torch
from typing import Tuple


try:
    from saccade_perception_ext import Cropper as CropperCpp

    _CROPPER_CPP_AVAILABLE = True
except ImportError:
    _CROPPER_CPP_AVAILABLE = False


class ZeroCopyCropper:
    """
    Saccade 零拷貝裁切器 (Phase 1)

    直接在 GPU 顯存中接收原始高解析度 Frame Tensor 與 YOLO Bounding Boxes，
    優先使用 C++ Cropper (CUDA kernel)，否則使用 torchvision.ops.roi_align 進行批次裁切與縮放。
    產出 CLIP/SigLIP 相容的 Tensor。
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        mode: str = "tight",
        padding: float = 0.0,
    ) -> None:
        if mode not in {"tight", "square", "square_mean"}:
            raise ValueError(f"Unsupported crop mode: {mode}")
        self.output_size = output_size
        self.mode = mode
        self.padding = padding

        if _CROPPER_CPP_AVAILABLE and mode == "tight" and padding == 0.0:
            self._cpp = CropperCpp(output_size[1], output_size[0])  # w, h
        else:
            self._cpp = None

    @staticmethod
    def _roi_align(
        frame_tensor: torch.Tensor, rois: torch.Tensor, output_size: Tuple[int, int]
    ) -> torch.Tensor:
        # Import torchvision only on the Python fallback path to avoid loading
        # image codecs before the C++ extension resolves its own dependencies.
        from torchvision.ops import roi_align

        return roi_align(
            input=frame_tensor,
            boxes=rois,
            output_size=output_size,
            spatial_scale=1.0,
            aligned=True,
        )

    def _prepare_boxes(
        self, boxes: torch.Tensor, frame_h: int, frame_w: int
    ) -> torch.Tensor:
        if self.mode == "tight" and self.padding <= 0.0:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)

        if self.mode in {"square", "square_mean"}:
            side = torch.maximum(bw, bh) * (1.0 + self.padding)
            half_w = side * 0.5
            half_h = side * 0.5
        else:
            half_w = bw * (1.0 + self.padding) * 0.5
            half_h = bh * (1.0 + self.padding) * 0.5

        prepared = torch.stack(
            (
                (cx - half_w).clamp(0.0, float(frame_w - 1)),
                (cy - half_h).clamp(0.0, float(frame_h - 1)),
                (cx + half_w).clamp(0.0, float(frame_w - 1)),
                (cy + half_h).clamp(0.0, float(frame_h - 1)),
            ),
            dim=1,
        )
        return prepared

    def _fill_extra_with_mean(
        self,
        crops: torch.Tensor,
        original_boxes: torch.Tensor,
        prepared_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode != "square_mean" or crops.numel() == 0:
            return crops

        out_h, out_w = self.output_size
        px1, py1, px2, py2 = prepared_boxes.unbind(dim=1)
        ox1, oy1, ox2, oy2 = original_boxes.unbind(dim=1)
        pw = (px2 - px1).clamp(min=1.0)
        ph = (py2 - py1).clamp(min=1.0)

        sx1 = ((ox1 - px1) / pw * out_w).floor().clamp(0, out_w - 1).to(torch.long)
        sx2 = ((ox2 - px1) / pw * out_w).ceil().clamp(1, out_w).to(torch.long)
        sy1 = ((oy1 - py1) / ph * out_h).floor().clamp(0, out_h - 1).to(torch.long)
        sy2 = ((oy2 - py1) / ph * out_h).ceil().clamp(1, out_h).to(torch.long)

        xs = torch.arange(out_w, device=crops.device).view(1, 1, out_w)
        ys = torch.arange(out_h, device=crops.device).view(1, out_h, 1)
        inside = (
            (xs >= sx1.view(-1, 1, 1))
            & (xs < sx2.view(-1, 1, 1))
            & (ys >= sy1.view(-1, 1, 1))
            & (ys < sy2.view(-1, 1, 1))
        )
        outside = ~inside
        outside_f = outside.unsqueeze(1).to(crops.dtype)
        denom = outside_f.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        fill = (crops * outside_f).sum(dim=(2, 3), keepdim=True) / denom
        return torch.where(inside.unsqueeze(1), crops, fill)

    def process(self, frame_tensor: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        執行零拷貝裁切與縮放

        :param frame_tensor: 原始或縮放後的 GPU 影格，形狀 [1, C, H, W]，型別 Float (0~1)
        :param boxes: YOLO 輸出的 Bounding Boxes，形狀 [N, 4]，格式為 (x1, y1, x2, y2)，相對於 frame_tensor 的絕對座標
        :return: 裁切並縮放後的 Tensor，形狀 [N, C, output_size[0], output_size[1]]
        """
        # 若無目標，回傳空 Tensor 以免報錯
        if boxes is None or boxes.numel() == 0:
            return torch.empty(
                (0, frame_tensor.shape[1], *self.output_size),
                device=frame_tensor.device,
            )

        # 確保 boxes 與 frame_tensor 在相同的裝置上，並視需要在 GPU 上調整 RoI。
        boxes = boxes.to(frame_tensor.device)
        original_boxes = boxes
        boxes = self._prepare_boxes(
            boxes, frame_tensor.shape[-2], frame_tensor.shape[-1]
        )

        if self._cpp is not None:
            # Use C++ Implementation (CUDA kernel)
            h, w = frame_tensor.shape[-2], frame_tensor.shape[-1]
            num = boxes.shape[0]
            out = torch.empty(
                (num, 3, *self.output_size),
                device=frame_tensor.device,
                dtype=torch.float32,
            )

            # C++ process_gpu expects HWC RGB interleaved float32 [0, 1] on GPU for input.
            # If frame_tensor is [1, 3, H, W], we permute it.
            hwc_frame = frame_tensor.squeeze(0).permute(1, 2, 0).contiguous()

            self._cpp.process_gpu(
                hwc_frame.data_ptr(),
                w,
                h,
                boxes.contiguous().data_ptr(),
                num,
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream,
            )
            return out

        # roi_align 需要的 boxes 格式為 [N, 5]，第一欄是 batch_index。
        # 由於我們每次只處理單張圖片 (Batch=1)，所以 index 全為 0。
        batch_indices = torch.zeros(
            (boxes.shape[0], 1), device=boxes.device, dtype=boxes.dtype
        )
        rois = torch.cat([batch_indices, boxes], dim=1)

        # 使用 RoI Align 進行硬體加速的裁切與對齊
        crops = self._roi_align(frame_tensor, rois, self.output_size)

        crops = self._fill_extra_with_mean(crops, original_boxes, boxes)
        return crops

    def process_parts(
        self, frame_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        裁切 full / upper / lower 三個視角，輸出順序為 [full, upper, lower]。
        """
        if boxes is None or boxes.numel() == 0:
            return torch.empty(
                (0, frame_tensor.shape[1], *self.output_size),
                device=frame_tensor.device,
            )

        boxes = boxes.to(frame_tensor.device)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        h = (y2 - y1).clamp(min=1.0)

        upper = torch.stack((x1, y1, x2, y1 + h * 0.55), dim=1)
        lower = torch.stack((x1, y1 + h * 0.45, x2, y2), dim=1)
        part_boxes = torch.cat((boxes, upper, lower), dim=0)
        return self.process(frame_tensor, part_boxes)


if __name__ == "__main__":
    # 微秒級延遲測試 (Dry Run)
    import time

    print("🚀 Testing ZeroCopyCropper Initialization...")
    cropper = ZeroCopyCropper(output_size=(224, 224))

    # 模擬 1080p 影格 (1, 3, 1080, 1920) 在 GPU 上
    dummy_frame = torch.rand((1, 3, 1080, 1920), device="cuda", dtype=torch.float32)

    # 模擬 YOLO 抓到 5 個目標的 BBox (x1, y1, x2, y2)
    dummy_boxes = torch.tensor(
        [
            [100, 150, 300, 450],
            [500, 200, 800, 900],
            [50, 50, 100, 100],
            [1000, 500, 1200, 800],
            [1500, 100, 1800, 600],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    # 預熱 CUDA
    _ = cropper.process(dummy_frame, dummy_boxes)
    torch.cuda.synchronize()

    # 測速
    start = time.perf_counter()
    for i in range(100):
        out_crops = cropper.process(dummy_frame, dummy_boxes)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / 100 * 1000 * 1000  # 轉換為微秒 (µs)

    print(f"✅ Cropped Tensor Shape: {out_crops.shape} (N, C, H, W)")
    print(f"⚡ Average Latency for 5 objects: {latency:.2f} µs")
