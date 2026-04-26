import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional, Any, Dict, Deque
from torchvision.ops import box_iou
from .reorder import ReorderingBuffer
from .tracker_gpu import GPUByteTracker
from perception.feature_bank import FeatureBank


class SmartTracker:
    """
    Saccade 智能追蹤器 (Synchronous Edition)

    整合 C++ GPUByteTracker 與 SigLIP 2 Saccade Heartbeat ReID。
    每 heartbeat_interval 幀對偵測框提取 embedding，送入 C++ 做
    Sinkhorn 融合匹配 ((1-w)*IoU + w*CosSim)；幀間用純 IoU。

    GMC (全域運動補償) 與光線因子同步計算，不阻塞主迴圈。
    FeatureBank 保存確認軌跡特徵，供跨鏡頭 Re-ID 查詢使用。
    """

    def __init__(
        self,
        iou_threshold: float = 0.7,
        max_objects: int = 2048,
        embedding_dim: int = 768,
        feature_bank: Optional[FeatureBank] = None,
        extractor: Optional[Any] = None,   # TRTFeatureExtractor
        cropper: Optional[Any] = None,     # ZeroCopyCropper
        heartbeat_interval: int = 10,
        geometry_mid_scale: bool = False,
        geometry_ref_height_ratio: float = 0.12,
        geometry_min_scale: float = 0.875,
        geometry_max_scale: float = 1.20,
        geometry_ema_beta: float = 0.80,
        geometry_loosen_step: float = 0.08,
        geometry_tighten_step: float = 0.03,
        geometry_min_samples: int = 5,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.heartbeat_interval = heartbeat_interval
        self.geometry_mid_scale = geometry_mid_scale
        self.geometry_ref_height_ratio = geometry_ref_height_ratio
        self.geometry_min_scale = geometry_min_scale
        self.geometry_max_scale = geometry_max_scale
        self.geometry_ema_beta = geometry_ema_beta
        self.geometry_loosen_step = geometry_loosen_step
        self.geometry_tighten_step = geometry_tighten_step
        self.geometry_min_samples = geometry_min_samples
        self._geometry_scale_prev = 1.0
        self._geometry_median_ratio_ema = 0.0
        self._geometry_scale_initialized = False

        self.gpu_tracker = GPUByteTracker(max_objects, embedding_dim)
        self.gpu_tracker.set_params(
            track_thresh=0.1,
            high_thresh=0.5,
            match_thresh=iou_threshold,
        )

        self.reorder_buffer = ReorderingBuffer()
        self.feature_bank = feature_bank or FeatureBank()

        # L2 組件：由外部注入，None 時退化為純 IoU 追蹤
        self.extractor = extractor
        self.cropper = cropper

        self.frame_count = 0

        # Non-blocking ReID: crop + extract run on a dedicated CUDA stream.
        # _pending_reid holds (embs, det_boxes, event) while GPU work is in flight.
        # _ready_reid holds the latest completed result for the main stream to consume.
        self._reid_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._pending_reid: Optional[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = None
        self._ready_reid: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        # GMC 狀態（per-tracker，不跨流）
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

        # Farewell ReID：緩衝最近 5 幀已確認追蹤的 (frame_tensor, ids, boxes)
        # 軌跡消失時從中取最近 3 幀做 SigLIP2 → L2-normalized average → FeatureBank
        self._farewell_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=5)
        self._prev_tracked_ids: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_degradation_params(self, level: int) -> None:
        """根據 ResourceManager 降級等級動態調整追蹤參數。"""
        if level >= 3:  # EMERGENCY
            self.gpu_tracker.set_params(track_thresh=0.2, high_thresh=0.6, track_buffer=10)
        else:
            self.gpu_tracker.set_params(track_thresh=0.1, high_thresh=0.5, track_buffer=30)

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        frame_tensor: Optional[torch.Tensor] = None,
        stream_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        同步追蹤更新。

        :param boxes:        偵測框 [N, 4]，格式 (x1,y1,x2,y2)，絕對像素座標
        :param scores:       信心分 [N]
        :param classes:      類別 ID [N]
        :param frame_tensor: 當前幀 Tensor [3,H,W] 或 [1,3,H,W]，float 0~1
                             提供時啟用 GMC + 光線補償 + Heartbeat ReID
        :param stream_id:    串流 ID，用於 FeatureBank 跨鏡頭索引
        :return: (tracked_ids [M], tracked_boxes [M,4], tracked_classes [M])
        """
        # 1. GMC + 光線補償
        gmc_matrix = None
        light_factor = 0.0
        if frame_tensor is not None:
            gmc_matrix = self._calculate_gmc(frame_tensor)
            light_factor = self._calculate_light_factor(frame_tensor)

        # 2. Poll for completed async ReID from the previous heartbeat (non-blocking)
        self._poll_reid()

        # Saccade Heartbeat：每 heartbeat_interval 幀非阻塞提交 embedding 提取到 _reid_stream
        is_heartbeat = (
            self.extractor is not None
            and self.cropper is not None
            and boxes.numel() > 0
            and frame_tensor is not None
            and self.frame_count % self.heartbeat_interval == 0
        )
        if is_heartbeat:
            self._submit_reid_async(frame_tensor, boxes)

        # Use embeddings from the most recently completed async heartbeat
        embeddings: Optional[torch.Tensor] = None
        reid_det_boxes: Optional[torch.Tensor] = None
        if self._ready_reid is not None:
            embeddings, reid_det_boxes = self._ready_reid

        # 3. C++ GPUByteTracker 核心更新
        #    C++ 內部：匹配後自動以偵測 embedding 更新對應追蹤槽的參考特徵
        results = self.gpu_tracker.update(
            boxes, scores, classes,
            embeddings=embeddings,
            gmc=gmc_matrix,
            light_factor=light_factor,
            mid_thresh_scale=self._geometry_mid_thresh_scale(boxes, frame_tensor),
        )

        self.frame_count += 1

        # 4. 轉換為 Tensor（results 為空時回傳空張量）
        dev = boxes.device
        if results:
            tracked_ids = torch.tensor(
                [r.obj_id for r in results], dtype=torch.int32, device=dev
            )
            tracked_boxes = torch.tensor(
                [[r.x1, r.y1, r.x2, r.y2] for r in results], dtype=torch.float32, device=dev
            )
            tracked_classes = torch.tensor(
                [r.class_id for r in results], dtype=torch.int32, device=dev
            )
        else:
            tracked_ids = torch.empty((0,), dtype=torch.int32, device=dev)
            tracked_boxes = torch.empty((0, 4), dtype=torch.float32, device=dev)
            tracked_classes = torch.empty((0,), dtype=torch.int32, device=dev)

        # 5. FeatureBank 更新（跨鏡頭 Re-ID 用）
        #    使用前一 heartbeat 完成的 embeddings；reid_det_boxes 為對應的偵測框
        if embeddings is not None and reid_det_boxes is not None and tracked_ids.numel() > 0:
            self._update_feature_bank_reuse(tracked_ids, tracked_boxes, reid_det_boxes, embeddings, stream_id)

        # 6. Farewell ReID：軌跡消失瞬間從最近 3 幀補提 embedding
        current_ids = set(tracked_ids.tolist())
        lost_ids = self._prev_tracked_ids - current_ids
        if lost_ids and self.extractor is not None and self.cropper is not None:
            self._extract_farewell_embeddings(lost_ids, dev, stream_id)

        # 更新 buffer：持有 frame_tensor ref 延遲 GC，讓 farewell 可往回查
        if frame_tensor is not None and tracked_ids.numel() > 0:
            self._farewell_buffer.append((
                frame_tensor,
                tracked_ids.clone(),
                tracked_boxes.clone(),
            ))
        self._prev_tracked_ids = current_ids

        return tracked_ids, tracked_boxes, tracked_classes

    def find_cross_camera_matches(
        self,
        query_embeddings: torch.Tensor,
        lost_ids: list,
        stream_id: int = 0,
    ) -> Dict[int, int]:
        """跨鏡頭 Re-ID：利用 FeatureBank 批量矩陣匹配。"""
        return self.feature_bank.find_matches_batch(
            query_embeddings, lost_ids, self.frame_count, stream_id
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _geometry_mid_thresh_scale(
        self,
        boxes: torch.Tensor,
        frame_tensor: Optional[torch.Tensor],
    ) -> float:
        if not self.geometry_mid_scale or boxes.numel() == 0 or frame_tensor is None:
            return 1.0
        frame_h = frame_tensor.shape[-2]
        if frame_h <= 0:
            return 1.0
        if boxes.shape[0] < self.geometry_min_samples:
            return self._geometry_scale_prev if self._geometry_scale_initialized else 1.0
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
        median_ratio = float(torch.median(heights).item()) / float(frame_h)
        if median_ratio <= 1e-6:
            return self._geometry_scale_prev if self._geometry_scale_initialized else 1.0
        if not self._geometry_scale_initialized:
            self._geometry_median_ratio_ema = median_ratio
            raw_scale = max(
                self.geometry_min_scale,
                min(self.geometry_max_scale, self.geometry_ref_height_ratio / self._geometry_median_ratio_ema),
            )
            self._geometry_scale_prev = raw_scale
            self._geometry_scale_initialized = True
            return raw_scale

        self._geometry_median_ratio_ema = (
            self.geometry_ema_beta * self._geometry_median_ratio_ema
            + (1.0 - self.geometry_ema_beta) * median_ratio
        )
        raw_scale = max(
            self.geometry_min_scale,
            min(self.geometry_max_scale, self.geometry_ref_height_ratio / self._geometry_median_ratio_ema),
        )
        diff = raw_scale - self._geometry_scale_prev
        if diff < 0.0:
            step = max(diff, -self.geometry_loosen_step) if self.geometry_loosen_step > 0.0 else diff
        else:
            step = min(diff, self.geometry_tighten_step) if self.geometry_tighten_step > 0.0 else diff
        self._geometry_scale_prev = max(
            self.geometry_min_scale,
            min(
                self.geometry_max_scale,
                self._geometry_scale_prev + step,
            )
        )
        return self._geometry_scale_prev

    def _extract_embeddings(
        self, frame_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """裁切偵測框並用 SigLIP 2 提取 embedding（同步，供 fallback 使用）。"""
        frame_4d = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
        with torch.no_grad():
            crops = self.cropper.process(frame_4d, boxes)
            if crops.numel() == 0:
                return None
            return self.extractor.extract(crops)

    def _poll_reid(self) -> None:
        """Non-blocking check: promote _pending_reid to _ready_reid if GPU work is done."""
        if self._pending_reid is None:
            return
        embs, det_boxes, event = self._pending_reid
        if event.query():
            self._ready_reid = (embs, det_boxes)
            self._pending_reid = None

    def _submit_reid_async(self, frame_tensor: torch.Tensor, boxes: torch.Tensor) -> None:
        """Submit crop + extract to _reid_stream without blocking the main stream.

        The result is collected on the next _poll_reid() call after the GPU event fires.
        Skips silently if the previous extraction is still in flight.
        """
        if self._pending_reid is not None:
            return
        if self._reid_stream is None:
            embs = self._extract_embeddings(frame_tensor, boxes)
            if embs is not None:
                self._ready_reid = (embs, boxes)
            return
        with torch.no_grad(), torch.cuda.stream(self._reid_stream):
            frame_4d = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
            crops = self.cropper.process(frame_4d, boxes)
            if crops.numel() == 0:
                return
            embs = self.extractor.extract(crops)
        event = torch.cuda.Event()
        event.record(self._reid_stream)
        self._pending_reid = (embs, boxes.clone(), event)

    def _update_feature_bank_reuse(
        self,
        tracked_ids: torch.Tensor,
        tracked_boxes: torch.Tensor,
        det_boxes: torch.Tensor,
        det_embeddings: torch.Tensor,
        stream_id: int,
    ) -> None:
        """FeatureBank 更新：以 IoU 匹配將偵測 embedding 重用於追蹤框，避免第二次 SigLIP 推理。"""
        if det_boxes.numel() == 0 or tracked_boxes.numel() == 0:
            return
        iou_mat = box_iou(tracked_boxes, det_boxes)   # [M, N]
        max_iou, best_det = iou_mat.max(dim=1)        # [M]
        valid = max_iou > 0.5
        if not valid.any():
            return
        valid_ids = tracked_ids[valid]
        valid_embs = det_embeddings[best_det[valid]]
        self.feature_bank.update_batch(valid_ids, valid_embs, self.frame_count, stream_id)
        self.gpu_tracker.update_reference_features(valid_ids, valid_embs)

    def _extract_farewell_embeddings(
        self, lost_ids: set, dev: torch.device, stream_id: int
    ) -> None:
        """消失軌跡補提 embedding：從最近 3 幀 buffer 裁切 crops，
        L2-normalized average 後寫入 FeatureBank，避免 heartbeat 間隔造成的特徵過期。"""
        if not self._farewell_buffer:
            return

        farewell_ids: list[int] = []
        farewell_embs: list[torch.Tensor] = []

        for tid in lost_ids:
            crops: list[torch.Tensor] = []
            # 倒序掃 buffer（最近幀優先），最多取 3 幀
            for frame_t, buf_ids, buf_boxes in reversed(self._farewell_buffer):
                if len(crops) >= 3:
                    break
                if buf_ids.numel() == 0:
                    continue
                mask = buf_ids == tid
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0][0]
                bbox = buf_boxes[idx : idx + 1]  # [1, 4]
                frame_4d = frame_t.unsqueeze(0) if frame_t.dim() == 3 else frame_t
                with torch.no_grad():
                    crop = self.cropper.process(frame_4d, bbox)
                if crop.numel() > 0:
                    crops.append(crop)

            if not crops:
                continue

            all_crops = torch.cat(crops, dim=0)  # [K, C, H, W], K ≤ 3
            with torch.no_grad():
                embs = self.extractor.extract(all_crops)  # [K, D]
            # L2-normalize per-frame → average → L2-normalize（抵銷特徵震盪）
            embs = F.normalize(embs, dim=1)
            mean_emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=1)  # [1, D]

            farewell_ids.append(tid)
            farewell_embs.append(mean_emb)

        if not farewell_ids:
            return

        ids_t = torch.tensor(farewell_ids, dtype=torch.int32, device=dev)
        embs_t = torch.cat(farewell_embs, dim=0)  # [N_lost, D]
        self.feature_bank.update_batch(ids_t, embs_t, self.frame_count, stream_id)

    def _calculate_gmc(self, frame_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """OpenCV LK 光流計算仿射 GMC 矩陣。"""
        with torch.no_grad():
            t = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
            small = torch.nn.functional.interpolate(
                t, size=(240, 320), mode="area"
            ).squeeze(0)
            gray_t = 0.299 * small[0] + 0.587 * small[1] + 0.114 * small[2]
            curr_gray = (gray_t * 255).byte().cpu().numpy()

        h_gmc = None
        if self.prev_gray is not None:
            if self.prev_points is None or len(self.prev_points) < 20:
                self.prev_points = cv2.goodFeaturesToTrack(
                    self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10
                )
            if self.prev_points is not None:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, self.prev_points, None
                )
                good_prev = self.prev_points[status == 1]
                good_curr = curr_pts[status == 1]
                if len(good_prev) > 10:
                    H, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                    if H is not None:
                        # 縮放平移分量回原始解析度
                        orig = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
                        H[0, 2] *= orig.shape[-1] / 320.0   # width
                        H[1, 2] *= orig.shape[-2] / 240.0   # height
                        h_gmc = torch.from_numpy(H.astype(np.float32)).to(frame_tensor.device)
                self.prev_points = good_curr.reshape(-1, 1, 2)

        self.prev_gray = curr_gray
        return h_gmc

    def _calculate_light_factor(self, frame_tensor: torch.Tensor) -> float:
        """計算亮度因子，越暗 factor 越高，供 Kalman R 矩陣補償。"""
        with torch.no_grad():
            return max(0.0, 1.0 - frame_tensor.mean().item() * 1.5)
