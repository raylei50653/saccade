import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Any, Dict
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
    ) -> None:
        self.embedding_dim = embedding_dim
        self.heartbeat_interval = heartbeat_interval

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

        # GMC 狀態（per-tracker，不跨流）
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

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

        # 2. Saccade Heartbeat：每 heartbeat_interval 幀提取偵測框 embedding
        #    偵測 embedding 直接傳入 C++，作為 Sinkhorn 成本矩陣的 ReID 項目
        embeddings: Optional[torch.Tensor] = None
        if (
            self.extractor is not None
            and self.cropper is not None
            and boxes.numel() > 0
            and frame_tensor is not None
            and self.frame_count % self.heartbeat_interval == 0
        ):
            embeddings = self._extract_embeddings(frame_tensor, boxes)

        # 3. C++ GPUByteTracker 核心更新
        #    C++ 內部：匹配後自動以偵測 embedding 更新對應追蹤槽的參考特徵
        results = self.gpu_tracker.update(
            boxes, scores, classes,
            embeddings=embeddings,
            gmc=gmc_matrix,
            light_factor=light_factor,
        )

        self.frame_count += 1

        if not results:
            dev = boxes.device
            return (
                torch.empty((0,), dtype=torch.int32, device=dev),
                torch.empty((0, 4), dtype=torch.float32, device=dev),
                torch.empty((0,), dtype=torch.int32, device=dev),
            )

        # 4. 轉換為 Tensor
        dev = boxes.device
        tracked_ids = torch.tensor(
            [r.obj_id for r in results], dtype=torch.int32, device=dev
        )
        tracked_boxes = torch.tensor(
            [[r.x1, r.y1, r.x2, r.y2] for r in results], dtype=torch.float32, device=dev
        )
        tracked_classes = torch.tensor(
            [r.class_id for r in results], dtype=torch.int32, device=dev
        )

        # 5. FeatureBank 更新（跨鏡頭 Re-ID 用）
        #    在 heartbeat 幀，對確認軌跡框再做一次裁切+提取，確保特徵對齊追蹤 ID
        if (
            self.extractor is not None
            and self.cropper is not None
            and frame_tensor is not None
            and tracked_ids.numel() > 0
            and self.frame_count % self.heartbeat_interval == 0
        ):
            self._update_feature_bank(frame_tensor, tracked_ids, tracked_boxes, stream_id)

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

    def _extract_embeddings(
        self, frame_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """裁切偵測框並用 SigLIP 2 提取 embedding。"""
        frame_4d = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
        with torch.no_grad():
            crops = self.cropper.process(frame_4d, boxes)
            if crops.numel() == 0:
                return None
            return self.extractor.extract(crops)

    def _update_feature_bank(
        self,
        frame_tensor: torch.Tensor,
        tracked_ids: torch.Tensor,
        tracked_boxes: torch.Tensor,
        stream_id: int,
    ) -> None:
        """對確認軌跡框提取 embedding，寫入 FeatureBank 與 C++ 參考特徵庫。"""
        ref_embeddings = self._extract_embeddings(frame_tensor, tracked_boxes)
        if ref_embeddings is None:
            return
        for i in range(tracked_ids.size(0)):
            self.feature_bank.update(
                int(tracked_ids[i]),
                ref_embeddings[i],
                self.frame_count,
                stream_id,
            )
        # 同步更新 C++ 參考特徵（強化長期 Re-ID）
        self.gpu_tracker.update_reference_features(tracked_ids, ref_embeddings)

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
