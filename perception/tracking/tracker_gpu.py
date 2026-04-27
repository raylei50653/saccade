import torch
from typing import List, Any, cast, Optional

try:
    from saccade_tracking_ext import GPUByteTracker as CppGPUByteTracker, TrackResult
except ImportError:
    # Fallback for environments where the extension is not available or has library conflicts
    class TrackResult: # type: ignore
        def __init__(self, *args, **kwargs):
            self.track_id = -1
            self.tlbr = [0, 0, 0, 0]
            self.score = 0.0
            self.class_id = -1

    class CppGPUByteTracker: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def set_params(self, *args: Any, **kwargs: Any) -> None: pass
        def set_reid_params(self, *args: Any, **kwargs: Any) -> None: pass
        def update(self, *args: Any, **kwargs: Any) -> List[TrackResult]: return []
        def get_state_snapshots(self, *args: Any, **kwargs: Any) -> List[Any]: return []
        def get_tentative_candidates(self, *args: Any, **kwargs: Any) -> List[Any]: return []


class GPUByteTracker:
    """
    Saccade GPU 追蹤器封裝。
    直接對接 C++ / CUDA 實作，確保 Zero-Copy。
    """
    def __init__(self, max_objects: int = 2048, embedding_dim: int = 768) -> None:
        self.tracker = CppGPUByteTracker(max_objects, embedding_dim)

    def set_params(
        self,
        track_thresh: float = 0.1,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        mid_thresh: float = 0.40,
        confirm_streak: int = 3,
        confirm_score_thresh: float = 0.50,
        adaptive_confirmation: bool = False,
        new_track_thresh: float = -1.0,
    ) -> None:
        """調整追蹤器門檻與參數。"""
        self.tracker.set_params(
            track_thresh,
            high_thresh,
            match_thresh,
            track_buffer,
            mid_thresh,
            confirm_streak,
            confirm_score_thresh,
            adaptive_confirmation,
            new_track_thresh,
        )

    def set_reid_params(
        self,
        cos_threshold: float = 0.90,
        iou_low: float = 0.30,
        iou_high: float = 0.60,
        weight: float = 0.40,
    ) -> None:
        """調整 C++ ReID fusion 門檻。"""
        set_reid_params = getattr(self.tracker, "set_reid_params", None)
        if set_reid_params is not None:
            set_reid_params(cos_threshold, iou_low, iou_high, weight)

    def update_reference_features(
        self,
        track_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        """更新追蹤器的參考特徵（用於 ReID）。"""
        num = track_ids.size(0)
        if num == 0:
            return
        
        ids_contig = track_ids.to(torch.int32).contiguous()
        features_contig = features.to(torch.float32).contiguous()
        stream = torch.cuda.current_stream().cuda_stream
        
        self.tracker.update_reference_features(
            ids_contig.data_ptr(),
            features_contig.data_ptr(),
            num,
            stream
        )

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        gmc: Optional[torch.Tensor] = None,
        light_factor: float = 0.0,
        mid_thresh_scale: float = 1.0,
    ) -> List[TrackResult]:
        """
        更新追蹤器狀態。
        """
        num_dets = boxes.size(0)

        # 💡 重要：必須保留這些 Tensor 的引用，防止在 data_ptr() 使用期間 be GC
        boxes_contig = boxes.to(torch.float32).contiguous()
        scores_contig = scores.to(torch.float32).contiguous()
        classes_contig = classes.to(torch.int32).contiguous()
        
        embed_contig = None
        embed_ptr = 0
        if embeddings is not None:
            embed_contig = embeddings.to(torch.float32).contiguous()
            embed_ptr = embed_contig.data_ptr()

        gmc_contig = None
        gmc_ptr = 0
        if gmc is not None:
            gmc_contig = gmc.to(torch.float32).contiguous()
            gmc_ptr = gmc_contig.data_ptr()

        stream = torch.cuda.current_stream().cuda_stream
        
        return cast(List[TrackResult], self.tracker.update(
            boxes_contig.data_ptr(),
            scores_contig.data_ptr(),
            classes_contig.data_ptr(),
            num_dets,
            stream,
            embed_ptr,
            gmc_ptr,
            light_factor,
            mid_thresh_scale,
        ))

    def get_state_snapshots(self) -> List[Any]:
        """Return active Kalman state/covariance snapshots from the C++ tracker."""
        stream = torch.cuda.current_stream().cuda_stream
        return cast(List[Any], self.tracker.get_state_snapshots(stream))

    def get_tentative_candidates(self) -> List[Any]:
        """Return active tentative tracks for lazy ReID arbitration/profiling."""
        stream = torch.cuda.current_stream().cuda_stream
        get_candidates = getattr(self.tracker, "get_tentative_candidates", None)
        if get_candidates is None:
            return []
        return cast(List[Any], get_candidates(stream))
