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
        def update(self, *args: Any, **kwargs: Any) -> List[TrackResult]: return []
        def get_state_snapshots(self, *args: Any, **kwargs: Any) -> List[Any]: return []


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
    ) -> None:
        """調整追蹤器門檻與參數。"""
        self.tracker.set_params(track_thresh, high_thresh, match_thresh, track_buffer)

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
    ) -> List[TrackResult]:
        """
        更新追蹤器狀態。
        """
        num_dets = boxes.size(0)
        if num_dets == 0:
            return []

        # 確保資料連續且在 GPU 上
        boxes_contig = boxes.to(torch.float32).contiguous()
        scores_contig = scores.to(torch.float32).contiguous()
        classes_contig = classes.to(torch.int32).contiguous()
        
        embed_ptr = embeddings.to(torch.float32).contiguous().data_ptr() if embeddings is not None else None
        gmc_ptr = gmc.to(torch.float32).contiguous().data_ptr() if gmc is not None else None

        stream = torch.cuda.current_stream().cuda_stream
        
        return cast(List[TrackResult], self.tracker.update(
            boxes_contig.data_ptr(),
            scores_contig.data_ptr(),
            classes_contig.data_ptr(),
            num_dets,
            stream,
            embed_ptr,
            gmc_ptr,
            light_factor
        ))

    def get_state_snapshots(self) -> List[Any]:
        """Return active Kalman state/covariance snapshots from the C++ tracker."""
        stream = torch.cuda.current_stream().cuda_stream
        return cast(List[Any], self.tracker.get_state_snapshots(stream))
