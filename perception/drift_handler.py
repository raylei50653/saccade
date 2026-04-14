import torch
import torch.nn.functional as F
import time
from typing import List, Tuple, Dict, Optional
from cognition.resource_manager import DegradationLevel


class SemanticDriftHandler:
    """
    Saccade 語義漂移處理器 (Industrial Grade - L2)
    
    1. 實作「語義質心 (Semantic Centroid)」比對與「熱身期 (Warm-up)」動態 Alpha。
    2. 實作「顯著性優先 (Salience-based)」截斷策略。
    3. 新增：基於超時的語義清理機制 (Pruning)，防止記憶體長尾。
    """

    def __init__(self, similarity_threshold: float = 0.95, base_alpha: float = 0.3) -> None:
        self.base_threshold = similarity_threshold
        self.base_alpha = base_alpha
        
        # 歷史特徵質心快取 {track_id: centroid_tensor}
        self.feature_history: Dict[int, torch.Tensor] = {}
        # 物件更新計數器 {track_id: update_count}
        self.track_update_count: Dict[int, int] = {}
        # 活躍時間追蹤 {track_id: last_active_timestamp}
        self.last_active_time: Dict[int, float] = {}
        
        self.N_OPT = 8

    def _get_dynamic_alpha(self, track_id: int) -> float:
        count = self.track_update_count.get(track_id, 0)
        if count == 0:
            return 1.0
        elif count < 5:
            return 0.7
        else:
            return self.base_alpha

    def calculate_drift(self, track_id: int, current_feature: torch.Tensor) -> float:
        if track_id not in self.feature_history:
            return 0.0
            
        centroid = self.feature_history[track_id]
        sim = F.cosine_similarity(current_feature.unsqueeze(0), centroid.unsqueeze(0))
        return sim.item()

    def filter_for_batch(
        self, 
        track_ids: List[int], 
        boxes: torch.Tensor,
        degradation_level: DegradationLevel
    ) -> List[int]:
        active_threshold = self.base_threshold
        max_batch = 32
        
        if degradation_level >= DegradationLevel.FAST_PATH:
            max_batch = self.N_OPT
        elif degradation_level >= DegradationLevel.REDUCED:
            max_batch = 16

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        priority_list = []
        for i, tid in enumerate(track_ids):
            count = self.track_update_count.get(tid, 0)
            area = areas[i].item()
            if count == 0: priority = 0
            elif count < 5: priority = 1
            else: priority = 2
            priority_list.append((priority, -area, tid))
            
        priority_list.sort()
        return [item[2] for item in priority_list[:max_batch]]

    def update_history(self, track_ids: List[int], new_features: torch.Tensor) -> None:
        """更新質心並標記活躍時間"""
        now = time.time()
        for i, tid in enumerate(track_ids):
            new_feat = new_features[i].detach().clone()
            alpha = self._get_dynamic_alpha(tid)
            
            if tid not in self.feature_history:
                self.feature_history[tid] = new_feat
                self.track_update_count[tid] = 1
            else:
                old_centroid = self.feature_history[tid]
                updated_centroid = alpha * new_feat + (1.0 - alpha) * old_centroid
                self.feature_history[tid] = F.normalize(updated_centroid, p=2, dim=0)
                self.track_update_count[tid] += 1
            
            self.last_active_time[tid] = now

    def prune_expired_centroids(self, timeout_sec: float = 300.0) -> int:
        """
        清理過期語義 (應定期由 Orchestrator 呼叫)
        """
        now = time.time()
        expired_ids = [tid for tid, last_ts in self.last_active_time.items() if (now - last_ts) > timeout_sec]
        for tid in expired_ids:
            self.clear_history(tid)
        return len(expired_ids)

    def clear_history(self, track_id: int) -> None:
        self.feature_history.pop(track_id, None)
        self.track_update_count.pop(track_id, None)
        self.last_active_time.pop(track_id, None)
