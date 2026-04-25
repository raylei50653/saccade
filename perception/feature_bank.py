import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class FeatureBank:
    """
    Saccade 高效能特徵銀行 (Vectorized Edition - Multi-Stream Ready)
    使用矩陣運算取代 Python 迴圈，支援萬級特徵秒級檢索與跨鏡頭 Re-ID。
    """
    def __init__(self, max_ids: int = 1000, feat_dim: int = 768, similarity_threshold: float = 0.85, device: str = "cuda") -> None:
        self.max_ids = max_ids
        self.feat_dim = feat_dim
        self.threshold = similarity_threshold
        self.device = device
        
        # 🚀 預配置矩陣
        self.features = torch.zeros((max_ids, feat_dim), device=device)
        self.id_map = torch.full((max_ids,), -1, dtype=torch.long, device=device)
        # 新增 stream_map 以支援跨鏡頭區分
        self.stream_map = torch.full((max_ids,), -1, dtype=torch.long, device=device)
        self.last_seen = torch.zeros((max_ids,), dtype=torch.long, device=device)
        self.is_active = torch.zeros((max_ids,), dtype=torch.bool, device=device)
        self.ptr = 0

    def reset(self) -> None:
        """清理所有特徵庫資料"""
        self.features.zero_()
        self.id_map.fill_(-1)
        self.stream_map.fill_(-1)
        self.last_seen.zero_()
        self.is_active.zero_()
        self.ptr = 0

    def update(self, track_id: int, embedding: torch.Tensor, frame_id: int, stream_id: int = 0) -> None:
        """更新或新增 ID 特徵 (支援 stream_id)"""
        # 檢查是否已存在 (需同時比對 track_id 與 stream_id)
        mask = (self.id_map == track_id) & (self.stream_map == stream_id)
        idx = mask.nonzero()
        
        if idx.numel() > 0:
            target_idx = idx[0][0].item()
        else:
            target_idx = self.ptr
            self.ptr = (self.ptr + 1) % self.max_ids
            self.id_map[target_idx] = track_id
            self.stream_map[target_idx] = stream_id
            
        self.features[target_idx] = F.normalize(embedding.view(-1), dim=-1)
        self.last_seen[target_idx] = frame_id
        self.is_active[target_idx] = True

    def find_matches_batch(self, query_embeddings: torch.Tensor, lost_ids: List[int], current_frame: int, stream_id: int = 0) -> Dict[int, int]:
        """
        🚀 批量矩陣匹配：一次計算所有偵測與所有同串流遺失 ID 的關係
        :return: {det_index: track_id}
        """
        if query_embeddings.numel() == 0 or not lost_ids:
            return {}

        # 1. 建立 Lost Mask (限定同一個 stream)
        valid_indices = []
        for lid in lost_ids:
            mask = (self.id_map == lid) & (self.stream_map == stream_id)
            idx = mask.nonzero()
            if idx.numel() > 0:
                valid_indices.append(idx[0][0].item())
        
        if not valid_indices:
            return {}

        v_idxs = torch.tensor(valid_indices, device=self.device)
        
        # 2. 矩陣乘法計算相似度
        queries = F.normalize(query_embeddings, dim=-1) # [N, D]
        targets = self.features[v_idxs]                 # [M, D]
        
        sim_matrix = torch.mm(queries, targets.t())
        max_sims, max_idxs = torch.max(sim_matrix, dim=1)
        
        results = {}
        for i in range(queries.shape[0]):
            if max_sims[i] > self.threshold:
                match_id = self.id_map[v_idxs[max_idxs[i]]].item()
                results[i] = int(match_id)
                
        return results

    def find_cross_camera_matches(self, query_embeddings: torch.Tensor, current_stream_id: int) -> Dict[int, Tuple[int, int]]:
        """
        🌐 跨鏡頭矩陣匹配：尋找其他串流中外觀高度相似的活躍物件。
        用於解決人物穿梭於不同攝影機畫面的 Re-ID 問題。
        :return: {det_index: (matched_stream_id, matched_track_id)}
        """
        if query_embeddings.numel() == 0:
            return {}

        # 篩選條件：活躍的、且屬於「其他」串流的特徵
        valid_mask = self.is_active & (self.stream_map != -1) & (self.stream_map != current_stream_id)
        valid_indices = valid_mask.nonzero().squeeze(-1)

        if valid_indices.numel() == 0:
            return {}

        queries = F.normalize(query_embeddings, dim=-1)  # [N, D]
        targets = self.features[valid_indices]           # [M, D]
        
        # 跨鏡頭匹配的門檻通常需要更嚴格 (e.g., 0.85 -> 0.88)
        strict_threshold = self.threshold + 0.03
        
        sim_matrix = torch.mm(queries, targets.t())
        max_sims, max_idxs = torch.max(sim_matrix, dim=1)

        results = {}
        for i in range(queries.shape[0]):
            if max_sims[i] > strict_threshold:
                best_match_idx = valid_indices[max_idxs[i]]
                match_stream = self.stream_map[best_match_idx].item()
                match_id = self.id_map[best_match_idx].item()
                results[i] = (int(match_stream), int(match_id))
                
        return results
