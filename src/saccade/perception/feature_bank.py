import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class FeatureBank:
    """
    Saccade 高效能特徵銀行 (Vectorized Edition - Multi-Stream Ready)
    使用矩陣運算取代 Python 迴圈，支援萬級特徵秒級檢索與跨鏡頭 Re-ID。
    _slot_map 提供 O(1) slot 查找，消除 GPU nonzero() 掃描。
    """

    def __init__(
        self,
        max_ids: int = 1000,
        feat_dim: int = 768,
        similarity_threshold: float = 0.85,
        device: str = "cuda",
    ) -> None:
        self.max_ids = max_ids
        self.feat_dim = feat_dim
        self.threshold = similarity_threshold
        self.device = device

        # 預配置矩陣
        self.features = torch.zeros((max_ids, feat_dim), device=device)
        self.id_map = torch.full((max_ids,), -1, dtype=torch.long, device=device)
        self.stream_map = torch.full((max_ids,), -1, dtype=torch.long, device=device)
        self.last_seen = torch.zeros((max_ids,), dtype=torch.long, device=device)
        self.is_active = torch.zeros((max_ids,), dtype=torch.bool, device=device)
        self.ptr = 0
        # O(1) slot lookup: (track_id, stream_id) -> slot_index
        self._slot_map: Dict[Tuple[int, int], int] = {}

    def reset(self) -> None:
        """清理所有特徵庫資料"""
        self.features.zero_()
        self.id_map.fill_(-1)
        self.stream_map.fill_(-1)
        self.last_seen.zero_()
        self.is_active.zero_()
        self.ptr = 0
        self._slot_map.clear()

    def _alloc_slot(self, track_id: int, stream_id: int) -> int:
        """Allocate or return existing slot for (track_id, stream_id)."""
        key = (track_id, stream_id)
        if key in self._slot_map:
            return self._slot_map[key]
        slot = self.ptr
        # Evict stale entry if this slot was previously occupied
        old_key = (int(self.id_map[slot].item()), int(self.stream_map[slot].item()))
        if old_key in self._slot_map and self._slot_map[old_key] == slot:
            del self._slot_map[old_key]
        self._slot_map[key] = slot
        self.id_map[slot] = track_id
        self.stream_map[slot] = stream_id
        self.ptr = (self.ptr + 1) % self.max_ids
        return slot

    def update(
        self, track_id: int, embedding: torch.Tensor, frame_id: int, stream_id: int = 0
    ) -> None:
        """更新或新增 ID 特徵 (支援 stream_id)"""
        target_idx = self._alloc_slot(track_id, stream_id)
        self.features[target_idx] = F.normalize(embedding.view(-1), dim=-1)
        self.last_seen[target_idx] = frame_id
        self.is_active[target_idx] = True

    def update_batch(
        self,
        track_ids: torch.Tensor,
        embeddings: torch.Tensor,
        frame_id: int,
        stream_id: int = 0,
    ) -> None:
        """Vectorized batch update — single scatter write instead of N sequential updates."""
        n = track_ids.size(0)
        if n == 0:
            return
        embeddings_norm = F.normalize(embeddings, dim=-1)
        target_indices: List[int] = [
            self._alloc_slot(int(track_ids[i].item()), stream_id) for i in range(n)
        ]
        idxs = torch.tensor(target_indices, dtype=torch.long, device=self.device)
        self.features[idxs] = embeddings_norm
        self.last_seen[idxs] = frame_id
        self.is_active[idxs] = True

    def find_matches_batch(
        self,
        query_embeddings: torch.Tensor,
        lost_ids: List[int],
        current_frame: int,
        stream_id: int = 0,
    ) -> Dict[int, int]:
        """
        批量矩陣匹配：一次計算所有偵測與所有同串流遺失 ID 的關係
        :return: {det_index: track_id}
        """
        if query_embeddings.numel() == 0 or not lost_ids:
            return {}

        valid_indices: List[int] = []
        valid_ids: List[int] = []
        for lid in lost_ids:
            key = (lid, stream_id)
            if key in self._slot_map:
                valid_indices.append(self._slot_map[key])
                valid_ids.append(lid)

        if not valid_indices:
            return {}

        v_idxs = torch.tensor(valid_indices, device=self.device)
        queries = F.normalize(query_embeddings, dim=-1)  # [N, D]
        targets = self.features[v_idxs]  # [M, D]

        sim_matrix = torch.mm(queries, targets.t())
        max_sims, max_idxs = torch.max(sim_matrix, dim=1)

        results = {}
        for i in range(queries.shape[0]):
            if max_sims[i] > self.threshold:
                results[i] = valid_ids[int(max_idxs[i].item())]

        return results

    def find_cross_camera_matches(
        self, query_embeddings: torch.Tensor, current_stream_id: int
    ) -> Dict[int, Tuple[int, int]]:
        """
        🌐 跨鏡頭矩陣匹配：尋找其他串流中外觀高度相似的活躍物件。
        用於解決人物穿梭於不同攝影機畫面的 Re-ID 問題。
        :return: {det_index: (matched_stream_id, matched_track_id)}
        """
        if query_embeddings.numel() == 0:
            return {}

        # 篩選條件：活躍的、且屬於「其他」串流的特徵
        valid_mask = (
            self.is_active
            & (self.stream_map != -1)
            & (self.stream_map != current_stream_id)
        )
        valid_indices = valid_mask.nonzero().squeeze(-1)

        if valid_indices.numel() == 0:
            return {}

        queries = F.normalize(query_embeddings, dim=-1)  # [N, D]
        targets = self.features[valid_indices]  # [M, D]

        # 跨鏡頭匹配的門檻通常需要更嚴格 (e.g., 0.85 -> 0.88)
        strict_threshold = self.threshold + 0.03

        sim_matrix = torch.mm(queries, targets.t())
        max_sims, max_idxs = torch.max(sim_matrix, dim=1)

        results = {}
        for i in range(queries.shape[0]):
            if max_sims[i] > strict_threshold:
                best_match_idx = valid_indices[int(max_idxs[i].item())]
                match_stream = self.stream_map[best_match_idx].item()
                match_id = self.id_map[best_match_idx].item()
                results[i] = (int(match_stream), int(match_id))

        return results
