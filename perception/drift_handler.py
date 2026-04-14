import torch
from typing import Dict, Tuple

class SemanticDriftHandler:
    """
    Saccade 語義漂移與去重管理器 (Phase 4)
    
    在 GPU VRAM 中維護近期物件的特徵快取。
    利用極速的 Cosine Similarity 排除高度重複的語義向量，避免 ChromaDB 垃圾資料堆積。
    """
    def __init__(self, similarity_threshold: float = 0.95, max_objects: int = 1000, feature_dim: int = 768):
        self.similarity_threshold = similarity_threshold
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        
        # 極致優化：使用大 Tensor 儲存特徵，ID 作為索引
        self.feature_cache_tensor = torch.zeros((max_objects, feature_dim), device="cuda")
        self.has_cache_tensor = torch.zeros(max_objects, dtype=torch.bool, device="cuda")

    def filter_novel_features(self, obj_ids: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        過濾出具有「新語義」的特徵 (全 GPU 向量化，移除所有同步點)。
        """
        if obj_ids.numel() == 0:
            return (torch.empty(0, device=obj_ids.device, dtype=obj_ids.dtype), 
                    torch.empty((0, features.size(1)), device=features.device, dtype=features.dtype))
            
        # 使用取模運算處理長效 ID
        ids = (obj_ids % self.max_objects).long()
        
        # 1. GPU 向量化提取歷史特徵
        has_cache = self.has_cache_tensor[ids]
        cached_feats = self.feature_cache_tensor[ids]
        
        # 2. 一次性 GPU 計算相似度
        sims = torch.nn.functional.cosine_similarity(features, cached_feats, dim=1)
        
        # KEEP 邏輯：全新物件 OR 相似度低於閾值
        should_keep = (~has_cache) | (sims < self.similarity_threshold)
        
        # 3. 更新快取 (直接遮罩寫入，不進行 .any() 檢查)
        keep_ids = ids[should_keep]
        keep_feats = features[should_keep]
        self.feature_cache_tensor[keep_ids] = keep_feats
        self.has_cache_tensor[keep_ids] = True
                        
        return obj_ids[should_keep], features[should_keep]

if __name__ == "__main__":
    print("🚀 Testing SemanticDriftHandler...")
    handler = SemanticDriftHandler(similarity_threshold=0.95)
    
    # 模擬 1 筆特徵
    ids = torch.tensor([1], device="cuda")
    feat1 = torch.randn(1, 1152, device="cuda")
    
    # 第一次寫入 (必過)
    novel_ids, _ = handler.filter_novel_features(ids, feat1)
    print(f"Test 1 - Expect [1]: {novel_ids.tolist()}")
    
    # 第二次寫入一模一樣的特徵 (應被過濾)
    novel_ids, _ = handler.filter_novel_features(ids, feat1)
    print(f"Test 2 - Expect []: {novel_ids.tolist()}")
    
    # 第三次寫入加了微小雜訊的特徵 (相似度 > 0.95，應被過濾)
    feat2 = feat1 + torch.randn_like(feat1) * 0.01
    novel_ids, _ = handler.filter_novel_features(ids, feat2)
    print(f"Test 3 - Expect []: {novel_ids.tolist()}")
    
    # 第四次寫入完全不同的特徵 (語義漂移，應通過)
    feat3 = torch.randn(1, 1152, device="cuda")
    novel_ids, _ = handler.filter_novel_features(ids, feat3)
    print(f"Test 4 - Expect [1]: {novel_ids.tolist()}")
