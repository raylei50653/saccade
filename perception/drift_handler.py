import torch
from typing import Dict, Tuple

class SemanticDriftHandler:
    """
    Saccade 語義漂移與去重管理器 (Phase 4)
    
    在 GPU VRAM 中維護近期物件的特徵快取。
    利用極速的 Cosine Similarity 排除高度重複的語義向量，避免 ChromaDB 垃圾資料堆積。
    """
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        # 熱資料快取：obj_id -> feature_tensor [1, Feature_Dim]
        self.feature_cache: Dict[int, torch.Tensor] = {}

    def filter_novel_features(self, obj_ids: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        過濾出具有「新語義」的特徵。
        
        :param obj_ids: 追蹤 ID 列表 [N]
        :param features: 剛提取出的高維特徵矩陣 [N, Feature_Dim]
        :return: (novel_ids, novel_features) 僅保留相似度低於閾值的特徵
        """
        novel_indices = []
        
        for i, obj_id_tensor in enumerate(obj_ids):
            obj_id = int(obj_id_tensor.item())
            # 取出單筆特徵，維持 2D 形狀 [1, D]
            current_feat = features[i:i+1]
            
            if obj_id in self.feature_cache:
                cached_feat = self.feature_cache[obj_id]
                # 在 GPU 上直接計算 Cosine Similarity
                sim = torch.nn.functional.cosine_similarity(current_feat, cached_feat)
                
                # 如果相似度低於閾值，代表語義發生漂移 (例如：人轉身、拿出武器)
                if sim.item() < self.similarity_threshold:
                    novel_indices.append(i)
                    # 更新快取為新的語義狀態
                    self.feature_cache[obj_id] = current_feat
            else:
                # 全新物件，必定寫入
                novel_indices.append(i)
                self.feature_cache[obj_id] = current_feat
                
        if not novel_indices:
            return torch.empty(0, device=obj_ids.device), torch.empty((0, features.size(1)), device=features.device)
            
        indices_tensor_feat = torch.tensor(novel_indices, device=features.device, dtype=torch.long)
        indices_tensor_ids = torch.tensor(novel_indices, device=obj_ids.device, dtype=torch.long)
        return obj_ids[indices_tensor_ids], features[indices_tensor_feat]

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
