import chromadb
import uuid
import time
from typing import Dict, Any, List, Optional, cast

class ChromaStore:
    """
    Saccade 向量記憶庫 (強化版)
    
    支援混合檢索：語義搜尋 + Metadata (時間、物件、異常標籤) 過濾。
    實踐 Pillar 5 中的 Feature & Timestamp Correlation。
    """
    def __init__(self, path: str = "./storage/chroma_db", collection_name: str = "saccade_memories"):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_memory(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """新增一條記憶，包含多維度元數據"""
        memory_id = doc_id or str(uuid.uuid4())
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
            
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    def hybrid_query(self, 
                     query_text: str, 
                     n_results: int = 5, 
                     start_time: Optional[float] = None, 
                     is_anomaly: Optional[int] = None,
                     object_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        核心功能：關聯檢索
        支援同時搜尋：
        1. 語義描述 (如 'person with knife')
        2. 時間區間 (最近一小時)
        3. 異常標籤 (僅看 ALERT)
        4. 特定物件 (包含 car)
        """
        where_clauses: List[Dict[str, Any]] = []
        
        if start_time:
            where_clauses.append({"timestamp": {"$gte": start_time}})
        if is_anomaly is not None:
            where_clauses.append({"is_anomaly": is_anomaly})
        if object_filter:
            # 支援簡單的物件名稱包含查詢
            where_clauses.append({"objects": {"$contains": object_filter}})

        # 構建 ChromaDB $and 條件
        where: Any = None
        if len(where_clauses) > 1:
            where = {"$and": where_clauses}
        elif len(where_clauses) == 1:
            where = where_clauses[0]

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        return cast(Dict[str, Any], results)

    def delete_memories(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
