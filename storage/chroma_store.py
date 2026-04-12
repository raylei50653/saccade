import chromadb
import uuid
import time
from typing import Dict, Any, List, Optional, cast

class ChromaStore:
    """
    Saccade 向量記憶庫 (Storage 模組)
    
    基於 ChromaDB，負責儲存 Cognition 的 VLM 分析結果，支援語義相似性檢索。
    實踐 Pillar 5 中的 Vector-indexed memory。
    """
    def __init__(self, path: str = "./storage/chroma_db", collection_name: str = "saccade_memories"):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        # 預設使用 Chroma 的 Embedding 函數，未來可自定義
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_memory(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """新增一條記憶"""
        memory_id = doc_id or str(uuid.uuid4())
        
        # 確保 metadata 中包含 timestamp
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
            
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    def query_memories(self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """語義搜尋相似的記憶"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        return cast(Dict[str, Any], results)

    def get_memories_by_time(self, start_time: float, end_time: Optional[float] = None) -> Dict[str, Any]:
        """按時間範圍搜尋記憶"""
        end_time_val = end_time or time.time()
        where_clause = {
            "$and": [
                {"timestamp": {"$gte": start_time}},
                {"timestamp": {"$lte": end_time_val}}
            ]
        }
        # ChromaDB 的 get 方法期望特定的 Where 格式
        results = self.collection.get(where=cast(Any, where_clause))
        return cast(Dict[str, Any], results)

    def delete_memories(self, ids: List[str]) -> None:
        """刪除指定記憶"""
        self.collection.delete(ids=ids)
