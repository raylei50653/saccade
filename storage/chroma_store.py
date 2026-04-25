import chromadb
import uuid
import time
import shutil
import os
from typing import Dict, Any, List, Optional, cast


class ChromaStore:
    """
    Saccade 向量記憶庫 (強化版)

    支援混合檢索：語義搜尋 + Metadata (時間、物件、異常標籤) 過濾。
    實踐 Pillar 5 中的 Feature & Timestamp Correlation。
    """

    def __init__(
        self,
        path: str = "./storage/chroma_db",
        collection_name: str = "saccade_memories",
    ):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def backup(self, backup_dir: str = "./storage/backups") -> Optional[str]:
        """
        建立 ChromaDB 冷備份 (Cold Backup) Snapshot。
        將目前的資料庫目錄壓縮並儲存至備份目錄。
        """
        try:
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"chroma_backup_{timestamp}")
            
            # 使用 shutil.make_archive 建立 zip 壓縮檔
            archive_path = shutil.make_archive(backup_path, 'zip', self.path)
            print(f"📦 [ChromaStore] Backup successfully created at {archive_path}")
            return archive_path
        except Exception as e:
            print(f"❌ [ChromaStore] Backup failed: {e}")
            return None

    def add_memory(
        self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None, embedding: Optional[List[float]] = None
    ) -> str:
        """新增一條記憶，包含多維度元數據與視覺特徵"""
        memory_id = doc_id or str(uuid.uuid4())
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()

        if embedding is not None:
            self.collection.add(documents=[content], metadatas=[metadata], ids=[memory_id], embeddings=[embedding])
        else:
            self.collection.add(documents=[content], metadatas=[metadata], ids=[memory_id])
        return memory_id

    def hybrid_query(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 5,
        start_time: Optional[float] = None,
        is_anomaly: Optional[int] = None,
        object_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        核心功能：關聯檢索 (支援純文字、純向量或混合)
        支援同時搜尋：
        1. 語義描述 (如 'person with knife')
        2. 視覺特徵 (SigLIP embedding)
        3. 時間區間 (最近一小時)
        4. 異常標籤 (僅看 ALERT)
        5. 特定物件 (包含 car)
        """
        where_clauses: List[Dict[str, Any]] = []

        if start_time:
            where_clauses.append({"timestamp": {"$gte": start_time}})
        if is_anomaly is not None:
            where_clauses.append({"is_anomaly": is_anomaly})
        if object_filter:
            where_clauses.append({"objects": {"$contains": object_filter}})

        where: Any = None
        if len(where_clauses) > 1:
            where = {"$and": where_clauses}
        elif len(where_clauses) == 1:
            where = where_clauses[0]

        query_kwargs: Dict[str, Any] = {"n_results": n_results}
        if where:
            query_kwargs["where"] = where
        if query_text:
            query_kwargs["query_texts"] = [query_text]
        if query_embedding:
            query_kwargs["query_embeddings"] = [query_embedding]

        results = self.collection.query(**query_kwargs)
        return cast(Dict[str, Any], results)

    def delete_memories(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
