from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from saccade.storage.redis_cache import RedisCache
from saccade.storage.chroma_store import ChromaStore

_redis_cache: Optional[RedisCache] = None
_chroma_store: Optional[ChromaStore] = None


def _get_redis_cache() -> RedisCache:
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache


def _get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await _get_redis_cache().connect()
    try:
        yield
    finally:
        await _get_redis_cache().disconnect()


app = FastAPI(title="Saccade Spatiotemporal Retrieval API", lifespan=lifespan)


class SearchQuery(BaseModel):
    text: str
    n_results: Optional[int] = 5
    start_time: Optional[float] = None
    is_anomaly: Optional[bool] = None


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "online", "system": "Saccade", "api_version": "1.0"}


@app.get("/objects")
async def list_active_objects() -> Dict[str, Any]:
    """獲取目前所有活躍 (最近 5 分鐘內出現) 的目標 ID"""
    try:
        object_ids = await _get_redis_cache().get_active_objects()
        return {"count": len(object_ids), "active_objects": object_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/objects/{obj_id}")
async def get_object_history(obj_id: int) -> Dict[str, Any]:
    """獲取特定物件的詳細時空紀錄與軌跡"""
    try:
        history = await _get_redis_cache().get_object_history(obj_id)
        if not history:
            raise HTTPException(
                status_code=404, detail=f"Object {obj_id} not found or expired."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 計算停留時間 (Dwell Time)
    if "last_seen" in history and "first_seen" in history:
        duration = history["last_seen"] - history["first_seen"]
        history["dwell_time_seconds"] = round(duration, 2)

    return history


@app.post("/search")
async def semantic_search(query: SearchQuery) -> Dict[str, Any]:
    """
    執行時空語義檢索
    範例：查詢 'person with suspicious bag' 且只看異常紀錄
    """
    is_anomaly_int = (
        1 if query.is_anomaly is True else (0 if query.is_anomaly is False else None)
    )

    try:
        results = _get_chroma_store().hybrid_query(
            query_text=query.text,
            n_results=query.n_results if query.n_results is not None else 5,
            start_time=query.start_time,
            is_anomaly=is_anomaly_int,
        )

        # 格式化輸出
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        return {"query": query.text, "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
