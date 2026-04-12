from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from storage.redis_cache import RedisCache
from storage.chroma_store import ChromaStore
import time
import os

app = FastAPI(title="Saccade Spatiotemporal Retrieval API")

# 初始化存儲組件
redis_cache = RedisCache()
chroma_store = ChromaStore()

class SearchQuery(BaseModel):
    text: str
    n_results: Optional[int] = 5
    start_time: Optional[float] = None
    is_anomaly: Optional[bool] = None

@app.on_event("startup")
async def startup():
    await redis_cache.connect()

@app.on_event("shutdown")
async def shutdown():
    await redis_cache.disconnect()

@app.get("/")
async def root():
    return {"status": "online", "system": "Saccade", "api_version": "1.0"}

@app.get("/objects")
async def list_active_objects():
    """獲取目前所有活躍 (最近 5 分鐘內出現) 的目標 ID"""
    try:
        object_ids = await redis_cache.get_active_objects()
        return {"count": len(object_ids), "active_objects": object_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/objects/{obj_id}")
async def get_object_history(obj_id: int):
    """獲取特定物件的詳細時空紀錄與軌跡"""
    history = await redis_cache.get_object_history(obj_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Object {obj_id} not found or expired.")
    
    # 計算停留時間 (Dwell Time)
    duration = history["last_seen"] - history["first_seen"]
    history["dwell_time_seconds"] = round(duration, 2)
    
    return history

@app.post("/search")
async def semantic_search(query: SearchQuery):
    """
    執行時空語義檢索
    範例：查詢 'person with suspicious bag' 且只看異常紀錄
    """
    is_anomaly_int = 1 if query.is_anomaly is True else (0 if query.is_anomaly is False else None)
    
    try:
        results = chroma_store.hybrid_query(
            query_text=query.text,
            n_results=query.n_results or 5,
            start_time=query.start_time,
            is_anomaly=is_anomaly_int
        )
        
        # 格式化輸出
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
                
        return {"query": query.text, "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
