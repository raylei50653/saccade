import json
import os
import redis.asyncio as redis
from typing import Dict, Any, Optional, List, cast, Awaitable

class RedisCache:
    """
    Saccade Redis 時空快取 (整合 YOLO + 時間戳)
    """
    def __init__(self, url: Optional[str] = None) -> None:
        env_url = os.getenv("REDIS_URL")
        self.url: str = url or env_url or "redis://localhost:6379/0"
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        if self.client is None:
            self.client = redis.from_url(self.url)

    async def disconnect(self) -> None:
        if self.client:
            await cast(Awaitable[Any], self.client.aclose())
            self.client = None

    async def update_object_track(self, obj_id: int, label: str, box: List[float], timestamp: float) -> None:
        """核心整合：更新 YOLO 目標的時空軌跡"""
        if not self.client:
            await self.connect()
        if self.client:
            key = f"saccade:obj:{obj_id}"
            raw_data = await cast(Awaitable[Optional[str]], self.client.get(key))
            if raw_data:
                data = json.loads(raw_data)
                data["last_seen"] = timestamp
                data["last_box"] = box
                data["trajectory"].append({"t": timestamp, "pos": [(box[0]+box[2])/2, (box[1]+box[3])/2]})
                if len(data["trajectory"]) > 10:
                    data["trajectory"].pop(0)
            else:
                data = {
                    "id": obj_id,
                    "label": label,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "last_box": box,
                    "trajectory": [{"t": timestamp, "pos": [(box[0]+box[2])/2, (box[1]+box[3])/2]}]
                }
            await cast(Awaitable[Any], self.client.set(key, json.dumps(data), ex=300))

    async def get_active_objects(self) -> List[str]:
        """獲取目前所有活躍 (最近 5 分鐘內出現) 的目標 ID"""
        if not self.client:
            await self.connect()
        if self.client:
            keys = await cast(Awaitable[List[bytes]], self.client.keys("saccade:obj:*"))
            return [k.decode('utf-8').split(':')[-1] for k in keys]
        return []

    async def get_object_history(self, obj_id: int) -> Optional[Dict[str, Any]]:
        """獲取特定物件的時空歷史"""
        if not self.client:
            await self.connect()
        if self.client:
            key = f"saccade:obj:{obj_id}"
            data = await cast(Awaitable[Optional[str]], self.client.get(key))
            return cast(Dict[str, Any], json.loads(data)) if data else None
        return None

    async def publish_event(self, queue: str, event_data: Dict[str, Any]) -> None:
        if not self.client:
            await self.connect()
        if self.client:
            await cast(Awaitable[Any], self.client.rpush(queue, json.dumps(event_data)))
            await cast(Awaitable[Any], self.client.expire(queue, 3600))
