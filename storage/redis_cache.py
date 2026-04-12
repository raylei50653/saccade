import json
import os
import redis.asyncio as redis
from typing import Dict, Any, Optional, List, cast, Awaitable

class RedisCache:
    """
    Saccade Redis 快取與事件佇列封裝
    
    負責處理 Perception (快路徑) 的事件發布以及 Cognition (慢路徑) 的狀態同步。
    """
    def __init__(self, url: Optional[str] = None) -> None:
        env_url = os.getenv("REDIS_URL")
        self.url: str = url or env_url or "redis://localhost:6379/0"
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """建立 Redis 連線"""
        if self.client is None:
            self.client = redis.from_url(self.url)

    async def disconnect(self) -> None:
        """關閉 Redis 連線"""
        if self.client:
            await cast(Awaitable[Any], self.client.aclose())
            self.client = None

    async def publish_event(self, queue: str, event_data: Dict[str, Any]) -> None:
        """發布事件至指定的 Redis 佇列 (List)"""
        if not self.client:
            await self.connect()
        if self.client:
            await cast(Awaitable[Any], self.client.rpush(queue, json.dumps(event_data)))
            # 設定預設 TTL 為 1 小時，防止記憶體溢出
            await cast(Awaitable[Any], self.client.expire(queue, 3600))

    async def update_object_state(self, obj_id: str, state: Dict[str, Any], ttl: int = 60) -> None:
        """更新追蹤目標的即時狀態 (如最後位置、時間)"""
        if not self.client:
            await self.connect()
        if self.client:
            key = f"saccade:obj:{obj_id}"
            await cast(Awaitable[Any], self.client.set(key, json.dumps(state), ex=ttl))

    async def get_object_state(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """獲取目標的最後已知狀態"""
        if not self.client:
            await self.connect()
        if self.client:
            key = f"saccade:obj:{obj_id}"
            data = await cast(Awaitable[Optional[str]], self.client.get(key))
            return cast(Dict[str, Any], json.loads(data)) if data else None
        return None

    async def get_active_objects(self) -> List[str]:
        """獲取目前所有活躍 (未過期) 的追蹤目標 ID"""
        if not self.client:
            await self.connect()
        if self.client:
            keys = await cast(Awaitable[List[bytes]], self.client.keys("saccade:obj:*"))
            return [k.decode('utf-8').split(':')[-1] for k in keys]
        return []
