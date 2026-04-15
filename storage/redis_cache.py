import json
import os
import redis
import redis.asyncio as aioredis
from typing import Dict, Any, Optional, List, Tuple, cast


class RedisCache:
    """
    Saccade Redis 緩衝與快取 (L3 Shock Absorber)

    支援 Redis Streams 用於高頻感知事件緩衝。
    """

    def __init__(self, url: Optional[str] = None) -> None:
        env_url = os.getenv("REDIS_URL")
        self.url: str = url or env_url or "redis://localhost:6379/0"
        self.client: Optional[aioredis.Redis] = None
        # 串流名稱與限制
        self.stream_name = "saccade:events:stream"
        self.max_len = 10000  # 限制緩衝區大小，防止資料庫斷線時 Redis 爆掉

    async def connect(self) -> None:
        if self.client is None:
            # 建立連線池以支撐高頻並發
            self.client = aioredis.from_url(
                self.url, decode_responses=True, max_connections=32
            )
            # 初始化消費者群組 (如果不存在)
            try:
                await self.client.xgroup_create(
                    self.stream_name, "orchestrator_group", id="0", mkstream=True
                )
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise e

    async def add_to_stream(self, event_data: Dict[str, Any]) -> str:
        """
        高壓生產者：將事件推入 Redis Stream (L3 衝擊吸收器)
        """
        if not self.client:
            await self.connect()

        assert self.client is not None

        # 序列化資料
        payload: Dict[Any, Any] = {"data": json.dumps(event_data)}

        # XADD: 使用 '~' (Approximate Maxlen) 提升寫入效能
        msg_id = await self.client.xadd(
            self.stream_name, payload, maxlen=self.max_len, approximate=True
        )
        return cast(str, msg_id)

    async def add_to_stream_batch(self, events: List[Dict[str, Any]]) -> List[str]:
        """
        批次生產者：利用 Redis Pipeline 一次性寫入多個事件 (D 優化)
        """
        if not self.client:
            await self.connect()

        assert self.client is not None

        if not events:
            return []

        async with self.client.pipeline(transaction=False) as pipe:
            for event in events:
                payload: Dict[Any, Any] = {"data": json.dumps(event)}
                pipe.xadd(
                    self.stream_name, payload, maxlen=self.max_len, approximate=True
                )

            results = await pipe.execute()
        return cast(List[str], results)

    async def read_stream_batch(
        self, count: int = 100, timeout_ms: int = 500
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        批次消費者：從 Stream 撈取多筆事件 (微批次處理)
        """
        if not self.client:
            await self.connect()

        assert self.client is not None

        # 從消費者群組讀取尚未確認的訊息
        streams = await self.client.xreadgroup(
            "orchestrator_group",
            "worker_1",
            {self.stream_name: ">"},
            count=count,
            block=timeout_ms,
        )

        events = []
        if streams:
            for stream, msgs in streams:
                for msg_id, payload in msgs:
                    event_data = json.loads(payload["data"])
                    events.append((msg_id, event_data))
        return events

    async def acknowledge(self, message_ids: List[str]) -> None:
        """確認訊息已處理，從待處理列表中移除"""
        if not self.client:
            await self.connect()

        assert self.client is not None

        if message_ids:
            await self.client.xack(self.stream_name, "orchestrator_group", *message_ids)

    async def disconnect(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def get_active_objects(self) -> List[int]:
        """獲取目前所有活躍的目標 ID (透過 KEY 掃描或集合)"""
        if not self.client:
            await self.connect()

        assert self.client is not None

        # 掃描目前的 track keys
        keys = await self.client.keys("saccade:track:*")
        ids = [int(k.split(":")[-1]) for k in keys]
        return ids

    async def get_object_history(self, obj_id: int) -> Optional[Dict[str, Any]]:
        """獲取特定物件的詳細時空紀錄"""
        if not self.client:
            await self.connect()

        assert self.client is not None

        data = await cast(Any, self.client.hgetall(f"saccade:track:{obj_id}"))
        if not data:
            return None

        # 將資料轉型並回傳
        return {
            "track_id": obj_id,
            "first_seen": float(data.get("first_seen", 0)),
            "last_seen": float(data.get("last_seen", 0)),
            "classes": json.loads(data.get("classes", "[]")),
            "avg_similarity": float(data.get("avg_similarity", 1.0)),
        }
