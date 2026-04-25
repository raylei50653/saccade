import json
import os
import time
import asyncio
import redis.asyncio as redis
from typing import Dict, Any, Optional, List, cast, Awaitable

class MicroBatcher:
    """
    非同步微批次處理器，用於減少 Redis 寫入次數。
    """
    def __init__(self, client: redis.Redis, queue: str, window_ms: int = 100, max_size: int = 50):
        self.client = client
        self.queue = queue
        self.window_ms = window_ms
        self.max_size = max_size
        self._buf: List[str] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.TimerHandle] = None

    async def add(self, event_data: Dict[str, Any]) -> None:
        async with self._lock:
            self._buf.append(json.dumps(event_data))
            if len(self._buf) >= self.max_size:
                await self._flush_locked()
            elif self._timer is None:
                loop = asyncio.get_running_loop()
                self._timer = loop.call_later(self.window_ms / 1000.0, 
                                             lambda: asyncio.create_task(self.flush()))

    async def _flush_locked(self) -> None:
        if not self._buf:
            return
        if self._timer:
            self._timer.cancel()
            self._timer = None
        
        await cast(Awaitable[Any], self.client.rpush(self.queue, *self._buf))
        await cast(Awaitable[Any], self.client.expire(self.queue, 3600))
        self._buf.clear()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()


class RedisCache:
    """
    Saccade Redis 緩衝與快取 (L3 Shock Absorber)

    支援 Redis Streams 用於高頻感知事件緩衝。
    """

    def __init__(self, url: Optional[str] = None) -> None:
        env_url = os.getenv("REDIS_URL")
        self.url: str = url or env_url or "redis://localhost:6379/0"
        self.client: Optional[redis.Redis] = None
        self.batchers: Dict[str, MicroBatcher] = {}

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
        # 確保所有批次都已送出
        for batcher in self.batchers.values():
            await batcher.flush()
        if self.client:
            await self.client.aclose()
            self.client = None

    async def cleanup_expired_objects(self, max_memory_mb: int = 500) -> None:
        """
        定期監控 Redis 記憶體使用量，若超過閾值 (預設 500MB) 
        則強制清理部分 saccade:obj:* 快取以防溢出。
        """
        if not self.client:
            await self.connect()
        if self.client:
            try:
                info = await cast(Awaitable[Dict[str, Any]], self.client.info(section="memory"))
                used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)
                
                if used_memory_mb > max_memory_mb:
                    print(f"⚠️ [RedisCache] Memory {used_memory_mb:.1f}MB exceeds limit {max_memory_mb}MB. Initiating cleanup...")
                    # 獲取所有物件鍵
                    keys = await cast(Awaitable[List[bytes]], self.client.keys("saccade:obj:*"))
                    if keys:
                        # 隨機抽樣或直接刪除一半的鍵 (因為已經有 TTL 300s，這裡僅作緊急記憶體釋放)
                        keys_to_delete = keys[:len(keys)//2]
                        await cast(Awaitable[Any], self.client.delete(*keys_to_delete))
                        print(f"🧹 [RedisCache] Emergency cleanup: Deleted {len(keys_to_delete)} objects.")
            except Exception as e:
                print(f"❌ [RedisCache] Cleanup error: {e}")

    async def update_object_track(
        self, obj_id: int, label: str, box: List[float], timestamp: float
    ) -> None:
        """核心整合：更新 YOLO 目標的時空軌跡"""
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

    async def publish_event(self, queue: str, event_data: Dict[str, Any]) -> None:
        if not self.client:
            await self.connect()
        if self.client:
            if queue not in self.batchers:
                self.batchers[queue] = MicroBatcher(self.client, queue)
            await self.batchers[queue].add(event_data)
