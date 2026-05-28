import json
import math
import time
import uuid
import asyncio
import os
from typing import List, Any
from dotenv import load_dotenv
from saccade.media.rtsp import build_rtsp_url, DEFAULT_RTSP_SINGLE_STREAM_PATH
from saccade.storage.redis_cache import RedisCache

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class EntropyTrigger:
    """
    資訊熵觸發器 (Perception 快路徑)

    負責評估影格價值，並在達到閾值時向 Redis 發布事件以觸發慢路徑 (Cognition)。
    """

    def __init__(
        self, threshold: float = 0.8, redis_url: str = REDIS_URL, cooldown: float = 2.0
    ):
        self.threshold = threshold
        self.redis_url = redis_url
        self.cache = RedisCache(redis_url)
        self.last_emit_time = 0.0
        self.cooldown = cooldown

    async def _ensure_cache(self) -> RedisCache:
        if self.cache.client is None:
            await self.cache.connect()
        return self.cache

    def calculate_entropy(self, detections: List[Any], density_max: int = 10) -> float:
        """
        計算影格資訊熵，結合 Shannon Entropy（類別分佈）與 Object Density（物體密度）。

        detections 元素可為：
          - 有 class_id / label 屬性的偵測物件
          - 字串類別標籤
        """
        if not detections:
            return 0.0

        # 提取類別標籤
        labels: List[Any] = []
        for d in detections:
            if hasattr(d, "class_id"):
                labels.append(d.class_id)
            elif hasattr(d, "label"):
                labels.append(d.label)
            else:
                labels.append(str(d))

        n = len(labels)

        # Shannon entropy over class distribution, normalized to [0, 1]
        counts: dict[int, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1

        raw_entropy = 0.0
        for count in counts.values():
            p = count / n
            raw_entropy -= p * math.log2(p)

        n_classes = len(counts)
        max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0
        shannon_score = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        # Object density: linear scale up to density_max objects
        density_score = min(n / density_max, 1.0)

        return 0.5 * shannon_score + 0.5 * density_score

    async def emit_event(
        self, entropy_value: float, frame_id: int, source_path: str, objects: List[str]
    ) -> bool:
        """
        按照 docs/api_spec.md 規範發布事件至 Redis
        """
        event_id = str(uuid.uuid4())
        timestamp = time.time()

        event_data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "type": "entropy_trigger",
            "metadata": {
                "entropy_value": round(entropy_value, 3),
                "source_path": source_path,
                "frame_id": frame_id,
                "objects": objects,
            },
        }

        cache = await self._ensure_cache()
        assert cache.client is not None
        try:
            await cache.client.rpush("saccade:events", json.dumps(event_data))
            await cache.client.expire("saccade:events", 3600)

            print(
                f"📡 Event emitted: {event_id} (Entropy: {entropy_value:.2f}, Frame: {frame_id})"
            )
            return True
        except Exception as e:
            print(f"❌ Failed to emit event: {str(e)}")
            return False

    async def process_frame(
        self, frame_id: int, detections: List[Any], source_path: str
    ) -> bool:
        """
        處理單個影格的邏輯 (增加冷卻時間檢查)
        """
        current_time = time.time()
        if current_time - self.last_emit_time < self.cooldown:
            return False

        entropy = self.calculate_entropy(detections)

        if entropy >= self.threshold:
            objects = [str(d) for d in detections]  # 簡化轉字串
            success = await self.emit_event(entropy, frame_id, source_path, objects)
            if success:
                self.last_emit_time = current_time
            return success

        return False

    async def close(self) -> None:
        await self.cache.disconnect()


async def main() -> None:
    # 測試執行
    trigger = EntropyTrigger(threshold=0.5)
    # 模擬偵測到三個物體，觸發事件
    await trigger.process_frame(
        frame_id=1001,
        detections=["person", "car", "dog"],
        source_path=build_rtsp_url(DEFAULT_RTSP_SINGLE_STREAM_PATH),
    )
    await trigger.close()


if __name__ == "__main__":
    asyncio.run(main())
