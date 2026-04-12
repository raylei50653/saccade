import json
import time
import uuid
import asyncio
import os
import redis.asyncio as redis
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class EntropyTrigger:
    """
    資訊熵觸發器 (Perception 快路徑)
    
    負責評估影格價值，並在達到閾值時向 Redis 發布事件以觸發慢路徑 (Cognition)。
    """
    def __init__(self, threshold: float = 0.8, redis_url: str = REDIS_URL):
        self.threshold = threshold
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

    async def _ensure_redis(self) -> redis.Redis:
        """確保 Redis 連線已建立"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url)
        return self.redis_client

    def calculate_entropy(self, detections: List[Any]) -> float:
        """
        計算影格資訊熵 (目前的簡化實作)
        
        可以根據偵測到的物體數量、類別分佈或邊界框的位移量來計算。
        """
        # TODO: 實作基於 Shannon Entropy 或 Object Density 的計算邏輯
        # 這裡先用簡單的偵測物體數量模擬
        if not detections:
            return 0.0
        
        # 模擬：偵測到越多物體，熵值越高
        score = len(detections) * 0.2
        return min(score, 1.0)

    async def emit_event(self, entropy_value: float, frame_id: int, source_path: str, objects: List[str]) -> bool:
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
                "objects": objects
            }
        }

        r = await self._ensure_redis()
        try:
            # 推送到 Redis List (saccade:events)
            await r.rpush("saccade:events", json.dumps(event_data))
            # 設定過期時間 (TTL)，防止 Redis 記憶體溢出
            await r.expire("saccade:events", 3600) 
            
            print(f"📡 Event emitted: {event_id} (Entropy: {entropy_value:.2f}, Frame: {frame_id})")
            return True
        except Exception as e:
            print(f"❌ Failed to emit event: {str(e)}")
            return False

    async def process_frame(self, frame_id: int, detections: List[Any], source_path: str) -> bool:
        """
        處理單個影格的邏輯
        """
        entropy = self.calculate_entropy(detections)
        
        if entropy >= self.threshold:
            objects = [str(d) for d in detections] # 簡化轉字串
            return await self.emit_event(entropy, frame_id, source_path, objects)
        
        return False

    async def close(self):
        """關閉連線"""
        if self.redis_client:
            await self.redis_client.aclose()

if __name__ == "__main__":
    # 測試執行
    async def main():
        trigger = EntropyTrigger(threshold=0.5)
        # 模擬偵測到三個物體，觸發事件
        await trigger.process_frame(
            frame_id=1001, 
            detections=["person", "car", "dog"], 
            source_path="rtsp://localhost:8554/live"
        )
        await trigger.close()

    asyncio.run(main())
