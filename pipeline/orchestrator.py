import asyncio
import json
import os
import time
import redis.asyncio as redis
from typing import Dict, Any, Optional, cast, Awaitable, List
from storage.chroma_store import ChromaStore
from pipeline.health import HealthChecker, render
from dotenv import load_dotenv

load_dotenv()


class PipelineOrchestrator:
    """
    Saccade 純視覺向量調度器 (Vision-Only Architecture)

    接收 YOLO 的高頻事件，將結構化數據轉化為語義記憶，存入 ChromaDB。
    """

    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(self.redis_url)
        self.memory_store = ChromaStore()

        # 由於拔除 VLM，並發控制只需限制資料庫寫入頻率
        self.semaphore = asyncio.Semaphore(32)

    def _generate_scene_description(self, objects: List[str], entropy: float) -> str:
        """基於 YOLO 標籤生成結構化的場景描述，供 ChromaDB Embedding 使用"""
        if not objects:
            return "Empty scene."

        # 簡單的物件計數
        obj_counts: Dict[str, int] = {}
        for obj in objects:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1

        desc_parts = []
        for obj, count in obj_counts.items():
            desc_parts.append(f"{count} {obj}{'s' if count > 1 else ''}")

        base_desc = "Scene contains: " + ", ".join(desc_parts) + "."

        # 根據 Entropy 附加動態標籤
        if entropy > 0.8:
            base_desc += " High dynamic activity or complex scene detected."

        return base_desc

    async def handle_cognitive_event(self, event_data: Dict[str, Any]) -> None:
        """處理 YOLO 觸發的事件並持久化"""
        async with self.semaphore:
            metadata = event_data.get("metadata", {})
            frame_id = metadata.get("frame_id", 0)
            entropy = metadata.get("entropy_value", 0.0)
            yolo_objects = metadata.get("objects", [])

            # 1. 生成場景語義描述
            scene_description = self._generate_scene_description(yolo_objects, entropy)

            # 2. 定義異常邏輯 (基於規則)
            # 例如：如果在不該出現人的地方出現人，或是偵測到特定危險物品
            risk_objects = {"knife", "gun", "fire", "smoke", "accident"}
            is_anomaly = any(obj.lower() in risk_objects for obj in yolo_objects)

            # 3. 寫入 ChromaDB
            try:
                self.memory_store.add_memory(
                    content=scene_description,
                    metadata={
                        "frame_id": frame_id,
                        "entropy": entropy,
                        "objects": ", ".join(yolo_objects),
                        "is_anomaly": 1 if is_anomaly else 0,
                        "timestamp": time.time(),
                    },
                )
                status_tag = "🚨" if is_anomaly else "✅"
                print(f"{status_tag} [Frame {frame_id}] Indexed: {scene_description}")
            except Exception as e:
                print(f"❌ [Storage Error] {e}")

    async def start_cognition_loop(self) -> None:
        print("🚀 [Orchestrator] High-Speed Vector Indexing Loop Active...")
        while True:
            try:
                result = await cast(
                    Awaitable[Optional[list[Any]]],
                    self.redis_client.blpop("saccade:events", timeout=0),
                )
                if result:
                    _, raw_event = result
                    event_data = json.loads(raw_event)
                    asyncio.create_task(self.handle_cognitive_event(event_data))
            except Exception as e:
                print(f"⚠️ [Loop Error] {e}")
                await asyncio.sleep(1)

    async def run(self) -> None:
        checker = HealthChecker()
        report = await checker.run()
        print(render(report))

        try:
            await self.start_cognition_loop()
        finally:
            await cast(Awaitable[Any], self.redis_client.aclose())


async def main() -> None:
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
