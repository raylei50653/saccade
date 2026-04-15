import asyncio
import time
import uuid
from typing import Dict, Any, List, Tuple, Optional, cast, Awaitable
from storage.chroma_store import ChromaStore
from storage.redis_cache import RedisCache
from pipeline.health import HealthChecker, render
from dotenv import load_dotenv

load_dotenv()


class PipelineOrchestrator:
    """
    Saccade 高效能編排器 (Micro-batching Edition)

    1. 從 Redis Stream 批次撈取事件 (L3 -> L4)。
    2. 執行向量化寫入 ChromaDB，最大化 I/O 效率。
    3. 整合健康檢查與多路併發控制。
    """

    def __init__(self) -> None:
        self.redis_cache = RedisCache()
        self.memory_store = ChromaStore()
        # 微批次大小
        self.batch_size = 100
        # 併發控制 (用於處理多個微批次)
        self.semaphore = asyncio.Semaphore(16)

    def _generate_scene_description(self, objects: List[str], entropy: float) -> str:
        """基於 YOLO 標籤生成結構化的場景描述"""
        if not objects:
            return "Empty scene."

        obj_counts: Dict[str, int] = {}
        for obj in objects:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1

        desc_parts = []
        for obj, count in obj_counts.items():
            desc_parts.append(f"{count} {obj}{'s' if count > 1 else ''}")

        base_desc = "Scene contains: " + ", ".join(desc_parts) + "."
        if entropy > 0.8:
            base_desc += " High dynamic activity detected."
        return base_desc

    async def process_event_batch(
        self, batch: List[Tuple[str, Dict[str, Any]]]
    ) -> None:
        """處理一組微批次事件並寫入 ChromaDB"""
        if not batch:
            return

        msg_ids, events = zip(*batch)
        contents = []
        metadatas = []
        ids = []

        for event in events:
            metadata = event.get("metadata", {})
            frame_id = metadata.get("frame_id", 0)
            entropy = metadata.get("entropy_value", 0.0)
            yolo_objects = metadata.get("objects", [])

            # 1. 生成場景描述
            scene_description = self._generate_scene_description(yolo_objects, entropy)

            # 2. 異常檢測
            risk_objects = {"knife", "gun", "fire", "smoke", "accident"}
            is_anomaly = any(obj.lower() in risk_objects for obj in yolo_objects)

            contents.append(scene_description)
            metadatas.append(
                {
                    "frame_id": frame_id,
                    "entropy": entropy,
                    "objects": ", ".join(yolo_objects),
                    "is_anomaly": 1 if is_anomaly else 0,
                    "timestamp": time.time(),
                }
            )
            ids.append(str(uuid.uuid4()))

        # 3. 向量化寫入 (Vectorized Write)
        try:
            # 透過 to_thread 執行同步的 ChromaDB 批量操作
            await asyncio.to_thread(
                self.memory_store.collection.add,
                documents=contents,
                metadatas=cast(Any, metadatas),
                ids=ids,
            )
            # 4. 確認訊息已處理 (ACK)
            await self.redis_cache.acknowledge(list(msg_ids))
            print(
                f"📦 [Orchestrator] Micro-batch committed: {len(events)} events indexed."
            )
        except Exception as e:
            print(f"❌ [Storage Error] {e}")

    async def start_cognition_loop(self) -> None:
        print(
            "🚀 [Orchestrator] High-Speed Stream Indexing Loop Active (Micro-batching)..."
        )
        await self.redis_cache.connect()

        while True:
            try:
                # 1. 從 Redis Stream 撈取批次
                batch = await self.redis_cache.read_stream_batch(count=self.batch_size)

                if batch:
                    # 2. 異步處理該批次
                    asyncio.create_task(self.process_event_batch(batch))
                else:
                    # 沒資料時稍微休息
                    await asyncio.sleep(0.1)
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
            await self.redis_cache.disconnect()


async def main() -> None:
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
