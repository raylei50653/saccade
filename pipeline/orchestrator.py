import asyncio
import json
import os
import cv2
import base64
import time
import redis.asyncio as redis
from typing import Dict, Any, Optional, cast, Awaitable
from cognition.llm_engine import LLMEngine
from media.mediamtx_client import MediaMTXClient
from storage.chroma_store import ChromaStore
from pipeline.health import HealthChecker, render
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class PipelineOrchestrator:
    """
    Saccade 雙軌管線調度器
    
    負責監聽 Perception 的觸發事件，並調度 Cognition (VLM) 進行深度視覺推理。
    將分析結果存儲於 ChromaDB 向量記憶庫中。
    """
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.rtsp_url = os.getenv("MEDIAMTX_URL", "rtsp://localhost:8554/live")
        self.use_local = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"
        self.dummy_video = os.getenv("DUMMY_VIDEO_PATH")
        
        # 初始化客戶端
        self.media_client = MediaMTXClient(
            self.rtsp_url, 
            use_local=self.use_local, 
            dummy_video=self.dummy_video
        )
        self.llm_engine = LLMEngine()
        self.redis_client = redis.from_url(self.redis_url)
        self.memory_store = ChromaStore()

    async def _encode_image(self, frame: np.ndarray) -> str:
        """將 OpenCV 影格轉換為 Base64 字串以傳輸至 VLM"""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    async def handle_cognitive_event(self, event_data: Dict[str, Any]) -> None:
        """處理來自 Redis 的感知觸發事件 (慢路徑)"""
        frame_id = event_data["metadata"]["frame_id"]
        entropy = event_data["metadata"]["entropy_value"]

        print(f"🧠 [Cognition] High-value event received (Frame: {frame_id}, Entropy: {entropy})")

        # 1. 抓取關鍵影格
        ret, frame = self.media_client.grab_frame()
        if not ret or frame is None:
            print(f"❌ [Cognition] Failed to grab frame {frame_id} from stream.")
            return

        # 2. 準備 VLM 推理
        labels = ", ".join(event_data["metadata"]["objects"])
        prompt = (
            f"You are a security AI. The following objects were detected via YOLO: {labels}. "
            "Examine the image carefully. Describe the scene, the interactions between these objects, "
            "and identify any unusual activity or safety risks. Be concise and professional."
        )

        # 3. 執行視覺推理 (VLM)
        image_b64 = await self._encode_image(frame)
        response = await self.llm_engine.generate(prompt, image_data=image_b64)

        print(f"💡 [VLM Result] {response}")

        # 4. 寫入狀態層 (ChromaDB 儲存)
        try:
            self.memory_store.add_memory(
                content=response,
                metadata={
                    "frame_id": frame_id,
                    "entropy": entropy,
                    "objects": labels,
                    "timestamp": time.time()
                }
            )
            print(f"💾 [Storage] Analysis saved to ChromaDB.")
        except Exception as e:
            print(f"❌ [Storage] Failed to save memory: {e}")

    async def start_cognition_loop(self) -> None:
        """啟動慢路徑事件監聽循環"""
        print("🚀 [Cognition Loop] Listening for events on Redis...")
        while True:
            try:
                # 使用 BLPOP 進行阻塞式監聽，減少 CPU 負擔
                result = await cast(Awaitable[Optional[list[Any]]], self.redis_client.blpop("saccade:events", timeout=0))
                if result:
                    _, raw_event = result
                    event_data = json.loads(raw_event)
                    # 在背景執行推理，不阻塞事件監聽
                    asyncio.create_task(self.handle_cognitive_event(event_data))
                
            except Exception as e:
                print(f"⚠️ [Cognition Loop] Error: {str(e)}")
                await asyncio.sleep(1)

    async def run(self) -> None:
        """啟動全域管線協調"""
        # 1. 啟動前健康檢查
        checker = HealthChecker()
        report = await checker.run()
        print(render(report))
        
        if not report.overall_ok:
            print("❌ System check failed. Please check infrastructure status.")
            # 視需求決定是否退出
            # return

        # 2. 建立串流連線 (增加重試機制)
        source_name = "Local Camera" if self.use_local else self.rtsp_url
        while not self.media_client.connect():
            print(f"⏳ Waiting for media source {source_name}... retrying in 2s")
            await asyncio.sleep(2)
        
        print(f"✅ Orchestrator connected to {source_name}")

        # 3. 啟動雙軌協調
        # 目前 Perception 單獨作為 Systemd 服務運行，Orchestrator 專注於事件調度
        try:
            await self.start_cognition_loop()
        finally:
            self.media_client.release()
            await cast(Awaitable[Any], self.redis_client.aclose())

async def main() -> None:
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
