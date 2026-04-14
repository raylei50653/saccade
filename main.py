"""
Saccade CLI Entrypoint

此模組僅作為應用程式的 CLI 進入點（Entrypoint）。
其主要職責為：
1. 解析命令列參數（CLI Arguments）。
2. 初始化環境變數與全局設定。
3. 根據指定的模式（如 perception, orchestrator）實例化並啟動對應的核心邏輯。

注意：核心的系統調度、非同步事件循環與各層級之間的資料流動，皆由 `pipeline/orchestrator.py` 負責處理。本檔案不應包含任何業務邏輯或感知流程細節。
"""

import os
import torch.multiprocessing as mp
# CUDA 必須使用 spawn 模式
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 必須在 import numpy / torch 之前設定，防止執行緒暴風
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import asyncio
import argparse
import torch
import numpy as np
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("⚡ [System] Fast Event Loop (uvloop) & Single-thread NumPy config enabled.")
except ImportError:
    pass

from typing import Optional, List
from perception.detector_trt import TRTYoloDetector
from perception.entropy import EntropyTrigger
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor
from perception.tracker import SmartTracker
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from storage.redis_cache import RedisCache
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

from media.ffmpeg_utils import RTSPStreamer

load_dotenv()


from perception.dispatcher import AsyncDispatcher

async def run_stream_producer(stream_id: str, dispatcher: AsyncDispatcher, source_url: Optional[str] = None) -> None:
    """
    單路串流生產者：負責抓圖並推入分發器
    """
    media = MediaMTXClient(dummy_video=source_url)
    while not media.connect():
        await asyncio.sleep(2)
    
    print(f"✅ Stream [{stream_id}] connected.")
    
    try:
        while True:
            ret, tensor = media.grab_tensor()
            if ret and tensor is not None:
                # [H, W, C] -> [3, 640, 640]
                input_tensor = tensor.permute(2, 0, 1).float() / 255.0
                yolo_input = torch.nn.functional.interpolate(
                    input_tensor.unsqueeze(0), size=(640, 640)
                ).squeeze(0)
                
                await dispatcher.put_frame(stream_id, yolo_input, time.time())
            
            await asyncio.sleep(0.01) # 控頻
    finally:
        media.release()

async def run_perception() -> None:
    """感知層：多路並行處理 (Async-Batching Dispatcher)"""
    print("🚀 Initializing Multi-stream Perception Pipeline...")
    
    detector = TRTYoloDetector()
    dispatcher = AsyncDispatcher(detector, max_batch=8)
    dispatcher.start()

    # 模擬 4 路串流 (可從環境變數配置)
    streams = ["stream_1", "stream_2", "stream_3", "stream_4"]
    tasks = []
    
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    
    for sid in streams:
        tasks.append(asyncio.create_task(run_stream_producer(sid, dispatcher, dummy_video)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        dispatcher.stop()
        print("🛑 Perception Pipeline shutting down...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saccade - Dual-Track Video Perception"
    )
    parser.add_argument(
        "--mode", choices=["perception", "orchestrator", "full"], default="full"
    )
    args = parser.parse_args()

    if args.mode == "perception":
        asyncio.run(run_perception())
    elif args.mode == "orchestrator":
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())
    else:
        print("💡 Running in full mode - starting orchestrator.")
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
