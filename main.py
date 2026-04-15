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
import time
import torch.multiprocessing as mp

# CUDA 必須使用 spawn 模式
try:
    mp.set_start_method("spawn", force=True)
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

from typing import Optional, List, Tuple, Any, Dict, cast
from perception.dispatcher import AsyncDispatcher
from perception.embedding_dispatcher import EmbeddingDispatcher
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor
from perception.detector_trt import TRTYoloDetector
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from media.dali_pipeline import DALIMediaClient
from cognition.resource_manager import ResourceManager
from storage.redis_cache import RedisCache
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

load_dotenv()

# 全域資源
resource_manager = ResourceManager()
redis_cache = RedisCache()
drift_handler = SemanticDriftHandler()
cropper = ZeroCopyCropper(output_size=(224, 224))


async def on_detection_finished(
    stream_id: str,
    results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any],
    frame_tensor: torch.Tensor,
    timestamp: float,
) -> None:
    """
    橋接回調：L1 (YOLO) -> L2 (Embedding)
    """
    boxes, scores, classes, _ = results
    if boxes.size(0) == 0:
        return

    # 1. 執行 L2 過濾 (漂移檢測)
    # 這裡我們在主 Loop 中進行快速過濾，決定哪些物件需要「重度嵌入」

    # 為了簡化展示，我們假設所有物件都送去嵌入，
    # 實際生產環境會在這裡調用 drift_handler.filter_for_batch

    # 2. 執行零拷貝裁切 [N, 3, 224, 224]
    # 注意：frame_tensor 需為 [1, 3, H, W]
    crops = cropper.process(frame_tensor.unsqueeze(0), boxes)

    # 3. 推入 Embedding 分發器
    metadata = [
        {"frame_id": int(timestamp * 1000), "cls": str(cls.item()), "track_id": i}
        for i, cls in enumerate(classes)
    ]

    # 取得全域的 embedding_dispatcher (在 run_perception 中初始化)
    if _embedding_dispatcher is not None:
        await _embedding_dispatcher.put_crops(stream_id, crops, metadata)


async def on_embeddings_ready(
    stream_id: str, embeddings: np.ndarray, metadata: List[Dict[str, Any]]
) -> None:
    """
    橋接回調：L2 (SigLIP2) -> L3 (Redis Stream) (Batch Optimized)
    """
    level = resource_manager.current_level
    events = []

    for i, emb_np in enumerate(embeddings):
        emb_tensor = torch.from_numpy(emb_np).to("cuda")
        track_id = metadata[i]["track_id"]

        # 再次確認語義漂移 (精確判定)
        sim, should_persist = drift_handler.calculate_drift(track_id, emb_tensor, level)
        drift_handler.update_history([track_id], emb_tensor.unsqueeze(0), level)

        if should_persist:
            event = {
                "stream_id": stream_id,
                "metadata": {
                    "frame_id": metadata[i]["frame_id"],
                    "track_id": track_id,
                    "objects": [metadata[i]["cls"]],
                    "entropy_value": 0.9,  # 佔位
                    "similarity": float(sim),
                },
            }
            events.append(event)

    # 🚀 批次寫入 Redis Stream (D 優化)
    if events:
        await redis_cache.add_to_stream_batch(events)


_embedding_dispatcher: Optional[EmbeddingDispatcher] = None


async def run_stream_producer(
    stream_id: str, dispatcher: AsyncDispatcher, source_url: Optional[str] = None
) -> None:
    """
    單路串流生產者：負責抓圖並推入分發器 (DALI Optimized)
    """
    # 判斷是否為影片檔案以決定是否使用 DALI
    is_file = source_url and os.path.isfile(source_url)

    if is_file:
        print(
            f"🎬 [Stream {stream_id}] Using DALI GPU-Preprocessing for file: {source_url}"
        )
        media: Any = DALIMediaClient(video_path=cast(str, source_url), batch_size=1)
    else:
        print(f"📡 [Stream {stream_id}] Using MediaMTXClient for source: {source_url}")
        media = MediaMTXClient(dummy_video=source_url)

    while not media.connect():
        await asyncio.sleep(2)

    print(f"✅ Stream [{stream_id}] connected.")

    try:
        while True:
            ret, tensor = media.grab_tensor()
            if ret and tensor is not None:
                if is_file:
                    # DALI 已經完成 [3, 640, 640] float32 [0,1] 的預處理 (來自 [1, 3, 640, 640])
                    yolo_input = tensor.squeeze(0)
                else:
                    # 原有 MediaMTXClient 輸出為 [H, W, 3] uint8，需進行預處理 (佔用 CPU/GPU 同步)
                    input_tensor = tensor.permute(2, 0, 1).float() / 255.0
                    yolo_input = torch.nn.functional.interpolate(
                        input_tensor.unsqueeze(0), size=(640, 640)
                    ).squeeze(0)

                await dispatcher.put_frame(stream_id, yolo_input, time.time())

            await asyncio.sleep(0.01)  # 控頻 (約 100 FPS)
    finally:
        media.release()


async def run_perception() -> None:
    """感知層：極速雙路並行 (Industrial Pipeline)"""
    global _embedding_dispatcher
    print("🚀 Initializing Optimized Multi-stream Perception Pipeline...")

    # 初始化偵測器與嵌入器
    detector = TRTYoloDetector()
    extractor = TRTFeatureExtractor(max_batch=64)

    # 啟動雙路分發器
    _embedding_dispatcher = EmbeddingDispatcher(extractor, max_batch=64)
    _embedding_dispatcher.start(callback=on_embeddings_ready)

    # YOLO 分發器：傳入橋接回調，實現 L1 -> L2 連動
    dispatcher = AsyncDispatcher(detector, max_batch=8)
    dispatcher.start(callback=on_detection_finished)

    # 模擬 4 路串流
    streams = ["stream_1", "stream_2", "stream_3", "stream_4"]
    tasks = []

    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")

    for sid in streams:
        tasks.append(
            asyncio.create_task(run_stream_producer(sid, dispatcher, dummy_video))
        )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        dispatcher.stop()
        _embedding_dispatcher.stop()
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
