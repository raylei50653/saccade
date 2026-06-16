"""
Saccade CLI Entrypoint

此模組僅作為應用程式的 CLI 進入點（Entrypoint）。
其主要職責為：
1. 解析命令列參數（CLI Arguments）。
2. 初始化環境變數與全局設定。
3. 根據指定的模式（如 perception, orchestrator）實例化並啟動對應的核心邏輯。

注意：核心的系統調度、非同步事件循環與各層級之間的資料流動，皆由 `src/saccade/cognition/orchestrator.py` 負責處理。本檔案不應包含任何業務邏輯或感知流程細節。
"""

import os
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
import time

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("⚡ [System] Fast Event Loop (uvloop) & Single-thread NumPy config enabled.")
except ImportError:
    print("⚠️  [System] uvloop not available, falling back to default event loop.")

from typing import Optional, List, Dict, Any, cast  # noqa: E402
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402
from saccade.media.mediamtx_client import MediaMTXClient  # noqa: E402
from saccade.media.dali_pipeline import DALIMediaClient  # noqa: E402
from saccade.perception.dispatcher import AsyncDispatcher  # noqa: E402
from saccade.perception.embedding_dispatcher import (  # noqa: E402
    AsyncEmbeddingDispatcher as EmbeddingDispatcher,
)
from saccade.perception.drift_handler import SemanticDriftHandler  # noqa: E402
from saccade.resource.resource_manager import ResourceManager  # noqa: E402
from saccade.storage.redis_cache import RedisCache  # noqa: E402
from saccade.cognition.orchestrator import PipelineOrchestrator  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# 全域資源
resource_manager = ResourceManager()
redis_cache = RedisCache()
drift_handler = SemanticDriftHandler()


async def on_detection_finished(
    stream_id: str,
    ts: float,
    ids: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    kps: Optional[torch.Tensor],
) -> None:
    """橋接回調：L1 (YOLO) -> downstream. Embedding 路徑由 EmbeddingDispatcher 獨立處理。"""
    pass


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

    # 啟動自動重連監控
    asyncio.create_task(media.watchdog_loop())

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
    """感知層：多路並行處理 (Async-Batching Dispatcher + ReID)"""
    global _embedding_dispatcher
    print("🚀 Initializing Multi-stream Perception Pipeline...")

    detector = TRTYoloDetector()

    # L2 ReID 組件（選配，缺少 engine 時退化為純 IoU）
    extractor: Optional[TRTFeatureExtractor] = None
    try:
        extractor = TRTFeatureExtractor(max_batch=64)
        print("✅ [ReID] SigLIP 2 extractor ready.")
        _embedding_dispatcher = EmbeddingDispatcher(
            extractor, on_embeddings_ready=on_embeddings_ready
        )
        _embedding_dispatcher.start()
    except Exception as e:
        print(
            f"⚠️  [ReID] Extractor unavailable ({e}), falling back to IoU-only tracking."
        )

    dispatcher = AsyncDispatcher(
        detector,
        resource_manager,
        extractor=extractor,
        heartbeat_interval=10,
        max_batch=8,
        on_track_result=on_detection_finished,
    )
    # Note: in perception/dispatcher.py, start() is async and doesn't take callback.
    # It calls on_track_result which we can set in __init__.

    await dispatcher.start()

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
        await dispatcher.stop()
        print("🛑 Perception Pipeline shutting down...")


async def run_full() -> None:
    print("💡 Running in full mode - starting perception + orchestrator.")
    await asyncio.gather(
        run_perception(),
        PipelineOrchestrator().run(),
    )


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
        asyncio.run(run_full())


if __name__ == "__main__":
    main()
