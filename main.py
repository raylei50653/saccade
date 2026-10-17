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

from typing import Any, Awaitable, Callable, Coroutine, Optional, Sequence, cast  # noqa: E402
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402
from saccade.media.mediamtx_client import MediaMTXClient  # noqa: E402
from saccade.media.dali_pipeline import DALIMediaClient  # noqa: E402
from saccade.perception.dispatcher import (  # noqa: E402
    AsyncDispatcher,
    DispatcherFailed,
    DispatcherState,
)
from saccade.resource.resource_manager import ResourceManager  # noqa: E402
from saccade.cognition.orchestrator import PipelineOrchestrator  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# 全域資源
resource_manager = ResourceManager()

# #537 R-1: the online path tracks geometry only. The SigLIP extractor and the
# embedding→Redis bridge were never fed by this path, so they are not started.
REID_UNSUPPORTED = "ReID is unsupported in the online tracking path (#537 R-1)"


async def on_detection_finished(
    stream_id: str,
    ts: float,
    ids: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    kps: Optional[torch.Tensor],
) -> None:
    """橋接回調：L1 (YOLO) -> downstream。

    Still a no-op: tracking results go nowhere, so ``--mode full`` does not
    connect detection→embedding→Redis→cognition (#537 R-1).
    """
    pass


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


def same_failure(a: BaseException, b: BaseException) -> bool:
    """Both report the same underlying worker failure (same ``__cause__``)."""
    return a is b or (a.__cause__ is not None and a.__cause__ is b.__cause__)


async def supervise(
    tasks: Sequence["asyncio.Task[Any]"],
    watchers: Sequence["asyncio.Task[Any]"] = (),
) -> None:
    """Wait until every task in ``tasks`` ends (#538 caller lifecycle).

    The first failure in ``tasks`` or ``watchers`` cancels everything else and
    is re-raised once all of it has finished; later failures become notes.
    Cancelling the caller cancels everything too.
    """
    everything = [*tasks, *watchers]
    first: Optional[BaseException] = None
    try:
        pending = set(everything)
        while first is None and any(t in pending for t in tasks):
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if first is None and not t.cancelled() and t.exception() is not None:
                    first = t.exception()
    finally:
        for t in everything:
            t.cancel()
        await asyncio.gather(*everything, return_exceptions=True)
    if first is not None:
        for t in everything:
            other = None if t.cancelled() else t.exception()
            if other is not None and not same_failure(other, first):
                first.add_note(f"also failed: {other!r}")
        raise first


async def raise_when_failed(dispatcher: AsyncDispatcher, poll_s: float = 0.1) -> None:
    """Surface a dispatcher worker failure even when no producer is sending."""
    while dispatcher.state is not DispatcherState.FAILED:
        await asyncio.sleep(poll_s)
    raise DispatcherFailed(
        f"AsyncDispatcher worker failed: {dispatcher.failure!r}"
    ) from dispatcher.failure


async def stop_after(
    primary: Optional[BaseException], step: Callable[[], Awaitable[None]], name: str
) -> None:
    """Run a cleanup step. With a primary error, a cleanup error becomes its note."""
    if primary is None:
        await step()
        return
    try:
        await step()
    except Exception as cleanup_error:
        if not same_failure(cleanup_error, primary):
            primary.add_note(f"cleanup {name} also failed: {cleanup_error!r}")
        else:  # the same failure re-raised; keep what its cleanup reported
            for note in getattr(cleanup_error, "__notes__", []):
                primary.add_note(f"cleanup {name}: {note}")


async def run_perception(
    *,
    detector_factory: Callable[[], Any] = TRTYoloDetector,
    producer: Optional[Callable[..., Coroutine[Any, Any, None]]] = None,
    streams: Sequence[str] = ("stream_1", "stream_2", "stream_3", "stream_4"),
    source: Optional[str] = None,
) -> None:
    """感知層：多路並行處理 (Async-Batching Dispatcher, geometry-only tracking)

    The dispatcher is stopped on every exit (normal end, worker failure,
    producer error, cancellation). The first error is re-raised; a cleanup
    error never replaces it.
    """
    producer = producer or run_stream_producer
    print("🚀 Initializing Multi-stream Perception Pipeline...")
    print(f"ℹ️  [Perception] online tracking = geometry-only; {REID_UNSUPPORTED}.")

    detector = detector_factory()
    dispatcher = AsyncDispatcher(
        detector,
        resource_manager,
        heartbeat_interval=10,
        max_batch=8,
        on_track_result=on_detection_finished,
    )
    await dispatcher.start()

    if source is None:
        source = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    tasks = [asyncio.create_task(producer(sid, dispatcher, source)) for sid in streams]
    watcher = asyncio.create_task(raise_when_failed(dispatcher))

    try:
        await supervise(tasks, [watcher])
    except BaseException as primary:
        await stop_after(primary, dispatcher.stop, "dispatcher.stop()")
        print("🛑 Perception Pipeline shutting down...")
        raise
    await dispatcher.stop()


async def run_full(
    perception: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
    orchestrator_factory: Callable[[], Any] = PipelineOrchestrator,
) -> None:
    """Perception + orchestrator; if either fails, the other is cancelled."""
    print("💡 Running in full mode - starting perception + orchestrator.")
    await supervise(
        [
            asyncio.create_task((perception or run_perception)()),
            asyncio.create_task(orchestrator_factory().run()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saccade - Dual-Track Video Perception"
    )
    parser.add_argument(
        "--mode", choices=["perception", "orchestrator", "full"], default="full"
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Not supported: the online tracking path is geometry-only (#537 R-1).",
    )
    args = parser.parse_args()
    if args.reid and args.mode != "orchestrator":
        parser.error(REID_UNSUPPORTED)

    if args.mode == "perception":
        asyncio.run(run_perception())
    elif args.mode == "orchestrator":
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())
    else:
        asyncio.run(run_full())


if __name__ == "__main__":
    main()
