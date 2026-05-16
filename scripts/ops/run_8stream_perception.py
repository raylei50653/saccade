import argparse
import asyncio
import time
from typing import Optional

import torch

from saccade.media.mediamtx_client import MediaMTXClient
from saccade.media.rtsp import (
    DEFAULT_RTSP_HOST,
    DEFAULT_RTSP_PORT,
    DEFAULT_RTSP_READ_PASSWORD,
    DEFAULT_RTSP_READ_USER,
    DEFAULT_RTSP_STREAM_PREFIX,
    build_reader_url,
    build_stream_path,
)
from saccade.perception.detector_trt import TRTYoloDetector
from saccade.perception.dispatcher import AsyncDispatcher
from saccade.resource.resource_manager import ResourceManager


async def run_stream_producer(
    stream_id: str,
    dispatcher: AsyncDispatcher,
    rtsp_url: str,
) -> None:
    media = MediaMTXClient(rtsp_url=rtsp_url)

    print(f"⏳ Connecting to {rtsp_url}...")
    while not media.connect():
        await asyncio.sleep(1)

    print(f"✅ Stream [{stream_id}] connected.")

    target_hw = dispatcher.input_hw
    try:
        last_frame_time = time.time()
        while True:
            ret, tensor = media.grab_tensor()
            if ret and tensor is not None:
                input_tensor = tensor.permute(2, 0, 1).float() / 255.0
                yolo_input = torch.nn.functional.interpolate(
                    input_tensor.unsqueeze(0), size=target_hw
                ).squeeze(0)

                await dispatcher.put_frame(stream_id, yolo_input, time.time())
                last_frame_time = time.time()
            else:
                if time.time() - last_frame_time > 5.0:
                    print(f"⚠️ Stream [{stream_id}] has not received frames for 5s.")
                    last_frame_time = time.time()

            await asyncio.sleep(0.001)
    except Exception as e:
        print(f"❌ Stream [{stream_id}] error: {e}")
    finally:
        media.release()


async def run_stream_producer_workbench(
    stream_id: str,
    pool: "WorkbenchPool",  # noqa: F821
    rtsp_url: str,
) -> None:
    """Async RTSP producer that pushes to a WorkbenchPool per-stream queue."""
    media = MediaMTXClient(rtsp_url=rtsp_url)

    print(f"⏳ Connecting to {rtsp_url}...")
    while not media.connect():
        await asyncio.sleep(1)

    print(f"✅ Stream [{stream_id}] connected (workbench mode).")

    try:
        last_frame_time = time.time()
        while True:
            ret, tensor = media.grab_tensor()
            if ret and tensor is not None:
                # Pass HWC tensor directly; workbench worker handles letterbox
                pool.put_frame(stream_id, tensor, time.time())
                last_frame_time = time.time()
            else:
                if time.time() - last_frame_time > 5.0:
                    print(f"⚠️ Stream [{stream_id}] has not received frames for 5s.")
                    last_frame_time = time.time()

            await asyncio.sleep(0.001)
    except Exception as e:
        print(f"❌ Stream [{stream_id}] error: {e}")
    finally:
        media.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-stream perception demo.")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=float, default=3.0)
    parser.add_argument("--engine", default="models/yolo/yolo26m_960_batch16.engine")
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument(
        "--frame-width", type=int, default=1920, help="Source frame width"
    )
    parser.add_argument(
        "--frame-height", type=int, default=1080, help="Source frame height"
    )
    parser.add_argument("--stats-interval-sec", type=float, default=5.0)
    parser.add_argument(
        "--workbench",
        action="store_true",
        default=False,
        help="Use per-thread WorkbenchPool (GIL-free, cross-thread parallel)",
    )
    parser.add_argument(
        "--rtsp-prefix",
        default=None,
        help="Deprecated raw prefix. Prefer --rtsp-host/--rtsp-port/--rtsp-user/--rtsp-password/--rtsp-path-prefix.",
    )
    parser.add_argument("--rtsp-host", default=DEFAULT_RTSP_HOST)
    parser.add_argument("--rtsp-port", type=int, default=DEFAULT_RTSP_PORT)
    parser.add_argument("--rtsp-user", default=DEFAULT_RTSP_READ_USER)
    parser.add_argument("--rtsp-password", default=DEFAULT_RTSP_READ_PASSWORD)
    parser.add_argument("--rtsp-path-prefix", default=DEFAULT_RTSP_STREAM_PREFIX)
    return parser


def build_stream_url(args: argparse.Namespace, stream_index: int) -> str:
    if args.rtsp_prefix:
        return f"{args.rtsp_prefix}{stream_index}"
    return build_reader_url(
        build_stream_path(stream_index, prefix=args.rtsp_path_prefix),
        host=args.rtsp_host,
        port=args.rtsp_port,
        username=args.rtsp_user,
        password=args.rtsp_password,
    )


async def _on_result_workbench(
    sid: str,
    ts: float,
    ids: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    kps: Optional[torch.Tensor],
) -> None:
    """Callback for WorkbenchPool results — just print for demo."""
    n = ids.numel()
    if n == 0:
        return
    # Print top-3 tracks per stream
    top = min(3, n)
    print(
        f"  [{sid}@{ts:.1f}] {n} tracks | "
        f"IDs={ids[:top].tolist()} "
        f"scores={boxes[:top, 0].tolist()[:top]}"
    )


async def _on_result_legacy(
    sid: str,
    ts: float,
    ids: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    kps: Optional[torch.Tensor],
) -> None:
    """Callback for AsyncDispatcher results."""
    n = ids.numel()
    if n == 0:
        return
    top = min(3, n)
    print(f"  [{sid}@{ts:.1f}] {n} tracks | IDs={ids[:top].tolist()}")


async def run_multistream_perception(args: argparse.Namespace) -> None:
    n = args.streams
    print("🚀 Initializing multi-stream Perception Pipeline...")

    if args.workbench:
        # WorkbenchPool: per-thread Workbenches sharing one BatchingTRTDetector
        print(f"  Using WorkbenchPool ({n} threads, GIL-free hot path)")
        pool = WorkbenchPool(  # noqa: F821
            engine_path=args.engine,
            n_streams=n,
            on_result=_on_result_workbench,
            input_hw=(args.input_size, args.input_size),
            frame_size=(args.frame_width, args.frame_height),
        )
        await pool.start()

        tasks = []
        for i in range(n):
            sid = f"stream_{i}"
            url = build_stream_url(args, i)
            tasks.append(
                asyncio.create_task(run_stream_producer_workbench(sid, pool, url))
            )

        try:
            while True:
                await asyncio.sleep(args.stats_interval_sec)
                stats = pool.get_stats()
                print(
                    "📈 WorkbenchPool Stats: "
                    f"level={stats['current_level']} "
                    f"active={stats['active_streams']} "
                    f"queue={stats['queue_depth']} "
                    f"drops={stats['drops']} "
                    f"e2e_p95={stats['e2e_ms_p95']:.1f}ms "
                    f"e2e_p99={stats['e2e_ms_p99']:.1f}ms "
                    f"queue_wait_p95={stats['queue_wait_ms_p95']:.1f}ms"
                )
        except asyncio.CancelledError:
            print("🛑 Shutting down WorkbenchPool...")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.stop()
    else:
        # AsyncDispatcher: legacy single-async-loop batching
        print(f"  Using AsyncDispatcher ({n} streams, async batching)")
        detector = TRTYoloDetector(engine_path=args.engine)
        resource_manager = ResourceManager()
        dispatcher = AsyncDispatcher(
            detector,
            resource_manager,
            max_batch=args.max_batch,
            input_hw=(args.input_size, args.input_size),
            batch_timeout_ms=args.batch_timeout_ms,
            max_streams=max(n, args.max_batch),
            on_track_result=_on_result_legacy,
        )
        await dispatcher.start()

        tasks = []
        for i in range(n):
            sid = f"stream_{i}"
            url = build_stream_url(args, i)
            tasks.append(asyncio.create_task(run_stream_producer(sid, dispatcher, url)))

        try:
            while True:
                await asyncio.sleep(args.stats_interval_sec)
                stats = dispatcher.get_stats()
                print(
                    "📈 Dispatcher Stats: "
                    f"level={stats['current_level']} "
                    f"active={stats['active_streams']} "
                    f"queue={stats['queue_depth']} "
                    f"batch_mean={stats['mean_batch_size']:.2f} "
                    f"batch_p95={stats['batch_size_p95']:.2f} "
                    f"queue_wait_p95={stats['queue_wait_ms_p95']:.2f}ms "
                    f"infer_p95={stats['infer_ms_p95']:.2f}ms "
                    f"e2e_p95={stats['end_to_end_ms_p95']:.2f}ms "
                    f"e2e_p99={stats['end_to_end_ms_p99']:.2f}ms"
                )
        except asyncio.CancelledError:
            print("🛑 Shutting down AsyncDispatcher...")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await dispatcher.stop()
            resource_manager.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_multistream_perception(build_parser().parse_args()))
    except KeyboardInterrupt:
        pass
