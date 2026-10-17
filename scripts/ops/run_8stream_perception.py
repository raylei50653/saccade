"""Run multi-stream (8-stream) perception demo."""

# status: stable
import argparse
import asyncio
import inspect
import time
from typing import Any, Callable, Optional, Sequence

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
from saccade.perception.dispatcher import (
    AsyncDispatcher,
    DispatcherFailed,
    DispatcherState,
    PoolState,
    WorkbenchPool,
    WorkbenchPoolFailed,
)
from saccade.perception.online_telemetry import OnlineTelemetry
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
        print(f"❌ Stream [{stream_id}] error: {e!r}")
        raise
    finally:
        media.release()


async def run_stream_producer_workbench(
    stream_id: str,
    pool: WorkbenchPool,
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
        print(f"❌ Stream [{stream_id}] error: {e!r}")
        raise
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
        "--online-telemetry",
        action="store_true",
        help="Print rolling CPU/GPU utilization and stage latency telemetry.",
    )
    parser.add_argument(
        "--workbench",
        action="store_true",
        default=False,
        help=(
            "Use per-thread WorkbenchPool (GIL-free, cross-thread parallel). "
            "Unsupported: not verified end to end (#538)."
        ),
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


def _on_result_workbench(
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


def _same_failure(a: BaseException, b: BaseException) -> bool:
    """Both report the same underlying worker failure (same ``__cause__``)."""
    return a is b or (a.__cause__ is not None and a.__cause__ is b.__cause__)


async def _supervise(
    tasks: Sequence["asyncio.Task[Any]"], watchers: Sequence["asyncio.Task[Any]"]
) -> None:
    """Wait until every task in ``tasks`` ends; the first failure in ``tasks`` or
    ``watchers`` cancels the rest and is re-raised (later ones become notes)."""
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
            if other is not None and not _same_failure(other, first):
                first.add_note(f"also failed: {other!r}")
        raise first


async def _raise_when_failed(
    target: AsyncDispatcher | WorkbenchPool, poll_s: float = 0.1
) -> None:
    """Surface a worker failure even when no producer is sending frames."""
    if isinstance(target, WorkbenchPool):
        while target.state is not PoolState.FAILED:
            await asyncio.sleep(poll_s)
        raise WorkbenchPoolFailed(
            f"WorkbenchPool worker failed: {target.failure!r}"
        ) from target.failure
    while target.state is not DispatcherState.FAILED:
        await asyncio.sleep(poll_s)
    raise DispatcherFailed(
        f"AsyncDispatcher worker failed: {target.failure!r}"
    ) from target.failure


async def _cleanup(
    primary: Optional[BaseException], steps: Sequence[tuple[str, Callable[[], Any]]]
) -> None:
    """Run every cleanup step even if an earlier one raises.

    With a primary error, cleanup errors become its notes; otherwise the first
    cleanup error is raised after all steps ran.
    """
    errors: list[Exception] = []
    for name, step in steps:
        try:
            result = step()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            errors.append(e)
            if primary is None:
                continue
            if not _same_failure(e, primary):
                primary.add_note(f"cleanup {name} also failed: {e!r}")
            else:  # the same failure re-raised; keep what its cleanup reported
                for note in getattr(e, "__notes__", []):
                    primary.add_note(f"cleanup {name}: {note}")
    if primary is None and errors:
        raise errors[0]


async def _stats_loop(interval_s: float, report: Callable[[], None]) -> None:
    while True:
        await asyncio.sleep(interval_s)
        report()


async def run_multistream_perception(args: argparse.Namespace) -> None:
    """Run until a producer or worker fails, or the task is cancelled.

    Every resource created is released on exit, in order, even if an earlier
    release raises. The first error is re-raised; release errors become its
    notes and never replace it (#538 caller lifecycle).
    """
    n = args.streams
    print("🚀 Initializing multi-stream Perception Pipeline...")
    telemetry = (
        OnlineTelemetry(sample_interval_sec=1.0) if args.online_telemetry else None
    )
    cleanup: list[tuple[str, Callable[[], Any]]] = []
    primary: Optional[BaseException] = None
    try:
        if telemetry is not None:
            telemetry.start()
            cleanup.append(("telemetry.stop()", telemetry.stop))
        if args.workbench:
            await _run_workbench(args, n, telemetry, cleanup)
        else:
            await _run_dispatcher(args, n, telemetry, cleanup)
    except BaseException as e:
        primary = e
        if isinstance(e, asyncio.CancelledError):
            print("🛑 Shutting down...")
        raise
    finally:
        # Most recently created first: workers before what they depend on.
        await _cleanup(primary, cleanup[::-1])


async def _run_workbench(
    args: argparse.Namespace,
    n: int,
    telemetry: Optional[OnlineTelemetry],
    cleanup: list[tuple[str, Callable[[], Any]]],
) -> None:
    # WorkbenchPool: per-thread Workbenches sharing one BatchingTRTDetector
    print(f"  Using WorkbenchPool ({n} threads, GIL-free hot path; unsupported)")
    pool = WorkbenchPool(
        engine_path=args.engine,
        n_streams=n,
        on_result=_on_result_workbench,
        input_hw=(args.input_size, args.input_size),
        frame_size=(args.frame_width, args.frame_height),
        telemetry=telemetry,
    )
    # pool.stop() keeps a batcher that a live worker may still use.
    cleanup.append(("pool.stop()", pool.stop))

    def report() -> None:
        stats = pool.get_stats()
        print(
            "📈 WorkbenchPool Stats: "
            f"state={stats['state']} "
            f"level={stats['current_level']} "
            f"active={stats['active_streams']} "
            f"queue={stats['queue_depth']} "
            f"drops={stats['drops']} "
            f"e2e_p95={stats['e2e_ms_p95']:.1f}ms "
            f"e2e_p99={stats['e2e_ms_p99']:.1f}ms "
            f"queue_wait_p95={stats['queue_wait_ms_p95']:.1f}ms"
        )
        if telemetry is not None:
            print(
                "🔬 Online Telemetry: "
                f"cpu_mean={stats.get('cpu_util_mean', 0.0):.1f}% "
                f"gpu_mean={stats.get('gpu_util_mean', 0.0):.1f}% "
                f"gpu_p95={stats.get('gpu_util_p95', 0.0):.1f}% "
                f"gpu_mem={stats.get('gpu_mem_used_mb_mean', 0.0):.0f}MB "
                f"pre_p95={stats.get('workbench_preprocess_ms_p95', 0.0):.1f}ms "
                f"wb_p95={stats.get('workbench_process_ms_p95', 0.0):.1f}ms "
                f"worker_p95={stats.get('workbench_worker_ms_p95', 0.0):.1f}ms"
            )

    await pool.start()
    tasks = [
        asyncio.create_task(
            run_stream_producer_workbench(
                f"stream_{i}", pool, build_stream_url(args, i)
            )
        )
        for i in range(n)
    ]
    await _supervise(
        tasks,
        [
            asyncio.create_task(_stats_loop(args.stats_interval_sec, report)),
            asyncio.create_task(_raise_when_failed(pool)),
        ],
    )


async def _run_dispatcher(
    args: argparse.Namespace,
    n: int,
    telemetry: Optional[OnlineTelemetry],
    cleanup: list[tuple[str, Callable[[], Any]]],
) -> None:
    # AsyncDispatcher: legacy single-async-loop batching
    print(f"  Using AsyncDispatcher ({n} streams, async batching)")
    detector = TRTYoloDetector(engine_path=args.engine)
    resource_manager = ResourceManager()
    cleanup.append(("resource_manager.close()", resource_manager.close))
    dispatcher = AsyncDispatcher(
        detector,
        resource_manager,
        max_batch=args.max_batch,
        input_hw=(args.input_size, args.input_size),
        batch_timeout_ms=args.batch_timeout_ms,
        max_streams=max(n, args.max_batch),
        on_track_result=_on_result_legacy,
        telemetry=telemetry,
    )
    cleanup.append(("dispatcher.stop()", dispatcher.stop))

    def report() -> None:
        stats = dispatcher.get_stats()
        print(
            "📈 Dispatcher Stats: "
            f"state={stats['state']} "
            f"level={stats['current_level']} "
            f"active={stats['active_streams']} "
            f"queue={stats['queue_depth']} "
            f"batch_mean={stats['mean_batch_size']:.2f} "
            f"batch_p95={stats['batch_size_p95']:.2f} "
            f"batch_build_p95={stats['batch_build_ms_p95']:.2f}ms "
            f"queue_wait_p95={stats['queue_wait_ms_p95']:.2f}ms "
            f"infer_p95={stats['infer_ms_p95']:.2f}ms "
            f"e2e_p95={stats['end_to_end_ms_p95']:.2f}ms "
            f"e2e_p99={stats['end_to_end_ms_p99']:.2f}ms"
        )
        if telemetry is not None:
            print(
                "🔬 Online Telemetry: "
                f"cpu_mean={stats.get('cpu_util_mean', 0.0):.1f}% "
                f"gpu_mean={stats.get('gpu_util_mean', 0.0):.1f}% "
                f"gpu_p95={stats.get('gpu_util_p95', 0.0):.1f}% "
                f"gpu_mem={stats.get('gpu_mem_used_mb_mean', 0.0):.0f}MB "
                f"build_p95={stats.get('dispatcher_batch_build_ms_p95', 0.0):.1f}ms "
                f"infer_p95={stats.get('dispatcher_infer_ms_p95', 0.0):.1f}ms "
                f"track_p95={stats.get('dispatcher_track_ms_p95', 0.0):.1f}ms"
            )

    await dispatcher.start()
    tasks = [
        asyncio.create_task(
            run_stream_producer(f"stream_{i}", dispatcher, build_stream_url(args, i))
        )
        for i in range(n)
    ]
    await _supervise(
        tasks,
        [
            asyncio.create_task(_stats_loop(args.stats_interval_sec, report)),
            asyncio.create_task(_raise_when_failed(dispatcher)),
        ],
    )


if __name__ == "__main__":
    try:
        asyncio.run(run_multistream_perception(build_parser().parse_args()))
    except KeyboardInterrupt:
        pass
