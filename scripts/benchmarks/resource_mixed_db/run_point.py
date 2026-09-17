"""Run saturated double-buffer frames against bounded native elastic bursts."""

# status: diagnostic
import argparse
import ctypes as c
import hashlib
import json
import runpy
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "build", ROOT / "scripts/eval"):
    sys.path.insert(0, str(path))
from scripts.benchmarks.resource_partition import green_owner as go  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy", choices=["control", "fixed", "shared", "dynamic"], required=True
    )
    p.add_argument("--window", type=int, choices=[1, 4], default=1)
    p.add_argument("--iterations", type=int, choices=[2048, 8192], default=2048)
    p.add_argument("--frames", type=int, default=350)
    p.add_argument("--deadline-ms", type=float, default=1000 / 60)
    p.add_argument("--bursts", type=int, default=50)
    p.add_argument("--burst-period-ms", type=float, default=100.0)
    p.add_argument("--units", type=int, default=256)
    p.add_argument("--libraries", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        args.frames <= 52
        or args.deadline_ms <= 0
        or args.bursts <= 0
        or args.burst_period_ms <= 0
    ):
        p.error("positive deadline/workload and >52 frames required")
    args.output.mkdir(parents=True, exist_ok=False)
    from cuda.bindings import driver as d

    call = go.call
    partition_groups = []

    def capture_split(name, *params):
        if name == "cuDevSmResourceSplitByCount":
            groups, count, remainder = call(name, 4, params[1], 0, 8)
            if count != 4:
                raise RuntimeError("four 8-SM groups required")
            partition_groups.extend(groups)
            return [groups[0]], 1, remainder
        if name == "cuDevResourceGenerateDesc":
            selected = partition_groups[: size // 8]
            return call(name, selected, len(selected))
        return call(name, *params)

    go.call = capture_split
    size = 32 if args.policy == "shared" else 16
    owner = go.GreenExecutionOwner(
        size,
        args.libraries / "partition_audit.so" if args.audit else None,
        args.libraries / "smid_probe.so",
        args.output / "cupti.trace",
    )
    original_create = owner._create_stream
    owner._create_stream = lambda priority: original_create(owner._priority_range[1])
    owner.install()
    go.call = call
    import torch

    if args.policy in ("shared", "control"):
        elastic_green, elastic_context = owner.green, owner.context
        elastic_size = size
    else:
        groups = partition_groups[size // 8 :]
        desc = call("cuDevResourceGenerateDesc", groups, len(groups))
        elastic_green = call(
            "cuGreenCtxCreate",
            desc,
            0,
            d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
        )
        elastic_context = call("cuCtxFromGreenCtx", elastic_green)
        elastic_size = call(
            "cuGreenCtxGetDevResource",
            elastic_green,
            d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
        ).sm.smCount
        if elastic_size != 32 - size:
            raise RuntimeError("elastic actual SM mismatch")
    elastic_streams = []
    contexts = [
        elastic_context,
        owner.context if args.policy == "dynamic" else elastic_context,
    ]
    for green in [
        elastic_green,
        owner.green if args.policy == "dynamic" else elastic_green,
    ]:
        elastic_streams.append(
            call(
                "cuGreenCtxStreamCreate",
                green,
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,
            )
        )
    original_main = owner.main_stream
    owner.main_stream = torch.cuda.ExternalStream(int(elastic_streams[0]))
    call("cuCtxSetCurrent", elastic_context)
    with torch.cuda.stream(owner.main_stream):
        elastic_probe = owner.probe_partition()
    owner.main_stream = original_main
    call("cuCtxSetCurrent", owner.context)
    torch.cuda.set_stream(original_main)

    lib = c.CDLL(str(args.libraries / "elastic.so"))
    lib.mixed_create.argtypes = (
        [c.c_void_p] * 4 + [c.c_int] * 4 + [c.c_ulonglong, c.c_int]
    )
    lib.mixed_create.restype = c.c_void_p
    lib.mixed_error.argtypes = [c.c_void_p]
    lib.mixed_error.restype = c.c_char_p
    lib.mixed_busy.argtypes = [c.c_void_p, c.c_int]
    lib.mixed_run.argtypes = [c.c_void_p, c.c_ulonglong]
    lib.mixed_now.restype = c.c_ulonglong
    lib.mixed_save.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p]
    lib.mixed_reference.argtypes = [c.c_int]
    lib.mixed_reference.restype = c.c_float
    native = lib.mixed_create(
        *[int(s) for s in elastic_streams],
        *[int(ctx) for ctx in contexts],
        args.window,
        args.iterations,
        args.bursts,
        args.units,
        int(args.burst_period_ms * 1_000_000),
        args.policy == "dynamic",
    )
    if lib.mixed_error(native):
        raise RuntimeError(lib.mixed_error(native).decode())

    call("cuCtxSetCurrent", owner.context)
    owner.mark("pipeline_begin")
    import saccade.perception.detector_trt  # noqa: F401
    from saccade.perception.eval import evaluator, stages, streaming

    original_run = evaluator._run_frame
    original_launch = evaluator._launch_double_buffer_detect
    original_record = stages._record_frame_timing
    original_next = streaming.TorchvisionGpuStreamer.__next__
    starts = {}
    completion = {}
    cycle_finished = {}
    transitions = []
    windows = []
    route = {}
    origin = None
    native_origin = None
    native_anchor_after = None
    worker = None
    worker_result = []
    pending_release = None

    def start_measurement():
        nonlocal origin, native_origin, native_anchor_after, worker
        if origin is not None:
            return
        owner.mark("mixed_begin")
        origin = time.perf_counter()
        native_origin = lib.mixed_now()
        native_anchor_after = time.perf_counter()
        if args.policy != "control":
            worker = threading.Thread(
                target=lambda: worker_result.append(
                    lib.mixed_run(native, native_origin)
                )
            )
            worker.start()
            if lib.mixed_busy(native, 1):
                raise RuntimeError("initial borrow freeze failed")
        call("cuCtxSetCurrent", owner.context)

    def next_frame(streamer):
        nonlocal pending_release
        scheduled_frame = streamer._idx + 1
        if scheduled_frame == 51:
            start_measurement()
        if 51 <= scheduled_frame <= args.frames:
            request = time.perf_counter()
            if lib.mixed_busy(native, 1):
                raise RuntimeError("borrow drain failed")
            drained = time.perf_counter()
            call("cuCtxSetCurrent", owner.context)
            admitted = time.perf_counter()
            transitions.append(
                dict(
                    scheduled_frame=scheduled_frame,
                    request=request,
                    drained=drained,
                    admitted=admitted,
                )
            )
            if pending_release is not None:
                windows.append(
                    dict(
                        **pending_release,
                        reclaim_before_scheduled_frame=scheduled_frame,
                        request=request,
                        drained=drained,
                        admitted=admitted,
                        within_stable_service=True,
                    )
                )
                pending_release = None
        return original_next(streamer)

    def launch(state, **kwargs):
        frame_id = kwargs["frame_id"]
        result = original_launch(state, **kwargs)
        if frame_id >= 51:
            starts[frame_id] = kwargs["latency_started_at"]
        if not route:
            side = state.double_buffer_stream
            route.update(
                enabled=side is not None,
                stream_handle=int(side.cuda_stream) if side is not None else None,
                owned_class=isinstance(side, owner.stream_class)
                if side is not None
                else False,
                warmup_frames=state.warmup_frames,
            )
        return result

    def record(state, *, frame_id, latency_started_at):
        original_record(state, frame_id=frame_id, latency_started_at=latency_started_at)
        if frame_id > state.warmup_frames:
            completion[frame_id] = state.throughput_finished_at

    def run_frame(state, *, frame_id, prepared_detection=None):
        nonlocal pending_release
        if state.double_buffer_stream is None or prepared_detection is None:
            raise RuntimeError("this contract requires double-buffer frame admission")
        result = original_run(
            state, frame_id=frame_id, prepared_detection=prepared_detection
        )
        if 50 <= frame_id <= args.frames:
            for stream in owner.streams:
                call("cuStreamSynchronize", d.CUstream(stream["handle"]))
            finished = time.perf_counter()
            if frame_id >= 51:
                cycle_finished[frame_id] = dict(
                    finished=finished, context_owned=owner.current_context_owned()
                )
            # There is no schedule(args.frames + 1) before the final frame, so
            # do not lend after its predecessor. The final post-GPU CPU tail is
            # a separate, post-service window.
            should_release = (
                frame_id == 50
                or 51 <= frame_id < args.frames - 1
                or frame_id == args.frames
            )
            if should_release:
                release_requested = time.perf_counter()
                if lib.mixed_busy(native, 0):
                    raise RuntimeError("borrow release failed")
                released = time.perf_counter()
                pending_release = dict(
                    release_after_frame=frame_id,
                    release_requested=release_requested,
                    released=released,
                )
            call("cuCtxSetCurrent", owner.context)
        return result

    streaming.TorchvisionGpuStreamer.__next__ = next_frame
    evaluator._launch_double_buffer_detect = launch
    evaluator._run_frame = run_frame
    evaluator._record_frame_timing = stages._record_frame_timing = record
    sys.argv = [
        str(ROOT / "scripts/eval/mot17.py"),
        "--preset",
        "mamba_whole_graph_m",
        "--detector",
        "SDP",
        "--sequences",
        "MOT17-04-SDP",
        "--warmup-frames",
        "50",
        "--max-frames",
        str(args.frames),
        "--detect-barrier",
        "event",
        "--double-buffer",
        "--output",
        str(args.output / "eval"),
    ]
    error = None
    evidence = None
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if worker:
            if pending_release is not None:
                request = time.perf_counter()
                if lib.mixed_busy(native, 1):
                    error = error or "final borrow drain failed"
                drained = time.perf_counter()
                windows.append(
                    dict(
                        **pending_release,
                        reclaim_before_scheduled_frame=None,
                        request=request,
                        drained=drained,
                        admitted=None,
                        within_stable_service=False,
                    )
                )
                pending_release = None
            lib.mixed_busy(native, 0)
            worker.join(timeout=125)
            if worker.is_alive() or worker_result != [0]:
                error = (
                    error
                    or "elastic worker failed: " + lib.mixed_error(native).decode()
                )
            if not worker.is_alive():
                if lib.mixed_save(
                    native,
                    str(args.output / "elastic.records").encode(),
                    str(args.output / "elastic.stamps").encode(),
                ):
                    error = error or "elastic save failed"
        call("cuCtxSetCurrent", owner.context)
        torch.cuda.set_stream(original_main)
        owner.mark("mixed_end")
        evidence = owner.finalize()
        frame_rows = [
            dict(
                frame=frame_id,
                start=starts[frame_id],
                output=completion[frame_id],
                cycle_finished=cycle_finished[frame_id]["finished"],
                context_owned=cycle_finished[frame_id]["context_owned"],
            )
            for frame_id in sorted(set(starts) & set(completion) & set(cycle_finished))
        ]
        result = dict(
            schema="saccade-mixed-db-point-v1",
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            origin=origin,
            native_origin=native_origin,
            native_anchor_after=native_anchor_after,
            frames=frame_rows,
            transitions=transitions,
            windows=windows,
            route=route,
            owner=evidence,
            elastic_pool=dict(
                actual_sms=elastic_size,
                context=int(elastic_context),
                cupti_context=owner._context_id(elastic_context)
                if args.audit
                else None,
                probe=elastic_probe,
                streams=[int(s) for s in elastic_streams],
                contexts=[int(ctx) for ctx in contexts],
            ),
            reference=lib.mixed_reference(args.iterations),
            error=error,
        )
        result["output_sha256"] = {
            str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (args.output / "eval").glob("*.txt")
            if not f.name.startswith("_")
        }
        (args.output / "point.json").write_text(json.dumps(result, indent=2) + "\n")
        if error:
            raise RuntimeError(error)


if __name__ == "__main__":
    main()
