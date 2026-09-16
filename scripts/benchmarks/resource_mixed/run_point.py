"""Run scheduled real pipeline frames against bounded native elastic bursts."""

# status: diagnostic
import argparse
import ctypes as c
import hashlib
import json
import math
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
        "--policy",
        choices=["control", "fixed", "headroom", "shared", "dynamic"],
        required=True,
    )
    p.add_argument("--window", type=int, choices=[1, 4], default=1)
    p.add_argument("--iterations", type=int, choices=[2048, 8192], default=2048)
    p.add_argument("--frames", type=int, default=350)
    p.add_argument("--period-ms", type=float, default=1000 / 60)
    p.add_argument("--deadline-ms", type=float, default=1000 / 60)
    p.add_argument("--units", type=int, default=256)
    p.add_argument("--libraries", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.frames <= 50 or args.period_ms <= 0 or args.deadline_ms <= 0:
        p.error("positive period/deadline and >50 frames required")
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
    size = 32 if args.policy == "shared" else 24 if args.policy == "headroom" else 16
    owner = go.GreenExecutionOwner(
        size,
        args.libraries / "partition_audit.so" if args.audit else None,
        args.libraries / "smid_probe.so",
        args.output / "cupti.trace",
    )
    # Stable stream priority is high in every policy, including matched sharing.
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
    # Probe the independent elastic pool, restoring all stable owner state.
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
    bursts = math.ceil((args.frames - 50) * args.period_ms / 100 - 1e-9)
    native = lib.mixed_create(
        *[int(s) for s in elastic_streams],
        *[int(ctx) for ctx in contexts],
        args.window,
        args.iterations,
        bursts,
        args.units,
        100_000_000,
        args.policy == "dynamic",
    )
    if lib.mixed_error(native):
        raise RuntimeError(lib.mixed_error(native).decode())
    call("cuCtxSetCurrent", owner.context)
    owner.mark("pipeline_begin")
    # Preserve the evaluator's libjpeg-sensitive import order.
    import saccade.perception.detector_trt  # noqa: F401
    from saccade.perception.eval import evaluator, stages

    original_run = evaluator._run_frame
    original_record = stages._record_frame_timing
    rows = []
    completion = {}
    origin = None
    native_origin = None
    native_anchor_after = None
    worker = None
    worker_result = []

    def record(state, *, frame_id, latency_started_at):
        original_record(state, frame_id=frame_id, latency_started_at=latency_started_at)
        if frame_id > state.warmup_frames:
            completion[frame_id] = state.throughput_finished_at

    def run_frame(state, *, frame_id, prepared_detection=None):
        nonlocal origin, native_origin, native_anchor_after, worker
        if prepared_detection is not None or state.double_buffer_stream is not None:
            raise RuntimeError("this contract requires serial frame admission")
        if frame_id <= state.warmup_frames:
            return original_run(state, frame_id=frame_id)
        if origin is None:
            # perf_counter and native steady_clock epochs are independently
            # paired here; all frame durations use perf_counter exclusively.
            for stream in owner.streams:
                call("cuStreamSynchronize", d.CUstream(stream["handle"]))
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
                if lib.mixed_busy(native, 0):
                    raise RuntimeError("initial release failed")
            call("cuCtxSetCurrent", owner.context)
            owner.mark("mixed_begin")
        arrival = origin + (frame_id - state.warmup_frames) * args.period_ms / 1000
        delay = arrival - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        request = time.perf_counter()
        if lib.mixed_busy(native, 1):
            raise RuntimeError("borrow drain failed")
        drained = time.perf_counter()
        call("cuCtxSetCurrent", owner.context)
        admitted = time.perf_counter()
        result = original_run(state, frame_id=frame_id)
        if frame_id not in completion:
            raise RuntimeError("frame completion not observed")
        # Finish all work in this context before lending it. Include this
        # conservative barrier in stable response for every policy.
        for stream in owner.streams:
            call("cuStreamSynchronize", d.CUstream(stream["handle"]))
        finished = time.perf_counter()
        rows.append(
            dict(
                frame=frame_id,
                arrival=arrival,
                request=request,
                drained=drained,
                admitted=admitted,
                output=completion[frame_id],
                finished=finished,
                context_owned=owner.current_context_owned(),
            )
        )
        if lib.mixed_busy(native, 0):
            raise RuntimeError("borrow release failed")
        call("cuCtxSetCurrent", owner.context)
        return result

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
        result = dict(
            schema="saccade-mixed-point-v1",
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            origin=origin,
            native_origin=native_origin,
            native_anchor_after=native_anchor_after,
            frames=rows,
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
            bursts=bursts,
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
