"""Run one full-pipeline MOT17 point inside a verified Green Context partition."""

# status: diagnostic
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.resource_partition.green_owner import (  # noqa: E402
    GreenExecutionOwner,
)
from scripts.benchmarks.resource_partition.routing import (  # noqa: E402
    build_point_report,
)


def parse_sm_count(value):
    if value == "full":
        return None
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("sm-count must be positive or 'full'")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm-count", type=parse_sm_count, required=True)
    parser.add_argument("--audit-library", type=Path)
    parser.add_argument("--probe-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # The evaluator's run manifest claims an empty directory; keep it separate.
    eval_output = args.output / "eval"
    for path in (ROOT, ROOT / "src", ROOT / "build", ROOT / "scripts/eval"):
        sys.path.insert(0, str(path))
    trace_path = args.output / "cupti_activity.trace"
    owner = GreenExecutionOwner(
        args.sm_count, args.audit_library, args.probe_library, trace_path
    )
    # Install before any Saccade import so no pool stream or graph exists yet.
    owner.install()
    # Match mot17.py's libjpeg-sensitive import order.
    import saccade.perception.detector_trt  # noqa: F401
    from saccade.perception.eval import evaluator, stages

    original = stages._record_frame_timing
    completions = {}
    frame_starts = {}
    routes = {}
    unowned_frames = []

    def record(state, *, frame_id, latency_started_at):
        original(state, frame_id=frame_id, latency_started_at=latency_started_at)
        seq = state.seq
        if seq not in routes:
            owner.mark(f"first_completion {seq}")
            side = state.double_buffer_stream
            routes[seq] = {
                "double_buffer_enabled": side is not None,
                "double_buffer_stream_handle": side.cuda_stream if side else None,
                "double_buffer_stream_owned_class": isinstance(side, owner.stream_class)
                if side
                else None,
                "warmup_frames": state.warmup_frames,
            }
        frame_starts.setdefault(seq, []).append([frame_id, latency_started_at])
        if not owner.current_context_owned():
            unowned_frames.append([seq, frame_id])
        if frame_id > state.warmup_frames:
            completions.setdefault(seq, []).append(
                [frame_id, state.throughput_finished_at]
            )

    stages._record_frame_timing = record
    evaluator._record_frame_timing = record
    eval_args = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    sys.argv = [
        str(ROOT / "scripts/eval/mot17.py"),
        *eval_args,
        "--output",
        str(eval_output),
    ]
    clock_anchor = {"epoch": time.time(), "monotonic": time.perf_counter()}
    error = None
    evidence = None
    owner.mark("eval_begin")
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        owner.mark("eval_end")
        try:
            evidence = owner.finalize()
        except Exception as exc:
            error = error or repr(exc)
            raise
        finally:
            point = {
                "schema": "saccade-partition-point-v1",
                "kind": "green_context_true_partition"
                if args.sm_count is not None
                else "primary_full_device_baseline",
                "clock_anchor": clock_anchor,
                "requested_sm_count": args.sm_count,
                "audit_enabled": args.audit_library is not None,
                "owner": evidence,
                "routes": routes,
                "frame_starts": frame_starts,
                "completions": completions,
                "frames_completed_with_unowned_main_thread_context": unowned_frames,
                "double_buffer_enabled": {
                    seq: r["double_buffer_enabled"] for seq, r in routes.items()
                },
                "error": error,
            }
            report = build_point_report(point, trace_path if evidence else None)
            (args.output / "partition_point.json").write_text(json.dumps(report) + "\n")
            print(
                json.dumps(
                    {
                        k: report.get(k)
                        for k in (
                            "requested_sm_count",
                            "actual_sm_count",
                            "true_partition_validated",
                            "failed_checks",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
