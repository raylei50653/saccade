"""Run one full-pipeline MOT17 point with bounded SM residency pressure."""

# status: diagnostic
import argparse
import ctypes
import json
import runpy
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class Blocker:
    def __init__(self, library, blocks, pulse_us):
        self.lib = ctypes.CDLL(str(library))
        for name in ("blocker_error", "blocker_info", "blocker_pulse"):
            getattr(self.lib, name).restype = ctypes.c_char_p
        self.lib.blocker_init.argtypes = [ctypes.c_int]
        self.lib.blocker_pulse.argtypes = [ctypes.c_int]
        if self.lib.blocker_init(blocks):
            raise RuntimeError(self.lib.blocker_error().decode())
        self.info = json.loads(self.lib.blocker_info())
        self.pulse_us = pulse_us
        self.rows = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                begin = time.perf_counter()
                raw = self.lib.blocker_pulse(self.pulse_us)
                end = time.perf_counter()
                if not raw:
                    raise RuntimeError(self.lib.blocker_error().decode())
                self.rows.append(
                    {"host_begin": begin, "host_end": end, "blocks": json.loads(raw)}
                )
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("blocker did not stop within 10 seconds")
        if self.lib.blocker_close():
            raise RuntimeError(self.lib.blocker_error().decode())
        if self.error:
            raise RuntimeError(self.error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--pulse-us", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.blocks < 0 or not 100 <= args.pulse_us <= 100000:
        parser.error("blocks must be nonnegative; pulse-us must be 100..100000")
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (ROOT, ROOT / "src", ROOT / "build", ROOT / "scripts/eval"):
        sys.path.insert(0, str(path))
    # Match mot17.py's libjpeg-sensitive import order.
    import saccade.perception.detector_trt  # noqa: F401
    from saccade.perception.eval import evaluator, stages

    original = stages._record_frame_timing
    blocker = None
    completions = {}
    db_enabled = {}

    def record(state, *, frame_id, latency_started_at):
        nonlocal blocker
        original(state, frame_id=frame_id, latency_started_at=latency_started_at)
        # First completed frame is after the detector/tracker graph setup.
        # Start in warmup, so native allocations are outside measured frames.
        if args.blocks and blocker is None:
            if state.warmup_frames < 10:
                raise ValueError("resource sweep requires at least 10 warmup frames")
            blocker = Blocker(args.library, args.blocks, args.pulse_us)
        if frame_id > state.warmup_frames:
            completions.setdefault(state.seq, []).append(
                [frame_id, state.throughput_finished_at]
            )
            db_enabled[state.seq] = state.double_buffer_stream is not None

    stages._record_frame_timing = record
    evaluator._record_frame_timing = record
    eval_args = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    sys.argv = [
        str(ROOT / "scripts/eval/mot17.py"),
        *eval_args,
        "--output",
        str(args.output),
    ]
    clock_anchor = {"epoch": time.time(), "monotonic": time.perf_counter()}
    error = None
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            if blocker:
                blocker.close()
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            data = {
                "clock_anchor": clock_anchor,
                "schema": "saccade-resource-point-v1",
                "kind": "bounded_residency_pressure_proxy",
                "blocks_requested": args.blocks,
                "pulse_us": args.pulse_us,
                "blocker_info": blocker.info if blocker else None,
                "pulses": blocker.rows if blocker else [],
                "completions": completions,
                "double_buffer_enabled": db_enabled,
                "error": error,
            }
            (args.output / "resource_point.json").write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
