"""Check whether primary-context synchronize waits for an independent blocker."""

# status: diagnostic
import argparse
import ctypes
import json
import threading
import time
from pathlib import Path


def probe(library, blocks=16):
    import torch

    torch.cuda.init()
    lib = ctypes.CDLL(str(library))
    lib.blocker_error.restype = ctypes.c_char_p
    lib.blocker_pulse.restype = ctypes.c_char_p
    if lib.blocker_init(blocks):
        raise RuntimeError(lib.blocker_error().decode())
    work = torch.cuda.Stream()
    x = torch.ones(32, device="cuda")
    done = torch.cuda.Event()
    with torch.cuda.stream(work):
        x.add_(1)
        done.record()
    torch.cuda.synchronize()
    pulse = []

    def launch():
        raw = lib.blocker_pulse(100000)
        if raw:
            pulse.extend(json.loads(raw))

    thread = threading.Thread(target=launch)
    try:
        thread.start()
        time.sleep(0.01)
        start = time.perf_counter()
        with torch.cuda.stream(work):
            x.add_(1)
            done.record()
        done.synchronize()
        own_done = time.perf_counter()
        torch.cuda.synchronize()
        all_done = time.perf_counter()
        thread.join(timeout=5)
        if thread.is_alive() or len(pulse) != blocks:
            raise RuntimeError("probe pulse failed or did not terminate")
        return {
            "blocks": blocks,
            "pulse_us": 100000,
            "own_stream_work_and_event_wait_ms": (own_done - start) * 1000,
            "subsequent_device_synchronize_ms": (all_done - own_done) * 1000,
            "pulse_blocks": pulse,
            "kind": "synchronization_dependency_control_not_full_pipeline",
        }
    finally:
        thread.join(timeout=5)
        if not thread.is_alive():
            lib.blocker_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.library), indent=2) + "\n")
