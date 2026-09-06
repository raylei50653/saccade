"""Run one Python entry point with diagnostic-only CUPTI/Python attribution.

This changes observation overhead, never graph arguments or capture modes.
It is not a failure-rate harness and has no repetition option.
"""

# status: diagnostic

import argparse
import ctypes
import functools
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import traceback


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command_output(*argv: str) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else result.stderr


def asset_paths(files, trees):
    paths = {p.resolve() for p in files}
    for tree in trees:
        if not tree.is_dir():
            raise ValueError(f"Asset tree does not exist: {tree}")
        paths.update(p.resolve() for p in tree.rglob("*") if p.is_file())
    return sorted(paths)


def label_for(stack: list[dict], frames: list) -> str:
    # repo wrapper labels take priority; read only the label, never other locals.
    for frame in frames:
        if frame.f_code.co_name == "graph_capture":
            label = frame.f_locals.get("label")
            if isinstance(label, str):
                return label
    sites = {
        "_whole_graph_capture": "detector.whole",
        "_whole_graph_capture_preprocessed": "detector.whole_preprocessed",
    }
    for entry in stack:
        if entry["function"] in sites:
            return sites[entry["function"]]
        if entry["file"].endswith("/tracking/tracker_gpu.py"):
            return "tracker.update"
    return "unclassified.python"


def install(torch, native, emit):
    graph_class = torch.cuda.CUDAGraph
    original_begin = graph_class.capture_begin
    original_end = graph_class.capture_end
    signature = inspect.signature(original_begin)
    ids = itertools.count(1)
    active = {}

    @functools.wraps(original_begin)
    def begin(graph, *args, **kwargs):
        bound = signature.bind(graph, *args, **kwargs)
        bound.apply_defaults()
        frames = []
        frame = sys._getframe(1)
        while frame is not None:
            frames.append(frame)
            frame = frame.f_back
        stack = [
            {
                "file": f.f_code.co_filename,
                "line": f.f_lineno,
                "function": f.f_code.co_name,
            }
            for f in frames
        ]
        label = label_for(stack, frames)
        del frames, frame
        record = {
            "site_id": next(ids),
            "label": label,
            "stack": stack,
            "mode": bound.arguments.get("capture_error_mode", "unknown"),
            "stream": torch.cuda.current_stream().cuda_stream,
            "device": torch.cuda.current_device(),
        }
        active[id(graph)] = record
        native.attribution_site(record["site_id"])
        emit("python_begin_enter", **record)
        try:
            result = original_begin(graph, *args, **kwargs)
        except BaseException as exc:
            emit("python_begin_error", **record, error=repr(exc))
            active.pop(id(graph), None)
            native.attribution_site(0)
            raise
        emit("python_begin_exit", **record)
        return result

    @functools.wraps(original_end)
    def end(graph, *args, **kwargs):
        record = active.get(id(graph), {"site_id": 0, "label": "unmatched.end"})
        native.attribution_site(record["site_id"])
        emit("python_end_enter", **record)
        try:
            result = original_end(graph, *args, **kwargs)
        except BaseException as exc:
            emit("python_end_error", **record, error=repr(exc))
            raise
        else:
            emit("python_end_exit", **record)
            return result
        finally:
            active.pop(id(graph), None)
            native.attribution_site(0)

    graph_class.capture_begin = begin
    graph_class.capture_end = end

    def restore():
        graph_class.capture_begin = original_begin
        graph_class.capture_end = original_end

    return restore


# Both helpers read ``sys.modules`` instead of importing: teardown runs while the
# observer is still subscribed, and importing a module the target never loaded
# would add library loads and API calls to the trace that the workload never made.
def _shutdown_compile_workers() -> None:
    module = sys.modules.get("torch._inductor.async_compile")
    if module is not None:
        module.shutdown_compile_workers()


def _shutdown_tqdm_monitor() -> None:
    module = sys.modules.get("tqdm")
    monitor = getattr(getattr(module, "tqdm", None), "monitor", None)
    if monitor is not None:
        monitor.exit()


def quiesce(timeout: float) -> dict:
    """Bring library-owned daemon threads down before the observer stops.

    torch's inductor compile-worker pool and tqdm's monitor outlive the target's
    return, so a production target reaches ``attribution_stop`` with threads
    still alive. They are shut down here rather than excused: the shutdown check
    stays "no live thread", and this is what lets a production trace satisfy it
    honestly. Runs before ``attribution_stop`` so that anything these threads do
    on the way down is still observed. Its own failures are recorded, never
    raised -- a teardown convenience must not destroy the trace.
    """
    start = time.monotonic()
    errors = {}
    for name, action in (
        ("inductor_compile_workers", _shutdown_compile_workers),
        ("tqdm_monitor", _shutdown_tqdm_monitor),
    ):
        try:
            action()
        except BaseException as exc:
            errors[name] = repr(exc)
    deadline = start + timeout
    while True:
        alive = [
            t for t in threading.enumerate() if t is not threading.current_thread()
        ]
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            break
        for thread in alive:
            thread.join(min(0.2, max(0.0, deadline - time.monotonic())))
    return {
        "seconds": round(time.monotonic() - start, 2),
        "errors": errors,
        "timed_out": bool(alive),
    }


def teardown_log(output: Path):
    """Open ``tail.log`` and return it with a step recorder.

    Teardown is the one stretch where a failure erases its own evidence: the
    manifest is only rewritten at the very end, so a hang before that leaves a
    directory that says nothing beyond "no final manifest". Each step is flushed
    as it happens so the last line names where the tail stopped. Exclusive
    creation, because a second writer would rewrite that record.
    """
    tail = (output / "tail.log").open("x")

    def note(step: str) -> None:
        tail.write(f"{time.monotonic_ns()} {step}\n")
        tail.flush()

    return tail, note


def run(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = Path(args.target[0]).resolve()
    native_path = args.observer.resolve()
    inputs = asset_paths([target, *args.asset], args.asset_tree)
    repo = Path(command_output("git", "rev-parse", "--show-toplevel").strip())
    tracked = command_output("git", "ls-files", "-z").split("\0")
    source_hashes = {
        name: digest(repo / name)
        for name in tracked
        if name
        and (repo / name).is_file()
        and Path(name).suffix
        in {".py", ".cpp", ".cu", ".h", ".hpp", ".yaml", ".toml", ".lock"}
    }
    manifest = {
        "purpose": "single attribution run; not incidence or throughput evidence",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": [str(target), *args.target[1:]],
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "head": command_output("git", "rev-parse", "HEAD").strip(),
        "status": command_output("git", "status", "--porcelain"),
        "source_sha256_before": source_hashes,
        "packages": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SACCADE_", "CUDA_", "TORCH_", "PYTORCH_"))
            or k in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"}
        },
        "gpu": command_output(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ),
        "observer_sha256": digest(native_path),
        "explicit_input_sha256": {str(p): digest(p) for p in inputs},
        "asset_trees": [str(p.resolve()) for p in args.asset_tree],
        "harness_sha256": {
            str(p.resolve()): digest(p)
            for p in Path(__file__).parent.iterdir()
            if p.suffix in {".py", ".cpp"}
        },
        "asset_coverage": "explicit --asset files and complete --asset-tree inventories; not automatic workload attestation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source.patch").write_text(command_output("git", "diff", "HEAD", "--"))
    native = ctypes.CDLL(str(native_path))
    native.attribution_start.argtypes = [ctypes.c_char_p]
    native.attribution_start.restype = ctypes.c_int
    native.attribution_site.argtypes = [ctypes.c_uint64]
    native.attribution_site.restype = None
    native.attribution_stop.argtypes = []
    native.attribution_stop.restype = ctypes.c_int
    rc = native.attribution_start(os.fsencode(output / "cuda.jsonl"))
    if rc:
        raise RuntimeError(f"CUPTI subscription failed ({rc}); target was not started")
    python_log = (output / "python.jsonl").open("x")
    lock = threading.Lock()

    def emit(event, **fields):
        row = {
            "event": event,
            "ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            **fields,
        }
        with lock:
            python_log.write(json.dumps(row) + "\n")
            python_log.flush()

    def restore():
        pass

    old_hook = threading.excepthook
    old_argv = sys.argv
    old_path = sys.path[:]
    saved_fds = [os.dup(fd) for fd in (1, 2)]
    for fd, name in ((1, "stdout.log"), (2, "stderr.log")):
        log_fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(log_fd, fd)
        os.close(log_fd)

    def thread_error(details):
        emit(
            "thread_exception",
            exception_type=details.exc_type.__name__,
            traceback="".join(
                traceback.format_exception(
                    details.exc_type, details.exc_value, details.exc_traceback
                )
            ),
        )
        old_hook(details)

    outcome = "unreported"
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "graphs_file": torch.cuda.graphs.__file__,
            "graphs_sha256": digest(Path(torch.cuda.graphs.__file__)),
        }
        restore = install(torch, native, emit)
        threading.excepthook = thread_error
        emit("harness_ready")
        sys.argv = [str(target), *args.target[1:]]
        sys.path.insert(0, str(target.parent))
        namespace = runpy.run_path(str(target), run_name="__main__")
        resolved = namespace.get("args")
        if isinstance(resolved, argparse.Namespace):
            manifest["resolved_cli_args"] = vars(resolved)
        manifest["runtime_env_after"] = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SACCADE_", "CUDA_", "TORCH_", "PYTORCH_"))
        }
        emit("target_returned")
        outcome = "target_returned"
    except BaseException as exc:
        emit(
            "target_exception",
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        outcome = f"target_exception:{type(exc).__name__}"
        raise
    finally:
        tail, note = teardown_log(output)
        note(f"workload_completed:{outcome}")
        # Passive process mappings; no CUDA status queries at failure time.
        maps = Path("/proc/self/maps").read_text()
        (output / "maps.txt").write_text(maps)
        note("maps_written")
        restore()
        threading.excepthook = old_hook
        sys.argv, sys.path = old_argv, old_path
        note(f"quiesce_begin:timeout={args.quiesce_timeout}")
        quiesced = quiesce(args.quiesce_timeout)
        note(f"quiesce_end:{json.dumps(quiesced)}")
        rc = native.attribution_stop()
        note(f"attribution_stopped:{rc}")
        emit(
            "harness_stopped",
            cupti_rc=rc,
            quiesce=quiesced,
            live_threads=[
                {"name": t.name, "native_id": t.native_id}
                for t in threading.enumerate()
                if t is not threading.current_thread()
            ],
        )
        note("harness_stopped_emitted")
        python_log.close()
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, saved in zip((1, 2), saved_fds):
            os.dup2(saved, fd)
            os.close(saved)
        libraries = {
            line.split(maxsplit=5)[5]
            for line in maps.splitlines()
            if len(line.split(maxsplit=5)) == 6
            and line.split(maxsplit=5)[5].startswith("/")
        }
        note(f"hashing_mapped_files:{len(libraries)}")
        manifest["mapped_file_sha256_after"] = {
            name: digest(Path(name))
            for name in sorted(libraries)
            if Path(name).is_file()
        }
        note("mapped_files_hashed")
        manifest["source_drift"] = [
            name
            for name, sha in source_hashes.items()
            if not (repo / name).is_file() or digest(repo / name) != sha
        ]
        manifest["cupti_stop_rc"] = rc
        manifest["quiesce"] = quiesced
        inputs_after = asset_paths([target, *args.asset], args.asset_tree)
        manifest["asset_drift"] = [
            str(p)
            for p in inputs
            if not p.is_file() or digest(p) != manifest["explicit_input_sha256"][str(p)]
        ]
        manifest["asset_inventory_added"] = [
            str(p) for p in set(inputs_after) - set(inputs)
        ]
        # Both excluded files are still being written at this point, so hashing
        # them here would only record a value that is wrong by the time anyone
        # checks it. The exclusion is named in the manifest rather than silent.
        excluded = ("manifest.json", "tail.log")
        manifest["artifacts_sha256"] = {
            p.name: digest(p) for p in output.iterdir() if p.name not in excluded
        }
        manifest["artifacts_sha256_excluded"] = list(excluded)
        note("artifacts_hashed")
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
        note("manifest_written")
        tail.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--asset-tree", type=Path, action="append", default=[])
    # torch's compile-worker quiesce timer only checks its exit flag every
    # ``quiesce_async_compile_time / 2`` seconds, so the bound has to clear that.
    parser.add_argument("--quiesce-timeout", type=float, default=60.0)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.target[:1] == ["--"]:
        args.target = args.target[1:]
    if not args.target:
        parser.error("a Python script is required after --")
    run(args)
