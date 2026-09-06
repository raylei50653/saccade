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
    except BaseException as exc:
        emit(
            "target_exception",
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        # Passive process mappings; no CUDA status queries at failure time.
        maps = Path("/proc/self/maps").read_text()
        (output / "maps.txt").write_text(maps)
        restore()
        threading.excepthook = old_hook
        sys.argv, sys.path = old_argv, old_path
        rc = native.attribution_stop()
        emit(
            "harness_stopped",
            cupti_rc=rc,
            live_threads=[
                {"name": t.name, "native_id": t.native_id}
                for t in threading.enumerate()
                if t is not threading.current_thread()
            ],
        )
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
        manifest["mapped_file_sha256_after"] = {
            name: digest(Path(name))
            for name in sorted(libraries)
            if Path(name).is_file()
        }
        manifest["source_drift"] = [
            name
            for name, sha in source_hashes.items()
            if not (repo / name).is_file() or digest(repo / name) != sha
        ]
        manifest["cupti_stop_rc"] = rc
        inputs_after = asset_paths([target, *args.asset], args.asset_tree)
        manifest["asset_drift"] = [
            str(p)
            for p in inputs
            if not p.is_file() or digest(p) != manifest["explicit_input_sha256"][str(p)]
        ]
        manifest["asset_inventory_added"] = [
            str(p) for p in set(inputs_after) - set(inputs)
        ]
        manifest["artifacts_sha256"] = {
            p.name: digest(p) for p in output.iterdir() if p.name != "manifest.json"
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--asset-tree", type=Path, action="append", default=[])
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.target[:1] == ["--"]:
        args.target = args.target[1:]
    if not args.target:
        parser.error("a Python script is required after --")
    run(args)
