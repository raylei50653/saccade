"""One bounded mechanism control. Run each case in its own attributed process."""

# status: diagnostic

import ctypes
import json
from pathlib import Path
import argparse
import threading

import torch


def loaded_cudart():
    names = {
        line.split(maxsplit=5)[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libcudart.so" in line and len(line.split(maxsplit=5)) == 6
    }
    if len(names) != 1:
        raise RuntimeError(f"Expected one loaded cudart, found {names}")
    return ctypes.CDLL(names.pop())


def bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, ctypes.c_int
    return fn


def checked(rc):
    if rc != 0:
        raise RuntimeError(f"control CUDA rc={rc}")


def load_owner_helper(path: Path):
    helper = ctypes.CDLL(str(path.resolve()))
    word = ctypes.c_size_t
    pword = ctypes.POINTER(word)
    signatures = {
        "attribution_control_cuda_create_default": [pword],
        "attribution_control_cuda_create_flags": [pword, ctypes.c_uint],
        "attribution_control_cuda_create_priority": [
            pword,
            ctypes.c_uint,
            ctypes.c_int,
        ],
        "attribution_control_cuda_destroy": [word],
        "attribution_control_cu_create_flags": [pword, ctypes.c_uint],
        "attribution_control_cu_create_priority": [
            pword,
            ctypes.c_uint,
            ctypes.c_int,
        ],
        "attribution_control_cu_destroy": [word],
    }
    for name, args in signatures.items():
        bind(helper, name, args)
    return helper


def create_stream(helper, kind, flags):
    raw = ctypes.c_size_t()
    if kind == "runtime-default":
        checked(helper.attribution_control_cuda_create_default(ctypes.byref(raw)))
    elif kind == "runtime-flags":
        checked(helper.attribution_control_cuda_create_flags(ctypes.byref(raw), flags))
    elif kind == "runtime-priority":
        checked(
            helper.attribution_control_cuda_create_priority(ctypes.byref(raw), flags, 0)
        )
    elif kind == "driver-flags":
        checked(helper.attribution_control_cu_create_flags(ctypes.byref(raw), flags))
    elif kind == "driver-priority":
        checked(
            helper.attribution_control_cu_create_priority(ctypes.byref(raw), flags, 0)
        )
    else:
        raise ValueError(f"unknown creation kind: {kind}")
    if not raw.value:
        raise RuntimeError(f"{kind} returned a null stream")
    return ctypes.c_void_p(raw.value)


def destroy_stream(helper, driver, stream):
    raw = ctypes.c_size_t(stream.value)
    if driver:
        checked(helper.attribution_control_cu_destroy(raw))
    else:
        checked(helper.attribution_control_cuda_destroy(raw))


def mechanism(
    blocking, driver, creation_kind, helper_path, joined=False, recreate=False
):
    torch.cuda.init()
    torch.cuda.synchronize()
    helper = load_owner_helper(helper_path)
    rt = loaded_cudart()
    ptr = ctypes.c_void_p
    stream = ptr()
    graph = ptr()
    pptr = ctypes.POINTER(ptr)
    get_flags = bind(rt, "cudaStreamGetFlags", [ptr, ctypes.POINTER(ctypes.c_uint)])
    query = bind(rt, "cudaStreamIsCapturing", [ptr, ctypes.POINTER(ctypes.c_int)])
    begin = bind(rt, "cudaStreamBeginCapture", [ptr, ctypes.c_int])
    end = bind(rt, "cudaStreamEndCapture", [ptr, pptr])
    destroy_graph = bind(rt, "cudaGraphDestroy", [ptr])
    if driver:
        drv = ctypes.CDLL("libcuda.so.1")
        get_proc = bind(
            drv,
            "cuGetProcAddress_v2",
            [
                ctypes.c_char_p,
                pptr,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_int),
            ],
        )
        address, status = ptr(), ctypes.c_int(-1)
        checked(
            get_proc(
                b"cuStreamBeginCapture",
                ctypes.byref(address),
                12000,
                0,
                ctypes.byref(status),
            )
        )
        if status.value != 0 or not address.value:
            raise RuntimeError(f"cuGetProcAddress query status={status.value}")
        begin = ctypes.CFUNCTYPE(ctypes.c_int, ptr, ctypes.c_int)(address.value)
        end = bind(drv, "cuStreamEndCapture", [ptr, pptr])
    stream = create_stream(helper, creation_kind, 0 if blocking and not joined else 1)
    flags = ctypes.c_uint(99)
    checked(get_flags(stream, ctypes.byref(flags)))
    side, event_out, event_back = ptr(), ptr(), ptr()
    if joined:
        side = create_stream(helper, "runtime-flags", 0)
        create_event = bind(rt, "cudaEventCreateWithFlags", [pptr, ctypes.c_uint])
        record = bind(rt, "cudaEventRecord", [ptr, ptr])
        wait = bind(rt, "cudaStreamWaitEvent", [ptr, ptr, ctypes.c_uint])
        destroy_event = bind(rt, "cudaEventDestroy", [ptr])
        checked(create_event(ctypes.byref(event_out), 2))
        checked(create_event(ctypes.byref(event_back), 2))
    opened, done = threading.Event(), threading.Event()
    result = {
        "blocking": blocking,
        "creation_kind": creation_kind,
        "driver_getproc": driver,
        "flags": flags.value,
    }

    def worker():
        if not opened.wait(10):
            result["timeout"] = True
        else:
            status = ctypes.c_int(-1)
            # Explicit legacy handle; no kernels or allocations in this worker.
            result["query_rc"] = query(ptr(1), ctypes.byref(status))
            result["query_status"] = status.value
        done.set()

    thread = threading.Thread(target=worker, name="legacy-query-control")
    thread.start()
    checked(begin(stream, 1))
    if joined:
        checked(record(event_out, stream))
        checked(wait(side, event_out, 0))
        side_status = ctypes.c_int(-1)
        checked(query(side, ctypes.byref(side_status)))
        result["joined_stream_status"] = side_status.value
    opened.set()
    if not done.wait(10):
        raise RuntimeError("control worker timed out")
    thread.join()
    if joined:
        checked(record(event_back, side))
        checked(wait(stream, event_back, 0))
    result["end_rc"] = end(stream, ctypes.byref(graph))
    assert result["query_rc"] == (906 if blocking else 0), result
    assert result["end_rc"] == 0, result
    checked(destroy_graph(graph))
    first_handle = stream.value
    destroy_stream(helper, driver, stream)
    if joined:
        destroy_stream(helper, False, side)
        checked(destroy_event(event_out))
        checked(destroy_event(event_back))
    if recreate:
        replacement = create_stream(helper, creation_kind, 0)
        result["recreate"] = {
            "first": first_handle,
            "second": replacement.value,
            "same_handle": first_handle == replacement.value,
        }
        destroy_stream(helper, driver, replacement)
    print(json.dumps(result), flush=True)


def _whole_graph_capture(sample):
    def fn(value):
        return value * 2

    return torch.cuda.make_graphed_callables(fn, (sample,))


def python_sites():
    from saccade.perception.eval.cuda_capture import graph_capture

    sample = torch.ones(8, device="cuda")
    _whole_graph_capture(sample)(sample)
    for label in ("nms.main_nocopyback", "gmc.direct"):
        graph = torch.cuda.CUDAGraph()
        with graph_capture(graph, label=label):
            sample.mul_(2)
    torch.cuda.synchronize()
    print(json.dumps({"python_sites": "completed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--helper", type=Path)
    args = parser.parse_args()
    if args.case == "python":
        python_sites()
    else:
        if args.helper is None:
            parser.error("--helper is required for native creation controls")
        creation_kinds = {
            "blocking-runtime": "runtime-default",
            "nonblocking-runtime": "runtime-priority",
            "blocking-driver": "driver-flags",
            "nonblocking-driver": "driver-priority",
            "blocking-joined": "runtime-flags",
        }
        if args.case not in creation_kinds:
            parser.error(f"unknown control case: {args.case}")
        mechanism(
            blocking=args.case.startswith("blocking"),
            driver=args.case.endswith("driver"),
            creation_kind=creation_kinds[args.case],
            helper_path=args.helper,
            joined=args.case == "blocking-joined",
            recreate=args.case == "blocking-runtime",
        )
