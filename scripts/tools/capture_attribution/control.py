"""One bounded mechanism control. Run each case in its own attributed process."""

import ctypes
import json
from pathlib import Path
import sys
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


def mechanism(blocking, driver, joined=False):
    torch.cuda.init()
    torch.cuda.synchronize()
    rt = loaded_cudart()
    ptr = ctypes.c_void_p
    stream = ptr()
    graph = ptr()
    pptr = ctypes.POINTER(ptr)
    create = bind(rt, "cudaStreamCreateWithFlags", [pptr, ctypes.c_uint])
    get_flags = bind(rt, "cudaStreamGetFlags", [ptr, ctypes.POINTER(ctypes.c_uint)])
    query = bind(rt, "cudaStreamIsCapturing", [ptr, ctypes.POINTER(ctypes.c_int)])
    begin = bind(rt, "cudaStreamBeginCapture", [ptr, ctypes.c_int])
    end = bind(rt, "cudaStreamEndCapture", [ptr, pptr])
    destroy_graph = bind(rt, "cudaGraphDestroy", [ptr])
    destroy_stream = bind(rt, "cudaStreamDestroy", [ptr])
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
    checked(create(ctypes.byref(stream), 0 if blocking and not joined else 1))
    flags = ctypes.c_uint(99)
    checked(get_flags(stream, ctypes.byref(flags)))
    side, event_out, event_back = ptr(), ptr(), ptr()
    if joined:
        checked(create(ctypes.byref(side), 0))
        create_event = bind(rt, "cudaEventCreateWithFlags", [pptr, ctypes.c_uint])
        record = bind(rt, "cudaEventRecord", [ptr, ptr])
        wait = bind(rt, "cudaStreamWaitEvent", [ptr, ptr, ctypes.c_uint])
        destroy_event = bind(rt, "cudaEventDestroy", [ptr])
        checked(create_event(ctypes.byref(event_out), 2))
        checked(create_event(ctypes.byref(event_back), 2))
    opened, done = threading.Event(), threading.Event()
    result = {"blocking": blocking, "driver_getproc": driver, "flags": flags.value}

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
    print(json.dumps(result), flush=True)
    assert result["query_rc"] == (906 if blocking else 0), result
    assert result["end_rc"] == 0, result
    checked(destroy_graph(graph))
    checked(destroy_stream(stream))
    if joined:
        checked(destroy_stream(side))
        checked(destroy_event(event_out))
        checked(destroy_event(event_back))


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
    case = sys.argv[1]
    if case == "python":
        python_sites()
    else:
        mechanism(
            blocking=case.startswith("blocking"),
            driver=case.endswith("driver"),
            joined=case == "blocking-joined",
        )
