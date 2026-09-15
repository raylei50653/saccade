"""Own Green Context routing for PyTorch, TensorRT, native CUDA and graph launches."""

# status: diagnostic
#
# Diagnostic execution owner for #419 phase B. It does not modify production
# sources; it installs process-level routing before the evaluator imports:
#
# 1. Creates one Green Context with a requested SM count (or none for the
#    full-device primary baseline) and records the context-reported SM count.
# 2. Replaces ``torch.cuda.Stream`` so every stream the pipeline creates (the
#    double-buffer detector lane, graph capture streams, worker streams) is an
#    explicit ``ExternalStream`` created with ``cuGreenCtxStreamCreate``.
#    ``torch.cuda.default_stream`` returns an explicit owned stream too, and the
#    owned main stream is made current, so ordinary PyTorch launches, TensorRT
#    ``execute_async_v3(current_stream)`` and native launches that receive
#    ``torch.cuda.current_stream().cuda_stream`` all target owned streams.
# 3. Makes the Green Context current on the main thread and on every thread
#    started afterwards, so runtime-created streams (nvJPEG, TensorRT internals)
#    and NULL-stream launches also land in the Green Context. TorchInductor's
#    generated code calls ``torch.cuda.set_device(0)`` on every call, which
#    reverts the current context to the primary context; the owner wraps
#    ``set_device`` to re-assert its context and counts each re-assertion.
# 4. Records a CUPTI activity trace of every kernel/memcpy/memset with its
#    execution context, and an SM-id probe on the owned stream, so a reporter
#    can fail closed instead of trusting steps 1-3.
#
# Context binding facts measured on this host (feasibility, 2026-09-15): a
# ``cuCtxSetCurrent`` survives ordinary tensor ops, guards and events but is
# reverted by ``torch.cuda.set_device``; a CUDA graph executes in the context
# of its *capture* stream regardless of the replay stream; ``cudaStreamCreate``
# under a current Green Context yields a Green stream; pre-existing PyTorch pool
# streams stay in the primary context. Everything above is re-verified per run.
import ctypes
import json
import threading
import time


class OwnerError(RuntimeError):
    pass


def _driver():
    from cuda.bindings import driver as d

    return d


def call(name, *args):
    d = _driver()
    value = getattr(d, name)(*args)
    if int(value[0]):
        raise OwnerError(f"{name}: {value[0]}")
    return value[1] if len(value) == 2 else value[1:]


class GreenExecutionOwner:
    def __init__(self, sm_count, audit_library, probe_library, trace_path):
        """``sm_count`` is an int for a Green Context, or None for the primary
        full-device baseline routed through the same explicit-stream layer."""
        self.sm_count = sm_count
        self.trace_path = str(trace_path)
        self.audit = ctypes.CDLL(str(audit_library)) if audit_library else None
        self.probe = ctypes.CDLL(str(probe_library))
        self.probe.smid_probe_launch.restype = ctypes.c_char_p
        self.probe.smid_probe_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        if self.audit:
            self.audit.audit_error.restype = ctypes.c_char_p
            self.audit.audit_start.argtypes = [ctypes.c_char_p]
            self.audit.audit_mark.argtypes = [ctypes.c_char_p]
            self.audit.audit_timestamp.restype = ctypes.c_uint64
            self.audit.audit_context_id.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
        self.green = None
        self.context = None
        self.primary = None
        self.streams = []  # (handle, priority, thread ident, creating site)
        self.thread_switches = 0
        self.thread_switch_errors = []
        self.set_device_reasserts = 0
        self.calibration = []
        self.marks = []
        self.probe_results = {}
        self.report = {}
        self.installed = False
        self._original_stream = None
        self._original_default_stream = None
        self._original_thread_run = None
        self._lock = threading.Lock()

    # -- installation -------------------------------------------------------
    def install(self):
        import torch

        if self.installed:
            raise OwnerError("owner already installed")
        d = _driver()
        if self.audit:
            self._audit_call("audit_start", self.trace_path.encode())
        call("cuInit", 0)
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise OwnerError("the owner is qualified on a single visible GPU only")
        # torch.cuda.init() alone does not make the primary context current;
        # a device synchronize does, without launching any kernel.
        torch.cuda.synchronize()
        self.primary = call("cuCtxGetCurrent")
        retained = call("cuDevicePrimaryCtxRetain", 0)
        if int(self.primary) == 0 or int(self.primary) != int(retained):
            raise OwnerError("PyTorch is not running on the device primary context")
        resource = call(
            "cuDeviceGetDevResource", 0, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
        self.report["device_sm_count"] = resource.sm.smCount
        self.report["min_partition"] = resource.sm.minSmPartitionSize
        self.report["alignment"] = resource.sm.smCoscheduledAlignment
        self.report["requested_sm_count"] = self.sm_count
        if self.sm_count is not None:
            groups, actual_groups, _ = call(
                "cuDevSmResourceSplitByCount", 1, resource, 0, self.sm_count
            )
            if actual_groups != 1:
                raise OwnerError(f"unexpected group count {actual_groups}")
            desc = call("cuDevResourceGenerateDesc", groups, 1)
            self.green = call(
                "cuGreenCtxCreate",
                desc,
                0,
                d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
            actual = call(
                "cuGreenCtxGetDevResource",
                self.green,
                d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
            self.report["actual_sm_count"] = actual.sm.smCount
            self.context = call("cuCtxFromGreenCtx", self.green)
            call("cuCtxSetCurrent", self.context)
        else:
            self.report["actual_sm_count"] = resource.sm.smCount
            self.context = self.primary
        self.report["primary_context"] = int(self.primary)
        self.report["execution_context"] = int(self.context)
        self.report["green_context"] = int(self.green) if self.green else None
        if self.audit:
            self.report["cupti_primary_context_id"] = self._context_id(self.primary)
            self.report["cupti_execution_context_id"] = self._context_id(self.context)
        least, greatest = call("cuCtxGetStreamPriorityRange")
        self._priority_range = (int(least), int(greatest))
        self._install_stream_patch(torch)
        self._install_set_device_patch(torch)
        self._install_thread_hook()
        self.main_stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.main_stream)
        self.installed = True
        self.mark("owner_installed")
        self.probe_results["before"] = self.probe_partition()

    def _install_stream_patch(self, torch):
        owner = self
        original = torch.cuda.Stream
        self._original_stream = original
        self._original_default_stream = torch.cuda.default_stream

        class OwnedStream(torch.cuda.ExternalStream):
            """Explicit stream created in the owner's context."""

            def __new__(cls, device=None, priority=0, **kwargs):
                if "stream_id" in kwargs:
                    # torch.cuda.current_stream() reconstructs wrappers this way.
                    return original.__new__(
                        original, device=device, priority=priority, **kwargs
                    )
                handle = owner._create_stream(int(priority))
                return super().__new__(cls, handle, device=device)

        self.stream_class = OwnedStream
        torch.cuda.Stream = OwnedStream

        def default_stream(device=None):
            return owner.main_stream

        torch.cuda.default_stream = default_stream

    def _create_stream(self, priority):
        d = _driver()
        least, greatest = self._priority_range
        priority = max(min(priority, least), greatest)
        if self.green is not None:
            handle = call(
                "cuGreenCtxStreamCreate",
                self.green,
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                priority,
            )
        else:
            handle = call(
                "cuStreamCreateWithPriority",
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                priority,
            )
        with self._lock:
            self.streams.append(
                {
                    "handle": int(handle),
                    "priority": priority,
                    "thread": threading.get_ident(),
                    "created_at": time.perf_counter(),
                }
            )
        return int(handle)

    def _install_set_device_patch(self, torch):
        owner = self
        original = torch.cuda.set_device
        self._original_set_device = original

        def set_device(device):
            original(device)
            if torch.cuda.current_device() != 0:
                raise OwnerError("the owner only routes device 0")
            call("cuCtxSetCurrent", owner.context)
            with owner._lock:
                owner.set_device_reasserts += 1

        torch.cuda.set_device = set_device

    def current_context_owned(self):
        return int(call("cuCtxGetCurrent")) == int(self.context)

    def _install_thread_hook(self):
        owner = self
        original_run = threading.Thread.run
        self._original_thread_run = original_run

        def run(thread_self):
            try:
                call("cuCtxSetCurrent", owner.context)
                with owner._lock:
                    owner.thread_switches += 1
            except Exception as exc:  # pragma: no cover - reported, not raised
                with owner._lock:
                    owner.thread_switch_errors.append(repr(exc))
            return original_run(thread_self)

        threading.Thread.run = run

    # -- evidence -----------------------------------------------------------
    def _audit_call(self, name, *args):
        if getattr(self.audit, name)(*args):
            raise OwnerError(f"{name}: {self.audit.audit_error().decode()}")

    def _context_id(self, ctx):
        out = ctypes.c_uint32(0)
        self._audit_call(
            "audit_context_id", ctypes.c_void_p(int(ctx)), ctypes.byref(out)
        )
        return out.value

    def mark(self, label):
        """Record a labelled CUPTI timestamp with a host monotonic pair."""
        host = time.perf_counter()
        if self.audit:
            cupti = int(self.audit.audit_timestamp())
            self._audit_call("audit_mark", label.encode())
        else:
            cupti = None
        self.marks.append({"label": label, "host": host, "cupti": cupti})
        if cupti is not None:
            self.calibration.append((host, cupti))

    def probe_partition(self, blocks=4096, spin_ns=20_000):
        """Distinct SM ids reached by a spin kernel on the owned main stream,
        directly and through a CUDA graph captured on that stream."""
        import torch

        out = torch.zeros(blocks, dtype=torch.int32, device="cuda")
        result = {}
        with torch.cuda.stream(self.main_stream):
            error = self.probe.smid_probe_launch(
                ctypes.c_void_p(self.main_stream.cuda_stream),
                ctypes.c_void_p(out.data_ptr()),
                blocks,
                spin_ns,
            )
            if error:
                raise OwnerError(f"smid probe: {error.decode()}")
        torch.cuda.synchronize()
        result["direct_sm_ids"] = sorted(set(out.cpu().tolist()))
        out.zero_()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.main_stream):
            error = self.probe.smid_probe_launch(
                ctypes.c_void_p(self.main_stream.cuda_stream),
                ctypes.c_void_p(out.data_ptr()),
                blocks,
                spin_ns,
            )
            if error:
                raise OwnerError(f"smid probe capture: {error.decode()}")
        torch.cuda.synchronize()
        with torch.cuda.stream(self.main_stream):
            graph.replay()
        torch.cuda.synchronize()
        result["graph_sm_ids"] = sorted(set(out.cpu().tolist()))
        del graph, out
        return result

    def stream_ownership(self):
        d = _driver()
        rows = []
        for entry in self.streams:
            handle = entry["handle"]
            row = dict(entry)
            status, ctx = d.cuStreamGetCtx(d.CUstream(handle))
            row["context"] = int(ctx) if int(status) == 0 else None
            status, green = d.cuStreamGetGreenCtx(d.CUstream(handle))
            row["green_context"] = int(green) if int(status) == 0 else None
            row["owned"] = row["context"] == int(self.context) and (
                row["green_context"] == (int(self.green) if self.green else 0)
            )
            rows.append(row)
        return rows

    def finalize(self):
        """Stop the audit and return the ownership evidence (no verdict)."""
        import torch

        torch.cuda.synchronize()
        self.mark("owner_finalize")
        self.probe_results["after"] = self.probe_partition()
        torch.cuda.synchronize()
        current = call("cuCtxGetCurrent")
        if self.audit:
            self._audit_call("audit_stop")
        streams = self.stream_ownership()
        return {
            **self.report,
            "main_thread_context_still_owned_at_finalize": int(current)
            == int(self.context),
            "priority_range": self._priority_range,
            "main_stream": self.main_stream.cuda_stream,
            "streams": streams,
            "thread_context_switches": self.thread_switches,
            "set_device_reasserts": self.set_device_reasserts,
            "thread_context_switch_errors": self.thread_switch_errors,
            "marks": self.marks,
            "calibration": self.calibration,
            "probe": self.probe_results,
            "trace_path": self.trace_path if self.audit else None,
        }


def parse_trace(path):
    """Parse the CUPTI audit trace into kernels, copies, contexts and labels."""
    names = {}
    kernels, copies, memsets, contexts, labels, attributes = [], [], [], [], [], {}
    dropped = 0
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "N":
                names[int(parts[1])] = " ".join(parts[2:])
            elif kind == "K":
                kernels.append(
                    {
                        "start": int(parts[1]),
                        "end": int(parts[2]),
                        "context": int(parts[3]),
                        "stream": int(parts[4]),
                        "graph": int(parts[5]),
                        "name": names.get(int(parts[7]), "?"),
                    }
                )
            elif kind in "MS":
                (copies if kind == "M" else memsets).append(
                    {
                        "start": int(parts[1]),
                        "end": int(parts[2]),
                        "context": int(parts[3]),
                        "stream": int(parts[4]),
                        "graph": int(parts[5]),
                        "bytes": int(parts[8]),
                    }
                )
            elif kind == "C":
                contexts.append(
                    {
                        "context": int(parts[1]),
                        "device": int(parts[2]),
                        "null_stream": int(parts[3]),
                        "parent": int(parts[4]),
                        "is_green": int(parts[5]),
                        "sm_count": int(parts[6]),
                    }
                )
            elif kind == "G":
                contexts.append(
                    {
                        "context": int(parts[1]),
                        "device": int(parts[3]),
                        "null_stream": None,
                        "parent": int(parts[2]),
                        "is_green": 1,
                        "sm_count": int(parts[5]),
                        "tpcs": int(parts[4]),
                        "tpc_mask": [int(w) for w in parts[6:]],
                    }
                )
            elif kind == "L":
                labels.append({"cupti": int(parts[1]), "label": " ".join(parts[2:])})
            elif kind == "A":
                attributes[parts[1]] = " ".join(parts[2:])
            elif kind == "D":
                dropped += int(parts[1])
            elif kind == "E":
                dropped = max(dropped, int(parts[1].split("=")[1]))
    return {
        "kernels": kernels,
        "copies": copies,
        "memsets": memsets,
        "contexts": contexts,
        "labels": labels,
        "attributes": attributes,
        "dropped": dropped,
    }


def host_to_cupti(calibration, host_seconds):
    """Map a perf_counter reading to the CUPTI clock using the median offset."""
    offsets = sorted(cupti - host * 1e9 for host, cupti in calibration)
    spread_ns = offsets[-1] - offsets[0]
    return host_seconds * 1e9 + offsets[len(offsets) // 2], spread_ns


def audit_window(trace, execution_context_id, begin, end, null_stream_id=None):
    """Count records inside [begin, end] by execution context (CUPTI ids)."""
    summary = {}
    escapes = {}
    for kind in ("kernels", "memsets", "copies"):
        inside = [r for r in trace[kind] if begin <= r["start"] <= end]
        owned = [r for r in inside if r["context"] == execution_context_id]
        summary[kind] = {
            "total": len(inside),
            "in_execution_context": len(owned),
            "outside": len(inside) - len(owned),
            "graph_launched": sum(1 for r in owned if r["graph"]),
            "on_null_stream": sum(1 for r in owned if r["stream"] == null_stream_id)
            if null_stream_id is not None
            else None,
        }
        if kind == "kernels":
            for r in inside:
                if r["context"] != execution_context_id:
                    key = (r["context"], r["stream"], r["name"])
                    escapes[key] = escapes.get(key, 0) + 1
    summary["kernel_escapes"] = [
        {"context": c, "stream": s, "name": n, "count": v}
        for (c, s, n), v in sorted(escapes.items(), key=lambda kv: -kv[1])
    ]
    return summary


def dump(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
        handle.write("\n")
