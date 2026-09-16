"""Measure future-work routing and lane reclamation on reserved SM pools."""

# status: diagnostic
import argparse
import ctypes as c
import itertools
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from cuda.bindings import driver as d

HERE = Path(__file__).resolve().parent
from importlib.util import spec_from_file_location, module_from_spec

_helper_spec = spec_from_file_location(
    "elastic_helper", HERE.parent / "resource_elastic" / "probe.py"
)
_helper = module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
call, digest = _helper.call, _helper.digest
from report import derive, summarize, POLICIES  # noqa: E402


def run(args, output):
    call("cuInit", 0)
    if call("cuDeviceGetCount") != 1:
        raise RuntimeError("requires one visible device")
    primary = call("cuDevicePrimaryCtxRetain", 0)
    call("cuCtxSetCurrent", primary)
    least, highest = call("cuCtxGetStreamPriorityRange")
    if least == highest:
        raise RuntimeError("distinct stream priorities required")
    resource = call(
        "cuDeviceGetDevResource", 0, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    )
    groups, count, remainder = call(
        "cuDevSmResourceSplitByCount", 3, resource, 0, args.sms
    )
    combined, ncombined, _ = call(
        "cuDevSmResourceSplitByCount", 1, resource, 0, 3 * args.sms
    )
    if count != 3 or ncombined != 1:
        raise RuntimeError("requires three equal pools and one matched combined pool")
    lib = c.CDLL(str(output / "routing.so"))
    lib.routing_create.restype = c.c_void_p
    lib.admission_error.restype = c.c_char_p
    lib.routing_destroy.argtypes = [c.c_void_p]
    lib.routing_run.argtypes = (
        [c.c_void_p] * 4 + [c.c_int] * 3 + [c.c_ulonglong] + [c.c_void_p] * 5
    )
    smid = c.CDLL(str(output / "smid_probe.so"))
    smid.smid_probe_launch.restype = c.c_char_p
    smid.smid_probe_launch.argtypes = [c.c_void_p, c.c_void_p, c.c_int, c.c_ulonglong]
    greens, streams, pools = [], [], []
    state = None
    try:
        for group, size in [
            *[(g, args.sms) for g in groups],
            (combined[0], 3 * args.sms),
        ]:
            desc = call("cuDevResourceGenerateDesc", [group], 1)
            green = call(
                "cuGreenCtxCreate",
                desc,
                0,
                d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
            greens.append(green)
            context = call("cuCtxFromGreenCtx", green)
            actual = call(
                "cuGreenCtxGetDevResource",
                green,
                d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            ).sm.smCount
            if actual != size:
                raise RuntimeError("actual SM count mismatch")
            pair = []
            for priority in (highest, 0):
                stream = call(
                    "cuGreenCtxStreamCreate",
                    green,
                    d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                    priority,
                )
                streams.append(stream)
                if int(call("cuStreamGetCtx", stream)) != int(context):
                    raise RuntimeError("stream context mismatch")
                pair.append(int(stream))
            pools.append({"actual_sms": actual, "streams": pair})
        call("cuCtxSetCurrent", primary)
        full = []
        for priority in (highest, 0):
            stream = call(
                "cuStreamCreateWithPriority",
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                priority,
            )
            streams.append(stream)
            full.append(int(stream))
        pools.append({"actual_sms": resource.sm.smCount, "streams": full})

        def ids(stream):
            ptr = call("cuMemAlloc", 4096 * 4)
            try:
                error = smid.smid_probe_launch(stream, int(ptr), 4096, 100000)
                if error:
                    raise RuntimeError(error.decode())
                call("cuStreamSynchronize", d.CUstream(stream))
                host = (c.c_uint * 4096)()
                call("cuMemcpyDtoH", c.addressof(host), ptr, c.sizeof(host))
                return sorted(set(host))
            finally:
                call("cuMemFree", ptr)

        for pool in pools:
            pool["sm_ids_before"] = ids(pool["streams"][0])
            if len(pool["sm_ids_before"]) != pool["actual_sms"]:
                raise RuntimeError("incomplete SM probe")
        for a, b in itertools.combinations(pools[:3], 2):
            if set(a["sm_ids_before"]) & set(b["sm_ids_before"]):
                raise RuntimeError("disjoint pools overlap")
        state = lib.routing_create()
        if not state:
            raise RuntimeError(lib.admission_error().decode())
        cases = list(itertools.product(range(7), (1, 4), args.iterations, (0, 1)))
        rng = random.Random(args.seed)
        summaries = []
        for rep in range(args.repeats):
            rng.shuffle(cases)
            for index, (policy, window, iterations, swap) in enumerate(cases):
                lane_pools = (
                    [swap, 1 - swap, 2] if policy < 5 else [3 if policy == 5 else 4] * 3
                )
                lane_streams = [
                    pools[lane_pools[0]]["streams"][0],
                    pools[lane_pools[1]]["streams"][1],
                    pools[lane_pools[2]]["streams"][0],
                ]
                # A/C stable priority; B elastic priority. C never has competing
                # owners: it is drained before stable launches on that stream.
                case = dict(
                    repeat=rep,
                    policy=POLICIES[policy],
                    window=window,
                    iterations=iterations,
                    swap=swap,
                    lane_pools=lane_pools,
                )
                rows = []
                files = []
                for sample in range(-args.warmups, args.samples):
                    host = np.zeros(4, dtype=np.uint64)
                    records = np.zeros((4096, 6), dtype=np.uint64)
                    arrivals = np.zeros((16, 7), dtype=np.uint64)
                    stamps = np.zeros((4096, 256, 3), dtype=np.uint64)
                    values = np.zeros((4096, 256), dtype=np.float32)
                    arrays = dict(
                        host=host,
                        records=records,
                        arrivals=arrivals,
                        stamps=stamps,
                        values=values,
                    )
                    if lib.routing_run(
                        state,
                        *lane_streams,
                        policy,
                        window,
                        iterations,
                        args.period_us * 1000,
                        *[v.ctypes.data for v in arrays.values()],
                    ):
                        raise RuntimeError(lib.admission_error().decode())
                    n = int(host[3])
                    arrays = {
                        k: v[:n] if k in ("records", "stamps", "values") else v
                        for k, v in arrays.items()
                    }
                    row = derive(case, arrays, pools, args.period_us * 1000)
                    if sample >= 0:
                        name = f"r{rep}-c{index}-s{sample}.npz"
                        np.savez_compressed(output / name, **arrays)
                        files.append(name)
                        rows.append(row)
                summaries.append(dict(**case, raw=files, metrics=summarize(rows)))
                print(json.dumps(summaries[-1]), flush=True)
        for pool in pools:
            pool["sm_ids_after"] = ids(pool["streams"][0])
            if pool["sm_ids_before"] != pool["sm_ids_after"]:
                raise RuntimeError("SM identities changed")
        return dict(
            status="validated_synthetic_routing",
            pools=pools,
            summary=summaries,
            device_sms=resource.sm.smCount,
            remainder_sms=remainder.sm.smCount,
            highest_priority=highest,
        )
    finally:
        call("cuCtxSetCurrent", primary)
        for stream in streams:
            call("cuStreamSynchronize", stream)
        lib.routing_destroy(state)
        for stream in streams:
            call("cuStreamDestroy", stream)
        for green in greens:
            call("cuGreenCtxDestroy", green)
        call("cuDevicePrimaryCtxRelease", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sms", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", nargs="+", type=int, default=[2048, 8192])
    parser.add_argument("--period-us", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if (
        min(args.sms, args.samples, args.repeats, args.period_us, *args.iterations) < 1
        or args.warmups < 0
    ):
        parser.error("invalid counts")
    if len(set(args.iterations)) != len(args.iterations):
        parser.error("duplicate iteration classes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "execution_sources"
    sources.mkdir()
    for path in [
        HERE / "routing.cu",
        HERE / "probe.py",
        HERE / "report.py",
        HERE.parent / "resource_elastic" / "admission.cu",
        HERE.parent / "resource_partition" / "smid_probe.cu",
    ]:
        shutil.copy2(path, sources / path.name)
    shutil.copy2(
        HERE.parent / "resource_elastic" / "probe.py", sources / "elastic_probe.py"
    )
    manifest = dict(
        args={**vars(args), "output": str(output)},
        source_sha256={p.name: digest(p) for p in sources.iterdir()},
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        nvcc=subprocess.check_output(["nvcc", "--version"], text=True),
        device=subprocess.check_output(["nvidia-smi"], text=True),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for stem in ("routing", "smid_probe"):
            subprocess.run(
                [
                    "nvcc",
                    "-shared",
                    "-Xcompiler",
                    "-fPIC",
                    "-O2",
                    "-std=c++17",
                    f"-arch={args.arch}",
                    str(sources / f"{stem}.cu"),
                    "-o",
                    str(output / f"{stem}.so"),
                ],
                check=True,
            )
        with (output / "telemetry.csv").open("w") as telemetry:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv",
                    "-lms",
                    "100",
                ],
                stdout=telemetry,
                stderr=subprocess.STDOUT,
            )
            try:
                result = run(args, output)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        (output / "result.json").write_text(
            json.dumps(dict(status="failed", error=str(exc))) + "\n"
        )
        raise
    finally:
        (output / "SHA256SUMS").write_text(
            "".join(
                f"{digest(p)}  {p.relative_to(output)}\n"
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "SHA256SUMS"
            )
        )


if __name__ == "__main__":
    main()
