"""Measure confirmed-start arrivals and bounded rolling CUDA admission."""

# status: diagnostic
import argparse
import ctypes as c
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from cuda.bindings import driver as d

from admission_report import derive, summarize
from probe import call, digest

HERE = Path(__file__).resolve().parent


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
        "cuDevSmResourceSplitByCount", 2, resource, 0, args.sms
    )
    combined, ncombined, _ = call(
        "cuDevSmResourceSplitByCount", 1, resource, 0, 2 * args.sms
    )
    if count != 2 or ncombined != 1:
        raise RuntimeError("requires two equal pools and one matched combined pool")
    lib = c.CDLL(str(output / "admission.so"))
    lib.admission_create.restype = c.c_void_p
    lib.admission_error.restype = c.c_char_p
    lib.admission_destroy.argtypes = [c.c_void_p]
    lib.admission_run.argtypes = (
        [c.c_void_p] * 3 + [c.c_int] * 2 + [c.c_ulonglong] + [c.c_void_p] * 4
    )
    smid = c.CDLL(str(output / "smid_probe.so"))
    smid.smid_probe_launch.restype = c.c_char_p
    smid.smid_probe_launch.argtypes = [c.c_void_p, c.c_void_p, c.c_int, c.c_ulonglong]
    greens, streams, pools = [], [], []
    state = None
    try:
        for group, size in [
            *[(g, args.sms) for g in groups],
            (combined[0], 2 * args.sms),
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
        if set(pools[0]["sm_ids_before"]) & set(pools[1]["sm_ids_before"]):
            raise RuntimeError("disjoint pools overlap")
        state = lib.admission_create()
        if not state:
            raise RuntimeError(lib.admission_error().decode())
        cases = list(
            itertools.product(
                ("disjoint", "shared_budget_priority", "full_priority"),
                ((1, 1), (16, 1), (16, 4), (16, 16)),
                args.offset_us,
                (0, 1),
            )
        )
        rng = random.Random(args.seed)
        summaries, stable_value = [], None
        for rep in range(args.repeats):
            rng.shuffle(cases)
            for case_index, (route, (chunks, window), offset, swap) in enumerate(cases):
                pa, pb = (
                    (pools[swap], pools[1 - swap])
                    if route == "disjoint"
                    else (pools[2 if route == "shared_budget_priority" else 3],) * 2
                )
                case = {
                    "repeat": rep,
                    "route": route,
                    "chunks": chunks,
                    "window": window,
                    "offset_us": offset,
                    "swap": swap,
                }
                collected = {k: [] for k in ("host", "chunk_host", "stamps", "values")}
                for sample in range(-args.warmups, args.samples):
                    host = np.zeros(15, dtype=np.uint64)
                    chunk_host = np.zeros((16, 3), dtype=np.uint64)
                    stamps = np.zeros((128 + 1024 * chunks, 3), dtype=np.uint64)
                    values = np.zeros(128 + 1024 * chunks, dtype=np.float32)
                    if lib.admission_run(
                        state,
                        pa["streams"][0],
                        pb["streams"][1],
                        chunks,
                        window,
                        offset * 1000,
                        host.ctypes.data,
                        chunk_host.ctypes.data,
                        stamps.ctypes.data,
                        values.ctypes.data,
                    ):
                        raise RuntimeError(lib.admission_error().decode())
                    arrays = dict(
                        host=host, chunk_host=chunk_host, stamps=stamps, values=values
                    )
                    derive(case, arrays, pa["sm_ids_before"], pb["sm_ids_before"])
                    if stable_value is None:
                        stable_value = float(values[0])
                    if not np.all(values[:128] == stable_value):
                        raise RuntimeError("stable output differs between conditions")
                    if sample >= 0:
                        for key, value in arrays.items():
                            collected[key].append(value)
                name = f"raw-r{rep}-c{case_index}.npz"
                arrays = {k: np.stack(v) for k, v in collected.items()}
                np.savez_compressed(output / name, **arrays)
                rows = [
                    derive(
                        case,
                        {k: v[i] for k, v in arrays.items()},
                        pa["sm_ids_before"],
                        pb["sm_ids_before"],
                    )
                    for i in range(args.samples)
                ]
                summary = {
                    **case,
                    "raw": name,
                    "samples": args.samples,
                    **summarize(rows),
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                (output / "partial.json").write_text(
                    json.dumps(summaries, indent=2) + "\n"
                )
        for pool in pools:
            pool["sm_ids_after"] = ids(pool["streams"][0])
            if pool["sm_ids_before"] != pool["sm_ids_after"]:
                raise RuntimeError("SM identities changed")
        return {
            "status": "validated_synthetic_admission",
            "pools": pools,
            "device_sms": resource.sm.smCount,
            "remainder_sms": remainder.sm.smCount,
            "highest_priority": highest,
            "stable_value": stable_value,
            "summary": summaries,
        }
    finally:
        call("cuCtxSetCurrent", primary)
        for stream in streams:
            call("cuStreamSynchronize", stream)
        lib.admission_destroy(state)
        for stream in streams:
            call("cuStreamDestroy", stream)
        for green in greens:
            call("cuGreenCtxDestroy", green)
        call("cuDevicePrimaryCtxRelease", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sms", default=16, type=int)
    parser.add_argument("--samples", default=50, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--warmups", default=5, type=int)
    parser.add_argument("--offset-us", nargs="+", default=[0, 250], type=int)
    parser.add_argument("--seed", default=420, type=int)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if (
        min(args.sms, args.samples, args.repeats) < 1
        or args.warmups < 0
        or min(args.offset_us) < 0
        or len(set(args.offset_us)) != len(args.offset_us)
    ):
        parser.error("invalid counts or offsets")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "execution_sources"
    sources.mkdir()
    for path in (
        HERE / "admission.cu",
        HERE / "admission_probe.py",
        HERE / "admission_report.py",
        HERE / "probe.py",
        HERE.parent / "resource_partition" / "smid_probe.cu",
    ):
        shutil.copy2(path, sources / path.name)
    try:
        metadata = {
            "command": sys.argv,
            "args": {**vars(args), "output": str(output)},
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {p.name: digest(p) for p in sources.iterdir()},
            "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
            "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True),
        }
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for stem in ("admission", "smid_probe"):
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
        result["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        (output / "result.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}) + "\n"
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
