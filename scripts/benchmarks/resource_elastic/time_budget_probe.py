"""Measure calibrated time-budget admission under repeated stable arrivals."""

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

from time_budget_report import derive, summarize, calibrate
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
    lib = c.CDLL(str(output / "time_budget.so"))
    lib.budget_create.restype = c.c_void_p
    lib.admission_error.restype = c.c_char_p
    lib.budget_destroy.argtypes = [c.c_void_p]
    lib.budget_run.argtypes = (
        [c.c_void_p] * 3 + [c.c_int] * 3 + [c.c_ulonglong] * 2 + [c.c_void_p] * 7
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
        state = lib.budget_create()
        if not state:
            raise RuntimeError(lib.admission_error().decode())

        def execute(pa, pb, iterations, estimates, window, budget, arrivals):
            n = len(iterations)
            arrays = {
                "host": np.zeros(3, dtype=np.uint64),
                "chunk_host": np.zeros((n, 5), dtype=np.uint64),
                "arrivals": np.zeros((arrivals, 9), dtype=np.uint64),
                "stamps": np.zeros((arrivals * 128 + n * 1024, 3), dtype=np.uint64),
                "values": np.zeros(arrivals * 128 + n * 1024, dtype=np.float32),
            }
            it = np.asarray(iterations, dtype=np.int32)
            est = np.asarray(estimates, dtype=np.uint64)
            if lib.budget_run(
                state,
                pa["streams"][0],
                pb["streams"][1],
                n,
                arrivals,
                window,
                budget,
                args.period_us * 1000,
                it.ctypes.data,
                est.ctypes.data,
                *[v.ctypes.data for v in arrays.values()],
            ):
                raise RuntimeError(lib.admission_error().decode())
            return arrays

        def lanes(route, swap):
            return (
                (pools[swap], pools[1 - swap])
                if route == "disjoint"
                else (pools[2 if route == "shared_budget_priority" else 3],) * 2
            )

        routes = ("disjoint", "shared_budget_priority", "full_priority")
        classes = [512, 2048, 4096]
        calibration = []
        # Isolated per-route/class training data precedes and is excluded from evaluation.
        for route, swap in itertools.product(routes, (0, 1)):
            pa, pb = lanes(route, swap)
            for iterations in classes:
                for _ in range(args.warmups):
                    execute(pa, pb, [iterations], [1], 1, 0, 0)
                raw = [
                    execute(pa, pb, [iterations], [1], 1, 0, 0)
                    for _ in range(args.calibration_samples)
                ]
                name = f"calibration-{route}-{swap}-{iterations}.npz"
                arrays = {k: np.stack([r[k] for r in raw]) for k in raw[0]}
                np.savez_compressed(output / name, **arrays)
                estimate = calibrate(arrays, pb["sm_ids_before"])
                calibration.append(
                    dict(
                        route=route,
                        swap=swap,
                        iterations=iterations,
                        raw=name,
                        estimate_ns=estimate,
                    )
                )
        estimates = {
            (r["route"], r["swap"], r["iterations"]): r["estimate_ns"]
            for r in calibration
        }
        if max(estimates.values()) > min(args.budgets_ms) * 1e6:
            raise RuntimeError(
                "calibrated unit exceeds smallest budget; no oversize bypass"
            )
        policies = [(w, 0) for w in (1, 4, 16)] + [
            (0, int(b * 1e6)) for b in args.budgets_ms
        ]
        cases = list(itertools.product(routes, policies, (0, 1)))
        rng = random.Random(args.seed)
        summaries, stable_value = [], None
        for rep in range(args.repeats):
            rng.shuffle(cases)
            for case_index, (route, (window, budget), swap) in enumerate(cases):
                pa, pb = lanes(route, swap)
                case = dict(
                    repeat=rep, route=route, window=window, budget_ns=budget, swap=swap
                )
                collected = []
                for sample in range(-args.warmups, args.samples):
                    # Same ordered work sequence for every policy, route and role.
                    workload = [512, 2048, 4096, 2048] * 64
                    random.Random(args.seed + 10000 * rep + sample).shuffle(workload)
                    est = [estimates[route, swap, i] for i in workload]
                    arrays = execute(pa, pb, workload, est, window, budget, 4)
                    derive(
                        case,
                        arrays,
                        pa["sm_ids_before"],
                        pb["sm_ids_before"],
                        workload,
                        est,
                        args.period_us * 1000,
                    )
                    if stable_value is None:
                        stable_value = float(arrays["values"][0])
                    if not np.all(arrays["values"][:512] == stable_value):
                        raise RuntimeError("stable output differs between conditions")
                    if sample >= 0:
                        collected.append(arrays)
                name = f"raw-r{rep}-c{case_index}.npz"
                arrays = {k: np.stack([v[k] for v in collected]) for k in collected[0]}
                np.savez_compressed(output / name, **arrays)
                rows = []
                for sample, raw in enumerate(collected):
                    workload = [512, 2048, 4096, 2048] * 64
                    random.Random(args.seed + 10000 * rep + sample).shuffle(workload)
                    rows.extend(
                        derive(
                            case,
                            raw,
                            pa["sm_ids_before"],
                            pb["sm_ids_before"],
                            workload,
                            [estimates[route, swap, i] for i in workload],
                            args.period_us * 1000,
                        )
                    )
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
            "status": "validated_synthetic_time_budget",
            "pools": pools,
            "device_sms": resource.sm.smCount,
            "remainder_sms": remainder.sm.smCount,
            "highest_priority": highest,
            "stable_value": stable_value,
            "summary": summaries,
            "calibration": calibration,
        }
    finally:
        call("cuCtxSetCurrent", primary)
        for stream in streams:
            call("cuStreamSynchronize", stream)
        lib.budget_destroy(state)
        for stream in streams:
            call("cuStreamDestroy", stream)
        for green in greens:
            call("cuGreenCtxDestroy", green)
        call("cuDevicePrimaryCtxRelease", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sms", default=16, type=int)
    parser.add_argument("--samples", default=20, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--warmups", default=3, type=int)
    parser.add_argument("--period-us", default=12000, type=int)
    parser.add_argument("--budgets-ms", nargs="+", default=[3, 6, 12], type=int)
    parser.add_argument("--calibration-samples", default=30, type=int)
    parser.add_argument("--seed", default=421, type=int)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if (
        min(args.sms, args.samples, args.repeats) < 1
        or args.warmups < 0
        or args.period_us < 1
        or args.calibration_samples < 2
        or min(args.budgets_ms) < 1
        or len(set(args.budgets_ms)) != len(args.budgets_ms)
    ):
        parser.error("invalid counts or offsets")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "execution_sources"
    sources.mkdir()
    for path in (
        HERE / "admission.cu",
        HERE / "time_budget.cu",
        HERE / "time_budget_probe.py",
        HERE / "time_budget_report.py",
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
        for stem in ("time_budget", "smid_probe"):
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
