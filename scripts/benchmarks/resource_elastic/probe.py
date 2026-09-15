"""Measure synthetic co-load on shared versus disjoint Green Context SM pools."""

# status: diagnostic
import argparse
import ctypes as c
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from cuda.bindings import driver as d

HERE = Path(__file__).resolve().parent


def call(name, *args):
    result = getattr(d, name)(*args)
    if int(result[0]):
        raise RuntimeError(f"{name}: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(values):
    return dict(
        zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99]).tolist())
    )


def run(args, output):
    call("cuInit", 0)
    if call("cuDeviceGetCount") != 1:
        raise RuntimeError("probe requires exactly one visible device")
    primary = call("cuDevicePrimaryCtxRetain", 0)
    call("cuCtxSetCurrent", primary)
    least_priority, highest_priority = call("cuCtxGetStreamPriorityRange")
    if highest_priority == least_priority:
        raise RuntimeError("priority control requires distinct stream priorities")
    resource = call(
        "cuDeviceGetDevResource", 0, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    )
    groups, count, remainder = call(
        "cuDevSmResourceSplitByCount", 2, resource, 0, args.sms
    )
    if count != 2:
        raise RuntimeError(f"expected two disjoint groups, got {count}")
    lib = c.CDLL(str(output / "probe.so"))
    lib.probe_create.restype = c.c_void_p
    lib.probe_create.argtypes = [c.c_int]
    lib.probe_error.restype = c.c_char_p
    lib.probe_destroy.argtypes = [c.c_void_p]
    lib.probe_run.argtypes = [
        c.c_void_p,
        c.c_void_p,
        c.c_void_p,
        c.c_int,
        c.POINTER(c.c_double),
        c.POINTER(c.c_uint),
    ]
    smid = c.CDLL(str(output / "smid_probe.so"))
    smid.smid_probe_launch.restype = c.c_char_p
    smid.smid_probe_launch.argtypes = [c.c_void_p, c.c_void_p, c.c_int, c.c_ulonglong]
    pools, greens, streams = [], [], []
    state = None
    try:
        for group in groups:
            desc = call("cuDevResourceGenerateDesc", [group], 1)
            green = call(
                "cuGreenCtxCreate",
                desc,
                0,
                d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
            greens.append(green)
            ctx = call("cuCtxFromGreenCtx", green)
            actual = call(
                "cuGreenCtxGetDevResource",
                green,
                d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            ).sm.smCount
            if actual != args.sms:
                raise RuntimeError(f"requested {args.sms}, got {actual}")
            pair = []
            for priority in (0, 0, highest_priority):
                stream = call(
                    "cuGreenCtxStreamCreate",
                    green,
                    d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                    priority,
                )
                streams.append(stream)
                if int(call("cuStreamGetCtx", stream)) != int(ctx):
                    raise RuntimeError("stream context mismatch")
                pair.append(stream)
            pools.append(
                {
                    "context": int(ctx),
                    "actual_sms": actual,
                    "streams": [int(s) for s in pair],
                }
            )
        call("cuCtxSetCurrent", primary)
        full = []
        for priority in (0, 0, highest_priority):
            stream = call(
                "cuStreamCreateWithPriority",
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                priority,
            )
            streams.append(stream)
            full.append(int(stream))

        def probe_ids(stream):
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
            pool["sm_ids_before"] = probe_ids(pool["streams"][0])
            if len(pool["sm_ids_before"]) != pool["actual_sms"]:
                raise RuntimeError("SM probe count mismatch")
        if set(pools[0]["sm_ids_before"]) & set(pools[1]["sm_ids_before"]):
            raise RuntimeError("pools overlap")
        full_ids = probe_ids(full[0])
        if len(full_ids) != resource.sm.smCount:
            raise RuntimeError("full device probe count mismatch")
        state = lib.probe_create(args.compute_chunks)
        if not state:
            raise RuntimeError(lib.probe_error().decode())
        a, b = pools
        routes = {
            "reserved_solo": (a["streams"][0], a["streams"][1], a, a),
            "same_pool": (*a["streams"][:2], a, a),
            "same_pool_priority": (a["streams"][2], a["streams"][1], a, a),
            "disjoint": (a["streams"][0], b["streams"][0], a, b),
            "full_shared": (
                *full[:2],
                {"sm_ids_before": full_ids},
                {"sm_ids_before": full_ids},
            ),
            "full_solo": (
                *full[:2],
                {"sm_ids_before": full_ids},
                {"sm_ids_before": full_ids},
            ),
            "full_priority": (
                full[2],
                full[1],
                {"sm_ids_before": full_ids},
                {"sm_ids_before": full_ids},
            ),
        }
        cases = [
            (name, kind)
            for name in routes
            for kind in ([0] if name.endswith("solo") else [1, 2])
        ]
        rng = random.Random(args.seed)
        rows, summaries = [], []
        expected_value = None
        for rep in range(args.repeats):
            rng.shuffle(cases)
            for name, kind in cases:
                sa, sb, pa, pb = routes[name]
                samples = []
                for sample in range(-10, args.samples):
                    metrics, ids = (
                        (c.c_double * 9)(),
                        (c.c_uint * (128 + 1024 * args.compute_chunks))(),
                    )
                    if lib.probe_run(state, sa, sb, kind, metrics, ids):
                        raise RuntimeError(lib.probe_error().decode())
                    observed_a, observed_b = (
                        set(ids[:128]),
                        set(ids[128:]) - {2**32 - 1},
                    )
                    if not observed_a <= set(pa["sm_ids_before"]):
                        raise RuntimeError("stable work escaped its pool")
                    if kind and not observed_b <= set(pb["sm_ids_before"]):
                        raise RuntimeError("burst work escaped its pool")
                    if expected_value is None:
                        expected_value = metrics[4]
                    if not (metrics[4] == metrics[5] == expected_value):
                        raise RuntimeError("stable output changed")
                    if sample < 0:
                        continue
                    row = {
                        "repeat": rep,
                        "route": name,
                        "kind": kind,
                        "sample": sample,
                        "host_response_ms": metrics[0],
                        "stable_gpu_ms": metrics[1],
                        "burst_gpu_ms": metrics[2],
                        "envelope_overlap_ms": metrics[3],
                        "stable_start_offset_ms": metrics[6],
                        "stable_finish_offset_ms": metrics[7],
                        "stable_blocks_overlapping_burst_fraction": metrics[8],
                        "stable_sm_ids": sorted(observed_a),
                        "burst_sm_ids": sorted(observed_b) if kind else [],
                    }
                    samples.append(row)
                rows.extend(samples)
                summary = {
                    "repeat": rep,
                    "route": name,
                    "kind": kind,
                    "samples": len(samples),
                }
                for key in (
                    "host_response_ms",
                    "stable_gpu_ms",
                    "burst_gpu_ms",
                    "stable_start_offset_ms",
                    "stable_finish_offset_ms",
                    "stable_blocks_overlapping_burst_fraction",
                ):
                    summary[key] = percentiles([s[key] for s in samples])
                summary["overlap_fraction"] = sum(
                    s["envelope_overlap_ms"] > 0 for s in samples
                ) / len(samples)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                (output / "samples.json").write_text(json.dumps(rows) + "\n")
        for pool in pools:
            pool["sm_ids_after"] = probe_ids(pool["streams"][0])
            if pool["sm_ids_before"] != pool["sm_ids_after"]:
                raise RuntimeError("pool SM identities changed")
        return {
            "status": "validated_synthetic_probe",
            "device_sms": resource.sm.smCount,
            "remainder_sms_unused": remainder.sm.smCount,
            "pools": pools,
            "highest_priority": highest_priority,
            "compute_chunks": args.compute_chunks,
            "stable_value": expected_value,
            "summary": summaries,
            "full_pipeline_validated": False,
            "dynamic_routing_validated": False,
        }
    finally:
        call("cuCtxSetCurrent", primary)
        for stream in streams:
            call("cuStreamSynchronize", stream)
        if state:
            lib.probe_destroy(state)
        for stream in streams:
            call("cuStreamDestroy", stream)
        for green in greens:
            call("cuGreenCtxDestroy", green)
        call("cuDevicePrimaryCtxRelease", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sms", type=int, default=16)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--compute-chunks", type=int, choices=[1, 4, 16], default=1)
    parser.add_argument("--arch", default="sm_120")
    args = parser.parse_args()
    if min(args.sms, args.samples, args.repeats) < 1:
        parser.error("sms, samples and repeats must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "execution_sources"
    sources.mkdir()
    for path in (
        HERE / "probe.py",
        HERE / "probe.cu",
        HERE.parent / "resource_partition" / "smid_probe.cu",
    ):
        shutil.copy2(path, sources / path.name)
    metadata = {
        "command": sys.argv,
        "seed": args.seed,
        "samples": args.samples,
        "repeats": args.repeats,
        "compute_chunks": args.compute_chunks,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {p.name: digest(p) for p in sources.iterdir()},
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        for stem in ("probe", "smid_probe"):
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
        result = run(args, output)
    except Exception as exc:
        (output / "result.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n"
        )
        raise
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{digest(p)}  {p.relative_to(output)}\n"
            for p in sorted(output.rglob("*"))
            if p.is_file()
        )
    )


if __name__ == "__main__":
    main()
