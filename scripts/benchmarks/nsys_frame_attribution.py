#!/usr/bin/env python
"""Per-frame overhead attribution from an nsys trace (node-mode CUDA graph trace).

Usage:
    nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
        -o /tmp/wg .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m ...
    .venv/bin/python scripts/benchmarks/nsys_frame_attribution.py /tmp/wg.nsys-rep

Frame anchor = selective_scan_fwd_kernel (3 launches/frame inside the detect
whole-graph). Steady window trims 5% head/tail. Sections:
  1. kernel-category busy per frame (+ per-stream / per-graph split)
  2. frame phases: detect-graph span vs tail (tracker/GMC overlap check)
  3. host-side CUDA API activity in the GPU-idle tail (quiet zones = pure Python)
  4. device-idle gap histogram, memcpy kinds

Caveat: node-mode tracing inflates wall/frame on the host side (~0.7-1.7 ms
observed); kernel/graph spans are hardware timestamps and stay trustworthy.
Compute the production bubble as (production wall/frame - GPU busy/frame here),
never from the trace's own gaps. See docs/reference/runbooks/nsys_profiling.md.
"""

from __future__ import annotations

import bisect
import re
import sqlite3
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CATS = [
    ("scan", r"selective_scan"),
    ("trt", r"_trt$|__myl"),
    ("gmc_fft", r"fft|dpVector"),
    (
        "tracker_cpp",
        r"occlusion|sinkhorn|auction|stage1_cost|nms_|iou|kalman|track|assign|farewell|relink|crop|reid|bank|embed",
    ),
    (
        "decode_pre",
        r"nvjpeg|jpeg|decode|huffman|dct|yuv|nv12|letterbox|rgba2rgb|grayscale",
    ),
    ("pointwise", r"elementwise|pointwise|copy_|cast|fill|where|clamp|sigmoid|silu"),
    (
        "reduce_index",
        r"reduce|index|gather|scatter|topk|Topk|TopK|nonzero|masked|sort|cub|arange|cumsum|Device(Select|Compact|RadixSort|Reduce)",
    ),
    ("gemm_conv", r"gemm|conv|cutlass|cudnn|fprop|implicit"),
    ("transpose_cat", r"nchwToNhwc|nhwcToNchw|transpose|cat_|CatArray|permute"),
    ("upsample_pool", r"upsample|pool"),
    ("triton", r"^triton_"),
]


def cat_of(name: str) -> str:
    for c, pat in CATS:
        if re.search(pat, name):
            return c
    return "other"


def union_ms(ivals: list[tuple[int, int]]) -> float:
    if not ivals:
        return 0.0
    ivals = sorted(ivals)
    total = 0
    cs, ce = ivals[0]
    for s, e in ivals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return (total + ce - cs) / 1e6


def group_runs(rows, split_ns=5e5):
    runs = []
    cs = ce = None
    for r in rows:
        if cs is None:
            cs, ce = r["start"], r["end"]
        elif r["start"] - ce > split_ns:
            runs.append((cs, ce))
            cs, ce = r["start"], r["end"]
        else:
            ce = max(ce, r["end"])
    if cs is not None:
        runs.append((cs, ce))
    return runs


def main() -> None:
    path = Path(sys.argv[1])
    if path.suffix == ".nsys-rep":
        sq = path.with_suffix(".sqlite")
        if not sq.exists():
            subprocess.run(
                ["nsys", "export", "--type", "sqlite", "-o", str(sq), str(path)],
                check=True,
                capture_output=True,
            )
        path = sq
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row

    kern = db.execute(
        """SELECT k.start, k.end, k.streamId, k.graphId, s.value AS name
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
           ORDER BY k.start"""
    ).fetchall()

    scan_starts = sorted(r["start"] for r in kern if "selective_scan" in r["name"])
    if len(scan_starts) < 30:
        sys.exit(
            "too few selective_scan anchors — is this a whole-graph node-mode trace?"
        )
    n = len(scan_starts)
    lo, hi = scan_starts[int(n * 0.05)], scan_starts[int(n * 0.95) - 1]
    frames = len([s for s in scan_starts if lo <= s <= hi]) / 3.0
    wall_ms = (hi - lo) / 1e6
    print(
        f"steady window: {wall_ms:.1f} ms, frames={frames:.0f}, wall/frame={wall_ms / frames:.3f} ms"
    )

    # ---- 1. category busy ----
    busy: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    by_stream: dict[int, float] = defaultdict(float)
    by_graph: dict[int, float] = defaultdict(float)
    tops: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ivals: list[tuple[int, int]] = []
    for r in kern:
        if not lo <= r["start"] <= hi:
            continue
        dur = (r["end"] - r["start"]) / 1e6
        c = cat_of(r["name"])
        busy[c] += dur
        cnt[c] += 1
        tops[c][r["name"]] += dur
        by_stream[r["streamId"]] += dur
        by_graph[r["graphId"] or 0] += dur
        ivals.append((r["start"], r["end"]))

    memcpy_rows = []
    for tbl, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        try:
            rows = db.execute(f"SELECT * FROM {tbl}").fetchall()
        except sqlite3.OperationalError:
            continue
        for r in rows:
            if not lo <= r["start"] <= hi:
                continue
            dur = (r["end"] - r["start"]) / 1e6
            key = label if label == "memset" else f"memcpy_k{r['copyKind']}"
            busy[key] += dur
            cnt[key] += 1
            ivals.append((r["start"], r["end"]))
            if label == "memcpy":
                memcpy_rows.append(r)

    dev_union = union_ms(ivals)
    tot = sum(busy.values())
    print(
        f"GPU union busy/frame: {dev_union / frames:.3f} ms | sum-of-kernels/frame: {tot / frames:.3f} ms "
        f"(overlap {tot / dev_union:.2f}x) | trace bubble/frame: {(wall_ms - dev_union) / frames:.3f} ms "
        f"(host-inflated — calibrate vs production wall)\n"
    )
    print(f"{'category':<16}{'ms/frame':>9}{'k/frame':>9}  top kernels")
    for c, ms in sorted(busy.items(), key=lambda x: -x[1]):
        t = (
            "; ".join(
                f"{k[:40]}={v / frames * 1e3:.0f}us"
                for k, v in sorted(tops[c].items(), key=lambda x: -x[1])[:3]
            )
            if c in tops
            else ""
        )
        print(f"{c:<16}{ms / frames:9.3f}{cnt[c] / frames:9.1f}  {t}")

    print(
        "\nper-stream busy ms/frame:",
        {
            s: round(m / frames, 3)
            for s, m in sorted(by_stream.items(), key=lambda x: -x[1])[:8]
        },
    )
    print(
        "per-graph busy ms/frame (0=eager):",
        {
            g: round(m / frames, 3)
            for g, m in sorted(by_graph.items(), key=lambda x: -x[1])[:6]
        },
    )
    if memcpy_rows:
        agg: dict[int, list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for r in memcpy_rows:
            a = agg[r["copyKind"]]
            a[0] += 1
            a[1] += (r["end"] - r["start"]) / 1e6
            a[2] += r["bytes"]
        print(
            "memcpy kinds (1=HtoD 2=DtoH 8=DtoD): "
            + "; ".join(
                f"k{k}: {v[0] / frames:.1f}/f {v[1] / frames * 1e3:.0f}us {v[2] / frames / 1024 / 1024:.1f}MB"
                for k, v in sorted(agg.items(), key=lambda x: -x[1][1])
            )
        )

    # ---- 2. frame phases ----
    byg: dict[int, list] = defaultdict(list)
    for r in kern:
        if r["graphId"] and lo <= r["start"] <= hi:
            byg[r["graphId"]].append(r)
    det_gid = max(byg, key=lambda g: sum(x["end"] - x["start"] for x in byg[g]))
    det = group_runs(byg[det_gid])
    print(f"\ndetect graph = graphId {det_gid}")
    for g in sorted(byg, key=lambda g: -sum(x["end"] - x["start"] for x in byg[g]))[:4]:
        spans = [(e - s) / 1e6 for s, e in group_runs(byg[g])]
        print(
            f"  graph {g}: n={len(spans)} span mean={st.mean(spans):.3f} p95={sorted(spans)[int(len(spans) * 0.95)]:.3f} ms"
        )

    periods = [(det[i + 1][0] - det[i][0]) / 1e6 for i in range(len(det) - 1)]
    print(
        f"detect period mean={st.mean(periods):.3f} median={st.median(periods):.3f} ms; "
        f"span mean={st.mean([(e - s) / 1e6 for s, e in det]):.3f} ms"
    )

    non_det = [
        (r["start"], r["end"])
        for r in kern
        if lo <= r["start"] <= hi and r["graphId"] != det_gid
    ]
    tails, tail_busy = [], []
    for i in range(len(det) - 1):
        det_end, nxt = det[i][1], det[i + 1][0]
        tails.append((nxt - det_end) / 1e6)
        tail_busy.append(union_ms([(s, e) for s, e in non_det if det_end <= s < nxt]))
    print(
        f"tail (detect end -> next detect): mean={st.mean(tails):.3f} ms, "
        f"other-work busy={st.mean(tail_busy):.3f} ms, idle={st.mean(tails) - st.mean(tail_busy):.3f} ms"
    )

    # ---- 3. host APIs in tail ----
    try:
        api = db.execute(
            """SELECT r.start, r.end, r.globalTid, s.value AS name
               FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id
               WHERE r.start BETWEEN ? AND ? ORDER BY r.start""",
            (lo, hi),
        ).fetchall()
    except sqlite3.OperationalError:
        api = []
    if api:
        tid_launch: dict[int, int] = defaultdict(int)
        for a in api:
            if "GraphLaunch" in a["name"]:
                tid_launch[a["globalTid"]] += 1
        main_tid = max(tid_launch, key=lambda t: tid_launch[t])
        api_time: dict[str, float] = defaultdict(float)
        quiet: list[float] = []
        nf = 0
        for i in range(len(det) - 1):
            det_end, nxt = det[i][1], det[i + 1][0]
            if nxt - det_end < 2e5:
                continue
            nf += 1
            prev = det_end
            for a in api:
                if a["start"] < det_end or a["start"] >= nxt:
                    continue
                api_time[a["name"]] += (min(a["end"], nxt) - a["start"]) / 1e6
                if a["globalTid"] == main_tid:
                    if a["start"] - prev > 3e5:
                        quiet.append((a["start"] - prev) / 1e6)
                    prev = max(prev, a["end"])
            if nxt - prev > 3e5:
                quiet.append((nxt - prev) / 1e6)
        print("\nCUDA API time inside tail (ms/frame):")
        for k in sorted(api_time, key=lambda x: -api_time[x])[:8]:
            print(f"  {k:<44} {api_time[k] / nf:7.3f}")
        print(
            f"main-thread quiet zones >0.3ms (pure Python): {len(quiet) / nf:.2f}/frame, {sum(quiet) / nf:.3f} ms/frame"
        )

    # ---- 4. gap histogram ----
    merged: list[tuple[int, int]] = []
    for s, e in sorted(ivals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = [(merged[i + 1][0] - merged[i][1]) / 1e6 for i in range(len(merged) - 1)]
    big = [g for g in gaps if g > 0.02]
    nf = len(det) - 1
    print(
        f"\ndevice-idle gaps >20us: {len(big) / nf:.1f}/frame, total {sum(big) / nf:.3f} ms/frame"
    )
    bins = [0.05, 0.1, 0.2, 0.5, 1.0, 1e9]
    acc: dict[float, float] = defaultdict(float)
    cbin: dict[float, int] = defaultdict(int)
    for g in big:
        b = bins[bisect.bisect_left(bins, g)]
        acc[b] += g
        cbin[b] += 1
    for b in bins:
        if acc[b]:
            lbl = f"<= {b} ms" if b < 1e9 else "> 1.0 ms"
            print(f"  {lbl}: {cbin[b] / nf:.1f}/frame, {acc[b] / nf:.3f} ms/frame")


if __name__ == "__main__":
    main()
