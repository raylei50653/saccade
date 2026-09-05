#!/usr/bin/env python3
# status: diagnostic
"""Re-verify the 2026-09-05 L2 source-attribution evidence pack from its raw CSVs.

Why this is in the repository when the evidence is not
------------------------------------------------------
The raw ``.ncu-rep`` reports, CSVs and sweep outputs are local retained evidence
(``~/.local/state/saccade/perf/l2-source-attribution-20260905/``).  They need to be
*auditable*, which is not the same as needing to be version-controlled.  What is
worth carrying is the conclusion layer -- ``docs/reference/benchmarks/frame_budget_20260905.md``
-- and the ability to recompute every number in it, which is this script.

``check_collection`` is the reusable part and applies to any ``ncu`` collection, not
only to this investigation.  It enforces two failure modes that are silent by default:

1. ``ncu`` accepts a **misspelled metric name without any error**.  The column is
   simply absent from the CSV, so a subtraction built on it yields a residual that
   looks entirely plausible and is fabricated.  This is not hypothetical -- it is the
   first thing that went wrong in this investigation.
2. Adding metrics silently raises the replay pass count, and only single-pass counters
   may be subtracted as a simultaneous ledger.  Multi-pass ``lts__`` counters on this
   host fail ``total == hit + miss`` and exceed 100% even when scoped to ``srcnode_gpc``.

Conservation is a consistency test, not a precision test: ``hit == total, miss == 0``
satisfies it trivially, which is exactly the shape the quantised pipeline rows take.

Usage: python3 scripts/eval/diagnostics/verify_l2_evidence.py [--evidence-dir DIR]
"""

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

DEFAULT_EVIDENCE = (
    Path.home() / ".local/state/saccade/perf/l2-source-attribution-20260905"
)

# Display mode the framebuffer size is derived from; see environment.json in the pack.
FRAME_SECTORS = 2560 * 1600 * 4 // 32
REFRESH_HZ = 165.0

NODE = [
    "lts__t_sectors_realtime.sum",
    "lts__t_sectors_srcnode_gpc_realtime.sum",
    "lts__t_sectors_srcnode_hub_realtime.sum",
    "lts__t_sectors_srcnode_fbp_realtime.sum",
]
UNIT = [
    "lts__t_sectors_realtime.sum",
    "lts__t_sectors_srcunit_tex_realtime.sum",
    "lts__t_sectors_srcunit_gcc_realtime.sum",
    "lts__t_sectors_srcunit_ce_realtime.sum",
]
GPC_STD = [
    "lts__t_sectors_srcnode_gpc.sum",
    "lts__t_sectors_srcnode_gpc_lookup_hit.sum",
    "lts__t_sectors_srcnode_gpc_lookup_miss.sum",
]
GPC_RT = [
    "lts__t_sectors_srcnode_gpc_realtime.sum",
    "lts__t_sectors_srcnode_gpc_lookup_hit_realtime.sum",
    "lts__t_sectors_srcnode_gpc_lookup_miss_realtime.sum",
]
PIPELINE = GPC_RT + [
    "lts__t_sectors_srcnode_hub_realtime.sum",
    "gpu__time_duration.sum",
]
DECOMP = [
    "lts__t_sectors_srcnode_gpc_realtime.sum",
    "lts__t_sectors_srcunit_tex_realtime.sum",
    "lts__t_sectors_srcunit_gcc_realtime.sum",
    "lts__t_sectors_srcunit_tex_lookup_hit_realtime.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss_realtime.sum",
]
SWEEP = NODE + ["gpu__time_duration.sum"]
SPINS = (250000, 1000000, 4000000, 16000000, 64000000)

SINGLE_PASS_JOBS = [
    "e1-idle-node",
    "e1-idle-unit",
    "e1-nms-node",
    "e1-nms-unit",
    "e2-idle-dur",
    "e2-nms-dur",
    "e5-nms-gpc-rt",
    "e6-pipeline-nms",
    "e6-pipeline-occ",
    "e7-nms-probe",
    "e7-pipeline-nms",
    "e7-pipeline-occ",
]


def passes_of(root, name):
    """Replay pass counts reported in the ncu log, one entry per distinct value."""
    log = (root / f"{name}.log").read_text()
    return sorted({int(m) for m in re.findall(r"- (\d+) pass(?:es)?\b", log)})


def check_collection(root, name, metrics, passes, expect_passes=None):
    """Collector invariant: every requested metric must be present, and the pass
    count must be what the caller requires. Raises SystemExit on violation."""
    csv_path = root / f"{name}.csv"
    if not csv_path.exists():
        raise SystemExit(f"{name}: no CSV produced; collection failed")
    with csv_path.open() as f:
        header = set(next(csv.reader(f)))
    missing = [m for m in metrics if m not in header]
    if missing:
        raise SystemExit(
            f"{name}: ncu silently dropped {len(missing)} requested metric(s), "
            f"check spelling: {missing}"
        )
    if not passes:
        raise SystemExit(f"{name}: no pass count in log; cannot certify single-pass")
    if expect_passes is not None and passes != [expect_passes]:
        raise SystemExit(f"{name}: expected {expect_passes} pass(es), got {passes}")


def load(root, name, metrics):
    """Per-launch rows. Fails loudly rather than returning a partial record."""
    with (root / f"{name}.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("ID", "").strip().isdigit()]
    out = []
    for r in rows:
        rec = {}
        for m in metrics:
            if m not in r:
                raise SystemExit(
                    f"{name}: metric column missing (silently dropped?): {m}"
                )
            rec[m] = float(r[m].replace(",", ""))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    args = ap.parse_args()
    root = args.evidence_dir
    if not root.is_dir():
        print(
            f"evidence pack not found: {root}\n"
            "This is local retained evidence and is deliberately not in the repository; "
            "see docs/reference/benchmarks/frame_budget_20260905.md for what it contains.",
            file=sys.stderr,
        )
        return 2

    results, out = [], {}

    def check(name, ok, detail):
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + f"{name}: {detail}")

    # 1. every job whose counters get subtracted must have collected in one pass
    for n in SINGLE_PASS_JOBS:
        p = passes_of(root, n)
        check(f"single-pass {n}", p == [1], f"passes={p}")
    p = passes_of(root, "e5-nms-gpc-std")
    check("e5-nms-gpc-std is multi-pass (control)", p == [4], f"passes={p}")

    # 2. srcnode partitions the L2 total exactly; srcunit does not
    bad = tot = 0
    for n in ("e1-idle-node", "e1-nms-node"):
        for r in load(root, n, NODE):
            t, g, h, f = (r[m] for m in NODE)
            tot += 1
            bad += abs(t - (g + h + f)) > 1e-9
    check(
        "srcnode partition total=gpc+hub+fbp", bad == 0, f"{tot - bad}/{tot} rows exact"
    )

    bad = tot = 0
    for n in ("e1-idle-unit", "e1-nms-unit"):
        for r in load(root, n, UNIT):
            t, x, gc, ce = (r[m] for m in UNIT)
            tot += 1
            bad += abs(t - (x + gc + ce)) > 1e-9
    check(
        "srcunit does not partition the total",
        bad == tot,
        f"{bad}/{tot} rows leave a residual",
    )

    # 3. hub traffic is one framebuffer per display refresh, independent of kernel work
    pts = []
    for spin in SPINS:
        rows = load(root, f"e3-spin{spin}", SWEEP)
        pts.append(
            (
                st.median(r["gpu__time_duration.sum"] / 1e3 for r in rows),
                st.mean(r["lts__t_sectors_srcnode_hub_realtime.sum"] for r in rows),
            )
        )
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    slope = (n * sum(a * b for a, b in pts) - sx * sy) / (
        n * sum(p[0] ** 2 for p in pts) - sx * sx
    )
    pred = FRAME_SECTORS / (1e6 / REFRESH_HZ)
    out["hub_slope_sectors_per_us"] = slope
    out["hub_slope_predicted"] = pred
    check(
        "hub rate = 1 framebuffer per refresh",
        abs(slope / pred - 1) < 0.05,
        f"measured {slope:.2f} vs predicted {pred:.2f} sectors/us, ratio {slope / pred:.3f}",
    )

    gpcs = [
        st.median(
            r["lts__t_sectors_srcnode_gpc_realtime.sum"]
            for r in load(root, f"e3-spin{s}", SWEEP)
        )
        for s in SPINS
    ]
    out["gpc_across_250x_window_sweep"] = gpcs
    check(
        "gpc invariant across 250x window sweep",
        max(gpcs) / min(gpcs) < 1.4,
        f"median gpc {gpcs} sectors (window 168us..42.3ms)",
    )

    # 4. excluding hub is necessary but not sufficient: multi-pass is broken on its own
    for label, name, metrics in (
        ("realtime 1-pass", "e5-nms-gpc-rt", GPC_RT),
        ("standard 4-pass", "e5-nms-gpc-std", GPC_STD),
    ):
        rows = load(root, name, metrics)
        ok = sum(
            abs(r[metrics[0]] - (r[metrics[1]] + r[metrics[2]])) < 1e-9
            and 0 <= r[metrics[1]] <= r[metrics[0]]
            for r in rows
        )
        rates = [100 * r[metrics[1]] / r[metrics[0]] for r in rows]
        out[f"nms_probe_gpc_{label}"] = {
            "conserve": f"{ok}/{len(rows)}",
            "hit_pct": [min(rates), max(rates)],
        }
        check(
            f"gpc {label} conservation",
            (ok == len(rows)) == label.startswith("realtime"),
            f"{ok}/{len(rows)} rows, hit% {min(rates):.2f}-{max(rates):.2f}",
        )

    # 5. srcnode_gpc is not the data path: instruction/constant fetch is 4-63% of it
    for name in ("e7-nms-probe", "e7-pipeline-nms", "e7-pipeline-occ"):
        rows = load(root, name, DECOMP)
        gpc = st.median(r[DECOMP[0]] for r in rows)
        tex = st.median(r[DECOMP[1]] for r in rows)
        gcc = st.median(r[DECOMP[2]] for r in rows)
        resid = [r[DECOMP[0]] - r[DECOMP[1]] - r[DECOMP[2]] for r in rows]
        hit = [
            100 * r[DECOMP[3]] / r[DECOMP[1]] if r[DECOMP[1]] else float("nan")
            for r in rows
        ]
        out[name] = {
            "gpc": gpc,
            "tex": tex,
            "gcc": gcc,
            "gcc_share_pct": 100 * gcc / gpc,
            "gpc_minus_tex_gcc": [min(resid), max(resid)],
            "tex_hit_pct": [min(hit), max(hit)],
            "tex_zero_miss_rows": f"{sum(1 for r in rows if r[DECOMP[4]] == 0)}/{len(rows)}",
        }
        check(
            f"{name} gpc != tex+gcc",
            min(resid) > 0,
            f"gpc {gpc:.0f}, tex {tex:.0f}, gcc {gcc:.0f} ({100 * gcc / gpc:.1f}% of gpc), "
            f"residual {min(resid):.0f}-{max(resid):.0f}",
        )

    # 6. pipeline rows under the gpc-scoped single-pass method (reported, not resolved)
    for name in ("e6-pipeline-nms", "e6-pipeline-occ"):
        rows = load(root, name, PIPELINE)
        ok = sum(
            abs(r[PIPELINE[0]] - (r[PIPELINE[1]] + r[PIPELINE[2]])) < 1e-9 for r in rows
        )
        rates = [100 * r[PIPELINE[1]] / r[PIPELINE[0]] for r in rows]
        tots = [r[PIPELINE[0]] for r in rows]
        out[name] = {
            "conserve": f"{ok}/{len(rows)}",
            "gpc_sectors": [min(tots), max(tots)],
            "gpc_hit_pct": [min(rates), max(rates)],
            "median": st.median(rates),
        }
        check(
            f"{name} gpc conservation",
            ok == len(rows),
            f"{ok}/{len(rows)}, gpc {min(tots):.0f}-{max(tots):.0f} sectors, "
            f"hit% {min(rates):.2f}-{max(rates):.2f}",
        )

    # 7. the collector invariant catches the real silent failure that started this:
    #    lts__t_sectors_srcunit_hubother_realtime does not exist -- it is a syslts__
    #    metric -- and ncu accepted it without error, omitting the column.
    bad_set = UNIT + ["lts__t_sectors_srcunit_hubother_realtime.sum"]
    try:
        check_collection(root, "e1-idle-srcunit", bad_set, [1])
        check(
            "invariant rejects silently dropped metric",
            False,
            "accepted an uncollected metric",
        )
    except SystemExit as e:
        check(
            "invariant rejects silently dropped metric",
            "silently dropped" in str(e),
            str(e).split(":", 1)[1].strip()[:110],
        )
    try:
        check_collection(root, "e5-nms-gpc-std", GPC_STD, [4], expect_passes=1)
        check(
            "invariant rejects multi-pass when single-pass required",
            False,
            "accepted 4 passes",
        )
    except SystemExit as e:
        check(
            "invariant rejects multi-pass when single-pass required",
            "expected 1 pass" in str(e),
            str(e).split(":", 1)[1].strip(),
        )

    (root / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    nfail = sum(1 for ok in results if not ok)
    print(
        f"\n{len(results) - nfail}/{len(results)} checks passed; summary.json refreshed"
    )
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
