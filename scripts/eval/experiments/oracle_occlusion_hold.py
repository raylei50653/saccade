#!/usr/bin/env python3
"""Phase-0 oracle ceiling for the `Occluded(by=A)` state policy.

Upper-bounds the end-to-end gain of an *ideal* occludee-hold policy on the real
tracker output, BEFORE any CUDA is written (mirrors the #38 "oracle before C++"
precedent). It does NOT change the tracker — it post-hoc relabels the substrate
hypothesis so that the identities involved in occlusion crossing-swaps stay
consistent, then re-scores with the canonical evaluator.

What it measures
----------------
A crossing-swap (offline_relink_candidate_analysis.md §8 / mamba-score-distribution §10)
is a short, occlusion-driven identity switch: two confirmed tracks mutually occlude
1-2 frames, then separate, and the ids swap. The `Occluded` state would hold the
occludee's identity through the gap and re-acquire the depth-consistent box.

Oracle = "what if every such swap were resolved perfectly". For each switch we keep
(short `match_gap` + prior low GT visibility = the §8 profile), we relabel the
swap-involved GT identity's matched detections into a disjoint per-GT id namespace
(collision-free, since motmetrics matches <=1 detection per GT per frame), then
re-score. FP/FN are unchanged, so the IDF1/AssA deltas are pure association gains.
Two variants:
  - crossing : only short + occlusion-driven switches  (the §8 ceiling — the real target)
  - all      : every switch                            (loose upper bound, for context)

Discount the crossing ceiling by the depth-probe accuracy (~90 %, 97 % decisive) to
get the realistic end-to-end expectation.

GO bar (to justify the CUDA work): meaningful IDs reduction with non-negative IDF1
AND per-sequence consistency (no sequence materially negative).

Usage
-----
  .venv/bin/python scripts/eval/experiments/oracle_occlusion_hold.py \
      --substrate results/mamba_whole_graph_current_7seq_recheck
"""
# status: experiment

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# NumPy 2.0 compat for motmetrics (same shim the repo uses elsewhere)
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402

from saccade.perception.eval.metrics import run_motmetrics_evaluation  # noqa: E402

SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]
VIS_THRESH = 0.3  # below = effectively occluded (matches diagnose_id_switches.py)
ID_OFFSET = 1_000_000  # disjoint namespace for oracle-relabeled identities


def gt_visibility(gt_path: Path) -> dict[int, dict[int, float]]:
    by: dict[int, dict[int, float]] = defaultdict(dict)
    for line in gt_path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        frame, gid, conf, cls = int(p[0]), int(p[1]), int(p[6]), int(p[7])
        if conf != 1 or cls != 1:
            continue
        by[gid][frame] = float(p[8])
    return dict(by)


def gt_frames(gt_path: Path) -> dict[int, list[int]]:
    by: dict[int, list[int]] = defaultdict(list)
    for line in gt_path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9:
            continue
        frame, gid, conf, cls = int(p[0]), int(p[1]), int(p[6]), int(p[7])
        if conf != 1 or cls != 1:
            continue
        by[gid].append(frame)
    return {k: sorted(v) for k, v in by.items()}


def swap_remap(
    gt_path: Path, hyp_path: Path, max_gap: int, mode: str
) -> dict[tuple[int, int], int]:
    """Return {(frame, hyp_id): new_hyp_id} relabel map for swap-involved GT ids.

    mode="crossing": keep only short (match_gap<=max_gap) + occlusion-driven (prior
    low GT visibility) switches. mode="all": every switch.
    """
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)
    events = acc.events

    # per-frame MATCH map: gt history + (frame,hyp)->gt
    match_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frame_match: dict[tuple[int, int], int] = {}
    for (frame_id, _), row in events.iterrows():
        if row["Type"] != "MATCH":
            continue
        oid, hid = row["OId"], row["HId"]
        if isinstance(oid, float) and np.isnan(oid):
            continue
        gid, hyp = int(oid), int(hid)
        match_hist[gid].append((int(frame_id), hyp))
        frame_match[(int(frame_id), hyp)] = gid

    vis_by = gt_visibility(gt_path)
    frames_by = gt_frames(gt_path)

    swap_gt: set[int] = set()
    n_events = 0  # qualifying SWITCH events = directly-addressable IDs
    for (frame_id, _), row in events[events["Type"] == "SWITCH"].iterrows():
        gid = int(row["OId"])
        f = int(frame_id)
        if mode == "all":
            swap_gt.add(gid)
            n_events += 1
            continue
        # match_gap: frames since last MATCH for this gt
        prev = [mf for mf, _ in match_hist.get(gid, []) if mf < f]
        match_gap = (f - prev[-1] - 1) if prev else f
        if match_gap > max_gap:
            continue
        # occlusion-driven: low GT visibility in the frames just before the switch
        vis = vis_by.get(gid, {})
        occluded_before = False
        for ff in sorted(frames_by.get(gid, []), reverse=True):
            if ff >= f:
                continue
            occluded_before = vis.get(ff, 1.0) < VIS_THRESH
            break
        if occluded_before:
            swap_gt.add(gid)
            n_events += 1

    # Give each swap-involved GT its OWN consistent identity in a disjoint id
    # namespace (OFFSET + gt_id). Collision-free: motmetrics matches <=1 detection
    # per GT per frame, so no two same-frame detections get the same new id, and
    # OFFSET keeps these clear of the original small hyp ids.
    remap: dict[tuple[int, int], int] = {}
    for (frame_id, hyp), gid in frame_match.items():
        if gid in swap_gt:
            new_id = ID_OFFSET + gid
            if hyp != new_id:
                remap[(frame_id, hyp)] = new_id
    return remap, n_events


def write_remapped(
    hyp_path: Path, remap: dict[tuple[int, int], int], out_path: Path
) -> None:
    out_lines = []
    for line in hyp_path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        frame, hyp = int(p[0]), int(p[1])
        new = remap.get((frame, hyp))
        if new is not None:
            p[1] = str(new)
        out_lines.append(",".join(p))
    out_path.write_text("\n".join(out_lines) + "\n")


def evaluate(output_dir: Path, seq: str | None) -> dict | None:
    return run_motmetrics_evaluation(
        data_root=str(PROJECT_ROOT / "datasets" / "MOT17"),
        split="train",
        output=str(output_dir),
        sequences=seq or "",
        detector="SDP",
    )


def _num(metrics: dict | None, key: str) -> float:
    if not metrics or key not in metrics:
        return float("nan")
    return float(str(metrics[key]).rstrip("%"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--substrate",
        type=Path,
        default=Path("results/mamba_whole_graph_current_7seq_recheck"),
    )
    ap.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="max match_gap (frames) for a crossing-swap",
    )
    ap.add_argument("--mode", choices=["crossing", "all", "both"], default="both")
    args = ap.parse_args()

    modes = ["crossing", "all"] if args.mode == "both" else [args.mode]
    gt_root = PROJECT_ROOT / "datasets" / "MOT17" / "train"

    # build remapped output dirs (one per mode) + count addressable switch events
    tmp = Path(tempfile.mkdtemp(prefix="occ_oracle_"))
    n_events: dict[str, dict[str, int]] = {m: {} for m in modes}  # mode -> seq -> count
    out_dirs: dict[str, Path] = {}
    for mode in modes:
        d = tmp / mode
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[mode] = d
        for seq in SEQS:
            hyp = args.substrate / f"{seq}.txt"
            gt = gt_root / seq / "gt" / "gt.txt"
            if not hyp.exists() or not gt.exists():
                print(f"  ! missing {seq} (hyp={hyp.exists()} gt={gt.exists()})")
                continue
            remap, n = swap_remap(gt, hyp, args.max_gap, mode)
            n_events[mode][seq] = n
            write_remapped(hyp, remap, d / f"{seq}.txt")

    # per-seq + overall ceiling table (identity metrics are authoritative; FP/FN are
    # unchanged by relabeling, so IDF1/AssA gains are pure association improvements).
    print(f"\nSubstrate: {args.substrate}")
    print(f"max_gap={args.max_gap}  (crossing = short + occlusion-driven switches)\n")

    hdr = f"{'scope':<14}{'base IDF1':>10}{'base AssA':>10}{'base HOTA':>10}   {'crossing ΔIDF1 / ΔAssA  (N)':>30}   {'all ΔIDF1 / ΔAssA':>20}"
    print(hdr)
    print("─" * len(hdr))
    for scope, seq in [("OVERALL", None)] + [
        (s.replace("MOT17-", "").replace("-SDP", ""), s) for s in SEQS
    ]:
        base = evaluate(args.substrate, seq)
        if base is None:
            continue
        cells = {}
        for mode in modes:
            m = evaluate(out_dirs[mode], seq)
            cells[mode] = (
                _num(m, "IDF1") - _num(base, "IDF1"),
                _num(m, "AssA") - _num(base, "AssA"),
            )
        nc = (
            n_events.get("crossing", {}).get(seq)
            if seq
            else sum(n_events.get("crossing", {}).values())
        )
        cx = cells.get("crossing", (float("nan"), float("nan")))
        al = cells.get("all", (float("nan"), float("nan")))
        print(
            f"{scope:<14}{_num(base, 'IDF1'):>10.1f}{_num(base, 'AssA'):>10.1f}{_num(base, 'HOTA'):>10.1f}   "
            f"{cx[0]:>+9.1f} /{cx[1]:>+7.1f}   (N={nc if nc is not None else '—'})".ljust(
                33 + 14
            )
            + f"{al[0]:>+12.1f} /{al[1]:>+7.1f}"
        )

    tot_c = sum(n_events.get("crossing", {}).values())
    tot_a = sum(n_events.get("all", {}).values())
    print(
        f"\nAddressable switch events: crossing={tot_c}  (vs all switches={tot_a}); "
        f"baseline motmetrics IDs={_num(evaluate(args.substrate, None), 'IDs'):.0f}"
    )
    print(
        "Note: 'crossing' ΔIDF1/ΔAssA is the perfect-fix ceiling for occlusion crossing-swaps; "
        "discount by depth-probe accuracy (~90%, 97% decisive) for the realistic expectation."
    )
    print(f"(temp remapped outputs: {tmp})")


if __name__ == "__main__":
    main()
