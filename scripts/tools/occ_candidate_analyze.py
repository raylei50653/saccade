#!/usr/bin/env python3
# mypy: ignore-errors
"""Phase B: threshold discriminability + GT-movement from the real-run candidate dump.

Joins the per-candidate ACTUAL gate values dumped during one unprocessed run
(`SACCADE_OCC_DUMP` → seq,frame,track_id,partner_id,peak_iou,gap_h,foot_y_t,foot_y_p,h_ref)
with the GT id-switch events of that SAME run (motmetrics), so every (occ_iou_thresh,
occ_foot_gap) is evaluated offline with the *right* label — "does this candidate sit on a real
GT id-switch" (会动 GT) — instead of the front-call label that misled the GT-synthetic Tier-1.

Gate semantics replicated from compute_track_occlusion_kernel (tracker_gpu.cu:346-357): track t
is flagged FRONT iff peak_iou >= occ_iou_thresh AND 0 < gap_h <= occ_foot_gap (lower foot, same
depth).  So the "gated" set = candidates meeting that condition.

Outputs: per-candidate labelled CSV, per-parameter AUC, a (iou × foot) sweep of
coverage/precision/addressable-switches, and the interval-intersection band.

NOTE (boundary): this is the FIRST-ORDER effect — which candidates get gated and whether they
sit on a real switch.  It does not model downstream trajectory divergence, so the predicted-best
band still needs ONE confirmation eval.

Usage
-----
  .venv/bin/python scripts/tools/occ_candidate_analyze.py \
      --dump results/occ_tune/occ_candidates_cur.csv \
      --hyp-dir results/occ_nointerp_cur
"""
# status: stable

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import motmetrics as mm

if not hasattr(np, "asfarray"):  # NumPy 2.0 removed it; motmetrics still calls it
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

SEQS = [
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
]


def gt_switches(gt_path: Path, hyp_path: Path) -> list[tuple[int, int, int]]:
    """Return real id-switch events as (frame, old_hyp_id, new_hyp_id) via motmetrics."""
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt, hyp, "iou", distth=0.5)
    events = acc.events
    match_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (f, _), row in events.iterrows():
        if row["Type"] == "MATCH" and not (
            isinstance(row["OId"], float) and np.isnan(row["OId"])
        ):
            match_hist[int(row["OId"])].append((int(f), int(row["HId"])))
    sw = []
    for (f, _), row in events[events["Type"] == "SWITCH"].iterrows():
        g, h_new, f = int(row["OId"]), int(row["HId"]), int(f)
        prev = [(mf, mh) for mf, mh in match_hist.get(g, []) if mf < f]
        h_old = prev[-1][1] if prev else -1
        sw.append((f, h_old, h_new))
    return sw


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dump", type=Path, default=Path("results/occ_tune/occ_candidates_cur.csv")
    )
    ap.add_argument("--hyp-dir", type=Path, default=Path("results/occ_nointerp_cur"))
    ap.add_argument("--gt-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--det", default="SDP")
    ap.add_argument(
        "--window",
        type=int,
        default=4,
        help="frame window to match a candidate to a switch",
    )
    ap.add_argument("--iou-grid", default="0.40,0.45,0.50")
    ap.add_argument("--foot-grid", default="0.10,0.125,0.15,0.20,0.25")
    ap.add_argument("--default-iou", type=float, default=0.45)
    ap.add_argument("--default-foot", type=float, default=0.15)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/occ_tune/occ_candidates_labelled.csv"),
    )
    args = ap.parse_args()

    # candidate rows per seq, aggregated into contiguous-overlap EPISODES (the kernel latches
    # once per crossing, not once per frame): group by (seq, track_id, partner_id) over
    # consecutive frames (gap <= 1) and keep the peak-IoU frame as the event.
    raw_by_seq: dict[str, list[dict]] = defaultdict(list)
    with open(args.dump) as f:
        for r in csv.DictReader(f):
            raw_by_seq[r["seq"]].append(r)

    cand_by_seq: dict[str, list[dict]] = defaultdict(list)
    for seq, rows in raw_by_seq.items():
        rows.sort(
            key=lambda r: (int(r["track_id"]), int(r["partner_id"]), int(r["frame"]))
        )
        cur = None  # (track_id, partner_id, last_frame, best_row)
        for r in rows:
            ti, pi, fr = int(r["track_id"]), int(r["partner_id"]), int(r["frame"])
            if cur and cur[0] == ti and cur[1] == pi and fr - cur[2] <= 1:
                cur = (
                    ti,
                    pi,
                    fr,
                    r
                    if abs(float(r["peak_iou"])) > abs(float(cur[3]["peak_iou"]))
                    else cur[3],
                )
            else:
                if cur:
                    cand_by_seq[seq].append(cur[3])
                cur = (ti, pi, fr, r)
        if cur:
            cand_by_seq[seq].append(cur[3])

    # switch signatures per seq: (frame, old, new). Key by the candidate-dump seq string
    # ("MOT17-02-SDP"), which already carries the -SDP suffix.
    switch_keys = {}  # seq_key -> list of (frame, old, new)
    for seq in SEQS:
        key = f"{seq}-{args.det}"
        hyp = args.hyp_dir / f"{key}.txt"
        gt = args.gt_root / key / "gt" / "gt.txt"
        if not hyp.exists() or not gt.exists():
            continue
        switch_keys[key] = gt_switches(gt, hyp)

    # label each FRONT candidate (gap_h > 0) with whether it sits on a real switch, and which
    labelled = []
    for seq, rows in cand_by_seq.items():
        sws = switch_keys.get(seq, [])
        sw_by_frame = defaultdict(list)
        for f, o, nw in sws:
            for df in range(-args.window, args.window + 1):
                sw_by_frame[f + df].append((f, o, nw))
        for r in rows:
            gap = float(r["gap_h"])
            if gap <= 0:  # only the FRONT track (lower foot) is gate-eligible
                continue
            fr, ti, pi = int(r["frame"]), int(r["track_id"]), int(r["partner_id"])
            hit, both = None, False
            for sf, o, nw in sw_by_frame.get(fr, []):
                pair = {ti, pi}
                inv = pair & {o, nw}
                if inv:
                    hit = (sf, o, nw)
                    both = both or ({o, nw} <= pair)
                    if both:
                        break
            labelled.append(
                dict(
                    seq=seq,
                    frame=fr,
                    track_id=ti,
                    partner_id=pi,
                    peak_iou=float(r["peak_iou"]),
                    gap_h=gap,
                    is_gt_switch=int(hit is not None),
                    switch_pair=int(both),
                    switch_key=(f"{hit[0]}:{hit[1]}>{hit[2]}" if hit else ""),
                )
            )

    if not labelled:
        raise SystemExit("no front candidates labelled — check dump / hyp dir")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(labelled[0]))
        w.writeheader()
        w.writerows(labelled)

    arr_iou = np.array([r["peak_iou"] for r in labelled])
    arr_gap = np.array([r["gap_h"] for r in labelled])
    y = np.array([r["is_gt_switch"] for r in labelled])
    total_sw = sum(len(v) for v in switch_keys.values())
    # distinct switches that have >=1 front candidate anywhere (the ceiling for coverage)
    addressable_all = {(r["seq"], r["switch_key"]) for r in labelled if r["switch_key"]}

    def auc(score, pos_high=True):
        p, n = score[y == 1], score[y == 0]
        if len(p) == 0 or len(n) == 0:
            return float("nan")
        allv = np.concatenate([p, n])
        order = allv.argsort()
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, ranks)
        ranks = (s / cnt)[inv]
        a = (ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
        return a if pos_high else 1 - a

    print(
        f"\nfront candidates: {len(labelled)}  | on real switch: {int(y.sum())}  "
        f"| total GT switches: {total_sw}  | switches w/ a front candidate: {len(addressable_all)}"
    )
    print("\n── per-parameter AUC (predictor of 'sits on a real GT switch') ──")
    print(
        f"  peak_iou : AUC {auc(arr_iou):.3f}  (higher IoU → more likely a real swap)"
    )
    print(
        f"  gap_h    : AUC {auc(arr_gap, pos_high=False):.3f}  "
        f"(>0.5 ⇒ SMALLER gap = more likely a real swap → validates same-height)"
    )

    def distinct_sw(mask):
        return {
            (labelled[i]["seq"], labelled[i]["switch_key"])
            for i in np.where(mask)[0]
            if labelled[i]["switch_key"]
        }

    hdr = f"{'iou≥':>6}{'gap≤':>7}{'gated':>7}{'prec%':>7}{'addr_sw':>9}{'cover%':>8}{'false':>7}"
    print("\n── (iou × foot_gap) sweep: 会动 GT 的覆蓋 vs 誤觸 ──")
    print("  prec% = gated that sit on a switch; addr_sw = distinct switches gated;")
    print(
        f"  cover% = addr_sw / {len(addressable_all)} (switches reachable by any front candidate)"
    )
    print(hdr)
    print("─" * len(hdr))
    for it in [float(x) for x in args.iou_grid.split(",")]:
        for fg in [float(x) for x in args.foot_grid.split(",")]:
            m = (arr_iou >= it) & (arr_gap <= fg)
            ng = int(m.sum())
            if ng == 0:
                continue
            prec = 100 * y[m].mean()
            addr = distinct_sw(m)
            cover = 100 * len(addr) / max(len(addressable_all), 1)
            false_fire = int(ng - y[m].sum())
            mark = (
                " <-DEF"
                if (
                    abs(it - args.default_iou) < 1e-9
                    and abs(fg - args.default_foot) < 1e-9
                )
                else ""
            )
            print(
                f"{it:>6.2f}{fg:>7.3f}{ng:>7}{prec:>6.0f} {len(addr):>8} {cover:>7.0f} {false_fire:>6}{mark}"
            )
    print("─" * len(hdr))
    print(f"\nlabelled per-candidate CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
