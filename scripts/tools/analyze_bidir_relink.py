#!/usr/bin/env python3
"""Per-candidate analysis of the bidirectional bridge-relink dumps.

Consumes the 14-column ``{seq}_raw_data.npy`` emitted by the GPU tracker
(``relink_bidir_propose_kernel`` in src/tracking/tracker_gpu.cu) together with
the MOT output and ground truth, and answers four questions:

  (1) Which candidate IDs the bridge never linked, split into genuine
      *missed relinks* (a same-GT lost track existed but was gated out) vs
      *correct rejects*.
  (2) Forward (lost->cand) vs backward (cand->lost) minimum Euclidean and
      Mahalanobis residual distributions, split by GT correctness.
  (3) Per-attempt GT-correctness confusion (accept/reject x same/diff GT) by gap-bin.
  (4) Long-term vs short-term ID behaviour, bucketed by *track lifespan*
      (frames the ID lives in the MOT output).  NOTE: the logged hit_streak is
      degenerate at bridge time (candidate fires exactly when hit_streak==bridge_at,
      a lost track's streak is reset to 0), so lifespan is the usable signal.

Raw-data columns (see RAW_COLS in tracker_gpu.cu):
  0 gap  1 bridge(midpoint Euclid)  2 fwd_maha(=kalman_d2, -1 if none)  3 dir_cos
  4 speed  5 outcome(0=accept,1=bridge_px,2=kalman,3=dir,4=speed)  6 source(0=live,1=arch)
  7 lost_id  8 cand_id  9 cand_hit_streak  10 lost_hit_streak
  11 fwd_eucl  12 bwd_eucl  13 bwd_maha(-1 if none)
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── reused loaders/mapper from analyze_missed_relinks.py ──────────────────────


def load_mot_tracks(path: Path):
    tracks = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            tracks[tid].append((frm, x + w / 2, y + h / 2, h))
    return dict(tracks)


def load_gt_tracks(path: Path):
    tracks = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, gid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            if gid <= 0:
                continue
            tracks[gid].append((frm, x + w / 2, y + h / 2, h))
    return dict(tracks)


def map_track_to_gt(traj, gt_tracks):
    """Map a predicted track to the most-overlapping GT id by center proximity."""
    best_gt, best_overlap = -1, 0
    pred_frames = {f for f, _, _, _ in traj}
    pred_at = {f: (cx, cy, h) for f, cx, cy, h in traj}
    for gid, gtraj in gt_tracks.items():
        common = pred_frames & {f for f, _, _, _ in gtraj}
        if not common:
            continue
        gt_at = {f: (cx, cy, h) for f, cx, cy, h in gtraj}
        overlap = 0
        for f in common:
            pcx, pcy, ph = pred_at[f]
            gcx, gcy, gh = gt_at[f]
            if np.hypot(pcx - gcx, pcy - gcy) < max(ph, gh) * 2:
                overlap += 1
        if overlap > best_overlap:
            best_overlap, best_gt = overlap, gid
    return best_gt if best_overlap >= max(3, len(traj) * 0.3) else -1


# ── small stats helpers ───────────────────────────────────────────────────────

GAP_BINS = [(2, 5), (6, 15), (16, 30), (31, 45), (46, 120)]
GAP_LABELS = ["2-5", "6-15", "16-30", "31-45", "46-120"]
OUTCOME_NAME = {
    0: "accept",
    1: "rej_bridge",
    2: "rej_kalman",
    3: "rej_dir",
    4: "rej_speed",
}


def gap_bin(g):
    for i, (lo, hi) in enumerate(GAP_BINS):
        if lo <= g <= hi:
            return i
    return len(GAP_BINS) - 1


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1)
    )
    return (a.mean() - b.mean()) / sp if sp > 1e-12 else float("nan")


def quant(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "n=0"
    q = np.percentile(x, [25, 50, 75, 90])
    return f"n={len(x):4d} mean={x.mean():7.2f} p25={q[0]:6.2f} p50={q[1]:6.2f} p75={q[2]:6.2f} p90={q[3]:6.2f}"


# ── per-sequence analysis ─────────────────────────────────────────────────────

SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]
DATA_ROOT = Path("datasets/MOT17/train")
MOT_ROOT = Path("results/MOT17_eval")
FIG_DIR = Path("docs/modules/semantic/research/figures")  # canonical home for charts


def label_rows(raw, track_to_gt):
    """Return per-row correctness: 1=same_gt, 0=diff_gt, -1=unmapped."""
    lab = np.full(len(raw), -1, dtype=int)
    for i in range(len(raw)):
        lg = track_to_gt.get(int(raw[i, 7]), -1)
        cg = track_to_gt.get(int(raw[i, 8]), -1)
        if lg >= 0 and cg >= 0:
            lab[i] = 1 if lg == cg else 0
    return lab


def analyze_seq(seq, out):
    raw_path = DATA_ROOT / f"{seq}_raw_data.npy"
    mot_path = MOT_ROOT / f"{seq}.txt"
    gt_path = DATA_ROOT / seq / "gt/gt.txt"
    if not raw_path.exists() or not mot_path.exists():
        out.append(f"[skip] {seq}: missing dumps")
        return None
    raw = np.load(raw_path)
    if raw.shape[1] != 14:
        out.append(
            f"[skip] {seq}: raw has {raw.shape[1]} cols, expected 14 (re-run eval)"
        )
        return None
    mot = load_mot_tracks(mot_path)
    gt = load_gt_tracks(gt_path)
    track_to_gt = {
        tid: g for tid, traj in mot.items() if (g := map_track_to_gt(traj, gt)) >= 0
    }
    track_len = {tid: len(traj) for tid, traj in mot.items()}
    lab = label_rows(raw, track_to_gt)

    out.append(
        f"\n{'=' * 78}\n{seq}: {len(raw)} attempts, {len(mot)} pred tracks, "
        f"{len(track_to_gt)} mapped to GT"
    )

    # ── (1) Unhandled candidate IDs ──────────────────────────────────────────
    by_cand = defaultdict(list)
    for i in range(len(raw)):
        by_cand[int(raw[i, 8])].append(i)
    missed, correct_rej, unmapped_un = [], 0, 0
    for cid, idxs in by_cand.items():
        rows = raw[idxs]
        handled = bool((rows[:, 5] == 0).any())  # had a viable accepted proposal
        if handled:
            continue
        cg = track_to_gt.get(cid, -1)
        # same-GT lost target among this candidate's rejected attempts?
        same_hit = [i for i in idxs if lab[i] == 1]
        if same_hit and cg >= 0:
            j = min(same_hit, key=lambda i: raw[i, 11] + raw[i, 12])  # smallest fwd+bwd
            r = raw[j]
            missed.append(
                (
                    cid,
                    int(r[7]),
                    int(r[0]),
                    r[11],
                    r[12],
                    r[2],
                    r[13],
                    int(r[5]),
                    track_len.get(cid, 0),
                )
            )
        elif cg >= 0:
            correct_rej += 1
        else:
            unmapped_un += 1
    out.append(
        f"\n(1) UNHANDLED candidates (no accepted proposal): "
        f"{len(missed)} missed-relink, {correct_rej} correct-reject, "
        f"{unmapped_un} unmapped"
    )
    if missed:
        out.append(
            "    cand  ->lost  gap  fwd_eucl bwd_eucl  fwd_maha  bwd_maha  reason       cand_len"
        )
        for cid, lid, g, fe, be, fm, bm, oc, clen in sorted(missed, key=lambda x: x[2]):
            out.append(
                f"    {cid:5d} {lid:6d} {g:4d}  {fe:8.2f} {be:8.2f} {fm:9.1f} {bm:9.1f}  "
                f"{OUTCOME_NAME.get(oc, oc):11s} {clen:5d}"
            )

    # ── (2) fwd/bwd residual distributions by GT correctness ─────────────────
    out.append("\n(2) Forward vs backward residual distributions (per attempt):")
    same = raw[lab == 1]
    diff = raw[lab == 0]
    for name, col, drop_neg in [
        ("fwd_eucl", 11, False),
        ("bwd_eucl", 12, False),
        ("fwd_maha", 2, True),
        ("bwd_maha", 13, True),
    ]:
        s = same[:, col]
        d = diff[:, col]
        if drop_neg:
            s, d = s[s >= 0], d[d >= 0]
        out.append(f"    {name:9s} same-GT: {quant(s)}")
        out.append(f"    {name:9s} diff-GT: {quant(d)}   Cohen_d={cohens_d(d, s):+.2f}")

    # ── (3) correctness confusion by gap-bin ─────────────────────────────────
    out.append("\n(3) Accept/Reject x correctness by gap-bin:")
    out.append("    gap-bin   acc_same acc_diff acc_unm | rej_same rej_diff rej_unm")
    for b, glab in enumerate(GAP_LABELS):
        m = np.array([gap_bin(int(g)) == b for g in raw[:, 0]])
        acc = m & (raw[:, 5] == 0)
        rej = m & (raw[:, 5] != 0)
        c = lambda sel, v: int(((lab == v) & sel).sum())  # noqa: E731
        out.append(
            f"    {glab:8s}  {c(acc, 1):8d} {c(acc, 0):8d} {c(acc, -1):7d} | "
            f"{c(rej, 1):8d} {c(rej, 0):8d} {c(rej, -1):7d}"
        )

    # ── (4) long/short ID behaviour by track lifespan ────────────────────────
    out.append("\n(4) Behaviour by candidate track lifespan (frames in MOT output):")
    out.append("    bucket        n_att  accept%  same-GT%  fwd_eucl bwd_eucl")
    buckets = [("short<20", 0, 20), ("med20-100", 20, 100), ("long>=100", 100, 10**9)]
    for bname, lo, hi in buckets:
        sel = np.array([lo <= track_len.get(int(c), 0) < hi for c in raw[:, 8]])
        rs = raw[sel]
        if len(rs) == 0:
            out.append(f"    {bname:12s}  {0:5d}")
            continue
        acc_pct = 100.0 * (rs[:, 5] == 0).mean()
        ls = lab[sel]
        mapped = ls >= 0
        same_pct = 100.0 * (ls[mapped] == 1).mean() if mapped.any() else float("nan")
        out.append(
            f"    {bname:12s}  {len(rs):5d}  {acc_pct:6.1f}  {same_pct:7.1f}  "
            f"{rs[:, 11].mean():8.2f} {rs[:, 12].mean():8.2f}"
        )

    candlen = np.array([track_len.get(int(c), 0) for c in raw[:, 8]], dtype=int)
    return {
        "raw": raw,
        "lab": lab,
        "candlen": candlen,
        "missed": len(missed),
        "correct_rej": correct_rej,
    }


# ── plotting ──────────────────────────────────────────────────────────────────


def _roc(score_same, score_diff):
    """ROC treating same-GT as positive; LOWER score => more 'link'. Returns fpr,tpr,auc."""
    s = np.concatenate([score_same, score_diff])
    y = np.concatenate([np.ones(len(score_same)), np.zeros(len(score_diff))])
    order = np.argsort(s)  # ascending: smallest distance first = predicted link
    y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def make_plots(raw, lab, candlen, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    same, diff = raw[lab == 1], raw[lab == 0]

    # Fig 1: forward/backward residual distributions, same vs diff GT
    feats = [
        ("fwd_eucl", 11, False, 12),
        ("bwd_eucl", 12, False, 12),
        ("fwd_maha", 2, True, None),
        ("bwd_maha", 13, True, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, col, dn, clip) in zip(axes.ravel(), feats):
        s, d = same[:, col], diff[:, col]
        if dn:
            s, d = s[s >= 0], d[d >= 0]
        if clip is None:  # maha: log scale
            s, d = s[s > 0], d[d > 0]
            bins = np.logspace(0, np.log10(max(d.max(), s.max(), 10)), 40)
            ax.set_xscale("log")
        else:
            s, d = s[s <= clip], d[d <= clip]
            bins = np.linspace(0, clip, 40)
        ax.hist(
            s,
            bins=bins,
            density=True,
            alpha=0.6,
            label="same-GT (should link)",
            color="tab:green",
        )
        ax.hist(
            d,
            bins=bins,
            density=True,
            alpha=0.6,
            label="diff-GT (should reject)",
            color="tab:red",
        )
        ax.set_title(
            f"{name}  Cohen_d={cohens_d(diff[:, col][diff[:, col] >= 0] if dn else diff[:, col], same[:, col][same[:, col] >= 0] if dn else same[:, col]):+.2f}"
        )
        ax.set_xlabel(name)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Forward vs backward residual distributions (same-GT vs diff-GT)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_residual_dist.png", dpi=110)
    plt.close(fig)

    # Fig 2: fwd vs bwd Euclidean scatter, colored by correctness
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        diff[:, 11], diff[:, 12], s=6, alpha=0.3, color="tab:red", label="diff-GT"
    )
    ax.scatter(
        same[:, 11], same[:, 12], s=10, alpha=0.6, color="tab:green", label="same-GT"
    )
    ax.plot([0, 12], [0, 12], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_xlabel("fwd_eucl (lost extrapolated -> candidate)")
    ax.set_ylabel("bwd_eucl (candidate extrapolated -> lost)")
    ax.set_title("Forward vs backward Euclidean residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_fwd_vs_bwd_scatter.png", dpi=110)
    plt.close(fig)

    # Fig 3: ROC — which feature separates true from false links best?
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, col, dn in [
        ("fwd_eucl", 11, False),
        ("bwd_eucl", 12, False),
        ("bridge(midpoint)", 1, False),
        ("bwd_maha", 13, True),
    ]:
        s, d = same[:, col], diff[:, col]
        if dn:
            s, d = s[s >= 0], d[d >= 0]
        fpr, tpr, auc = _roc(s, d)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (diff-GT wrongly linked)")
    ax.set_ylabel("TPR (same-GT correctly linked)")
    ax.set_title("Gate separability: linking by each residual (lower = link)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_gate_roc.png", dpi=110)
    plt.close(fig)

    # Fig 4: behaviour by candidate lifespan bucket
    buckets = [
        ("short\n<20", 0, 20),
        ("med\n20-100", 20, 100),
        ("long\n>=100", 100, 10**9),
    ]
    names, same_pct, fwd_m, bwd_m = [], [], [], []
    for bn, lo, hi in buckets:
        sel = (candlen >= lo) & (candlen < hi)
        ls = lab[sel]
        mapped = ls >= 0
        names.append(bn)
        same_pct.append(100.0 * (ls[mapped] == 1).mean() if mapped.any() else 0)
        fwd_m.append(raw[sel, 11].mean() if sel.any() else 0)
        bwd_m.append(raw[sel, 12].mean() if sel.any() else 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.bar(names, same_pct, color="tab:blue")
    a1.set_ylabel("same-GT %% of mapped attempts")
    a1.set_title("Correctness rate by candidate lifespan")
    x = np.arange(len(names))
    a2.bar(x - 0.2, fwd_m, 0.4, label="fwd_eucl", color="tab:green")
    a2.bar(x + 0.2, bwd_m, 0.4, label="bwd_eucl", color="tab:orange")
    a2.set_xticks(x)
    a2.set_xticklabels(names)
    a2.set_ylabel("mean residual")
    a2.set_title("Residual magnitude by lifespan")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_lifespan.png", dpi=110)
    plt.close(fig)

    return [
        "fig1_residual_dist.png",
        "fig2_fwd_vs_bwd_scatter.png",
        "fig3_gate_roc.png",
        "fig4_lifespan.png",
    ]


def main():
    do_plot = "--plot" in sys.argv or "--no-plot" not in sys.argv
    out = []
    out.append("BIDIRECTIONAL BRIDGE-RELINK ANALYSIS  (mamba_whole_graph SDP)")
    agg_raw, agg_lab, agg_cl = [], [], []
    tot_missed = tot_rej = 0
    for seq in SEQS:
        r = analyze_seq(seq, out)
        if r:
            agg_raw.append(r["raw"])
            agg_lab.append(r["lab"])
            agg_cl.append(r["candlen"])
            tot_missed += r["missed"]
            tot_rej += r["correct_rej"]

    out_dir = Path("scripts/tools/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    if agg_raw:
        raw = np.vstack(agg_raw)
        lab = np.concatenate(agg_lab)
        candlen = np.concatenate(agg_cl)
        out.append(
            f"\n{'=' * 78}\nAGGREGATE (7 seqs): {len(raw)} attempts | "
            f"unhandled missed-relink={tot_missed} correct-reject={tot_rej}"
        )
        same, diff = raw[lab == 1], raw[lab == 0]
        out.append(
            "  feature   same-GT mean   diff-GT mean   Cohen_d (separation of false from true)"
        )
        for name, col, dn in [
            ("fwd_eucl", 11, False),
            ("bwd_eucl", 12, False),
            ("fwd_maha", 2, True),
            ("bwd_maha", 13, True),
        ]:
            s, d = same[:, col], diff[:, col]
            if dn:
                s, d = s[s >= 0], d[d >= 0]
            sm = s.mean() if len(s) else float("nan")
            dm = d.mean() if len(d) else float("nan")
            out.append(f"  {name:9s} {sm:13.2f} {dm:14.2f}   {cohens_d(d, s):+.2f}")

        if do_plot:
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            figs = make_plots(raw, lab, candlen, FIG_DIR)
            out.append("\n[plots] " + ", ".join(figs))

    report = "\n".join(out)
    print(report)
    (out_dir / "bidir_relink_report.txt").write_text(report + "\n")
    print(f"\n[saved] {out_dir / 'bidir_relink_report.txt'}")
    if agg_raw and do_plot:
        print(f"[saved] 4 figures -> {FIG_DIR}/fig*.png")


if __name__ == "__main__":
    sys.exit(main())
