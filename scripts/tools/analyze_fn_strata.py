#!/usr/bin/env python3
"""False-negative stratification: is the recall gap resolution-limited (small boxes,
visible -> a high-res P2 head can recover) or occlusion-limited (low visibility ->
P2 won't help)?

For each MOT17-SDP seq, match predictions (results/MOT17_eval/{seq}.txt) to GT
pedestrians (class==1) per frame at IoU>=0.5; unmatched GT = false negative.  Stratify
GT and FN by box height x visibility and report recall per cell.

GT columns: frame,id,x,y,w,h,consider,class,visibility(0..1).
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
FIG_DIR = Path("docs/modules/semantic/research/figures")

H_BINS = [(0, 50), (50, 100), (100, 200), (200, 10**9)]
H_LABELS = ["<50", "50-100", "100-200", "200+"]
V_BINS = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.01)]
V_LABELS = ["occ<0.3", "part0.3-0.7", "vis>=0.7"]


def load_gt(path):
    """{frame: [(x,y,w,h,vis), ...]} for pedestrian class==1."""
    by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            cls = int(p[7])
            if cls != 1:
                continue
            frm = int(p[0])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            vis = float(p[8])
            by_frame[frm].append((x, y, w, h, vis))
    return by_frame


def load_pred(path):
    by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm = int(p[0])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            by_frame[frm].append((x, y, w, h))
    return by_frame


def iou_matrix(g, d):
    """g:(N,4) d:(M,4) xywh -> (N,M) IoU."""
    if len(g) == 0 or len(d) == 0:
        return np.zeros((len(g), len(d)))
    gx1, gy1 = g[:, 0], g[:, 1]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    dx1, dy1 = d[:, 0], d[:, 1]
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    ix1 = np.maximum(gx1[:, None], dx1[None, :])
    iy1 = np.maximum(gy1[:, None], dy1[None, :])
    ix2 = np.minimum(gx2[:, None], dx2[None, :])
    iy2 = np.minimum(gy2[:, None], dy2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = (g[:, 2] * g[:, 3])[:, None]
    da = (d[:, 2] * d[:, 3])[None, :]
    return inter / np.clip(ga + da - inter, 1e-9, None)


def h_bin(h):
    for i, (lo, hi) in enumerate(H_BINS):
        if lo <= h < hi:
            return i
    return len(H_BINS) - 1


def v_bin(v):
    for i, (lo, hi) in enumerate(V_BINS):
        if lo <= v < hi:
            return i
    return len(V_BINS) - 1


def match_frame(gt, pred, thr=0.5):
    """Greedy IoU match; return boolean array over gt: True=matched."""
    g = np.array([[x, y, w, h] for x, y, w, h, _ in gt], float)
    d = np.array([[x, y, w, h] for x, y, w, h in pred], float)
    matched = np.zeros(len(g), bool)
    if len(g) == 0 or len(d) == 0:
        return matched
    iou = iou_matrix(g, d)
    used_d = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, di in order:
        if iou[gi, di] < thr:
            break
        if matched[gi] or di in used_d:
            continue
        matched[gi] = True
        used_d.add(int(di))
    return matched


def main():
    out = ["FALSE-NEGATIVE STRATIFICATION  (mamba_whole_graph SDP, IoU>=0.5)\n"]
    # global accumulators: gt_total[h][v], fn[h][v]
    gt_tot = np.zeros((len(H_BINS), len(V_BINS)), int)
    fn_tot = np.zeros((len(H_BINS), len(V_BINS)), int)

    for seq in SEQS:
        gt = load_gt(DATA_ROOT / seq / "gt/gt.txt")
        pred = load_pred(MOT_ROOT / f"{seq}.txt")
        if not gt:
            out.append(f"[skip] {seq}: no GT")
            continue
        for frm, gboxes in gt.items():
            matched = match_frame(gboxes, pred.get(frm, []))
            for (x, y, w, h, vis), m in zip(gboxes, matched):
                hb, vb = h_bin(h), v_bin(vis)
                gt_tot[hb, vb] += 1
                if not m:
                    fn_tot[hb, vb] += 1

    tot, fn = gt_tot.sum(), fn_tot.sum()
    out.append(f"Overall: GT={tot}  FN={fn}  Recall={100 * (1 - fn / tot):.1f}%\n")

    # recall table height x visibility
    out.append("Recall %% by box-height (rows) x visibility (cols):")
    out.append("  height    " + "".join(f"{vl:>13s}" for vl in V_LABELS) + "      row")
    for hi, hl in enumerate(H_LABELS):
        cells = []
        for vi in range(len(V_BINS)):
            g, f = gt_tot[hi, vi], fn_tot[hi, vi]
            cells.append(f"{(100 * (1 - f / g)) if g else 0:5.0f}% ({g:5d})")
        rg, rf = gt_tot[hi].sum(), fn_tot[hi].sum()
        out.append(
            f"  {hl:8s} "
            + "".join(f"{c:>13s}" for c in cells)
            + f"   {100 * (1 - rf / rg) if rg else 0:4.0f}%"
        )
    out.append(
        "  col-recall "
        + "".join(
            f"{100 * (1 - fn_tot[:, vi].sum() / max(gt_tot[:, vi].sum(), 1)):11.0f}% "
            for vi in range(len(V_BINS))
        )
    )

    # FN composition + the verdict
    out.append("\nFN composition (where the misses are):")
    for hi, hl in enumerate(H_LABELS):
        out.append(
            f"  height {hl:8s}: {fn_tot[hi].sum():6d} FN "
            f"({100 * fn_tot[hi].sum() / max(fn, 1):4.1f}% of all FN)"
        )
    small = fn_tot[:2]  # <100px
    small_vis = fn_tot[:2, 2].sum()  # <100px AND visible>=0.7
    small_all = small.sum()
    out.append(
        f"\nVERDICT — resolution-recoverable headroom:\n"
        f"  small boxes (<100px) hold {100 * small_all / max(fn, 1):.1f}% of all FN\n"
        f"  of those, {100 * small_vis / max(small_all, 1):.1f}% are HIGH-visibility (>=0.7)\n"
        f"  => {small_vis} FN are small+visible (P2 should recover); "
        f"{small.sum() - small_vis} are small+occluded (P2 limited)."
    )

    report = "\n".join(out)
    print(report)
    out_dir = Path("scripts/tools/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fn_strata_report.txt").write_text(report + "\n")

    # figure: recall heatmap + FN-by-height bar
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recall = np.where(
            gt_tot > 0, 100 * (1 - fn_tot / np.maximum(gt_tot, 1)), np.nan
        )
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        im = a1.imshow(recall, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        a1.set_xticks(range(len(V_LABELS)))
        a1.set_xticklabels(V_LABELS)
        a1.set_yticks(range(len(H_LABELS)))
        a1.set_yticklabels(H_LABELS)
        a1.set_xlabel("visibility")
        a1.set_ylabel("box height (px)")
        a1.set_title("Recall %% by height x visibility")
        for hi in range(len(H_LABELS)):
            for vi in range(len(V_LABELS)):
                if gt_tot[hi, vi]:
                    a1.text(
                        vi,
                        hi,
                        f"{recall[hi, vi]:.0f}\n({gt_tot[hi, vi]})",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        fig.colorbar(im, ax=a1, fraction=0.046)
        # FN by height, split visible vs occluded
        vis_fn = fn_tot[:, 2]
        occ_fn = fn_tot[:, :2].sum(axis=1)
        x = np.arange(len(H_LABELS))
        a2.bar(x, vis_fn, label="visible (>=0.7) — P2 recoverable", color="tab:green")
        a2.bar(
            x,
            occ_fn,
            bottom=vis_fn,
            label="occluded (<0.7) — P2 limited",
            color="tab:red",
        )
        a2.set_xticks(x)
        a2.set_xticklabels(H_LABELS)
        a2.set_xlabel("box height (px)")
        a2.set_ylabel("# false negatives")
        a2.set_title("FN by height, visible vs occluded")
        a2.legend()
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "fig5_fn_strata.png", dpi=110)
        plt.close(fig)
        print(
            f"\n[saved] {out_dir / 'fn_strata_report.txt'}  and  {FIG_DIR / 'fig5_fn_strata.png'}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    sys.exit(main())
