#!/usr/bin/env python3
"""Frame-level occlusion event analysis: label every tracker→GT pair by occlusion state.

For each frame, for each active track:
  1. Compute occ_coeff = max IoU with any other track (proxy for occlusion intensity)
  2. Match tracker output to GT via IoU → classify as TP/FP/FN/ID_swap
  3. Aggregate metrics stratified by occlusion state (occ vs non-occ)

Inputs:
  scripts/eval/output/ablation_mot17/baseline/MOT17-XX-SDP.txt  (τ=0)
  scripts/eval/output/ablation_mot17/association/oao_tau03/MOT17-XX-SDP.txt  (τ=0.3)
  datasets/MOT17/train/MOT17-XX-SDP/gt/gt.txt

Outputs:
  stdout tables
  occlusion_events.csv (per-track-per-frame occlusion annotation)
  occlusion_metrics.json (stratified metrics)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)


SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

ABLATION_DIR = Path("scripts/eval/output/ablation_mot17")
GT_ROOT = Path("datasets/MOT17/train")


def bbox_iou(a, b):
    """IoU between two [x1,y1,x2,y2] or [cx,cy,w,h] (both same convention)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-9 else 0.0


def xywh_to_xyxy(box):
    x, y, w, h = box[2:6]
    return [x, y, x + w, y + h]


def load_mot_file(path: Path):
    """Load MOT format: frame, id, x, y, w, h, score, -1, -1, -1"""
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            score = float(parts[6]) if len(parts) > 6 else 0.0
            frames[frame].append(
                {
                    "track_id": tid,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "score": score,
                    "xyxy": [x, y, x + w, y + h],
                }
            )
    return frames


def load_gt_file(path: Path):
    """Load GT: frame, id, x, y, w, h, active, class, visibility"""
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            active = int(parts[6]) if len(parts) > 6 else 1
            cls_id = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if active != 1 or cls_id != 1:
                continue
            frames[frame].append(
                {
                    "gt_id": gt_id,
                    "xyxy": [x, y, x + w, y + h],
                    "height": h,
                    "visibility": vis,
                }
            )
    return frames


def compute_occ_coeff(tracks_in_frame):
    """For each track, compute max IoU with any other track in the same frame."""
    n = len(tracks_in_frame)
    occ = [0.0] * n
    boxes = [t["xyxy"] for t in tracks_in_frame]
    for i in range(n):
        max_iou = 0.0
        for j in range(n):
            if i == j:
                continue
            iou = bbox_iou(boxes[i], boxes[j])
            if iou > max_iou:
                max_iou = iou
        occ[i] = max_iou
    return occ


def match_frame(tracks, gt_boxes, iou_thresh=0.5):
    """Greedy match: assign each GT to best track by IoU >= threshold.

    Returns:
      matched: list of (track_idx, gt_idx, iou) for matched pairs
      unmatched_tracks: list of track_idx
      unmatched_gt: list of gt_idx
    """
    n_trk = len(tracks)
    n_gt = len(gt_boxes)
    trk_boxes = [t["xyxy"] for t in tracks]
    gt_boxes_xyxy = [g["xyxy"] for g in gt_boxes]

    pairs = []
    for i in range(n_trk):
        for j in range(n_gt):
            iou = bbox_iou(trk_boxes[i], gt_boxes_xyxy[j])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    trk_matched = set()
    gt_matched = set()
    matched = []
    for iou, i, j in pairs:
        if i not in trk_matched and j not in gt_matched:
            matched.append((i, j, iou))
            trk_matched.add(i)
            gt_matched.add(j)

    unmatched_tracks = [i for i in range(n_trk) if i not in trk_matched]
    unmatched_gt = [j for j in range(n_gt) if j not in gt_matched]
    return matched, unmatched_tracks, unmatched_gt


def analyze_experiment(result_frames, gt_frames, label="baseline"):
    """For each frame, classify every track by occlusion state and match outcome."""
    rows = []
    all_frames = sorted(set(result_frames.keys()) & set(gt_frames.keys()))

    summary = {
        "tp_occ": 0,
        "tp_nonocc": 0,
        "fp_occ": 0,
        "fp_nonocc": 0,
        "fn_occ": 0,
        "fn_nonocc": 0,
        "total_occ_tracks": 0,
        "total_nonocc_tracks": 0,
        "frames_with_occ": 0,
        "total_frames": 0,
    }

    id_history = defaultdict(dict)  # track_id -> {prev_gt_id, swap_count}

    OCC_THRESH = 0.3  # IoU >= 0.3 counts as "occluded/overlapping"

    for frame in all_frames:
        tracks = result_frames[frame]
        gts = gt_frames[frame]
        if not tracks or not gts:
            continue

        occ_coeffs = compute_occ_coeff(tracks)
        matched, unmatched_trk, unmatched_gt = match_frame(tracks, gts)

        has_occ = any(c >= OCC_THRESH for c in occ_coeffs)
        summary["total_frames"] += 1
        if has_occ:
            summary["frames_with_occ"] += 1

        # TPs
        for trk_idx, gt_idx, iou in matched:
            is_occ = occ_coeffs[trk_idx] >= OCC_THRESH
            track_id = tracks[trk_idx]["track_id"]
            gt_id = gts[gt_idx]["gt_id"]
            if is_occ:
                summary["tp_occ"] += 1
                summary["total_occ_tracks"] += 1
            else:
                summary["tp_nonocc"] += 1
                summary["total_nonocc_tracks"] += 1

            prev_gt = id_history[track_id].get("gt_id")
            is_swap = prev_gt is not None and prev_gt != gt_id
            if is_swap:
                id_history[track_id]["swaps"] = id_history[track_id].get("swaps", 0) + 1
            id_history[track_id]["gt_id"] = gt_id
            id_history[track_id]["is_occ"] = is_occ

            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "gt_id": gt_id,
                    "is_occluded": int(is_occ),
                    "occ_coeff": round(occ_coeffs[trk_idx], 3),
                    "match_type": "TP",
                    "iou": round(iou, 3),
                    "gt_height": round(gts[gt_idx]["height"], 1),
                    "gt_visibility": round(gts[gt_idx]["visibility"], 3),
                }
            )

        # FPs (detections without GT match)
        for trk_idx in unmatched_trk:
            is_occ = occ_coeffs[trk_idx] >= OCC_THRESH
            if is_occ:
                summary["fp_occ"] += 1
            else:
                summary["fp_nonocc"] += 1
            track_id = tracks[trk_idx]["track_id"]
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "gt_id": -1,
                    "is_occluded": int(is_occ),
                    "occ_coeff": round(occ_coeffs[trk_idx], 3),
                    "match_type": "FP",
                    "iou": 0.0,
                    "gt_height": -1,
                    "gt_visibility": -1,
                }
            )

        # FNs (GTs without track match)
        for gt_idx in unmatched_gt:
            # Determine if this GT is in an occluded region by checking overlap with any track
            gt_box = gts[gt_idx]["xyxy"]
            is_occ = any(
                bbox_iou(gt_box, tracks[t]["xyxy"]) >= 0.3 for t in range(len(tracks))
            )
            # Also check GT-GT overlap
            other_gt = [g for k, g in enumerate(gts) if k != gt_idx]
            if any(bbox_iou(gt_box, g["xyxy"]) >= OCC_THRESH for g in other_gt):
                is_occ = True
            if is_occ:
                summary["fn_occ"] += 1
            else:
                summary["fn_nonocc"] += 1
            rows.append(
                {
                    "frame": frame,
                    "track_id": -1,
                    "gt_id": gts[gt_idx]["gt_id"],
                    "is_occluded": int(is_occ),
                    "occ_coeff": -1,
                    "match_type": "FN",
                    "iou": 0.0,
                    "gt_height": round(gts[gt_idx]["height"], 1),
                    "gt_visibility": round(gts[gt_idx]["visibility"], 3),
                }
            )

    (
        summary["tp_occ"]
        + summary["tp_nonocc"]
        + summary["fn_occ"]
        + summary["fn_nonocc"]
    )
    (
        summary["tp_occ"]
        + summary["tp_nonocc"]
        + summary["fp_occ"]
        + summary["fp_nonocc"]
    )

    # Compute stratified metrics
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    metrics = {
        "occ": {
            "tp": summary["tp_occ"],
            "fp": summary["fp_occ"],
            "fn": summary["fn_occ"],
            "total_tracks": summary["total_occ_tracks"],
        },
        "non_occ": {
            "tp": summary["tp_nonocc"],
            "fp": summary["fp_nonocc"],
            "fn": summary["fn_nonocc"],
            "total_tracks": summary["total_nonocc_tracks"],
        },
        "overall": {
            "frames_with_occ": summary["frames_with_occ"],
            "total_frames": summary["total_frames"],
        },
    }
    return rows, metrics


def print_stratified_results(baseline: dict, oao: dict):
    print("=" * 70)
    print("Per-frame occlusion-stratified analysis")
    print("=" * 70)

    for name, m in [("Baseline (τ=0)", baseline), ("OAO τ=0.3", oao)]:
        occ = m["occ"]
        nonocc = m["non_occ"]
        overall = m["overall"]
        occ_prec = (
            occ["tp"] / (occ["tp"] + occ["fp"]) if (occ["tp"] + occ["fp"]) > 0 else 0
        )
        occ_rec = (
            occ["tp"] / (occ["tp"] + occ["fn"]) if (occ["tp"] + occ["fn"]) > 0 else 0
        )
        non_prec = (
            nonocc["tp"] / (nonocc["tp"] + nonocc["fp"])
            if (nonocc["tp"] + nonocc["fp"]) > 0
            else 0
        )
        non_rec = (
            nonocc["tp"] / (nonocc["tp"] + nonocc["fn"])
            if (nonocc["tp"] + nonocc["fn"]) > 0
            else 0
        )
        occ_pct = (
            overall["frames_with_occ"] / overall["total_frames"] * 100
            if overall["total_frames"] > 0
            else 0
        )
        print(f"\n{name}:")
        print(
            f"  Occluded tracks:   TP={occ['tp']:>6}  FP={occ['fp']:>6}  FN={occ['fn']:>6}  "
            f"Prec={occ_prec:.3f}  Rec={occ_rec:.3f}"
        )
        print(
            f"  Non-occluded:      TP={nonocc['tp']:>6}  FP={nonocc['fp']:>6}  FN={nonocc['fn']:>6}  "
            f"Prec={non_prec:.3f}  Rec={non_rec:.3f}"
        )
        print(
            f"  Frames with occ:   {overall['frames_with_occ']}/{overall['total_frames']} "
            f"({occ_pct:.1f}%)"
        )

    # Comparison
    print(f"\n{'Metric':<30} {'Baseline':>10} {'OAO τ=0.3':>10} {'Change':>10}")
    print("-" * 65)
    for key, label in [("tp", "Occ TP"), ("fp", "Occ FP"), ("fn", "Occ FN")]:
        bv = baseline["occ"][key]
        ov = oao["occ"][key]
        delta = ov - bv
        print(f"{label:<30} {bv:>10} {ov:>10} {delta:>+10}")


def compare_per_seq(baseline_rows, oao_rows, label_b, label_o):
    print(
        f"\n{'Sequence':<18} {'Occ FP delta':>12} {'Occ FN delta':>12} {'Occ Prec delta':>14} {'Occ Rec delta':>14}"
    )
    print("-" * 75)
    for seq in SEQUENCES:
        seq.replace("MOT17-", "").replace("-SDP", "")
        [r for r in baseline_rows if r.get("sequence", "MOT17-02-SDP") == seq or True]
        # Actually, rows don't have sequence tag since we process per-seq
        pass


def main():
    parser = argparse.ArgumentParser(description="Frame-level occlusion event analysis")
    parser.add_argument(
        "--output",
        default="scripts/eval/output/oao_analysis",
        help="Output directory for CSVs and JSON",
    )
    parser.add_argument(
        "--seq",
        default=None,
        choices=SEQUENCES,
        help="Analyze a single sequence (default: all)",
    )
    parser.add_argument("--csv", action="store_true", help="Write per-event CSV")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = [args.seq] if args.seq else SEQUENCES
    exp_pairs = [
        ("baseline", ABLATION_DIR / "baseline"),
        ("oao_tau03", ABLATION_DIR / "association" / "oao_tau03"),
    ]

    all_results = {}

    for exp_name, exp_dir in exp_pairs:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {exp_name}")
        print(f"{'=' * 60}")
        all_rows = []
        all_metrics = defaultdict(lambda: defaultdict(int))

        for seq in sequences:
            result_file = exp_dir / f"{seq}.txt"
            gt_file = GT_ROOT / seq / "gt" / "gt.txt"
            if not result_file.exists():
                # try without detector suffix
                base_part = seq.rsplit("-", 1)[0]
                gt_file = GT_ROOT / f"{base_part}-SDP" / "gt" / "gt.txt"

            if not gt_file.exists():
                print(f"  [skip] {seq}: GT not found at {gt_file}")
                continue

            print(f"  {seq}...")
            result_frames = load_mot_file(result_file)
            gt_frames = load_gt_file(gt_file)
            rows, metrics = analyze_experiment(result_frames, gt_frames, exp_name)
            all_rows.extend(rows)
            for k, v in metrics.items():
                for k2, v2 in v.items():
                    all_metrics[k][k2] += v2

        all_results[exp_name] = {"rows": all_rows, "metrics": dict(all_metrics)}

        if args.csv:
            csv_path = output_dir / f"occlusion_events_{exp_name}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sequence",
                        "frame",
                        "track_id",
                        "gt_id",
                        "is_occluded",
                        "occ_coeff",
                        "match_type",
                        "iou",
                        "gt_height",
                        "gt_visibility",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"  → {csv_path} ({len(all_rows)} events)")

    # Print comparison
    print("\n")
    print_stratified_results(
        all_results["baseline"]["metrics"], all_results["oao_tau03"]["metrics"]
    )

    # Save metrics JSON
    metrics_out = {name: data["metrics"] for name, data in all_results.items()}
    with open(output_dir / "occlusion_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=int)
    print(f"\nMetrics saved to {output_dir}/occlusion_metrics.json")

    # Occlusion severity histogram
    try:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[skip] matplotlib not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (exp_name, data) in enumerate(all_results.items()):
            coeffs = [
                r["occ_coeff"]
                for r in data["rows"]
                if r["match_type"] == "TP" and r["occ_coeff"] > 0.01
            ]
            axes[idx].hist(
                coeffs,
                bins=30,
                alpha=0.7,
                color="tab:blue" if idx == 0 else "tab:orange",
            )
            axes[idx].set_title(f"Occ coeff distribution (TPs only) — {exp_name}")
            axes[idx].set_xlabel("occ_coeff (max track-track IoU)")
            axes[idx].set_ylabel("count")
            axes[idx].axvline(
                x=0.3, color="red", linestyle="--", alpha=0.5, label="occ threshold"
            )
            axes[idx].legend()
        plt.tight_layout()
        plt.savefig(output_dir / "occlusion_coeff_hist.png", dpi=120)
        plt.close()
        print(f"Chart saved to {output_dir}/occlusion_coeff_hist.png")
    except Exception as e:
        print(f"Chart generation failed: {e}")


if __name__ == "__main__":
    main()
