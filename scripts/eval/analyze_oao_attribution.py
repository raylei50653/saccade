#!/usr/bin/env python3
"""OAO causal attribution: classify FP/FN changes between baseline (τ=0) and OAO (τ=0.3).

For each frame, compares the tracker output of two experiments:
  - baseline: oao_tau=0
  - oao:      oao_tau=0.3

Classifies every detection (FP) and missed GT (FN) by cause:
  FP categories:
    stolen_recovered: track in baseline that mapped to wrong GT → OAO drops it (FP→no output)
    noise_both:       FP in both (detector noise, not OAO-related)
    new_fp:           FP only in OAO (OAO regression introducing new FPs, should be rare)
    fp_resolved:      FP in baseline, OAO reassigns to correct GT (FP→TP, ideal)

  FN categories:
    occluded_lost:    GT occluded in baseline, correctly tracked; OAO penalty kills it (bad)
    noise_both:       FN in both (detector gap, not OAO-related)
    recovered:         FN in baseline, OAO recovers (ideally FP→TP from fp_resolved)
    new_fn:           FN only in OAO (OAO suppressing a correct but occluded track)

Inputs:
  scripts/eval/output/ablation_mot17/baseline/MOT17-XX-SDP.txt
  scripts/eval/output/ablation_mot17/association/oao_tau03/MOT17-XX-SDP.txt
  datasets/MOT17/train/MOT17-XX-SDP/gt/gt.txt

Outputs:
  stdout attribution table
  oao_attribution.json (per-category counts)
  oao_attribution.csv (per-event details)
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
OCC_THRESH = 0.3


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-9 else 0.0


def load_mot_frames(path: Path):
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frm = int(float(p[0]))
            tid, x, y, w, h = (
                int(float(p[1])),
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(p[5]),
            )
            frames[frm].append(
                {
                    "id": tid,
                    "xyxy": [x, y, x + w, y + h],
                    "score": float(p[6]) if len(p) > 6 else 0.0,
                }
            )
    return dict(frames)


def load_gt_frames(path: Path):
    frames = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frm = int(float(p[0]))
            active = int(p[6]) if len(p) > 6 else 1
            cls_id = int(p[7]) if len(p) > 7 else 1
            if active != 1 or cls_id != 1:
                continue
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            frames[frm].append(
                {
                    "gt_id": int(float(p[1])),
                    "xyxy": [x, y, x + w, y + h],
                    "height": h,
                    "vis": float(p[8]) if len(p) > 8 else 1.0,
                }
            )
    return dict(frames)


def compute_occ_coeff(detections):
    """max IoU with any other detection in frame."""
    n = len(detections)
    occ = [0.0] * n
    boxes = [d["xyxy"] for d in detections]
    for i in range(n):
        m = 0.0
        for j in range(n):
            if i == j:
                continue
            m = max(m, bbox_iou(boxes[i], boxes[j]))
        occ[i] = m
    return occ


def greedy_match(detections, targets, iou_thresh=0.5):
    """Greedy bipartite matching by IoU."""
    pairs = []
    for i, d in enumerate(detections):
        for j, t in enumerate(targets):
            iou = bbox_iou(d["xyxy"], t["xyxy"])
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    d_matched = set()
    t_matched = set()
    matches = []
    for iou, i, j in pairs:
        if i not in d_matched and j not in t_matched:
            matches.append((i, j, iou))
            d_matched.add(i)
            t_matched.add(j)
    return matches, d_matched, t_matched


def analyze_sequence(seq, result_baseline, result_oao, gt_all):
    """Per-sequence attribution analysis."""
    common_frames = sorted(
        set(result_baseline.keys()) & set(result_oao.keys()) & set(gt_all.keys())
    )

    attribution = {
        "fp_stolen_recovered": 0,
        "fp_both": 0,
        "fp_new": 0,
        "fp_resolved_to_tp": 0,
        "fn_occluded_lost": 0,
        "fn_both": 0,
        "fn_recovered": 0,
        "fn_new": 0,
        "occ_frames": 0,
        "total_frames": 0,
    }
    events = []

    for frame in common_frames:
        det_base = result_baseline.get(frame, [])
        det_oao = result_oao.get(frame, [])
        gts = gt_all.get(frame, [])

        if not gts:
            continue

        occ_base = compute_occ_coeff(det_base)
        occ_oao = compute_occ_coeff(det_oao)

        has_occ = any(c >= OCC_THRESH for c in occ_base) or any(
            c >= OCC_THRESH for c in occ_oao
        )
        attribution["total_frames"] += 1
        if has_occ:
            attribution["occ_frames"] += 1

        # Match detections → GT
        m_base, d_m_base, gt_m_base = greedy_match(det_base, gts)
        m_oao, d_m_oao, gt_m_oao = greedy_match(det_oao, gts)

        # Build mappings
        {i: j for i, j, _ in m_base}
        oao_d_to_gt = {i: j for i, j, _ in m_oao}
        {j: i for i, j, _ in m_base}
        {j: i for i, j, _ in m_oao}

        base_gt_matched = set(j for _, j, _ in m_base)
        oao_gt_matched = set(j for _, j, _ in m_oao)

        # --- FP attribution ---
        # FPs in baseline: detections not matched to GT
        for i, det in enumerate(det_base):
            if i in d_m_base:
                continue  # already TP in baseline

            is_occ = occ_base[i] >= OCC_THRESH
            any(d["id"] == det["id"] for d in det_oao)

            # Find matching detection in OAO (by IoU to same position)
            best_oao_match = -1
            best_oao_iou = 0.0
            for j, d_o in enumerate(det_oao):
                iou = bbox_iou(det["xyxy"], d_o["xyxy"])
                if iou > best_oao_iou:
                    best_oao_iou = iou
                    best_oao_match = j

            if best_oao_match >= 0 and best_oao_iou >= 0.5:
                # Same detection exists in OAO — check if it became TP in OAO
                if best_oao_match in oao_d_to_gt:
                    # Baseline FP → OAO TP (ideal! OAO suppressed this from wrong GT and got correct)
                    attribution["fp_resolved_to_tp"] += 1
                    events.append(
                        {
                            "frame": frame,
                            "type": "fp_resolved_to_tp",
                            "seq": seq,
                            "is_occ": int(is_occ),
                            "base_id": det["id"],
                            "gt_id_oao": gts[oao_d_to_gt[best_oao_match]]["gt_id"],
                        }
                    )
                elif j not in d_m_oao:
                    # Same detection exists, still FP in both
                    attribution["fp_both"] += 1
                    events.append(
                        {
                            "frame": frame,
                            "type": "fp_both",
                            "seq": seq,
                            "is_occ": int(is_occ),
                            "base_id": det["id"],
                        }
                    )
            elif best_oao_match >= 0 and best_oao_iou < 0.5:
                # Detection disappeared in OAO (potentially suppressed by OAO penalty)
                # Check if it was supposed to match a GT
                attribution["fp_stolen_recovered"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fp_stolen_recovered",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "base_id": det["id"],
                        "best_oao_iou": round(best_oao_iou, 3),
                    }
                )
            else:
                # No OAO detection nearby — disappeared (possibly suppressed)
                attribution["fp_stolen_recovered"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fp_stolen_recovered",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "base_id": det["id"],
                        "best_oao_iou": 0.0,
                    }
                )

        # FPs in OAO but not in baseline
        for i, det in enumerate(det_oao):
            if i in d_m_oao:
                continue
            # Check if this detection also exists in baseline
            best_base_match = -1
            best_base_iou = 0.0
            for j, d_b in enumerate(det_base):
                iou = bbox_iou(det["xyxy"], d_b["xyxy"])
                if iou > best_base_iou:
                    best_base_iou = iou
                    best_base_match = j
            if best_base_match < 0 or best_base_iou < 0.5:
                # New detection in OAO that wasn't in baseline
                is_occ = occ_oao[i] >= OCC_THRESH
                attribution["fp_new"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fp_new",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "oao_id": det["id"],
                    }
                )

        # --- FN attribution ---
        # FNs in baseline: GTs not matched
        for j, gt in enumerate(gts):
            gt_id = gt["gt_id"]
            in_baseline = j in base_gt_matched
            in_oao = j in oao_gt_matched

            # Check occlusion: any other GT overlap this one
            other_gt_boxes = [g["xyxy"] for k, g in enumerate(gts) if k != j]
            is_occ = any(
                bbox_iou(gt["xyxy"], og) >= OCC_THRESH for og in other_gt_boxes
            )

            if not in_baseline and not in_oao:
                attribution["fn_both"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fn_both",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "gt_id": gt_id,
                        "gt_height": round(gt["height"], 1),
                        "gt_vis": round(gt["vis"], 3),
                    }
                )
            elif in_baseline and not in_oao:
                # Was tracked in baseline but lost in OAO — OAO penalty killed a correct match
                attribution["fn_occluded_lost"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fn_occluded_lost",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "gt_id": gt_id,
                        "gt_height": round(gt["height"], 1),
                        "gt_vis": round(gt["vis"], 3),
                    }
                )
            elif not in_baseline and in_oao:
                # Was missed in baseline but tracked in OAO — OAO recovered it (FP→TP resolved)
                attribution["fn_recovered"] += 1
                events.append(
                    {
                        "frame": frame,
                        "type": "fn_recovered",
                        "seq": seq,
                        "is_occ": int(is_occ),
                        "gt_id": gt_id,
                        "gt_height": round(gt["height"], 1),
                        "gt_vis": round(gt["vis"], 3),
                    }
                )
            # else: both have it → already accounted in m_base/m_oao

    return attribution, events


def main():
    parser = argparse.ArgumentParser(description="OAO causal FP/FN attribution")
    parser.add_argument(
        "--output", default="scripts/eval/output/oao_analysis", help="Output directory"
    )
    parser.add_argument(
        "--seq", default=None, choices=SEQUENCES, help="Analyze single sequence"
    )
    parser.add_argument("--csv", action="store_true", help="Write per-event CSV")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = [args.seq] if args.seq else SEQUENCES
    baseline_dir = ABLATION_DIR / "baseline"
    oao_dir = ABLATION_DIR / "association" / "oao_tau03"

    total_attr = defaultdict(int)
    all_events = []

    for seq in sequences:
        gt_file = GT_ROOT / seq / "gt" / "gt.txt"
        if not gt_file.exists():
            base_part = seq.rsplit("-", 1)[0]
            gt_file = GT_ROOT / f"{base_part}-SDP" / "gt" / "gt.txt"

        print(f"  {seq}...")
        r_base = load_mot_frames(baseline_dir / f"{seq}.txt")
        r_oao = load_mot_frames(oao_dir / f"{seq}.txt")
        gts = load_gt_frames(gt_file)

        attr, events = analyze_sequence(seq, r_base, r_oao, gts)
        for k, v in attr.items():
            total_attr[k] += v
        all_events.extend(events)

    # Print attribution table
    print("\n" + "=" * 70)
    print("OAO (τ=0.3) FP/FN Attribution vs Baseline (τ=0)")
    print("=" * 70)

    print(f"\n{'Category':<35} {'Count':>8}  {'Note'}")
    print("-" * 70)
    fp_items = [
        ("fp_resolved_to_tp", "FP→TP (OAO recovers a correct match)", "positive"),
        ("fp_stolen_recovered", "FP→gone (OAO suppressed stray detection)", "positive"),
        ("fp_both", "FP in both (detector noise, not OAO)", "neutral"),
        ("fp_new", "FP only in OAO (OAO regression, new noise)", "negative"),
    ]
    fn_items = [
        ("fn_recovered", "FN→TP (OAO recovers missed GT)", "positive"),
        (
            "fn_occluded_lost",
            "FN only in OAO (suppressed correct occl track)",
            "negative",
        ),
        ("fn_both", "FN in both (detector gap, not OAO)", "neutral"),
        ("fn_new", "FN only in OAO (OAO regression)", "negative"),
    ]

    pos_total = neg_total = 0
    for cat, desc, sentiment in fp_items + fn_items:
        count = total_attr[cat]
        print(f"  {cat:<33} {count:>8}  [{sentiment}] {desc}")
        if sentiment == "positive":
            pos_total += count
        elif sentiment == "negative":
            neg_total += count

    print("-" * 70)
    net_effect = pos_total - neg_total
    print(f"  {'Net effect':<33} {net_effect:>+8}")
    print(f"    Positive (FP_removed + FN_recovered) = {pos_total}")
    print(f"    Negative (FP_new + FN_occluded_lost + FN_new) = {neg_total}")

    occ_keys = [
        "fp_stolen_recovered",
        "fp_resolved_to_tp",
        "fp_both",
        "fp_new",
        "fn_occluded_lost",
        "fn_recovered",
        "fn_both",
        "fn_new",
    ]
    occ_attr = {}
    for k in occ_keys:
        occ_attr[k] = sum(
            1 for e in all_events if e["type"] == k and e.get("is_occ", 0)
        )
    print(f"\nOcclusion-only events (occ_coeff ≥ {OCC_THRESH}):")
    print(f"  fp_stolen_recovered(occ): {occ_attr['fp_stolen_recovered']}")
    print(f"  fn_occluded_lost(occ):    {occ_attr['fn_occluded_lost']}")
    print(f"  fn_recovered(occ):       {occ_attr['fn_recovered']}")
    print(f"  fp_resolved_to_tp(occ):  {occ_attr['fp_resolved_to_tp']}")

    occ_frames_pct = total_attr["occ_frames"] / max(total_attr["total_frames"], 1) * 100
    print(
        f"\n  Frames with occlusion: {total_attr['occ_frames']}/{total_attr['total_frames']} "
        f"({occ_frames_pct:.1f}%)"
    )

    # Save results
    with open(output_dir / "oao_attribution.json", "w") as f:
        json.dump(dict(total_attr), f, indent=2, default=int)

    if args.csv:
        with open(output_dir / "oao_attribution.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "seq",
                    "type",
                    "is_occ",
                    "base_id",
                    "oao_id",
                    "gt_id",
                    "best_oao_iou",
                    "gt_height",
                    "gt_vis",
                ],
            )
            writer.writeheader()
            writer.writerows(all_events)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
