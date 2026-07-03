#!/usr/bin/env python3
"""mnv4 appearance-veto separability at PRIMARY association decision points.

Question: at the moment the tracker hands a box to the WRONG track (SWITCH on
the substrate output), would a conservative cosine veto against the track's own
clean reference embedding have blocked the claim — without also vetoing
ordinary correct continuations?

This is the runtime-faithful version of the bridge_app_veto question, moved to
the main association: the veto only needs the easy end of the AUC curve
("clearly a different person"), so the decisive numbers are operating points at
FIXED false-veto rates measured on a stream of ordinary continuations, not the
aggregate AUC.

Populations (all crops via the native C++/CUDA + TRT mnv4 path, PIL fallback):
  P_wrong   : per SWITCH event, cosine(track reference, claimed box at swap
              frame). Reference = up to --n-ref clean (GT vis >= --ref-vis)
              hyp-box crops of the track's own PREVIOUS identity — what an
              occ-freeze bank would actually hold. Events without a clean
              reference are counted as ABSTAIN (runtime veto cannot fire).
  P_stream  : ordinary correct continuations sampled across each sequence
              (same reference recipe, probe = the track's own matched box).
              This population sets the veto threshold: tau = quantile at a
              target false-veto rate.
  P_correct : at each swap frame, cosine(reference, the track's TRUE GT box)
              — does the correct runner-up SURVIVE the veto (ranking use,
              same contract as bridge_app_veto)?

Stratified by claimed-box GT visibility (dirty-crop risk) and REBORN/ABSORB.

Usage
-----
  .venv/bin/python scripts/eval/probe_assoc_appearance_veto.py \
      --substrate results/analysis_m_semantic_delayed_claim_control_20260703
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402
import torch  # noqa: E402

from saccade.perception.eval.cheb_gr_merge import (  # noqa: E402
    _extract_native_crops_trt,
)
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
FALSE_VETO_TARGETS = (0.001, 0.005, 0.01, 0.02)
VIS_BUCKETS = (
    (0.0, 0.3, "vis<0.3"),
    (0.3, 0.7, "vis 0.3-0.7"),
    (0.7, 1.01, "vis>=0.7"),
)


def load_hyp_boxes(path: Path) -> dict[int, dict[int, tuple]]:
    by: dict[int, dict[int, tuple]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        by[int(p[1])][int(p[0])] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return dict(by)


def load_gt(
    path: Path,
) -> tuple[dict[int, dict[int, tuple]], dict[int, dict[int, float]]]:
    box: dict[int, dict[int, tuple]] = defaultdict(dict)
    vis: dict[int, dict[int, float]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 9 or int(p[6]) != 1 or int(p[7]) != 1:
            continue
        f, tid = int(p[0]), int(p[1])
        box[tid][f] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        vis[tid][f] = float(p[8])
    return dict(box), dict(vis)


def xywh_to_xyxy(b: tuple) -> tuple[float, float, float, float]:
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


class CropBatch:
    """Collects (frame, xyxy) crop requests; one grouped extraction per seq."""

    def __init__(self) -> None:
        self.samples: list[tuple[int, int, tuple[float, float, float, float]]] = []

    def add(self, frame: int, xyxy: tuple[float, float, float, float]) -> int:
        idx = len(self.samples)
        self.samples.append((idx, frame, xyxy))
        return idx

    def extract(self, seq_img_dir: Path, extractor, batch: int) -> torch.Tensor:
        by_frame: dict[int, list[int]] = defaultdict(list)
        for si, (_, frame, _) in enumerate(self.samples):
            by_frame[frame].append(si)
        crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))
        feats = _extract_native_crops_trt(
            self.samples,
            by_frame,
            str(seq_img_dir),
            extractor,
            crop_hw=crop_hw,
            im_ext=".jpg",
            batch=batch,
        )
        if feats is None:  # PIL fallback (same crop contract)
            from PIL import Image

            out_h, out_w = crop_hw
            device = getattr(extractor, "device", "cuda")
            feats = torch.empty(
                (len(self.samples), extractor.feature_dim), device=device
            )
            arrs = [None] * len(self.samples)
            for frame, idxs in by_frame.items():
                img = Image.open(seq_img_dir / f"{frame:06d}.jpg").convert("RGB")
                fw, fh = img.size
                for si in idxs:
                    x0, y0, x1, y1 = self.samples[si][2]
                    box = (
                        max(0, int(round(x0))),
                        max(0, int(round(y0))),
                        min(fw, int(round(x1))),
                        min(fh, int(round(y1))),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        box = (0, 0, fw, fh)
                    crop = img.crop(box).resize((out_w, out_h), Image.BILINEAR)
                    arrs[si] = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)
            for s in range(0, len(arrs), max(1, batch)):
                chunk = [a for a in arrs[s : s + batch] if a is not None]
                t = torch.from_numpy(np.stack(chunk)).to(device).float().div_(255.0)
                feats[s : s + t.shape[0]] = extractor.extract(t)
        return torch.nn.functional.normalize(feats.float(), dim=1)


def clean_ref_frames(
    hist: list[tuple[int, int]],
    oid: int,
    before: int,
    vis: dict[int, dict[int, float]],
    *,
    n_ref: int,
    ref_vis: float,
    lookback: int,
) -> list[int]:
    """Last n_ref frames < `before` where hyp matched `oid` with clean GT vis."""
    frames = [
        mf
        for mf, o in hist
        if o == oid
        and before - lookback <= mf < before
        and vis.get(oid, {}).get(mf, 0.0) >= ref_vis
    ]
    return frames[-n_ref:]


def collect_seq(
    seq: str,
    gt_root: Path,
    substrate: Path,
    extractor,
    args,
) -> tuple[list[dict], list[dict], dict]:
    gt_path = gt_root / seq / "gt" / "gt.txt"
    hyp_path = substrate / f"{seq}.txt"
    gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
    hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5)
    events = acc.events

    hyp_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)  # HId -> [(f,OId)]
    matched = events[events["Type"].isin(["MATCH", "SWITCH"])]
    for (f, _), row in matched.iterrows():
        if isinstance(row["OId"], float) and np.isnan(row["OId"]):
            continue
        hyp_hist[int(row["HId"])].append((int(f), int(row["OId"])))
    for h in hyp_hist:
        hyp_hist[h].sort()

    gt_box, gt_vis = load_gt(gt_path)
    hyp_box = load_hyp_boxes(hyp_path)
    hyp_birth = {tid: min(fr) for tid, fr in hyp_box.items()}

    batch_req = CropBatch()
    switch_recs: list[dict] = []
    counts = {"switch_total": 0, "no_prev_identity": 0, "no_clean_ref": 0, "no_box": 0}

    for (f, _), row in events[events["Type"] == "SWITCH"].iterrows():
        f = int(f)
        g, h_new = int(row["OId"]), int(row["HId"])
        counts["switch_total"] += 1
        hist = hyp_hist.get(h_new, [])
        prev_other = [(mf, o) for mf, o in hist if mf < f and o != g]
        if not prev_other:
            counts["no_prev_identity"] += 1  # fresh/REBORN: veto has no reference
            continue
        g2 = prev_other[-1][1]
        refs = clean_ref_frames(
            hist,
            g2,
            f,
            gt_vis,
            n_ref=args.n_ref,
            ref_vis=args.ref_vis,
            lookback=args.lookback,
        )
        if len(refs) < args.min_ref:
            counts["no_clean_ref"] += 1  # runtime veto abstains
            continue
        if f not in hyp_box.get(h_new, {}):
            counts["no_box"] += 1
            continue
        rec = {
            "seq": seq,
            "frame": f,
            "hyp": h_new,
            "gt_claimed": g,
            "gt_prev": g2,
            "vis_claim": gt_vis.get(g, {}).get(f, 1.0),
            "reborn": (f - hyp_birth.get(h_new, f)) < 2,
            "ref_gap": f - refs[-1],
            "n_ref_used": len(refs),
            "ref_idx": [
                batch_req.add(mf, xywh_to_xyxy(hyp_box[h_new][mf])) for mf in refs
            ],
            "wrong_idx": batch_req.add(f, xywh_to_xyxy(hyp_box[h_new][f])),
            "correct_idx": None,
            "vis_correct": None,
        }
        if f in gt_box.get(g2, {}):
            rec["correct_idx"] = batch_req.add(f, xywh_to_xyxy(gt_box[g2][f]))
            rec["vis_correct"] = gt_vis.get(g2, {}).get(f, 1.0)
        # Delayed audit: first clean crops of the wrongly-claimed target AFTER
        # the swap, while h_new still has a box (post-occlusion identity check).
        audit = []
        for fa in range(f + 1, f + 1 + args.audit_window):
            if len(audit) >= args.audit_crops:
                break
            if gt_vis.get(g, {}).get(fa, 0.0) >= args.ref_vis and fa in hyp_box.get(
                h_new, {}
            ):
                audit.append((fa, batch_req.add(fa, xywh_to_xyxy(hyp_box[h_new][fa]))))
        rec["audit"] = audit
        switch_recs.append(rec)

    # Stream controls: ordinary correct continuations, stride-sampled.
    stream_cand: list[tuple[int, int, int]] = []  # (f, h, oid)
    switch_frames = {(r["hyp"], r["frame"]) for r in switch_recs}
    for h, hist in hyp_hist.items():
        for i in range(1, len(hist)):
            f, o = hist[i]
            if hist[i - 1][1] == o and (h, f) not in switch_frames:
                stream_cand.append((f, h, o))
    stream_cand.sort()
    stride = max(1, len(stream_cand) // max(1, args.stream_samples))
    stream_recs: list[dict] = []
    for f, h, o in stream_cand[::stride]:
        refs = clean_ref_frames(
            hyp_hist[h],
            o,
            f,
            gt_vis,
            n_ref=args.n_ref,
            ref_vis=args.ref_vis,
            lookback=args.lookback,
        )
        if len(refs) < args.min_ref or f not in hyp_box.get(h, {}):
            continue
        stream_recs.append(
            {
                "seq": seq,
                "vis_probe": gt_vis.get(o, {}).get(f, 1.0),
                "ref_idx": [
                    batch_req.add(mf, xywh_to_xyxy(hyp_box[h][mf])) for mf in refs
                ],
                "probe_idx": batch_req.add(f, xywh_to_xyxy(hyp_box[h][f])),
            }
        )

    if not batch_req.samples:
        return switch_recs, stream_recs, counts

    feats = batch_req.extract(gt_root / seq / "img1", extractor, args.batch)

    def ref_embed(idxs: list[int]) -> torch.Tensor:
        return torch.nn.functional.normalize(
            feats[idxs].mean(dim=0, keepdim=True), dim=1
        )[0]

    for r in switch_recs:
        ref = ref_embed(r["ref_idx"])
        r["cos_wrong"] = float(ref @ feats[r["wrong_idx"]])
        r["cos_correct"] = (
            float(ref @ feats[r["correct_idx"]])
            if r["correct_idx"] is not None
            else None
        )
        r["audit_cos"] = [
            (fa - r["frame"], float(ref @ feats[ai])) for fa, ai in r["audit"]
        ]
        del r["audit"]
    for r in stream_recs:
        r["cos"] = float(ref_embed(r["ref_idx"]) @ feats[r["probe_idx"]])
    return switch_recs, stream_recs, counts


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that `pos` (correct, should be high) ranks above `neg` (wrong)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort().astype(np.float64) + 1.0
    r_pos = order[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    ap.add_argument("--n-ref", type=int, default=5)
    ap.add_argument("--min-ref", type=int, default=2)
    ap.add_argument("--ref-vis", type=float, default=0.7)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--stream-samples", type=int, default=300, help="per sequence")
    ap.add_argument("--audit-window", type=int, default=30, help="frames after swap")
    ap.add_argument("--audit-crops", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None, help="dump raw records as JSON")
    args = ap.parse_args()

    gt_root = PROJECT_ROOT / "datasets" / "MOT17" / "train"
    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=args.batch,
    )

    all_switch: list[dict] = []
    all_stream: list[dict] = []
    counts = defaultdict(int)
    for seq in args.seqs:
        sw, st, c = collect_seq(seq, gt_root, args.substrate, extractor, args)
        all_switch.extend(sw)
        all_stream.extend(st)
        for k, v in c.items():
            counts[k] += v
        print(
            f"{seq}: switches={c['switch_total']} scored={len(sw)} "
            f"(no_prev={c['no_prev_identity']} no_clean_ref={c['no_clean_ref']}) "
            f"stream={len(st)}"
        )

    wrong = np.array([r["cos_wrong"] for r in all_switch])
    stream = np.array([r["cos"] for r in all_stream])
    correct = np.array(
        [r["cos_correct"] for r in all_switch if r["cos_correct"] is not None]
    )
    n_abstain = counts["no_prev_identity"] + counts["no_clean_ref"]

    print("\n=== populations ===")
    print(
        f"wrong claims scored : {len(wrong)}  (abstain: {n_abstain} / "
        f"{counts['switch_total']} switches have no clean reference)"
    )
    print(f"stream correct      : {len(stream)}")
    print(
        f"cos median  wrong={np.median(wrong):.3f}  stream={np.median(stream):.3f}"
        f"  correct-at-swap={np.median(correct):.3f}"
    )
    print(f"AUC (stream > wrong): {rank_auc(stream, wrong):.3f}")

    print("\n=== veto operating points (tau = stream quantile) ===")
    print(
        f"{'false-veto':>10} {'tau':>7} {'wrong vetoed':>13} {'runner-up survives':>19}"
    )
    for q in FALSE_VETO_TARGETS:
        tau = float(np.quantile(stream, q))
        vetoed = float((wrong < tau).mean()) if len(wrong) else float("nan")
        survive = float((correct >= tau).mean()) if len(correct) else float("nan")
        print(f"{q:>10.1%} {tau:>7.3f} {vetoed:>12.1%} {survive:>18.1%}")

    print("\n=== wrong-vetoed by claimed-box GT visibility (tau @ 0.5% false-veto) ===")
    tau05 = float(np.quantile(stream, 0.005))
    for lo, hi, name in VIS_BUCKETS:
        sub = np.array(
            [r["cos_wrong"] for r in all_switch if lo <= r["vis_claim"] < hi]
        )
        if len(sub):
            print(
                f"{name:>12}: n={len(sub):>4}  vetoed={float((sub < tau05).mean()):.1%}"
            )
    for flag, name in ((False, "ABSORB"), (True, "REBORN")):
        sub = np.array([r["cos_wrong"] for r in all_switch if r["reborn"] == flag])
        if len(sub):
            print(
                f"{name:>12}: n={len(sub):>4}  vetoed={float((sub < tau05).mean()):.1%}"
            )

    print("\n=== delayed audit (clean-crop check within --audit-window after swap) ===")
    stream_clean = np.array(
        [r["cos"] for r in all_stream if r["vis_probe"] >= args.ref_vis]
    )
    audited = [r for r in all_switch if r["audit_cos"]]
    print(
        f"switches with a clean post-swap crop: {len(audited)}/{len(all_switch)} scored"
    )
    for q in (0.001, 0.005):
        tau = float(np.quantile(stream_clean, q))
        flagged = [r for r in audited if min(c for _, c in r["audit_cos"]) < tau]
        delays = [min(dt for dt, c in r["audit_cos"] if c < tau) for r in flagged]
        med = int(np.median(delays)) if delays else -1
        print(
            f"  clean-stream fv={q:.1%} tau={tau:.3f}: flagged {len(flagged)}/{len(audited)}"
            f"  (median delay {med} frames)"
        )

    if args.out:
        args.out.write_text(
            json.dumps(
                {"switch": all_switch, "stream": all_stream, "counts": dict(counts)},
                indent=1,
                default=lambda o: None,
            )
        )
        print(f"\nraw records -> {args.out}")


if __name__ == "__main__":
    main()
