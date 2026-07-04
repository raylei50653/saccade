#!/usr/bin/env python3
"""Probe: TrackAppearanceBank / OutputAppearanceBank FIFO replacement.

On a frozen substrate, compare:
  (a) TrackAppearanceBank (primary): score-gated top-K + EMA representative
  (b) CleanFifoBank: visclean-gated FIFO-20 + raw samples / mean rep

Metrics: per-track representative cosine similarity, consistency diff,
clean_ids set diff, and post-merge appearance-gate accept/reject diff
(OutputAppearanceBank consumer).

The probe extracts embeddings once (max(fifo_n, K) clean crops per track),
then builds both banks from the same pool — a fair same-budget comparison.

Usage:
  .venv/bin/python scripts/eval/diagnostics/probe_track_bank_fifo_replacement.py \
      --substrate results/diag_m_no_reid_current_20260704 \
      --data-root datasets/MOT17 --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from saccade.perception.eval.clean_fifo_bank import (  # noqa: E402
    CleanFifoBank,
)
from saccade.perception.eval.helpers import front_occlusion_mask_xyxy  # noqa: E402
from saccade.perception.eval.post_merge import _parse_mot_lines  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def _extract_all_clean_crops(
    lines: list[str],
    seq_dir: str,
    extractor: object,
    *,
    max_per_track: int = 50,
    cov: float = 0.4,
    crop_hw: tuple[int, int] = (224, 224),
) -> dict[int, list[tuple[int, float, torch.Tensor]]]:
    """Extract up to ``max_per_track`` clean crops per track.

    Returns ``{track_id: [(frame_id, score, embedding), ...]}`` sorted by frame.
    """
    from PIL import Image
    from saccade.perception.eval.cheb_gr_merge import _extract_native_crops_trt

    records = _parse_mot_lines(lines)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(records):
        by_frame[r.frame].append(ri)

    dirty: set[int] = set()
    for idxs in by_frame.values():
        boxes = torch.tensor(
            [
                (
                    records[i].x,
                    records[i].y,
                    records[i].x + records[i].w,
                    records[i].y + records[i].h,
                )
                for i in idxs
            ],
            dtype=torch.float32,
        )
        mask = front_occlusion_mask_xyxy(boxes, cov)
        dirty.update(i for i, d in zip(idxs, mask.tolist()) if d)

    clean_by_id: dict[int, list] = defaultdict(list)
    for ri, r in enumerate(records):
        if ri not in dirty:
            clean_by_id[r.track_id].append(r)

    pool: list[tuple[int, int, tuple[float, float, float, float]]] = []
    track_crops: dict[int, list[int]] = defaultdict(list)
    for tid, items in clean_by_id.items():
        items.sort(key=lambda r: r.frame)
        for r in items[:max_per_track]:
            idx = len(pool)
            pool.append((tid, r.frame, (r.x, r.y, r.x + r.w, r.y + r.h)))
            track_crops[tid].append(idx)
    if not pool:
        return {}

    by_frame_s: dict[int, list[int]] = defaultdict(list)
    for si, (_, fr, _) in enumerate(pool):
        by_frame_s[fr].append(si)

    feats = _extract_native_crops_trt(
        pool,
        by_frame_s,
        seq_dir,
        extractor,
        crop_hw=crop_hw,
        im_ext=".jpg",
        batch=256,
    )
    if feats is None:
        import numpy as _np

        out_h, out_w = crop_hw
        arrs: list[_np.ndarray | None] = [None] * len(pool)
        for fr, si_list in by_frame_s.items():
            img = Image.open(f"{seq_dir}/{fr:06d}.jpg").convert("RGB")
            fw, fh = img.size
            for si in si_list:
                x1, y1, x2, y2 = pool[si][2]
                box = (
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    min(fw, int(round(x2))),
                    min(fh, int(round(y2))),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    box = (0, 0, fw, fh)
                crop = img.crop(box).resize((out_w, out_h), Image.BILINEAR)  # type: ignore[attr-defined]
                arrs[si] = _np.asarray(crop, dtype=_np.uint8).transpose(2, 0, 1)
        device = getattr(extractor, "device", "cuda")
        feats = torch.empty((len(pool), extractor.feature_dim), device=device)
        for s in range(0, len(pool), 256):
            chunk = [a for a in arrs[s : s + 256] if a is not None]
            t = torch.from_numpy(_np.stack(chunk)).to(device).float().div_(255.0)
            feats[s : s + t.shape[0]] = extractor.extract(t)
        feats = F.normalize(feats, dim=1)

    result: dict[int, list[tuple[int, float, torch.Tensor]]] = {}
    for tid, indices in track_crops.items():
        crops = []
        for i, idx in enumerate(indices):
            _, frame_id, _ = pool[idx]
            score = clean_by_id[tid][i].score
            crops.append((frame_id, score, feats[idx]))
        result[tid] = crops
    return result


def _sim_consistency(embs: torch.Tensor) -> float:
    if embs.shape[0] < 2:
        return 1.0
    sim = embs @ embs.T
    n = sim.shape[0]
    return float((sim.sum() - n) / max(1, n * (n - 1)))


def simulate_track_appearance_bank(
    crops_by_track: dict[int, list[tuple[int, float, torch.Tensor]]],
    *,
    k: int = 5,
    min_score: float = 0.45,
    ema_alpha: float = 0.8,
    consistency_threshold: float = 0.82,
) -> dict:
    """Simulate TrackAppearanceBank: score-gated top-K + EMA representative."""
    reps: dict[int, torch.Tensor] = {}
    ema_reps: dict[int, torch.Tensor] = {}
    consistency: dict[int, float] = {}
    clean_ids: set[int] = set()

    for tid, crops in crops_by_track.items():
        filtered = [(f, s, e) for f, s, e in crops if s >= min_score]
        if not filtered:
            continue
        filtered.sort(key=lambda x: (x[1], x[0]), reverse=True)
        top_k = filtered[:k]
        embs = torch.stack([e for _, _, e in top_k])

        ema = None
        for _, _, e in sorted(top_k, key=lambda x: x[0]):
            e_norm = F.normalize(e.float(), dim=0)
            if ema is None:
                ema = e_norm
            else:
                ema = F.normalize(ema_alpha * ema + (1.0 - ema_alpha) * e_norm, dim=0)
        ema_reps[tid] = ema

        mean_rep = F.normalize(embs.mean(dim=0), dim=0)
        reps[tid] = mean_rep
        consistency[tid] = _sim_consistency(embs)
        if consistency[tid] >= consistency_threshold:
            clean_ids.add(tid)

    return {
        "reps": reps,
        "ema_reps": ema_reps,
        "consistency": consistency,
        "clean_ids": clean_ids,
        "n_tracks": len(reps),
    }


def simulate_clean_fifo_bank(
    crops_by_track: dict[int, list[tuple[int, float, torch.Tensor]]],
    *,
    fifo_n: int = 20,
) -> dict:
    """Simulate CleanFifoBank: visclean-gated FIFO-20 (crops already clean)."""
    bank = CleanFifoBank(fifo_n=fifo_n, stride=1, decide_n=5)
    for tid, crops in crops_by_track.items():
        for frame_id, _, emb in crops:
            bank.store(tid, emb, frame_id)

    reps: dict[int, torch.Tensor] = {}
    consistency: dict[int, float] = {}
    clean_ids: set[int] = set()

    for tid in bank.clean_ids():
        samples = bank.samples(tid)
        assert samples is not None
        reps[tid] = F.normalize(samples.mean(dim=0), dim=0)
        consistency[tid] = _sim_consistency(samples)
        clean_ids.add(tid)

    return {
        "reps": reps,
        "consistency": consistency,
        "clean_ids": clean_ids,
        "n_tracks": len(reps),
        "bank": bank,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    ap.add_argument("--fifo-n", type=int, default=20)
    ap.add_argument("--track-k", type=int, default=5)
    ap.add_argument("--track-min-score", type=float, default=0.45)
    ap.add_argument("--track-ema-alpha", type=float, default=0.8)
    ap.add_argument("--consistency-threshold", type=float, default=0.82)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )

    max_crops = max(args.fifo_n, args.track_k, 50)
    all_results = {}

    for seq in args.seqs:
        p = args.substrate / f"{seq}.txt"
        if not p.exists():
            continue
        lines = p.read_text().splitlines()
        seq_dir = str(PROJECT_ROOT / args.data_root / args.split / seq / "img1")

        print(f"\n=== {seq} ===")
        crops_by_track = _extract_all_clean_crops(
            lines,
            seq_dir,
            extractor,
            max_per_track=max_crops,
            cov=args.cov,
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
        )
        print(f"  tracks with clean crops: {len(crops_by_track)}")

        track_bank = simulate_track_appearance_bank(
            crops_by_track,
            k=args.track_k,
            min_score=args.track_min_score,
            ema_alpha=args.track_ema_alpha,
            consistency_threshold=args.consistency_threshold,
        )
        fifo_bank = simulate_clean_fifo_bank(crops_by_track, fifo_n=args.fifo_n)

        print(
            f"  TrackAppearanceBank: {track_bank['n_tracks']} reps, "
            f"{len(track_bank['clean_ids'])} clean_ids"
        )
        print(
            f"  CleanFifoBank:       {fifo_bank['n_tracks']} reps, "
            f"{len(fifo_bank['clean_ids'])} clean_ids"
        )

        common_ids = set(track_bank["reps"]) & set(fifo_bank["reps"])
        cos_sims = []
        cons_diffs = []
        for tid in common_ids:
            r1 = track_bank["ema_reps"][tid]
            r2 = fifo_bank["reps"][tid]
            cos = float(r1 @ r2)
            cos_sims.append(cos)
            c1 = track_bank["consistency"].get(tid, 1.0)
            c2 = fifo_bank["consistency"].get(tid, 1.0)
            cons_diffs.append(c1 - c2)

        cos_arr = np.array(cos_sims) if cos_sims else np.array([])
        cons_arr = np.array(cons_diffs) if cons_diffs else np.array([])

        clean_only_track = track_bank["clean_ids"] - fifo_bank["clean_ids"]
        clean_only_fifo = fifo_bank["clean_ids"] - track_bank["clean_ids"]

        print(
            f"  Rep cosine sim (common {len(common_ids)}): "
            f"mean={cos_arr.mean():.4f} median={np.median(cos_arr):.4f} "
            f"min={cos_arr.min():.4f}"
            if len(cos_arr)
            else "  No common tracks"
        )
        print(
            f"  Consistency diff (track - fifo): "
            f"mean={cons_arr.mean():.4f} std={cons_arr.std():.4f}"
            if len(cons_arr)
            else ""
        )
        print(
            f"  clean_ids only in track: {len(clean_only_track)}, "
            f"only in fifo: {len(clean_only_fifo)}"
        )

        all_results[seq] = {
            "n_tracks": len(crops_by_track),
            "track_bank_reps": track_bank["n_tracks"],
            "track_bank_clean_ids": len(track_bank["clean_ids"]),
            "fifo_bank_reps": fifo_bank["n_tracks"],
            "fifo_bank_clean_ids": len(fifo_bank["clean_ids"]),
            "rep_cosine_mean": float(cos_arr.mean()) if len(cos_arr) else None,
            "rep_cosine_median": float(np.median(cos_arr)) if len(cos_arr) else None,
            "rep_cosine_min": float(cos_arr.min()) if len(cos_arr) else None,
            "consistency_diff_mean": float(cons_arr.mean()) if len(cons_arr) else None,
            "consistency_diff_std": float(cons_arr.std()) if len(cons_arr) else None,
            "clean_ids_only_track": len(clean_only_track),
            "clean_ids_only_fifo": len(clean_only_fifo),
        }

    print("\n=== Overall ===")
    all_cos_means = [
        r["rep_cosine_mean"]
        for r in all_results.values()
        if r["rep_cosine_mean"] is not None
    ]
    if all_cos_means:
        print(f"  Rep cosine mean across seqs: {np.mean(all_cos_means):.4f}")
    all_clean_only_track = sum(r["clean_ids_only_track"] for r in all_results.values())
    all_clean_only_fifo = sum(r["clean_ids_only_fifo"] for r in all_results.values())
    print(f"  clean_ids only in track (total): {all_clean_only_track}")
    print(f"  clean_ids only in fifo (total): {all_clean_only_fifo}")

    result = {
        "schema": "track_bank_fifo_replacement_probe/v1",
        "substrate": str(args.substrate),
        "params": {
            "fifo_n": args.fifo_n,
            "track_k": args.track_k,
            "track_min_score": args.track_min_score,
            "track_ema_alpha": args.track_ema_alpha,
            "consistency_threshold": args.consistency_threshold,
            "cov": args.cov,
        },
        "per_seq": all_results,
        "overall": {
            "rep_cosine_mean": float(np.mean(all_cos_means)) if all_cos_means else None,
            "clean_ids_only_track": all_clean_only_track,
            "clean_ids_only_fifo": all_clean_only_fifo,
        },
    }

    out_path = args.out_json or Path(
        f"results/probe_track_bank_fifo_replacement_{args.substrate.name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult JSON: {out_path}")


if __name__ == "__main__":
    main()
