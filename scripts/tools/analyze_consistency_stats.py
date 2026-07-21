"""GT-level statistics for the cross-frame consistency term (route 3).

Pre-flight for run_v14replica_consistency.sh: characterises how many matched
track-id pairs the consistency loss actually penalises per adjacent-frame pair,
per clip, and per batch step — before spending GPU hours. No images, no GPU:
reads only the dataset's parsed GT (``_gt``) and clip index (``_clips``).

Usage:
    .venv/bin/python scripts/tools/analyze_consistency_stats.py \
        --data-root datasets/MOT17 --clip-len 3 --clip-stride 6 --batch-size 4
"""
# status: diagnostic

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.dataset import MOT17TemporalClip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--clip-len", type=int, default=3)
    ap.add_argument("--clip-stride", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--detector", default="SDP")
    ap.add_argument("--seqs", default="", help="comma-separated; empty = all train")
    args = ap.parse_args()

    seqs = [s for s in args.seqs.split(",") if s] or None
    ds = MOT17TemporalClip(
        data_root=args.data_root,
        split="train",
        clip_len=args.clip_len,
        stride=args.clip_stride,
        seqs=seqs,
        detector=args.detector,
        preload_to_ram=False,
        load_images=False,
    )

    frame_lists = ds._frame_lists  # seq -> list[Path]
    gt = ds._gt  # seq -> {fid: (boxes, ids)}

    def ids_at(seq: str, idx: int) -> set[int]:
        fid = int(frame_lists[seq][idx].stem)
        entry = gt[seq].get(fid)
        if entry is None or entry[1].numel() == 0:
            return set()
        return set(entry[1].tolist())

    # Per adjacent-frame-pair match counts, globally and per sequence.
    pair_matches: list[int] = []
    per_seq: dict[str, list[int]] = {}
    n_prev: list[int] = []
    n_cur: list[int] = []

    for seq, start in ds._clips:
        bucket = per_seq.setdefault(seq, [])
        for t in range(1, args.clip_len):
            prev = ids_at(seq, start + t - 1)
            cur = ids_at(seq, start + t)
            m = len(prev & cur)
            pair_matches.append(m)
            bucket.append(m)
            n_prev.append(len(prev))
            n_cur.append(len(cur))

    n_pairs = len(pair_matches)
    n_zero = sum(1 for m in pair_matches if m == 0)
    T = args.clip_len
    B = args.batch_size

    def pct(xs: list[int], q: float) -> float:
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * len(s)))]

    print(f"\n{'=' * 64}")
    print(f"Consistency-term GT statistics  (clip_len={T}, stride={args.clip_stride})")
    print(f"{'=' * 64}")
    print(f"clips                : {len(ds._clips)}")
    print(f"adjacent-frame pairs : {n_pairs}  ({T - 1} per clip)")
    print(f"sequences            : {len(per_seq)}")
    print("\n--- matched track-id pairs per adjacent-frame pair (= L2 terms) ---")
    print(f"mean                 : {statistics.mean(pair_matches):.1f}")
    print(f"median               : {statistics.median(pair_matches):.0f}")
    print(
        f"p10 / p90            : {pct(pair_matches, 0.10)} / {pct(pair_matches, 0.90)}"
    )
    print(f"min / max            : {min(pair_matches)} / {max(pair_matches)}")
    print(
        f"pairs with 0 matches : {n_zero}  ({100 * n_zero / n_pairs:.1f}%  <- term inactive)"
    )
    print("\n--- box density (continuity check) ---")
    mean_prev = statistics.mean(n_prev)
    mean_match = statistics.mean(pair_matches)
    print(f"mean boxes/frame     : {mean_prev:.1f}")
    print(
        f"mean matched/frame   : {mean_match:.1f}  ({100 * mean_match / max(mean_prev, 1e-9):.1f}% of boxes carried)"
    )
    print("\n--- per-batch-step term count (mean) ---")
    print(f"B={B}, (T-1)={T - 1}: ~{mean_match * (T - 1) * B:.0f} L2 terms/step")
    print("\n--- per sequence: mean matched pairs ---")
    for seq in sorted(per_seq):
        ms = per_seq[seq]
        z = 100 * sum(1 for m in ms if m == 0) / len(ms)
        print(
            f"  {seq:18s} mean={statistics.mean(ms):5.1f}  median={statistics.median(ms):3.0f}  zero={z:4.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
