"""
cache_gt_tracks.py — Phase 1 GT Oracle Cache Generator

從 MOT17 gt.txt 讀取 GT 軌跡，轉換成 TrackerGateInput 相容格式並
存成 per-frame .pt 檔。供 train_conditioned.py Phase 1 或消融實驗使用。

輸出格式（每幀一個 .pt）：
    {out_dir}/{seq}/frame_{fid:06d}.pt
    內容: {
        'boxes':  Tensor(N, 4)  [x1,y1,x2,y2] 在縮放後圖片的絕對像素座標
        'scores': Tensor(N,)    全為 1.0（GT 軌跡視為完美信心度）
        'ids':    Tensor(N,)    GT track ID
    }

沒有 velocities（GT 軌跡無 Kalman 速度），符合 TrackerGateInput.from_boxes_scores()
的最簡化介面。Phase 2 的 ByteTrack 快取由 cache_tracker_states.py 處理。

使用方式：
    uv run scripts/tools/cache_gt_tracks.py \\
        --data-root /path/to/MOT17 \\
        --out-dir datasets/mot17_gt_cache \\
        --img-size 640

驗證已存在的快取：
    uv run scripts/tools/cache_gt_tracks.py \\
        --data-root /path/to/MOT17 \\
        --out-dir datasets/mot17_gt_cache \\
        --verify
"""

from __future__ import annotations

import argparse
import configparser
import csv
import sys
from pathlib import Path

import torch


def _get_seq_scale(seq_dir: Path, img_size: int) -> tuple[float, float]:
    """讀 seqinfo.ini 取得 GT → resized 座標的 scale 係數。"""
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        orig_h = int(cfg["Sequence"]["imHeight"])
        orig_w = int(cfg["Sequence"]["imWidth"])
    else:
        # fallback: 讀第一張圖的 shape
        import torchvision.io as tv_io

        frames = sorted((seq_dir / "img1").glob("*.jpg"))
        img = tv_io.read_image(str(frames[0]))
        _, orig_h, orig_w = img.shape

    return img_size / orig_h, img_size / orig_w


def _parse_gt(gt_file: Path) -> dict[int, tuple[list, list, list]]:
    """
    Parse gt.txt → {frame_id: (boxes_xyxy, scores, ids)}.

    Only conf=1, class=1 (pedestrian), visibility≥0.1 rows are kept.
    """
    per_frame: dict[int, tuple[list, list, list]] = {}
    with gt_file.open() as f:
        for row in csv.reader(f):
            fid = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            conf = int(row[6]) if len(row) > 6 else 1
            cls = int(row[7]) if len(row) > 7 else 1
            vis = float(row[8]) if len(row) > 8 else 1.0

            if conf != 1 or cls != 1 or vis < 0.1:
                continue

            if fid not in per_frame:
                per_frame[fid] = ([], [], [])
            per_frame[fid][0].append([x, y, x + w, y + h])
            per_frame[fid][1].append(1.0)
            per_frame[fid][2].append(tid)
    return per_frame


def cache_sequence(
    seq_dir: Path,
    out_seq_dir: Path,
    img_size: int,
    overwrite: bool,
) -> int:
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        print(f"  [SKIP] {seq_dir.name}: no gt.txt")
        return 0

    scale_h, scale_w = _get_seq_scale(seq_dir, img_size)
    per_frame = _parse_gt(gt_file)
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for fid, (boxes, scores, ids) in per_frame.items():
        out_path = out_seq_dir / f"frame_{fid:06d}.pt"
        if out_path.exists() and not overwrite:
            n_saved += 1
            continue

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        if boxes_t.numel() > 0:
            boxes_t[:, [0, 2]] *= scale_w
            boxes_t[:, [1, 3]] *= scale_h

        torch.save(
            {
                "boxes": boxes_t,
                "scores": torch.tensor(scores, dtype=torch.float32),
                "ids": torch.tensor(ids, dtype=torch.int64),
            },
            out_path,
        )
        n_saved += 1

    return n_saved


def verify_cache(out_dir: Path, seq_dir: Path) -> bool:
    """Spot-check that cache exists and tensors have expected shapes."""
    seq_name = seq_dir.name
    cache_seq = out_dir / seq_name
    if not cache_seq.exists():
        print(f"  [MISSING] {seq_name}")
        return False

    pts = sorted(cache_seq.glob("frame_*.pt"))
    if not pts:
        print(f"  [EMPTY] {seq_name}")
        return False

    # Check first and last frame
    for pt in [pts[0], pts[-1]]:
        d = torch.load(pt, map_location="cpu", weights_only=True)
        boxes = d["boxes"]
        scores = d["scores"]
        ids = d["ids"]
        assert boxes.ndim == 2 and boxes.shape[1] == 4, f"bad boxes shape {boxes.shape}"
        assert scores.ndim == 1, f"bad scores shape {scores.shape}"
        assert ids.ndim == 1, f"bad ids shape {ids.shape}"
        assert boxes.shape[0] == scores.shape[0] == ids.shape[0]
        # Boxes should be in resized pixel space (≥ 0, reasonable max for 640-ish img)
        # Note: MOT17 gt.txt legitimately contains boxes with negative coords
        # (persons partially outside frame edges). No bounds check needed.

    print(
        f"  [OK] {seq_name}: {len(pts)} frames  "
        f"boxes[0]={d['boxes'].shape}  max_x={d['boxes'][:, 2].max():.0f}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache GT tracks for Option D Phase 1")
    parser.add_argument(
        "--data-root", required=True, help="MOT17 root (contains train/)"
    )
    parser.add_argument("--out-dir", required=True, help="Output cache directory")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--detector", default="SDP", help="MOT17 detector suffix (SDP, DPM, FRCNN)"
    )
    parser.add_argument(
        "--seqs", default="", help="Comma-separated sequence names (empty = all)"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing cache, do not regenerate",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    split_dir = data_root / args.split

    if not split_dir.exists():
        print(f"[Error] Split dir not found: {split_dir}")
        sys.exit(1)

    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(",")]
    else:
        seq_names = sorted(
            d.name
            for d in split_dir.iterdir()
            if d.is_dir() and d.name.endswith(f"-{args.detector}")
        )

    if not seq_names:
        print(f"[Error] No sequences found in {split_dir} with suffix -{args.detector}")
        sys.exit(1)

    print(f"Sequences ({len(seq_names)}): {seq_names}")
    print(f"img_size={args.img_size}  out_dir={out_dir}\n")

    all_ok = True
    for seq_name in seq_names:
        seq_dir = split_dir / seq_name
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_name}: directory not found")
            continue

        if args.verify:
            all_ok &= verify_cache(out_dir, seq_dir)
        else:
            n = cache_sequence(
                seq_dir, out_dir / seq_name, args.img_size, args.overwrite
            )
            print(f"  {seq_name}: {n} frames cached")

    if args.verify:
        print("\n[PASS]" if all_ok else "\n[FAIL] Some sequences missing or malformed")
        sys.exit(0 if all_ok else 1)
    else:
        print(f"\nDone. Cache at: {out_dir}")


if __name__ == "__main__":
    main()
