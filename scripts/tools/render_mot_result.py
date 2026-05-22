"""
Render a MOT-format tracking result onto the source frames and encode a video.

Usage:
    uv run python scripts/tools/render_mot_result.py \
        --result  results/yolo26m_full/MOT17-02-SDP.txt \
        --seq     datasets/MOT17/train/MOT17-02-SDP \
        --output  results/renders/MOT17-02-SDP.mp4 \
        [--gt     datasets/MOT17/train/MOT17-02-SDP/gt/gt.txt] \
        [--start  1]  [--end 600] \
        [--fps    30] [--scale 0.5]
"""

from __future__ import annotations

import argparse
import colorsys
import configparser
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ── colour palette ────────────────────────────────────────────────────────────


def _id_colour(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR colour per ID."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ── MOT txt loader ─────────────────────────────────────────────────────────────


def load_mot_txt(path: Path) -> dict[int, list[tuple]]:
    """Returns {frame_id: [(track_id, x, y, w, h, score), ...]}."""
    data: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            fid, tid = int(row[0]), int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            score = float(row[6]) if len(row) > 6 else -1.0
            data[fid].append((tid, x, y, w, h, score))
    return data


# ── drawing ───────────────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS = 2
_LABEL_PAD = 3


def _draw_box(
    img: np.ndarray,
    tid: int,
    x: float,
    y: float,
    w: float,
    h: float,
    score: float,
    colour: tuple[int, int, int],
    show_score: bool,
) -> None:
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, _THICKNESS)

    label = f"#{tid}" if not show_score or score < 0 else f"#{tid} {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
    bg_y1 = max(y1 - th - baseline - _LABEL_PAD * 2, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + _LABEL_PAD * 2, y1), colour, -1)
    cv2.putText(
        img,
        label,
        (x1 + _LABEL_PAD, y1 - baseline - _LABEL_PAD),
        _FONT,
        _FONT_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ── main ──────────────────────────────────────────────────────────────────────


def render(
    result_txt: Path,
    seq_dir: Path,
    output_mp4: Path,
    gt_txt: Path | None = None,
    start: int = 1,
    end: int | None = None,
    fps: int | None = None,
    scale: float = 1.0,
    show_score: bool = True,
) -> None:
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"img1 not found under {seq_dir}")

    # Read sequence FPS from seqinfo.ini
    if fps is None:
        ini = seq_dir / "seqinfo.ini"
        if ini.exists():
            cfg = configparser.ConfigParser()
            cfg.read(ini)
            fps = cfg.getint("Sequence", "frameRate", fallback=30)
        else:
            fps = 30

    pred = load_mot_txt(result_txt)
    gt = load_mot_txt(gt_txt) if gt_txt else {}

    frame_files = sorted(img_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(img_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg/png frames in {img_dir}")

    if end is None:
        end = len(frame_files)
    frame_files = frame_files[start - 1 : end]

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for idx, fpath in enumerate(frame_files):
            fid = start + idx
            img = cv2.imread(str(fpath))
            if img is None:
                continue

            # Draw GT (faint grey, dashed effect via thin rect)
            for tid, x, y, w, h, _ in gt.get(fid, []):
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Draw predictions
            for tid, x, y, w, h, score in pred.get(fid, []):
                colour = _id_colour(tid)
                _draw_box(img, tid, x, y, w, h, score, colour, show_score)

            # Frame counter
            cv2.putText(
                img,
                f"frame {fid}",
                (8, 24),
                _FONT,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            if scale != 1.0:
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = tmp / f"{fid:06d}.jpg"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Encode with ffmpeg (h264_nvenc if available, else libx264)
        for codec in ("h264_nvenc", "libx264"):
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                str(start),
                "-i",
                str(tmp / "%06d.jpg"),
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
            ]
            if codec == "h264_nvenc":
                cmd += ["-preset", "p4", "-cq", "20"]
            else:
                cmd += ["-crf", "20", "-preset", "fast"]
            cmd.append(str(output_mp4))

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                print(f"✅ Encoded with {codec}: {output_mp4}")
                break
            else:
                print(f"⚠️  {codec} failed, trying next...")
        else:
            print("❌ All codecs failed.")
            print(result.stderr.decode())


def main() -> None:
    ap = argparse.ArgumentParser(description="Render MOT tracking result to video.")
    ap.add_argument("--result", required=True, type=Path, help="MOT txt result file")
    ap.add_argument(
        "--seq", required=True, type=Path, help="Sequence root dir (contains img1/)"
    )
    ap.add_argument("--output", required=True, type=Path, help="Output mp4 path")
    ap.add_argument(
        "--gt", default=None, type=Path, help="GT txt (optional, drawn in grey)"
    )
    ap.add_argument("--start", default=1, type=int, help="First frame (1-indexed)")
    ap.add_argument("--end", default=None, type=int, help="Last frame (inclusive)")
    ap.add_argument(
        "--fps", default=None, type=int, help="Output FPS (reads seqinfo.ini if unset)"
    )
    ap.add_argument(
        "--scale", default=0.5, type=float, help="Resize scale (default 0.5 for speed)"
    )
    ap.add_argument("--no-score", action="store_true", help="Hide score labels")
    args = ap.parse_args()

    render(
        result_txt=args.result,
        seq_dir=args.seq,
        output_mp4=args.output,
        gt_txt=args.gt,
        start=args.start,
        end=args.end,
        fps=args.fps,
        scale=args.scale,
        show_score=not args.no_score,
    )


if __name__ == "__main__":
    main()
