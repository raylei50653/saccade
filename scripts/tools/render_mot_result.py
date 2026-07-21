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
# status: stable

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
    # Adaptively scale font size and thickness based on image height
    ref_h = img.shape[0]
    scale_factor = ref_h / 1080.0

    font_scale = 0.55 * scale_factor
    box_thickness = max(1, int(round(2.0 * scale_factor)))
    text_thickness = max(1, int(round(1.0 * scale_factor)))
    label_pad = max(1, int(round(4.0 * scale_factor)))

    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, box_thickness)

    label = f"#{tid}" if not show_score or score < 0 else f"#{tid} {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
    bg_y1 = max(y1 - th - baseline - label_pad * 2, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + label_pad * 2, y1), colour, -1)
    cv2.putText(
        img,
        label,
        (x1 + label_pad, y1 - baseline - label_pad),
        _FONT,
        font_scale,
        (255, 255, 255),
        text_thickness,
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
    trail_len: int = 30,
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

    # Track center history: {track_id: list[(cx, cy)]}
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for idx, fpath in enumerate(frame_files):
            fid = start + idx
            img = cv2.imread(str(fpath))
            if img is None:
                continue

            # Update track centers history
            if trail_len > 0:
                for tid, x, y, w, h, _ in pred.get(fid, []):
                    cx = int(round(x + w / 2))
                    cy = int(round(y + h / 2))
                    track_history[tid].append((cx, cy))
                    if len(track_history[tid]) > trail_len:
                        track_history[tid].pop(0)

            # Draw GT (faint grey, dashed effect via thin rect)
            for tid, x, y, w, h, _ in gt.get(fid, []):
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Draw trajectory trails
            if trail_len > 0:
                ref_h = img.shape[0]
                scale_factor = ref_h / 1080.0
                trail_thickness = max(1, int(round(2.5 * scale_factor)))

                for tid, _, _, _, _, _ in pred.get(fid, []):
                    pts = track_history[tid]
                    if len(pts) > 1:
                        colour = _id_colour(tid)
                        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            img,
                            [pts_arr],
                            isClosed=False,
                            color=colour,
                            thickness=trail_thickness,
                            lineType=cv2.LINE_AA,
                        )

            # Draw predictions
            for tid, x, y, w, h, score in pred.get(fid, []):
                colour = _id_colour(tid)
                _draw_box(img, tid, x, y, w, h, score, colour, show_score)

            # Frame counter
            ref_h = img.shape[0]
            scale_factor = ref_h / 1080.0
            counter_scale = 0.75 * scale_factor
            counter_thick = max(1, int(round(1.0 * scale_factor)))
            counter_x = int(8 * scale_factor)
            counter_y = int(28 * scale_factor)

            cv2.putText(
                img,
                f"frame {fid}",
                (counter_x, counter_y),
                _FONT,
                counter_scale,
                (220, 220, 220),
                counter_thick,
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
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=out_range=tv",
                "-color_range",
                "tv",
                "-profile:v",
                "main",
                "-bf",
                "0",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
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
    ap.add_argument(
        "--trail-len",
        default=30,
        type=int,
        nargs="?",
        const=30,
        help="Length of trajectory trail in frames (default 30, 0 to disable)",
    )
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
        trail_len=args.trail_len,
    )


if __name__ == "__main__":
    main()
