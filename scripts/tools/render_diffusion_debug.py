#!/usr/bin/env python3
"""Render bidirectional relink debug events emitted by ``mot17.py``.

This tool intentionally does not reconstruct tracker motion. It visualizes the
raw bridge-attempt data written by the GPU tracker:

    datasets/<split>/<sequence>_raw_data.npy

The MOT result and ``_global_id_map.txt`` from ``mot17.py`` are used only to draw
boxes and map raw tracker-local IDs to the global IDs shown in the result video.
For current 18-column raw dumps, the event frame is inferred from the candidate
track's ``cand_hit``-th emitted box. If a future raw dump appends an event-frame
column, this renderer uses it directly.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class Box:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass(frozen=True)
class RawEvent:
    event_frame: int
    start_frame: int
    gap: int
    bridge: float
    kalman_d2: float
    dir_cos: float
    speed_mps: float
    outcome: int
    source: int
    lost_id: int
    cand_id: int
    cand_hit: int
    lost_hit: int
    fwd_eucl: float
    bwd_eucl: float
    bwd_maha: float
    lost_mid_x: float
    lost_mid_y: float
    cand_mid_x: float
    cand_mid_y: float


def id_colour(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = cv2.cvtColor(np.uint8([[[int(hue * 179), 217, 242]]]), cv2.COLOR_HSV2BGR)[
        0, 0
    ]
    return int(r), int(g), int(b)


def load_mot(path: Path) -> tuple[dict[int, list[Box]], dict[int, list[Box]]]:
    by_frame: dict[int, list[Box]] = defaultdict(list)
    by_track: dict[int, list[Box]] = defaultdict(list)
    with path.open() as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            box = Box(
                frame=int(row[0]),
                track_id=int(row[1]),
                x=float(row[2]),
                y=float(row[3]),
                w=float(row[4]),
                h=float(row[5]),
                score=float(row[6]) if len(row) > 6 else -1.0,
            )
            by_frame[box.frame].append(box)
            by_track[box.track_id].append(box)
    for boxes in by_track.values():
        boxes.sort(key=lambda b: b.frame)
    return by_frame, by_track


def parse_global_id_map(path: Path, sequence: str) -> dict[int, int]:
    local_to_global: dict[int, int] = {}
    if not path.exists():
        return local_to_global
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3 or parts[0] != sequence:
                continue
            local_id = int(parts[1].split("=", 1)[1])
            global_id = int(parts[2].split("=", 1)[1])
            local_to_global[local_id] = global_id
    return local_to_global


def parse_id_map(raw: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        mapping[int(left.strip())] = int(right.strip())
    return mapping


def read_fps(seq_dir: Path) -> int:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        return 30
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    return cfg.getint("Sequence", "frameRate", fallback=30)


def infer_seq_len(seq_dir: Path) -> int:
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        length = cfg.getint("Sequence", "seqLength", fallback=0)
        if length > 0:
            return length
    img_dir = seq_dir / "img1"
    return len(sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png")))


def infer_event_frame(cand_boxes: list[Box], cand_hit: int, fallback: int) -> int:
    if cand_boxes:
        idx = max(0, min(len(cand_boxes) - 1, cand_hit - 1))
        return cand_boxes[idx].frame
    return fallback


def load_raw_events(
    raw_path: Path,
    by_track: dict[int, list[Box]],
    local_to_global: dict[int, int],
    lost_id_map: dict[int, int],
    cand_id_map: dict[int, int],
    start_frame: int,
    end_frame: int,
) -> list[RawEvent]:
    arr = np.load(raw_path)
    if arr.ndim != 2 or arr.shape[1] < 18:
        raise ValueError(f"Expected raw data with at least 18 columns, got {arr.shape}")

    events: list[RawEvent] = []
    for row in arr:
        raw_lost = int(row[7])
        raw_cand = int(row[8])
        lost_id = lost_id_map.get(raw_lost, local_to_global.get(raw_lost, raw_lost))
        cand_id = cand_id_map.get(raw_cand, local_to_global.get(raw_cand, raw_cand))
        cand_hit = int(row[9])
        gap = int(row[0])

        if arr.shape[1] >= 19:
            event_frame = int(row[18])
        else:
            event_frame = infer_event_frame(
                by_track.get(cand_id, []), cand_hit, end_frame
            )
        window_start = max(start_frame, event_frame - gap)
        window_end = min(end_frame, event_frame)
        if window_end < start_frame or window_start > end_frame:
            continue

        events.append(
            RawEvent(
                event_frame=event_frame,
                start_frame=window_start,
                gap=gap,
                bridge=float(row[1]),
                kalman_d2=float(row[2]),
                dir_cos=float(row[3]),
                speed_mps=float(row[4]),
                outcome=int(row[5]),
                source=int(row[6]),
                lost_id=lost_id,
                cand_id=cand_id,
                cand_hit=cand_hit,
                lost_hit=int(row[10]),
                fwd_eucl=float(row[11]),
                bwd_eucl=float(row[12]),
                bwd_maha=float(row[13]),
                lost_mid_x=float(row[14]),
                lost_mid_y=float(row[15]),
                cand_mid_x=float(row[16]),
                cand_mid_y=float(row[17]),
            )
        )
    events.sort(key=lambda e: (e.event_frame, e.source, e.lost_id, e.cand_id))
    return events


def draw_label(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    thickness = max(1, int(round(scale * 2)))
    pad = max(3, int(round(scale * 5)))
    font_scale = 0.55 * scale
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin
    x = max(0, min(x, img.shape[1] - tw - pad * 2 - 1))
    y = max(th + base + pad * 2 + 1, min(y, img.shape[0] - 1))
    cv2.rectangle(img, (x, y - th - base - pad * 2), (x + tw + pad * 2, y), colour, -1)
    cv2.putText(
        img,
        text,
        (x + pad, y - base - pad),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_box(img: np.ndarray, box: Box, selected: bool, scale: float) -> None:
    colour = id_colour(box.track_id)
    thickness = max(1, int(round((3 if selected else 2) * scale)))
    x1, y1 = int(round(box.x)), int(round(box.y))
    x2, y2 = int(round(box.x + box.w)), int(round(box.y + box.h))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    label = f"#{box.track_id}"
    if selected:
        label += " raw"
    draw_label(img, label, (x1, max(y1, 28)), colour, scale)


def draw_event(img: np.ndarray, event: RawEvent, fid: int, scale: float) -> None:
    lost_colour = id_colour(event.lost_id)
    cand_colour = id_colour(event.cand_id)
    line_colour = (60, 220, 220) if event.outcome == 0 else (40, 40, 230)
    lost_pt = (int(round(event.lost_mid_x)), int(round(event.lost_mid_y)))
    cand_pt = (int(round(event.cand_mid_x)), int(round(event.cand_mid_y)))

    cv2.line(
        img, lost_pt, cand_pt, line_colour, max(1, int(round(2 * scale))), cv2.LINE_AA
    )
    cv2.circle(
        img, lost_pt, max(4, int(round(7 * scale))), lost_colour, -1, cv2.LINE_AA
    )
    cv2.circle(
        img, lost_pt, max(1, int(round(2 * scale))), (255, 255, 255), -1, cv2.LINE_AA
    )
    cv2.circle(
        img, cand_pt, max(4, int(round(7 * scale))), cand_colour, -1, cv2.LINE_AA
    )
    cv2.circle(img, cand_pt, max(1, int(round(2 * scale))), (0, 0, 0), -1, cv2.LINE_AA)

    source = "archive" if event.source == 1 else "live"
    label = (
        f"{source} #{event.lost_id}<->#{event.cand_id} "
        f"gap={event.gap} bridge={event.bridge:.2f} out={event.outcome} "
        f"event={event.event_frame}"
    )
    mid = (
        int(round((event.lost_mid_x + event.cand_mid_x) * 0.5)),
        int(round((event.lost_mid_y + event.cand_mid_y) * 0.5)),
    )
    draw_label(img, label, (mid[0] + 8, mid[1] + 18), line_colour, scale)


def encode_frames(tmp: Path, output_mp4: Path, start: int, fps: int) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    last_error = b""
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
            print(f"Encoded with {codec}: {output_mp4}")
            return
        last_error = result.stderr
        print(f"{codec} failed, trying next...")
    raise RuntimeError(last_error.decode(errors="replace"))


def render(args: argparse.Namespace) -> None:
    eval_output = Path(args.eval_output)
    seq_dir = Path(args.data_root) / args.split / args.sequence
    result_path = eval_output / f"{args.sequence}.txt"
    raw_path = Path(args.data_root) / args.split / f"{args.sequence}_raw_data.npy"
    if args.result:
        result_path = Path(args.result)
    if args.seq:
        seq_dir = Path(args.seq)
    if args.raw_data_npy:
        raw_path = Path(args.raw_data_npy)

    start = args.start if args.start is not None else 1
    end = args.end if args.end is not None else infer_seq_len(seq_dir)
    output = (
        Path(args.output)
        if args.output
        else eval_output / "renders" / f"{args.sequence}_diffusion_debug_raw.mp4"
    )

    by_frame, by_track = load_mot(result_path)
    local_to_global = parse_global_id_map(
        eval_output / "_global_id_map.txt", args.sequence
    )
    lost_id_map = parse_id_map(args.raw_lost_id_map)
    cand_id_map = parse_id_map(args.raw_candidate_id_map)
    events = load_raw_events(
        raw_path,
        by_track,
        local_to_global,
        lost_id_map,
        cand_id_map,
        start,
        end,
    )
    selected_ids = {e.lost_id for e in events} | {e.cand_id for e in events}

    events_by_frame: dict[int, list[RawEvent]] = defaultdict(list)
    for event in events:
        for fid in range(event.start_frame, event.event_frame + 1):
            if start <= fid <= end:
                events_by_frame[fid].append(event)

    print(
        f"[debug] events={len(events)} active_frames={len(events_by_frame)} "
        f"result={result_path} raw={raw_path}"
    )

    img_dir = seq_dir / "img1"
    frame_files = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {img_dir}")

    fps = args.fps or read_fps(seq_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for fid in range(start, end + 1):
            if fid - 1 >= len(frame_files):
                break
            img = cv2.imread(str(frame_files[fid - 1]))
            if img is None:
                continue
            draw_scale = img.shape[0] / 1080.0

            for box in by_frame.get(fid, []):
                if args.selected_only and box.track_id not in selected_ids:
                    continue
                draw_box(img, box, box.track_id in selected_ids, draw_scale)

            for event in events_by_frame.get(fid, []):
                draw_event(img, event, fid, draw_scale)

            draw_label(img, f"frame {fid}", (12, 42), (30, 30, 30), draw_scale)
            if args.scale != 1.0:
                img = cv2.resize(
                    img,
                    (
                        int(round(img.shape[1] * args.scale)),
                        int(round(img.shape[0] * args.scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(tmp / f"{fid:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

        encode_frames(tmp, output, start, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output", type=Path, default=Path("results/demo"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--split", default="demo")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--seq", type=Path, default=None)
    parser.add_argument("--raw-data-npy", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument(
        "--raw-lost-id-map",
        default="",
        help="Optional raw local lost-id to rendered-id map, e.g. '3:15'.",
    )
    parser.add_argument(
        "--raw-candidate-id-map",
        default="",
        help="Optional raw local candidate-id to rendered-id map, e.g. '8:18'.",
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
