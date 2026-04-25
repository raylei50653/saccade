import argparse
import configparser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from saccade_tracking_ext import GPUByteTracker


def load_seq_info(seq_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(seq_path / "seqinfo.ini")
    return {
        "name": config.get("Sequence", "name"),
        "imDir": config.get("Sequence", "imDir"),
        "seqLength": int(config.get("Sequence", "seqLength")),
        "imWidth": int(config.get("Sequence", "imWidth")),
        "imHeight": int(config.get("Sequence", "imHeight")),
        "fps": int(config.get("Sequence", "frameRate")),
    }


def load_public_detections(det_path: Path, min_conf: float) -> dict[int, np.ndarray]:
    detections_by_frame: dict[int, list[list[float]]] = defaultdict(list)
    if not det_path.exists():
        return {}

    dets = np.loadtxt(det_path, delimiter=",", ndmin=2)
    for row in dets:
        frame_id = int(row[0])
        x1 = float(row[2])
        y1 = float(row[3])
        w = float(row[4])
        h = float(row[5])
        score = float(row[6])
        if score < min_conf or w <= 0.0 or h <= 0.0:
            continue
        detections_by_frame[frame_id].append([x1, y1, x1 + w, y1 + h, score])

    return {
        frame_id: np.asarray(rows, dtype=np.float32)
        for frame_id, rows in detections_by_frame.items()
    }


def run_eval(
    data_root: str,
    split: str,
    output_dir: str,
    min_conf: float,
) -> None:
    data_path = Path(data_root) / split
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted(p for p in data_path.iterdir() if p.name.startswith("MOT17-"))
    print(f"Found {len(sequences)} MOT17 {split} sequences. Using public detections.")

    for seq_path in sequences:
        info = load_seq_info(seq_path)
        dets_by_frame = load_public_detections(seq_path / "det" / "det.txt", min_conf)
        tracker = GPUByteTracker(max_objects=2048)
        result_lines: list[str] = []

        print(f"Processing {seq_path.name}...")
        for frame_id in range(1, info["seqLength"] + 1):
            img_path = seq_path / info["imDir"] / f"{frame_id:06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            img_gpu = torch.from_numpy(img).to("cuda")
            img_ptr = img_gpu.data_ptr()

            frame_dets = dets_by_frame.get(frame_id)
            if frame_dets is None or frame_dets.size == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32, device="cuda")
                scores = torch.empty((0,), dtype=torch.float32, device="cuda")
                classes = torch.empty((0,), dtype=torch.int32, device="cuda")
            else:
                boxes_np = frame_dets[:, :4].copy()
                scores_np = frame_dets[:, 4].copy()
                classes_np = np.zeros((frame_dets.shape[0],), dtype=np.int32)

                boxes = torch.from_numpy(boxes_np).to("cuda")
                scores = torch.from_numpy(scores_np).to("cuda")
                classes = torch.from_numpy(classes_np).to("cuda")

            current_stream = torch.cuda.current_stream().cuda_stream
            track_results = tracker.update(
                boxes.data_ptr(),
                scores.data_ptr(),
                classes.data_ptr(),
                boxes.size(0),
                current_stream,
                0,
                img_ptr,
                w,
                h,
            )

            for res in track_results:
                result_lines.append(
                    f"{frame_id},{res.obj_id},{res.x1:.2f},{res.y1:.2f},"
                    f"{res.x2 - res.x1:.2f},{res.y2 - res.y1:.2f},"
                    f"{res.score:.4f},-1,-1,-1"
                )

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id}/{info['seqLength']}...")

        result_file = output_root / f"{seq_path.name}.txt"
        result_file.write_text("\n".join(result_lines))
        print(f"Saved {seq_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="results/MOT17_public_det_tracker")
    parser.add_argument("--min-conf", type=float, default=0.0)
    args = parser.parse_args()
    run_eval(args.data_root, args.split, args.output, args.min_conf)
