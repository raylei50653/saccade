# mypy: ignore-errors
from ultralytics import YOLO
import time
from pathlib import Path
import argparse


def run_official_eval() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo/yolo11n.pt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="results/MOT17_ultralytics_eval")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Ultralytics inference size.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument(
        "--detector",
        choices=["SDP", "DPM", "FRCNN"],
        default=None,
        help="Filter sequences by detector suffix.",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    split_root = Path(args.data_root) / args.split
    seqs = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
    if args.detector:
        seqs = [s for s in seqs if s.endswith(f"-{args.detector}")]
    train_root = split_root

    fps_records = []

    for seq in seqs:
        seq_path = train_root / seq
        img_dir = seq_path / "img1"
        if not img_dir.exists():
            continue

        results_lines = []
        img_files = sorted(list(img_dir.glob("*.jpg")))
        num_frames = len(img_files)

        print(f"🚀 Running Official Ultralytics Tracking on {seq}...")
        start_time = time.time()

        results = model.track(
            source=str(img_dir),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            persist=True,
            stream=True,
            verbose=False,
            device="cuda:0",
            classes=[0],
        )

        for i, res in enumerate(results):
            frame_id = i + 1
            if res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                for j in range(len(ids)):
                    x1, y1, x2, y2 = boxes[j]
                    w, h = x2 - x1, y2 - y1
                    results_lines.append(
                        f"{frame_id},{ids[j]},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{scores[j]:.4f},-1,-1,-1"
                    )

        elapsed = time.time() - start_time
        fps = num_frames / elapsed if elapsed > 0 else 0
        mean_ms = (elapsed / num_frames) * 1000 if num_frames > 0 else 0

        # 儲存個別結果
        (output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        fps_records.append(
            f"{seq}\tfps={fps:.2f}\tmean_ms={mean_ms:.2f}\tframes={num_frames}"
        )
        print(f"✅ Finished {seq}: {fps:.2f} FPS")

    # 儲存 FPS 總表
    (output_root / "_fps_summary.txt").write_text("\n".join(fps_records))


if __name__ == "__main__":
    run_official_eval()
