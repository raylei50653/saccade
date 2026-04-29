from pathlib import Path
from perception.eval.runner import run_eval

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "models/yolo/yolo26s_batch6.engine"
REID_ENGINE = ROOT / "models/embedding/google_siglip2-base-patch16-224.engine"
DATA_ROOT = ROOT / "datasets/MOT17"
SEQ = "MOT17-04-SDP"


def main() -> None:
    print("🚀 Starting End-to-End Latency Benchmark...")
    print(f"Sequence: {SEQ}")

    run_eval(
        engine=str(ENGINE),
        output="results/latency_benchmark",
        data_root=str(DATA_ROOT),
        split="train",
        sequences=SEQ,
        max_frames=100,
        conf_threshold=0.25,
        reid_mode="semantic",
        warmup_frames=10,
        track_thresh=0.1,
        high_thresh=0.5,
        match_thresh=0.8,
        profile_stages=True,
        reid_engine_path=str(REID_ENGINE),
    )


if __name__ == "__main__":
    main()
