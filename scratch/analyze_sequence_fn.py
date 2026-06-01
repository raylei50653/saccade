import sys
from pathlib import Path

# Setup paths to import saccade modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from saccade.perception.eval.metrics import _evaluate_single_sequence  # noqa: E402

data_root = "datasets/MOT17"
split = "train"
output_dir = "results/MOT17_eval"
sequences = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

print(
    "| Sequence | Total Objects (GT Bboxes) | Detections (TP) | False Negatives (FN) | Recall | MOTA | FP | IDs |"
)
print("|---|---|---|---|---|---|---|---|")

total_objs = 0
total_fns = 0

for seq in sequences:
    gt_path = f"{data_root}/{split}/{seq}/gt/gt.txt"
    ts_path = f"{output_dir}/{seq}.txt"

    if not Path(gt_path).exists() or not Path(ts_path).exists():
        print(f"Skipping {seq}: files not found")
        continue

    res = _evaluate_single_sequence(seq, gt_path, ts_path)

    num_objs = res["num_objects"]
    num_dets = res["num_detections"]
    num_fns = res["num_misses"]
    num_fps = res["num_false_positives"]
    num_ids = res["num_switches"]

    recall = num_dets / max(num_objs, 1) * 100
    mota = (1.0 - (num_fns + num_fps + num_ids) / max(num_objs, 1)) * 100

    total_objs += num_objs
    total_fns += num_fns

    print(
        f"| {seq} | {num_objs:,} | {num_dets:,} | {num_fns:,} | {recall:.1f}% | {mota:.1f}% | {num_fps:,} | {num_ids:,} |"
    )

overall_recall = (total_objs - total_fns) / max(total_objs, 1) * 100
print(
    f"| **OVERALL** | **{total_objs:,}** | **{total_objs - total_fns:,}** | **{total_fns:,}** | **{overall_recall:.1f}%** | - | - | - |"
)
