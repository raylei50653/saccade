#!/usr/bin/env bash
# status: experiment
# Full v14 lineage rebuild with a teacher trained through epoch 30:
# yolo26s -> gated teacher e30 -> cache -> legacy N=1 distill -> legacy N=1
# GT warm-start -> fixed N=16 GT adaptation -> detector/tracking evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-datasets/MOT17}"
YOLO="${YOLO:-models/yolo/yolo26s.pt}"
SEQS="${SEQS:-MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP}"
SEED="${SEED:-42}"
RUN_PREFIX="${RUN_PREFIX:-v14_full_e30}"
PRINT_ONLY=0

if [[ "${1:-}" == "--print-only" ]]; then
  PRINT_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--print-only]" >&2
  exit 2
fi

TEACHER_RUN="runs/gated_det_${RUN_PREFIX}"
TEACHER="${TEACHER_RUN}/epoch_0030.ckpt"
CACHE="runs/mamba_teacher_cache_${RUN_PREFIX}"
DISTILL="runs/mamba_distill_${RUN_PREFIX}_n1"
GT_N1="runs/mamba_gt_${RUN_PREFIX}_n1"
GT_N16="runs/mamba_gt_${RUN_PREFIX}_n16"
REPORT="results/mamba_${RUN_PREFIX}_n16_recall_02.json"
RESULTS="results/mamba_${RUN_PREFIX}_n16"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$PRINT_ONLY" -eq 0 ]]; then
    "$@"
  fi
}

checkpoint_epoch() {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
actual = int(checkpoint.get("epoch", -1))
if actual != expected:
    raise SystemExit(f"{path}: expected epoch {expected}, found {actual}")
PY
}

run_training_stage() {
  local label="$1"
  local run_dir="$2"
  local final_epoch="$3"
  shift 3

  if checkpoint_epoch "$run_dir/epoch_$(printf '%04d' "$final_epoch").ckpt" "$final_epoch" 2>/dev/null; then
    echo "=== ${label}: complete (epoch ${final_epoch}) ==="
    return
  fi

  local resume_args=()
  if [[ -s "$run_dir/latest.ckpt" ]]; then
    resume_args=(--resume "$run_dir/latest.ckpt")
    echo "=== ${label}: resume $run_dir/latest.ckpt ==="
  else
    echo "=== ${label}: start ==="
  fi
  run "$@" "${resume_args[@]}"

  if [[ "$PRINT_ONLY" -eq 0 ]]; then
    checkpoint_epoch "$run_dir/epoch_$(printf '%04d' "$final_epoch").ckpt" "$final_epoch"
  fi
}

if [[ "$PRINT_ONLY" -eq 0 ]]; then
  run "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to run the full replication on CPU.")
print("CUDA:", torch.cuda.get_device_name(0))
PY
fi

run_training_stage "STAGE T: gated teacher e30" "$TEACHER_RUN" 30 \
  "$PYTHON" scripts/train/temporal_yolo/train_gated_detector.py \
  --data-root "$DATA_ROOT" \
  --yolo-weights "$YOLO" \
  --run-dir "$TEACHER_RUN" \
  --epochs 30 --batch-size 4 --clip-len 2 --clip-stride 2 \
  --img-size 640 \
  --lr-gate 1e-3 --lr-yolo 1e-5 \
  --gt-ratio 0.5 --seed "$SEED" \
  --warmup-epochs 0 --clip-grad 10 \
  --save-every 1 --best-by train-loss \
  --protocol-revision V14-FULL-E30-N1-N16-20260614

if [[ "$PRINT_ONLY" -eq 0 ]]; then
  checkpoint_epoch "$TEACHER" 30
fi

if [[ -s "$CACHE/manifest.json" ]] \
  && "$PYTHON" - "$CACHE/manifest.json" "$TEACHER" "$SEQS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
teacher = Path(sys.argv[2])
digest = hashlib.sha256(teacher.read_bytes()).hexdigest()
expected_sequences = sys.argv[3].split(",")
ok = (
    manifest.get("status") == "complete"
    and manifest.get("teacher_checkpoint_sha256") == digest
    and manifest.get("sequences") == expected_sequences
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "=== STAGE 0: cache complete ==="
else
  echo "=== STAGE 0: build/resume teacher cache ==="
  run env \
    DATA_ROOT="$DATA_ROOT" \
    YOLO_WEIGHTS="$YOLO" \
    TEACHER_CKPT="$TEACHER" \
    IMG_SIZE=640 \
    SEQS="$SEQS" \
    scripts/train/temporal_yolo/build_mamba_teacher_cache.sh "$CACHE"
fi

run_training_stage "STAGE 1: legacy N=1 distill" "$DISTILL" 30 \
  "$PYTHON" scripts/train/temporal_yolo/train_mamba_head.py \
  --data-root "$DATA_ROOT" \
  --yolo-weights "$YOLO" \
  --teacher-ckpt "$TEACHER" \
  --cache-dir "$CACHE" \
  --seqs "$SEQS" \
  --run-dir "$DISTILL" \
  --use-pixel-shuffle --use-cross-scan --d-state 16 \
  --scan-stop-grad --legacy-n1-scan \
  --epochs 30 --batch-size 8 --lr 1e-3 \
  --warmup-epochs 0 --seed "$SEED"

run_training_stage "STAGE 2: legacy N=1 live-teacher GT" "$GT_N1" 30 \
  "$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root "$DATA_ROOT" \
  --yolo-weights "$YOLO" \
  --teacher-ckpt "$TEACHER" \
  --mamba-ckpt "$DISTILL/best.ckpt" \
  --run-dir "$GT_N1" \
  --seqs "$SEQS" \
  --img-size 640 --clip-len 4 --clip-stride 8 \
  --epochs 30 --batch-size 4 --lr 1e-4 \
  --warmup-epochs 5 --clip-grad 1.0 \
  --gt-ratio 0.5 --seed "$SEED" \
  --scan-stop-grad --legacy-n1-scan \
  --best-by train-loss --save-every 5

run_training_stage "STAGE 3: fixed N=16 cached GT adaptation" "$GT_N16" 30 \
  "$PYTHON" scripts/train/temporal_yolo/train_mamba_gt.py \
  --data-root "$DATA_ROOT" \
  --yolo-weights "$YOLO" \
  --teacher-ckpt "$TEACHER" \
  --mamba-ckpt "$GT_N1/best.ckpt" \
  --cache-dir "$CACHE" \
  --run-dir "$GT_N16" \
  --seqs "$SEQS" \
  --img-size 640 --clip-len 4 --clip-stride 8 \
  --epochs 30 --batch-size 4 --lr 1e-4 \
  --warmup-epochs 5 --clip-grad 1.0 \
  --gt-ratio 0 --seed "$SEED" \
  --scan-stop-grad --no-legacy-n1-scan \
  --best-by train-loss --save-every 5

echo "=== EVAL: detector recall ==="
run "$PYTHON" scripts/eval/mamba_size_binned_recall.py \
  --teacher-ckpt "$TEACHER" \
  --mamba-ckpt "$GT_N16/best.ckpt" \
  --sequences MOT17-02-SDP \
  --score-thresholds 0.001,0.10,0.25 \
  --output "$REPORT"

echo "=== EVAL: seven-sequence tracking ==="
run "$PYTHON" scripts/eval/mot17.py \
  --preset mamba_whole_graph --detector SDP \
  --mamba-ckpt "$GT_N16/best.ckpt" \
  --mamba-teacher-ckpt "$TEACHER" \
  --output "$RESULTS"

echo "=== V14 FULL E30 REPLICATION COMPLETE ==="
echo "teacher: $TEACHER"
echo "final:   $GT_N16/best.ckpt"
echo "recall:  $REPORT"
echo "track:   $RESULTS"
