#!/usr/bin/env bash
# status: stable
# Build mamba teacher feature cache used by distill/GT training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CACHE_DIR="${1:-runs/mamba_teacher_cache_v14r_holdout02}"
DATA_ROOT="${DATA_ROOT:-datasets/MOT17}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-models/yolo/yolo26s.pt}"
TEACHER_CKPT="${TEACHER_CKPT:-runs/gated_det_v14r_holdout02/latest.ckpt}"
IMG_SIZE="${IMG_SIZE:-640}"
SEQS="${SEQS:-MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP}"
GPU_DECODE="${GPU_DECODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cmd=(
  .venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py
  --data-root "$DATA_ROOT"
  --yolo-weights "$YOLO_WEIGHTS"
  --teacher-ckpt "$TEACHER_CKPT"
  --img-size "$IMG_SIZE"
  --seqs "$SEQS"
  --precompute-dir "$CACHE_DIR"
)
if [[ "$GPU_DECODE" == "1" ]]; then
  cmd+=(--gpu-decode --num-workers "$NUM_WORKERS")
fi

printf 'Building immutable Mamba teacher cache:\n'
printf '  output: %s\n' "$CACHE_DIR"
printf '  sequences: %s\n' "$SEQS"
printf '  teacher checkpoint: %s\n' "${TEACHER_CKPT:-<base YOLO only>}"
printf '  decode backend: %s\n' "$([[ "$GPU_DECODE" == "1" ]] && echo "torchvision/nvJPEG" || echo "CPU loader")"
printf '  cache payload: P3/P4/P5 + cls/reg targets\n'
printf '  training behavior: no YOLO forward in cached epochs\n'

exec "${cmd[@]}"
