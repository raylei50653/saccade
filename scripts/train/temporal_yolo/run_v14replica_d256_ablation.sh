#!/usr/bin/env bash
# status: experiment
# d_model 128->256 capacity ablation for the v14replica (MOT17) main-line head.
# Mirrors the v14replica curriculum EXACTLY (args recovered from the embedded
# ckpt args of runs/mamba_distill_v14replica, runs/mamba_gt_v14replica_stage1,
# runs/mamba_gt_v14replica_t3_t1) — only --d-model changes 128->256, and
# batch/accum are halved+doubled to fit 12GB at 4x params (effective batch kept).
#
# Stages: cache (rebuilt, d_model-independent) -> distill -> GT1 -> T3 -> T1 -> eval.
# Final deploy candidate: runs/mamba_gt_v14replica_d256_t3_t1/best.ckpt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=.venv/bin/python
DATA=datasets/MOT17
YOLO=models/yolo/yolo26s.pt
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica   # rebuilt below (d_model-independent)
LOG=runs/v14replica_d256_ablation.log
DM=256

echo "[pipeline] d_model=$DM TEACHER=$TEACHER  $(date)" | tee -a "$LOG"

echo "[1/5] cache (rebuild, d_model-independent)  $(date)" | tee -a "$LOG"
if [ ! -d "$CACHE" ]; then
  $PY scripts/train/temporal_yolo/train_mamba_head.py \
      --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
      --img-size 640 --precompute-dir "$CACHE" 2>&1 | tee -a "$LOG"
else
  echo "  cache exists, reuse $CACHE" | tee -a "$LOG"
fi

echo "[2/5] distill (d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --cache-dir "$CACHE" --run-dir runs/mamba_distill_v14replica_d256 \
    --img-size 640 --clip-len 1 --lr 1e-3 --scan-stop-grad \
    --use-pixel-shuffle --use-cross-scan --d-state 16 --d-model "$DM" \
    --epochs 30 --warmup-epochs 5 --seed 20260612 \
    --batch-size 4 --accum-steps 2 --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[3/5] GT1 (live teacher, d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt runs/mamba_distill_v14replica_d256/best.ckpt \
    --run-dir runs/mamba_gt_v14replica_d256_stage1 \
    --img-size 640 --clip-len 4 --clip-stride 8 --lr 1e-4 --lr-gate 0 \
    --gt-ratio 0.5 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 30 --warmup-epochs 5 --best-by train-loss --seed 20260612 \
    --batch-size 2 --accum-steps 2 2>&1 | tee -a "$LOG"

echo "[4/5] T3 (add temporal, d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt runs/mamba_gt_v14replica_d256_stage1/best.ckpt \
    --add-temporal --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_d256_t3 \
    --img-size 640 --clip-len 3 --clip-stride 6 --lr 1e-4 --lr-gate 0 \
    --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 15 --warmup-epochs 3 --clip-grad 1.0 --best-by train-loss \
    --seed 42 --save-every 5 --batch-size 2 --accum-steps 2 \
    --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[5/5] T1 (deploy candidate, d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt runs/mamba_gt_v14replica_d256_t3/best.ckpt \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_d256_t3_t1 \
    --img-size 640 --clip-len 1 --clip-stride 2 --lr 1e-4 --lr-gate 0 \
    --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 15 --warmup-epochs 3 --clip-grad 1.0 --best-by train-loss \
    --seed 42 --save-every 5 --batch-size 2 --accum-steps 2 \
    --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[eval] tracking on MOT17 train/SDP  $(date)" | tee -a "$LOG"
$PY scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --split train \
    --mamba-ckpt runs/mamba_gt_v14replica_d256_t3_t1/best.ckpt \
    --mamba-teacher-ckpt "$TEACHER" \
    --output results/mamba_v14replica_d256 2>&1 | tee -a "$LOG"

echo "[done] final ckpt=runs/mamba_gt_v14replica_d256_t3_t1/best.ckpt  $(date)" | tee -a "$LOG"
