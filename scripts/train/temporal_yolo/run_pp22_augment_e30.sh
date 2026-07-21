#!/usr/bin/env bash
# status: experiment
# Downstream pipeline from the augment teacher (gated_det_pp22_augment/best.ckpt).
# Mirrors run_pp22_heldout_e30.sh exactly — only TEACHER, CACHE, and run-dirs differ
# (_augment_e30 suffix) so the e30 baseline stays intact for comparison.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=.venv/bin/python
DATA=datasets/PersonPath22
YOLO=models/yolo/yolo26s.pt
TEACHER="${TEACHER:-runs/gated_det_pp22_augment/best.ckpt}"
CACHE=runs/mamba_teacher_cache_pp22_augment_e30
SEQS="$(paste -sd, datasets/PersonPath22/train_seqs.txt)"
LOG=runs/pp22_augment_e30.log
common=(--data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
        --img-size 640 --seqs "$SEQS")

echo "[pipeline] TEACHER=$TEACHER CACHE=$CACHE  $(date)" | tee -a "$LOG"

echo "[1/5] cache  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_head.py "${common[@]}" \
    --precompute-dir "$CACHE" 2>&1 | tee -a "$LOG"

echo "[2/5] distill  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_head.py "${common[@]}" \
    --cache-dir "$CACHE" --run-dir runs/mamba_distill_pp22_augment_e30 \
    --clip-len 1 --lr 1e-3 --scan-stop-grad --use-pixel-shuffle --use-cross-scan \
    --d-state 16 --d-model 128 --epochs 30 --warmup-epochs 5 --seed 20260612 \
    --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[3/5] GT1 (live teacher)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py "${common[@]}" \
    --mamba-ckpt runs/mamba_distill_pp22_augment_e30/best.ckpt \
    --run-dir runs/mamba_gt_pp22_augment_stage1_e30 \
    --clip-len 4 --clip-stride 8 --lr 1e-4 --lr-gate 0 --gt-ratio 0.5 \
    --scan-stop-grad --d-state 16 --d-model 128 --epochs 30 --warmup-epochs 5 \
    --best-by train-loss --seed 20260612 2>&1 | tee -a "$LOG"

echo "[4/5] T3 (add temporal)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py "${common[@]}" \
    --cache-dir "$CACHE" --mamba-ckpt runs/mamba_gt_pp22_augment_stage1_e30/best.ckpt \
    --run-dir runs/mamba_gt_pp22_augment_t3_e30 \
    --add-temporal --clip-len 3 --clip-stride 6 --lr 1e-4 --lr-gate 0 --gt-ratio 0 \
    --scan-stop-grad --d-state 16 --d-model 128 --epochs 15 --warmup-epochs 3 \
    --best-by train-loss --seed 42 --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[5/5] T1 (deploy candidate)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py "${common[@]}" \
    --cache-dir "$CACHE" --mamba-ckpt runs/mamba_gt_pp22_augment_t3_e30/best.ckpt \
    --run-dir runs/mamba_gt_pp22_augment_t3_t1_e30 \
    --clip-len 1 --clip-stride 2 --lr 1e-4 --lr-gate 0 --gt-ratio 0 \
    --scan-stop-grad --d-state 16 --d-model 128 --epochs 15 --warmup-epochs 3 \
    --best-by train-loss --seed 42 --no-preload-cache 2>&1 | tee -a "$LOG"

echo "[done] final ckpt=runs/mamba_gt_pp22_augment_t3_t1_e30/best.ckpt  $(date)" | tee -a "$LOG"
