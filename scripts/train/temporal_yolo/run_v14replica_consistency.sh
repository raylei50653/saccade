#!/usr/bin/env bash
# status: experiment
# Route 3 of mamba-t3t1-curriculum-20260613.md §5: explicit consistency protection.
#
# Context (§4.1): full-grad SSM-ft from the T3→T1 checkpoint ERASES the shaping
# (AssA 66.0 → 62.8) because the per-frame v8 loss has no consistency term — the
# extra SSM freedom chases single-frame DetA and drifts off the cross-frame-
# consistent construction. This run adds --consistency-weight (cross-frame cls
# logit L2 at matched-track GT centres) to that same full-grad phase to test
# whether it neutralises the antagonism: keep AssA near 66 while gaining DetA.
#
# Baseline to beat: T3T1→ssmft = IDF1 73.8 / MOTA 79.4 / DetA 71.0 / AssA 62.8.
# Star to preserve: T3→T1     = IDF1 75.4 / MOTA 77.6 / DetA 69.7 / AssA 66.0.
#
# CONS_WEIGHT is the knob to sweep (logit-MSE scale differs from det loss ~2.85).
# Start coarse: {0.1, 0.3, 1.0}. Pass as $1, default 0.3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONS_WEIGHT="${1:-0.3}"
TAG="cw${CONS_WEIGHT/./p}"  # e.g. 0.3 -> cw0p3

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
START=runs/mamba_gt_v14replica_t3_t1/best.ckpt  # the shaped (75.4) checkpoint

echo "=== PHASE A: full-grad T=3 SSM-ft + consistency protection (w=${CONS_WEIGHT}) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$START" \
    --add-temporal \
    --no-scan-stop-grad \
    --consistency-weight "$CONS_WEIGHT" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3t1_consist_${TAG}" \
    --img-size 640 --clip-len 3 --clip-stride 6 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 \
    --best-by train-loss --save-every 5

echo ""
echo "=== PHASE B: T=1 spatial re-adaptation (deploy config) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "runs/mamba_gt_v14replica_t3t1_consist_${TAG}/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3t1_consist_${TAG}_t1" \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 \
    --no-scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== TRACKING EVAL (whole-graph T=1 deploy) ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "runs/mamba_gt_v14replica_t3t1_consist_${TAG}_t1/best.ckpt" \
    --output "results/mamba_v14replica_t3t1_consist_${TAG}"

echo ""
echo "=== DONE (w=${CONS_WEIGHT}) — compare DetA/AssA vs T3T1→ssmft (71.0/62.8) ==="
