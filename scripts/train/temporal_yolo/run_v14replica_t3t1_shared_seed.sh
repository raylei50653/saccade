#!/usr/bin/env bash
# status: experiment
# Explicit T3→T1 at an arbitrary seed, warm-started from the SHARED GT1
# (runs/mamba_gt_v14replica_stage1/best.ckpt) — identical to run_v14replica_t3t1.sh
# (the seed-42 run) except --seed is threaded and run-dirs are tagged. Use this
# (NOT run_v14replica_t3t1_seed.sh, which needs a seed-specific GT1) to pair against
# the implicit baseline, which also starts from the shared GT1. Only the seed (GT2
# stage randomness) and implicit-vs-staging differ.
#
# Usage: run_v14replica_t3t1_shared_seed.sh <seed>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEED="$1"
TAG="s${SEED: -2}"
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt   # shared GT1 (same as implicit + seed-42)

echo "=== PHASE A (${TAG}): T=3 temporal GT2 (15 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --add-temporal \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3_shared_${TAG}" \
    --img-size 640 --clip-len 3 --clip-stride 6 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== PHASE B (${TAG}): T=1 spatial re-adaptation (15 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "runs/mamba_gt_v14replica_t3_shared_${TAG}/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3_t1_shared_${TAG}" \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== TRACKING EVAL (${TAG}) ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "runs/mamba_gt_v14replica_t3_t1_shared_${TAG}/best.ckpt" \
    --output "results/mamba_v14replica_t3t1_shared_${TAG}"

echo "T3T1 (shared GT1) SEED ${SEED} DONE"
