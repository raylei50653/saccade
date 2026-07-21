#!/usr/bin/env bash
# status: experiment
# T3->T1 GT2 multi-seed validation: paired comparison against each seed's
# plain GT2 baseline (mamba_gt_v14replica_${TAG}_final). Warm-starts from the
# same seed's GT1 checkpoint so the only difference is the GT2 curriculum.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEED="$1"
TAG="s${SEED: -2}"
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1="runs/mamba_gt_v14replica_${TAG}_stage1/best.ckpt"

echo "=== PHASE A (${TAG}): T=3 temporal GT2 (15 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --add-temporal \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3_${TAG}" \
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
    --mamba-ckpt "runs/mamba_gt_v14replica_t3_${TAG}/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_t3_t1_${TAG}" \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== TRACKING EVAL (${TAG}) ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "runs/mamba_gt_v14replica_t3_t1_${TAG}/best.ckpt" \
    --output "results/mamba_v14replica_t3t1_${TAG}"

echo "T3T1 SEED ${SEED} DONE"
