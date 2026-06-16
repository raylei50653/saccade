#!/usr/bin/env bash
# T=3→T=1 GT2 variant: test "temporal enhances spatial consistency" hypothesis.
# Phase A: 15ep GT2 with temporal (T=3), Phase B: 30ep GT2 T=1 re-adaptation.
# Warm-start from replica GT1 best, same teacher/cache as replica.
# Result: T3→T1 > 73.8 (0.4pp noise band above replica 73.4) = positive signal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt

echo "=== PHASE A: T=3 temporal GT2 (15 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --add-temporal \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_t3 \
    --img-size 640 --clip-len 3 --clip-stride 6 \
    --epochs 15 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== PHASE B: T=1 spatial re-adaptation (30 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt runs/mamba_gt_v14replica_t3/best.ckpt \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_t3_t1_15x30 \
    --img-size 640 --clip-len 1 --clip-stride 2 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== RECALL EVAL ==="
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt runs/mamba_gt_v14replica_t3_t1_15x30/best.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output report_data/mamba_size_recall_v14replica_t3t1_15x30_02.json

echo ""
echo "=== TRACKING EVAL ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt runs/mamba_gt_v14replica_t3_t1_15x30/best.ckpt \
    --mamba-teacher-ckpt "$TEACHER" \
    --output results/mamba_v14replica_t3t1_15x30

echo ""
echo "=== DONE ==="
