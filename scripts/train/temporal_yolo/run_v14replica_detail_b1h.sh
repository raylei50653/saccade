#!/usr/bin/env bash
# B1-H detail branch on the clean replica lineage (the dual-resolution plan's
# prescribed "controlled v14-R retrain"). GT2 stage from replica GT1 with
# --detail-source high (1280x768 bucket, 32ch shallow encoder, 3x3 token).
# Paired against plain replica GT2 (IDF1 73.4).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt

echo "=== GT2 + detail high (30 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_detail_b1h \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --detail-source high \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== RECALL EVAL ==="
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt runs/mamba_gt_v14replica_detail_b1h/best.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output results/mamba_size_recall_v14replica_detail_b1h_02.json

echo ""
echo "=== TRACKING EVAL ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt runs/mamba_gt_v14replica_detail_b1h/best.ckpt \
    --output results/mamba_v14replica_detail_b1h

echo "DETAIL B1H DONE"
