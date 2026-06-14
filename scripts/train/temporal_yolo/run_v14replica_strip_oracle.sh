#!/usr/bin/env bash
# Strip detail routing (Phase 1, GT-driven oracle mask) on the clean replica
# lineage. GT2 stage from replica GT1 with --detail-source strip-oracle:
# fixed-budget GT-routed Detail Mamba reads raw original-image pixel strips at
# small-object P3 cells (min-side < 8px). Targets the min_4to8 recall gap
# (0.826 @0.25 / 0.884 @0.001 on MOT17-02). Paired against plain replica GT2
# (IDF1 73.4) and against the dense B1-H detail branch.
# Design: docs/modules/detection/research/mamba-strip-detail-routing-design.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt

echo "=== GT2 + strip-oracle detail (30 epochs) ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --cache-dir "$CACHE" \
    --run-dir runs/mamba_gt_v14replica_strip_oracle \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --detail-source strip-oracle \
    --strip-route-budget 640 \
    --strip-small-threshold 8.0 \
    --strip-length 32 \
    --strip-width 3 \
    --strip-stem-channels 32 \
    --strip-route-chunk-size 16 \
    --epochs 30 --batch-size 2 --accum-steps 2 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== RECALL EVAL ==="
.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt runs/mamba_gt_v14replica_strip_oracle/best.ckpt \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output report_data/mamba_size_recall_v14replica_strip_oracle_02.json

echo ""
echo "=== TRACKING EVAL ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt runs/mamba_gt_v14replica_strip_oracle/best.ckpt \
    --output results/mamba_v14replica_strip_oracle

echo "STRIP ORACLE DONE"
