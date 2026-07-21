#!/usr/bin/env bash
# status: experiment
# v14replica student-chain reseed: distill -> GT1(live) -> GT2(cache) -> eval.
# Teacher (gated_det_v14replica/epoch_0012) and cache stay fixed; this measures
# the student-chain seed noise band for the replication comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEED="$1"
TAG="s${SEED: -2}"
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
SEQS=MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP

.venv/bin/python scripts/train/temporal_yolo/train_mamba_head.py \
    --data-root datasets/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt "$TEACHER" \
    --cache-dir "$CACHE" \
    --seqs "$SEQS" \
    --run-dir "runs/mamba_distill_v14replica_${TAG}" \
    --use-pixel-shuffle --use-cross-scan --d-state 16 \
    --scan-stop-grad \
    --epochs 30 --batch-size 8 --lr 1e-3 --seed "$SEED"

.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "runs/mamba_distill_v14replica_${TAG}/best.ckpt" \
    --run-dir "runs/mamba_gt_v14replica_${TAG}_stage1" \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0.5 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "runs/mamba_gt_v14replica_${TAG}_stage1/best.ckpt" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_${TAG}_final" \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 5 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

.venv/bin/python scripts/eval/mamba_size_binned_recall.py \
    --mamba-ckpt "runs/mamba_gt_v14replica_${TAG}_final/best.ckpt" \
    --sequences MOT17-02-SDP \
    --score-thresholds 0.001,0.10,0.25 \
    --output "results/mamba_size_recall_v14replica_${TAG}_02.json"

.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "runs/mamba_gt_v14replica_${TAG}_final/best.ckpt" \
    --output "results/mamba_v14replica_${TAG}"

echo "SEED ${SEED} DONE"
