#!/usr/bin/env bash
# Implicit-consistency baseline matched to T3→T1 (explicit), per v14 recipe (git 77fcc262:
# clip_len=4 + all-frames loss + Cross-Scan N=16, "restore all-frames loss as default").
#
# This is the clean control that did NOT exist: same GT1 start, lineage, budget and seeds
# as run_v14replica_t3t1.sh, differing ONLY in implicit (all-frames, no temporal block,
# 10.13M) vs explicit (T3→T1 staging + temporal block, 11.37M). The earlier `_final`
# checkpoints were the full distill→GT1→GT2 chain, a different tier — not comparable.
#
# Usage: run_v14replica_implicit_seed.sh <seed>   (seeds 42 / 20260613 / 20260614)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SEED="$1"
TAG="s${SEED: -2}"
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica
GT1=runs/mamba_gt_v14replica_stage1/best.ckpt   # same start as T3→T1

echo "=== IMPLICIT baseline (all-frames T=4, no temporal) seed=${SEED} ==="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root datasets/MOT17 \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1" \
    --cache-dir "$CACHE" \
    --run-dir "runs/mamba_gt_v14replica_implicit_${TAG}" \
    --img-size 640 --clip-len 4 --clip-stride 8 \
    --epochs 30 --batch-size 4 --lr 1e-4 \
    --warmup-epochs 3 --clip-grad 1.0 \
    --gt-ratio 0 --seed "$SEED" \
    --scan-stop-grad \
    --best-by train-loss --save-every 5

echo ""
echo "=== TRACKING EVAL (whole-graph) ==="
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    --mamba-ckpt "runs/mamba_gt_v14replica_implicit_${TAG}/best.ckpt" \
    --output "results/mamba_v14replica_implicit_${TAG}"

echo ""
echo "=== seed=${SEED} DONE — pair vs explicit t3t1_${TAG} (75.4/65.3 @13, 73.4/62.5 @14) ==="
