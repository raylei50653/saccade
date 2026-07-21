#!/usr/bin/env bash
# status: experiment
# Strengthened PP22 teacher — strategies A + B (held-out plan §11). NOT yet run.
#
#   A (schedule/LR, the directly-evidenced lever): lr-yolo 1e-5 → 1e-4 (the loss was
#     still dropping −0.025/epoch when the cosine throttled LR to ~0; the backbone was
#     never fed enough), 60 epochs, --min-lr-ratio 0.1 so it keeps learning late.
#   B (data): --augment (clip-consistent hflip + scale/translate jitter + brightness/
#     contrast) for scale/small-object robustness, + --balance-by-seq so dense/long
#     scenes don't dominate (PP22 densities span 13–96 boxes/frame).
#
# Mirrors the gated_det_pp22 recipe otherwise (gt-ratio 0.5, clip-len 2, stride 2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
SEQS="$(paste -sd, datasets/PersonPath22/train_seqs.txt)"

.venv/bin/python scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/PersonPath22 --yolo-weights models/yolo/yolo26s.pt \
    --run-dir runs/gated_det_pp22_strong \
    --epochs 60 --batch-size 4 --clip-len 2 --clip-stride 2 --img-size 640 \
    --lr-gate 1e-3 --lr-yolo 1e-4 --gt-ratio 0.5 --seed 20260612 \
    --warmup-epochs 5 --min-lr-ratio 0.1 \
    --augment --balance-by-seq \
    --num-workers 16 \
    --save-every 1 --best-by train-loss --seqs "$SEQS"

# Then rebuild the downstream chain on the new teacher (own cache dir):
#   TEACHER=runs/gated_det_pp22_strong/best.ckpt \
#     bash scripts/train/temporal_yolo/run_pp22_heldout_e30.sh
# (edit CACHE= in that script to runs/mamba_teacher_cache_pp22_strong first — cache is
#  teacher-specific.) Eval with --preset mamba_pyt_backbone (never the TRT-engine preset).
#
# Still pending (strategy E): hold out a PP22 val split and select epochs by recall,
# not train-loss — train-loss ≠ generalization in the many-scene regime.
