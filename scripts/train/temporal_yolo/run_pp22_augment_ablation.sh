#!/usr/bin/env bash
# status: experiment
# Augment single-factor ablation: e30 recipe + --augment, nothing else.
#
# Isolates the (now fixed, no-clip) augmentation from the lr/epochs/balance changes
# of the strong recipe.  Compares to the e30 baseline (same recipe, no augment).
#
#   e30 baseline:  runs/gated_det_pp22/best.ckpt  (no augment, lr 1e-5, 30ep)
#   augment-only:  runs/gated_det_pp22_augment/best.ckpt
#
# Both evaluated on held-out MOT17 with the same gate-off recall protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
SEQS="$(paste -sd, datasets/PersonPath22/train_seqs.txt)"

.venv/bin/python scripts/train/temporal_yolo/train_gated_detector.py \
    --data-root datasets/PersonPath22 --yolo-weights models/yolo/yolo26s.pt \
    --run-dir runs/gated_det_pp22_augment \
    --epochs 30 --batch-size 4 --clip-len 2 --clip-stride 2 --img-size 640 \
    --lr-gate 1e-3 --lr-yolo 1e-5 --gt-ratio 0.5 --seed 20260612 \
    --warmup-epochs 5 \
    --augment \
    --num-workers 16 \
    --save-every 1 --best-by train-loss --seqs "$SEQS"
