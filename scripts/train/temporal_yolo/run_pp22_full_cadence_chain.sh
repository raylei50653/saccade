#!/usr/bin/env bash
# PP22 full-cadence + interp + live-teacher + GPU-decode training chain.
# Reproduces docs/research/training/pp22_full_cadence_interp_training_plan.md §4.
#
#   GT1 (clip4, 30ep) --latest.ckpt--> T3 (add-temporal, clip3, 15ep)
#       --latest.ckpt--> T1 (clip1, 15ep)
#
# Distill stage (clip_len 1) is cadence-invariant -> reused from
# runs/mamba_distill_pp22_augment_e30/best.ckpt (not retrained here).
# Stage handoffs use latest.ckpt (cosine LR annealed to end = warm-start);
# final checkpoint selection is detector-only on MOT17 (plan §5), not here.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import nvidia.cublas.lib,os;print(os.path.dirname(nvidia.cublas.lib.__file__))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

DATA=datasets/PersonPath22_full
YOLO=models/yolo/yolo26s.pt
TEACHER=runs/gated_det_pp22_augment/best.ckpt
DISTILL=runs/mamba_distill_pp22_augment_e30/best.ckpt
SEQS="$(paste -sd, datasets/PersonPath22/train_seqs.txt)"

S1=runs/mamba_gt_pp22_aug_full_stage1
T3=runs/mamba_gt_pp22_aug_full_t3
T1=runs/mamba_gt_pp22_aug_full_t3_t1

COMMON=(
  --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER"
  --img-size 640 --seqs "$SEQS"
  --interpolate-gt --gpu-decode --num-workers 12
  --batch-size 16
  --lr 1e-4 --lr-gate 0 --scan-stop-grad --d-state 16 --d-model 128
  --best-by none --save-every 1
)

echo "=========== STAGE GT1 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$DISTILL" \
  --clip-len 4 --clip-stride 8 --gt-ratio 0.5 \
  --epochs 30 --warmup-epochs 5 --seed 20260612 --run-dir "$S1"

echo "=========== STAGE T3 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$S1/latest.ckpt" --add-temporal \
  --clip-len 3 --clip-stride 6 --gt-ratio 0 \
  --epochs 15 --warmup-epochs 3 --seed 42 --run-dir "$T3"

echo "=========== STAGE T1 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$T3/latest.ckpt" \
  --clip-len 1 --clip-stride 2 --gt-ratio 0 \
  --epochs 15 --warmup-epochs 3 --seed 42 --run-dir "$T1"

echo "=========== CHAIN DONE ($(date)) ==========="
echo "T1 epochs in $T1 -> run plan §5 detector-only MOT17 selection next."
