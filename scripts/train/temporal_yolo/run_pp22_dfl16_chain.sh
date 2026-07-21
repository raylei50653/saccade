#!/usr/bin/env bash
# status: experiment
# DFL reg_max=16 retrain — clean single-variable test vs the reg_max=1 baseline.
# Mirrors the baseline keyframe recipe EXACTLY (data, stages, hyperparams, B=4)
# except --reg-max 16. Reuses the teacher FPN-feature cache (reg_max-independent)
# for the cache-mode T3/T1 stages. GT1 is live (gate feedback, gt_ratio 0.5).
#
#   GT1 (live, clip4, 30ep) -> T3 (cache, add-temporal, clip3, 15ep)
#       -> T1 (cache, clip1, 15ep)   [latest.ckpt handoffs]
set -euo pipefail
cd "$(dirname "$0")/../../.."
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import nvidia.cublas.lib,os;print(os.path.dirname(nvidia.cublas.lib.__file__))' 2>/dev/null):${LD_LIBRARY_PATH:-}"

DATA=datasets/PersonPath22
YOLO=models/yolo/yolo26s.pt
TEACHER=runs/gated_det_pp22_augment/best.ckpt
DISTILL=runs/mamba_distill_pp22_augment_e30/best.ckpt
CACHE=runs/mamba_teacher_cache_pp22_augment_e30
SEQS="$(paste -sd, datasets/PersonPath22/train_seqs.txt)"

S1=runs/mamba_gt_pp22_dfl16_stage1
T3=runs/mamba_gt_pp22_dfl16_t3
T1=runs/mamba_gt_pp22_dfl16_t3_t1

COMMON=(
  --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER"
  --img-size 640 --seqs "$SEQS" --reg-max 16 --batch-size 4
  --lr 1e-4 --lr-gate 0 --scan-stop-grad --d-state 16 --d-model 128
  --best-by none --save-every 1
)

echo "=========== DFL STAGE GT1 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$DISTILL" \
  --clip-len 4 --clip-stride 8 --gt-ratio 0.5 \
  --epochs 30 --warmup-epochs 5 --seed 20260612 --run-dir "$S1"

echo "=========== DFL STAGE T3 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$S1/latest.ckpt" --cache-dir "$CACHE" --add-temporal \
  --clip-len 3 --clip-stride 6 --gt-ratio 0 \
  --epochs 15 --warmup-epochs 3 --seed 42 --run-dir "$T3"

echo "=========== DFL STAGE T1 ($(date)) ==========="
.venv/bin/python scripts/train/temporal_yolo/train_mamba_gt.py \
  "${COMMON[@]}" --mamba-ckpt "$T3/latest.ckpt" --cache-dir "$CACHE" \
  --clip-len 1 --clip-stride 2 --gt-ratio 0 \
  --epochs 15 --warmup-epochs 3 --seed 42 --run-dir "$T1"

echo "=========== DFL CHAIN DONE ($(date)) ==========="
echo "T1 epochs in $T1 -> detector-only PP22 held-out + MOT17 vs baseline 45.4/75.8"
