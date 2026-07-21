#!/usr/bin/env bash
# status: experiment
# Parametric leave-one-out (LOO) augmentation experiment for the v14replica head.
#
# Tests whether GT1-stage augmentation reduces over-fit, measured OUT-OF-SAMPLE on
# a held-out MOT17 sequence (the only place augmentation can be applied cleanly —
# GT1 runs the teacher live, so frame + gt_boxes are transformed together; distill/
# T3/T1 read a fixed-frame feature cache and cannot be augmented).
#
# Usage:
#   run_aug_loo.sh <d_model> <holdout> <augment|noaug>
#   e.g.  run_aug_loo.sh 128 MOT17-10 augment
#         run_aug_loo.sh 128 MOT17-10 noaug
#         run_aug_loo.sh 128 ""       noaug      # full-7, no holdout (deployable)
#
# Per (width, holdout) the distill stage is built ONCE and shared by both arms
# (Arm A/noaug and Arm B/augment differ only in GT1 augmentation). The full-7
# un-augmented teacher cache is reused as-is. All GT/T3/T1 stages use
# --best-by none --save-every 1 so every epoch is a selectable candidate; the
# deploy ckpt is then chosen by held-out recall via select_ckpt_by_recall.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

DM="${1:?usage: run_aug_loo.sh <d_model> <holdout> <augment|noaug>}"
HOLDOUT="${2-}"
AUG="${3:?usage: run_aug_loo.sh <d_model> <holdout> <augment|noaug>}"

PY=.venv/bin/python
DATA=datasets/MOT17
YOLO=models/yolo/yolo26s.pt
TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
CACHE=runs/mamba_teacher_cache_v14replica          # shared, un-augmented, full-7
ALL_SEQS="MOT17-02 MOT17-04 MOT17-05 MOT17-09 MOT17-10 MOT17-11 MOT17-13"

# Per-width per-fold/effective-batch knobs (effective batch: GT=4, distill=8).
if [ "$DM" -ge 256 ]; then
  GT_BS=2;  GT_ACC=2;  DI_BS=4;  DI_ACC=2
else
  GT_BS=4;  GT_ACC=1;  DI_BS=8;  DI_ACC=1
fi

# Training-seq list = all 7 minus the holdout (empty holdout => full 7).
# Sequence dirs carry the detector suffix (MOT17-10-SDP), and both the dataset
# (explicit seqs used verbatim) and resolve_training_sequences validate against
# the real dir names — so every name must be -SDP suffixed.
TRAIN_SEQS=""
for s in $ALL_SEQS; do
  [ "$s" = "$HOLDOUT" ] && continue
  TRAIN_SEQS="${TRAIN_SEQS:+$TRAIN_SEQS,}${s}-SDP"
done

TAG="d${DM}_ho${HOLDOUT:-none}"
DISTILL_DIR="runs/aug_loo/distill_${TAG}"
GT1_DIR="runs/aug_loo/gt1_${TAG}_${AUG}"
T3_DIR="runs/aug_loo/t3_${TAG}_${AUG}"
T1_DIR="runs/aug_loo/t1_${TAG}_${AUG}"
LOG="runs/aug_loo/${TAG}_${AUG}.log"
mkdir -p runs/aug_loo

# GT-stage holdout flag (empty holdout => no flag => train on all 7).
HO_FLAG=()
[ -n "$HOLDOUT" ] && HO_FLAG=(--holdout-seqs "${HOLDOUT}-SDP")
AUG_FLAG=()
[ "$AUG" = "augment" ] && AUG_FLAG=(--augment)

echo "[aug-loo] DM=$DM HOLDOUT=${HOLDOUT:-none} AUG=$AUG  train_seqs=$TRAIN_SEQS  $(date)" | tee -a "$LOG"

# ---- [0/5] shared un-augmented cache (build once, full-7) -----------------
if [ ! -d "$CACHE" ]; then
  echo "[0/5] cache (rebuild, d_model-independent)  $(date)" | tee -a "$LOG"
  $PY scripts/train/temporal_yolo/train_mamba_head.py \
      --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
      --img-size 640 --precompute-dir "$CACHE" 2>&1 | tee -a "$LOG"
else
  echo "[0/5] cache exists, reuse $CACHE" | tee -a "$LOG"
fi

# ---- [1/5] distill (per width+fold, shared by both arms) ------------------
if [ ! -f "$DISTILL_DIR/best.ckpt" ]; then
  echo "[1/5] distill (d_model=$DM, exclude holdout)  $(date)" | tee -a "$LOG"
  $PY scripts/train/temporal_yolo/train_mamba_head.py \
      --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
      --cache-dir "$CACHE" --run-dir "$DISTILL_DIR" --seqs "$TRAIN_SEQS" \
      --img-size 640 --clip-len 1 --lr 1e-3 --scan-stop-grad \
      --use-pixel-shuffle --use-cross-scan --d-state 16 --d-model "$DM" \
      --epochs 30 --warmup-epochs 5 --seed 20260612 \
      --batch-size "$DI_BS" --accum-steps "$DI_ACC" --no-preload-cache 2>&1 | tee -a "$LOG"
else
  echo "[1/5] distill exists, reuse $DISTILL_DIR/best.ckpt" | tee -a "$LOG"
fi

# ---- [2/5] GT1 (live teacher, AUGMENTABLE) --------------------------------
echo "[2/5] GT1 (d_model=$DM, aug=$AUG, $([ -n "$HOLDOUT" ] && echo holdout=$HOLDOUT || echo full7))  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$DISTILL_DIR/best.ckpt" --run-dir "$GT1_DIR" \
    --img-size 640 --clip-len 4 --clip-stride 8 --lr 1e-4 --lr-gate 0 \
    --gt-ratio 0.5 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 30 --warmup-epochs 5 --best-by none --save-every 1 --seed 20260612 \
    --batch-size "$GT_BS" --accum-steps "$GT_ACC" \
    "${HO_FLAG[@]}" "${AUG_FLAG[@]}" 2>&1 | tee -a "$LOG"

# GT1 has no best.ckpt under --best-by none; pick the GT1 init for T3 by held-out
# recall too (small sweep) so the augmented representation isn't discarded.
echo "[2b/5] select GT1 init by recall  $(date)" | tee -a "$LOG"
$PY scripts/eval/select_ckpt_by_recall.py --run-dir "$GT1_DIR" \
    --teacher-ckpt "$TEACHER" --preset mamba_whole_graph --detector SDP --split train \
    ${HOLDOUT:+--sequences ${HOLDOUT}-SDP} 2>&1 | tee -a "$LOG"

# ---- [3/5] T3 (add temporal, cached -> no augment) ------------------------
echo "[3/5] T3 (d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$GT1_DIR/best_recall.ckpt" --add-temporal --cache-dir "$CACHE" \
    --run-dir "$T3_DIR" --img-size 640 --clip-len 3 --clip-stride 6 --lr 1e-4 \
    --lr-gate 0 --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 15 --warmup-epochs 3 --clip-grad 1.0 --best-by none --save-every 1 \
    --seed 42 --batch-size "$GT_BS" --accum-steps "$GT_ACC" --no-preload-cache \
    "${HO_FLAG[@]}" 2>&1 | tee -a "$LOG"
echo "[3b/5] select T3 init by recall  $(date)" | tee -a "$LOG"
$PY scripts/eval/select_ckpt_by_recall.py --run-dir "$T3_DIR" \
    --teacher-ckpt "$TEACHER" --preset mamba_whole_graph --detector SDP --split train \
    ${HOLDOUT:+--sequences ${HOLDOUT}-SDP} 2>&1 | tee -a "$LOG"

# ---- [4/5] T1 (deploy candidate, cached -> no augment) --------------------
echo "[4/5] T1 (d_model=$DM)  $(date)" | tee -a "$LOG"
$PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" --yolo-weights "$YOLO" --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$T3_DIR/best_recall.ckpt" --cache-dir "$CACHE" \
    --run-dir "$T1_DIR" --img-size 640 --clip-len 1 --clip-stride 2 --lr 1e-4 \
    --lr-gate 0 --gt-ratio 0 --scan-stop-grad --d-state 16 --d-model "$DM" \
    --epochs 15 --warmup-epochs 3 --clip-grad 1.0 --best-by none --save-every 1 \
    --seed 42 --batch-size "$GT_BS" --accum-steps "$GT_ACC" --no-preload-cache \
    "${HO_FLAG[@]}" 2>&1 | tee -a "$LOG"

# ---- [5/5] select deploy ckpt by HELD-OUT recall --------------------------
echo "[5/5] select T1 deploy ckpt by ${HOLDOUT:-all}-recall  $(date)" | tee -a "$LOG"
$PY scripts/eval/select_ckpt_by_recall.py --run-dir "$T1_DIR" \
    --teacher-ckpt "$TEACHER" --preset mamba_whole_graph --detector SDP --split train \
    ${HOLDOUT:+--sequences ${HOLDOUT}-SDP} 2>&1 | tee -a "$LOG"

echo "[done] deploy=$T1_DIR/best_recall.ckpt  $(date)" | tee -a "$LOG"
