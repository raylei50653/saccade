#!/usr/bin/env bash
# status: experiment
# Matched reduction-variant experiment for the Mamba recall bottleneck.
#
# Arms:
#   baseline_conv      original stride-4 conv reduction
#   blur_conv          fixed 3x3 blur + stride-4 conv
#   space_to_depth     pixel-unshuffle phase packing + 1x1 projection
#   wavelet            Haar wavelet frequency packing + 1x1 projection
#
# Usage:
#   scripts/train/temporal_yolo/run_reduction_candidates.sh train
#   scripts/train/temporal_yolo/run_reduction_candidates.sh eval
#   scripts/train/temporal_yolo/run_reduction_candidates.sh bootstrap
#   scripts/train/temporal_yolo/run_reduction_candidates.sh all
#
# Useful smoke:
#   EPOCHS=1 MAX_BATCHES=1 SEQS=MOT17-02-SDP SELECT=0 BEST_BY=train-loss \
#     scripts/train/temporal_yolo/run_reduction_candidates.sh train
#
# Useful full validation after the 3-epoch screen:
#   ARMS="baseline_conv space_to_depth wavelet" scripts/train/temporal_yolo/run_reduction_candidates.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
PY="${PY:-uv run python}"
DATA="${DATA:-datasets/MOT17}"
YOLO="${YOLO:-models/yolo/yolo26s.pt}"
TEACHER="${TEACHER:-runs/gated_det_v1/best.ckpt}"
BASE_CKPT="${BASE_CKPT:-runs/mamba_gt_vgt_mamba_v14/best.ckpt}"
OUT_ROOT="${OUT_ROOT:-runs/reduction_candidates}"
REPORT_ROOT="${REPORT_ROOT:-results/reduction_candidates}"
SEQS="${SEQS:-MOT17-02-SDP}"
EVAL_SEQS="${EVAL_SEQS:-MOT17-02-SDP}"
EPOCHS="${EPOCHS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
BEST_BY="${BEST_BY:-none}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
CLIP_LEN="${CLIP_LEN:-4}"
CLIP_STRIDE="${CLIP_STRIDE:-8}"
LR="${LR:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-20000}"
SELECT="${SELECT:-1}"
ARMS="${ARMS:-baseline_conv blur_conv space_to_depth wavelet}"

mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

available_arms=(
  "baseline_conv:conv"
  "blur_conv:blur-conv"
  "space_to_depth:space-to-depth"
  "wavelet:wavelet"
)
arms=()

add_arm() {
  local requested="$1"
  local arm name variant
  if [ "$requested" = "all" ]; then
    for arm in "${available_arms[@]}"; do
      arms+=("$arm")
    done
    return
  fi
  for arm in "${available_arms[@]}"; do
    IFS=: read -r name variant <<<"$arm"
    if [ "$name" = "$requested" ]; then
      arms+=("$arm")
      return
    fi
  done
  echo "unknown arm '$requested' in ARMS='$ARMS'" >&2
  echo "available arms: baseline_conv blur_conv space_to_depth wavelet" >&2
  exit 2
}

for requested_arm in ${ARMS//,/ }; do
  add_arm "$requested_arm"
done

if [ "${#arms[@]}" -eq 0 ]; then
  echo "no arms selected by ARMS='$ARMS'" >&2
  exit 2
fi

max_batches_arg=()
if [ "$MAX_BATCHES" != "0" ]; then
  max_batches_arg=(--max-batches-per-epoch "$MAX_BATCHES")
fi

run_train() {
  local name="$1"
  local variant="$2"
  local run_dir="$OUT_ROOT/$name"
  local log="$REPORT_ROOT/${name}_train.log"
  echo "[train] $name variant=$variant run_dir=$run_dir $(date)" | tee -a "$log"
  $PY scripts/train/temporal_yolo/train_mamba_gt.py \
    --data-root "$DATA" \
    --yolo-weights "$YOLO" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$BASE_CKPT" \
    --run-dir "$run_dir" \
    --seqs "$SEQS" \
    --epochs "$EPOCHS" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --clip-len "$CLIP_LEN" \
    --clip-stride "$CLIP_STRIDE" \
    --lr "$LR" \
    --gt-ratio 0.5 \
    --scan-stop-grad \
    --reduction-variant "$variant" \
    --best-by "$BEST_BY" \
    --save-every "$SAVE_EVERY" \
    "${max_batches_arg[@]}" 2>&1 | tee -a "$log"

  if [ "$SELECT" = "1" ]; then
    echo "[select] $name $(date)" | tee -a "$log"
    $PY scripts/eval/select_ckpt_by_recall.py \
      --run-dir "$run_dir" \
      --teacher-ckpt "$TEACHER" \
      --preset mamba_whole_graph \
      --detector SDP \
      --split train \
      --sequences "$EVAL_SEQS" 2>&1 | tee -a "$log"
  fi
}

run_eval() {
  local name="$1"
  local ckpt="$OUT_ROOT/$name/best_recall.ckpt"
  if [ ! -f "$ckpt" ]; then
    ckpt="$OUT_ROOT/$name/best.ckpt"
  fi
  if [ ! -f "$ckpt" ]; then
    ckpt="$OUT_ROOT/$name/latest.ckpt"
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[eval] missing checkpoint for $name: $ckpt" >&2
    return 1
  fi
  local out="$REPORT_ROOT/${name}_size_recall.json"
  echo "[eval] $name ckpt=$ckpt out=$out $(date)"
  $PY scripts/eval/detector/mamba_size_binned_recall.py \
    --data-root "$DATA" \
    --sequences "$EVAL_SEQS" \
    --yolo-weights "$YOLO" \
    --teacher-ckpt "$TEACHER" \
    --mamba-ckpt "$ckpt" \
    --score-thresholds "0.001,0.10,0.25" \
    --save-frame-records \
    --output "$out"
}

run_bootstrap() {
  local baseline="$REPORT_ROOT/baseline_conv_size_recall.json"
  local arm name variant
  for arm in "${arms[@]}"; do
    IFS=: read -r name variant <<<"$arm"
    if [ "$name" = "baseline_conv" ]; then
      continue
    fi
    local candidate="$REPORT_ROOT/${name}_size_recall.json"
    local out="$REPORT_ROOT/${name}_vs_baseline_bootstrap.json"
    echo "[bootstrap] $name vs baseline $(date)"
    $PY scripts/eval/bootstrap_mamba_size_recall.py \
      --baseline "$baseline" \
      --candidate "$candidate" \
      --sequence "$EVAL_SEQS" \
      --thresholds "0.001,0.1,0.25" \
      --bins "all,min_4to8,min_8to16" \
      --samples "$BOOTSTRAP_SAMPLES" \
      --output "$out"
  done
}

case "$MODE" in
  train)
    for arm in "${arms[@]}"; do
      IFS=: read -r name variant <<<"$arm"
      run_train "$name" "$variant"
    done
    ;;
  eval)
    for arm in "${arms[@]}"; do
      IFS=: read -r name variant <<<"$arm"
      run_eval "$name"
    done
    ;;
  bootstrap)
    run_bootstrap
    ;;
  all)
    "$0" train
    "$0" eval
    "$0" bootstrap
    ;;
  *)
    echo "usage: $0 {train|eval|bootstrap|all}" >&2
    exit 2
    ;;
esac
