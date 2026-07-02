#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_param_search.txt; : > "$LOG"
CKPT=runs/gated_det_native_full/epoch_0026.ckpt
PC="--private-continuation --private-candidate-nms-iou 0.70 --private-prior-iou-threshold 0.30 --private-min-score 0.10 --private-max-candidates 50 --private-selection-mode global"
run() { # tag  extra-args...
  local tag=$1; shift
  local out line fps
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --teacher-head-ckpt "$CKPT" \
    "$@" --output "results/native_ps_${tag}" 2>&1)
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|Rcll:|Prcn:|IDs:|FP:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[$tag] $line $fps" | tee -a "$LOG"
}
echo "=== native e26 param search (baseline + private_continuation + assoc) ===" | tee -a "$LOG"
run base_cst15         --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20
run pc_cst15           --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20 $PC
run pc_cst10           --confirm-streak 2 --confirm-score-thresh 0.10 --new-track-thresh 0.15 $PC
run pc_cst10_minsc05   --confirm-streak 2 --confirm-score-thresh 0.10 --new-track-thresh 0.15 --private-continuation --private-candidate-nms-iou 0.70 --private-prior-iou-threshold 0.30 --private-min-score 0.05 --private-max-candidates 50 --private-selection-mode global
echo "=== done ===" | tee -a "$LOG"
