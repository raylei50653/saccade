#!/usr/bin/env bash
# Matched-baseline: original YOLO detect head (teacher) vs Mamba head through the
# IDENTICAL tracker on the SAME PyTorch backbone (gated_det_v14replica/epoch_0012).
# Single variable = detection head. See docs/reference/ (findings write-up).
#
# All 7 MOT17 SDP sequences. PyTorch backbone (pyt_backbone preset) so both heads
# sit on the same backbone object; eager (no-compile) so FPS is a lower bound.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"

TEACHER=runs/gated_det_v14replica/epoch_0012.ckpt
LOG=results/teacher_matched_baseline_summary.txt
: > "$LOG"

emit() {  # tag  <eval-stdout>
  local tag=$1; shift
  local out=$1
  local line
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|IDs:|FP:|FN:|Rcll:|Prcn:" | tr '\n' ' ')
  local fps
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[$tag] $line $fps" | tee -a "$LOG"
}

run_mamba() {  # tag  cst  ntt
  local tag=$1 cst=$2 ntt=$3
  local out
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-teacher-ckpt "$TEACHER" \
    --confirm-score-thresh "$cst" --new-track-thresh "$ntt" \
    --output "results/matched_$tag" 2>&1)
  emit "$tag" "$out"
}

run_teacher() {  # tag  cs  cst  ntt
  local tag=$1 cs=$2 cst=$3 ntt=$4
  local out
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --teacher-head-ckpt "$TEACHER" \
    --confirm-streak "$cs" --confirm-score-thresh "$cst" --new-track-thresh "$ntt" \
    --output "results/matched_$tag" 2>&1)
  emit "$tag" "$out"
}

echo "=== Matched baseline: teacher (original YOLO head) vs Mamba head, 7-seq ===" | tee -a "$LOG"
# Treatment: Mamba head at frozen headline confirm (0.50) and at its own best (0.30).
run_mamba  mamba_headline_cst050  0.50 0.28
run_mamba  mamba_best_cst030      0.30 0.28
# Control: teacher head at the frozen (mamba-tuned) thresholds, and at its fair point.
run_teacher teacher_deploy_cst050 3 0.50 0.28
run_teacher teacher_fair_cst020   2 0.20 0.20
run_teacher teacher_fair_cst015   2 0.15 0.20
run_teacher teacher_fair_cst030   3 0.30 0.28
echo "=== done ===" | tee -a "$LOG"
