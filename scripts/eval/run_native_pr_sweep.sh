#!/usr/bin/env bash
# status: experiment
# Native precision/recall operating-point sweep.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_pr_sweep.txt; : > "$LOG"
CKPT=runs/gated_det_native_full/epoch_0026.ckpt
run() { # cst ntt
  local cst=$1 ntt=$2 out line fps
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --teacher-head-ckpt "$CKPT" \
    --confirm-streak 2 --confirm-score-thresh "$cst" --new-track-thresh "$ntt" \
    --output "results/native_pr_cst${cst}_ntt${ntt}" 2>&1)
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|Rcll:|Prcn:|IDs:|FP:|AssA:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[e26 cst${cst} ntt${ntt}] $line $fps" | tee -a "$LOG"
}
echo "=== native e26 PR sweep (recall@precision vs mamba) ===" | tee -a "$LOG"
run 0.10 0.15
run 0.05 0.10
run 0.02 0.05
echo "=== done ===" | tee -a "$LOG"
