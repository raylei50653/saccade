#!/usr/bin/env bash
# status: experiment
# Native full-config sweep driver (matched baseline suite).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_full_sweep.txt; : > "$LOG"
run() { # epoch cs cst ntt
  local ep=$1 cs=$2 cst=$3 ntt=$4
  local out
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --teacher-head-ckpt "runs/gated_det_native_full/epoch_$(printf %04d $ep).ckpt" \
    --confirm-streak "$cs" --confirm-score-thresh "$cst" --new-track-thresh "$ntt" \
    --output "results/native_full_e${ep}_cs${cs}_cst${cst}" 2>&1)
  local line fps
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|IDs:|FP:|FN:|Rcll:|Prcn:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[e${ep} cs${cs} cst${cst} ntt${ntt}] $line $fps" | tee -a "$LOG"
}
echo "=== native full-training sweep (fair point cs2 cst0.15 ntt0.20) ===" | tee -a "$LOG"
for ep in 12 20 26 30; do run "$ep" 2 0.15 0.20; done
echo "=== done ===" | tee -a "$LOG"
