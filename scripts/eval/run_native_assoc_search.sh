#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_assoc_search.txt; : > "$LOG"
CKPT=runs/gated_det_native_full/epoch_0026.ckpt
BASE="--confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20"
run() { local tag=$1; shift; local out line fps
  out=$(.venv/bin/python scripts/eval/mot17.py --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --teacher-head-ckpt "$CKPT" $BASE "$@" --output "results/native_as_${tag}" 2>&1)
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|Rcll:|Prcn:|IDs:|FP:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[$tag] $line $fps" | tee -a "$LOG"; }
echo "=== native e26 association-only search (no PC) ===" | tee -a "$LOG"
run bridge_relax   --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.7 --relink-bridge-px 0.4
run bridge_maxage  --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.7 --relink-bridge-px 0.4 --relink-bridge-max-age 240
run relink_sim     --relink-sim-thresh 0.85
echo "=== done ===" | tee -a "$LOG"
