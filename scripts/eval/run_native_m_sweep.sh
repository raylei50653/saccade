#!/usr/bin/env bash
# status: experiment
# Native M-architecture sweep driver.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_m_sweep.txt; : > "$LOG"
run() { local ep=$1 out line fps
  out=$(.venv/bin/python scripts/eval/mot17.py \
    --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --mamba-yolo-weights models/yolo/yolo26m.pt \
    --teacher-head-ckpt "runs/gated_det_native_full_m/epoch_$(printf %04d $ep).ckpt" \
    --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20 \
    --output "results/native_m_e${ep}" 2>&1)
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|Rcll:|Prcn:|IDs:|FP:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[m_e${ep} cs2 cst0.15]" $line $fps | tee -a "$LOG"; }
echo "=== native-m full-training sweep (fair point) ===" | tee -a "$LOG"
for ep in 12 20 26 30; do run "$ep"; done
echo "=== done ===" | tee -a "$LOG"
