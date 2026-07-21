#!/usr/bin/env bash
# status: experiment
# Native M-parameter sweep driver.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_m_param.txt; : > "$LOG"
CKPT=runs/gated_det_native_full_m/epoch_0026.ckpt
Y="--mamba-ckpt --mamba-yolo-weights models/yolo/yolo26m.pt"
PC="--private-continuation --private-candidate-nms-iou 0.70 --private-prior-iou-threshold 0.30 --private-min-score 0.10 --private-max-candidates 50 --private-selection-mode global"
run() { local tag=$1; shift; local out line fps
  out=$(.venv/bin/python scripts/eval/mot17.py --preset mamba_pyt_backbone --detector SDP --no-compile \
    --mamba-ckpt "" --mamba-yolo-weights models/yolo/yolo26m.pt --teacher-head-ckpt "$CKPT" \
    --confirm-streak 2 "$@" --output "results/native_m_ps_${tag}" 2>&1)
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|Rcll:|Prcn:|IDs:|FP:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[$tag] $line $fps" | tee -a "$LOG"; }
echo "=== native-m e26 param search ===" | tee -a "$LOG"
run base            --confirm-score-thresh 0.15 --new-track-thresh 0.20
run bridge_relax    --confirm-score-thresh 0.15 --new-track-thresh 0.20 --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.7 --relink-bridge-px 0.4
run pc              --confirm-score-thresh 0.15 --new-track-thresh 0.20 $PC
run bridge_pc       --confirm-score-thresh 0.15 --new-track-thresh 0.20 --relink-bridge-h-lo 0.6 --relink-bridge-h-hi 1.7 --relink-bridge-px 0.4 $PC
echo "=== done ===" | tee -a "$LOG"
