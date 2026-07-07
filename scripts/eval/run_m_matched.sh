#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/m_matched.txt; : > "$LOG"
MTEACH=runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt
MHEAD=runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt
emit() { local tag=$1 out=$2 line fps
  line=$(echo "$out" | grep -E "IDF1:|MOTA:|HOTA:|AssA:|Rcll:|Prcn:|IDs:|FP:" | tr '\n' ' ')
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  echo "[$tag] $line $fps" | tee -a "$LOG"; }
echo "=== m matched control (same m teacher backbone epoch_0012) ===" | tee -a "$LOG"
# mamba-m head at its saturated fair point
o=$(.venv/bin/python scripts/eval/mot17.py --preset mamba_pyt_backbone --detector SDP --no-compile \
  --mamba-yolo-weights models/yolo/yolo26m.pt --mamba-teacher-ckpt "$MTEACH" --mamba-ckpt "$MHEAD" \
  --confirm-score-thresh 0.50 --new-track-thresh 0.28 --output results/m_matched_mamba_cst050 2>&1)
emit "mamba_m_cst050" "$o"
# native-m head (same backbone) at its calibrated fair point
o=$(.venv/bin/python scripts/eval/mot17.py --preset mamba_pyt_backbone --detector SDP --no-compile \
  --mamba-ckpt "" --mamba-yolo-weights models/yolo/yolo26m.pt --teacher-head-ckpt "$MTEACH" \
  --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20 --output results/m_matched_native_cst015 2>&1)
emit "native_m_matched_cst015" "$o"
echo "=== done ===" | tee -a "$LOG"
