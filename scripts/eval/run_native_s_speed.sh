#!/usr/bin/env bash
# status: experiment
# Native S-speed variant sweep.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"
export SACCADE_BUILD_PATH="$ROOT/build"
LOG=results/native_s_speed.txt; : > "$LOG"
NCKPT=runs/gated_det_native_full/epoch_0026.ckpt
MTEACH=runs/gated_det_v14replica/epoch_0012.ckpt
run() { # tag  head-args...   (compile controlled by presence of --no-compile in args)
  local tag=$1; shift; local out fps perseq
  out=$(.venv/bin/python scripts/eval/mot17.py --preset mamba_pyt_backbone --detector SDP \
    "$@" --output "results/native_s_speed_${tag}" 2>&1)
  fps=$(echo "$out" | grep -oE "Overall throughput: [0-9.]+ FPS" | tail -1)
  perseq=$(echo "$out" | grep -oE "MOT17-[0-9]+-SDP.*[0-9.]+ FPS" | grep -oE "[0-9.]+ FPS" | tr '\n' ' ')
  echo "[$tag] $fps  | per-seq: $perseq" | tee -a "$LOG"
}
echo "=== native-s speed (eager vs compiled) + mamba-s reference ===" | tee -a "$LOG"
run native_eager    --mamba-ckpt "" --teacher-head-ckpt "$NCKPT" --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20 --no-compile
run native_compiled --mamba-ckpt "" --teacher-head-ckpt "$NCKPT" --confirm-streak 2 --confirm-score-thresh 0.15 --new-track-thresh 0.20
run mamba_eager     --mamba-teacher-ckpt "$MTEACH" --confirm-score-thresh 0.50 --new-track-thresh 0.28 --no-compile
run mamba_compiled  --mamba-teacher-ckpt "$MTEACH" --confirm-score-thresh 0.50 --new-track-thresh 0.28
echo "=== done ===" | tee -a "$LOG"
