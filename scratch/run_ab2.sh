#!/usr/bin/env bash
set -u
cd /home/ray/developer/ai/saccade || exit 1
TORCHLIB=$(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
NVLIBS=$(.venv/bin/python -c "import glob,os,nvidia; base=os.path.dirname(nvidia.__file__); print(':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(base+'/*/lib/*.so*')))))")
export LD_LIBRARY_PATH="$TORCHLIB:$NVLIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"
OUT=scratch/ab_runs; mkdir -p "$OUT"; SUM="$OUT/summary2.txt"; : > "$SUM"
run() {
  local label="$1"; shift
  echo "[$(date +%H:%M:%S)] RUN $label : $*" | tee -a "$SUM"
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
    "$@" --output "$OUT/$label" > "$OUT/$label.log" 2>&1
  [ $? -ne 0 ] && { echo "  $label FAILED" | tee -a "$SUM"; return; }
  awk '/=== OVERALL METRICS ===/{f=1} f{print} /Rcll:/{if(f)exit}' "$OUT/$label.log" | sed 's/^/    /' >> "$SUM"
  echo "" >> "$SUM"
}
# GMC always ON; add one more module each (compare against plus_gmc=75.9 already measured)
run gmc_bridge --gmc --relink-bridge-enabled --no-occ-state-enabled --oao-tau 0
run gmc_occ    --gmc --no-relink-bridge-enabled --occ-state-enabled --oao-tau 0
run gmc_oao    --gmc --no-relink-bridge-enabled --no-occ-state-enabled --oao-tau 0.50 --oao-ramp-frames 25
echo "[$(date +%H:%M:%S)] DONE" | tee -a "$SUM"
