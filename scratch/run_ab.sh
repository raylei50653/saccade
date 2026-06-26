#!/usr/bin/env bash
# Per-feature ablation: bare (all off) -> bare + each single feature.
# Full 7-seq MOT17 train / SDP. Writes per-run logs + a summary table.
set -u
cd /home/ray/developer/ai/saccade || exit 1
TORCHLIB=$(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
NVLIBS=$(.venv/bin/python -c "import glob,os,nvidia; base=os.path.dirname(nvidia.__file__); print(':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(base+'/*/lib/*.so*')))))")
export LD_LIBRARY_PATH="$TORCHLIB:$NVLIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/build:${PYTHONPATH:-}"

OUT=scratch/ab_runs
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
: > "$SUM"

# label | the four toggles (gmc | bridge | occ | oao)
run() {
  local label="$1"; shift
  echo "[$(date +%H:%M:%S)] RUN $label : $*" | tee -a "$SUM"
  .venv/bin/python scripts/eval/mot17.py \
    --preset mamba_whole_graph --detector SDP \
    "$@" \
    --output "$OUT/$label" > "$OUT/$label.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  $label FAILED rc=$rc" | tee -a "$SUM"; return; fi
  # pull the OVERALL METRICS block
  awk '/=== OVERALL METRICS ===/{f=1} f{print} /Rcll:/{if(f)exit}' "$OUT/$label.log" \
    | sed 's/^/    /' >> "$SUM"
  echo "" >> "$SUM"
}

ALLOFF="--no-gmc --no-relink-bridge-enabled --no-occ-state-enabled --oao-tau 0"

run bare           $ALLOFF
run plus_gmc       --gmc --no-relink-bridge-enabled --no-occ-state-enabled --oao-tau 0
run plus_bridge    --no-gmc --relink-bridge-enabled --no-occ-state-enabled --oao-tau 0
run plus_occ       --no-gmc --no-relink-bridge-enabled --occ-state-enabled --oao-tau 0
run plus_oao       --no-gmc --no-relink-bridge-enabled --no-occ-state-enabled --oao-tau 0.50 --oao-ramp-frames 25

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$SUM"
