#!/usr/bin/env bash
# status: stable
# Threshold-strategy sweep on the full MOT17 train SDP set.
# Grounded in docs/modules/detection/research/mamba-score-distribution-20260613.md:
# the tracker thresholds sit in the saturated left tail and act mainly as a
# small-object / occlusion-miss filter. Each strategy perturbs one knob and we
# read end-to-end IDF1/HOTA/MOTA/AssA/IDs/Rcll/Prcn.
set -uo pipefail
cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

SEQS="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP"
LOGDIR="results/threshold_strategies"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "strategy\toverride\tIDF1\tMOTA\tHOTA\tDetA\tAssA\tIDs\tRcll\tPrcn\n" > "$SUMMARY"

# name : extra CLI flags (threshold overrides on top of preset defaults)
run_one() {
  local name="$1"; shift
  local override="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo "=== [$name] $override ==="
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph \
    --sequences "$SEQS" --output "results/strat_${name}" "$@" > "$log" 2>&1
  # Parse the OVERALL METRICS block.
  local idf1 mota hota deta assa ids rcll prcn
  idf1=$(grep -E "^\s*IDF1:" "$log" | tail -1 | grep -oE "[0-9.]+")
  mota=$(grep -E "^\s*MOTA:" "$log" | tail -1 | grep -oE "[0-9.]+")
  hota=$(grep -E "^\s*HOTA:" "$log" | tail -1 | grep -oE "[0-9.]+")
  deta=$(grep -E "^\s*DetA:" "$log" | tail -1 | grep -oE "[0-9.]+")
  assa=$(grep -E "^\s*AssA:" "$log" | tail -1 | grep -oE "[0-9.]+")
  ids=$(grep -E "^\s*IDs:" "$log" | tail -1 | grep -oE "[0-9]+")
  rcll=$(grep -E "^\s*Rcll:" "$log" | tail -1 | grep -oE "[0-9.]+")
  prcn=$(grep -E "^\s*Prcn:" "$log" | tail -1 | grep -oE "[0-9.]+")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$override" "$idf1" "$mota" "$hota" "$deta" "$assa" "$ids" "$rcll" "$prcn" \
    | tee -a "$SUMMARY"
}

run_one baseline      "defaults(ntt0.28,tt0.05,mt0.10)"
run_one ntt020        "new_track_thresh=0.20"   --new-track-thresh 0.20
run_one ntt015        "new_track_thresh=0.15"   --new-track-thresh 0.15
run_one ntt035        "new_track_thresh=0.35"   --new-track-thresh 0.35
run_one tt002         "track_thresh=0.02"       --track-thresh 0.02
run_one mt005         "mid_thresh=0.05"         --mid-thresh 0.05
run_one combo         "ntt0.20+tt0.02"          --new-track-thresh 0.20 --track-thresh 0.02

echo ""
echo "=== SUMMARY ($SUMMARY) ==="
column -t -s $'\t' "$SUMMARY"
