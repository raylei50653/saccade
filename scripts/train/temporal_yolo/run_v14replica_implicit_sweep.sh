#!/usr/bin/env bash
# Train the implicit-consistency baseline at 3 seeds paired to T3→T1 (explicit).
# See run_v14replica_implicit_seed.sh for the recipe rationale (v14 git 77fcc262).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# saccade_tracking_ext needs torch's cu13 libs ahead of system CUDA (cublas conflict).
CU13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"

mkdir -p logs
SEEDS=(42 43 44)

echo "=== implicit baseline sweep: seeds ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do
    tag="s${s: -2}"
    log="logs/implicit_${tag}.log"
    echo ""
    echo "######################################################################"
    echo "### seed=${s}  ->  ${log}   ($(date '+%H:%M:%S'))"
    echo "######################################################################"
    bash scripts/train/temporal_yolo/run_v14replica_implicit_seed.sh "$s" 2>&1 | tee "$log"
done

echo ""
echo "=== IMPLICIT SWEEP DONE ($(date '+%H:%M:%S')) ==="
echo "Pair vs explicit: implicit s13/s14 vs t3t1 75.4/73.4 (IDF1), 65.3/62.5 (AssA)."
