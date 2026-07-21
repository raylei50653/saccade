#!/usr/bin/env bash
# status: experiment
# Sweep the consistency-protection weight (route 3, mamba-t3t1-curriculum-20260613.md).
#
# Runs run_v14replica_consistency.sh for each weight sequentially. Order puts the
# decisive control-vs-treatment pair first (0 = pure T=3 full-grad control, 0.3 =
# recommended treatment), then the brackets 0.1 / 1.0. Each weight: Phase A
# (T=3 full-grad + consistency, 15ep) → Phase B (T=1 re-adapt, 15ep) → whole-graph
# tracking eval. Per-weight logs under logs/.
#
# Pre-flight stats (analyze_consistency_stats.py + dry run): 0% inactive pairs,
# 99% box carry-over, ~143 L2 terms/step; raw cons MSE 2.04 at the T3→T1 ckpt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# The saccade_tracking_ext C++ module needs torch's bundled cu13 libs ahead of the
# system CUDA libs (cublas version conflict) — see project_scan_cuda_backward.
CU13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"

mkdir -p logs
WEIGHTS=(0 0.3 0.1 1.0)

echo "=== consistency sweep: weights ${WEIGHTS[*]} ==="
for w in "${WEIGHTS[@]}"; do
    tag="cw${w/./p}"
    log="logs/consistency_${tag}.log"
    echo ""
    echo "######################################################################"
    echo "### weight=${w}  ->  ${log}   ($(date '+%H:%M:%S'))"
    echo "######################################################################"
    bash scripts/train/temporal_yolo/run_v14replica_consistency.sh "$w" 2>&1 | tee "$log"
done

echo ""
echo "=== SWEEP DONE ($(date '+%H:%M:%S')) ==="
echo "Results: results/mamba_v14replica_t3t1_consist_cw{0,0p3,0p1,1p0}/"
echo "Compare DetA/AssA vs T3→T1 (69.7/66.0) and T3T1→ssmft (71.0/62.8)."
