#!/usr/bin/env bash
# Confirmation-gate sweep: shorten establishment latency (the 57% "establish/
# confirm" bucket of detected-but-untracked FN, see analyze_assoc_fn.py) while
# guarding precision/IDs. Each strategy writes a full per-seq output dir so the
# downstream per-seq consistency check (avoid ntt0.20-style overfit) can run.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

SEQS="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP"

run() {  # name, flags...
  local name="$1"; shift
  echo "=== [$name] $* ==="
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph \
    --sequences "$SEQS" --output "results/conf_${name}" "$@" \
    > "results/conf_${name}.log" 2>&1
  grep -E "^\s*(IDF1|MOTA|HOTA|AssA|IDs|Rcll|Prcn):" "results/conf_${name}.log" | tr '\n' ' '
  echo ""
}

run baseline
run cst040  --confirm-score-thresh 0.40
run cst035  --confirm-score-thresh 0.35
run streak2 --confirm-streak 2
run combo   --confirm-score-thresh 0.40 --confirm-streak 2
echo "ALL DONE"
