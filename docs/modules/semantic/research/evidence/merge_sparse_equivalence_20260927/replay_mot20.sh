#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
R=results/xval459_20260924
O=$R/supp_sparse_resume_20260927
mkdir -p $O
SUB=$(git rev-parse b3ac5725)
M17=MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP
DT=$(ls datasets/DanceTrack/train | paste -sd,)
EQ=scripts/eval/experiments/merge_impl_equivalence.py
H=scripts/eval/experiments/run_output_layer_repair_chaining.py
run() { # tag substrate root seqs detector extra...
  local tag=$1 sub=$2 root=$3 seqs=$4 det=$5; shift 5
  .venv/bin/python $H --substrate $R/substrate_$sub --out $O/replay_$tag --artifact-dir $O/$tag \
    --data-root $root --split train --seqs "$seqs" --detector "$det" --substrate-commit $SUB "$@" > $O/replay_$tag.log 2>&1
}
run e2e_mot20 mot20 datasets/MOT20/MOT20 MOT20-01,MOT20-02 "" --arms merge_only
run mot20_full mot20 datasets/MOT20/MOT20 MOT20-01,MOT20-02,MOT20-03,MOT20-05 "" --arms base,merge_only --repeats 3 --repeat-arms base,merge_only
run mot20_full_noint mot20 datasets/MOT20/MOT20 MOT20-01,MOT20-02,MOT20-03,MOT20-05 "" --arms base,merge_only --no-interpolate
echo SUPP_DONE
