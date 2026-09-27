#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
R=results/xval459_20260924
O=$R/qualified_ordered_20260927
mkdir -p "$O"
EQ=scripts/eval/experiments/merge_impl_equivalence.py
.venv/bin/python "$EQ" --substrate "$R/substrate_mot17" --data-root datasets/MOT17 --seqs MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP --out "$O/equivalence_mot17.json" > "$O/mot17.log" 2>&1
.venv/bin/python "$EQ" --substrate "$R/substrate_mot20" --data-root datasets/MOT20/MOT20 --seqs MOT20-01,MOT20-02,MOT20-03,MOT20-05 --out "$O/equivalence_mot20.json" > "$O/mot20.log" 2>&1
DT=$(ls datasets/DanceTrack/train | paste -sd,)
.venv/bin/python "$EQ" --substrate "$R/substrate_dancetrack" --data-root "$R/dancetrack_6d" --seqs "$DT" --out "$O/equivalence_dancetrack.json" > "$O/dancetrack.log" 2>&1
