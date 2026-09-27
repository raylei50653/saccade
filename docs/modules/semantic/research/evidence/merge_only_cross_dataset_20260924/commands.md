# Commands — merge-only cross-dataset validation (#459, 2026-09-24)

Substrates captured on clean `main` `b3ac5725` (post-#457). All replays ran at clean
`eb465b49` (`dirty=false`); harness
commits `0df87c65` / `30b94e7b` (measurement-only instrumentation: stage wall
time, accepted merge rows, detector label; merge decisions unchanged). All
GPU work ran under `tools/resctl.py run machine-bench`.

```bash
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
common="--preset mamba_whole_graph_m --double-buffer --no-gpu-decode --no-interpolate-tracklets"

# 1. Frozen tracker substrates (interp OFF). MOT20/DanceTrack captured twice
#    in separate processes for determinism (substrate_*_rep).
.venv/bin/python scripts/eval/mot17.py $common --detector SDP --output results/xval459_20260924/substrate_mot17
.venv/bin/python scripts/eval/mot17.py $common --data-root datasets/MOT20/MOT20 --split train --output results/xval459_20260924/substrate_mot20
.venv/bin/python scripts/eval/mot17.py $common --data-root datasets/DanceTrack --split train --output results/xval459_20260924/substrate_dancetrack

# 2. base vs merge_only replay, n=3 in-process repeats of each arm,
#    shipping interpolation applied after the identity stage.
.venv/bin/python scripts/eval/experiments/run_output_layer_repair_chaining.py \
  --substrate results/xval459_20260924/substrate_<ds> --out results/xval459_20260924/replay_<ds> \
  --artifact-dir <this dir>/<ds> --data-root <root> --split train --seqs <seqs> --detector <SDP|""> \
  --substrate-commit b3ac5725 --arms base,merge_only --repeats 3 --repeat-arms base,merge_only
#   mot17:      datasets/MOT17, 7 SDP sequences
#   mot20:      datasets/MOT20/MOT20, MOT20-01,MOT20-02 (03/05 OOM, see mot20/full_set_oom_excerpt.txt)
#   dancetrack: results/xval459_20260924/dancetrack_6d (symlink mirror, 6-digit frame names), all 40 train sequences

# 3. Interpolation-off decomposition (same substrates, n=1): add --no-interpolate,
#    --artifact-dir <this dir>/<ds>/no_interp.

# 4. Accepted-merge event labelling against GT
.venv/bin/python scripts/eval/experiments/characterize_merge_events.py \
  --results <this dir>/<ds>/results.json --substrate results/xval459_20260924/substrate_<ds> \
  --data-root <root> --out <this dir>/<ds>/merge_events.json
```

DanceTrack mirror: `train/<seq>/img1/%06d.jpg -> datasets/DanceTrack/train/<seq>/img1/%08d.jpg`,
`gt/` and `seqinfo.ini` symlinked. Pixels, GT and frame numbers are unchanged.

Operating point (not retuned): `mobilenetv4_reid` / `models/embedding/mobilenetv4_reid_visclean_224.engine`,
merge `max_cost=0.45`, `max_gap=60`, `n_samples=50`, `pool_frac=0.3`, `cheb_lambda=2.0`, `k2=6`, `max_fwd=50`,
`fuse_lambda=0.3`; interpolation `max_gap=35`, `min_track_len=5`, `min_h=0`.
