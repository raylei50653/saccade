# Commands — output-layer repair chaining revalidation (2026-09-12)

Substrate captured on clean `main` `85b95ac151bc7dd399ffc25fc83ac103c0e7321a`.
Replay harness is measurement-only; it does not change shipping presets or
evaluator dispatch.

## 1. Frozen tracker substrate (interpolation off)

```bash
uv run python scripts/eval/mot17.py \
  --preset mamba_whole_graph_m \
  --detector SDP \
  --double-buffer \
  --no-gpu-decode \
  --no-interpolate-tracklets \
  --output results/olr_reval_20260912/substrate
```

Pre-interpolation OVERALL from this capture (not the scored base arm):
IDF1 79.5 / MOTA 79.2 / HOTA 73.2 / AssA 72.7 / IDs 737.
Shipping interpolation is applied later, identically, after each arm's identity
stages.

## 2. Five-arm replay + repeats

```bash
uv run python scripts/eval/experiments/run_output_layer_repair_chaining.py \
  --substrate results/olr_reval_20260912/substrate \
  --out results/olr_reval_20260912 \
  --artifact-dir docs/modules/semantic/research/evidence/output_layer_repair_chaining_revalidation_20260912 \
  --substrate-commit 85b95ac151bc7dd399ffc25fc83ac103c0e7321a \
  --expected-substrate-sha256 docs/modules/semantic/research/evidence/output_layer_repair_chaining_revalidation_20260912/substrate_sha256.json \
  --repeats 3 \
  --repeat-arms merge_only,handover_then_merge,merge_then_handover
```

Operating points (not retuned):

- embedding: `mobilenetv4_reid` / `models/embedding/mobilenetv4_reid_visclean_224.engine`
- merge: `max_cost=0.45`, `max_gap=60`, `n_samples=50`
- handover: `min_head=2`, `margin=0.05`, `max_cost=0.45`, `decide_n=5`
- interpolation after stages: `max_gap=35`, `min_track_len=5` (shipping `mamba_whole_graph_m`)
