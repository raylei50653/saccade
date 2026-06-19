# Saccade Pipeline Flow Reference

> This reference is intentionally short. The detailed current dataflow lives in
> [DATAFLOW.md](../DATAFLOW.md). Keeping one detailed flow document avoids
> stale duplicate stage tables.
>
> Last updated: 2026-06-19.

---

## 1. Scope

This file maps the current MOT17 evaluation path to its source files and stable
contracts. It does not record ablation results or profiling tables.

Primary path:

- [scripts/eval/mot17.py](../../scripts/eval/mot17.py)
- [src/saccade/perception/eval/evaluator.py](../../src/saccade/perception/eval/evaluator.py)

Compatibility shim:

- [src/saccade/perception/eval/runner.py](../../src/saccade/perception/eval/runner.py)
  re-exports `run_eval` from `evaluator.py`; it is not the implementation source
  of truth.

Current recommended baseline:

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP
```

Preset source:

- [configs/presets/mamba_whole_graph.yaml](../../configs/presets/mamba_whole_graph.yaml)

---

## 2. Stage Order

`evaluator.py` currently profiles these top-level stages:

```text
fetch
ingest_preprocess
detect
postprocess
reid_bank_sync
reid_budget
reid_crop
reid_extract
lazy_reid
gmc
track
materialize
bg_relink_wait
relink_write
frame_total
```

See [DATAFLOW.md](../DATAFLOW.md) for stage responsibilities and current
`mamba_whole_graph` behavior.

---

## 3. Stable Source Map

| Area | Primary implementation |
|:--|:--|
| Eval orchestration | [evaluator.py](../../src/saccade/perception/eval/evaluator.py) |
| Legacy import shim | [runner.py](../../src/saccade/perception/eval/runner.py) |
| Detection helpers | [detection.py](../../src/saccade/perception/eval/detection.py) |
| Native postprocess facade | [include/tracking/pipeline.hpp](../../include/tracking/pipeline.hpp), [src/tracking/pipeline.cpp](../../src/tracking/pipeline.cpp) |
| GPU tracker | [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu), [include/tracking/tracker_gpu.hpp](../../include/tracking/tracker_gpu.hpp) |
| Python tracker wrapper | [tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py) |
| Semantic / output relink | [relink.py](../../src/saccade/perception/eval/relink.py) |
| Event queue | [redis_cache.py](../../src/saccade/storage/redis_cache.py), [entropy.py](../../src/saccade/perception/entropy.py) |
| Cognition memory path | [orchestrator.py](../../src/saccade/cognition/orchestrator.py), [chroma_store.py](../../src/saccade/storage/chroma_store.py) |
| Health | [health.py](../../src/saccade/pipeline/health.py) |

---

## 4. Related Contracts

- Stable architecture boundaries: [architecture/README.md](../architecture/README.md)
- MOT17 config and preset defaults: [mot17_default_config.md](mot17_default_config.md)
- Event / API / storage schema: [api_spec.md](../modules/storage/api_spec.md)
- Current work ordering: [TODO.md](../TODO.md)

When a pipeline behavior changes, update [DATAFLOW.md](../DATAFLOW.md) first.
Only update this file if source locations, stage names, or contract links change.
