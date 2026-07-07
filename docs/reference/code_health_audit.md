# Code Health Audit

Last updated: 2026-06-29

## Verified Gates

Commands run from repo root:

```bash
uv run ruff check src tests scripts
uv run mypy .
uv run pytest
scripts/test_native.sh
```

Current result:

- `ruff`: passed
- `mypy`: passed
- `pytest`: 877 passed, 10 skipped, total coverage 48%
- native tests: 6/6 passed via `scripts/test_native.sh`

## Fixed In This Pass

- Unified perception/cognition event transport on Redis Stream `saccade:stream`.
  `EntropyTrigger.emit_event()` now writes through `RedisCache.add_to_stream()`;
  `PipelineOrchestrator` already consumes that stream.
- Updated health queue depth to inspect `XLEN saccade:stream`, and ensured the
  Redis connection closes even when the depth check fails.
- Fixed `ChromaStore.add_memory(..., embedding=...)` to pass one embedding per
  document instead of wrapping the vector in an extra list.
- Fixed `ChromaStore.hybrid_query(start_time=0.0)` so zero is a valid lower
  bound filter.
- Fixed native test entrypoint portability:
  `scripts/test_native.sh` now pins CUDA toolkit discovery and builds all native
  test targets before running `ctest`.
- Removed tracked hardcoded local paths from:
  - `scripts/model/compile_mamba_head_aot.py`
  - `scripts/tools/setup_services.sh`
- Synced storage/API docs to the current Stream-based event contract.
- Added FastAPI endpoint coverage for root, active object listing, object
  history, search result formatting, and error mapping.
- Migrated FastAPI startup/shutdown from deprecated `on_event` hooks to a
  lifespan handler.
- Removed stale mypy override entries for modules that no longer exist.
- Refreshed `scripts/README.md` inventory counts from tracked files.
- Added CPU-only unit coverage for `TRTFeatureExtractor` Python-side
  normalization, fused 3-part ReID fallback, default stability fallback, and
  LaSt-ViT top-k validation.
- Made `extract_parts_fused()` reject batches that are not a multiple of three
  before entering either the C++ bridge or Python fallback.
- Made `extract_with_stability()` and the Python LaSt-ViT helper reject
  `top_k_ratio` values outside `(0, 1]` before they can reach `torch.topk()` or
  the C++ bridge.
- Made `RTSPStreamer.start()` idempotent for an already-running ffmpeg process,
  added BGR frame shape validation, and made `stop()` wait/kill so subprocesses
  do not linger.
- Made `DALIRTSPOptimizer` reject invalid batch sizes and empty tensor lists
  before batch-padding logic can fail with an index error.
- Added CPU-only media tests for ffmpeg command construction, frame resizing,
  process lifecycle behavior, DALI batch padding, priming, and conversion
  interface handling.
- Fixed `AsyncEmbeddingDispatcher.submit()` empty-batch results to use the
  extractor's `feature_dim` instead of a hardcoded 768.
- Made `AsyncEmbeddingDispatcher.start()` idempotent and added CPU-only async
  coverage for empty submissions, worker extraction, fire-and-forget callbacks,
  and queue-skipping empty crop batches.
- Quoted `MediaMTXClient` GStreamer `location=` property values so RTSP URLs
  and local files containing spaces or pipeline metacharacters are not parsed as
  pipeline syntax.
- Made `MediaMTXClient.release()` clear pipeline/C++ references and join the
  GLib loop thread during cleanup. Added tests for first-frame timeout cleanup,
  idempotent release behavior, and pipeline quoting.
- Extracted `TRTYoloDetector._decode_outputs()` so TensorRT raw outputs can be
  decoded without launching inference again.
- Fixed `TRTYoloDetector.detect_batch()` to return immediately for empty input
  batches, and fixed empty detection class tensors to use `int32`.
- Fixed `BatchedDetectorProxy.detect_batch()` so it decodes the raw batcher
  output directly instead of passing `[1, 300, 6]` raw output back through
  `base.detect_batch()` as if it were an image tensor.
- Added CPU-only detector tests for raw output decoding, pose extra reshape,
  embedding side outputs, empty-batch behavior, and batched-proxy decode flow.
- Quoted `GstZeroCopyDecoder` GStreamer `location=` values for RTSP URLs and
  file paths so spaces and pipeline metacharacters are not parsed as pipeline
  syntax.
- Made `GstZeroCopyDecoder._on_new_sample()` return `Gst.FlowReturn.ERROR`
  when a pulled sample has no caps instead of raising an AttributeError.
- Added zero-copy decoder tests for RTSP/file pipeline quoting and missing caps
  handling.
- Fixed 12 race conditions (S1–S12) in the C++ `GstClient` GPU-decode buffer
  pool (`src/media/gst_client.cpp`). Extracted a GStreamer-independent
  `BufferPool` class (`include/media/buffer_pool.hpp`) with CAS-based slot
  acquisition, grow-only buffer reallocation, stream-synced teardown, and a
  proper `EMPTY→WRITING→READY→PROCESSING→EMPTY` state machine. Implemented the
  ADR-009 Python contract in `_on_cpp_frame`: `with frame_data` RAII +
  `sync_buffer` before read + `ExternalStream` stream-ordered handoff +
  `clone()` to decouple tensor lifetime from pool slots. Added 8 native race
  tests (`tests/native/test_gst_buffer_pool.cpp`, `ctest -R gst_buffer_pool`).
- Fixed `perception/eval/evaluator.py` softmax3 torch parameter caching so two
  same-shape models with different learned values do not share stale tensors.
- Made softmax3 feature-name mismatches fail with a clear `ValueError` instead
  of an incidental `KeyError`, and broadened environment flag parsing for
  common false values (`no`, `off`, and whitespace-wrapped values).
- Removed a tensor-copy warning from FP filter ranking and added CPU-only tests
  for the evaluator softmax3 cache, unsupported features, and env flag parsing.
- Fixed nested `scripts/eval/*/*.py` entrypoints that still computed repo root
  with fixed parent depth after being copied from root-level scripts. Direct
  execution from `appearance/`, `baselines/`, `detector/`, `diagnostics/`, and
  `experiments/` now resolves the workspace by repo markers instead of landing
  on `/repo/scripts` or `/repo/scripts/eval`.
- Added eval-script path guard tests so nested entrypoints cannot reintroduce
  fixed-depth repo-root resolution, and so scripts with unqualified eval-root
  imports add `scripts/eval` to `sys.path`.
- Collapsed the exact duplicate `scripts/eval` source entrypoint pairs to
  root-level compatibility wrappers plus canonical implementations under
  `diagnostics/`, `baselines/`, and `experiments/`. A duplicate scan now reports
  zero exact duplicate source groups among tracked `.py`/`.sh` eval scripts
  outside generated outputs.
- Added `scripts/eval/_redirect.py` and wrapper tests that verify canonical
  targets exist, CLI `sys.argv[0]` is set/restored, shell wrappers forward
  arguments, and root modules that are imported by tests still re-export their
  canonical APIs.
- Collapsed a second batch of legacy root-level `scripts/eval` entrypoints to
  conditional compatibility wrappers for canonical implementations under
  `diagnostics/`, `appearance/`, `detector/`, and `experiments/`.
- Updated Python wrappers so direct CLI execution skips package re-export and
  delegates through `_redirect`, while module imports still re-export the
  canonical APIs used by tests and downstream scripts.
- The eval duplicate scan now reports zero exact duplicate source groups and 31
  remaining duplicate basenames among tracked `.py`/`.sh` eval scripts outside
  generated outputs.
- Removed the last tracked hardcoded local paths from tools scripts:
  - `scripts/tools/gate_clean_color.py` now resolves its repo root via
    `Path(__file__).resolve().parents[2]` instead of a hardcoded
    `/home/ray/developer/ai/saccade`, matching the sibling
    `probe_relink_occlusion_signal.py` idiom.
  - `scripts/tools/analyze_kalman_h_signal.py` derives its rescreen-artifact
    defaults from the repo root's sibling (`<repo>/../saccade-rescreen/...`)
    rather than absolute `/home/ray/...` paths, and its usage docstring uses the
    relative form. A grep for `/home/` across tracked `scripts/**.py`/`.sh` now
    returns nothing.
  - Replaced the cosmetic `cd /home/ray/developer/ai/saccade` usage line in
    `scripts/eval/sweep_density_gating.py` with `cd <repo root>`.
- Triaged the remaining 13 non-identical duplicate `scripts/eval` entrypoint
  pairs into compatibility wrappers. Each root-level copy differed from its
  canonical subdirectory implementation only in the usage docstring path and the
  repo-root resolution method (fixed `parents[2]` vs marker-based); the
  subdirectory copy (under `diagnostics/`, `appearance/`, `baselines/`,
  `experiments/`) is the canonical version with marker-based root resolution.
  The root copies now delegate through `scripts/eval/_redirect.py`. After this
  pass, all 28 tracked duplicate eval basenames resolve to exactly one real
  implementation plus wrappers (zero basenames with more than one real impl), and
  `tests/unit/test_eval_script_paths.py::COMPAT_WRAPPERS` covers all 28.

## Remaining Findings

- Test coverage is still broad but shallow in important runtime boundaries.
  The full suite reports 48% total coverage. Notable low-coverage modules:
  `detector_trt.py` at 25%, `perception/eval/evaluator.py` at 19%,
  `feature_extractor.py` at 40%, `mediamtx_client.py` at 46%, and
  `zero_copy.py` at 55%. The C++ `BufferPool` race fixes are covered by native
  tests (`tests/native/test_gst_buffer_pool.cpp`, 8 scenarios) rather than the
  Python coverage report.
- `src/saccade/api/server.py` is now covered at 97%, but it still depends on
  global singleton stores. That is testable, but future dependency-injection
  cleanup would make startup/shutdown and endpoint testing simpler.
- `scripts/` cleanup is incomplete. Current tracked inventory is 134 files
  under `scripts/eval` and 68 under `scripts/tools`.
- `scripts/eval` duplicate-basename triage is complete: all 28 tracked duplicate
  basenames now resolve to a single canonical subdirectory implementation plus a
  root-level `_redirect` compatibility wrapper (28 wrappers + one shell wrapper).
  No tracked eval basename has more than one real implementation. Future work is
  limited to deciding whether any of the canonical implementations are
  themselves archive candidates, which is a content-relevance question rather
  than a duplication one.
- `GET /objects/{obj_id}` still assumes richer object history than
  `RedisCache.update_object_track()` stores. This is already documented in
  `docs/modules/storage/api_spec.md`, but the endpoint remains unsuitable as a
  complete dwell-time/history API until the schema is expanded.
- Hardcoded local paths have been removed from tracked `scripts/`. A grep for
  `/home/` across tracked `scripts/**.py`/`.sh` returns nothing. (Rescreen-style
  tooling still assumes a sibling `saccade-rescreen` checkout, but now resolves
  it relative to the repo root rather than an absolute home path.)

## Suggested Next Pass

1. Review the canonical `scripts/eval` implementations for content relevance and
   archive any that are obsolete (duplication itself is now resolved).
2. Extend `feature_extractor.py` coverage around TRT/native paths using fake
   engine/context objects where practical; real engine loading remains outside
   unit-test scope.
3. Add focused tests for remaining detector/TRT context paths, evaluator frame
   orchestration helpers, and `MediaMTXClient`/`GstZeroCopyDecoder`
   sample/callback paths that require heavier GStreamer/GPU fakes.
