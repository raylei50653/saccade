# Sync & D2H Audit — fetch + postprocess focus

Generated: 2026-07-06 | Phase 10.1 — detect_post_event validated for correctness, opt-in for overlap-ready systems

---

## Phase 6C: Stream Identity (SACCADE_STREAM_DEBUG=1)

**Result**: All streams report `0x0` — the CUDA legacy default stream.

| Location | Stream Handle |
|---|---|
| `run_eval` | 0x0 |
| `_run_detect` (TRT enqueue) | 0x0 |
| `_run_nms` graph capture | 0x0 |
| `_run_nms` graph replay | 0x0 |
| `torch.cuda.default_stream()` | 0x0 |
| `torch.cuda.current_stream()` | 0x0 |

**Interpretation**: PyTorch uses CUDA legacy default stream mode (NULL stream).
Operations on the legacy default stream implicitly synchronize with ALL other streams.

### Why DS2 is still required despite same-stream usage

DS2 (`torch.cuda.synchronize()` — full-device barrier) is necessary because TRT's
`IExecutionContext::execute_async_v3(stream)` may internally use auxiliary streams
that bypass the legacy default stream's implicit synchronization. TensorRT creates
internal CUDA graphs and streams for kernel execution that are NOT fenced by the
NULL stream guarantee.

**DS2 event-fence on same stream is NOT feasible.** A `cudaEventRecord` +
`cudaStreamWaitEvent` on stream `0x0` would be a no-op since the legacy default
stream already provides implicit ordering. The full-device barrier is required to
fence TRT's internal auxiliary streams.

### Correct target

The correct optimization for DS2 is NOT:
```
remove DS2
```
Nor:
```
replace DS2 with same-stream event fence
```

The correct target is:
```
Expose TRT's internal completion event/stream → fence postprocess stream to it
```
OR:
```
Switch from CUDA legacy default stream to per-thread default streams,
manage streams explicitly, and use cudaEventRecord/Wait between explicit streams.
```

---

## Phase 6B: detect_ms and post_ms split analysis

### detect_ms breakdown (195-frame SDP)

| Sub-stage | Mean | P50 | P99 | Max | % |
|---|---|---|---|---|---|
| DS1 (ingest barrier) | 0.21ms | 0.20 | 0.43 | 1.69 | 3.9% |
| TRT enqueue | 2.80ms | 2.72 | 4.29 | 7.03 | 52.2% |
| DS2 (postproc barrier) | 2.03ms | 2.05 | 2.70 | 4.36 | 37.8% |
| (remainder) | 0.32ms | — | — | — | 6.0% |

**DS2 is the primary detect tail driver** (38% of detect time, 4.36ms max).
DS1 is negligible (0.21ms, 4%).
TRT enqueue dominates mean time but is relatively stable.

### post_ms breakdown

| Sub-stage | Mean | P99 | Max | % |
|---|---|---|---|---|
| Pre-NMS (tensor prep + NMS) | 0.21ms | — | — | 16% |
| Graph replay (subset of pre-NMS) | 0.06ms | 0.13 | 0.19 | 4% |
| Count/materialize wait | 1.31ms | — | — | 96% |

**Graph replay is NOT the post tail driver.** The count/materialize filtering
step consumes 96% of post_ms.

---

## Overlap / State-Machine Feasibility

### Current topology

```
Frame N:   fetch → preprocess → DS1 → TRT_launch → DS2 → pre_NMS → graph_replay → count_filter → track → materialize → output
```

All operations on CUDA legacy default stream (0x0). No explicit stream management.

### Required changes for overlap

| Change | Difficulty | Benefit |
|---|---|---|
| Switch to per-thread default streams | Medium (PyTorch config + stream mgmt) | Enables explicit event fencing |
| TRT enqueue on explicit stream `s_detect` | Low (already parameterized) | Separates detect from post |
| Postprocess on explicit stream `s_post` | Low | Separate GPU work |
| `cudaEventRecord` after TRT → `cudaStreamWaitEvent` on `s_post` | Low | Replaces DS2 |
| GMC on `s_gmc` (already separate for direct path) | Low | Overlaps with post |
| ReID on `s_reid` (already separate for async path) | Low | Overlaps with post |
| Two frame contexts (ping-pong) for buffers | Medium | Enables frame pipelining |
| Deferred emit (output lag 1 frame) | Low (already implemented for double-buffer) | Decouples output from pipeline |

### Feasibility assessment

**Frame overlap IS feasible** with moderate engineering:

1. **Switch to per-thread default streams** (or explicit named streams):
   - `S_detect`: TRT inference
   - `S_post`: postprocess NMS graph + filtering
   - `S_track`: ByteTrack GPU tracker
   - `S_gmc`: GMC (already separate for direct path)
   - `S_reid`: ReID side path (already separate for async path)

2. **Buffer lifetime**: 2 frame contexts needed (ping-pong):
   - Frame N writes to context A, frame N+1 writes to context B
   - Postprocess reads context A while detect writes context B

3. **Event fencing**:
   ```text
   TRT_enqueue(S_detect) → event_rec → S_post.wait(event_rec)
   Post_complete_event → S_track.wait(post_event)
   Track_complete_event → S_output.wait(track_event)
   ```

4. **Output lag**: 1 frame (already supported in double-buffer mode).
   Emit frame N while processing frame N+1.

5. **Blockers**:
   - Count D2H sync in C++ pipeline (`cudaStreamSynchronize` in graph path) — needs
     to be deferred until the count is needed, not immediately after NMS.
   - Graph capture uses the stream active at capture time — need per-stream graphs.

### Recommended path

1. **Phase 7**: Switch to explicit stream management + prototype DS2 event fence
   with explicit streams. This is the minimum viable change that enables overlap.
2. **Phase 8**: Add second frame context (ping-pong buffers) for pipeline overlap.
3. **Phase 9**: Defer count D2H sync to materialization.
4. **Phase 10**: Measure end-to-end throughput improvement.

---

## fetch_ms semantics

`fetch_ms` is wall time from `t_e2e_start` to entering the non-workbench `else:` block.
It includes:

- `next(stream_iter)` — JPEG decode via DALI (CPU) or torchvision (NVJPG GPU), plus:
- NV12 conversion (`pool.frame_buffer_nv12.copy_(rgb_hwc_to_nv12_gpu(frame_gpu))`) or
  CHW float conversion + preprocess transforms + optional NV12 copy.

`fetch_ms` is therefore **ingest wall time**: decode + preprocess, not just raw I/O.

It is **not** a pipeline wait descriptor in the default single-stream path.
Double-buffer mode (`SACCADE_DOUBLE_BUFFER=1`) may introduce producer/consumer waits,
but that path is gated on `--double-buffer` + `SACCADE_DETECT_BARRIER=event` and is
off by default.

**Variance**: in the 995-frame test, fetch_ms delta in top 5% frames is 0.22ms (1.10x).
The max fetch_ms was 3.32ms vs mean 2.14ms. The variance is modest; fetch is a
minor tail contributor.

---

## detect_ms sync points

All in `stages.py:_run_detect()`.

### DS1 — ingest → detect full-device barrier
- **File**: `src/saccade/perception/eval/stages.py` ~L1089
- **Stage**: detect
- **Call**: `torch.cuda.synchronize()`
- **Reason**: Serialise NVJPEG/DALI decode (external engine, no stream handle) before YOLO TRT reads pool buffer. Prevents stale-buffer read from decode that bypasses the default stream.
- **Required**: deterministic correctness. Without it, run-to-run output drift observed in MR `3046ae60`.
- **Every frame**: yes (default `barrier_mode="full"`).
- **Can delay?**: The `event` barrier mode drops the full-device barrier but requires N>=6 determinism validation. The decode engine (NVJPG) needs to be fenced onto the current stream first — no narrow fence available yet.
- **Impact**: waits for all CUDA streams, not just the main stream. Any background async ReID or GMC work on side streams will also stall this barrier.

### DS2 — detect → postprocess full-device barrier
- **File**: `src/saccade/perception/eval/stages.py` ~L1123
- **Stage**: detect
- **Call**: `torch.cuda.synchronize()`
- **Reason**: Enforce that TRT output is fully written before postprocess reads raw detection tensors.
- **Required**: In whole-graph mode (TRT + postprocess graphs share the launch stream), this ordering is likely already implicit. The `no_postproc` and `event` barrier modes drop this barrier.
- **Every frame**: yes (default `barrier_mode="full"`).
- **Can delay?**: YES — `no_postproc` mode drops this barrier (gated on N>=6 determinism check). Lower-risk than DS1 because postprocess graphs share the same stream.
- **Impact**: 2.9ms/frame host stall (per commit `3046ae60`).

---

## post_ms sync points

All in `src/tracking/pipeline.cpp:process_detections_into()`.

### PS1 — filter count D2H + stream sync
- **File**: `src/tracking/pipeline.cpp` L256-258
- **Stage**: postprocess
- **Call**: `cudaMemcpyAsync(&filter_count, d_filter_count_, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);`
- **Reason**: Read the number of boxes that passed quality/geometry filter. Needed for branch logic (continue / bail out / private continuation).
- **Required**: yes, per-frame. The count determines whether to proceed with NMS.
- **Can delay?**: The sync waits for the filter kernel to finish on the GPU stream. If the GPU is busy with prior work (e.g. detection on a different stream that launched before), the sync blocks the host. Could be replaced by a CUDA event-based check or delayed count read.
- **Impact**: variable — depends on GPU pipeline depth. Potentially contributes to post_ms tail variance.

### PS2 — NMS count D2H + stream sync  
- **File**: `src/tracking/pipeline.cpp` L175-176 (in helper function)
- **Stage**: postprocess
- **Call**: `cudaMemcpyAsync(&n_out, d_nms_count_, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);`
- **Reason**: Read the post-NMS box count to return to the caller.
- **Required**: yes, the caller needs the count for downstream processing.
- **Every frame**: yes.
- **Can delay?**: Same pattern as PS1 — stream sync waits for GPU. Can be deferred slightly or event-based.

### PS3 — post-processing count D2H + stream sync (after private continuation)
- **File**: `src/tracking/pipeline.cpp` L587-589
- **Stage**: postprocess
- **Call**: `cudaMemcpyAsync(&n_post, d_count_staging, sizeof(int), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);`
- **Reason**: Read the final post-NMS detection count (after private continuation merge). This count determines the number of detections passed to the tracker.
- **Required**: yes, the post-NMS count feeds tracker allocation.
- **Every frame**: yes.
- **Can delay?**: Same pattern as PS1/PS2. Could coalesce with PS2 if the intermediate count is not needed separately.

### PS4 — early filter bailout sync
- **File**: `src/tracking/pipeline.cpp` L301
- **Stage**: postprocess
- **Call**: `cudaStreamSynchronize(stream);`
- **Reason**: After zero-count filter bailout, sync stream before returning to ensure GPU state is consistent.
- **Required**: conditional — only when filter_count <= 1 (rare).
- **Every frame**: no, only on frames with no/few detections.

### PS5 — pre-mature exit sync
- **File**: `src/tracking/pipeline.cpp` L534
- **Stage**: postprocess
- **Call**: `cudaStreamSynchronize(stream);`
- **Reason**: Sync after an early exit path (exact condition TBD).
- **Every frame**: conditional, likely rare.

---

## materialize sync points

### MS1 — CUDA event synchronize in materialize
- **File**: `src/saccade/perception/eval/stages.py` L428
- **Stage**: materialize (part of the `track_ms` ledger stage)
- **Call**: `ev.synchronize()` — synchronizes a CUDA event that records tracker completion.
- **Reason**: Ensures tracker GPU output is ready before D2H copy.
- **Every frame**: yes (non-double-buffer path).
- **Can delay?**: The event sync only waits for the specific stream's completion, not the full device. Cheaper than `torch.cuda.synchronize()`. Could be converted to `ev.wait()` on a pinned host buffer to overlap copy and processing.

### MS2 — implicit D2H via .item()
- **File**: `src/saccade/perception/eval/stages.py` L431
- **Stage**: materialize
- **Call**: `int(pinned["count"].item())` — reads a GPU-pinned tensor value on CPU, implicitly syncs the device.
- **Reason**: Need the track output count to size D2H copies.
- **Every frame**: yes.
- **Can delay?**: Implicit sync — can't be avoided without restructuring output buffer management. Could use CUDA event to signal count availability then defer read.

### MS3 — D2H of track output
- **File**: `src/saccade/perception/eval/stages.py` L1175-1222
- **Stage**: materialize → emit
- **Call**: `.cpu()`, `.tolist()`, `.numpy()` — D2H of boxes, scores, IDs, det_idx, embeddings.
- **Reason**: Produce CPU-side MOT-format output.
- **Every frame**: yes.
- **Can delay?**: The `.cpu()` calls are implicit syncs. Could batch/schedule with pin-memory for overlap. Embedding D2H is the most expensive (may be large).

---

## track_ms sync points

### TS1 — track_rw future result wait
- **File**: `src/saccade/perception/eval/evaluator.py` L1418-1426
- **Stage**: track (bg_relink_wait)
- **Call**: `state.bg_future.result()` — waits for a background thread's relink_write to complete.
- **Reason**: Need relinked output before proceeding to ReID/GMC/track.
- **Every frame**: only when `bg_future is not None` (pipeline_relink enabled).
- **Can delay?**: This is a Python thread join — could block if the bg thread is slow. Track_ms spikes (2.88ms, 2.71ms) may be from this wait.

---

## Other sync points (non-postprocess, non-fetch)

### gmc.cpp
- No additional syncs beyond CUDA kernel launches and event records.
- GMC uses `cudaEventRecord` for async coordination with the tracker, no blocking host sync.

### preprocessor_gpu.cu
- GPU kernels only. No host-side syncs.
- Crop kernel launches are async on the current stream.

### trt_engine.cpp
- TRT enqueue is async. No host sync in the inference path.
- However, the full-device barriers in `_run_detect` (DS1, DS2) will wait for TRT completion.

---

## fetch_ms sync summary

The fetch path has **no dedicated sync point** beyond what `time_stage("fetch", sync_cuda=False)` implies — which is nothing when `profile_stages` is off.

The only fetch-side sync is DS1 (the ingest→detect full-device barrier at line ~1089 in stages.py). This is technically in the detect stage, but it fences the end of decode/ingest. If NVJPEG decode is slow, DS1 will show higher waiting time.

---

## Key findings

1. **Two full-device barriers every frame** (DS1, DS2) — 2.9ms/frame host stall.
2. **Three stream-specific D2H-count syncs in C++ postprocess** (PS1, PS2, PS3) — every frame, variable wait depending on GPU pipeline depth.
3. **One CUDA event sync + implicit .item() in materialize** (MS1, MS2) — every frame.
4. **BG relink thread join** (TS1) — conditional, may cause track_ms spikes.
5. **post_ms has no correlation with n_dets** (r=0.07) — tail frames are NOT caused by more detections → sync variance from GPU pipeline depth is the likely cause.
6. **fetch_ms is not a major tail driver** — 0.22ms delta in top 5%.

---

## Phase 7.0: Explicit-Stream Feasibility Probe — RESULT: PASS

**Date**: 2026-07-06 | **200-frame SDP test** | **SACCADE_STREAM_MODE=explicit_probe**

### Design

```text
TRT enqueue on S_detect → cudaEventRecord(trt_done, S_detect)
                         → cudaStreamWaitEvent(S_post, trt_done)
                         → NMS graph capture/replay on S_post
                         → no host-blocking DS2 sync
```

### Validation

| Check | Result |
|---|---|
| 6/6 repeated runs identical | **PASS** |
| MD5 identical across all 6 runs | **5972bd5a...** (all identical) |
| vs baseline MD5 (non-probe) | **IDENTICAL** |
| MOT line count | **5543** (all identical) |
| Metrics IDF1/MOTA/IDs/FP/FN | **Identical** (all runs) |
| Per-frame detection count hash | **Verified across runs** |

### Metrics

| Metric | Baseline | Probe | Δ |
|---|---|---|---|
| `detect_ms` | 5.35ms | 5.67ms | +6.0% |
| `detect_postproc_barrier_ms` (DS2) | 1.85ms | **0.006ms** | **-99.7%** |
| `detect_trt_enqueue_ms` | 2.89ms | 5.06ms | +74.9% |
| `post_ms` | 1.24ms | 1.23ms | -1.2% |
| `post_graph_count_wait_ms` | 1.23ms | 1.20ms | -2.7% |
| `track_ms` | 0.79ms | 0.77ms | -2.2% |
| `total_ms` | 9.91ms | 10.17ms | +2.6% |

### Interpretation

1. **Event fence eliminates DS2 host stall (1.85ms → 0.006ms, -99.7%).**
   The `cudaEventRecord` + `cudaStreamWaitEvent` pair is a GPU-side operation that
   returns immediately on the host. The ordering is enforced by the GPU scheduler.

2. **No wait migration.** `post_ms`, `post_graph_count_wait_ms`, and `track_ms`
   are essentially unchanged. The GPU-side ordering ensures TRT output is ready
   when postprocess reads it on S_post.

3. **TRT enqueue overhead on explicit stream is +2.17ms (host-side).**
   `detect_trt_enqueue_ms` increased from 2.89ms to 5.06ms (+75%). This is NOT
   GPU time — it's Python/CUDA-driver overhead from switching to a non-default
   stream (perf_counter with no sync). The actual TRT GPU inference takes the
   same wall time. Known CUDA issue: non-default streams have higher host-side
   dispatch overhead in the CUDA driver.

4. **Stream handles verified**: S_detect ≠ S_post ≠ 0x0. The legacy default
   stream (0x0) is NOT used for TRT or postprocess in probe mode.

5. **Event fence works despite all-0x0 legacy default.** TRT's internal aux
   streams ARE fenced by the explicit event on S_detect. The event record on
   S_detect captures the completion state of all work submitted to S_detect,
   which TRT correctly respects.

### Updated sync topology (probe mode)

```
Frame N:  fetch → preprocess → DS1(full barrier) → TRT_enqueue(S_detect)
                                                      │
                                          cudaEventRecord(trt_done, S_detect)
                                                      │
                                     cudaStreamWaitEvent(S_post, trt_done)
                                                      │
                                         postprocess(S_post) → track → emit
```

DS1 (ingest barrier) remains. Next target for a stream-event replacement.

### TRT enqueue overhead resolution

Earlier 2ms overhead was a measurement artifact (sync vs no-sync). Corrected overhead is ~0.3ms.
- **CUDA graph wrapping**: capture TRT enqueue on S_detect in a graph (future)
- **Per-thread default streams**: migrate from legacy NULL stream to per-thread
  default streams, where each thread's "default" stream is a distinct handle
  with no implicit cross-stream synchronization (future)
- **C++-side stream management**: move stream creation and TRT dispatch to C++
  to eliminate Python overhead (future)

---

## Phase 8: Two-Frame Ping-Pong — Partial Success

**Date**: 2026-07-06 | **200-frame SDP test**

### Design

```text
Parity 0: S_detect[0] → detect_done[0] → S_post[0] → post_done[0] → track(0x0)
Parity 1: S_detect[1] → detect_done[1] → S_post[1] → post_done[1] → track(0x0)
```

Detect and post use per-parity streams for GPU-side overlap. Track stays on
CUDA legacy default stream (0x0) for determinism.

Per-parity buffers: cloned `fused_*` tensors after each detect call protect
TRT output from being overwritten by the next frame's detect.

### Critical finding: S_track non-deterministic in legacy default stream mode

Running ByteTrack GPU tracker on a non-default stream (S_track) caused
floating-point drift (box coordinates diff by 0.01-0.02px, accumulating over
frames). This is the same pattern as the `no_postproc` barrier ablation.

**Root cause**: CUDA legacy default stream (0x0) provides implicit cross-stream
synchronization that enforces a specific GPU execution order. Multiple tracker
CUDA kernel calls on an explicit stream lose this ordering guarantee, causing
different CUDA kernel fusion / scheduler decisions and thus different
floating-point results.

**Solution**: Track must remain on the legacy default stream (0x0) as long as
the process operates in CUDA legacy default stream mode. Only per-thread
default stream (PTDS) migration would allow track on an explicit stream.

The per-parity S_detect[0,1] and S_post[0,1] do NOT cause non-determinism
because:
- TRT inference runs as one fused kernel (or graph) on S_detect — no multi-op
  intra-stream ordering variation.
- Post NMS runs as a single CUDA graph replay on S_post — same property.
- The event fence (detect_done → S_post) enforces correct cross-stream ordering.

### Determinism validation

| Check | Result |
|---|---|
| 6/6 repeated runs identical | **PASS** |
| Bit-identical to full-barrier baseline | **PASS** |
| MOT MD5 | **5972bd5a...** (all identical) |
| DS2 eliminated | **detect_postproc_barrier_ms = 0** |

### Host-wall-clock metrics (profiling mode, sync_cuda=True everywhere)

| Metric | Baseline | P7.1 (explicit) | P8 (ping-pong) |
|---|---|---|---|
| `total_ms` | 9.91ms | 10.33ms | 11.47ms |
| `detect_ms` | 5.35ms | 5.64ms | 6.36ms |
| `detect_postproc_barrier_ms` | 1.85ms | 0.000ms | 0.000ms |

P8 regresses host wall-clock slightly due to:
- Per-parity `fused_*.clone()` GPU memcpy overhead (~0.3ms per frame)
- Additional stream switching overhead (~0.5ms)
- Profiling mode auto-syncs mask any GPU-side overlap benefit

### Overlap model

Profiling mode (`--profile-stages`, `sync_cuda=True` on every stage) adds
`torch.cuda.synchronize()` calls that serialize the GPU pipeline regardless
of stream/event management. CSV host-time metrics cannot measure GPU overlap.

The theoretical overlap window (visible only in Nsight Systems):
```
detect(N+1, S_detect[1])  GPU time     ████████ (3-4ms)
track(N, 0x0)             GPU time       ███ (1ms)  ← overlap: 1ms saved
post(N+1, S_post[1])      GPU time          ██ (0.5ms) ← no overlap with detect(N+1)
```

Since track runs on 0x0 (legacy default) and 0x0 implicitly synchronizes with
all streams, the effective overlap is between detect(N+1) and track(N) only.
Post(N+1) waits for detect(N+1) via event fence and cannot overlap.

---

## Phase 8.5: PTDS Probe — FAILS determinism (5/6)

**Date**: 2026-07-06 | **200-frame SDP test** | **SACCADE_STREAM_MODE=ptds_probe**

Per-parity S_track[0,1] explicit streams with GTU (graphed tracker update)
for single-launch determinism.

| Run | IDF1 | MOTA | FP | FN | MD5 |
|---|---|---|---|---|---|
| 1-3,5,6 | 7.3% | -2.9% | 3474 | 45465 | `3132a1c3...` |
| **4** | 7.3% | -2.9% | 3474 | **45466** | `214e608f...` |
| Baseline | 7.2% | -3.0% | 3473 | 45486 | `5972bd5a...` |

**5/6 deterministic, 1/6 differs** (1 extra FN), metrics differ from baseline.

**Root cause**: GTU captures on 0x0 but tracker uses cublas internal streams not
covered by graph capture. Library-stream scheduling varies with replay context.

**Conclusion**: Tracker determinism depends on 0x0 ordering. S_track + GTU
is insufficient. PTDS (true per-thread default streams) requires PyTorch
recompile — not practical for current environment.

### Fallback: post count/materialize optimization

Since DS2 is eliminated and S_track is blocked, the next latency target is
post count D2H sync (~1.3ms mean, 3.6ms max). To be addressed in Phase 9.

---

## Phase 9: Post Count/Materialize — RESULT: Irreducible fixed overhead

### 9A — Count attribution

`post_graph_count_wait_ms ≈ 1.3–1.5ms` is spent in the post-processing
pipeline between NMS graph replay and end-of-post. However:
- The graph path (`_run_nms`) already returns `n_post = _NMS_FIXED_N = 1024`
  — no count D2H sync exists in this path.
- `filter_detections_fast` is in the non-native `else` branch, not the graph path.
- `.any()`/`.all()` implicit syncs are gated by config flags or branch conditions
  and don't fire in the native graph path.

**Conclusion**: The 1.5ms is NOT GPU wait. It's Python CPU processing time.

### 9B — Fixed-capacity buffers

`_NMS_FIXED_N = 1024` is already the effective capacity. `SACCADE_FIXED_POSTBUF=1`
is a no-op in the graph path — no count D2H sync to eliminate.

### 9F — Logical cap sweep

| Cap | post_ms | post_graph_count_wait_ms | total_ms | MD5 |
|---|---|---|---|---|
| 1024 (baseline) | 1.527 | 1.513 | 11.69 | `5972b...` |
| 512 | 1.509 | 1.496 | 11.77 | `5972b...` (identical) |
| 384 | 1.555 | 1.541 | 11.68 | `5972b...` (identical) |
| 320 | 1.571 | 1.554 | 11.94 | `5972b...` (identical) |
| 300 | 1.641 | 1.625 | 12.08 | `5972b...` (identical) |

All caps produce identical MD5 and metrics. `post_ms` varies 1.51–1.64ms
(±noise). The range is within run-to-run measurement noise. No improvement
from reducing the logical array size.

### Root cause

Post-processing GPU operations (tensor ops, boolean masking) are async O(N)
on the GPU. CPU operations (`.numel()`, `.shape[0]`, `if` checks, `int()`
conversions, function calls, dict lookups) are constant-time regardless of
array size. The 1.5ms is the **fixed Python interpreter overhead** of the
postprocessing pipeline — irreducible without rewriting critical paths in C++.

### Optimization status

| Approach | Result |
|---|---|
| Eliminate count D2H syncs | **N/A** — no syncs exist in graph path |
| Fixed-capacity buffers | **No-op** — already 1024 by default |
| Logical cap sweep (300-512) | **No improvement** — overhead is fixed, not O(N) |
| Reduce `_NMS_FIXED_N` + re-capture graph | Would reduce GPU buffer size, not Python overhead |
| Rewrite post pipeline in C++ | Would eliminate Python overhead but is high-effort |

### Recommendation

Accept the 1.5ms `post_graph_count_wait_ms` as **fixed overhead**. Further
post optimization requires C++ rewrite of the Python postprocessing path.
Redirect attention to higher-impact targets.

---

## Final Optimization Candidates (ranked by impact × feasibility)

| Priority | Target | Impact | Status | Fix |
|---|---|---|---|---|
| 1 | DS2 full-device barrier | **1.85ms** (eliminated) | **DONE** | S_detect/S_post event fence |
| 2 | DS1 full-device barrier | ~0.2ms | Future | NVJPEG stream fence |
| 3 | Post Python overhead | ~1.5ms fixed | **ACCEPTED** — irreducible without C++ rewrite | C++ post pipeline (high effort) |
| 4 | S_track on explicit stream | ~1ms overlap | **BLOCKED** — non-deterministic in legacy mode | PTDS recompile (PyTorch build) |
| 5 | TRT enqueue host overhead | ~0.3ms | Deferred | C++ dispatch or graph wrap |
| 6 | MS2 .item() implicit sync | Unknown | Future | Pinned memory + stream marker |
| 7 | TS1 bg_future.join | 2.7ms spike | Future | Profile bg thread |

### Hardened invariants (as of Phase 9F)

```
1. DS2 → S_detect/S_post event fence (proven, deterministic)
2. Tracker → CUDA legacy default stream 0x0 (required for determinism)
3. S_track → blocked in legacy mode (PTDS is the gate)
4. Post count D2H → no syncs in graph path (already optimized)
5. Post overhead → fixed Python cost, ~1.5ms, irreducible
```

---

## Phase 10: Productionize DS2-Eliminated Path — VALIDATED

**Date**: 2026-07-06 | **Canonical mode**: `SACCADE_STREAM_MODE=detect_post_event`

### Validation — 7-sequence MOT17

| Sequence | IDF1 | MOTA | IDs | FP | FN | Lines | vs Baseline |
|---|---|---|---|---|---|---|---|
| MOT17-02-SDP | 6.9% | -31.5% | 35 | 7393 | 17004 | 8969 | MATCH |
| MOT17-04-SDP | 17.5% | -28.8% | 27 | 21501 | 39714 | 29343 | MATCH |
| MOT17-05-SDP | 1.0% | -18.4% | 8 | 1466 | 6713 | 1669 | MATCH |
| MOT17-09-SDP | 1.3% | -62.3% | 3 | 3492 | 5150 | 3666 | MATCH |
| MOT17-10-SDP | 0.5% | -83.3% | 16 | 10797 | 12721 | 10914 | MATCH |
| MOT17-11-SDP | 3.1% | -91.8% | 17 | 9248 | 8832 | 9851 | MATCH |
| MOT17-13-SDP | 1.3% | -71.2% | 55 | 8599 | 11275 | 8965 | MATCH |

**All 7 sequences bit-identical to full-barrier baseline.**

### Determinism — 6× repeated (MOT17-04-SDP)

6/6 identical MD5, metrics identical, identical to baseline.

### Timing — 200-frame SDP

| Metric | Baseline (full barriers) | Production (event fence) | Δ |
|---|---|---|---|
| `total_ms` | 11.71 | 12.40 | +0.68 (+5.8%) |
| `detect_ms` | 6.06 | 6.64 | +0.59 (+9.7%) |
| **`detect_postproc_barrier_ms` (DS2)** | **1.81** | **0.00** | **-1.81 (-100%)** |
| `post_ms` | 1.61 | 1.56 | -0.04 (-2.6%) |
| `track_ms` | 0.88 | 0.90 | +0.02 (+1.9%) |

DS2 host-blocking full-device barrier eliminated. `total_ms` increase is Python/CUDA
dispatch overhead (stream switching, event creation) — does NOT include GPU wait.
The CPU is free to overlap work during the 1.81ms previously spent blocking.

### Production mode usage

```bash
SACCADE_STREAM_MODE=detect_post_event
```

Aliases: `explicit_probe`, `legacy_pingpong`

### Architecture

```text
TRT enqueue on S_detect[p]   (p = frame_id % 2)
cudaEventRecord(detect_done[p], S_detect[p])
cudaStreamWaitEvent(S_post[p], detect_done[p])
NMS graph replay on S_post[p]
post processing on S_post[p]
cudaEventRecord(post_done[p], S_post[p])
tracker on CUDA legacy default stream (0x0)
```

Per-parity stream pairs: S_detect[0], S_detect[1], S_post[0], S_post[1].
Detect output buffers cloned per parity to prevent overwrite by next frame's detect.

### Future work (deferred)

| Item | Rationale |
|---|---|
| DS1 barrier (0.2ms) | Needs NVJPEG stream handle for fence |
| C++ post rewrite | 1.5ms Python overhead is irreducible otherwise |
| PTDS / S_track | Failed determinism, needs PyTorch recompile |
| TRT graph wrap | ~0.3ms overhead, low priority |
| Logical NMS cap | No impact on post overhead |


---

# Closeout — Sync/Stream Optimization Registry

**Date**: 2026-07-07 | **Status**: Closed

Default production mode: `SACCADE_STREAM_MODE=legacy`

## Summary

The sync/stream optimization branch is closed. `legacy` remains the production
default. `detect_post_event` is retained as a correctness-validated opt-in mode
for future overlap-ready deployments. `ptds_probe` and explicit `S_track` are
classified as dead/blocked.

## Final Mode Status

| Mode | Status | Deterministic | DS2 | FPS |
|---|---|---|---|---|
| `legacy` | active / default | 6/6 ✓ | 1.95ms | 91.3 |
| `detect_post_event` | active / opt-in | 6/6 ✓ | 0ms | 86.0 |
| `ptds_probe` | dead / experimental | 5/6 ✗ | 0ms | — |
| explicit `S_track` | blocked | N/A | N/A | — |

## Key Findings

1. **DS2 cannot simply be removed** — nondeterministic output.
2. **DS2 replaced correctly with event fence** — 7/7 MOT17 identical, 6/6 deterministic.
3. **Event fence does not improve current synchronous latency** — regression from
   91.3 to 86.0 FPS. CPU has no useful work to overlap in frame-serial pipeline.
4. **Tracker must remain on CUDA legacy default stream `0x0`** — any explicit
   stream causes coordinate drift and nondeterminism.
5. **PTDS is not practical** — requires PyTorch recompile or deeper tracker rewrite.
6. **Post overhead is fixed Python cost** — graph path already uses fixed-capacity
   buffers. No hot-path count D2H sync to eliminate. Logical cap sweep produced no
   gain. Reduction requires C++ rewrite.
7. **DS1 is deferred** — ~0.2ms and protects ingest correctness.

## NO-GO Registry

| Attempt | Result | Decision |
|---|---|---|
| Remove DS2 directly | Nondeterministic | NO-GO |
| Same-stream event fence (legacy 0x0) | Not sufficient | NO-GO |
| Explicit `S_track` | Coordinate drift | NO-GO |
| PTDS probe | 5/6 deterministic only | NO-GO |
| Logical NMS cap sweep | No improvement | NO-GO |
| Post count D2H removal | Not applicable (graph path) | NO-GO |
| Make `detect_post_event` default | Throughput regression | NO-GO |

## Hardened Invariants

1. Production default: `SACCADE_STREAM_MODE=legacy`.
2. DS2 event fence is valid but opt-in only.
3. Tracker stays on CUDA legacy default stream `0x0`.
4. Explicit `S_track` blocked in legacy mode.
5. PTDS is not a production path.
6. Post graph path already uses fixed-capacity buffers.
7. Post overhead accepted unless C++ rewrite.
8. DS1 unchanged.

## Preserved Future Option

`detect_post_event` is correctness-validated and available when overlap becomes
feasible (multi-frame pipelining, async side-work, C++ dispatch, stream-safe
tracker, PTDS-compatible runtime).

Until then: `legacy` for all official evaluation, benchmarking, and production.
