# Production double-buffer critical-path measurement contract

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-17 -->
<!-- doc-module: pipeline -->

This contract is the owner of **how** unconstrained
`mamba_whole_graph_m + SDP + --double-buffer` production throughput is
measured and attributed. It does not own GPU partition / Green Context
routing (#419 / #431–#433). Those results may be cited as background
only; their harness FPS is not a production baseline.

The research TODO this contract executes is
[production_pipeline_profiling_todo.md](production_pipeline_profiling_todo.md).

## Question

Under the normal production scheduler

```text
detect(N+1) || tracker(N)
```

with no research routing-controller sync boundary, what work actually
limits throughput / frame period?

## Non-negotiable rules

1. **Production scheduling first.** Any instrumentation that inserts
   `cudaDeviceSynchronize`, extra stream waits, disables double-buffer,
   changes CUDA graph capture/replay, or changes detector/tracker stream
   dependencies is a *diagnostic trace*. Its FPS / frame period must not
   be reported as production.
2. **Production throughput is a separate clean run.** No nsys, no
   `--profile-stages`, no `SACCADE_ASSOC_STATS`, no NVTX.
3. **Do not re-sweep routing policy.** Fixed / shared / dynamic Green
   Context work is closed in #419/#431–#433.
4. **Exposed cost, not kernel duration.** A 0.5 ms kernel fully overlapped
   with detect is not a 0.5 ms throughput bottleneck.
5. **No guessing.** If a claim cannot be evidenced, write `UNRESOLVED`
   and name the missing measurement.

## Two-layer evidence

| Layer | What it may claim | What it must not claim |
|:--|:--|:--|
| **P — production** | FPS, frame period, per-seq latency, occupancy (`n_dets`), overlap *existence* via DB being eligible | Stage GPU durations, kernel names, memcpy bytes |
| **D — diagnostic** | Kernel/graph spans, overlap geometry, GMC/association breakdown, sync API placement, opportunity loss *structure* | Production FPS / frame period |

Calibrate D against P **without subtracting GPU-union busy from the
production period**. Those are different runs and different clocks;
their difference is not GPU idle and may be negative.

```text
outside_detect_remainder_ms
  = production_frame_period_ms − diagnostic_detect_span_ms
```

This remainder is a cross-run calibrated residual (P period minus D
detect span), not a production bubble. Diagnostic GPU-union busy is
recorded as a D-layer quantity only.

Never read production idle from nsys device-gap histograms. Node-mode
CUPTI inflates host wall time; kernel/graph hardware timestamps stay
usable. See [nsys_profiling.md](../../reference/runbooks/nsys_profiling.md).

Removal ceilings apply only to an **attackable slice** that is a proper
subset of the period (e.g. selective_scan). The detector whole-graph
span *is* the period; it is not removable and must not be converted into
an FPS ceiling.

## Allowed observers (default OFF)

| Observer | Production-safe? | Notes |
|:--|:--|:--|
| Clean `mot17.py` headline command | **P** | Only source of production FPS |
| `--profile-frame-csv` | **P-host** | Wall-clock ledger, no extra GPU sync, does not flip `_double_buffer_eligible` |
| nsys `--trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none` | **D** | Host-inflated wall; GPU spans OK |
| `SACCADE_ASSOC_STATS=1` | **D** | Extra diagnostic kernels inside tracker/NMS graphs; dump at sequence end |
| `--profile-stages` | **D-serial** | Disables double-buffer. Stage tables from this flag are not DB production |
| `SACCADE_ASSOC_DUMP` | **D-serial** | Host I/O + `cudaStreamSynchronize`; incompatible with tracker graph |
| GMC `set_profiling_enabled` | **D-serial** | Uses `cudaEventSynchronize` per sub-stage |
| NVTX / OSRT / CPU sampling | forbidden on this path | Deadlocks with whole-graph capture (runbook) |

`SACCADE_ASSOC_STATS` must be set **before** tracker / NMS graph capture
(environment at process start). Enabling it later cannot insert kernels
into an already-captured graph. Parsing is fail-closed and shared:
`1/true/yes/on` enable; unset, `0/false/no/off`, and unknown tokens stay
OFF in Python, tracker, and `PerceptionPipeline`.

## Headline command

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m \
  --detector SDP \
  --double-buffer
```

Resolved production identity (must hold for a P run):

```text
SACCADE_DOUBLE_BUFFER=1
SACCADE_DETECT_BARRIER=event
SACCADE_GPU_DECODE=1          # default
SACCADE_MAIN_NMS_GRAPHED=1    # preset
SACCADE_STREAM_MODE unset
profile_stages = false
workbench = false
cpp_threads = 0
```

## Procedure

Create a run directory, then execute layers in this order. Do not reuse
an nsys-traced process for P numbers.

### P0 — production 7-seq (throughput owner)

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer \
  --output "$RUN/p0_production"
```

Record overall FPS, per-seq FPS / mean / p95 / p99 from
`_fps_summary.txt` and `_latency_profile_*.json`.

### P1 — host ledger (no GPU sync)

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer \
  --profile-frame-csv \
  --output "$RUN/p1_frame_csv"
```

Optional occupancy slice (same flags, `--sequences` one of
`MOT17-05-SDP`, `MOT17-09-SDP`, `MOT17-04-SDP`) for low / medium / high
detection load. Measure P-layer FPS on each slice as well so occupancy
scaling of *period* is production-calibrated.

### D1 — nsys node-mode GPU timeline

One sequence at a time. MOT17-04-SDP is the high-occupancy 1080p
anchor; MOT17-05-SDP is the low-resolution / lower-count contrast;
MOT17-13-SDP is the heavy camera-motion contrast.

```bash
nsys profile --trace=cuda --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$RUN/d1_nsys_04" \
  uv run scripts/eval/mot17.py \
    --preset mamba_whole_graph_m --detector SDP --double-buffer \
    --sequences MOT17-04-SDP --latency-only \
    --output "$RUN/d1_eval_04"

uv run python scripts/benchmarks/nsys_frame_attribution.py \
  "$RUN/d1_nsys_04.nsys-rep" --json "$RUN/d1_nsys_04.json"
```

Repeat for `05` and `13` if the occupancy / GMC questions are still
open after 04.

### D2 — association / private-continuation counters

```bash
SACCADE_ASSOC_STATS=1 uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer \
  --output "$RUN/d2_assoc_stats"
```

Writes `_assoc_workload_{seq}.json` next to MOT output. Compare FPS
against P0 / P1 on the same sequences and record instrumentation
overhead. Do not substitute D2 FPS for P0.

### Derive

```bash
uv run python scripts/benchmarks/production_db_attribution.py \
  --production-dir "$RUN/p0_production" \
  --ledger-dir "$RUN/p1_frame_csv" \
  --nsys-json "$RUN/d1_nsys_04.json" \
  --assoc-dir "$RUN/d2_assoc_stats" \
  --out "$RUN/derived.json"
```

## Stage catalogue (must be identified)

```text
decode / ingest
backbone (TRT)
Mamba head (TRT + selective_scan)
postprocess / box decode / top-k
NMS (main graph)
private continuation (NMS 0.70 + serial append)
D2H metadata / count
GMC (frame staging copy, downscale, FFT, peak, prev-gray D2D)
tracker prediction (fused GMC+Kalman)
occlusion / cost / sinkhorn top-k
S0, S1, S1b, S1c, S2 auction+commit
bridge relink / lifecycle
materialize / output
```

Plus, on every timeline: CUDA streams, CUDA graph launches, CPU
blocking, event dependencies, explicit sync, detect/tracker overlap,
GPU idle bubbles.

## Exposed-cost identity

```text
production_frame_period
  ≈ exposed detector work
  + exposed tracker work
  + synchronization loss (opportunity, not just API duration)
  + unavoidable serial overhead
```

Per stage classification:

| Label | Rule |
|:--|:--|
| fully hidden | exposed / duration < 5% |
| partially exposed | 5–95% |
| critical-path | ≥ 95% of its duration sits in the detect-to-detect tail or on the period |
| serialization point | a wait that delays `detect(N+1)` admission |
| fixed scheduling overhead | launch / graph / host quiet time that does not scale with occupancy |

Distinguish:

* **A.** sync self-cost (e.g. D2H 30 µs)
* **B.** opportunity loss (e.g. that 30 µs delays the next detect by 250 µs)

B is what can dominate throughput.

## Bottleneck ranking rule

Rank by **exposed cost to production frame period**, not by kernel
duration. Each ranked item must carry: observed cost, exposed cost,
trigger frequency, occupancy scaling, overlap relationship, evidence
pointer, uncertainty, and a removal upper bound.

A **container** that *is* the period (the detector whole-graph span)
gets `removal_applicable=false` and null FPS ceiling. Attach an
`attackable_slice` (selective_scan) for the only in-container lever
that may carry a ceiling. Association passes are classified by
**activity frequency** (`frames_with_assignment` /
`frames_with_valid_topk` over frames), not by whether the assignment
count is exactly zero.

Class labels for the next optimization PR:

```text
A. fixed-capacity computation
B. synchronization / scheduling
C. memory staging / traffic
D. detector compute
E. tracker compute
F. launch / fixed scheduling overhead
G. no single bottleneck — overlap frontier already saturated
```

## What this contract does not authorize

GPU partition/routing sweeps, Green Context redesign, module-quality
ablation, tracker threshold tuning, detector retraining, algorithm
removal, large optimization implementation, or a new headline FPS
claim.
