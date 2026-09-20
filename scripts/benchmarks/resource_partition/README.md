# GPU true execution-resource partition (#419 phase B)

Standalone diagnostic entry points that run the full MOT17 serial/double-buffer
pipeline **inside a verified CUDA Green Context** and measure how throughput,
latency, period and jitter scale with the actual SM budget. Production
defaults and production sources are unchanged; every routing change lives in
the diagnostic execution owner below. The residency-pressure proxy in
[`../resource_sensitivity/`](../resource_sensitivity/README.md) is a separate
surface and is never mixed with these results.

## Run

From the repository root, with the project's CUDA/TensorRT environment loaded:

```bash
.venv/bin/python scripts/benchmarks/resource_partition/sweep.py \
  --output /absolute/new/output-directory \
  --preset mamba_whole_graph_m --sequences MOT17-04-SDP \
  --levels full 46 40 32 24 16 8 --repeats 2 --max-frames 300

.venv/bin/python scripts/benchmarks/resource_partition/summarize.py \
  /absolute/new/output-directory --deadline-ms 10
.venv/bin/python scripts/benchmarks/resource_partition/plot.py \
  /absolute/new/output-directory
```

Levels are **requested** SM counts; `full` is the primary full-device baseline
routed through the same owner. Requests are rounded to the device's partition
alignment (8 on this host) and the context-reported actual count is what the
report uses; a request whose actual count differs is not validated. Requires an
NVIDIA GPU with Green Context support, CUPTI headers/libraries at
`/opt/cuda/targets/x86_64-linux`, `nvcc`, `g++`, `nvidia-smi`, local MOT17 data
and the preset's model assets. The probe build defaults to `--arch sm_120`.
The owner is qualified on a single visible GPU (logical device 0). Every
output directory must be new; failed results are retained, never reused.

## Green Context execution owner (`green_owner.py`)

`GreenExecutionOwner.install()` runs before any Saccade import and owns:

| Resource | Mechanism | Evidence retained |
|---|---|---|
| Green Context | `cuDevSmResourceSplitByCount` → `cuGreenCtxCreate(CU_GREEN_CTX_DEFAULT_STREAM)` | requested vs `cuGreenCtxGetDevResource` actual SM count |
| Every `torch.cuda.Stream()` | replaced by an `ExternalStream` subclass whose handle comes from `cuGreenCtxStreamCreate` (or `cuStreamCreateWithPriority` for `full`) | registry of handles; `cuStreamGetCtx`/`cuStreamGetGreenCtx` per handle at finalize |
| Main/default stream | owned stream made current; `torch.cuda.default_stream` returns it | handle; per-frame check that the main thread's current context is still owned |
| Double-buffer detector lane | `pipeline.py`'s `torch.cuda.Stream()` therefore yields an owned stream | handle and class check per sequence |
| TensorRT | `execute_async_v3(torch.cuda.current_stream().cuda_stream)` receives the owned stream; runtime-created internal streams inherit the current Green Context | CUPTI kernel records |
| Native tracker kernels | receive `torch.cuda.current_stream().cuda_stream`; NULL-stream memsets/memcpys inherit the current Green Context | CUPTI kernel/memset/memcpy records |
| CUDA graphs | a graph executes in the context of its **capture** stream (measured); all capture streams are owned streams | SM-id probe through a captured graph; graph-launched kernel records |
| Background threads | `threading.Thread.run` makes the owned context current before the target runs | switch count and errors |
| `torch.cuda.set_device` | TorchInductor-generated code calls it on every compiled call and it reverts the current context; the owner re-asserts its context after each call | re-assertion count |

`cuCtxSetCurrent` alone is explicitly **not** trusted. Two independent
verifications are recorded for every audited run and both must pass:

1. **SM-id probe**: a spin kernel with 4096 resident blocks launched on the
   owned main stream directly and through a captured CUDA graph, before and
   after the evaluation, must reach exactly the context-reported SM count.
2. **CUPTI activity audit** (`partition_audit.cpp`): every kernel, memset and
   memcpy record carries the CUPTI context ID it executed in. From the first
   frame's latency start to the last completion, every kernel and memset must
   be in the owned context; graph-launched kernels must be present. Escaped
   kernels are listed by context, stream and name so an escape is localized,
   not just counted. Records before the first frame (model loading) are
   reported separately and are not fatal. Dropped CUPTI records fail the run.
   A kernel-name heuristic reports coverage by source (TensorRT, native,
   PyTorch, nvJPEG) for readability only; the verdict never uses it.

## Audited twin and measurement

CUPTI per-kernel tracing costs roughly 5–15% of throughput on this workload and
costs serial more than double-buffer, which would distort DB gain. The sweep
therefore runs every point twice with an identical command: `<point>_audit`
(CUPTI on, produces the verdict) and `<point>` (CUPTI off, produces the
timing). A measurement row is `true_partition_validated` only when:

- its own owner checks pass (stream registry ownership, main-thread context
  owned at every completed frame and at finalize, SM-id probes before/after,
  double-buffer stream owned, actual SM count equals the request);
- its audited twin passes every owner and audit check; and
- the twin is consistent with it: byte-identical MOT output, equal owned-stream
  count, equal thread switches and `set_device` re-assertions, equal actual SM
  count and the same double-buffer route.

The `full` baseline row can be `routing_validated` but is never
`true_partition_validated`. `summary.json` records `true_partition_validated`,
`actual_sm_count` and `routing_verification` per row and exits nonzero if any
row fails, while retaining every row's evidence.

## Timing and comparison contract

Identical to the proxy surface: the point wrapper delegates to the production
`_record_frame_timing` and reads its completion timestamp, adding no per-frame
GPU synchronization. Latency starts before decode and ends after completed
output (#35/#376). Period is the difference between consecutive completion
timestamps within a sequence. FPS is completed frames over the production
throughput interval. Both modes use `--detect-barrier event` and default GPU
decode; only double mode adds `--double-buffer`. Tracker-stage latency is
unobserved and stays `null`. Levels are shuffled per repetition (`--seed 419`),
mode order alternates, and each DB gain pairs the same-repetition serial run.

## Frozen benchmark and before/after comparison (`frozen_benchmark.py`)

`frozen_benchmark.py build <sweep-dir>` turns a completed sweep whose every
point validated into a compact record (`saccade-partition-frozen-benchmark-v1`):
per-budget aggregates over repetitions (FPS, DB gain, frame p50/p95/p99,
period p50/p95/p99 and σ, telemetry means) for serial and double, the scope
that a later run must reproduce (preset, sequences, frame bounds, warmup,
barrier, requested levels, device name/SM count/alignment) with the recorded
`HEAD`, harness/library hashes, driver and toolchain versions, the
**pre-declared knee verdict** and the **service-level frontier** (smallest
verified budget meeting each declared FPS / frame-p99 target in every
repetition). The rule constants at the top of the script are part of the
schema; changing them is a new schema version.

`frozen_benchmark.py compare --reference <record.json> --candidate <sweep-dir
or record> [--control <sweep-dir or record>]` exits 2 unless the scope and
criteria match exactly; otherwise it reports candidate/reference ratios of
means per budget and mode, DB-gain deltas, whether repetition ranges overlap,
both knee verdicts and frontier movement in SMs. It records (never rejects)
`HEAD`, source, driver and toolchain differences. Absolutes on one host have
drifted 7–10% between sessions at identical clocks, so a before/after claim
should pass the reference source re-run in the **same session** as
`--control`; without it the comparison says `host_drift: null`. The committed
reference is
[`resource_scaling_closure_20260920.json`](../../../docs/reference/benchmarks/resource_scaling_closure_20260920.json).

## Artifacts

- `input_identity_before.json` / `input_identity_after.json`, `green_probe.json`
  (capability probe), `execution_sources/`, `sweep.json` (commands, schedule,
  source/library hashes, HEAD, compiler, device, environment).
- `<point>/partition_point.json`: owner evidence, routes, frame starts,
  completions, routing verification, verdict; `<point>/cupti_activity.trace`
  for audited twins; `<point>/eval/` holds the ordinary evaluator outputs.
- `<point>.telemetry.csv`, `<point>.log`, `summary.json`, `resource_partition.png`.
- `sweep.json` also records `toolchain` (python/torch/TensorRT/cuda-bindings
  versions of the point interpreter) for the frozen record's scope.

See [the phase-B report](../../../docs/reference/benchmarks/resource_partition_20260915.md)
for measured results.

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `frozen_benchmark.py` | diagnostic | cli | Freeze the serial/double-buffer SM-scaling benchmark and compare a later sweep against it. |
| `green_owner.py` | diagnostic | - | Own Green Context routing for PyTorch, TensorRT, native CUDA and graph launches. |
| `plot.py` | diagnostic | cli | Plot verified-partition serial/double FPS, DB gain and frame p99 versus actual SM count. |
| `report.py` | diagnostic | cli | Build the machine-readable phase-B study record and its markdown tables. |
| `routing.py` | diagnostic | - | Fail-closed Green Context routing verdicts from owner evidence and CUPTI traces. |
| `run_point.py` | diagnostic | cli | Run one full-pipeline MOT17 point inside a verified Green Context partition. |
| `summarize.py` | diagnostic | cli | Validate routing evidence and summarize true-partition throughput, latency and jitter. |
| `sweep.py` | diagnostic | cli | Reproduce paired serial/double-buffer sweeps over verified Green Context SM budgets. |

<!-- END generated script index -->
