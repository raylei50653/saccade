# Full-pipeline mixed-workload frontier (#419)

Diagnostic only. The stable workload is the real serial MOT17 evaluator with
`mamba_whole_graph_m`, GPU decode, TensorRT detection, PyTorch features, native
tracking and CUDA Graphs. Its state and captured graphs stay in one verified
Green Context. Dynamic routing moves **future elastic kernels** onto that
context between frames; it freezes admission and drains them before admitting
the next frame. This tests temporal borrowing of stable capacity, not migration
of the stateful pipeline, live graph rebinding, or a permanently disjoint floor
while the pipeline is idle. Double-buffer execution is outside this contract.

## Predeclared contract (before pilot)

- One sequence: MOT17-04-SDP, first 350 frames, 50 warmup / 300 retained.
- Absolute stable arrivals at 60 FPS; FIFO, no frame dropping. Stable target:
  scheduled-arrival-to-output p99 <= 16.667 ms and observed misses <= 1% per
  repetition. Host scheduling, admission drain and queueing count toward it.
- Fixed: stable 16 SM / elastic 16 SM. Static headroom: stable 24 SM / elastic
  8 SM. Matched sharing: one 32-SM context with stable high-priority streams
  and elastic normal-priority streams. Dynamic: same split as fixed, plus a
  normal-priority elastic stream on stable's 16 SM, available only between
  frames. All use a 32-SM envelope; all counts and disjointness are verified.
- Identical burst arrivals: 256 independent elastic units every 100 ms,
  starting at the measurement origin; last burst strictly before the final
  stable arrival. Each unit: 256 blocks x 256 threads, 2048 or 8192 FP32 FMA
  iterations. Queue windows 1 or 4 per lane; shared has two elastic streams,
  matching dynamic's two submission lanes. Fixed/headroom use two streams
  within their single elastic pool. Compare unit sizes separately.
- Three shuffled repetitions, seed 431. Pilot is excluded. No operating point
  is removed or target adjusted after observing results. An unpressured
  16-SM stable control checks target feasibility.
- Elastic completed-before-stable-cutoff throughput and all-bursts completion
  time are separate. Final elastic drain is measured. Total throughput is
  reported as the pair (stable frames/s, elastic units/s), not a sum of
  unlike jobs. Every elastic output must equal its CPU FP32 reference.
- Request-to-drain and request-to-stable-admission are host timestamps.
  They do not claim exact GPU execution start. CUPTI twins validate all real
  pipeline kernels/graphs stay in the stable context and elastic kernels use
  only declared contexts. Unaudited runs supply frontier timing.
- Every run retains frame timestamps, elastic enqueue/completion observations,
  stream/pool identity, output hashes, source snapshots, commands and failures.
  Latency tails are descriptive on one sequence/host; no hardware utilization,
  deterministic deadline or production policy claim.

## Execution and replay

```bash
uv run python scripts/benchmarks/resource_mixed/sweep.py \
  --output /absolute/new/archive
uv run python scripts/benchmarks/resource_mixed/summarize.py /absolute/new/archive
uv run python scripts/benchmarks/resource_mixed/plot.py \
  /absolute/new/archive/summary.json --output /absolute/new/archive/frontier.svg
```

Each of the 17 conditions (16 mixed plus control) has one 350-frame audited
condition twin in repetition zero and three unaudited measurements. Later
repetitions reuse that condition twin only if output hashes, stream priorities,
context SM counts, direct/graph SM probes, thread-switch counts and set-device
reassertions agree. This is audited-twin evidence, not a CUPTI trace of every
timing run. `report.py` rejects observed borrowed/stable GPU-kernel overlap,
escaped kernels/memsets, graph absence, queue overflow, incorrect output and
lost work; `summarize.py` also checks sealed raw hashes and complete coverage.

The wrapper waits for every registered pipeline stream before lending its
capacity. The target uses this conservative completion (which includes output
completion); ordinary scheduled-arrival-to-output latency is also retained.
It never adds an all-device barrier to shared service. Native steady-clock and
Python perf-counter are paired in a measured host bracket (maximum 1 ms);
elastic cutoff counting uses the conservative end of that bracket. No GPU
clock is subtracted from a host clock. CUPTI overlap checks stay within CUPTI's
clock domain. `transition_ms` is request-to-admission, not GPU kernel start.

The elastic kernel performs FP32 FMA recurrence on all threads and a shared
memory reduction; every block result is checked both against native CPU FMA
and an independent NumPy replay. This workload provides known completed work,
not an application utility score or a memory-bandwidth interference study.
The final throughput pair must not be added into a single jobs/s number.

GPU stamp buffers are copied to the host only after the evaluator has completed
all stable frames. An incomplete initial main attempt was excluded because it
performed that evidence copy at elastic completion, potentially within stable
service. The replacement preserves the frozen target and workload.

Independent checks after the main sweep:

```bash
uv run python scripts/benchmarks/resource_mixed/audit.py /absolute/new/archive
uv run python scripts/benchmarks/resource_mixed/audit_copies.py /absolute/new/archive
```

Hardware-side replay of the sealed audited twins:

```bash
uv run python scripts/benchmarks/resource_mixed/hardware_report.py \
  /absolute/existing/archive \
  --json-output /absolute/new/resource_hardware.json
```

This derives stable/elastic kernel-presence intervals, execution overlap,
borrowed-lane use, exact SM-set coverage and per-SM elastic block residency.
It uses only clocks within their own domains and does not promote kernel
presence into a chip-wide SM-active counter. The measured
[hardware report](../../../docs/reference/benchmarks/resource_hardware_20260916.md)
records why direct GPU-counter profiling was excluded after it perturbed
multi-context routes asymmetrically.

The [measured report](../../../docs/reference/benchmarks/resource_mixed_20260916.md)
retains all repetitions, target passes, misses, paired comparisons and the
frontier figure. Execution snapshots in the archive remain authoritative when
this README later gains generated discovery indexes.

Priority applies to owner-created stable streams; runtime-internal auxiliary
streams retain their library behavior. CUPTI verifies their execution context.

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `audit.py` | diagnostic | cli | Independently audit the frozen service target, arrivals and borrowing intervals. |
| `audit_copies.py` | diagnostic | cli | Reject diagnostic GPU stamp transfers inside the stable-service horizon. |
| `hardware_report.py` | diagnostic | cli | Relate the sealed mixed frontier to CUPTI timelines and block residency. |
| `plot.py` | diagnostic | cli | Plot the measured full-pipeline latency versus elastic burst frontier. |
| `report.py` | diagnostic | cli | Replay mixed-workload timestamps, conservation, placement and CUPTI evidence. |
| `run_point.py` | diagnostic | cli | Run scheduled real pipeline frames against bounded native elastic bursts. |
| `summarize.py` | diagnostic | cli | Verify sealed runs and compare paired mixed-workload service frontiers. |
| `sweep.py` | diagnostic | cli | Freeze, execute and seal repeated full-pipeline mixed-workload comparisons. |

<!-- END generated script index -->
