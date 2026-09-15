# Elastic SM scheduling exploration (#419)

Bounded native CUDA experiments to distinguish spatial reservation from stream
priority and burst work granularity. This is a synthetic diagnostic, not a
production scheduler or a full-pipeline Green Context audit.

## Reproduce

Run on an otherwise idle CUDA host, sequentially (never concurrently):

```bash
.venv/bin/python scripts/benchmarks/resource_elastic/probe.py \
  --output /absolute/new/elastic-one --compute-chunks 1 --samples 100 --repeats 3
.venv/bin/python scripts/benchmarks/resource_elastic/probe.py \
  --output /absolute/new/elastic-sixteen --compute-chunks 16 --samples 100 --repeats 3
.venv/bin/python scripts/benchmarks/resource_elastic/probe.py \
  --output /absolute/new/elastic-return-one --compute-chunks 1 --samples 100 --repeats 3
.venv/bin/python scripts/benchmarks/resource_elastic/report.py \
  /absolute/new/elastic-one /absolute/new/elastic-sixteen \
  /absolute/new/elastic-return-one \
  --json-output /absolute/new/record.json --markdown-output /absolute/new/tables.md
```

Requires cuda-bindings, NumPy, nvcc, nvidia-smi and a device supporting two
16-SM Green Context groups. `--sms` changes the requested equal group size;
`--arch` defaults to this host's `sm_120`. Unused remainder SMs are reported.
Output directories must be new. A failed run retains `result.json` with failed
status; the reporter refuses it. The CLI exits nonzero on failures.

## Work and routing

- Stable work: one 128-block, 256-thread kernel, each thread running 4096
  dependent FP32 FMAs. Block-leader outputs must remain exactly equal across
  every condition. This small compute workload is not a memory-sensitive or
  TensorRT workload.
- Compute burst: 1024 blocks, 256 threads, 32768 dependent FMAs per thread.
  With `--compute-chunks 16`, sixteen 1024-block kernels each do 2048 FMAs.
  Total FMA count stays fixed; block setup/output and launch overhead change.
  Each chunk restarts its local accumulator: this models independent work
  units, not a semantics-preserving transformation of an arbitrary kernel.
- Memory burst: four read/modify/write passes over a separate 128 MiB array,
  1024 blocks, 256 threads, volatile accesses. It is unchanged by chunk count;
  its readback is not checked and this is not a correctness benchmark for it.
- Two disjoint groups come from **one** `cuDevSmResourceSplitByCount` call.
  Their stream context ownership, actual SM counts, and SM IDs are checked.
  SM-ID saturation probes run before/after; every measured block also records
  its SM ID and must stay inside the matching pool. This directly audits the
  native probe kernels; it does not audit library internals or CUDA graphs.
- Routes: 16-SM solo, same 16-SM pool, disjoint 16+16 pools, primary full-device
  solo/shared, and highest-supported-priority stable streams on the shared
  16-SM/full-device pools. Burst stream priority is zero throughout. Fixed pool
  roles are retained; the pools are not swapped between repetitions.

## Timing contract

Each trial submits the whole burst first, then submits stable work and waits
only for the stable stream. **Host response** starts immediately before stable
kernel submission and ends when its stream wait returns. It includes launch,
queueing and host wakeup; it excludes the host cost of submitting the preceding
burst. Afterwards the burst stream is drained before readback and the next
trial. There is no device-wide synchronization in the measured interval.

Each block records `%globaltimer` timestamps. Stable/burst GPU time is the
envelope from earliest block start to latest block end, including inter-kernel
gaps for a chunked burst. Stable start/finish offsets use the burst's earliest
GPU block start as origin; a negative start is possible if priority lets stable
work start first. These offsets are **not** enqueue-to-start latency. Per-block
interval intersections separately show whether stable blocks overlap any burst
block in time; envelope overlap alone is weaker evidence.

All burst chunks are enqueued before stable work; the experiment does not inject
a request at a controlled offset into an already resident burst. It measures
neither an open-loop arrival process nor actual resource reclamation. WSL host
scheduling and uncontrolled clocks/power can affect samples. Conditions are
shuffled per repetition with seed 419, ten warmups precede each condition,
and each condition retains every measured sample. Separate chunk sweeps are
sequential rather than a randomized paired chunk experiment.

## Evidence

Each archive retains source copies, source and artifact SHA-256, build/runtime
identity, all sample metrics and SM ID sets, per-repetition quantiles and pool
probes. The reporter verifies listed checksums, complete condition/sample
coverage and re-derives every saved quantile from samples. Raw per-block timer
arrays are reduced inside native code and are not retained. No CUPTI twin,
pipeline output, deadline, energy or service guarantee is claimed.

See [the exploration record](../../../docs/reference/benchmarks/resource_elastic_20260915.md)
for the measured decision and continuation.

## Confirmed-start arrival and bounded admission follow-up

`admission_probe.py` is an independent harness; the earlier three archives and
`probe.py` timing contract remain their own experiment. Run sequentially:

```bash
.venv/bin/python scripts/benchmarks/resource_elastic/admission_probe.py \
  --output /absolute/new/admission --samples 50 --repeats 3 --warmups 5
.venv/bin/python scripts/benchmarks/resource_elastic/admission_report.py \
  /absolute/new/admission --json-output /absolute/new/admission-record.json \
  --markdown-output /absolute/new/admission-tables.md
```

The frozen default matrix has 48 conditions per repetition: three routes,
`(chunks, window) = (1,1), (16,1), (16,4), (16,16)`, two arrival offsets
(0 and 250 µs), and two pool-role assignments. Three repetitions retain 7,200
trials plus 720 warmups. Conditions, including chunk size, are shuffled together
in each repetition with seed 420. Role labels on shared routes are repeated
controls; only the disjoint route physically swaps pools.

### Work, start confirmation and admission contract

- Compute work retains the original 128-block stable task and 1024-block burst
  with equal nominal 32768 iterations per burst thread across chunks. Each
  chunk resets its accumulator. All block-leader outputs and all block stamps
  are retained. There is no memory-burst condition in this follow-up.
- Routes are disjoint 16+16 SM pools, shared high-priority stable work on one
  32-SM Green Context, and shared high-priority stable work on the full device.
  The first two match total SM count. The 32-SM group comes from a separate
  split; its physical IDs need not equal the union of the 16-SM groups.
  Only one route runs at a time. Before/after SM probes and every measured
  block validate membership; reservation is static throughout.
- Block zero of the **actual first burst kernel** publishes its GPU start
  timestamp through an aligned mapped-host system-scope atomic **store**.
  The host polls with acquire loads, then starts the requested offset timer.
  No mapped-memory read/modify/write atomic is used. CUDA documents system-scope
  atomicity for naturally aligned mapped loads/stores in its
  [memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html).
  The signal itself adds overhead and confirms kernel entry, not full occupancy.
- The initial window is filled before observing the start signal. Before the
  arrival target, the host replenishes at most one chunk per polling iteration.
  A slot is released only after a per-chunk CUDA completion event succeeds.
  Thus the window bounds **submitted but not host-retired chunks**, including
  running work. It is a conservative count, not hardware queue occupancy.
- At the target, stop admission and submit stable work. Observe stable
  completion and the already-admitted prefix drain independently. Resume the
  remaining burst only after **both** complete, using the same window. This
  pause/resume policy keeps total burst work equal; it does not transfer pool
  ownership or implement borrowing/routing.
- No retry/filter removes late arrivals or tails. A 5-second host polling timeout
  fails the run. Burst kernels have finite work and do not wait on a host gate.

### Timing and evidence limits

`stable_response_ms` measures host launch begin to successful stable completion
**event polling**, unlike the earlier stream-synchronize response. Dispatch,
CUDA event recording and host polling are included. `stable_signal_observed_ms`
is an upper bound on host-enqueue-to-first-GPU-start delay, since block zero's
publication can be observed later than first execution. Its last-zero/first-value
poll interval brackets publication visibility; it does not bracket the first
block's execution. Host and GPU clock values are never subtracted.

`drain_observed_ms` is arrival submission to host observation that all admitted
chunks have completed; it includes host detection delay and has a floor from
stable launch/event overhead. It measures when this particular lane is observed
empty, not SM reclamation. `burst_host_ms` includes initial submission, arrival
control, stable pause, dispatch gaps and all resumed work. Per-block stamps give
GPU envelopes and the admitted-prefix finish relative to stable GPU start.

The requested offsets begin at **host observation**, not the exact GPU start.
The start signal may be observed late, including after some work has finished.
The archive reports arrival overshoot, retired work at arrival and whether the
admitted prefix finished before stable execution. Confirmed prior start does not
prove continuous residency at submission. Exact enqueue-to-GPU-start timing
would need host/device clock correlation with quantified error.

Archives retain raw arrays per condition (`host`, `chunk_host`, `stamps`,
`values`), every measured sample, source snapshots/hashes, build identity,
100-ms `nvidia-smi` clocks/power/temperature/utilization telemetry and checksums.
Telemetry is descriptive; clocks/power are not controlled. The reporter checks
complete conditions, sample counts, source identity, queue/freeze ordering,
start signals, outputs, SM membership and recomputes every saved quantile from
raw arrays. The pilot is separate and excluded from the main result.

See [the follow-up record](../../../docs/reference/benchmarks/resource_admission_20260915.md).

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `admission_probe.py` | diagnostic | cli | Measure confirmed-start arrivals and bounded rolling CUDA admission. |
| `admission_report.py` | diagnostic | cli | Replay raw CUDA admission evidence and summarize timing observations. |
| `probe.py` | diagnostic | cli | Measure synthetic co-load on shared versus disjoint Green Context SM pools. |
| `report.py` | diagnostic | cli | Replay synthetic probe summaries and render per-repetition ranges. |

<!-- END generated script index -->
