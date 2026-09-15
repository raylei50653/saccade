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
| `time_budget_audit.py` | diagnostic | cli | Independent vectorized ledger/timing audit; does not import producer code. |
| `time_budget_plot.py` | diagnostic | cli | Plot descriptive response/drain/throughput frontiers from a replayed record. |
| `time_budget_probe.py` | diagnostic | cli | Measure calibrated time-budget admission under repeated stable arrivals. |
| `time_budget_report.py` | diagnostic | cli | Replay time-budget ledgers, calibration, repeated arrivals and measured frontiers. |

<!-- END generated script index -->

## Time-budget admission with repeated arrivals

`time_budget_probe.py` augments the fixed-window study with a conservative
remaining-work ledger. Run the GPU experiment sequentially, then replay:

```bash
.venv/bin/python scripts/benchmarks/resource_elastic/time_budget_probe.py \
  --output /absolute/new/time-budget --samples 20 --repeats 3 --warmups 3 \
  --calibration-samples 30 --period-us 12000 --budgets-ms 3 6 12
.venv/bin/python scripts/benchmarks/resource_elastic/time_budget_report.py \
  /absolute/new/time-budget --json-output /absolute/new/time-budget.json \
  --markdown-output /absolute/new/time-budget.md
.venv/bin/python scripts/benchmarks/resource_elastic/time_budget_audit.py \
  /absolute/new/time-budget /absolute/new/time-budget-audit.json
.venv/bin/python scripts/benchmarks/resource_elastic/time_budget_plot.py \
  /absolute/new/time-budget.json /absolute/new/time-budget.svg
```

### Frozen controller and workload

- The same three routes and physical pool-role swaps as `admission_probe.py`;
  only the disjoint 16+16 route supplies a spatial reservation. The shared
  32-SM route is the matched-total-SM priority control. Full-device priority
  uses all device SMs. Pool ownership is static; no SM borrowing occurs.
- 256 independent elastic kernels per trial, each with 1024 blocks × 256
  threads. The ordered workload is a seeded shuffle of 64 repetitions of
  `[512, 2048, 4096, 2048]` FMA iterations. This is 272 equivalent 2048-iteration
  units, identical across policy, route and role for each repetition/sample.
  The mix includes an eightfold iteration-count range; shorter work is not selected
  ahead of a longer FIFO head to make a budget fit.
- Before evaluation, measure 30 isolated samples of each size on each
  route/role, with three warmups. Freeze each estimate at **1.2 × training
  p95 GPU envelope duration**, rounded up to integer ns. Retain calibration
  arrays and exclude them from evaluation. Shared-route role labels use separate
  calibration batches on the same physical pool; their variation includes
  training variation as well as execution drift. Only disjoint roles physically
  swap the pools. This is a calibrated empirical
  distribution, with no online learning or deterministic coverage claim.
- Window baselines admit at most 1/4/16 submitted-but-not-retired kernels.
  Budget policies admit only when the sum of frozen estimates of all
  submitted-but-not-retired work, including the candidate, is at most
  **B = 3/6/12 ms**. A completion event releases its charge. The running unit
  retains its full estimate until retirement; queued time is never subtracted
  as execution progress. This conservatively approximates remaining service
  demand but does not upper-bound actual duration when estimates are wrong.
- A unit larger than B fails before evaluation; there is no singleton bypass,
  splitting or dropping. The initial pilot rejected a 2-ms budget for this
  reason. The separate qualifying pilot used the final 3/6/12-ms matrix.
- Submit one elastic unit, observe its actual block-zero start publication,
  then define four targets at +12/+24/+36/+48 ms in the host clock. Dispatch at
  most one unit per polling iteration. At each due target, freeze admission,
  submit stable work, independently observe stable completion and admitted
  prefix drain, and resume only after both. Targets do not shift when an
  earlier pause runs late: overdue stable requests are serviced serially and
  their overshoot is retained. This is a finite four-arrival schedule, not an
  independent arrival thread or a general open-loop load generator.
- Three shuffled repetitions × 36 conditions × 20 retained trials = 2,160
  trials and 8,640 stable arrivals, plus 324 warmup trials. No tail filtering,
  retry, or removal of idle/exhausted arrivals. Fixed seed 421.

### Measurements and replay

Stable response is submission-to-completion observation. Scheduled response
adds target-to-submission overshoot. Drain is freeze/submission-to-observed
empty elastic lane, including polling delay and stable dispatch cost; this is
an **empty-lane proxy**, not ownership transfer or actual resource reclamation.
No host timestamp is subtracted from a GPU timestamp.

Throughput is 272 equivalent elastic units divided by whole-trial host seconds,
including all stable pauses, dispatch and polling. This is synthetic compute
throughput, not application throughput. Dispatch ms sums elastic launch/event
host calls for the entire trial; pause ms is per arrival. Trial-level metrics
are repeated across that trial's four rows, so percentiles must not be treated
as four independent throughput samples. Arrival p99 uses 80 correlated
observations per repetition/role; it is descriptive, not a rare-tail guarantee.

Raw arrays retain every block's start/end/SM ID, block-leader outputs, every
elastic dispatch and retirement, the estimate/ledger at dispatch and freeze,
all four arrival targets and pause boundaries, and trial/start-observation
times. Replay recomputes training estimates, matched workload sequences,
ledger and count bounds, freeze ordering, output identity, physical membership,
all metrics and quantiles. `rolling_dispatches` counts launches following an
observed retirement in that same active interval, excluding the initial refill
following a pause. Later-arrival counts demonstrate actual pre-arrival rolling
replenishment. Miss counts and per-unit estimate exceedance remain evidence;
a valid ledger is not sufficient to validate an observed latency bound.

The supplementary audit uses a separate vectorized implementation for ledger
and timing calculations, maximum drain, budget misses and later replenishment.
Run it after the full reporter: it does not replace checksum, calibration,
coverage or routing validation. The SVG shows medians and min–max ranges of
repetition/role quantiles, not confidence intervals.

Source snapshots, toolchain/device identity, 100-ms device telemetry and complete
SHA-256 manifests follow the previous harness. Failed runs are retained and
rejected by replay. The interactive WSL host has no locked clocks or isolated
host core. Memory-sensitive stable work, ownership/routing changes, production
scheduling and full-pipeline guarantees remain outside this experiment.

See [the time-budget results](../../../docs/reference/benchmarks/resource_time_budget_20260915.md).
