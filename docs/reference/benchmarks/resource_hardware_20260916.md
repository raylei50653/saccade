# #419 hardware behavior behind the mixed-workload frontier

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-16 -->
<!-- doc-module: cross -->

## Decision

**Sharing and borrowing win by using different kinds of otherwise unavailable
capacity.** Sharing gives every admitted elastic kernel the full 32-SM pool and
lets stable and elastic work coexist under stream priority. Dynamic routing
keeps two 16-SM pools, but sends future elastic kernels onto the stable pool
between frames and drains that lane before stable admission. Fixed reservation
does neither: when no stable kernel is present, elastic work still cannot enter
the reserved stable pool.

This explains the measured regions without requiring one universal bottleneck:

- At 2048 iterations / W4, sharing wins both axes. Its elastic-kernel p50 is
  **19.7 us**, versus **35.0 us** fixed and **31.7 us** dynamic, and it reaches
  all 32 pool SMs. Borrowing adds only **158.7 ms** of kernel presence to the
  stable pool over the five-second service horizon, raising its observed
  kernel-present fraction from **32.2% to 35.4%**. That recovered opportunity
  is too small to overcome 16-SM execution and reclamation cost at this point.
- At 8192 iterations / W4, borrowing becomes materially useful. **4,908 / 12,800
  units (38.3%)** execute on all 16 stable-pool SMs between frames, adding
  **601.4 ms** of borrowed kernel presence and raising stable-pool presence from
  **32.3% to 44.3%**. No borrowed kernel overlaps a stable kernel. This matches
  dynamic's intermediate frontier: stable p99 is **0.638–0.969 ms below
  sharing**, while burst p95 remains **21.2–23.1% longer** because each dynamic
  lane is still only 16 SMs.
- W4 succeeds where W1 does not because it keeps future launches available to
  the GPU. In the audited dynamic runs, **99.2–99.6%** of within-burst successor
  units at W4 are enqueued before the preceding unit is observed complete;
  W1 is **0%** by construction. W1 actually records more borrowed units, so
  borrowing count alone does not explain the win.

The elastic kernel reaches the same six-block/SM peak in almost every route and
averages **50.8–57.8%** of its six-block theoretical residency while an SM has
elastic work. The policies therefore do not separate mainly through a different
per-active-SM occupancy limit. Pool width, the time at which a pool accepts work,
and queue depth explain the observed ordering more directly.

## Evidence used

This is a replay of the audited repetition-zero twins from the sealed
[mixed-workload study](resource_mixed_20260916.md), not a new timing frontier.
Latency and burst-completion comparisons continue to use its 51 unprofiled
timing runs. The hardware replay uses:

- CUPTI kernel start/end time, execution context and graph records;
- all 3,276,800 elastic block stamps per condition (SM ID and per-SM `clock64`
  start/end);
- the verified direct/graph SM-ID pool sets;
- native enqueue/completion records and queue-window ledgers.

The [machine-readable replay](resource_hardware_20260916.json) contains all
derived rows. Raw evidence remains in
`~/.local/state/saccade/perf/resource-419-mixed-final-20260916/`.

## Kernel presence and concurrency

Percentages below are union time within the approximately five-second stable
service horizon. “Present” means at least one traced kernel of that class is
executing; it is not an SM-active percentage or an occupancy counter. Overlap is
wall time in which stable and elastic kernels both execute, whether on disjoint
or shared SM pools.

| Iterations | Policy | Stable kernel present | Elastic kernel present | Stable + elastic overlap | Elastic kernel p50 | Stable pool incl. borrow |
|---:|---|---:|---:|---:|---:|---:|
| 2048 | fixed | 32.14% | 7.71% | 49.5 ms | 35.0 us | — |
| 2048 | shared | 21.68% | 4.34% | 30.9 ms | 19.7 us | — |
| 2048 | dynamic | 32.19% | 5.79% | 36.9 ms | 31.7 us | 35.36% |
| 8192 | fixed | 32.12% | 29.79% | 327.2 ms | 142.1 us | — |
| 8192 | shared | 22.92% | 15.66% | 147.0 ms | 89.2 us | — |
| 8192 | dynamic | 32.32% | 19.56% | 172.9 ms | 118.2 us | 44.32% |

Two points matter when reading the shorter shared presence times. First, shared
elastic blocks visit all 32 SMs rather than 16, and the individual elastic
kernels finish faster. Second, less time with work in flight can be evidence of
faster completion, not unused capacity. This is why “percent active” alone would
misread the frontier.

Fixed reservation's stable context has no traced stable kernel for about
**67.9%** of the horizon at both unit sizes, but elastic work is prohibited from
that pool. That is an observed opportunity boundary, not a claim that all 16 SMs
are device-idle: copy engines, shared caches, the 14 SMs outside the experiment
and display activity remain outside this kernel-presence measure.

Shared execution uses the opposite rule. Both stable and elastic work occupy
one 32-SM context; high stable stream priority does not create an exclusive
floor. At large units the trace contains 147.0 ms of simultaneous stable and
elastic kernel execution. The concurrent execution, higher large-unit stable
p99, and shorter elastic completion are consistent with a latency/throughput
trade: capacity remains work-conserving, while stable work can still share
execution and non-SM resources with already active elastic work. These
observations do not isolate scheduling, cache, power or memory as the sole
latency cause.

## Borrowed capacity is real and protected

| Iterations | Borrowed units | Stable SM IDs reached | Borrowed kernel presence | Borrowed/stable kernel overlap | Stable-pool presence |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4,987 / 12,800 | 16 / 16, exact set | 158.7 ms | 0 ns | 32.19% → 35.36% |
| 8192 | 4,908 / 12,800 | 16 / 16, exact set | 601.4 ms | 0 ns | 32.32% → 44.32% |

The permanent elastic lane also reaches its exact 16-SM set. Borrowing is
therefore neither a controller-only label nor capacity that was reserved but
never exercised. Conversely, it is not migration or preemption: previously
launched borrowed kernels finish, the lane drains, and only future stable
kernels are admitted. Dynamic still runs its permanent 16-SM elastic lane
concurrently with stable work, so the result is protection from borrowed work,
not device-wide isolation.

## Elastic block residency

`cuobjdump` reports 14 registers/thread and 2,048 bytes static shared memory for
the 256-thread elastic block. This device reports 1,536 threads, 65,536
registers, 102,400 shared-memory bytes and 24 blocks per SM; the thread limit
therefore caps this kernel at six resident blocks (48 warps) per SM.

| Iterations | Policy / physical pool | SMs observed | Peak blocks/SM | Mean blocks while SM active | Residency / 6 |
|---:|---|---:|---:|---:|---:|
| 2048 | fixed elastic pool | 16 | 6 | 3.142 | 52.4% |
| 2048 | shared pool | 32 | 6 | 3.048 | 50.8% |
| 2048 | dynamic permanent pool | 16 | 6 | 3.058 | 51.0% |
| 2048 | dynamic borrowed pool | 16 | 5 | 3.068 | 51.1% |
| 8192 | fixed elastic pool | 16 | 6 | 3.467 | 57.8% |
| 8192 | shared pool | 32 | 6 | 3.415 | 56.9% |
| 8192 | dynamic permanent pool | 16 | 6 | 3.303 | 55.1% |
| 8192 | dynamic borrowed pool | 16 | 6 | 3.305 | 55.1% |

The mean is weighted by per-SM cycles with at least one elastic block resident.
It is derived within each SM clock domain, so it does not subtract timestamps
across SMs. It includes launch edges and tail waves; it is an achieved
block-residency measure for the elastic kernel, not whole-pipeline achieved
occupancy. The near-equal values make a policy-specific occupancy limit an
unlikely primary explanation for the frontier. Sharing wins elastic completion
by applying comparable per-active-SM residency across twice as many SMs.

## Admission window and scheduler supply

| Iterations | Window | Successor enqueued before prior completion | Borrowed units | Audit transition p99 |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 0 / 12,700 | 5,421 | 0.269 ms |
| 2048 | 4 | 12,654 / 12,700 (99.64%) | 4,987 | 0.341 ms |
| 8192 | 1 | 0 / 12,700 | 5,399 | 0.320 ms |
| 8192 | 4 | 12,602 / 12,700 (99.23%) | 4,908 | 0.678 ms |

W4 exposes multiple future launches per lane, reducing dependence on host
completion observation between units. W1 lends the stable lane just as often
by unit count, but cannot keep a successor queued. This matches the sealed
frontier: dynamic W1 is 1.5–6.5% slower than fixed depending on unit size,
whereas dynamic W4 is 5.0–32.2% faster. The transition interval still grows at
large-unit W4 and is part of stable response, so W4 is not free capacity.

## Why direct GPU counters are excluded

Nsight Systems 2026.3.2 counter collection was qualified with node tracing,
whole-graph tracing, CUDA tracing disabled, the 277-metric `gb20x-top` set, the
25-metric `gb20x` set, and 1,000/100 Hz sampling. It perturbed multi-context
Green execution asymmetrically even when CUDA activity tracing was disabled.

The fail-closed control is fixed/8192/W4: the narrow `gb20x` set reported burst
p95 **98.862 ms at 1,000 Hz** and **99.316 ms at 100 Hz**, versus
**40.714–41.477 ms** in the sealed timing runs; stable p99 moved to
**13.224–13.617 ms** versus **8.476–8.601 ms**. A shared/8192 metrics-only run
remained close to baseline (burst p95 22.782 ms), so the disturbance is not a
common additive cost that could be subtracted. No sampled SM-active or warp-
occupancy comparison from those runs is used here.

Excluded qualification archives are retained under
`~/.local/state/saccade/perf/resource-419-hardware-{preflight,preflight2,preflight3,preflight5,final,final2,final3,final4}-20260916/`.
The original 100-ms NVIDIA-SMI telemetry remains device-wide context and is
also not used for per-policy attribution.

## Interpretation boundary

The best-supported explanation is compositional:

1. **Fixed reservation protects latency by construction but strands future-work
   opportunity.** Its elastic work is confined to 16 SMs even when the stable
   context has no traced kernel.
2. **Sharing is the most work-conserving route.** Stable and elastic kernels use
   one 32-SM pool; elastic kernels finish fastest and both workloads spend less
   wall time present. Large elastic units can coexist with stable execution,
   which is consistent with sharing's higher large-unit stable tail.
3. **Borrowing converts some stable gaps into elastic progress without allowing
   borrowed/stable overlap.** It becomes valuable when units are long enough
   and the window is deep enough to amortize drain and keep launches supplied.
   It retains more stable protection than sharing but less completion capacity.

This does not prove that stream-priority scheduling, occupancy, cache pressure,
DVFS or one other mechanism singly causes the latency difference. Cache,
bandwidth, copy engines and power management remain shared; clocks were not
locked. One audited repetition supplies the detailed hardware replay, and
CUPTI overhead means its durations are descriptive diagnostics, not substitutes
for the three-repetition timing frontier.

## Reproduction

```bash
uv run python scripts/benchmarks/resource_mixed/hardware_report.py \
  ~/.local/state/saccade/perf/resource-419-mixed-final-20260916 \
  --json-output /absolute/new/resource_hardware.json
```

The replay fails on invalid audited points, missing kernels, incorrect borrowed
counts, borrowed/stable overlap, short SM coverage or corrupt block stamps. It
does not rerun the workload or accept the excluded profiler archives as timing
evidence.
