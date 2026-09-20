# #419 closure — serial vs double-buffer SM-scaling frozen benchmark

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-20 -->
<!-- doc-module: cross -->

## Scope and interpretation

This is the closure record for the SM-scaling half of
[#419](https://github.com/raylei50653/saccade/issues/419): serial versus
double-buffer (`detect(N+1) || tracker(N)`) throughput, frame latency,
completion period and jitter measured at the same full-pipeline boundary
(decode → detector → tracker → completed output, #35/#376) inside verified CUDA
Green Context partitions on the unconstrained device plus every creatable
constrained budget. It re-runs the phase-B harness
([resource_partition_20260915.md](resource_partition_20260915.md)) with one
more repetition, applies a knee rule that was fixed **before** the run (below),
and freezes the result as a benchmark record that a later optimization can be
compared against with `frozen_benchmark.py compare`.

It answers two things only: whether a double-buffer overlap knee exists on the
creatable budget grid of this device, and what the resource frontier (minimum
verified SM budget meeting a declared service target) is today. It does not
add or evaluate any routing policy, does not attribute the curve to a mechanism
(occupancy, clocks, scheduler, cache, power), and is not an edge-device claim.
Production presets, defaults and sources are unchanged.

## Pre-declaration

Written and committed before the closure sweep started. The constants live in
`scripts/benchmarks/resource_partition/frozen_benchmark.py` and are part of
the frozen record's schema; changing them is a new schema version, not an
edit.

**Sweep.** `mamba_whole_graph_m`, MOT17-04-SDP frames 1–300 (warmup 1–50,
250 timed frames), `--detect-barrier event`, default GPU decode; requested
budgets `full 46 40 32 24 16 8`; **3 repetitions**, levels shuffled per
repetition (`--seed 419`), mode order alternating; every point run as an
audited twin (CUPTI on, produces the routing verdict) and a measurement (CUPTI
off, produces the timing). Only `true_partition_validated` rows enter the
curve; the `full` primary-context rows are the reference and never a partition.
Run under the `machine-bench` lease.

**Curve.** Per validated Green Context budget `r`, `g(r)` = mean over
repetitions of the paired DB gain `FPS_double / FPS_serial` (same repetition);
`g_full` = the same mean for the primary full-device rows. Retained fraction
`f(r) = (g(r) − 1) / (g_full − 1)`. Segment slope between adjacent grid points
`s_i = (f(r_{i+1}) − f(r_i)) / (r_{i+1} − r_i)`.

**Knee rule.** A knee exists at `r_i` iff, for some segment `i` that has at
least one higher segment, both hold:

1. `s_i ≥ 2 × max_{j>i} s_j` — the retained-gain decline per SM in the segment starting at `r_i` is
   at least twice the steepest decline anywhere above it (if every higher
   slope is ≤ 0, any positive `s_i` qualifies); and
2. `f(r_i) < 0.5` — less than half the full-device DB gain survives at the low
   end of that segment.

Otherwise the outcome is **graceful scaling** if `f(r) ≥ 0.8` at every
constrained budget, else **smooth decline**. Separately, any budget with
`g(r) ≤ 1.05` is listed as having lost the overlap. The rule is evaluated on
the repetition means and again per repetition; the record states whether the
per-repetition verdicts agree. Fewer than three validated budgets or
`g_full ≤ 1` make the rule inapplicable and are reported as such.

The rule was written with knowledge of the 09-15 two-repetition curve, which it
classifies as *smooth decline* (steepest segment 8→16 SM at 0.031/SM versus
0.025/SM for 24→32; `f(8) = 0.30`). It is therefore a pre-declaration for the
closure run only, not a blind one, and a closure verdict other than *smooth
decline* would be reported as a disagreement with 09-15, not averaged away.

**Frontier targets** (each evaluated per mode; a budget meets a target only if
**every** repetition meets both bounds; the frontier value is the smallest
verified Green Context budget that meets it, with the full device reported
beside it):

| Target | FPS ≥ | Frame p99 ≤ |
|---|---:|---:|
| T1 | 300 | — |
| T2 | 200 | 10 ms |
| T3 | 150 | 15 ms |
| T4 | 100 | 25 ms |

These are probes of the frontier, not service commitments; no production
deadline exists (#419), so no deadline-miss rate is claimed.

**Comparison contract for future use.** `frozen_benchmark.py compare` refuses
(exit 2) unless preset, sequences, frame bounds, warmup, barrier, requested
levels, device name/SM count/partition alignment and the criteria constants are
identical; it records (does not reject) differences in runtime `HEAD`, harness
source hashes, driver and library versions. It reports candidate/reference
ratios of means per budget and mode, DB-gain deltas, whether repetition ranges
overlap, both knee verdicts and frontier movement. Cross-session absolutes on
this host have drifted 7–10% at identical clocks
([resource_mixed_db_20260918.md](resource_mixed_db_20260918.md)), so a
before/after claim needs a **same-session control** (the reference source
re-run alongside the candidate, passed as `--control`); without one the
comparison is emitted with `host_drift: null` and says so.

## Reproduction

```bash
.venv/bin/python tools/resctl.py run --timeout 3600 machine-bench -- \
  .venv/bin/python scripts/benchmarks/resource_partition/sweep.py \
  --output /absolute/new/resource-419-scaling-closure \
  --preset mamba_whole_graph_m --sequences MOT17-04-SDP \
  --levels full 46 40 32 24 16 8 --repeats 3 --max-frames 300
.venv/bin/python scripts/benchmarks/resource_partition/report.py \
  /absolute/new/resource-419-scaling-closure \
  --json-output resource_scaling_closure_20260920.study.json --markdown-output study_tables.md
.venv/bin/python scripts/benchmarks/resource_partition/frozen_benchmark.py build \
  /absolute/new/resource-419-scaling-closure \
  --study-record resource_scaling_closure_20260920.study.json \
  --json-output resource_scaling_closure_20260920.json --markdown-output tables.md
# later, for an optimization candidate:
.venv/bin/python scripts/benchmarks/resource_partition/frozen_benchmark.py compare \
  --reference docs/reference/benchmarks/resource_scaling_closure_20260920.json \
  --candidate /absolute/candidate-sweep [--control /absolute/same-session-reference-sweep] \
  --json-output comparison.json --markdown-output comparison.md
```

## Results

_Pending: filled from the closure sweep after this pre-declaration was
committed._
