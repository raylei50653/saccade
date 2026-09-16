# #419 time-budget elastic admission

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-15 -->
<!-- doc-module: cross -->

## Decision

**The time budget provides a useful measured drain-control frontier, while
retaining useful elastic throughput. It does not establish a hard drain bound
or uniformly outperform small fixed windows.** Across the three routes,
B=3 ms gives drain p99 **1.807–3.315 ms**, versus **7.222–20.232 ms** for
window 16. Its paired throughput medians are **93.1%–108.3%** of window 16,
comparing the same repetition/role within each route. These ranges describe
this workload on this host; they are not confidence intervals or a selected
production operating point.

The estimate ledger passed every admission check, but **28/4,320 budget-policy
arrivals (0.65%) drained later than B**. The observed frontier satisfies the
exploratory tradeoff objective; a deterministic interpretation of the budget
is rejected by the measurements. Keep a reservation plus a bounded admission
ledger as a candidate, and qualify estimator error before ownership changes.

## Question and contract

Does bounding estimated in-flight service time give a more portable drain
control than bounding the count of work units, while retaining elastic throughput?
Stable response and observed empty-lane drain are measured separately.

The [harness contract and reproduction commands](../../../scripts/benchmarks/resource_elastic/README.md#time-budget-admission-with-repeated-arrivals)
define the frozen workload, calibration and metric semantics. This augments the
[confirmed-start fixed-window study](resource_admission_20260915.md), whose
arrivals did not exercise pre-arrival replenishment.

The new controller retains each unit's calibrated duration charge until its
completion event is observed. It admits the next FIFO unit only if the resulting
sum is at most B. Training uses per-route/role/size p95 GPU duration × 1.2;
evaluation uses frozen estimates. This is a conservative remaining-service
estimator with possible calibration error, not a hardware remaining-time query.

Matched trials have 256 mixed-duration units (512/2048/4096 iterations), four
fixed host targets 12 ms apart, and the same ordered workload for every policy.
At each arrival, pause admission until both stable work and the admitted prefix
finish, then replenish. Compare window 1/4/16 with B=3/6/12 ms on disjoint
16+16 SMs, shared 32-SM priority, and full-device priority. Only the disjoint
route supplies a fixed spatial floor. No routing/ownership changes occur.

## Measured results

<!-- BEGIN measured tables -->
Ranges across repetition/role quantiles; p99 is descriptive over correlated arrivals.

| Route | Policy | Stable p99 ms | Drain p99 ms | Units/s p50 | Dispatch ms p50 |
|---|---|---:|---:|---:|---:|
| disjoint | B=3 ms | 0.506–0.832 | 1.807–2.182 | 1115.207–1200.351 | 2.298–2.722 |
| disjoint | B=6 ms | 0.631–0.680 | 4.116–5.435 | 1116.558–1235.599 | 2.317–2.700 |
| disjoint | B=12 ms | 0.638–0.750 | 7.712–9.292 | 1120.683–1206.327 | 2.268–2.602 |
| disjoint | W=1 | 0.582–0.675 | 1.607–1.880 | 1088.215–1171.393 | 2.552–3.188 |
| disjoint | W=4 | 0.640–0.810 | 5.086–6.112 | 1119.549–1204.478 | 2.502–2.855 |
| disjoint | W=16 | 0.611–0.949 | 17.185–20.232 | 1126.252–1213.127 | 2.370–2.684 |
| full_priority | B=3 ms | 0.518–1.018 | 2.708–3.290 | 2937.471–3212.560 | 1.778–2.019 |
| full_priority | B=6 ms | 0.499–1.047 | 5.609–6.681 | 2944.532–3183.919 | 1.752–2.007 |
| full_priority | B=12 ms | 0.517–0.673 | 10.332–11.954 | 2929.526–3019.799 | 1.644–1.971 |
| full_priority | W=1 | 0.469–0.701 | 0.771–1.025 | 2759.378–2858.922 | 2.008–2.547 |
| full_priority | W=4 | 0.544–0.699 | 2.011–2.846 | 2895.443–3211.115 | 1.884–2.386 |
| full_priority | W=16 | 0.596–2.374 | 7.222–8.160 | 2925.027–3111.792 | 1.722–2.093 |
| shared_budget_priority | B=3 ms | 0.578–0.721 | 2.071–3.315 | 2147.700–2325.412 | 1.991–2.431 |
| shared_budget_priority | B=6 ms | 0.646–1.634 | 4.344–6.640 | 2136.032–2202.781 | 1.911–2.112 |
| shared_budget_priority | B=12 ms | 0.528–0.744 | 7.907–11.149 | 2150.740–2328.260 | 1.970–2.188 |
| shared_budget_priority | W=1 | 0.371–0.714 | 0.916–1.130 | 2024.048–2259.290 | 2.107–2.677 |
| shared_budget_priority | W=4 | 0.332–0.828 | 2.814–3.334 | 2181.980–2313.749 | 2.079–2.309 |
| shared_budget_priority | W=16 | 0.588–0.999 | 9.950–10.392 | 2147.110–2324.079 | 1.941–2.456 |
<!-- END measured tables -->

## What the frontier establishes

1. **Time budgets adjust the admitted count to service speed and unit size.**
   At B=3 ms, maximum unretired counts at arrival were 5/10/12 on disjoint,
   shared 32-SM and full-device routes. A fixed window 16 has drain p99
   17.185–20.232 / 9.950–10.392 / 7.222–8.160 ms respectively. Calibrating
   time gives a more comparable drain scale across these routes, while
   reservation and shared priority remain distinct resource models.
2. **Fast stable completion can hide delayed arrivals.** Disjoint window 16
   has submission-response p99 0.611–0.949 ms but scheduled-response p99
   **5.680–8.719 ms**, with maximum arrival overshoot **9.248 ms**. Disjoint
   B=3 ms has both response p99 ranges 0.506–0.832 ms. The fixed target clock
   exposes delay carried from an earlier drain into a later request.
3. **This actually tests rolling replenishment.** Across all policies,
   6,240/6,480 later arrivals follow a retirement and new dispatch in the same
   active interval. Budget policies cover 3,225/3,240; B=3 ms covers all
   **1,080/1,080**. Disjoint window 16 covers only 136/360 because long pauses
   often pass the next target. Those cases remain in the data. No arrival
   observes the whole 256-unit workload already retired.
4. **Small fixed windows remain competitive.** Full-device window 4 gives
   drain p99 2.011–2.846 ms and throughput p50 2,895–3,211 units/s, versus
   B=3 ms at 2.708–3.290 ms and 2,937–3,213 units/s. Window 1 gives lower
   drain still, at 2,759–2,859 units/s. These overlapping ranges do not establish
   a universal superiority or a stable-tail ordering.
5. **Dispatch and pause have different meanings.** Full-device window 1
   spends 2.008–2.547 ms per trial in elastic launch/event calls, versus
   1.778–2.019 ms for B=3 ms. Its pause p50 is 0.248–0.307 ms versus
   1.793–2.317 ms for B=3 ms. Pause time includes useful elastic work draining;
   it is not all lost capacity. Whole-trial throughput includes these costs,
   host polling and GPU gaps; this study does not separately identify the
   causal cost of each component.

### Estimate error and observed exceedance

Each row below contains 480 stable arrivals. Counts are empirical observations,
not a tolerated miss rate or an estimated future failure probability.

| Route | B ms | Drain > B | Maximum observed drain ms |
|---|---:|---:|---:|
| disjoint | 3 | 0/480 | 2.735 |
| disjoint | 6 | 1/480 | 10.204 |
| disjoint | 12 | 0/480 | 9.435 |
| shared_budget_priority | 3 | 4/480 | 3.326 |
| shared_budget_priority | 6 | 2/480 | 10.746 |
| shared_budget_priority | 12 | 1/480 | 13.558 |
| full_priority | 3 | 12/480 | 3.477 |
| full_priority | 6 | 7/480 | 8.657 |
| full_priority | 12 | 1/480 | 12.047 |

Of 276,480 elastic units in budget-policy trials, **17,304 (6.26%)** have GPU
envelope duration above their frozen isolated estimate. B=6 ms still has a
maximum observed drain of 10.746 ms; selecting only p99 would hide this failure.
Service variation, co-load and host observation delay are not causally isolated
here. Retaining the full charge until retirement prevents fictitious queue-age
progress, but cannot fix underestimation or polling delays.

## Evidence boundary

Drain denotes host observation of an empty elastic lane; no SM reclamation or
ownership transfer is performed. Stable workload is synthetic compute, not a
memory-sensitive workload or full pipeline. Throughput counts equivalent
2048-iteration elastic units per whole-trial second, including stable pauses.

The four targets stay fixed if a preceding drain delays later submissions;
report scheduled response and overshoot alongside submission response. This
finite serial controller is not a general open-loop arrival generator. p99
uses 80 correlated arrivals per repetition/role, with no tail filtering or
latency guarantee. Static calibration cannot remove clock drift or contention. Shared-route role
labels use independently calibrated estimates on the same physical pool; only
disjoint roles swap physical pools. Role ranges therefore include estimator
variation as well as execution drift. This tests one frozen workload mixture,
not generalization to a different duration distribution.

The first pilot rejected B=2 ms because the longest calibrated unit exceeded
it. The qualifying pilot used B=3/6/12 ms and 12-ms targets. Both are separate
from the main data. All budget violations remain in the observed results;
passing the estimate ledger is not equivalent to meeting a real-time deadline.

## Retained artifacts and validation

Main archive: `~/.local/state/saccade/perf/resource-419-time-budget-20260915/`.
The failed initial calibration and qualifying pilot use separate
`resource-419-time-budget-pilot-20260915` and
`resource-419-time-budget-pilot2-20260915` directories and are excluded.
Main design: 108 conditions, 2,160 retained trials, 8,640 stable arrivals;
324 warmup trials and 540 retained isolated calibration units are separate.

The [machine-readable replay](resource_time_budget_20260915.json) retains
per-repetition/role quantiles, calibration estimates, source identities, pool
sets and coverage counts. The [separate audit](resource_time_budget_20260915.audit.json)
retains extrema, estimate exceedance and rolling-admission counts using a
vectorized implementation that does not import the producer's calculations.
The [frontier figure](resource_time_budget_20260915.svg) shows ranges rather
than hiding repetition/role variation.

Validation completed:

- All 108 conditions passed checksum/source, calibration, workload, ledger,
  freeze, output and SM-membership replay; every saved quantile was re-derived.
- A separate vectorized audit passed all 2,160 trials; its budget-miss and later
  replenishment counts agree with the reporter for all 18 policy/route groups.
- 39 focused unit tests passed. Pilot archive copies with regenerated checksums
  rejected changed quantiles, calibration estimates, condition coverage and
  admission ledgers. Faults were not injected into the measurement archives.
- Ruff lint/format, script/test indexes, document master map, stale-path checks
  and `git diff --check` passed. Index generation used a temporary Git index.

The GPU execution source snapshots remain authoritative. The replay record also
identifies its own reporter source hash, including the additional check that
first dispatch precedes start observation. This post-execution replay check
does not alter the controller or any timing metrics.

## Next bounded question

Before ownership/routing work, test calibration robustness with a frozen
held-out duration mix and a declared drain exceedance target. Compare a
conservative calibrated estimate with an estimator updated only from retired
units, retaining the same FIFO and oversize-unit rules. Any more aggressive
elapsed-time deduction must distinguish running time from queue age and account
for estimator overrun. This is a proposed continuation, not an executed result.
