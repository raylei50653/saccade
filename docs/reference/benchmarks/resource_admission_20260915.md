# #419 confirmed-start arrivals and bounded admission

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-15 -->
<!-- doc-module: cross -->

## Decision

**Bound both work-unit duration and admitted work count.** Small kernels with a
high-priority stable stream can give fast stable completion even while the
elastic lane has milliseconds of already-admitted work left. A fast stable
response alone does not establish a quick opportunity to change lane ownership.

For the full-device 16-chunk route at offset zero, reducing the window from 16
to 1 changes stable p50 only from **0.190–0.200 ms to 0.192–0.204 ms**, but
reduces observed prefix-drain p50 from **4.682–5.179 ms to 0.332–0.352 ms**.
Window 4 gives an intermediate **1.228–1.237 ms** drain. These are ranges across
six repetition/role quantiles, not confidence intervals or guarantees.

## Question and executed design

Follow up the [elastic scheduling exploration](resource_elastic_20260915.md):
after burst work has really entered a kernel, how long does stable work wait,
and how much does limiting admitted work shorten the wait to empty the lane?

The [harness contract and commands](../../../scripts/benchmarks/resource_elastic/README.md#confirmed-start-arrival-and-bounded-admission-follow-up)
define the exact experiment. The new harness preserves 32768 nominal burst
iterations per thread, compares one kernel against 16 independent kernels,
and bounds submitted-but-not-retired chunks to 1, 4 or 16. At arrival it freezes
admission, observes stable completion and prefix drain, then resumes the rest.
The total burst work remains constant. It does not dynamically assign SMs.

A system-scope mapped **load/store** handshake from block zero of the actual
burst kernel confirms entry before the host starts a 0 or 250 µs delay. This
uses the naturally aligned mapped load/store case in NVIDIA's
[CUDA memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html),
not a mapped read/modify/write atomic. The host reports when it observes stable
start publication and completion. GPU per-block timestamps remain in their own
clock domain. Thus start-observation latency is an upper bound, not an exact
host-enqueue-to-GPU-start measurement.

- Routes: 16+16 disjoint Green pools, one shared 32-SM Green pool with stable
  priority, and the 46-SM full-device priority baseline. The first two match
  total SM count; shared 32-SM physical IDs come from an independent split.
- 48 conditions per repetition: three routes × four chunk/window pairs × two
  offsets × two role assignments; all shuffled together with seed 420.
- Three repetitions, five warmups and 50 retained samples per condition:
  **7,200 retained trials and 720 warmups**. Roles physically swap only on the
  disjoint route; shared-route role labels are repeated controls.
- RTX 5070 Ti Laptop, WSL2. The archive retains exact driver/compiler identity
  and 100-ms clocks/power/temperature/utilization telemetry. No clock lock or
  dedicated host CPU isolation; this is an interactive development host.
- Archive: `~/.local/state/saccade/perf/resource-419-admission-20260915/`.
  The separate 96-trial `resource-419-admission-pilot-20260915` qualified the
  harness and is excluded from the main results.

## Measured results

<!-- BEGIN measured tables -->
Ranges are min–max of repetition/role quantiles, in ms; not confidence intervals.

| Route | Chunks | Window | Offset µs | Stable p50 | Stable p99 | Drain p50 | Burst host p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| disjoint | 1 | 1 | 0 | 0.270–0.279 | 0.297–0.538 | 12.378–13.023 | 12.411–13.121 |
| disjoint | 1 | 1 | 250 | 0.252–0.279 | 0.298–1.057 | 12.143–13.313 | 12.429–13.609 |
| disjoint | 16 | 1 | 0 | 0.286–0.289 | 0.297–0.683 | 0.776–0.782 | 13.312–13.807 |
| disjoint | 16 | 1 | 250 | 0.286–0.293 | 0.326–0.695 | 0.527–0.538 | 13.363–13.746 |
| disjoint | 16 | 4 | 0 | 0.279–0.288 | 0.312–0.765 | 3.126–3.382 | 13.015–13.859 |
| disjoint | 16 | 4 | 250 | 0.265–0.291 | 0.309–0.710 | 2.876–3.645 | 12.968–14.957 |
| disjoint | 16 | 16 | 0 | 0.276–0.287 | 0.323–0.755 | 12.871–15.162 | 12.968–15.444 |
| disjoint | 16 | 16 | 250 | 0.260–0.291 | 0.307–1.125 | 12.565–14.164 | 12.915–14.587 |
| full_priority | 1 | 1 | 0 | 1.130–1.203 | 1.215–1.691 | 4.643–4.986 | 4.674–5.065 |
| full_priority | 1 | 1 | 250 | 0.933–0.951 | 0.966–1.382 | 4.395–5.327 | 4.684–5.699 |
| full_priority | 16 | 1 | 0 | 0.192–0.204 | 0.208–0.724 | 0.332–0.352 | 5.144–6.286 |
| full_priority | 16 | 1 | 250 | 0.126–0.141 | 0.152–1.102 | 0.026–0.047 | 5.193–6.007 |
| full_priority | 16 | 4 | 0 | 0.202–0.205 | 0.211–0.426 | 1.228–1.237 | 4.824–4.858 |
| full_priority | 16 | 4 | 250 | 0.150–0.154 | 0.239–0.810 | 0.973–0.976 | 4.804–5.292 |
| full_priority | 16 | 16 | 0 | 0.190–0.200 | 0.231–0.582 | 4.682–5.179 | 4.788–5.312 |
| full_priority | 16 | 16 | 250 | 0.172–0.207 | 0.233–0.676 | 4.260–4.619 | 4.784–5.238 |
| shared_budget_priority | 1 | 1 | 0 | 1.189–1.203 | 1.229–2.543 | 6.433–7.810 | 6.475–8.001 |
| shared_budget_priority | 1 | 1 | 250 | 0.933–0.957 | 0.971–2.061 | 6.186–7.681 | 6.471–8.199 |
| shared_budget_priority | 16 | 1 | 0 | 0.204–0.204 | 0.210–0.634 | 0.494–0.499 | 6.990–7.397 |
| shared_budget_priority | 16 | 1 | 250 | 0.203–0.228 | 0.243–1.312 | 0.249–0.254 | 7.007–8.743 |
| shared_budget_priority | 16 | 4 | 0 | 0.201–0.204 | 0.207–0.722 | 1.704–1.711 | 6.619–7.360 |
| shared_budget_priority | 16 | 4 | 250 | 0.203–0.217 | 0.270–0.646 | 1.462–1.465 | 6.617–7.531 |
| shared_budget_priority | 16 | 16 | 0 | 0.192–0.204 | 0.244–1.238 | 6.462–7.415 | 6.577–7.580 |
| shared_budget_priority | 16 | 16 | 250 | 0.186–0.215 | 0.240–0.534 | 6.228–6.617 | 6.573–6.970 |
<!-- END measured tables -->

### What changed and what it costs

1. **Kernel granularity still matters after confirmed burst entry.** At offset
   zero, full-device one-kernel stable p50 is 1.130–1.203 ms; with 16 chunks it
   is 0.190–0.205 ms across tested windows. Stable start-publication observation
   p50 changes from 0.923–0.972 ms to 0.032–0.061 ms. The latter is a host
   observation upper bound on first-start delay, not exact queue latency.
2. **A bounded window controls empty-lane wait separately from stable response.**
   The 32-SM shared route repeats the distinction: at offset zero, window 16
   versus 1 gives drain p50 6.462–7.415 versus 0.494–0.499 ms while stable p50
   stays around 0.192–0.204 ms. Disjoint pools also require 12.871–15.162 ms to
   drain a fully queued burst versus 0.776–0.782 ms for a one-chunk window.
3. **Reservation protects against the monolithic burst in this matched budget.**
   At offset zero, disjoint 16+16 stable p50 is 0.270–0.279 ms versus
   1.189–1.203 ms for shared 32 SM. With 16 chunks, shared 32 SM instead gives
   0.192–0.204 ms versus 0.276–0.289 ms disjoint. This does not select a minimum
   reservation or prove reservation unnecessary for other workloads.
4. **Window 1 adds dispatch/pause cost.** Full-device burst host p50 at offset
   zero is 5.144–6.286 ms with window 1, 4.824–4.858 ms with window 4, and
   4.788–5.312 ms with window 16. The ranges overlap and are descriptive;
   there is no selected throughput optimum. Host totals include the whole
   equal-nominal-work burst and the stable pause. Useful application throughput
   is not measured.
5. **Arrival phase needs explicit accounting.** At offset 250 µs, 77/300
   full-device window-1 trials already finish the admitted prefix before
   stable GPU execution; shared 32-SM window 1 has 1/300. Across all 7,200
   trials, none has retired the entire burst at arrival. Prefix-finished
   cases are retained. The largest condition p99 arrival overshoot is
   1.971 ms (disjoint, 16 chunks/window 16, requested offset 250 µs).
   Requested offsets therefore do not establish tightly controlled GPU phases.

The two 16-SM pools were disjoint, and the independently constructed 32-SM pool
happened to equal their exact SM-ID union in this run. All before/after probes
and all measured block memberships passed. The
[machine-readable record](resource_admission_20260915.json) retains per-condition
p50/p95/p99, arrival/drain diagnostics, source identities and pool sets.

## Interpretation boundary

The host detects start and completion by polling; the new response measurement
is not identical to the original stream-synchronize response. Initial window
submission precedes start-signal observation, so especially with a large window,
“offset zero” can already be well into the burst. Requested offsets refer to
host observation, not a fixed offset from actual GPU start. Retain overshoot and
start-observation distributions when comparing windows.

An event that has not yet been observed complete does not prove a kernel is
still resident. The trace records retired chunks at arrival and whether the
admitted prefix finishes before stable GPU execution, without filtering such
samples. There is no open-loop arrival process, memory-sensitive stable control,
full pipeline, borrowing, owner transfer or service guarantee in this experiment.
No deadline or tolerated miss rate was chosen; p99 is descriptive over 50 samples
per condition, not an estimated rare-tail guarantee.

Raw stamps include every block's GPU begin/end/SM ID; raw host arrays include
all chunk enqueue and observed completion times. Replay checks the admission
window and freeze interval, completion order, start-signal identity, all pool
membership, stable outputs across conditions and all quantiles. This native
synthetic routing audit says nothing about hidden library streams or CUDA graphs.

## Next bounded question

The next synthetic step should calibrate a **remaining-time admission budget**:
use measured chunk-duration distributions to decide how many units may enter,
and evaluate the response/drain tradeoff under a frozen repeated-arrival model.
**None of the 7,200 trials replenishes a chunk before arrival**; the current
offsets exercise only the initial admitted prefix. Rolling replenishment is
exercised after the pause, not as a sustained pre-arrival controller test. This
is not validation of a controller under sustained replenishment. Include later arrivals
that demonstrably retire and replenish several chunks, and account for arrival
phase error before comparing an adaptive budget with fixed windows 1/4/16.

A minimum reservation plus such a budget remains the research direction.
Actual lane reassignment requires a separate routing/ownership experiment;
full-pipeline safe units and memory-sensitive service remain subsequent work.

## Validation and retained state

- Native pilot: 96 retained trials. Main: 7,200 retained trials and 720 warmups;
  all 144 conditions completed, with no sample filtering or retries.
- Main archive checksums and source identities verified; all saved quantiles
  independently re-derived from retained host/block/output arrays.
- 24 unit tests passed: 16 new arrival/admission/checksum cases plus the eight
  original elastic reporter tests.
- Fault-injected archive copies with regenerated checksums reject altered
  quantiles, an exceeded queue bound and inconsistent source identity.
- Ruff lint/format, script/test indexes, document master map, stale-path checks
  and `git diff --check` passed. Index generation included the new files through
  a temporary Git index; the real staging area stayed unchanged.
- Production sources/presets are unchanged. The measurements and checks above
  describe the local research handoff; PR publication is tracked separately.

## Time-budget continuation

The [time-budget study](resource_time_budget_20260915.md) now evaluates 2,160
mixed-duration trials with four scheduled stable arrivals each, calibrated
remaining-work ledgers, fixed-window controls and pre-arrival replenishment.
It records a useful drain/throughput frontier and empirical budget exceedances;
no ownership transfer or deterministic latency is established.
