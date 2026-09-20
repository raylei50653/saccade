# #419 acceptance closure — GPU resource-sensitivity characterization

<!-- doc-status: closed -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-20 -->
<!-- doc-module: cross -->

## Purpose

This is the closure ledger for
[#419](https://github.com/raylei50653/saccade/issues/419) (work package F of
#420). It aligns the existing evidence records against the issue's acceptance
checklist and states what the umbrella issue established. It adds **no new
measurement**, selects **no production routing policy**, attributes **no
mechanism** to any curve, and makes **no edge-hardware claim**. Every number
below is copied from the cited record; the records, not this page, are the
evidence.

## Decision statements

1. **SM scaling: `smooth_decline`, no resolved knee.** On the device's
   creatable Green Context grid (8/16/24/32/40/46 SM, 3 repetitions), the
   double-buffer gain falls monotonically 1.555 (full device) → 1.518 / 1.486
   / 1.391 / 1.303 / 1.158 at 40/32/24/16/8 SM (retained fraction 0.93 / 0.88
   / 0.70 / 0.55 / 0.29). Under the knee rule pre-declared before the run
   (segment slope ≥ 2× the steepest higher segment *and* < 50% gain retained
   at its low end) the steepest segment, 8→16 SM, is 1.5× the next; no budget's
   gain is within 0.05 of 1.0; all three repetitions agree. A knee below 8 SM
   or between grid points is not excluded.
   — [resource_scaling_closure_20260920.md](resource_scaling_closure_20260920.md),
   frozen record [`.json`](resource_scaling_closure_20260920.json); consistent
   with the 2-repetition phase-B curve
   ([resource_partition_20260915.md](resource_partition_20260915.md)).

2. **Shared execution is the most work-conserving route but can lose the
   stable tail under the large-unit load that was observed.** One 32-SM shared
   pool gives the highest elastic completion and DB throughput at every
   matched point (serial study: shortest burst p95 in all 12 same-repetition
   comparisons; DB study: 199.9–245.1 fps versus 157.5–168.9 for 16-SM
   policies) and the lowest stable p99 at three of four DB workload points —
   but at 8192-iteration / window-4 load it is the only policy that misses the
   re-declared 20 ms / 1% target (0/3, p99 22.9–23.6 ms, frame-period σ
   2.24–2.36 ms versus 0.93–1.19 ms partitioned) while fixed and dynamic hold
   (18.2–19.1 ms). The hardware replay's descriptive account is that shared
   large elastic units coexist with stable execution on the same SMs, which is
   consistent with the higher stable tail; it is not a mechanism proof.
   — [resource_mixed_20260916.md](resource_mixed_20260916.md),
   [resource_mixed_db_20260918.md](resource_mixed_db_20260918.md),
   [resource_hardware_20260916.md](resource_hardware_20260916.md).

3. **Reserved and headroom routes buy a lower stable tail with elastic
   completion / capacity.** Serial study, 8192 / W4: fixed 16+16 stable p99
   8.48–8.60 ms with burst p95 40.7–41.5 ms; 24-SM headroom p99 7.52–8.15 ms
   with burst p95 67.5–71.2 ms; shared 32 p99 9.78–9.96 ms with burst p95
   22.9–23.1 ms; dynamic borrowing sits between (p99 8.93–9.26 ms, burst p95
   27.8–28.5 ms, 30.7–32.2% faster than fixed, 21.2–23.1% slower than shared).
   Under the double-buffer schedule at 16 SM, dynamic borrowing tracks fixed
   reservation within the sweep (burst p95 ratio 0.894–1.151, DB fps ratio
   0.992–1.017; no consistent penalty resolved, not an equivalence result) and
   lends 0.4–1.9% of the offered load. Fixed reservation protects latency by
   construction and strands otherwise-idle stable-pool capacity for elastic
   work. None of these studies selects a policy; every one names sharing as a
   strong control that a spatial policy has not dominated on both axes.
   — same three records plus [resource_routing_20260916.md](resource_routing_20260916.md)
   (synthetic lanes).

## Acceptance checklist alignment

| # | Acceptance item (#419) | Evidence | Status |
|---|---|---|---|
| 1 | Reproducible serial vs double-buffer resource sweep exists | `scripts/benchmarks/resource_partition/` (`sweep.py`, `summarize.py`, `report.py`, `frozen_benchmark.py`), reproduction block in [resource_scaling_closure_20260920.md](resource_scaling_closure_20260920.md); re-run 09-15 → 09-20 within 3% | **satisfied** |
| 2 | ≥ 4 nonzero constrained levels + unconstrained baseline, or the limitation documented | 6 constrained levels (8/16/24/32/40/46 SM) + `full`, all `true_partition_validated`; 8-SM alignment and rejected 48 documented | **satisfied** |
| 3 | `DB_gain(r)` and absolute throughput both reported | per budget and per repetition, [closure](resource_scaling_closure_20260920.md) and [phase B](resource_partition_20260915.md) | **satisfied** |
| 4 | p50/p95/p99 latency and frame-period jitter for the same runs | same-run frame p50/p95/p99, period p50/p95/p99 and σ; DB/serial p99 ratio 2.0 (46–32 SM) → 3.1 (8 SM) | **satisfied** |
| 5 | Power/clock/utilization metadata retained | 100 ms `nvidia-smi` samples inside measured windows per run, summarized per budget/mode; clocks differ across budgets and are recorded, not attributed | **satisfied** |
| 6 | Persistent-blocker measurements labelled as a proxy | [resource_sensitivity_20260915.md](resource_sensitivity_20260915.md) is titled and read as a residency-pressure proxy; phase B presents proxy and true partition separately | **satisfied** |
| 7 | ≥ 1 true partition point validates or falsifies the proxy trend | Falsified: proxy K=8 ran 5.7 FPS with DB gain ≈1.03 versus a true 8-SM partition at 91 FPS / 1.17; proxy curve not citable for SM scaling ([phase B](resource_partition_20260915.md)) | **satisfied** |
| 8 | If detector/tracker partitioning is feasible, ≥ 1 shared-vs-reserved comparison reports tail-latency and throughput trade-off | Shared 32 SM vs fixed 16+16 vs 24-SM headroom vs dynamic borrowing, with the production serial and double-buffer pipelines as the stable lane and matched elastic bursts as the reserved-against load; tail (stable p99, period σ, target passes) and throughput/completion (burst p95, DB fps, units/s) reported per point ([09-16](resource_mixed_20260916.md), [09-18](resource_mixed_db_20260918.md)). **The reservation axis measured is stable-pipeline vs elastic work; a detector-vs-tracker spatial split (42/4, 38/8 …) was not executed** — the issue's own ladder C names it as optional and the studies above answer the shared-vs-reserved question the item asks for. | **satisfied (stable/elastic reservation; D/T split not measured)** |
| 9 | Final artifact reusable as a before/after optimization benchmark | frozen record + `frozen_benchmark.py compare` (scope fail-closed, frontier movement, same-session `--control` requirement), exercised against the 09-15 sweep | **satisfied** |

Decision-rule outcomes named in the issue: (1) graceful scaling — not met
(retained fraction < 0.8 below 32 SM); (2) knee — not resolved; (3)
shared-resource sensitivity — proxy and true partition disagree materially;
(4) tail-latency trade-off — observed (reserved/headroom lower stable tail at
elastic-completion cost; shared loses the tail only at 8192/W4); (5) no useful
isolation effect — not the outcome, but also not a policy selection.

## Non-claims and what stays open elsewhere

- **Edge hardware**: explicit non-claim, as the issue's scope states; every
  measurement is one RTX 5070 Ti Laptop / WSL2 host, one MOT17-04 prefix. It is
  not a blocker for closing this characterization issue.
- **Production routing policy**: not selected. Sharing has not been dominated
  on both axes by any spatial policy at the tested loads; the studies say a
  tighter, separately justified target or a held-out interference workload is
  the next question. That question belongs to a new issue, not to #419.
- **Mechanism**: not attributed. Clocks, L2, memory bandwidth, copy/NVJPG
  engines and power management stayed shared and varied across budgets.
- **Statistics**: three short repetitions with descriptive ranges; no
  confidence intervals, equivalence margins or rare-tail estimates. Absolute
  numbers are not comparable across sessions (7–10% host drift 09-17→09-18;
  3% 09-15→09-20).
- **Detector/tracker spatial split**: not measured (item 8 above).

## Frozen artifacts for future comparison

| Artifact | Use |
|---|---|
| [resource_scaling_closure_20260920.json](resource_scaling_closure_20260920.json) | `frozen_benchmark.py compare --reference` target; per-budget aggregates, knee verdict, frontier T1–T4 |
| [resource_scaling_closure_20260920.study.json](resource_scaling_closure_20260920.study.json) | full per-point evidence, routing verification, raw SHA-256 manifest |
| [resource_mixed_20260916.json](resource_mixed_20260916.json) / [.audit.json](resource_mixed_20260916.audit.json) | serial stable + elastic frontier, 51 timing runs |
| [resource_mixed_db_20260918.json](resource_mixed_db_20260918.json) / [.audit.json](resource_mixed_db_20260918.audit.json) | double-buffer stable + elastic frontier, revision 2 |
| [resource_hardware_20260916.json](resource_hardware_20260916.json) | CUPTI/block-stamp replay of the 09-16 audited twins |

Frontier today (every repetition must meet): 300 FPS → double-buffer at 32 SM
(serial never); 200 FPS at frame p99 ≤ 10 ms → serial at 32 SM (double-buffer
never; its p99 is 9.8–10.4 ms even unconstrained); 150 FPS / 15 ms → 24 SM;
100 FPS / 25 ms → 16 SM. An optimization claiming better resource efficiency
must move these under the comparer's same-session control.
