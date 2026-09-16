# Dynamic execution-resource routing (#419)

This synthetic experiment measures an actual change of workload destination
between pre-created Green Context lanes. It follows
[time-budget admission](../../../docs/reference/benchmarks/resource_time_budget_20260915.md).

## Frozen experiment

Three disjoint 8-SM pools provide A (permanent stable floor), B (elastic), and
C (policy-controlled). The 24-SM policy envelope leaves 22 of this 46-SM device's
SMs outside these workloads. Within that envelope C can be reserved, borrowed,
or deliberately idle. Matched sharing uses an independently created 24-SM pool;
full-device sharing is a separate 46-SM operational control. A/B physical roles
swap; C is fixed. Shared-route swaps are repeat labels, not physical swaps.

| Policy | A | B | C while stable idle | C during stable service |
|---|---|---|---|---|
| fixed_reservation | stable | elastic | reserved idle | stable |
| static_elastic | stable | elastic | elastic | elastic |
| borrow | stable | elastic | elastic | drain then stable on every arrival |
| headroom | stable | elastic | intentionally idle | stable only on high demand |
| dynamic | stable | elastic | elastic | drain then stable only on high demand |
| shared_24 | shared stable priority | shared elastic priority | unused handle | same 24-SM pool |
| full_shared | shared stable priority | shared elastic priority | unused handle | same full-device pool |

Contexts are never resized. C's software owner changes only after all prior
C events report completion. Future stable kernels then execute on C; after all
stable units finish, future elastic kernels may return there. No running kernel
is preempted or migrated. A never admits elastic work on partitioned routes.
This is lane ownership/routing, not hardware repartitioning latency.

- Sixteen stable arrivals at absolute host targets `origin + (j+1)*8 ms`.
  Demand repeats `[1, 1, 8, 8]` independent units. Each unit is 256 blocks ×
  256 threads × 4096 dependent FP32 FMA iterations. Same 72 units for every trial.
- Stable jobs remain FIFO even if late. Each eligible lane has at most one
  stable unit in flight. High demand is the known eight-unit job size, not a
  fitted predictor. No future arrival is used to initiate early reclamation.
- Saturated elastic work uses identical independent 256-block, 256-thread
  kernels of 2048 or 8192 iterations, compared separately. Per-lane admission
  windows are 1 and 4. Host polling retires events; no speculative cancellation.
- C admission freezes when the head stable job requests it. A serves work
  while C drains. A low-demand job may complete without using C; such a request
  has drain evidence but no first-C-launch sample. Borrow waits for C drain
  before releasing ownership and selecting the next job; stable completion is
  timed independently. Fixed reservation/headroom reclamation is an idle-lane
  control, not proof of transferring an elastic owner.
- All stable streams and C use highest device priority; B uses priority zero.
  C has a single stream with no competing owners. Shared controls use highest
  stable priority and zero elastic priority. These are explicit policies.
- Frozen descriptive service criterion: stable p99 ≤4 ms and observed fraction
  over 4 ms ≤1%, checked per repetition/role condition. Not a real-time guarantee.
- Default retained sweep: 3 shuffled repetitions × 56 conditions × 10 trials,
  one excluded warmup per condition; 26,880 stable arrivals. Seed 430. Pilot
  uses one repetition/sample without warmup, is excluded, and qualifies the
  harness. No parameter is selected by held-out production evidence.

## Measurements and limits

Stable response is **scheduled arrival to host-observed completion**, including
host lateness, queueing, reclaim interference and launch overhead. Request-to-
drain and request-to-first-C-enqueue are separate host measurements. GPU
last-elastic-block finish to first stable-C-block start is a third measurement
with its own clock domain; it also includes idle gaps. Host/GPU clocks are never
subtracted. Request-to-first-C-enqueue does not claim exact execution-start time.

Elastic throughput counts host-observed completed work before the last stable
job releases its lane, normalized to 2048-iteration units per second of that
same host horizon. Work admitted before the cutoff is drained afterward and
reported separately. Thus work and horizon differ between policies; this is a
saturated-throughput comparison, not equal-job-count completion time. Boundary
completion observation is conservative. First arrival is timed from origin,
not a start handshake; replay proves actual GPU ordering for transferred lanes.

`observed_sm_coverage` unions block-leader start/end intervals per physical SM,
over the GPU span from earliest block start to latest finish, including final
drain, divided by **all device SMs**. It measures observed block presence, not
occupancy, instruction utilization, or a hardware SM-active counter. It is not
substituted for total SM utilization. 100-ms NVIDIA-SMI telemetry is retained
as device-wide context and may include display activity; it cannot attribute
utilization to a condition. Cache, memory bandwidth, clocks and power remain
shared. Only one compute-heavy stable workload is tested.

Raw NPZs retain every kernel's host enqueue/completion, logical destination,
stable job identity, every block's GPU timestamps/SM ID, and block-leader output.
Replay checks condition/sample coverage, checksums, pool sets before/after,
outputs, fixed arrivals, work conservation, queue bounds, freeze/drain/release
ordering, and GPU owner non-overlap, then re-derives saved metrics. This is
synthetic routing evidence; it does not validate TensorRT/graph migration or
full-pipeline service. Reported quantiles are descriptive at finite sample size.

## Reproduction

```bash
uv run python scripts/benchmarks/resource_routing/probe.py \
  --output /absolute/new/routing --samples 10 --repeats 3 --warmups 1
uv run python scripts/benchmarks/resource_routing/report.py \
  /absolute/new/routing --json-output /absolute/new/routing.json \
  --markdown-output /absolute/new/routing.md
uv run python scripts/benchmarks/resource_routing/audit.py \
  /absolute/new/routing /absolute/new/routing.audit.json
uv run python scripts/benchmarks/resource_routing/plot.py \
  /absolute/new/routing.json /absolute/new/routing.svg
```

The output directory must be new. Source snapshots, source hashes, Git HEAD,
compiler/device identity, telemetry and a complete checksum manifest accompany
raw evidence. See the [research record](../../../docs/reference/benchmarks/resource_routing_20260916.md)
for the replayed results and limitations. The independent `audit.py` does not
import producer/reporter calculations; it checks transfer ordering and service
counts after the full integrity replay.

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `audit.py` | diagnostic | cli | Independently audit raw C-lane transfers and stable service target counts. |
| `plot.py` | diagnostic | cli | Plot measured routing service, transition and throughput comparisons. |
| `probe.py` | diagnostic | cli | Measure future-work routing and lane reclamation on reserved SM pools. |
| `report.py` | diagnostic | cli | Replay routing ownership, output, service and observed SM activity evidence. |

<!-- END generated script index -->
