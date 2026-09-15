# GPU resource sensitivity (#419)

Standalone diagnostic entry points for full-pipeline MOT17 serial/double-buffer
measurements. Production defaults and source files are unchanged.

## Run

From the repository root, with the project's CUDA/TensorRT environment loaded:

```bash
.venv/bin/python scripts/benchmarks/resource_sensitivity/sweep.py \
  --output /absolute/new/output-directory \
  --preset mamba_whole_graph_m --sequences MOT17-04-SDP \
  --levels 0 8 16 24 32 --repeats 2 --max-frames 300

.venv/bin/python scripts/benchmarks/resource_sensitivity/summarize.py \
  /absolute/new/output-directory --deadline-ms 10
```

The initial characterization uses 300 input frames, including 50 warmup frames;
use `--max-frames 0` for complete sequences. A short prefix is insufficient for
precise rare-tail or cross-scene claims. The optional deadline is a reporting
threshold, not a claim that the workload meets a service level.

Requires an NVIDIA GPU, the current project Python environment, `nvcc`,
`nvidia-smi`, local MOT17 data and the chosen preset's model assets. The blocker
build defaults to `--arch sm_120`; pass the correct architecture on another GPU.
The active implementation is qualified on a single-GPU host and targets logical
CUDA device 0 / physical telemetry GPU 0; run without device visibility remapping. GPU execution is sequential.
Every output directory must be new: failed results are retained, never reused.

## What the proxy controls

`blocker.cu` uses K one-thread blocks, each requesting more than half of an SM's
shared memory. The occupancy API must report at most one such block per SM.
Each block records its actual `%smid`, `%globaltimer` start and end; the reporter
rejects repeated SMs or a pulse without simultaneous K-block residency.
The loop uses `__nanosleep` and performs no repeated global-memory traffic.

This consumes **shared-memory residency**, not an exclusive execution partition.
Other kernels can still share registers, issue slots and remaining shared memory
on those SMs. **Do not use `46 - K` as an available SM count.**

Unbounded same-context persistence would prevent a device-wide synchronize from
returning. The harness therefore launches finite 5 ms pulses on a nonblocking
stream in the primary context, one pulse at a time, from a host thread. There
are restart gaps, device synchronization interference, and CPU launch/collection
overhead. K=0 has no blocker thread. The summary retains pulse-span coverage,
GPU-timestamp simultaneous-residency fraction, mean resident block count and
restart-gap distribution. These are measured proxy properties, not hardware
resource budgets or proof of detector/tracker overlap. Each sequence must have
post-warmup samples; short or failed runs are rejected.

Start occurs on the first completed frame after graph setup, inside warmup.
The blocker then remains active across sequence setup; its finite pulses permit
subsequent capture/synchronization. The runner imposes a process timeout and the
point worker stops and joins its blocker thread in `finally`.

## Timing and comparison contract

The point wrapper delegates to the production `_record_frame_timing`, then reads
its completion timestamp. It installs that wrapper at the evaluator and stages
references; it does not add per-frame GPU synchronization. Latency starts before
decode and ends after output completion, including deferred DB emit/drain, using
the #35/#376 timing surface. Period is the difference between consecutive
completion timestamps **within a sequence**, giving N-1 samples per sequence.
FPS is completed frames divided by the existing production throughput interval;
it is never computed as reciprocal mean latency. Quantiles use NumPy's linear
percentile convention, jitter uses population standard deviation.

Both modes use `--detect-barrier event`, default GPU decode, the same preset,
sequence prefix and warmup. Only double mode adds `--double-buffer`. The wrapper
records whether the side stream actually exists; a silent fallback fails report
validation. `--profile-stages` is not enabled because it disables ordinary DB
eligibility. Tracker-stage latency is explicitly **unobserved (`null`)**; full
frame latency and completion-period quantiles are available on both modes.

Levels are shuffled reproducibly within each repetition (`--seed 419`); mode
order alternates and each DB gain uses its own same-repetition, same-K serial
run. Each row also compares MOT output SHA-256 against its same-mode K=0 run.
Byte equality is an observation, not a general determinism proof. Prefix runs'
full-sequence GT metrics are not valid quality estimates; future #421 quality
contracts must use compatible sequence bounds before frontier promotion.

## Artifacts and reuse

- `input_identity_before.json` / `input_identity_after.json`: preset, referenced
  models, native binaries and selected JPEG/GT input fingerprints; changes fail.
- `execution_sources/`: a snapshot of the diagnostic source files.
- `sweep.json`: exact command per point, randomized schedule, source/library
  hashes, repository HEAD/status, compiler, device metadata and relevant env.
- `<point>/run_manifest.json`: ordinary eval provenance and effective arguments.
- `<point>/resource_point.json`: actual DB route, frame completion timestamps,
  blocker device timestamps/SM IDs, host pulse bounds and wall/monotonic anchor.
- `<point>/_latency_profile_<sequence>.json`: same-run raw frame latencies and
  throughput duration; MOT text outputs and normal eval logs are retained.
- `<point>.telemetry.csv`: 100 ms `nvidia-smi` samples including startup; report
  statistics select the measured intervals using the clock anchor. Unsupported
  fields remain absent, not zero. Telemetry is sampled metadata, not per-frame
  energy accounting.
- `summary.json`: absolute FPS, paired DB gain, latency and period p50/p95/p99,
  jitter, optional deadline misses, pressure qualification and output equality.
- `green_probe.json`: requested **and actual** Green Context resources and
  PyTorch stream-routing probe. It never certifies Saccade-wide partitioning.

For before/after comparison, fix model/data identity, sequence bounds, decode,
pulse waveform, telemetry cadence and quality tolerance. Compare rows and paired
gains with their repetitions; repeat the unconstrained anchor each session.
An actual minimum-SM frontier requires a verified execution partition, and a
shared-vs-reserved tail-latency comparison additionally requires stream/context
ownership verification across PyTorch, TensorRT and native CUDA.

The Green Context probe explicitly shows whether ordinary PyTorch pool streams
escape the selected context; an external-stream toy kernel passing is **not** an
integration certificate. The present harness intentionally reports
`true_partition_validated: false` and `available_sm_count: null`.

See [the characterization report](../../../docs/reference/benchmarks/resource_sensitivity_20260915.md)
for measured results and remaining integration work.

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `green_probe.py` | diagnostic | cli | Probe actual Green Context resources and PyTorch stream-pool escape. |
| `input_identity.py` | diagnostic | cli | Fingerprint resource-sweep presets, local models, native libraries and frames. |
| `plot.py` | diagnostic | cli | Plot observed resource-proxy throughput, paired gain and frame p99. |
| `run_point.py` | diagnostic | cli | Run one full-pipeline MOT17 point with bounded SM residency pressure. |
| `summarize.py` | diagnostic | cli | Validate and summarize same-run throughput, latency, jitter and pressure. |
| `sweep.py` | diagnostic | cli | Reproduce paired serial/double-buffer SM-pressure sweeps with telemetry. |
| `synchronization_probe.py` | diagnostic | cli | Check whether primary-context synchronize waits for an independent blocker. |

<!-- END generated script index -->
