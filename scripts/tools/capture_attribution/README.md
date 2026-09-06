# CUDA capture attribution (#340)

Scope: recover historical failure provenance, then validate an attribution observer.
No capture mode, stream ownership, production source, or capture scheduling is changed.
No failure-rate measurement is implemented. Historical provenance remains partial; see
[the recovery report](../../../docs/research/pipeline/capture_failure_provenance_20260906.md).

The original failure excerpt points at `currentStreamCaptureStatusMayInitCtx`, so the
observer includes `StreamIsCapturing`/`GetCaptureInfo`, not just `StreamWaitEvent`.

## Build and one bounded control

Run from the repository root. Output directories must be new. These commands run only
synthetic mechanism controls; they do not evaluate a sequence.

```bash
uv run python scripts/tools/capture_attribution/build.py --output /absolute/new/build
uv run python scripts/tools/capture_attribution/run.py \
  --observer /absolute/new/build/observer.so --output /absolute/new/trace \
  -- scripts/tools/capture_attribution/control.py blocking-runtime
uv run python scripts/tools/capture_attribution/analyze.py /absolute/new/trace
```

For the fixed six-case qualification (each case once, separate processes):

```bash
uv run python scripts/tools/capture_attribution/qualify.py \
  --observer /absolute/new/build/observer.so --output /absolute/new/qualification
```

The control cases are `blocking-runtime`, `nonblocking-runtime`, `blocking-driver`,
`nonblocking-driver`, `blocking-joined`, and `python`. Each belongs in a fresh process and output directory.
The driver cases obtain `cuStreamBeginCapture` through `cuGetProcAddress_v2`, exercising
a function-pointer path that a symbol-only LD_PRELOAD shim may miss. `python` covers
`make_graphed_callables` and the repo NMS/GMC wrapper with trivial tensor operations.
This is instrumentation validation, not a sample of production failures.
`blocking-joined` starts capture on a non-blocking origin and joins a blocking side
stream via an event before querying legacy. It checks the observer's ability to retain
capture participation without another BeginCapture call.

## What is recorded

- CUPTI subscribes before torch import to both runtime and driver API domains.
  Typed parameter decoders are generated from the installed CUDA metadata and callback
  IDs; `build.json` records header hashes, covered and unparsed variants, and build command.
- `cuda.jsonl`: monotonic timestamp, PID/native TID, context, callback/correlation IDs,
  enter/exit, stream, flags and their source, capture mode/status/ID (when queried by
  the application), event record/wait with event flags and event destruction, numeric return,
  Python site ID, native stack addresses on begin/error. Nonzero returns from other APIs
  are retained too. Runtime and driver callbacks may represent the same operation;
  do not add them as independent captures or failures.
- Stream flags are learned from successful application stream-creation and GetFlags
  calls. No CUDA API is called inside a CUPTI callback. Unknown flags remain `-1` and
  fail the trace structure check for captures. The observer never clears a CUDA error.
- `python.jsonl`: CUDAGraph begin/end spans, native TID, current stream/device,
  unmodified capture mode, site label and full Python source stack. Wrapping
  `CUDAGraph.capture_begin/end` covers both repo wrappers and `make_graphed_callables`.
  Exceptions preserve their original propagation; background uncaught exceptions are
  logged then passed to the existing thread hook. `site_id=0` means unclassified;
  it is not evidence that the capture belongs to an external library.
- `tail.log`: one flushed line per teardown stage, from how the workload finished
  through quiescence, `attribution_stop`, mapped-file hashing and the final manifest
  write. The manifest is only rewritten at the very end, so without this a teardown
  that dies part-way is indistinguishable from any other missing final manifest. It
  locates the last completed step; it is a diagnostic aid and neither the manifest nor
  the structure check defers to it.
- `stdout.log`, `stderr.log`, final `/proc/self/maps`, git HEAD/status/diff, source
  file hashes, package versions, selected CUDA/runtime environment, current GPU/driver,
  harness/observer/target hashes, and final mapped-file hashes. `--asset PATH` is repeatable
  for checkpoint/engine/config inputs. Unlisted assets and inherited environment outside
  the selected keys are not attested. Never substitute this current manifest for the
  original failure environment. Background native threads and unloaded libraries are
  not exhaustively inventoried by the final process mapping. `manifest.json` and
  `tail.log` are still open when artifacts are hashed, so they are excluded by name in
  `artifacts_sha256_excluded` rather than silently omitted.

Before `attribution_stop`, teardown shuts down the auxiliary workers this process owns
(torch's inductor compile pool, tqdm's monitor) and waits for the remaining threads,
bounded by `--quiesce-timeout` (default 60s, which has to clear torch's own
`quiesce_async_compile_time / 2` polling interval). This is quiescence, not an excuse:
the shutdown check still requires no live thread, so a worker that outlasts the bound
is reported and fails the structure check as before. Shutdown errors are recorded in
`quiesce.errors` and never raised, since a teardown convenience must not destroy the
trace it is serving.

## Interpretation and limits

`analyze.py` checks artifact hashes, event numbering, API pairs, begin/end pairing,
known capture metadata, source drift and observer shutdown. It reports per-domain
capture intervals and same-context overlap with errors 900/901/906. An overlap is an
attribution candidate, not a causal proof. A clean trace structure is not a claim of
complete site attribution: inspect `unclassified_captures` and resolve native addresses
against `maps.txt` (and matching binaries). No observed capture is not a passing check.
Missing metadata, missing exit, a failed observer, live Python workers at shutdown, or
unparsed capture variants are evidence gaps, never negative evidence.

The harness adds host callback/stack/logging overhead and may change race timing.
Its FPS and observed error count cannot estimate production performance or incidence.
It does not trace other processes. Starting before torch import and validating runtime,
driver-function-pointer and Python paths improves coverage but does not prove that all
private implementation paths of every library are observable. Fresh-process load history
and a failure-time trace are still required to attribute the production stream.
Event record/wait edges and observed stream statuses are retained separately. A stream
can join a capture without calling BeginCapture; enumerating origin streams alone does
not enumerate all participating streams. External event flags must not be interpreted
as an ordinary capture join. Missing event history remains unknown.

The launcher can observe one Python entry point using the same `-- script.py args...`
form. On 2026-09-06 one bounded production-path topology trace was performed with it
(`production-01` under the local state root; plan and outcome in
[the recovery report](../../../docs/research/pipeline/capture_failure_provenance_20260906.md)):
`mamba_whole_graph_m`, MOT17-02-SDP, GPU decode, double-buffer, 64 frames, one process,
no incidence loop. Its structure check did not pass: three Python workers were live at
stop, and the final manifest was never written because the process ended somewhere
after `harness_stopped`. Which teardown step it stopped at was not recoverable from
what that run retained, which is why `tail.log` exists. Under this README's rules
those are evidence gaps, so the trace must not be cited as a clean structure result.
It observed four classified single-stream captures with no capture errors, no
in-capture event joins and no blocking participant; that does not locate the synthetic
blocking-join mechanism in production, and it is not a failure reconstruction. The
user's stop boundary remains: no failure-rate runs or new capture-semantics changes
until provenance and attribution harness work are complete.

References: [CUPTI callbacks](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api)
(CUDA calls inside callbacks are generally unsupported),
[CUDA stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
(`cudaStreamIsCapturing` on legacy may return Implicit without invalidating capture).

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `analyze.py` | diagnostic | cli | Validate trace structure and attribute observed errors without exclusion claims. |
| `build.py` | diagnostic | cli | Build the CUPTI observer and typed decoders from installed CUDA headers. |
| `control.py` | diagnostic | - | One bounded mechanism control. Run each case in its own attributed process. |
| `qualify.py` | diagnostic | cli | Fixed six-case observer qualification. This never invokes production evaluation. |
| `recover_failure.py` | diagnostic | cli | Extract surviving primary tool evidence for #340; never rerun the workload. |
| `run.py` | diagnostic | cli | Run one Python entry point with diagnostic-only CUPTI/Python attribution. |

<!-- END generated script index -->
