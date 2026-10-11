<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
<!-- doc-module: cross -->

# #549 S2-2 bounded implementation candidate

Owner authorization was given in the working session: one bounded S2-2 implementation Draft PR,
based on `main@f63c9da41f2aa034a76311ab14a62ab69ec8c540`; no Ready transition or merge.
Base [CI 38122766092](https://github.com/raylei50653/saccade/actions/runs/38122766092)
completed with seven successful jobs and the exact base headSha.
The governing contracts are [ADR 028 D5](../decisions/028-model-bundle-runtime-separation.md),
[model-bundle contract](../architecture/model_bundle_contract_549.md),
[CC-536-01-02](../architecture/ship_export_contracts_536.md#cc-536-01-02), and #550.

This candidate delivers engine/head buffer loading and compatibility gates. **S2-2 is incomplete**:
N03 operator byte equality remains OPEN. Aggregate `load_verification.byte_scope` stays
`hashed_before_load`. The existing S2-1 checkpoint and its historical evidence retain their scope.
The PR body records the final base/head, CI and independent read-only review.

## Implementation and contract behavior

- Engine and head are each opened once in Gate B, read into an owned buffer, checked for a
  regular file, manifest size when available, complete read and SHA-256, then loaded from
  that exact buffer. TorchScript uses a seekable input stream over the buffer without a
  device map; TensorRT uses its pointer/size deserializer. Neither loader reopens these paths.
  Legacy mode also uses these buffers and retains its supplied expected SHA bindings.
- Operator keeps its existing pre-load hash and pathname `dlopen`, including the existing
  auditor policy. No sealed-memfd production implementation is introduced.
- Manifest Gate A compares the native binary's platform, linked TensorRT library version including
  build (its no-device version APIs), CUDA runtime header version and LibTorch package build version including its CUDA
  qualifier against **one whole target**. Missing build evidence or no matching target is a
  catchable `preflight: model bundle compatibility` refusal, exit 2 before CUDA. Versions are
  build coordinates. TRT SDK headers (`10.16.0.72` here) do not substitute for the actual
  linked runtime build (`10.16.1.11` here). This does not certify arbitrary dynamically
  substituted LibTorch/CUDA libraries; existing auditor enforcement remains in place.
- Gate A passes only version-matched targets and the ordered I/O contract through an immutable
  `DetectorPlan::manifest_contract`, alongside the existing resolved bindings. Gate B reads
  neither the manifest nor caller options again. GPU SM must belong to a version-matched
  target; an unsupported or unreadable device is refused before engine deserialization.
- Gate B checks TRT I/O direction, ordinal/name order, dtype, CUDA location, linear storage and
  concrete shape against the manifest, before execution. The existing float32 native path
  rejects other tensor types; no force flag or fallback is added. Head layout must also match
  the supported plan and its existing device, graph, operator and runtime-readback checks.
- Gate A identity remains immutable. Gate B alone writes aggregate `load_verification`:
  catchable loader/compatibility failures retain identity and write `failed` with null byte scope;
  uncatchable interruption can leave `running` with null load verification. Journal v3/report v5
  and historical reader semantics are retained. Successful report/probe load diagnostics add
  `artifact_byte_scopes` (head/engine `loaded_buffer`, operator `hashed_before_load`) and GPU SM;
  these per-object diagnostics never promote the aggregate scope.

## Coverage and evidence

Evidence under `results/549_s2_2/` is gitignored, local-only. CI runs and their exact heads
are recorded in the Draft PR; local results do not substitute for remote checks.
The tracked [evidence summary](model_bundle_s2_2_549.evidence.json) binds the frozen source,
binaries and raw local artifacts by SHA-256 and preserves the qualification boundaries.

| Contract control | Executable control and observation | Evidence boundary |
|:--|:--|:--|
| MB-41 engine/head | CPU immutable-buffer/stream control replaces a path after hashing; GPU control actually loads original head/engine buffers after replacing paths. Pre-read replacement, wrong size, missing/nonregular files and hash mismatch refuse | Buffer helper: 17 checks; native Gate B: 150 checks; both pass |
| MB-41 N03 | Existing pathname loader remains; sealed-memfd experiment fails provenance requirement | OPEN |
| MB-43 | Whole-target positive; TRT/LibTorch/CUDA wrong versions, unavailable build tuple and cross-target mixing negative; native CLI exit 2, no cuInit, failed journal and null load verification | Compatibility helper: 26 checks; native CLI suite passes; CPU helper also runs in CI |
| MB-44 | Supported SM positive; unsupported/version-unmatched target SM negative; native CLI refusal precedes invalid stand-in dlopen/deserialization and retains Gate A identity | CPU helper, native Gate B and both native CLI variants pass |
| MB-45, shape/order | Exact comparison helper; real engine vs hash-valid manifest dtype/shape/name-order mismatch; catchable Gate B refusal and failed journal | CPU helper, native Gate B and both native CLI variants pass |
| MB-60..62 | Existing schema/semantic matrix replays R-08 class channels, regression channels and level H/W mismatches | CPU ctest 12/12; manifest test 6,501 checks; native preflight controls pass |
| MB-58 caught | Gate B loader and compatibility failures: exit 2, identity unchanged, run/load failed | Native CLI four-suite run: 72 passed, no skips |
| MB-58 uncaught | Existing writer SIGKILL and actual during-load SIGKILL; auditor foreign-N03 `_exit(127)` with identity retained and journal running/null load | Native CLI completion suite and final-binary auditor negative pass |
| Serial/double buffer/graph/MOT/seven sequences | Shipping trace/no-trace, measurement serial/double-buffer and manifest-mode replays each process seven sequences / 5,316 frames; seven negative controls differ in their expected sections | Local diagnostics complete; qualified V5 remains UNRESOLVED, controls caught=false |
| Report/journal readers | Existing v3/v5, historical null/strict-reader and completion suites | CPU reader/schema/auditor selection: 350 passed; native CLI: 72 passed |
| Runtime identity | Fresh complete scratch capture; only implementation changes from 297 to 303 files; independent stacked review, outgoing append-only archive and attested verification accompany atomic delivery | Fresh probe computed; equivalence stays unproven; exact review recorded in PR |

Coordinate-only [Draft review #581](https://github.com/raylei50653/saccade/pull/581) preserves
review head `79ffd5278c8226d104d67beb243a6f3111c78afe`. Independent scratch and exact promotion
reviews found no findings. The reviewed commit is an ancestor in the principal delivery;
canonical and outgoing append-only archive travel with the implementation atomically.
The attested check passes; its normal unrecomputed host/input/probe warnings are retained.
The independent scratch review separately recomputes all five full axes, current inputs
and native build witnesses. Probe equality establishes no equivalence.

The complete Release C++ build and native/GPU ctest run pass (19/19). Ruff check/format,
mypy (604 source files), lockfile, document links and strict document structure checks pass.
LibTorch-origin negative CMake controls refuse a same-version foreign `Torch_DIR` and a
same-byte foreign literal `c10` library with exit 1. Initial failed generations, including
the SDK/runtime version mismatch and a CLI test assertion that wrongly treated the manifest
itself as trusted, are retained beside corrected runs.

The GPU evidence is bound to source
`9fe51e6606e5a72e1cd52acbdf72913be7aebef8`, clean tree
`178df336494d5e0dbb418c1742e4037a84a2fda8`, and the following Release binaries:

| Binary | SHA-256 |
|:--|:--|
| `saccade_track` | `555f5134ce3d9b9c5670f60b98463a96e0ca769230b8e1e1e887f7c6f195c1ab` |
| `saccade_track_measurement` | `611739b2b9f2357abe265a6f5067984ed20d0b91e2d1e06d8246a9b3c25461aa` |

The replay checks source, binary and frozen input hashes before and after execution; all
remain unchanged. Its raw evidence directory is `results/549_s2_2/gpu_9fe51e6606e5_r2/`,
whose `SHA256SUMS` digest is
`d9b34b9b1866f9c9195b5286739c5b87404bbec478ec18f743dc086dfcb9b276`.
Raw positive detector/MOT/graph section comparisons are equal where observed; no-trace
detector comparison is `NOT_RUN`. Manifest-mode MOT files equal the legacy trace files and
retain journal v3/report v5, `checksum_matched`, six matched member bindings, an unchecked
manifest binding, and verified aggregate `hashed_before_load`.

The after-hash replacement GPU control calls the actual head/engine deserializers on the
retained buffers; a timed whole-CLI replacement race is **UNVERIFIED**. A dedicated TRT I/O
ordinal mutation is covered by CPU/native tests; the separate CLI negative swaps names to
exercise ordered bindings. A separate CLI ordinal mutation and an actual `abort` interruption
are **UNVERIFIED**; existing SIGKILL and actual auditor exit 127 controls cover the retained
uncaught journal semantics. Alternate physical SM hardware and a newly generated engine
with alternate dtype are **UNVERIFIED**; rejection controls change the expected contract
against the same current device and float32 engine.

## N03 proposal evidence and residual risk

The candidate-only experiment uses unchanged frozen N03
`aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4`
(3,517,632 bytes) and an untouched existing installed auditor
`1ee968abb8f431c770723a3a1a9263d37ab58c92a9915b18b98506f91b95d74b`.
Exact commands, raw stderr, RUNPATH/NEEDED and dependency hashes are retained in
`results/549_s2_2/operator/` (`n03_audit.md`, `operator_probe.summary.json`,
`mapped_dependency_hashes.json`, `sha256.json`). No package is installed or re-pinned by it.

| Probe under the same auditor | Exit | Result |
|:--|--:|:--|
| N03 at fixed prefix path | 0 | Native dlopen succeeds |
| Same N03 bytes from foreign repository path | 127 | `operator library mapped from another path` |
| Same bytes copied to sealed memfd, then `/proc/self/fd/N` | 0 | All four seals present; realpath unresolved; N03 classification is bypassed |

Seals do not repair the auditor provenance contract. This concrete candidate is unsuitable;
production implementation requires a new proposal with auditor, N03 RUNPATH exception,
dependency-load and actual GPU evidence, then independent owner acceptance. Candidate memfd
GPU execution is **UNVERIFIED**. Existing auditor/launcher CPU regression controls: 14 passed.
Observed successful probe mappings included 21 pinned vendor objects per case; this is not a
complete system/driver closure claim. The original N03 RUNPATH exception is retained.

Independent final-binary native controls in `operator/final_binary_r2/` use direct execution,
the unchanged auditor via `LD_AUDIT`, and its exact pinned vendor `LD_LIBRARY_PATH`.
The manifest positive exits 0, completes one real GPU frame, verifies CUDA head parameters,
three native operator calls and detector/NMS/tracker graph capture/replay, and initializes
27 vendor objects whose hashes match the pinned copies. The foreign-N03 negative exits 127
with `operator library mapped from another path`; its immutable `checksum_matched` identity
remains in a `running` journal with null load verification. These source-native controls do
not attest an installed package.

The explicit dynamic-loader invocation `ld-linux ... --audit ... entrypoint --model-bundle`
also exposes an existing S2-1 runtime-root derivation limitation: `/proc/self/exe` resolves
the loader, producing `/usr/share/saccade` and a catchable Gate A exit 2. The failed attempt
on implementation head `9a60f52757a5b77d683a06443b5b6ba54d3aa07f` is retained under
`operator/final_binary/positive_manifest/`. Final-binary direct execution does not reverify
that invocation. Launcher/root policy repair
and installed launcher acceptance remain outside this candidate.

N03 can still change after rehash and before/while pathname dlopen consumes it. The auditor
checks path provenance, not equality with the prehash bytes. Passing its positive control
cannot establish `loaded_buffer`. Sequence contents/listing TOCTOU remains outside this slice.

## Remaining boundaries

- V5 requires driver `616.92`; current host `617.42` leaves qualification **UNRESOLVED** and
  qualification negative controls `caught=false`. Raw section equality/differences and
  manifest MOT text equality are local diagnostics, not qualified EXACT or equivalence.
- Installed package is **UNVERIFIED** for this source change. No package re-pin, production
  allowlist approval, launcher policy change, model byte change, S2-3 split, repository creation,
  public release or benchmark claim is included. The production allowlist remains empty.
- #547 remains `distribution.status=local-only`, `owner_confirmation=null`.
- The principal PR stays Draft/unmerged. Final-head CI and independent review do not authorize
  merge. Separate owner authorization must name the final head; #549/#536/#550 checkpoint
  writeback follows an authorized merge and successful exact-head main CI.
