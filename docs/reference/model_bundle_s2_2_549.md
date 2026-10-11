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

| Contract control | Executable control and observation | Evidence boundary |
|:--|:--|:--|
| MB-41 engine/head | CPU immutable-buffer/stream control replaces a path after hashing; GPU control actually loads original head/engine buffers after replacing paths. Pre-read replacement, wrong size, missing/nonregular files and hash mismatch refuse | Local CPU/GPU results pending |
| MB-41 N03 | Existing pathname loader remains; sealed-memfd experiment fails provenance requirement | OPEN |
| MB-43 | Whole-target positive; TRT/LibTorch/CUDA wrong versions, unavailable build tuple and cross-target mixing negative; native CLI exit 2, no cuInit, failed journal and null load verification | Local CPU/native CLI pending; CPU helper runs in CI |
| MB-44 | Supported SM positive; unsupported/version-unmatched target SM negative; native CLI refusal precedes invalid stand-in dlopen/deserialization and retains Gate A identity | Local CPU/GPU/native CLI pending |
| MB-45, shape/order | Exact comparison helper; real engine vs hash-valid manifest dtype/shape/name-order mismatch; catchable Gate B refusal and failed journal | Local CPU/GPU/native CLI pending |
| MB-60..62 | Existing schema/semantic matrix replays R-08 class channels, regression channels and level H/W mismatches | CPU ctest pending; unchanged contract rules |
| MB-58 caught | Gate B loader and compatibility failures: exit 2, identity unchanged, run/load failed | Local native CLI pending |
| MB-58 uncaught | Existing writer SIGKILL and actual during-load SIGKILL; auditor foreign-N03 `_exit(127)` with identity retained and journal running/null load | Local native controls pending |
| Serial/double buffer/graph/MOT/seven sequences | Fresh native tests and V5 replays; each result has its own source/binary/input coordinates | Pending; qualified V5 remains UNRESOLVED on driver 617.42 |
| Report/journal readers | Existing v3/v5, historical null/strict-reader and completion suites | Pending |
| Runtime identity | Fresh complete scratch capture, independent stacked review, outgoing append-only archive, atomic delivery and attested verification | Pending; equivalence stays unproven |

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
