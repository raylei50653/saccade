<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 MB-41: aggregate `loaded_buffer` (condition check and evidence)

**Status: Draft implementation candidate. Not merged.** The status is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2).
This page records the owner's conditions, the change, and the evidence for each condition.

## Authorization

The owner gave a conditional approval in session on 2026-10-12. It approves, in principle,
closing MB-41 and promoting the aggregate `load_verification.byte_scope` to `loaded_buffer`,
but only after three conditions are confirmed:

1. Operator, head and engine cover every load file that [contract 4.1](../architecture/model_bundle_contract_549.md#load-verification) requires.
2. Each file's loaded bytes are the bytes whose hash was verified, and there is no unverified
   alternative load path.
3. The corresponding tests, negative controls and journal consistency checks all pass.

The approval also fixes the scope and the fallback:

- **Scope**: one small PR that changes only the aggregate record and the MB-41 status.
- **Unchanged**: qualified V5, package admission and `distribution.status` keep their states
  unless each has its own independent evidence.
- **If a condition fails**: MB-41 stays OPEN, and the page lists the gaps, the evidence and the
  minimal follow-up.
- **Merge**: needs a final owner authorization.

## Change

The PR has two commits:

- `69476d4b`: the implementation and its tests.
- `1d3c76af`: the runtime coordinate republication.

| Where | What |
|:--|:--|
| [run_completion.hpp](../../shipping/include/saccade_shipping/run_completion.hpp) / [.cpp](../../shipping/src/run_completion.cpp) | `LoadByteScope` and `aggregate_load_byte_scope(manifest_mode, op, head, engine)`. The result is `LoadedBuffer` only in manifest mode, and only when all three per-object scopes are `loaded_buffer`. `record_load_verification(verified, scope)` throws `std::logic_error` for `LoadedBuffer` in a legacy-mode run or for a failed load. |
| [track_driver.hpp](../../shipping/tools/track_driver.hpp) `load_runtime` | After the runtime is built, records the aggregate of `load_report()`'s three per-object scopes. |
| Comments | `run_completion.hpp`, `detector_host.hpp` and `verified_model_bytes.hpp` no longer describe path-loaded objects. |

**Unchanged:**

- Journal v3 / report v5 keep their format strings. Contract 4.1 defined `loaded_buffer` as a
  value of this field from S1 on; until now no binary wrote it.
- The parity reader. It reads legacy-mode runs only, and those still write `hashed_before_load`.
- Gate A identity and the failure semantics.

**Legacy mode** keeps `hashed_before_load`, because contract 4.1 says "legacy 模式一律
`hashed_before_load`". Its loads also come from the checked bytes (same `DetectorHost`), so
promoting legacy runs as well would be a contract change, which this PR does not make.

## Condition 1: coverage of the load files

In the shipping tree, `git grep` for `dlopen(`, `torch::jit::load(` and the `TRTEngine`
constructors finds three load sites:

| Site | Object |
|:--|:--|
| `detector_host.cpp` `torch::jit::load(head_stream)` | head |
| `detector_host.cpp` `TRTEngine(engine_bytes.data(), size)` | engine |
| `operator_loader.cpp` `dlopen("/proc/self/fd/<M>")` | operator |

Other facts that complete the picture:

- Each runtime (`SerialRuntime`, `DoubleBufferRuntime`) builds exactly one `DetectorHost`.
- The path-taking `TRTEngine(path)` and `torch::jit::load(path)` calls are under `src/perception/`.
  No shipping code calls them.
- Contract 4.1 names exactly these three files. Config, lineage and attestation are Gate A
  inputs, not Gate B loads.

**Met.**

## Condition 2: loaded bytes are the hashed bytes, with no alternative path

**Head and engine.** `VerifiedModelBytes::read` opens each file once, reads it whole into a
`const` buffer and hashes that buffer. The buffer must be a regular file, and the read must
match the size it expects.

- TorchScript loads from a seekable stream over that buffer, and TensorRT deserializes from its
  pointer/size.
- Both loads happen in the `DetectorHost` constructor while the buffers are alive. Neither path
  is reopened.

**Operator.** The flow is:

1. The plan's digest must be the compiled N03 pin.
2. The file is opened once with `openat2` and hashed.
3. Its bytes are written into a memfd that carries four seals.
4. The memfd is read back, and both its hash and its bytes must equal the verified ones.
5. The loader runs `dlopen("/proc/self/fd/<M>")`, then checks that the returned object's mapping
   is `M`, by device and inode.

Any failure after `dlopen` poisons the registry. There is no pathname fallback. A registry reuse
returns the handle of an earlier verified load of the same SHA-256 and source.

**Alternative paths.** The aggregate is computed only from the per-object scopes. The binary
sets those after each verified load (`detector_host.cpp`), so no other path can produce a
`loaded_buffer` aggregate.

- The measurement build's mutations act on outputs after the load.
- The auditor still accepts the operator by path at the fixed location ("path mode"). That
  serves older entrypoints, such as the pinned `fbab8b79`, which do not run this code. This
  binary never loads the operator by path.
- Dependencies of N03 (`DT_NEEDED`) are vendor libraries under the auditor. They are not
  contract 4.1 load files (known limit, unchanged).

**Evidence (re-run on this head):**

| Object | Controls |
|:--|:--|
| head/engine (CPU) | `saccade_shipping_verified_model_bytes_test`: replacing the path after the hash leaves the loaded bytes unchanged |
| head/engine (GPU) | `saccade_shipping_model_bundle_gate_b_test`: after a post-hash replacement with zero bytes, LibTorch and TensorRT still deserialize the original buffers |
| operator | `saccade_shipping_operator_loader_test`: rename-replace and in-place writes after prepare and after load |
| operator, real loader and auditor | `test_shipping_n03_auditor.py` (37 cases, in the CLI suites): races, forgeries, aliases, foreign paths |

**Met.**

## Condition 3: tests, negative controls, journal consistency

All runs are local; CI does not run GPU or real-binary CLI tests. Raw logs are under
`results/549_mb41/` (gitignored, local-only), with their hashes below.

| Check | Result |
|:--|:--|
| Shipping ctest, 22 tests (CPU and GPU), under the gpu0 lease | 22/22 passed. run completion 329 checks, manifest preflight 6515 checks, 0 failures |
| New writer rules (run completion ctest) | Aggregate truth table: manifest × legacy, each object `hashed_before_load` or empty, wrong case. Legacy `loaded_buffer` refused; failed + `loaded_buffer` refused; the journal stays null after a refusal |
| New manifest record (manifest preflight ctest) | A manifest-mode Gate A followed by `verified/loaded_buffer` or `verified/hashed_before_load`. Identity unchanged, and a failed load with `loaded_buffer` is refused |
| CLI suites: the 12 files of the #584 chain, under the gpu0 lease | 656 passed, 0 skipped. The real bundle in manifest mode completes with `verified/loaded_buffer` and per-object `loaded_buffer` ×3, in journal and report. The legacy run on the same binary stays `verified/hashed_before_load` |
| Journal readers | `test_native_track_parity.py` (in the CLI suites) is unchanged: a complete legacy run must be `verified/hashed_before_load`, and `loaded_buffer` in a legacy record is a problem |
| CPU mutants (five) | All caught: legacy refusal removed, failed-load refusal removed, aggregate ignores the mode, aggregate ignores the engine, record ignores the scope |
| End-to-end mutant: the entrypoint always passes `hashed_before_load` | Caught by the real-bundle CLI test for both entrypoints (2 failed). After restoring, 2 passed |
| Runtime coordinate | Fresh `--run-probe` capture on a Release build. Only the implementation axis changes, `942e0e3c…` → `78741f99…`, with 312 files on both sides. `check_runtime_identity_staleness --mode attested` exits 0 |

**Met.**

## Outcome and what does not change

**MB-41 closes at source level when this PR merges.** In manifest mode the aggregate
`load_verification` becomes `verified/loaded_buffer`.

The following do not change:

- **Legacy mode** stays `hashed_before_load` (contract 4.1).
- **Qualified V5** stays **UNRESOLVED**: the host driver is `617.42`, and V5 requires `616.92`.
- **Installed package** stays **UNVERIFIED**. The pinned entrypoint `fbab8b79…` predates both
  this change and the N03 pin.
- **`distribution.status`** stays `local-only`.
- **`equivalence`** stays `unproven`.
- **The S2-2 label (PARTIAL).** All four S2-2 matrix rows (MB-41, MB-43..45) are now
  source-level implemented. Whether the label changes is left to the owner's merge decision;
  this PR does not change it.

## Follow-ups (not in this PR)

- **Legacy aggregate.** Promote legacy runs only if the owner amends contract 4.1.
- **Package re-pin.** It needs the N03 pin, and the bundle log tools need to understand memfd
  loads (resolved config §17.9).
- **V5.** Run on the qualified driver, or under an owner-approved versioned protocol change.

## Raw evidence

| File (`results/549_mb41/`) | SHA-256 |
|:--|:--|
| `ctest_shipping.log` | `c957ee162bf3e38a5e03ecc538cc901528f517e507d9f33af1efad8b20463859` |
| `mutation_cpu.log` | `ff72d29891f108db911c1deae183f7ed1c28fdbe6ee45ae6fe2670759875ee71` |
| `mutant_legacy_refusal_removed.log` | `7aac37b9fc58a6041b72e750b76884992517c6b3491a2949ba391481c00fd418` |
| `mutant_failed_refusal_removed.log` | `e7588e900b811dc35fac4a432f7f96678bc3e3e721327fe70dfd67043513cb41` |
| `mutant_aggregate_ignores_mode.log` | `caab7b908d42e64570bd5d6ad83b527dd83a1bedba6db0dcc8af281db9844dee` |
| `mutant_aggregate_ignores_engine.log` | `caab7b908d42e64570bd5d6ad83b527dd83a1bedba6db0dcc8af281db9844dee` |
| `mutant_record_ignores_scope.log` | `693359d27509a730bb868ec2fca70df896a467f9f29a68eaf411a972b5d1b3be` |
| `mutation_cli.log` | `a12328b05c0ce1e8b5174579ec788c454f3f2d28f882b4197403bebb6135e76a` |
| `cli_suites_69476d4b.log` | `8b8bf2a6552f66ea3d1413152c45723f94783d3ac33cd3c284dbbb3182f6d718` |
| `coordinate_capture.log` | `40036e54ed1ac4b5a938dfdb8e40c99608da7aaa95425d2bfa43c4c853e25b67` |
| `runtime_identity.candidate.json` | `5b0c6721185a9e7a5a9f1ebc9da9bec0cd0de5ed64378ca90e86c5e11f531e4c` |
| `attested_check.log` | `122942186718cbf3760ebc25ea265ecccbe557f8f8d9de4594d55d5e658058c8` |
