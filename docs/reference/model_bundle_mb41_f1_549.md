<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 MB-41 F-1 + F-2: head admission and aggregate byte scope

**Historical F-1 outcome (source-level, reviewed at `c9f8b053`):** Gate B checks a TorchScript head before and
after the load, using a runtime-owned allowlist. All five G-1 controls are refused (exit 2,
`load_verification` failed), and N02 is admitted. The loader-default and NamedTuple class-kind
findings were fixed; the completed independent read-only re-review has no findings. Those results
cover F-1 alone and do not verify the new integration. **F-1 + F-2 landed on main** with
[#591](https://github.com/raylei50653/saccade/pull/591) merge `c78f1319`: MB-41 is CLOSED within the bounded condition-2 declaration,
and manifest aggregate `load_verification.byte_scope` is `loaded_buffer` only when all three
objects report it. See [Landing on main](#landing-on-main-2026-10-12).

The status is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2).
This page records:

- the owner's decision;
- the design and what it relies on in LibTorch;
- the controls and their evidence.

The gap itself is in [MB-41 condition check](model_bundle_mb41_549.md#gap-g-1-the-heads-torchscript-can-read-external-unverified-bytes).

## Confirmed bounded closeout decision (2026-10-12)

The [confirmed #590 Owner decision](https://github.com/raylei50653/saccade/issues/590#issuecomment-6123060172)
supersedes the separate-F-2 plan below. [#591](https://github.com/raylei50653/saccade/pull/591)
now prepares F-1 + F-2 together; [#592](https://github.com/raylei50653/saccade/pull/592) must be
recaptured and independently reviewed against the integrated content. Requirement decision,
implementation candidate, local verification and merged status are separate claims; the
landing is recorded in [Landing on main](#landing-on-main-2026-10-12).

MB-41 condition 2 is now: **「能成功載入的 head，其載入 bytes 等於 hash 過的 bytes，
且通過 runtime 擁有的准入政策；其他 head 一律 exit 2。」** F-1 is a fail-closed admission
policy for N02's required functionality, bound to LibTorch `2.11.0+cu130`. Its operation allowlist
does not authenticate a source or promise arbitrary TorchScript safety. The importer, unpickler,
constant-folding and graph behavior assumptions listed below remain limits of this scoped claim.

The existing ctests are the maintenance obligation: N02 must load and all five G-1 controls must
be refused. A LibTorch upgrade or a new head operation updates the policy and these tests in the
same PR, using the existing coordinate and validation flow. There is no separate standing security
review or evidence ledger. If maintaining F-1 costs more than its value, the Owner decision permits
removing it while narrowing the declared guarantee and returning aggregate scope to
`hashed_before_load`.

F-2 candidate `69476d4b` was transplanted onto the latest F-1 branch as `ebef9c7c`; only one
aggregate-comment conflict needed resolution. No candidate-branch coordinate or status publication
was imported. After successful Gate B, the manifest aggregate is `loaded_buffer` only when all
three per-object scopes are `loaded_buffer`; every other case and legacy mode use
`hashed_before_load`. Gate A identity and caught/uncaught failure semantics are retained.
If bounded integration or coordinate adaptation required larger engineering, the authorized
fallback would be F-1 alone, MB-41 closed by declaring limits, aggregate `hashed_before_load`,
and `artifact_byte_scopes` authoritative per object, with no independently scheduled F-2.

This direction decision does not promote main's MB-41/aggregate state or authorize merge,
package re-pin or release. S2-2 remains governed by its canonical status; V5 **UNRESOLVED**,
installed package **UNVERIFIED**, `distribution.status=local-only`, `equivalence=unproven`.

## Original F-1 implementation authorization (historical, 2026-10-12)

- **F-1 approved:** a runtime-owned allowlist, fail-closed. The scope is what N02 needs, not a
  general TorchScript analyzer.
- **Before the load:** accept only fully validated ZIP TorchScript. Refuse FlatBuffer, scripted
  `__setstate__`, unsupported custom classes and unknown restore mechanisms. If the pre-load check
  cannot establish the reviewed admission conditions, refuse the archive. Do not rely on the post-load
  scan to catch load-time side effects.
- **After the load:** check every method, submodule, scripted function, nested block, Subgraph and
  call relation. Refuse unresolvable indirect calls, custom operations and graph structures.
- **The allowlist:** fixed in runtime source and bound to the reviewed LibTorch build and the
  implementation coordinate. No manifest or lineage may extend it.
- **The controls:** all four G-1 reproductions must be refused, with hash-consistent substitute
  heads so the existing checksum gate is not what refuses them. Each control checks:
  - exit 2;
  - a failed journal;
  - no inference;
  - no unverified external read during the load.

  N02 stays the positive control. The fifth `default` control was added during implementation
  review; the four controls above record the original owner decision.
- **Out of scope:** the legacy §4.1 contract, aggregate promotion, MB-41 closure, a general
  analyzer, and V5, package admission and distribution status.
- **Historical scope:** S2-2 PARTIAL. The original F-1 authorization kept F-2 separate;
  #590 now brings the bounded aggregate change into #591.

## Design

Code: [head_admission.hpp](../../shipping/include/saccade_shipping/head_admission.hpp) and
[head_admission.cpp](../../shipping/src/head_admission.cpp). Gate B calls them from
[detector_host.cpp](../../shipping/src/detector_host.cpp). Each refusal is a `DetectorLoadError`
whose message starts with `head admission (F-1):`. The entrypoints already record that as
`load_verification` failed, with exit 2.

**Before `torch::jit::load`.** Gate B runs this check after the head bytes are hashed and before
the operator is mapped. It reads only the verified buffer.

1. **Format.** `torch::jit::getFileFormat` must say ZIP. This is the same test the loader uses to
   choose its path, so FlatBuffer and unknown formats are refused.
2. **Records.** Every archive record must be one of these:
   - records the loader reads: `code/<qualifier>.py`, its `.debug_pkl`, `data.pkl`,
     `constants.pkl`, `data/<n>`, `constants/<n>`;
   - records it reads only as metadata, or never: `version`, `.data/version`, `byteorder`,
     `.data/serialization_id`, `traced_inputs*`. `torch::jit::load(std::istream&)` passes
     `restore_shapes=false`, so it never reads the traced inputs.

   Any other record is refused. A code record name must spell a qualifier, so the file a class
   name resolves to is the file that was checked.
3. **Code.** Every code record is parsed with LibTorch's TorchScript parser, using the importer's
   own file grammar: the version line, then imports, classes and functions. A record that does not
   parse is refused.
   - Only a class declared with the `Module` base is admitted. `NamedTuple`, `Enum`, `Interface`,
     `ModuleInterface`, classes without a base and every other class kind are refused before any
     Module metadata exception is applied. N02's 106 class declarations all have the `Module` base.
   - No class may define `__setstate__`, because the deserializer calls it while it unpickles
     (`import.cpp` `ObjLoaderFunc`).
   - Any parameter default anywhere in a code record is refused. A Module body may hold only
     methods, attribute annotations, `__parameters__` / `__buffers__` and `__annotations__` types.
     Class constants and other class-body values are refused.
4. **Pickles.** `data.pkl`, `constants.pkl` and each `.debug_pkl` are read opcode by opcode, using
   exactly the opcodes LibTorch's unpickler reads, starting from PROTO 2. Any other opcode, a
   truncated argument, or a malformed GLOBAL is refused. The GLOBALs allowed are:
   - in data and constants: classes defined in the archive's code, `_rebuild_tensor_v2`,
     `FloatStorage`/`LongStorage`, `OrderedDict`;
   - in debug ranges: `build_intlist`.

   A registered custom class is refused, because its C++ `__setstate__` would run. So is a class
   the code does not define.

**The importer evaluates defaults and constants.** LibTorch's `ScriptTypeParser::evaluateDefaults`
compiles and runs parameter defaults, `Final` class constants and NamedTuple field defaults during
the load. A hand-written code record can therefore read a sidecar through a parameter default
even when its `forward` graph uses only admitted operators. `a90ef994` added the parameter-default
and class-value refusals. The subsequent class-kind fix limits the body rules and metadata
exceptions to the Module importer; other class importers are refused before the load.

**Deserialization also compiles the code.** Apart from the evaluated values refused above, the
compiler's cleanup folds constants only for nodes
whose inputs and outputs are all immutable values (`ConstantPropagationImmutableTypes`, the
`NoAliasDb` variant). `aten::from_file` returns a Tensor, so it never folds. The 59 registered
operators that could fold with a string input are listed in `foldable_str_ops.txt` (raw evidence).
They are string methods, device parsing, formatting, the autocast-dtype lookup, and side-effecting
ops (print, warn, raise, stat) that folding skips anyway. None of them reads a file.

**After the load, before anything runs.** Every function in the head's compilation unit is walked
as compiled; no pass runs on it. That covers every method of every module and class, and every free
function. The walk covers nested blocks and graph-valued attributes such as `prim::fork`'s
`Subgraph`. Each node must be one of two things:

- one of the 28 reviewed operators in `kHeadAdmissionOperators`;
- a call that resolves statically to another function of the same compilation unit:
  `prim::CallMethod` on a class type, or `prim::CallFunction` on a function constant. That
  function is walked too.

Every other call is refused: an interface method, a function value, or a function outside the
head.

An earlier draft inlined each function first. With `graph_executor_optimize` off, which Gate B
sets before the load, `optimized_graph()` is not inlined, so `Inline` expands only one level and
nested calls survive. The resolved-call walk does not depend on that flag.

**The allowlist.** It is exactly N02's operator set. The CPU ctest checks that N02 uses every
listed operator, so the list is not wider than the reviewed head:

- **Tensor arithmetic:** `_convolution`, `linear`, `add.Tensor`, `mul.Tensor`, `neg`, `exp`,
  `silu`, `mean.dim`.
- **Shape and view:** `cat`, `stack`, `contiguous`, `expand`, `flatten.using_ints`, `flip`,
  `pixel_shuffle`, `reshape`, `select.int`, `slice.Tensor`, `transpose.int`, `size.int`,
  `to.dtype`, `Int.Tensor`.
- **The native scan:** `saccade_native::selective_scan_fwd`.
- **TorchScript plumbing:** `prim::Constant`, `GetAttr`, `ListConstruct`, `TupleConstruct`,
  `NumToTensor.Scalar`.

None of these reviewed operator implementations supplies a path for the head to read model bytes
from an external file. This is a model-byte integrity boundary; it is not a claim that tensor
execution observes no runtime state.

The list was reviewed for `2.11.0+cu130`. Gate B passes the build's
`SACCADE_LIBTORCH_BUILD_VERSION`, and any other value refuses every head. A head that needs
another operator needs a reviewed change to this source, which moves the implementation
coordinate.

**LibTorch behavior relied on.** Reviewed from the v2.11.0 sources:

- `serialization/import.cpp`: the ZIP/FlatBuffer dispatch, `ObjLoaderFunc` and `deserialize`;
- `unpickler.cpp`: `run`, `readInstruction`, `readGlobal`;
- `import_source.cpp`: `parseSourceIfNeeded`, `findNamedType` (which checks custom classes first),
  the Module class importer and `importNamedTuple`;
- `frontend/script_type_parser.cpp`: `evaluateDefaults`, parameter schemas and class constants;
- `import_export_helpers.cpp`: `qualifierToArchivePath`;
- `passes/constant_propagation.cpp`;
- `api/function_impl.cpp`: `optimized_graph`;
- `passes/inliner.cpp`: why inlining was dropped.

## Controls

The G-1 heads are built in-process: with LibTorch in C++
([g1_heads.hpp](../../tests/native/fixtures/g1_heads.hpp)) and with Torch in the CLI test. Each
head reads a sidecar float with `aten::from_file`:

- `setstate`: a scripted `__setstate__` reads it while the archive is deserialized;
- `forward`: `forward` reads it;
- `fork`: `forward` reads it through `fork`, so the read sits in the `prim::fork` Subgraph;
- `flatbuffer`: the `setstate` head saved as FlatBuffer;
- `default`: `forward` has a parameter default that reads the file during the load. The archive's
  code record is edited by hand, because saving a module normally carries the already evaluated
  value. This fifth G-1 control was added after the first interrupted review.

Every test first shows the head really reads the file: under plain `torch::jit::load`, the value
follows the sidecar, 60 then 7. It then removes the sidecar, so any read would fail with LibTorch's
own error instead of the admission message.

The table describes the controls and their final local results at source head `c9f8b053`.
The earlier `a90ef994` logs are preserved separately as historical evidence.

| Control | Where | Coverage and evidence |
|:--|:--|:--|
| [CPU ctest](../../tests/native/test_shipping_head_admission.cpp) | in process | `setstate`, `flatbuffer` and `default` are refused before the load. `forward` and `fork` load without reading, then are refused. For `fork`, a block-only walk does not see `aten::from_file`; the Subgraph walk does. There are 17 single-change pre-load negatives: the original 9, class constant, literal parameter default, two NamedTuple reserved-name fields, a class without a base, Enum, Interface and ModuleInterface. The five post-load scanner negatives remain: off-list operator, `prim::If`, an uncalled `from_file` method, an interface call and a call into another compilation unit. The interface archive is also refused before the load; its post-load scanner control loads it directly to exercise that scanner separately. Another LibTorch build is refused before and after the load. N02 is the positive control with exactly the 28 operators. Final `c9f8b053`: 82 checks, 0 failures. |
| [Gate B GPU ctest](../../tests/native/test_shipping_model_bundle_gate_b.cpp) | real `DetectorHost`, manifest mode | Each G-1 head is bound hash-consistently: the plan and resolved binding carry its path, size and sha256, so the rehash passes. `setstate`, `flatbuffer` and `default` are refused before the operator is mapped (`/proc/self/maps` has no memfd). `forward` and `fork` are refused after the load. `cross_root` (N02) is the positive control. Final `c9f8b053`: 175 checks, 0 failures; N02 loads. |
| [CLI suite](../../tests/unit/test_saccade_track_model_bundle_cli.py) `test_g1_head_is_refused_by_the_head_admission` | both entrypoints, install-shaped prefix | The real bundle with the head replaced and the lineage, attestation and manifest rebound. Gate A: `checksum_matched`, head binding `matched`. Then exit 2, journal `failed`, `load_verification` `{failed, null}`, the F-1 message, every sequence `pending`, no MOT output, and the sidecar never created. 10 G-1 tests (5 heads × 2 entrypoints). Final `c9f8b053`: four CLI files, 82 passed, including all 10 G-1 controls. |
| shipping ctest | all shipping and N03 ctests | Final `c9f8b053`: 23/23 passed. |

**Mutants.** Each mutant disables one check. The named test must fail, and the file is restored
from git afterwards.

| Mutant | Disabled | Caught by |
|:--|:--|:--|
| K1 | the `__setstate__` check | CPU ctest |
| K2 | the format check | CPU ctest |
| K3 | the Subgraph walk | CPU ctest |
| K4 | the operator list | CPU ctest |
| K5a | interface calls treated as resolved | CPU ctest |
| K5b | calls outside the head treated as resolved | CPU ctest |
| K6 | the pickle class check | CPU ctest |
| K7 | the debug-range check | CPU ctest |
| K8 | unknown records allowed | CPU ctest |
| K9 | the opcode check | CPU ctest |
| K10 | Gate B without the pre-load admission | GPU ctest |
| K11 | Gate B without the post-load admission | GPU ctest |
| K12 | the entrypoint without the post-load admission | CLI test |
| K13 | the parameter-default check | CPU ctest |
| K14 | the class-body value check | CPU ctest |
| K15 | the entrypoint without the parameter-default check | CLI test |
| K16 | the Module class-kind guard | CPU ctest |

Historical `a90ef994`: all 16 mutants K1–K15 (including K5a and K5b) were caught: 12 CPU,
2 GPU and 2 CLI. K16 was added for the class-kind fix. The final 17-mutant run (13 CPU, 2 GPU,
2 CLI) caught all 17; the historical result does not cover that fix.

## Independent review

Earlier reviews were interrupted by model-capacity or content-risk errors. Their partial output
is evidence for the findings below, not a completed `NO BLOCKER` verdict.

- **First finding, fixed in `a90ef994`:** the importer runs parameter defaults, `Final` class
  constants and NamedTuple defaults during the load. The pre-load scan had not refused them.
  The fix added parameter-default and class-value refusals, the fifth G-1 `default` control,
  literal-default and class-constant negatives, and K13–K15.
- **Subsequent independent static review, blocker at `a90ef994`:** Module metadata exceptions
  were applied to every class kind. A NamedTuple field named `__parameters__` or `__buffers__`
  could pass that scan even though the NamedTuple importer evaluates it during the load. The
  source fix `c9f8b053` checks that the class base is `Module` before applying any body or metadata
  exception. Six harmless class-kind regressions and K16 cover the refusal rule.
- **Final review at `c9f8b053364562efdc80349e710590aa8cc74134`: No findings / NO BLOCKER.**
  The independent reviewer checked the full F-1 source diff against `5da12874`, the local
  LibTorch 2.11.0 importer and graph/call traversal, and independently ran the existing CPU
  admission ctest (82 checks, 0 failures). GPU, CLI and shipping results above were run by the
  implementer. This is a completed source review, not a GitHub APPROVED review, remote CI,
  installed-package verification or merge authorization.

## Merged baseline and remaining scope

- **Main: MB-41 CLOSED within the bounded condition-2 declaration.** F-1 + F-2 landed with #591
  (see [Landing on main](#landing-on-main-2026-10-12)). Manifest aggregate is `loaded_buffer` only
  for a successful load in which all three per-object scopes are `loaded_buffer`; legacy stays
  `hashed_before_load`. Old F-1 heads and the old #592 coordinate remain historical evidence only.
- **Unchanged:** S2-2 PARTIAL. The legacy mode §4.1 contract. V5 **UNRESOLVED**. The installed
  package **UNVERIFIED** (the entrypoint pin predates F-1). `distribution.status=local-only`.
  `equivalence=unproven`.

## Known limits

- **Admitted heads:** only Module class declarations using N02's 28 operators, with no control
  flow, custom class, interface, parameter default or class constant. NamedTuple, Enum and classes
  without the Module base are refused. This is the owner's narrow scope; a broader head is
  refused, not analyzed.
- **LibTorch dependency:** the pre-load guarantee relies on LibTorch 2.11.0's deserializer as
  reviewed above. The version binding refuses other builds; it does not re-verify the library's
  bytes. Package admission does that.
- **Allowed opcodes:** the scan accepts every opcode LibTorch's unpickler accepts, which is wider
  than N02's pickles use. Only the GLOBALs are narrowed.
- **CI:** none of these controls runs in CI. CI builds no LibTorch target, and the GPU and CLI
  tests need the models and a GPU. The evidence is local.

## Raw evidence

All files are local-only (gitignored). Historical F-1 files remain under
`results/549_f1/` in the original `.claude/worktrees/docs-549-s2-2-status-writeback`
worktree. New integration evidence uses `results/549_f1/integration_20261012/` in the
root WSL workspace; old logs and the pinned operator are preserved.

**Historical evidence at `a90ef994`, before the class-kind fix.** These hashes record the existing
logs; they do not certify the final source tree.

| File | SHA-256 |
|:--|:--|
| `head_admission_ctest_a90ef994.log` | `288cdd07156237a4514ac13924fa9a3a36348741f468c3279f435fb83a91828f` |
| `gate_b_ctest_a90ef994.log` | `075bdd20f7d497426873709e4a410dbaaad7aa958b570f934353be3c9f50f8b4` |
| `cli_suites_a90ef994.log` | `9bf9e264025d05787655a97df0760620c19bf4609e3cb6a13c63662cfef8856e` |
| `shipping_ctest_a90ef994.log` | `41adf249870e6926b7636842656dfd8d82b4a51f28c78b65ea55c098c5c30ac3` |
| `mutants.log` (16-mutant run) | `5abfe7efab2a0a6049b3e881311f4f897dbd83045f7a04d788912cd1a5dd91ce` |

**Supporting operator review evidence.**

| File | SHA-256 |
|:--|:--|
| `foldable_str_ops.txt` | `7969abeb5974e79135fa982f4f3fa6ead4000e5bc3aa907ab17155eed9322df0` |
| `foldable_str_ops.py` | `c483bfdbb0e5950261e0b286315aadac4cb4953effc01d986caab2205632cc9f` |

**Final source evidence after the class-kind fix (`c9f8b053`).** The files below bind the local
tests, mutation controls and completed independent source re-review. Fresh coordinate evidence
is recorded after capture. The final coordinate-branch pre-push log hash and delivery head are
recorded in the Draft PR and #549 checkpoint after that run, so this page does not imply that
the implementation-only branch has a current publication or a remote CI pass.
The earlier coordinate candidate was captured at `2621b016`, before either review fix. The
complete pre-push at that same head had 5687 passed and 4 coordinate-related failures; it is
historical evidence, not final acceptance. Publication follows the
[runtime coordinate runbook §3.2](runbooks/runtime_identity_republication.md#32-review-boundary-與-landing-的關係)
with a separate stacked coordinate PR containing canonical and archive files only.

| File | SHA-256 |
|:--|:--|
| `head_admission_ctest_c9f8b053.log` | `4e1f795e67f7ea1c15aded2251bdd20d58005974df463de66d34fc8714e38caa` |
| `gate_b_ctest_c9f8b053.log` | `f9f2f6789779fe6fe19d8ed599775626af06816373326ea4d87876ec078bab3a` |
| `cli_suites_c9f8b053.log` | `d3ceca2a771a3b933e3391bd0c82cc31cdca53cbdde61b3e10e691a9f584cd62` |
| `shipping_ctest_c9f8b053.log` | `32a931aef137e4bf5db8e456039a9d9388ad067487567d7a69041237f3bfd029` |
| `mutants_c9f8b053.log` | `cc787b932bab1e864af57001abd2a4f1574a1f21af70332d52c8597931016523` |
| `mutants_c9f8b053.py` | `000c58cd8efa815f9cb0e00032835bebaa49cf3047ddadbe422c7cbba6570918` |
| `review/codex_static_a90ef994.md` | `05e1a313acefe0dbeb032fc46316a7dcd56800b80e6bd63b6bb889e6c2792bcc` |
| `review/codex_final_c9f8b053.md` | `995041707207ca1a0283598f5cc21958a62aad34564db06a5c4c6e831726ffb3` |

## Historical F-1 runtime coordinate candidate

This candidate belongs to F-1 alone and cannot verify F-1 + F-2. It was captured from
clean tree `e02b030410905158b2192d5b1f7e09beb43f2854` with `--run-probe` and
`--require-complete`. Only the probe targets `saccade_tracking_ext` and
`saccade_scan_plugin` were built in `build/runtime_identity`. The pinned shipping
operator in `build/libsaccade_scan_torchop.so` remained
`aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4`.

Only implementation moves: `942e0e3ce5c5717b853828bfd0d2496070ff35cd124c6c45131ab4436d2d7363`
to `23c072cfd9d76bff05655aee58281f75cad7b3609a2a9e8c85f662f997fc910c`, 312 to 314 files. The two additions are
`head_admission.hpp` and `head_admission.cpp`; the five modified implementation members
are `shipping/CMakeLists.txt`, `detector_host.hpp`, `run_completion.hpp`,
`verified_model_bytes.hpp` and `detector_host.cpp`. The other four axes are unchanged.
Source, observed host, current runtime inputs and captured fresh probe compare with
zero failures and zero warnings; probe artifact sizes and SHA-256 values also match.
The fresh probe digest equals the outgoing digest, which carries no equivalence claim:
`equivalence.state=unproven`. The head/tree witness is navigation metadata. Later evidence
documentation commits do not change an axis. Promotion retains the separate stacked
canonical-plus-archive review boundary; it does not authorize merge into main.

| File | SHA-256 |
|:--|:--|
| `runtime_identity.candidate.json` | `dc8ab54e1e53cdb2d056b7f4e00c2ba0067d1aad923182125127e721e7d01fd2` |
| `coordinate_capture_e02b0304.log` | `3a42a8d5023ff6ad8a3836e6b6863ba02fcc4b1ad0d51733a78fcd6bea62f728` |
| `build_runtime_identity_e02b0304.log` | `2adb13b22a50a85f24cb21908e9a87803ed10e860b284c3c82b2d07b042ef655` |
| `coordinate_comparison_e02b0304.json` | `a13325849efdf1f1586784bf4b7c43353e48e016708a544da0c14f06e5830862` |
| `capture_final.sh` | `e2e025c620aea03be78ba91a73ad7de38bb81ad6f0dd72c2e45cae7c2a75a857` |
| `compare_final_coordinate.py` | `f51f88775baaf5f28a8fdc4f757313aba2fb08e8336b553d6c503c923da15839` |


## Integrated F-1 + F-2 verification (2026-10-12)

**Chosen path: F-1 + F-2.** The eight-file `69476d4b` candidate contributed seven necessary
file changes as `ebef9c7c`; its `verified_model_bytes.hpp` comment was already corrected by F-1.
The only transplant conflict was an aggregate comment, with no broader engineering needed.
The necessary adaptation to two existing writer tests keeps removed-guard mutations visible as
ordinary CHECK failures rather than a later double-record exception. No attack controls or
new security framework were added.

The code, contracts and test adaptation were settled in clean
`433974513f22c383398d039a84a15a6686281aac` before capture and validation. All local results below
bind that source/test content. The baseline records 140 tracked code/test SHA-256 values and
actual tested binary hashes. `local_validation_final_43397451.json` records the final restored
binaries used by the repeated normal tests; some hashes changed after rebuilding unchanged
source, and no binary reproducibility is claimed. Later evidence documentation and coordinate publication change no
controlled source/test bytes; their exact delivery heads and final checks are recorded in
[#591](https://github.com/raylei50653/saccade/pull/591) and
[#592](https://github.com/raylei50653/saccade/pull/592).

| Current integration check on original WSL host | Result |
|:--|:--|
| Named shipping/N03 test targets and both entrypoints | Build succeeded; pinned operator unchanged |
| Shipping/N03 ctest | 23/23 PASS, no skipped |
| Focused CPU completion / manifest preflight | 329 / 6515 checks, zero failures |
| CPU head admission | 82 checks, zero failures; N02 and the five existing G-1 controls |
| GPU Gate B | 175 checks, zero failures; N02 loads, five hash-consistent G-1 heads refused |
| Four existing CLI suites, both entrypoints | 82 passed, zero skipped; manifest aggregate `loaded_buffer`, legacy `hashed_before_load`; raw MOT equality is diagnostic |
| Existing mutation controls | 23/23 expected assertions caught: 17 F-1, five F-2 CPU, one F-2 integration mutation tested across both entrypoints |
| Mutation restoration | Four source files restored byte-for-byte, six affected targets rebuilt, all 25 named targets rebuilt, normal ctest 23/23 and CLI 82/82 passed again |

A crash, timeout, build error, unavailable environment or skipped test cannot count as a caught
mutation. The two writer guard mutations now fail the expected CHECK normally with exit 1.
Complete pre-push and required remote CI are final-delivery-head checks; their logs, exact heads
and results are maintained in the Draft PR descriptions and #550 after delivery. Historical
`c6f6c83d` pre-push/CI results do not validate this integration. The stacked #592 workflow
filter may leave it without remote checks; that is not remote CI success.

**Fresh coordinate.** A new `--run-probe --require-complete` capture used clean `43397451` and
scratch `results/549_f1/integration_20261012/runtime_identity.candidate.json`. Only the existing
probe targets were built. Only implementation changes:
`942e0e3ce5c5717b853828bfd0d2496070ff35cd124c6c45131ab4436d2d7363` ->
`c0a319fa1207523a9d163dd9fc8741a4a0df0949792b6145297b024b66c91f9f`, 312 -> 314 files,
two added and seven modified. The other four axes stay unchanged. The 525-frame probe was
freshly executed; its equality to the outgoing probe is a fixture diagnostic, with
`equivalence.state=unproven`. Source, observed host, all runtime inputs and the captured probe
compare with zero failures and zero warnings. Both probe artifacts' physical sizes/hashes match.
The shipping operator remains
`aa84cccd5c5094b33d63217eeb377452ae19c69cd1382ec7de8ba96d0d8d31c4`.

**Independent read-only reviews.** Source and contract re-review at `43397451` reports
No findings / NO SOURCE BLOCKER; the reviewer independently rebuilt the exact snapshot's CPU
completion and manifest tests (329/0 and 6515/0), and checked unchanged F-1 source against its
independent 82/0 admission run. GPU, CLI and mutation execution belongs to the implementer.
The independent coordinate candidate review reports No findings / NO BLOCKER within publication
scope: all 344 source/recipe blobs match frozen Git objects, the five axes and original-host
inputs match, fresh-capture provenance and both artifacts match, and outgoing canonical archive
bytes are exact. Publication-diff and final-head re-reviews are recorded in the PRs after the
canonical-plus-archive commit is fixed. These are read-only findings, not GitHub APPROVED reviews
or merge authority.

**Review and landing boundary.** #592 retains a two-file canonical-plus-archive review surface
against a frozen implementation snapshot (`review/549-f1-f2-source`). This preserves the
runbook's separate implementation/coordinate review boundary while the same reviewed publication
content is committed to #591 for current-coordinate required CI and later atomic co-land.
The old `ae1f28ea` / `c6f6c83d` heads and their local evidence remain historical. Neither PR is
made Ready or merged by this task.

After Owner authorization binds the final heads, actual merge and its required main CI permit
MB-41 to close **within the bounded condition-2 declaration**, and manifest aggregate to become
`loaded_buffer` only when all three objects report it. Until then main remains MB-41 OPEN /
aggregate `hashed_before_load`. Legacy stays `hashed_before_load`. S2-2 stays governed by its
canonical PARTIAL status; V5 **UNRESOLVED**, installed package **UNVERIFIED**,
`distribution.status=local-only`, `equivalence=unproven`. No package re-pin, S2-3 or release is
part of this closeout.

All integration files below are local-only in `results/549_f1/integration_20261012/`:

| File | SHA-256 |
|:--|:--|
| `baseline.json` | `28290fd9b2aa199d09827c0286381e996e334dae420d01e7c1bb64651a4e5afe` |
| `validation_baseline_43397451.json` | `a698057c8500e9ace4711003618d625d79cc320cbd91c2a6dd40f9b1a97e5302` |
| `baseline_validation_result_43397451.json` | `f831a35ac63af965c1653ed1cfdedcca1a501b8f4ee2b08e4949f6412b3eb595` |
| `shipping_ctest_43397451.log` | `2fe228c6fa8221e1fb868086826c8568b36c7142deb46a6b36f7327d81a83b0b` |
| `focused_ctest_43397451.log` | `1617df9b40bcf51dec6427979b96ad01b58adb1d74dc4b5d1d291b273f126bcf` |
| `cli_suites_43397451.log` | `bff047a48921dc9016230f10b4a37b8df7ef58d9fa15e55978e4da882e5291b5` |
| `mutants_43397451.json` | `41d8e091e069e36f8fc34a66978423dd817ceecfd1d54233ab8fe194b1e9980c` |
| `mutation_controls.py` | `485c05771884a9e11ecd752589a5913cb504381fb26874731a3b5997820c7383` |
| `mutation_definitions.py` | `b36d176160c7d46a80548862c816777bf24e9abbcb36cba45adc80e7ee46bb01` |
| `shipping_ctest_post_mutation_43397451.log` | `1e35ba6fd2001639d72639daebcc1bee9654ed49400fbd7997299a664419ea36` |
| `focused_ctest_post_mutation_43397451.log` | `0bf1b33ad115f9cf97af4a27d221fda0c0cff4f55e959d7f133c45202dcaacb2` |
| `cli_suites_post_mutation_43397451.log` | `7b4718f2efcff5926b45ae9834df6c54505fba9b0a54a8844b3ab6bdc41839f2` |
| `runtime_identity.candidate.json` | `6df3bcfb5e1a23be754f2fc5f09bca4ef84f05cfef341ef99c0143e8b8dd4fe3` |
| `coordinate_capture_43397451.log` | `8f24b04474e7d13ccaf7cc5321cfdcc4a858c05991c54d592c952345d105a2de` |
| `build_runtime_identity_43397451.log` | `ca720c481f9d285614ef246670b493012419908964f530e019435dbe63a499ab` |
| `coordinate_comparison_43397451.json` | `3376b5dec4346d658381389aba71bd31b9ae4d6a391d5c97b22418177b1c1237` |
| `source_review_43397451.md` | `0f0bfda0f629ee17852b9f396bca402c6c122e92d83754ae2a7eb5f406d10cc9` |
| `coordinate_candidate_review_43397451.json` | `2fad81f0d05f4ea3182ae5f1a3868869249930a2ad387d57b82d58944cca87f6` |
| `local_validation_final_43397451.json` | `98fd72442106d45bde76dfade437d745297ba3730bd1a1138fe8f04d03992169` |

## Landing on main (2026-10-12)

The Owner authorized merging the final heads, implementation
`306efa4c1257801007b89e6fcd91e8e24cc87072` and coordinate `03bffa1e5bd24466b1bc3dadf135ff7cb4a4fbf6`.
[#591](https://github.com/raylei50653/saccade/pull/591) merged as merge commit `c78f1319b4170f31cf172b1861f7c8dcc2155f0e`, parents `5da12874` + `306efa4c`; its tree
`323fd98a7355e623adb3416cc2dc9fb76f6efd2f` equals both reviewed delivery trees. The reviewed
coordinate bytes landed atomically with it. [#592](https://github.com/raylei50653/saccade/pull/592) was closed without merging and
stays the two-file coordinate review record. Required checks at the merge SHA: [main CI 7/7](https://github.com/raylei50653/saccade/actions/runs/38249645847) + [main C++ 1/1](https://github.com/raylei50653/saccade/actions/runs/38249645844), SUCCESS.

Within the bounded condition-2 declaration of
[contract 4.1](../architecture/model_bundle_contract_549.md#mb41-head-admission), **MB-41 is CLOSED**.
Manifest aggregate `load_verification.byte_scope` is `loaded_buffer` only for a successful load in
which the operator, head and engine each report `loaded_buffer`; failed loads have no byte scope
and legacy mode stays `hashed_before_load`. Remote CI does not run GPU or real-entrypoint CLI; that
evidence is the local integration check above. Unchanged: S2-2 **PARTIAL**, V5 **UNRESOLVED**,
installed package **UNVERIFIED** (the entrypoint pin predates F-1), `distribution.status=local-only`,
`equivalence=unproven`. No package re-pin, S2-3 or release is part of this landing.
