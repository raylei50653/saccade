<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 MB-41 F-1: TorchScript head admission in Gate B

**Outcome:** Gate B refuses every G-1 head (exit 2, `load_verification` failed) before it can
read bytes outside its hashed archive. The four G-1 reproductions are now negative controls, and
N02 is still admitted. **MB-41 stays OPEN.** The aggregate `load_verification.byte_scope` stays
`hashed_before_load` until F-2 lands as a separate PR.

The status is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2).
This page records:

- the owner's decision;
- the design and what it relies on in LibTorch;
- the controls and their evidence.

The gap itself is in [MB-41 condition check](model_bundle_mb41_549.md#gap-g-1-the-heads-torchscript-can-read-external-unverified-bytes).

## Owner decision (2026-10-12, in session)

- **F-1 approved:** a runtime-owned allowlist, fail-closed. The scope is what N02 needs, not a
  general TorchScript analyzer.
- **Before the load:** accept only fully validated ZIP TorchScript. Refuse FlatBuffer, scripted
  `__setstate__`, unsupported custom classes and unknown restore mechanisms. If the pre-load check
  cannot show the deserialization path is safe, refuse the archive. Do not rely on the post-load
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

  N02 stays the positive control.
- **Out of scope:** the legacy §4.1 contract, aggregate promotion, MB-41 closure, a general
  analyzer, and V5, package admission and distribution status.
- **Kept:** S2-2 PARTIAL. F-1 and F-2 stay separate.

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
   own file grammar: the version line, then imports, classes and functions. No class may define
   `__setstate__`, because the deserializer calls a class's `__setstate__` while it unpickles
   (`import.cpp` `ObjLoaderFunc`). A record that does not parse is refused.
4. **Pickles.** `data.pkl`, `constants.pkl` and each `.debug_pkl` are read opcode by opcode, using
   exactly the opcodes LibTorch's unpickler reads, starting from PROTO 2. Any other opcode, a
   truncated argument, or a malformed GLOBAL is refused. The GLOBALs allowed are:
   - in data and constants: classes defined in the archive's code, `_rebuild_tensor_v2`,
     `FloatStorage`/`LongStorage`, `OrderedDict`;
   - in debug ranges: `build_intlist`.

   A registered custom class is refused, because its C++ `__setstate__` would run. So is a class
   the code does not define.

**Deserialization also compiles the code.** The compiler's cleanup folds constants only for nodes
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

None of them reads a file or any other external state.

The list was reviewed for `2.11.0+cu130`. Gate B passes the build's
`SACCADE_LIBTORCH_BUILD_VERSION`, and any other value refuses every head. A head that needs
another operator needs a reviewed change to this source, which moves the implementation
coordinate.

**LibTorch behavior relied on.** Reviewed from the v2.11.0 sources:

- `serialization/import.cpp`: the ZIP/FlatBuffer dispatch, `ObjLoaderFunc` and `deserialize`;
- `unpickler.cpp`: `run`, `readInstruction`, `readGlobal`;
- `import_source.cpp`: `parseSourceIfNeeded` and `findNamedType`, which checks custom classes
  first;
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
- `flatbuffer`: the `setstate` head saved as FlatBuffer.

Every test first shows the head really reads the file: under plain `torch::jit::load`, the value
follows the sidecar, 60 then 7. It then removes the sidecar, so any read would fail with LibTorch's
own error instead of the admission message.

| Control | Where | Result |
|:--|:--|:--|
| [CPU ctest](../../tests/native/test_shipping_head_admission.cpp) | in process | `setstate` and `flatbuffer` refused before the load. `forward` and `fork` load without reading, then are refused. For `fork`, a block-only walk does not see `aten::from_file`; the Subgraph walk does. Single-change negatives: before the load, 9 (extra record, custom class, undefined class, other global in data and in debug ranges, opcode, protocol, truncation, top-level code); after the load, 5 (off-list operator, `prim::If`, an uncalled `from_file` method, an interface call, a call into another compilation unit). Another LibTorch build is refused before and after the load. N02 is admitted with exactly the 28 operators. 63 checks. |
| [Gate B GPU ctest](../../tests/native/test_shipping_model_bundle_gate_b.cpp) | real `DetectorHost`, manifest mode | Each G-1 head is bound hash-consistently: the plan and resolved binding carry its path, size and sha256, so the rehash passes. `setstate` and `flatbuffer` are refused before the operator is mapped (`/proc/self/maps` has no memfd). `forward` and `fork` are refused after the load. `cross_root` (N02) still loads. 171 checks. |
| [CLI suite](../../tests/unit/test_saccade_track_model_bundle_cli.py) `test_g1_head_is_refused_by_the_head_admission` | both entrypoints, install-shaped prefix | The real bundle with the head replaced and the lineage, attestation and manifest rebound. Gate A: `checksum_matched`, head binding `matched`. Then exit 2, journal `failed`, `load_verification` `{failed, null}`, the F-1 message, every sequence `pending`, no MOT output, and the sidecar never created. 8 tests. |

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

All 13 are caught.

## What this does not change

- **MB-41 OPEN.** The aggregate `byte_scope` stays `hashed_before_load`. F-2 (a separate PR) does
  the rest:
  1. rebase the candidate `69476d4b` onto F-1;
  2. recapture the coordinate;
  3. rerun the full verification;
  4. decide on the promotion.
- **Unchanged:** S2-2 PARTIAL. The legacy mode §4.1 contract. V5 **UNRESOLVED**. The installed
  package **UNVERIFIED** (the entrypoint pin predates F-1). `distribution.status=local-only`.
  `equivalence=unproven`.

## Known limits

- **Admitted heads:** only heads that use N02's 28 operators, with no control flow, no custom
  class and no interface. This is the owner's narrow scope; a broader head is refused, not
  analyzed.
- **LibTorch dependency:** the pre-load guarantee relies on LibTorch 2.11.0's deserializer as
  reviewed above. The version binding refuses other builds; it does not re-verify the library's
  bytes. Package admission does that.
- **Allowed opcodes:** the scan accepts every opcode LibTorch's unpickler accepts, which is wider
  than N02's pickles use. Only the GLOBALs are narrowed.
- **CI:** none of these controls runs in CI. CI builds no LibTorch target, and the GPU and CLI
  tests need the models and a GPU. The evidence is local.

## Raw evidence

All files are local-only, under `results/549_f1/` (gitignored).

| File | SHA-256 |
|:--|:--|
