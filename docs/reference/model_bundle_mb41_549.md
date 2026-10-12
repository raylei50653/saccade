<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 MB-41: historical condition check and bounded F-1/F-2 closeout

**MB-41 is CLOSED on main within the bounded condition-2 declaration** ([#591](https://github.com/raylei50653/saccade/pull/591)
merge `c78f1319`, required main CI SUCCESS). Manifest aggregate `load_verification.byte_scope` is
`loaded_buffer` only when all three objects report it; legacy stays `hashed_before_load`.
The original condition check below found G-1 and is retained as historical evidence.
The [final owner ruling in #590](https://github.com/raylei50653/saccade/issues/590#issuecomment-6123060172)
settled the bounded requirement and the F-1/F-2 integration path; independent coordinate review
was [#592](https://github.com/raylei50653/saccade/pull/592). Local verification, remote CI,
independent review and the landing are recorded in [F-1/F-2 evidence](model_bundle_mb41_f1_549.md).
Old head and coordinate results do not verify the integration.

The status is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2).
This page records:

- the owner's conditions;
- what was checked and how;
- the gap that keeps MB-41 open, with its evidence;
- the bounded closeout already decided by the owner.

## Authorization

The owner first gave a conditional approval in session on 2026-10-12. The final #590 ruling
replaces the broad head guarantee with bounded condition 2 below. Closing MB-41 and promoting
the manifest-mode aggregate remain subject to applicable verification and final-head merge
authorization:

1. Operator, head and engine cover every load file that
   [contract 4.1](../architecture/model_bundle_contract_549.md#load-verification) requires.
2. 「能成功載入的 head，其載入 bytes 等於 hash 過的 bytes，且通過 runtime 擁有的准入政策；其他 head 一律 exit 2。」
   This is the N02-scoped, fail-closed admission contract for LibTorch `2.11.0+cu130`, not a
   guarantee that arbitrary TorchScript is safe or from a trusted source. Scope, LibTorch
   behavior assumptions and maintenance terms are authoritative in
   [contract 4.1](../architecture/model_bundle_contract_549.md#mb41-head-admission).
3. The corresponding tests, negative controls and journal consistency checks all pass.

Under the chosen F-1/F-2 path, if any condition fails, MB-41 stays OPEN; the gap, evidence and
minimal follow-up must be listed. A historical owner hold is not a reason. Qualified V5, package
admission and `distribution.status` keep their states.

## Historical check

The original check used a candidate on `feat/549-mb41-aggregate-loaded-buffer`.
That old branch is retained for reference; only the necessary `69476d4b` implementation changes
are integrated into #591. Its old coordinate publication and documentation do not accompany
the new integration. The historical commits were:

- `69476d4b`: the change.
  - `aggregate_load_byte_scope` gives `loaded_buffer` only in manifest mode, and only when all
    three per-object scopes are `loaded_buffer`.
  - Legacy mode keeps `hashed_before_load`.
  - `RunCompletion` refuses `loaded_buffer` for a legacy run or a failed load.
- `1d3c76af`: the coordinate republication. Only the implementation axis changes,
  `942e0e3c…` → `78741f99…`, and `--mode attested` exits 0.
- `0032ab38`: the documentation.

An independent read-only Codex review then covered `03b6a82d..0032ab38`. It found no blocker
in the writer logic, the coordinate or the status words. It found **one major issue**, which
this page reproduces below.

## Historical results

| Condition | Result | Evidence |
|:--|:--|:--|
| 1. Load files covered | **Met** for the three files that contract 4.1 names. In the shipping tree, `dlopen(`, `torch::jit::load(` and the `TRTEngine` constructors appear only at `detector_host.cpp` (head from a stream, engine from a pointer/size) and `operator_loader.cpp` (`/proc/self/fd/<M>`). Each runtime builds one `DetectorHost`. The pathname loaders under `src/perception/` are not reachable from shipping code. | `git grep` over `shipping/`, `src/`, `include/` |
| 2. Loaded bytes = hashed bytes, no unverified alternative path | **Not met.** The outer bytes hold: head and engine come from one immutable hashed buffer, and the operator comes from a sealed, read-back memfd whose mapping is checked. **But the head's TorchScript can read bytes that were never hashed**, both while it loads and while it runs (gap G-1 below). | `repro_from_file.log`, `repro_forward_from_file.log`, `codex_review_r1.md`. Outer-byte controls: `verified_model_bytes`, `model_bundle_gate_b`, `operator_loader` ctests and the N03 auditor suite |
| 3. Tests, negative controls, journal consistency | **Not met as a whole.** The shipping ctest, the CLI suites and the mutants pass on the candidate (below), but there is no negative control for G-1, so the controls do not cover condition 2. | Candidate runs below |

Historical candidate runs (local, gpu0 lease; CI runs neither GPU nor real-binary CLI tests):

- shipping ctest 22/22;
- the 12 CLI suites of the #584 chain: 656 passed, 0 skipped;
- five CPU mutants and one end-to-end CLI mutant, all caught;
- full pre-push on the candidate (`0032ab38`): 5671 passed, **1 failed**
  (`test_dali_pipeline::test_connect_returns_true`). Its video,
  `datasets/MOT17_videos/MOT17-05-SDP.mp4`, was not linked into the worktree.
- full pre-push on this branch (`af256e8a`): 5679 passed, **2 failed**. Both were
  `test_real_bundle_runs_from_two_roots`: the worktree binaries were still the candidate's,
  which writes `loaded_buffer`, while this branch's test expects `hashed_before_load`.
- After linking the video and rebuilding both entrypoints from this branch's source, the DALI
  test and the model-bundle CLI suite ran again: 55 passed (12 + 43).

## Gap G-1: the head's TorchScript can read external, unverified bytes

`aten::from_file` is a scriptable op that reads a file named at run time. Two paths use it:

- **G-1a, at load.** LibTorch runs a scripted `__setstate__` (of a module or a scripted class)
  while it deserializes the archive. The C++ `torch::jit::load(std::istream&)` that Gate B uses
  goes through the same import code, and it accepts **FlatBuffer** TorchScript as well as ZIP.
  The read happens during deserialization, before Gate B's post-deserialization head checks
  (the hashes and the manifest/SM checks come earlier). The bytes are neither in the hashed
  archive nor among the head's parameters or buffers.
- **G-1b, at run time.** `forward`, or any method it calls, can read a file on every inference.
  Gate B already walks the inlined forward graph (`detector_host.cpp`). That walk refuses
  `prim::PythonOp` and the Python scan op, but not `aten::from_file`, and it visits nested
  blocks only. A read passed to `torch.jit.fork` stays in the `prim::fork` node's graph-valued
  `Subgraph` attribute after inlining, so the walk never sees it.

**Reproduction of G-1a** (`repro_from_file.py`, Torch 2.11, entirely in memory):

1. Save a module whose `__setstate__` reads a sidecar file containing `60.0`.
2. Load it from an in-memory buffer (archive sha256 `1dd5b309…`). It restores `w = [60.0]`.
3. Rewrite the sidecar to `7.0` and load **the same archive bytes** again. It restores
   `w = [7.0]`.

The parameter and buffer lists are empty in both loads.

**Reproduction of G-1b** (`repro_forward_from_file.py`): one loaded module's `forward` returns
`[60.0]`, and after the sidecar is rewritten the same module returns `[7.0]`. The inlined
forward graph contains `aten::from_file`.

**Variants** (`repro_variants.py`; review round 2 reported both, and they are reproduced here):

- **fork.** `forward` reads through `torch.jit.fork(torch.from_file, …)`. It returns `[60.0]`,
  then `[7.0]` after the sidecar changes. A block walk of the inlined graph does not see
  `aten::from_file`; it sits inside the `prim::fork` node's `Subgraph`.
- **FlatBuffer.** The same `__setstate__` module saved as FlatBuffer (not a ZIP archive) loads
  through `torch.jit.load` and restores `[60.0]`, then `[7.0]` after the sidecar changes.

**Scope.**

- **Production head N02** (`mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt`, sha256
  `1663ec97…`) is not affected. Diagnostic `scan_head.py`: none of the 102 archive code files
  contains `__setstate__` or `from_file`, no loaded method is named `__setstate__`, and no node
  kind contains `file`.
- **At that check, the broad guarantee was not met.** It had been read as covering every head
  a manifest could bind, including a self-consistent substitute at `checksum_matched`. #590
  replaces that declaration with the runtime-owned admission contract above.
- **Engine.** TensorRT's embedded host code needs an opt-in that this code does not enable.
- **Operator.** It is a single compiled, reviewed digest (the N03 pin). What its own code does
  at load time is fixed by that pin.

## Bounded closeout decided in #590

**Chosen candidate path: F-1 plus F-2 in #591.** The admission policy remains limited to N02's
required TorchScript features and LibTorch `2.11.0+cu130`. The runtime owns it, and the manifest
or lineage cannot extend its operation allowlist. Passing this allowlist does not authenticate
source or change Gate A identity. The guarantee relies on the reviewed LibTorch importer,
unpickler, default/constant evaluation, constant propagation and graph/call behavior described
in [F-1 design](model_bundle_mb41_f1_549.md#design); the version binding does not re-verify
LibTorch's library bytes. Other builds or unsupported head functionality fail closed.

The integrated candidate includes the necessary `69476d4b` writer and test changes. Its aggregate
becomes `loaded_buffer` only for a successful manifest-mode load with all three object scopes
`loaded_buffer`. Legacy stays `hashed_before_load`; caught failure stays exit 2 with failed/null
scope, and uncaught auditor/process termination retains its existing null/running limit.
Per-object `artifact_byte_scopes` retain authority for their own scopes.

After code and contracts are settled, #592 needs one fresh capture and independent review for
the integrated content under the [coordinate runbook](runbooks/runtime_identity_republication.md).
Its review surface stays separate and the implementation/coordinate prepare for atomic co-land.
Existing ctest, GPU/Gate B, both entrypoint CLI, mutation controls, full pre-push and required
remote CI must be bound to the actual tested head/content. The old runs on this page remain historical,
and this page does not declare the integrated candidate locally verified or merged.

**Maintenance and exit.** Keep the existing ctest obligations: N02 loads and all five G-1
controls are refused. A LibTorch upgrade or new head op updates the policy and tests in the same
PR. No standing security re-review or new evidence ledger is required. If F-1 costs more to
maintain than it is worth, it may be removed under #590 with the guarantee and declarations
lowered together and aggregate returned to `hashed_before_load`.

If integrating F-2 or adapting its coordinate/contracts requires an engineering expansion,
#590's fallback is F-1 alone, MB-41 closed through explicit limits, permanent aggregate
`hashed_before_load` and authoritative per-object scopes. The concrete reason must be recorded;
F-1 still needs its verification, and no independent F-2 is scheduled. This fallback has not
been selected for the present F-1/F-2 candidate.

**Unchanged by this check:**

- the S2-2 label (PARTIAL);
- legacy mode (`hashed_before_load`, contract 4.1);
- V5 **UNRESOLVED** (driver `617.42` ≠ `616.92`);
- installed package **UNVERIFIED**;
- `distribution.status=local-only`;
- `equivalence=unproven`.

## Raw evidence

All files are local-only, under `results/549_mb41/` (gitignored).

| File | SHA-256 |
|:--|:--|
| `review/codex_prompt_r1.md` | `5f6e46598cf485b46b0f3cca91a50bb93f3f19b2bd69dac05626a78d47b2147e` |
| `review/codex_review_r1.md` | `ea58e2b82b52961151812abe2623cc5c954cf269ed98cae144a33074a24b7049` |
| `review/repro_from_file.py` | `515be846f51a43c6272d53393b92e54058f8c3998ffbacb9a83d11a1541357d2` |
| `review/repro_from_file.log` | `534ca8ae2f281776c44360a67d3bfa1d56655f0bc783c89b2db6d65878190514` |
| `review/scan_head.py` | `fdd17673696d9e71ae50d5209c7a6dc7a29d0113e2fc8d7fe24b409c5b68ecec` |
| `review/production_head_scan.log` | `b019305789472a3ef2b80859a321ecc557e9b0dd609df95763661252d39d63a5` |
| `review/repro_forward_from_file.py` | `dd4ba683b7caa5f553e7a9921f05c5bfc56213f62475ed57d8206f6cb93368a5` |
| `review/repro_forward_from_file.log` | `5544c2dd963fb01410661d2de55a584d324acd2c8394328358194866f2826c38` |
| `review/codex_prompt_r2.md` | `bc2538be9e43456d19e3c1ec422b17fad6c5d5d5a8c201e00367c156f9787ccd` |
| `review/codex_review_r2b.md` | `3d006ded8bcc9256a4e1d3967fdcbb528e79dfff01c1c91c22cd634f3b0fec52` |
| `review/repro_variants.py` | `fef95317c0956894eb5e6a66e1a465e0bff050c5e855bfbaf7e05a3f308b9945` |
| `review/repro_variants.log` | `da56137e226baa06e6ca6bc0432c931c3deb6862adba76f3c34bb02694098700` |
| `pre_push_candidate_0032ab38.log` | `ab24e2064edf2394478c39e1e75e85dd88089a33f12d8afde3405430ee7ad59e` |
| `pre_push_docs_af256e8a.log` | `bd466aa5898bf56883e1e48c6ae6148d85fbb099b99f00614c5fd25347379360` |
| `rerun_dali_and_model_bundle_cli_17777293.log` | `a5d1692034760847e45cac89be8a572d2912de49f913b7605bfaf2aba2f7a886` |
| `ctest_shipping.log` (candidate) | `c957ee162bf3e38a5e03ecc538cc901528f517e507d9f33af1efad8b20463859` |
| `cli_suites_69476d4b.log` (candidate) | `8b8bf2a6552f66ea3d1413152c45723f94783d3ac33cd3c284dbbb3182f6d718` |
| `mutation_cpu.log` (candidate) | `ff72d29891f108db911c1deae183f7ed1c28fdbe6ee45ae6fe2670759875ee71` |
| `mutation_cli.log` (candidate) | `a12328b05c0ce1e8b5174579ec788c454f3f2d28f882b4197403bebb6135e76a` |
| `coordinate_capture.log` (candidate) | `40036e54ed1ac4b5a938dfdb8e40c99608da7aaa95425d2bfa43c4c853e25b67` |
