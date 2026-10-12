<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 MB-41: condition check for the aggregate `loaded_buffer` (outcome: OPEN)

**Outcome: condition 2 is not met. MB-41 stays OPEN, and the aggregate
`load_verification.byte_scope` stays `hashed_before_load`.** No runtime change is landed.

The status is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2).
This page records:

- the owner's conditions;
- what was checked and how;
- the gap that keeps MB-41 open, with its evidence;
- the minimal follow-up.

## Authorization

The owner gave a conditional approval in session on 2026-10-12. It approves, in principle,
closing MB-41 and promoting the aggregate `byte_scope` to `loaded_buffer`, but only after three
conditions are confirmed:

1. Operator, head and engine cover every load file that
   [contract 4.1](../architecture/model_bundle_contract_549.md#load-verification) requires.
2. Each file's loaded bytes are the bytes whose hash was verified, and there is no unverified
   alternative load path.
3. The corresponding tests, negative controls and journal consistency checks all pass.

If any condition fails, MB-41 stays OPEN, and the owner wants the gap, the evidence and the
minimal follow-up listed. A historical owner hold is not a reason. Qualified V5, package
admission and `distribution.status` keep their states.

## How it was checked

The check used a candidate implementation on branch `feat/549-mb41-aggregate-loaded-buffer`.
It is pushed for reference but **not proposed for merge**. Its commits:

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

## Results

| Condition | Result | Evidence |
|:--|:--|:--|
| 1. Load files covered | **Met** for the three files that contract 4.1 names. In the shipping tree, `dlopen(`, `torch::jit::load(` and the `TRTEngine` constructors appear only at `detector_host.cpp` (head from a stream, engine from a pointer/size) and `operator_loader.cpp` (`/proc/self/fd/<M>`). Each runtime builds one `DetectorHost`. The pathname loaders under `src/perception/` are not reachable from shipping code. | `git grep` over `shipping/`, `src/`, `include/` |
| 2. Loaded bytes = hashed bytes, no unverified alternative path | **Not met.** The outer bytes hold: head and engine come from one immutable hashed buffer, and the operator comes from a sealed, read-back memfd whose mapping is checked. **But the head's TorchScript can read bytes that were never hashed**, both while it loads and while it runs (gap G-1 below). | `repro_from_file.log`, `repro_forward_from_file.log`, `codex_review_r1.md`. Outer-byte controls: `verified_model_bytes`, `model_bundle_gate_b`, `operator_loader` ctests and the N03 auditor suite |
| 3. Tests, negative controls, journal consistency | **Not met as a whole.** The shipping ctest, the CLI suites and the mutants pass on the candidate (below), but there is no negative control for G-1, so the controls do not cover condition 2. | Candidate runs below |

Candidate runs (local, gpu0 lease; CI runs neither GPU nor real-binary CLI tests):

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
- **The guarantee is still not met.** It must hold for every head that a manifest can bind,
  including a self-consistent substitute at `checksum_matched`.
- **Engine.** TensorRT's embedded host code needs an opt-in that this code does not enable.
- **Operator.** It is a single compiled, reviewed digest (the N03 pin). What its own code does
  at load time is fixed by that pin.

## Minimal follow-up

**F-1 (blocks MB-41): head closure in Gate B.** Gate B refuses (exit 2, `load_verification`
failed) any head whose TorchScript can read bytes outside its archive. All four repros become
negative controls in the GPU ctest and the CLI suite: `__setstate__`, `forward`, `fork` and
FlatBuffer.

A refusal after the load still meets the MB-41 row ("the loaded bytes are the hashed bytes, or
exit 2"), as long as it exits 2 with `failed`/null before any inference. It cannot undo a read
that already happened during deserialization. G-1a therefore has to be closed before the load.

What F-1 has to meet:

- **Before `torch::jit::load`:**
  - accept only the ZIP TorchScript format, and refuse FlatBuffer and anything unrecognized;
  - refuse any scripted `__setstate__` (module or class) found in the archive's code entries.
- **After the load, over every method of every submodule and class:**
  - traverse nested blocks **and** graph-valued attributes (`Subgraph`) and called functions;
  - refuse any executable path the walk cannot resolve.
- **The policy that decides which ops are allowed:**
  - **(a)** A denylist (`aten::from_file` and similar) is a **mitigation only**. It can never be
    shown complete, so on its own it does not establish condition 2. MB-41 would stay OPEN.
  - **(b)** An op allowlist establishes condition 2 only if the runtime owns it: built in and
    reviewed, like the N03 pin or the TR-1b allowlist. A list carried in the caller-controlled
    lineage or manifest could authorize `aten::from_file` itself at `checksum_matched`.

The design choice is the owner's.

**F-2 (after F-1): land the candidate.** Only after F-1 establishes condition 2 with option (b)
or an equivalent runtime-owned policy, not merely when its tests pass:

1. Rebase `69476d4b` onto F-1.
2. Recapture the coordinate.
3. Rerun the shipping ctest, the CLI suites, the mutants and pre-push.
4. Close MB-41 in the status writer.

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
