<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 S2-2 N03 Phase 2: sealed-memfd operator load (implementation candidate)

**Status: Draft implementation candidate. Not merged.** Until it is merged and main CI passes:

- N03 MB-41 stays **OPEN** and S2-2 stays **PARTIAL**;
- aggregate `load_verification.byte_scope` stays `hashed_before_load`;
- qualified V5 stays **UNRESOLVED**;
- `distribution.status=local-only`, `owner_confirmation=null`.

The #549, #536 and #550 checkpoints are written back only after merge and main CI.

Authorization (owner, in session): P-N03 rev 3c ACCEPT, D-N03-5 SELECTED, D-N03-6 NOT SELECTED,
Phase 2 implementation authorized, merge **not** authorized. In session the owner also chose to
apply D-N03-5 strictly in both modes, with the root derived from the kernel mapping (see Design).

The design, threat model and Phase 1 research evidence are in Draft
[#583](https://github.com/raylei50653/saccade/pull/583) (`docs/reference/model_bundle_s2_2_n03_549.md`
on its branch). Base: `main@74f2f5cb1a3db8a1f42da48d533e3322f243d138`. Evidence bindings:
[model_bundle_s2_2_n03_phase2_549.evidence.json](model_bundle_s2_2_n03_phase2_549.evidence.json).
The auditor rule is recorded in [§17.9](native_runtime_resolved_config.md).

**Design revision (2026-10-12).** The owner accepted the [offline admission proposal
(P-N03 rev 3d)](model_bundle_s2_2_n03_offline_proposal_549.md) (O-N03-1..3): the ELF check moved
to admission and the runtime accepts only a compiled singleton N03 digest. This document
describes the revised implementation. The proposal also records the partial r4 verification at
`1d0ffa1e`; replay and coordinate bindings before rev 3d are historical.

## Design

**D-N03-5 (both modes).** The shipping entrypoints (`track_driver.hpp`, Gate B) name the only
legal operator library:

- The kernel's `/proc/self/maps` record of the entrypoint mapping (the main program, the first
  `dl_iterate_phdr` object) gives `<prefix>/libexec/<entrypoint>`. The legal operator library is
  `<prefix>/share/saccade/build/libsaccade_scan_torchop.so`.
- This works under direct exec and under the launcher's explicit loader. Gate A's
  `/proc/self/exe` root and the launcher/root policy are unchanged.
- An entrypoint outside `<prefix>/libexec/`, a replaced (deleted) entrypoint, or a mapped path
  that does not name the mapped inode is refused.
- `DetectorHost` refuses a plan whose operator binding, normalized lexically, is another file,
  and refuses a plan that names none. Legacy mode under the launcher therefore needs
  `--model-root <prefix>/share/saccade`, as in the installed usage. Any other model root is a
  catchable refusal; main's auditor refused a foreign model root with exit 127.
- One deliberate tightening: a model root reached through a symlink that resolves to the
  package's own `share/saccade` passed main's `realpath`-based auditor. It is now refused,
  because the comparison is lexical and the open refuses symlinks. A spelling that normalizes
  lexically to the legal path (for example `<prefix>/share/saccade/../saccade`) is accepted;
  every other alias is refused.

**Load sequence (`operator_loader.hpp`):**

1. Require the plan's SHA-256, and Gate A's size when known, to be the compiled N03 pin
   (`n03_operator_pin()`, rev 3d), before the registry, the open or `dlopen`; a self-consistent
   plan for another operator is refused here.
2. Open the legal path once with `openat2`, allowing no symlink or magic link in any component.
   Require a regular file of the pin's size, read it whole and require the pin's SHA-256. The
   runtime does not parse the ELF object.
3. Write the buffer into a memfd labeled `saccade-n03:src-fd=<S>` (`MFD_NOEXEC_SEAL`, with a
   retry only on `EINVAL`).
4. Add `F_SEAL_WRITE | GROW | SHRINK | SEAL` and read the seals back. `F_SEAL_WRITE` itself
   must be set; `F_SEAL_FUTURE_WRITE` does not count.
5. Read the memfd back whole and require the SHA-256 again.
6. `dlopen("/proc/self/fd/<M>")`, then require that the complete maps record holding the
   returned object's dynamic section is `M` (device, inode and `M`'s own `/proc` name).
7. Keep `M` open for the process lifetime. The process-wide registry reuses the handle for the
   same SHA-256 and source path (also for a request prepared while an identical one loaded)
   and refuses anything else. It is filled only after validation, with non-throwing transfers.
8. If anything fails after `dlopen` succeeded, the registry is marked poisoned and `M` stays
   open, so its number is never reused; every later request in the process is refused.

**Pin and admission (rev 3d).**

- `shipping/n03_operator_pin.json` (`aa84cccd…`, 3,517,632 bytes, policy `n03-no-origin-v1`) is
  compiled at configure time into `saccade_shipping_n03_pin`, which the loader's library
  `saccade_shipping_operator_loader` links. Configure refuses a pin that is not the tracked
  realization attestation's `op_library` (`cmake/n03_pin.cmake`). Install compares against the pin
  this build compiled in, refuses a source pin changed since the build, and requires the
  attestation to be that pin. The package checker (`check_n03_pin`) compares the packaged
  operator with the pin at the source commit.
- `saccade_track`'s POST_BUILD (`cmake/check_n03_pin.cmake`) requires the pin's digest in the
  binary, and `saccade_shipping_n03_pin_test` requires the generated pin to equal the record.
- One loader object: CPU tests link a fixture pin
  (`tests/native/fixtures/n03_operator_fixture_pin.cpp`) ahead of the production one. The
  production pin has no setter, CLI or environment input.
- The `$ORIGIN` search-path rule and its layout rules moved unchanged to
  `saccade_shipping_n03_admission` and the CPU tool `saccade_n03_admission`, which no runtime
  target links or installs. The tool reads the candidate once, requires the pin's size and
  SHA-256, applies the rule to the same buffer and prints an admission record (any read error,
  or bytes beyond the size `fstat` reported, refuses). It runs when a
  digest is accepted and in ctest; install and package compare digests only. A parser that
  accepts a hostile ELF no longer loads anything: only a reviewed, pinned digest reaches
  `dlopen`.

Steps 1–5 run in Gate B step 1, before the head and engine are read. The `dlopen` in step 6 is
Gate B step 2, so every hash refusal still happens before any operator constructor runs. There is
no pathname fallback (D-N03-3).

**Per-object scope (D-N03-2).** The report and probe `artifact_byte_scopes.op_library` is
`loaded_buffer`. Head and engine were already `loaded_buffer`. Aggregate `load_verification` is
unchanged: `verified`/`hashed_before_load`, or `failed`/null.

**Auditor.** It identifies every mapped object from the kernel's record and refuses names that
do not identify their mapping, unless the object is the verified operator memfd. The reviewed
P-N03 rev 3c rule is applied verbatim apart from comments; see §17.9.

**Callers.**

- `saccade_detector_probe` is a developer tool and is not installed. It names its plan's own
  operator; the pin and the sealed-memfd load still apply, with no exception.
- `SerialRuntime`/`DoubleBufferRuntime` built from `DetectorInputs` take
  `runtime_operator_path` from their caller (tests). An empty path is refused.
- Build-tree runs that load the operator now run from an install-shaped prefix:
  - the preflight positive control;
  - the two-root legacy comparison;
  - the parity harness (`stage_runtime_prefix`).

**Failure semantics.**

- Every host refusal is a `DetectorLoadError`: exit 2, journal `failed` with null byte scope,
  Gate A identity retained.
- Auditor refusals and SIGKILL are uncatchable: journal `running`, null `load_verification`.
- Journal v3 and report v5 schemas are unchanged.

## Evidence

Raw evidence is under `results/549_s2_2_n03_phase2/` (gitignored, local-only). The last completed
historical runs are bound to clean head `c5510fdd` (code `0e09a85d`), before the r4 fix in
`4e49864a`; they are not final evidence for current head `1d0ffa1e`. The r4 chain has completed
CPU, GPU-native ctest and CLI checks, but its seven-sequence replay chain is partial (see the
offline proposal). The retained earlier runs at `e598d2f6`, `492bb3d3` and `85086d59` also predate
later review fixes. The evidence JSON retains the historical bindings.

| Acceptance condition | Control | Result |
|:--|:--|:--|
| 1. Fixed source, verified bytes, readback | CPU `saccade_shipping_operator_loader_test`, fork-isolated cases: load; registry reuse, overlapping identical requests and refusals; rename and in-place writes after prepare and after load; wrong SHA-256 or size; symlinked file or directory; directory, FIFO, relative and missing paths; `$ORIGIN` in `DT_RUNPATH`/`DT_RPATH` and non-ELF refused, absolute RUNPATH accepted; the D-N03-5 path function. GPU: `gate_b` `foreign_operator` and `detector_s2` refusals | 53 checks, 30/30 repeated runs; GPU native ctest 5/5 |
| 2. Seals and mapping identity | CPU: `F_SEAL_FUTURE_WRITE` without `F_SEAL_WRITE` fails the seal check; descriptor reuse through `load()` is refused and poisons the process; that control fails when the post-check is removed (mutation run). GPU replay observer: the live memfd read through `/proc/<pid>/fd` | seals `0x2f` (four seals plus the exec seal); mapped device and inode = the open memfd's; memfd SHA-256 = attested N03 `aa84cccd…` (manifest mode and launcher) |
| 3. Auditor negative controls | `tests/unit/test_shipping_n03_auditor.py`: production host and auditor on a real loader (launcher and direct exec) and a forger, including a memfd sealed with `F_SEAL_FUTURE_WRITE` and a host `$ORIGIN` refusal; the production auditor's `mapping_from_fd` on boundary texts (65,535/8,191-byte records with and without newline, oversized earlier records with decoys at each continuation start, `PATH_MAX`) | 37/37. The first 20 cases against main's auditor: 13 fail (every bypass, source-forgery, post-hash race, FD and alias case) |
| 3. Legitimate objects kept | existing auditor/launcher regressions; vDSO and vendor in every positive; WSL overlayfs `libcuda.so.1` in every GPU run | 14/14 and all GPU positives |
| 4. Full production verification | GPU replay: shipping trace/no-trace, measurement double-buffer and serial, 7 sequences / 5,316 frames each; 7 negative controls; manifest mode 7 sequences; policy refusal; launcher legacy run; surface checks | raw sections EXACT (no-trace detector NOT_RUN); MOT/trace hashes equal to the #582 replay; negative controls differ in their expected sections; manifest MOT equals legacy; all reports `loaded_buffer` ×3 with aggregate `hashed_before_load` |
| 4. Load identity measured directly | replay observer on manifest mode and the launcher run | no operator path mapped; operator segments from one memfd whose bytes are N03 |
| Failure states | launcher with a foreign model root; this entrypoint under another prefix's auditor; SIGKILL during load (CLI test) | exit 2 `failed`/null scope, identity kept, D-N03-5 message; exit 127 `running`/null, `operator library mapped from another path`; SIGKILL `running`/null |
| CLI suites | preflight, completion, schedule, model-bundle, measurement surface, parity harness, bundle checks, G2, N03 auditor | 517 passed, 0 skipped (GPU lease); before the test fixes, 5 build-tree operator runs failed |

Qualification: anchor and oracle rows refuse this host (driver 617.42 ≠ 616.92), so formal
verdicts are **UNRESOLVED** and negative controls are `caught=false`. Section equality is a local
diagnostic, not qualified parity.

The launcher prefix is an evidence-only assembly of this build: the Release entrypoint, the
repository launcher, the auditor compiled from this source with its CMake flags and no RUNPATH,
the #536 tree's pinned vendor set (hard links) and its `share/saccade`. It is not an install,
package or re-pin. `build/runtime_identity` has no shipping `TORCH_CUDA_ARCH_LIST`, so
`cmake --install` refuses it (#465 PR-12).

## Independent review

Read-only Codex reviews; prompts, logs and outputs are in `results/549_s2_2_n03_phase2/review/`.
Round 1 (`85eeb2af`) found no blocker. Its findings and their resolution in `492bb3d3`:

| Severity | Finding | Resolution |
|:--|:--|:--|
| major | the rev 3c `$ORIGIN` refusal was missing | ELF search-path check before sealing; CPU fixtures and a real-loader control |
| major | after a successful `dlopen`: unchecked second maps read, unowned handle with its memfd closed, non-atomic registry publication | one checked post-check; poison and keep the memfd on failure; publish after validation |
| major | descriptor reuse not tested through `load()`; no `F_SEAL_FUTURE_WRITE` control; production auditor parser untested | added; the descriptor-reuse control fails with the post-check removed |
| minor | overlapping identical requests refused | reused |
| minor | replay observer joined the memfd to the maps by inode only | device and inode |
| minor | "successful runs unchanged" ignored a symlinked legacy model root | stated as a deliberate tightening |
| nit | the 30/30 repetition claim had no recorded log | recorded (`cpu_operator_loader_repeat30_492bb3d3.log`) |

Round 2 (`94329dcf`) found no blocker. Its review of the coordinate republication found nothing.
Its findings and their resolution:

| Severity | Finding | Resolution |
|:--|:--|:--|
| major | the ELF check read the dynamic table at `p_offset` while the loader reads it at `p_vaddr` (a split table could hide `$ORIGIN`); no terminator or segment bounds | read through `p_vaddr` with agreement, terminator and segment bounds (overflow-safe); CPU controls for a split, an unterminated table, an oversized `DT_STRSZ` and two `PT_DYNAMIC`; the split is accepted by the round 1 parser and refused now (`elf_split_attack_old_vs_new.log`) |
| minor | a throwing copy after registry publication bypassed poisoning | staged before publication |
| minor | `..` spellings described as refused | corrected |

Round 3 (`8a7307fc`) found no blocker. Its review of the republished coordinate found nothing.
Its findings and their resolution:

| Severity | Finding | Resolution |
|:--|:--|:--|
| major | overlapping `PT_LOAD` segments: the check translated through an earlier segment that the loader overwrites (`MAP_FIXED`), hiding `$ORIGIN` | ascending, page-disjoint, congruent loadable segments required; CPU overlap control; the overlap is accepted by the round 2 parser and refused now (`elf_overlap_attack_r2_vs_new.log`) |
| minor | the malformed-ELF closure was overstated | scoped to the stated layout rule; `$ORIGIN` also refused in `DT_NEEDED`/`DT_AUXILIARY`/`DT_FILTER` (real-loader `DT_NEEDED` control) |

Round 4 (`7734ee10`; the first attempt was refused by the reviewer's safety filter and is retained as
`codex_review_p2_r4.refused.log`) found no blocker. It compared the checker's model with the
loader's on the dynamic array, the string table, the expanded tags, RPATH/RUNPATH, `e_phnum`,
`p_align`, BSS and RELRO, and found no divergence other than the major below. It found no ordinary
layout refusal across 3,798 local x86-64 shared objects. Its review of the coordinate found
nothing.

| Severity | Finding | Resolution |
|:--|:--|:--|
| major | a `PT_LOAD` with `p_memsz` 0 skipped the layout rule while the translation still used it (the loader maps its file bytes) | every `PT_LOAD` is validated; zero-size load segments are refused; CPU control |
| minor | the overflow guard could wrap | end and page rounding checked in separate ordered steps; CPU control |
| minor | the `DT_NEEDED` control relied on the linker keeping an unused dependency | `--no-as-needed` |
| minor | malformed-ELF refusal still described as complete | the validated properties are listed; other ELF checks stay `dlopen`'s |

## Known limits and residuals

- **Package checkers.** `check_shipping_bundle.py` `sources` (the `LD_DEBUG` `calling init` path)
  and the strace parsers (`open`/`openat`, not `openat2`) do not understand the memfd load. A
  package built from this source would fail `sources` until the reader is adapted. That belongs
  with a package re-pin, outside this authorization; the release strategy is unchanged.
- **Availability.** A host without `openat2`, `memfd_create`, `/proc` or an executable memfd
  mapping is refused (D-N03-3); such hosts are **UNVERIFIED**. This host runs WSL2 kernel 6.18
  with `vm.memfd_noexec=0`.
- **Compatibility restrictions** (auditor, fail closed): ordinary files named `… (deleted)`,
  objects whose `l_ld` is not file-backed, and mapped paths of `PATH_MAX` bytes or more are
  refused.
- **Provenance.** It is an audit-time observation on the auditor side. D-N03-5 adds the host-side
  binding of where the source is opened. The chunked auditor comparison is not a single-instant
  snapshot; byte identity does not depend on it.
- **Runtime identity.** Historical commit `456f948e` published a fresh `--run-probe` capture of
  `0e09a85d`, moving only the implementation axis: `bc378f68…` → `671bc0a9…`, 303 → 305 files.
  Its attested check passed for that source. After the r4 fix, `1d0ffa1e` reverted that publication
  and its outgoing archive; the current source has no final coordinate publication. Earlier
  republications `91a0037c` and `001ecec4` were also reverted before any push. Their local capture
  records remain historical evidence. A new publication must follow the fresh-capture,
  independent-review and append-only-archive runbook. Probe equality does not establish equivalence.
- **Not covered:**
  - the installed package and launcher acceptance (no re-pin);
  - another GPU, driver 616.92 and V5 qualification;
  - an `abort` interruption during the load;
  - manifest mode under the launcher (Gate A's `/proc/self/exe` limitation is unchanged and out
    of scope).
