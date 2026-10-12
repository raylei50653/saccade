<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-12 -->
<!-- doc-module: cross -->

# #549 S2-2 N03 Phase 2: sealed-memfd operator load (implementation candidate)

**Merge status (2026-10-12):** merged as [#584](https://github.com/raylei50653/saccade/pull/584)
(merge `4a79e50f4484fefa4d3f270a406eac016b229412`, head `c73ff7c1`); main CI on the merge SHA
passed 8/8. The #585 hardening followed in [#586](https://github.com/raylei50653/saccade/pull/586)
(merge `793c5d56ad6be2d0f85c98112dc63ae7948f4bec`). The status after merge is written only in
[CC-536-01-02 S2-2 實作狀態](../architecture/ship_export_contracts_536.md#cc-536-01-02-status-s2-2): the per-object scope is `loaded_buffer` for all three objects. S2-2 stays **PARTIAL**. A
separate change checked the owner's conditions of 2026-10-12 and closed MB-41 at source level:
the manifest-mode aggregate is now `loaded_buffer` ([MB-41 evidence](model_bundle_mb41_549.md)). The rest of this page is the candidate record as reviewed.

**Status (as reviewed): Draft implementation candidate. Not merged.** Until it is merged and main CI passes:

- N03 MB-41 stays **OPEN** and S2-2 stays **PARTIAL**;
- aggregate `load_verification.byte_scope` stays `hashed_before_load`;
- qualified V5 stays **UNRESOLVED**;
- `distribution.status=local-only`, `owner_confirmation=null`.

The #549, #536 and #550 checkpoints are written back only after merge and main CI.

Authorization (owner, in session): P-N03 rev 3c ACCEPT, D-N03-5 SELECTED, D-N03-6 NOT SELECTED,
Phase 2 implementation authorized, merge **not** authorized. In session the owner also chose to
apply D-N03-5 strictly in both modes, with the root derived from the kernel mapping (see Design).
On 2026-10-12 the owner accepted P-N03 rev 3d (O-N03-1..3) and authorized its implementation,
local evidence, push and a Draft PR; merge is still **not** authorized.

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
  this build compiled in, refuses a source pin changed since the build, requires this build's
  `saccade_track` (when the tree takes it) to carry the pin's digest, and requires the attestation
  snapshot it copies against to be that pin. A pinned entrypoint (`SACCADE_SHIPPING_ENTRYPOINT`)
  is not checked: the current one predates the pin until a package re-pin.
  The package checker compares the packaged operator with the pin at the source commit
  (`check_n03_pin`). After #585, `read_n03_pin` permits no pin only for
  `74f2f5cb1a3db8a1f42da48d533e3322f243d138` (the last pre-rev-3d main) and its
  proven ancestors (`git merge-base --is-ancestor`). Every other commit requires
  a readable pin, regardless of its CMake text; unknown commits, unavailable
  history and shallow repositories refuse.
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

Raw evidence is under `results/549_s2_2_n03_phase2/` (gitignored, local-only). The final runs are
bound to clean head `8b020ec0` (the frozen rev 3d source; chain `verify_chain_r3d.sh`, output
`verify_chain_8b020ec0.out`). Runs at `c5510fdd`, `1d0ffa1e` (partial r4 chain), `85086d59`,
`492bb3d3` and `e598d2f6` are historical rev 3c evidence. Hashes are in the evidence JSON.

| Acceptance condition | Control | Result |
|:--|:--|:--|
| 1. Fixed source, verified bytes, readback | CPU `saccade_shipping_operator_loader_test` (the runtime's loader object with a fixture pin), fork-isolated cases: load; registry reuse, overlapping identical requests and refusals; rename and in-place writes after prepare and after load; symlinked file or directory; directory, FIFO, relative and missing paths; the D-N03-5 path function. GPU: `gate_b` `foreign_operator` and `detector_s2` refusals | 47 checks, 30/30 repeated runs; GPU native ctest 8/8 |
| 1a. Pin (rev 3d) | CPU: a self-consistent request for another operator, another known size (also before a registry reuse), bytes other than the pin's at the path, no pin set; a constructor sentinel with `load()` attempted after the refusal. Mutation: the SHA pin check disabled. Real loader: the auditor suite's unknown-digest cases. CLI: stand-in bundles (legacy and manifest mode) refused with the pin diagnostic, Gate A identity kept | all refused before anything is opened; sentinel never set; the mutation fails 3 checks (`mutation_pin_check_removed_2fbd437e.log`); `saccade_shipping_n03_pin_test` 2/2; both `saccade_track` builds carry the pin digest and no admission strings |
| 1b. Admission (rev 3d) | `saccade_shipping_n03_admission_test`: `$ORIGIN`/`${ORIGIN}` under all five tags, non-ELF, split dynamic, unterminated, oversized `DT_STRSZ`, two `PT_DYNAMIC`, overlap, zero-size, wrap; pin pairing (digest, size, policy, malformed pin). `saccade_n03_admission` on the real N03; configure/install/package pin rules (`n03_pin.cmake`, install fixtures, `check_n03_pin`) | 32 checks, 0 failures; record `n03_admission_record_8b020ec0.json` `admitted` for `aa84cccd…`; pin-rule controls in the CLI suites |
| 2. Seals and mapping identity | CPU: `F_SEAL_FUTURE_WRITE` without `F_SEAL_WRITE` fails the seal check; descriptor reuse through `load()` is refused and poisons the process (fails with the post-check removed, rev 3c mutation run). GPU replay observer: the live memfd read through `/proc/<pid>/fd` | seals `0x2f` (four seals plus the exec seal); mapped device and inode = the open memfd's; memfd SHA-256 = the pin `aa84cccd…` (manifest mode and launcher) |
| 3. Auditor negative controls | `tests/unit/test_shipping_n03_auditor.py`: production host (with a pin naming the stand-in) and auditor on a real loader (launcher and direct exec) and a forger, including a memfd sealed with `F_SEAL_FUTURE_WRITE`; the production auditor's `mapping_from_fd` on boundary texts (65,535/8,191-byte records with and without newline, oversized earlier records with decoys at each continuation start, `PATH_MAX`) | 37/37. Rev 3c run of the first 20 cases against main's auditor: 13 fail (every bypass, source-forgery, post-hash race, FD and alias case) |
| 3. Legitimate objects kept | existing auditor/launcher regressions; vDSO and vendor in every positive; WSL overlayfs `libcuda.so.1` in every GPU run | in the CLI suites; all GPU positives |
| 4. Full production verification | GPU replay `gpu_8b020ec0240d`: shipping trace/no-trace, measurement double-buffer and serial, 7 sequences / 5,316 frames each; 7 negative controls; manifest mode 7 sequences; policy refusal; launcher legacy run; surface checks | raw sections EXACT (no-trace detector NOT_RUN); MOT/trace hashes equal to the #582 replay; negative controls differ in their expected sections; manifest MOT equals legacy; all reports `loaded_buffer` ×3 with aggregate `hashed_before_load` |
| 4. Load identity measured directly | replay observer on manifest mode and the launcher run | no operator path mapped; operator segments from one memfd whose bytes are N03 |
| Failure states | launcher with a foreign model root; this entrypoint under another prefix's auditor; SIGKILL during load (CLI test) | exit 2 `failed`/null scope, identity kept, D-N03-5 message; exit 127 `running`/null, `operator library mapped from another path`; SIGKILL `running`/null |
| CLI suites | preflight, completion, schedule, model-bundle, measurement surface, parity harness, bundle checks, G2, N03 auditor, N03 pin, headline publication install, package | 617 passed, 0 skipped (GPU lease) |

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
Rounds 1–4 reviewed rev 3c; the rev 3d rounds follow them. Round 1 (`85eeb2af`) found no blocker. Its findings and their resolution in `492bb3d3`:

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

**Rev 3d.** The parser's completeness was out of scope by design (proposal §3): it moved
unchanged and no longer guards the runtime.

Round 1 (`1d0ffa1e..cbd8f60c`) found no blocker and no runtime pin bypass. Its findings and
their resolution in `8f6dbdcc` and `2fbd437e`:

| Severity | Finding | Resolution |
|:--|:--|:--|
| major | the G2 and headline-publication install fixtures lacked the new pin input | pins and attestation sizes in both fixtures |
| minor | admission could accept a prefix after a buffered read error | one descriptor: `fstat`'s size, then end of file; any error refuses |
| minor | install compared live source records, not the built runtime's pin | compared against the pin this build compiled in |
| minor | `string(JSON)` at configure under a declared CMake 3.18 | shipping CMake minimum 3.19 |
| minor | binding/package controls did not exercise the refusals; four controls passed with their check removed or moved | behavioral rule, install and `check_n03_pin` controls; exact pin diagnostics; size refusal before reuse; sentinel with `load()` attempted |

Round 2 (`cbd8f60c..2fbd437e`) found no blocker or major. Resolution in `8b020ec0`:

| Severity | Finding | Resolution |
|:--|:--|:--|
| minor | a reconfigure without a rebuild could still pair this build's `saccade_track` with another operator at install | install requires that executable, when the tree takes it, to carry the pin's digest |
| minor (new) | install validated a second read of the attestation, copied against the first | one snapshot: the rule takes the text install already read |
| minor | `manifest_head`'s and configure's calls into the pin rules were checked by a source assertion and the shared helper, not end to end | `manifest_head` now has end-to-end controls (see the PR review below); configure's call stays covered by the shared-rule controls and every configured build |

**PR review (#584, owner-shared, head `e4e93bde`).** No blocker in the runtime pin, memfd,
mapping or registry. P2: `manifest_head` read any failed `git show` of the pin as an older commit
and skipped the check (fail-open). Resolved after the freeze, outside the coordinate's axes
(`scripts/native/`): the pin is required once the commit's shipping CMake compiles it, an
unreadable or non-JSON pin refuses, with end-to-end `manifest_head` controls on a fake source commit
(a mutation restoring the fail-open read fails the unreadable-pin control). P3, tracked separately:
`install_model_root.cmake` copies the attestation from its source after validating a snapshot, a
window that predates this PR.

The source was frozen at `8b020ec0` after self-review of the round 2 diff; the final chain and the
coordinate capture ran on it. The coordinate review of `bc4e77cd` found nothing: the archive is the
outgoing canonical, the canonical is the candidate, only the implementation axis moved (nine added,
nine modified), equivalence stays unproven.

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
- **Runtime identity.** `bc4e77cd` publishes a fresh `--run-probe` capture after a complete
  Release build of `8b020ec0`, moving only the implementation axis: `bc378f68…` → `5eb66833…`,
  303 → 312 files (nine added, nine modified shipping files, relative to main's canonical). The
  outgoing canonical bytes are archived append-only
  (`runtime_identity.2026-10-12.bc378f688f9cd71b.json`) and the attested check passes. Probe
  equality does not establish equivalence. The rev 3c republications (`91a0037c`, `001ecec4`,
  `456f948e`) were reverted before any push; their capture records remain historical evidence.
- **Pinned entrypoint.** The entrypoint the package carries (`shipping/entrypoint_pin.json`,
  `fbab8b79…`) predates the pin and does not enforce it; an installed package is protected only
  after an entrypoint re-pin, which is outside this authorization.
- **Not covered:**
  - the installed package and launcher acceptance (no re-pin);
  - another GPU, driver 616.92 and V5 qualification;
  - an `abort` interruption during the load;
  - manifest mode under the launcher (Gate A's `/proc/self/exe` limitation is unchanged and out
    of scope).
