<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
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
- One deliberate tightening: a model root reached through a symlink (or a `..` spelling) that
  resolves to the package's own `share/saccade` passed main's `realpath`-based auditor. It is now
  refused, because the comparison is lexical and the open refuses symlinks.

**Load sequence (`operator_loader.hpp`):**

1. Open the legal path once with `openat2`, allowing no symlink or magic link in any component.
2. Require a regular file and Gate A's size when known, read the whole file, and require the
   plan's SHA-256. Require a 64-bit ELF whose `DT_RPATH`/`DT_RUNPATH` do not use `$ORIGIN`
   (it would expand under `/proc/self/fd` when loaded from the memfd); a non-ELF object is
   refused here, before anything is sealed or `dlopen`'ed.
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
  operator; the sealed-memfd load still applies.
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
bound to clean source `492bb3d3`; the retained earlier run at `e598d2f6` predates the review round 1
fixes. Hashes are in the evidence JSON.

| Acceptance condition | Control | Result |
|:--|:--|:--|
| 1. Fixed source, verified bytes, readback | CPU `saccade_shipping_operator_loader_test`, fork-isolated cases: load; registry reuse, overlapping identical requests and refusals; rename and in-place writes after prepare and after load; wrong SHA-256 or size; symlinked file or directory; directory, FIFO, relative and missing paths; `$ORIGIN` in `DT_RUNPATH`/`DT_RPATH` and non-ELF refused, absolute RUNPATH accepted; the D-N03-5 path function. GPU: `gate_b` `foreign_operator` and `detector_s2` refusals | 48 checks, 30/30 repeated runs; GPU native ctest 5/5 |
| 2. Seals and mapping identity | CPU: `F_SEAL_FUTURE_WRITE` without `F_SEAL_WRITE` fails the seal check; descriptor reuse through `load()` is refused and poisons the process; that control fails when the post-check is removed (mutation run). GPU replay observer: the live memfd read through `/proc/<pid>/fd` | seals `0x2f` (four seals plus the exec seal); mapped device and inode = the open memfd's; memfd SHA-256 = attested N03 `aa84cccd…` (manifest mode and launcher) |
| 3. Auditor negative controls | `tests/unit/test_shipping_n03_auditor.py`: production host and auditor on a real loader (launcher and direct exec) and a forger, including a memfd sealed with `F_SEAL_FUTURE_WRITE` and a host `$ORIGIN` refusal; the production auditor's `mapping_from_fd` on boundary texts (65,535/8,191-byte records with and without newline, oversized earlier records with decoys at each continuation start, `PATH_MAX`) | 36/36. The first 20 cases against main's auditor: 13 fail (every bypass, source-forgery, post-hash race, FD and alias case) |
| 3. Legitimate objects kept | existing auditor/launcher regressions; vDSO and vendor in every positive; WSL overlayfs `libcuda.so.1` in every GPU run | 14/14 and all GPU positives |
| 4. Full production verification | GPU replay: shipping trace/no-trace, measurement double-buffer and serial, 7 sequences / 5,316 frames each; 7 negative controls; manifest mode 7 sequences; policy refusal; launcher legacy run; surface checks | raw sections EXACT (no-trace detector NOT_RUN); MOT/trace hashes equal to the #582 replay; negative controls differ in their expected sections; manifest MOT equals legacy; all reports `loaded_buffer` ×3 with aggregate `hashed_before_load` |
| 4. Load identity measured directly | replay observer on manifest mode and the launcher run | no operator path mapped; operator segments from one memfd whose bytes are N03 |
| Failure states | launcher with a foreign model root; this entrypoint under another prefix's auditor; SIGKILL during load (CLI test) | exit 2 `failed`/null scope, identity kept, D-N03-5 message; exit 127 `running`/null, `operator library mapped from another path`; SIGKILL `running`/null |
| CLI suites | preflight, completion, schedule, model-bundle, measurement surface, parity harness, bundle checks, G2, N03 auditor | 516 passed, 0 skipped (GPU lease); before the test fixes, 5 build-tree operator runs failed |

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
- **Runtime identity.** The coordinate is republished in this delivery. Only the implementation
  axis changed: `bc378f68…` → `02f4dd97…`, 303 → 305 files (the two new loader files and eight
  modified shipping files). It was a fresh `--run-probe` capture of `492bb3d3`; the outgoing
  canonical bytes are archived append-only, and the attested check passes. Probe equality does
  not establish equivalence.
- **Not covered:**
  - the installed package and launcher acceptance (no re-pin);
  - another GPU, driver 616.92 and V5 qualification;
  - an `abort` interruption during the load;
  - manifest mode under the launcher (Gate A's `/proc/self/exe` limitation is unchanged and out
    of scope).
