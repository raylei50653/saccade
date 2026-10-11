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
  catchable refusal. Main's auditor refused that case with exit 127, so the set of runs that
  succeed under the launcher is unchanged.

**Load sequence (`operator_loader.hpp`):**

1. Open the legal path once with `openat2`, allowing no symlink or magic link in any component.
2. Require a regular file and Gate A's size when known, read the whole file, and require the
   plan's SHA-256.
3. Write the buffer into a memfd labeled `saccade-n03:src-fd=<S>` (`MFD_NOEXEC_SEAL`, with a
   retry only on `EINVAL`).
4. Add `F_SEAL_WRITE | GROW | SHRINK | SEAL` and read the seals back. `F_SEAL_WRITE` itself
   must be set; `F_SEAL_FUTURE_WRITE` does not count.
5. Read the memfd back whole and require the SHA-256 again.
6. `dlopen("/proc/self/fd/<M>")`, then require that the complete maps record holding the
   returned object's dynamic section is `M` (device, inode and `M`'s own `/proc` name).
7. Keep `M` open for the process lifetime. The process-wide registry reuses the handle for the
   same SHA-256 and source path and refuses anything else.

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

Raw evidence is under `results/549_s2_2_n03_phase2/` (gitignored, local-only). The GPU replay is
bound to clean source `e598d2f6b07af20b5b5c1cee8d6a76468b8a7203`; hashes are in the evidence JSON.

| Acceptance condition | Control | Result |
|:--|:--|:--|
| 1. Fixed source, verified bytes, readback | CPU `saccade_shipping_operator_loader_test` (fork-isolated): load, registry reuse and refusals, rename/in-place after prepare and after load, wrong SHA/size, symlink (file and directory), directory, FIFO, relative, missing; D-N03-5 path function; GPU `gate_b` `foreign_operator` and `detector_s2` refusals | 36 checks pass; GPU native 5/5 ctest |
| 2. Seals and mapping identity | CPU: memfd mapping and post-check refusal of another memfd; GPU replay observer: live memfd via `/proc/<pid>/fd` | seals `0x2f` (four seals plus exec seal); mapped inode = open memfd inode; memfd SHA-256 = attested N03 `aa84cccd…` (manifest mode and launcher) |
| 3. Auditor negative controls | `tests/unit/test_shipping_n03_auditor.py`: production host and auditor on a real loader (launcher and direct exec) and a forger | 20/20. Against main's auditor 13 fail: every bypass, source-forgery, post-hash race, FD and alias case |
| 3. Legitimate objects kept | existing auditor/launcher regressions; vDSO and vendor in every positive; WSL overlayfs `libcuda.so.1` in every GPU run | 14/14 and all GPU positives |
| 4. Full production verification | GPU replay: shipping trace/no-trace, measurement double-buffer and serial, 7 sequences / 5,316 frames each; 7 negative controls; manifest mode 7 sequences; policy refusal; launcher legacy run; surface checks | raw sections EXACT (no-trace detector NOT_RUN); MOT/trace hashes equal to the #582 replay; negative controls differ in their expected sections; manifest MOT equals legacy; all reports `loaded_buffer` ×3 with aggregate `hashed_before_load` |
| 4. Load identity measured directly | replay observer on manifest mode and the launcher run | no operator path mapped; operator segments from one memfd whose bytes are N03 |
| Failure states | launcher with a foreign model root; this entrypoint under another prefix's auditor; SIGKILL during load (CLI test) | exit 2 `failed`/null scope, identity kept, D-N03-5 message; exit 127 `running`/null, `operator library mapped from another path`; SIGKILL `running`/null |
| CLI suites | preflight, completion, schedule, model-bundle, measurement surface, parity harness, bundle checks, G2, N03 auditor | 500 passed, 0 skipped (GPU lease); before the test fixes, 5 build-tree operator runs failed |

Qualification: anchor and oracle rows refuse this host (driver 617.42 ≠ 616.92), so formal
verdicts are **UNRESOLVED** and negative controls are `caught=false`. Section equality is a local
diagnostic, not qualified parity.

The launcher prefix is an evidence-only assembly of this build: the Release entrypoint, the
repository launcher, the auditor compiled from this source with its CMake flags and no RUNPATH,
the #536 tree's pinned vendor set (hard links) and its `share/saccade`. It is not an install,
package or re-pin. `build/runtime_identity` has no shipping `TORCH_CUDA_ARCH_LIST`, so
`cmake --install` refuses it (#465 PR-12).

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
- **Runtime identity.** The implementation axis moved. The coordinate republication travels with
  the final delivery (see the PR).
- **Not covered:**
  - the installed package and launcher acceptance (no re-pin);
  - another GPU, driver 616.92 and V5 qualification;
  - an `abort` interruption during the load;
  - manifest mode under the launcher (Gate A's `/proc/self/exe` limitation is unchanged and out
    of scope).
