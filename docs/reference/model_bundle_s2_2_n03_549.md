<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
<!-- doc-module: cross -->

# #549 S2-2 N03 TOCTOU: threat model and loading proposal

**Status: PROPOSAL (Phase 1, revision 3b).** N03 MB-41 remains **OPEN**, S2-2 remains **PARTIAL**,
aggregate `load_verification.byte_scope` remains `hashed_before_load`, and qualified V5 remains
**UNRESOLVED**. No production loader, auditor, launcher, N03 bytes, allowlist, installed package or
pin is changed. This document asks for independent review and owner acceptance of one design.
Implementation (Phase 2) starts only after both.

Base: `main@74f2f5cb1a3db8a1f42da48d533e3322f243d138` (#582 merged). Governing contracts:
[ADR 028 D5](../decisions/028-model-bundle-runtime-separation.md),
[model-bundle contract](../architecture/model_bundle_contract_549.md),
[CC-536-01-02](../architecture/ship_export_contracts_536.md#cc-536-01-02), the
[S2-2 bounded candidate](model_bundle_s2_2_549.md) and #550. Evidence bindings:
[model_bundle_s2_2_n03_549.evidence.json](model_bundle_s2_2_n03_549.evidence.json). Prototype
sources (revision 3): [prototype auditor](model_bundle_s2_2_n03_549/loader_audit_prototype.c)
(compare with `shipping/src/loader_audit.c`), [host probe](model_bundle_s2_2_n03_549/n03_host_probe.c),
[GPU smoke](model_bundle_s2_2_n03_549/n03_gpu_smoke.cpp), parser self-tests
([cases](model_bundle_s2_2_n03_549/parser_cases.h), [auditor test](model_bundle_s2_2_n03_549/auditor_parser_selftest.c)).

## Gate 1 result

Revision 3 keeps the auditor's N03 path and source refusals at the strength main has (an
audit-time observation, see P2), closes the descriptor bypasses found so far, and makes the file
the loader maps for N03 byte-identical to the verified buffer. Each claim below is a real-loader
control in matrix r9 (45/45), GPU smoke r5 (6/6) or the parser self-tests (3 × 12/12):

- Same N03 bytes from a foreign path: still exit 127 (C02; GPU G4).
- Hardlink under a foreign name: exit 127 (C03).
- Path-mode positives and negatives behave as on main (C18/C19, plus 14/14 existing auditor and
  launcher regressions on the revised source).
- Descriptor bypasses all exit 127:
  - unlabeled, unsealed, forged-content and forged-source memfds (C11–C15);
  - deleted- or regular-file `/proc/self/fd` loads (C16/C17);
  - a relative symlink alias to a memfd (E3);
  - an alias whose `realpath` is spoofed by a planted `… (deleted)` file (E6).
- Main and revision 1 accept both alias cases (E1/E2, E4/E5).
- Rename or in-place modification after the hash: refused (C05/C06). Before the hash: a catchable
  host refusal (C07).
- In-place modification after load: no longer reaches the compared mapped bytes (C08 vs B3).
- A second operator load in the same process can no longer return an earlier object unchecked
  (F1 reproduces the revision 1 host defect; F3/F4/F5 refuse a different SHA, a foreign second
  source path and a reused descriptor; F6 reuses the verified operator for the same SHA and path).

Gate 1 is therefore **not** a stop. This is research evidence on one host, not an accepted design.
The verdict for this phase is **PARTIAL**: production implementation waits on the review and owner
decisions below.

## Independent review

Read-only Codex reviews; prompts, logs and outputs are in `results/549_s2_2_n03/review/`. Round 1
(revision 1, commit `cdf86953`) blocked acceptance:

| # | Severity | Finding | Revision 2 |
|:--|:--|:--|:--|
| 1 | blocker | A relative alias to a memfd, or a planted `… (deleted)` path, evades the descriptor rule | Auditor identifies the mapped object from `/proc/self/maps` (E3, E6) |
| 2 | major | `d_path` at audit time does not prove the path used at open | I2 narrowed; residual stated below |
| 3 | major | A reused descriptor number lets glibc return an earlier operator without `la_objopen` | Host registry, memfd kept open, post-`dlopen` mapping check (F1–F5) |
| 4 | major | A chunked compare is not a single-instant snapshot | I2 narrowed; I1 does not depend on it |
| 5 | minor | Memory comparison overstated; G3 has none | Wording fixed |
| 6 | minor | I4 and dependency wording too strong; `$ORIGIN` refusal optional | Wording fixed; `$ORIGIN` refusal required in Phase 2 |
| 7 | minor | Policy denial is not always a catchable exit 2 | Failure semantics split |

Round 2 (revision 2, commit `4c7c0eba`) found no new byte-equality bypass and held acceptance on
wording:

| # | Severity | Finding | Revision 3 |
|:--|:--|:--|:--|
| A | major | P2 promised historical provenance that I2 cannot establish; registry reuse accepted a foreign subsequent binding (F4) | P2 restated as the audit-time observation; registry keyed by SHA-256 **and** source path (F4 refuses, F6 reuses); residual needs explicit owner acceptance (D-N03-6) |
| B | minor | `/proc/self/maps` parser could parse an overlong line's continuation as a record | auditor discards overlong records through their newline (not exercised) |
| C | minor | ordinary files named `… (deleted)` and objects whose `l_ld` is not file-backed are refused | stated as compatibility restrictions; E8 control; frozen N03's `PT_DYNAMIC` is file-backed |
| D | minor | written and GPU post-check compared only the inode | device and inode for the memfd (auditor, host, GPU smoke r3) |

Round 3 (revision 3, commit `1d8d9abe`) found no blocker or major issue and would accept revision 3
as the Phase 2 design if: complete-record parsing in all three identity parsers becomes a Phase 2
requirement with tests; the residuals below are accepted, or D-N03-5 is selected; and the
compatibility restrictions are accepted. Its two minor findings:

| # | Finding | Revision 3a |
|:--|:--|:--|
| R3-1 | host post-check parsers (`fgets`, 8 KiB) parse an overlong line's continuation as a record | probe and GPU smoke parse complete records only (matrix r8 45/45, GPU smoke r4 6/6); oversized-record tests required in Phase 2 |
| R3-2 | the claim that `PATH_MAX` bounds kernel-rendered paths is unfounded | removed |

Round 4 (revision 3a, commit `7d1d04d1`; the first attempt stalled on an open stdin and is retained
as `codex_review_r4.stalled_stdin.log`) found no blocker or major issue and cleared revision 3a for
owner review, with two minor items:

| # | Finding | Revision 3b |
|:--|:--|:--|
| R4-1 | the 3a host parsers falsely refuse a final record of exactly 8191 bytes without a newline | host and GPU smoke parsers use `getline` (no fixed-buffer boundary); the auditor parser reads from a descriptor so it can be tested; all three pass 12 synthetic boundary cases each |
| R4-2 | Phase 2 parser list differs between the proposal and the evidence JSON | unified: auditor, host post-check, GPU smoke (research), and any report reader |

The self-tests also exposed one restriction. The auditor refuses an object whose mapped path is at
least `PATH_MAX` (4096) bytes: path length 4096 is refused, 4095 is accepted. It cannot identify
such an object, and `realpath` cannot resolve its name either. This is listed under the known
limits.

## Current path and observed gaps

On main, `DetectorHost` runs `check_sha(op_path)`, then `dlopen(op_path, RTLD_NOW | RTLD_LOCAL)`
([detector_host.cpp](../../shipping/src/detector_host.cpp)). The auditor's
[`la_objopen`](../../shipping/src/loader_audit.c) resolves the mapped *name* with `realpath`. It
classifies an object as N03 by mapped or real basename, or by SONAME family, and requires
`real == <prefix>/share/saccade/build/libsaccade_scan_torchop.so`. An unresolvable name passes
whenever neither the bundled nor the operator rule matches its own text.

The probe reproduces these gaps under the untouched installed auditor `1ee968ab…`. Exit 3 means the
compared mapped bytes differ from the verified buffer.

| Case | Exit | Observation |
|:--|--:|:--|
| A1 pinned path | 0 | loads (#582 reproduced) |
| A2 same bytes, foreign path | 127 | `operator library mapped from another path` (#582) |
| A3 unlabeled sealed memfd | 0 | `realpath` unresolved; classification bypassed (#582) |
| B1 rename-replace after hash | 3 | auditor passes; mapped inode is the replacement |
| B2 in-place write after hash | 3 | auditor passes; loaded bytes ≠ verified |
| B3 in-place write after load | 3 (after) | clean `MAP_PRIVATE` pages follow the page cache |
| B4 `/proc/self/fd/N` of a deleted N03-named copy | 0 | **existing bypass, newly observed**: name `N` |
| E1 relative alias `./alias.so` → memfd | 0 | **existing bypass**: relative name does not resolve |
| E4 alias → deleted file, planted `… (deleted)` | 0 | **existing bypass**: `realpath` names the planted file |

## Threat model

Protected properties:

- **P1 (byte equality).** The file the dynamic loader maps for N03 holds the bytes whose SHA-256
  matched the plan (D5).
- **P2 (provenance and ownership, as main enforces it).** When the auditor runs, the N03 object
  is the runtime package's file at `<prefix>/share/saccade/build/libsaccade_scan_torchop.so`. A
  binding to the model bundle or another path is refused (same bytes included) unless that inode
  is, at audit time, linked at the fixed path. Main's `realpath` check is the same kind of
  audit-time observation. Neither main nor this proposal establishes the path through which a
  file was originally opened; this residual needs explicit owner acceptance (D-N03-6), or D-N03-5
  enforces the binding on the host side.
- **P3.** Existing vendor, Python-family, readiness and RUNPATH rules stay unchanged.

Adversaries in scope:

- A local writer who can replace, rename, hardlink or modify N03 or its directory around the load.
- Hostile or misconfigured manifest or legacy bindings that point at a foreign operator.
- Descriptor-backed objects (memfd, deleted or `O_TMPFILE` files, `/proc/self/fd`, `/dev/fd`,
  aliases to them) used to avoid path classification.

Out of scope and unchanged:

- Code already executing in the process, including concurrent threads closing or `dup2`ing
  descriptors. Such code can map arbitrary executable memory anyway.
- Kernel, root and ptrace compromise.
- A malicious auditor file at the prefix.
- Sequence-input TOCTOU.
- Name-based classification: N03 bytes renamed to another basename and loaded by path are not
  operator-classified, on main and in this proposal alike.

## Proposal P-N03 (revision 3)

**Host (Gate B; replaces `check_sha` and `dlopen` for N03 only):**

1. **Process-wide registry.** If an operator was already verified and loaded in this process,
   reuse its handle only when the plan's SHA-256 **and** the resolved source path are the same as
   the first load's. Refuse otherwise. (On main a second path `dlopen` of the same path also
   returns the first object by name.)
2. Open the resolved operator path once (`O_RDONLY | O_CLOEXEC`), giving the source descriptor `S`.
   Require a regular file and the manifest size when available. Read the whole file (EOF checked)
   into an owned buffer and require its SHA-256 to equal the plan's.
3. Create `M = memfd_create("saccade-n03:src-fd=<S>", MFD_CLOEXEC | MFD_ALLOW_SEALING |
   MFD_NOEXEC_SEAL)`. Retry without `MFD_NOEXEC_SEAL` only on `EINVAL` (a kernel before 6.3).
   Write the buffer, then add and read back `F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK |
   F_SEAL_SEAL`.
4. Read `M` back completely and require its size and SHA-256 to equal the plan's.
5. Call `dlopen("/proc/self/fd/<M>", RTLD_NOW | RTLD_LOCAL)` while `S` and `M` are open, then:
   - require that the complete `/proc/self/maps` record (read whole, any length) containing the
     returned object's `l_ld` has `M`'s device and inode;
   - close `S`, keep `M` open for the process lifetime, and record the SHA-256 and handle.
6. There is no fallback to pathname loading. Every step that returns an error is a
   `DetectorLoadError`.

**Auditor (`la_objopen`):**

1. **Identify the mapped object.** Read the `/proc/self/maps` entry that contains `l_ld`: path,
   device and inode. Records longer than the 64 KiB buffer are discarded whole. No entry → refuse. `[vdso]` is the vDSO; it passes unless its name matches the
   Python, bundled or operator rules (main's exception, now identified by the kernel).
2. **Names that cannot be trusted.** Each of the following goes to the operator memfd check
   (step 4) and is refused unless it passes:
   - a name under `/proc/` or `/dev/fd/`;
   - a name (relative or absolute) whose `realpath` fails;
   - a name whose real path differs from the kernel's mapped path;
   - a mapped path carrying ` (deleted)`;
   - an inode mismatch.

   The device is not compared: on overlayfs `/proc/self/maps` reports the lower device while
   `stat` reports the overlay's (r4: WSL `/usr/lib/wsl/lib/libcuda.so.1`).
3. **Path-classified objects.** Everything else is classified exactly as on main (basename and
   SONAME family, vendor directory, fixed operator path, Python family).
4. **Operator memfd check.** All of the following must hold:
   - the name is exactly `/proc/self/fd/<M>`;
   - `M`'s readlink is `/memfd:saccade-n03:src-fd=<S> (deleted)` with strict decimal `S ≠ M`, and
     it equals the kernel's mapped path with the same device and inode;
   - `M` carries the four seals;
   - `S`'s readlink (kernel `d_path`) is the fixed operator path, and `S` is a regular file with the
     same device and inode as the file the auditor opens there (`O_NOFOLLOW`);
   - `M` has the same size and the same bytes as that file, compared in 64 KiB chunks.
5. Any failure is the existing `fail()`: a message, then `_exit(127)`. The auditor uses libc only
   and needs no hash code.

**Invariants (worded to what the checks establish):**

- **I1 (loader input bytes).** The file the loader maps for N03 is `M`, a sealed memfd whose
  complete bytes had the planned SHA-256 after sealing (host steps 3–5). Seals forbid later
  changes. Relocation, `.bss` zero-fill and RELRO writes then change memory as for any loaded
  object. The direct memory observation is narrower: C01/C08 compare 170,457 bytes of the three
  non-writable `PT_LOAD` segments outside `PT_GNU_RELRO`.
- **I2 (source observation).** At audit time:
  - `S`'s inode is linked at the fixed operator path and is the inode at that path;
  - each 64 KiB chunk of that file equalled `M` when the auditor read it.

  It does **not** establish the path used when `S` was opened: a writer to the fixed directory can
  rename a foreign-opened inode onto the fixed path (review finding 2). That is the same capability
  as writing the fixed path directly, which main also accepts, and the bytes must still match the
  plan. It also does **not** establish single-instant equality with a mutable file (finding 4). I1
  does not depend on I2.
- **I3.** No object whose name does not identify its mapping (descriptor names, unresolved or
  spoofed names) passes unless it is the operator memfd satisfying I1 and I2.
- **I4.** Objects whose name identifies their mapping follow main's classification. New refusals
  apply to objects main accepted when their name does not identify the mapping (B4/B5 vs C16/C17,
  E1/E4 vs E3/E6).
- **I5.** A process holds at most one verified operator, for one SHA-256 and one source path. A
  later request with another SHA-256 or path is refused (F3/F4), and a `dlopen` that returns
  another object is refused by the host post-check (F5).

## Alternatives

| Alternative | Byte equality | Provenance | Disposition |
|:--|:--|:--|:--|
| Status quo: hash, then path `dlopen` | no (B1–B3) | path only; descriptor bypasses B4/E1/E4 | OPEN residual |
| #582 unlabeled sealed memfd | yes | **bypassed** (A3) | rejected |
| Revision 1 (name-based descriptor rule, memfd closed after load) | second load can return an earlier object (F1) | alias bypasses (E2/E5) | superseded |
| Revision 2 (registry by SHA-256 only, inode-only post-check) | yes | registry reuse accepted a foreign second path | superseded |
| Pin the inode: `dlopen /proc/self/fd/<S>` of the regular file | no: in-place writes still reach the mapping | main accepts even a deleted copy (B4) | rejected |
| Auditor compares the memfd with the fixed path only, no source descriptor | yes | **weakened**: a same-byte foreign source would pass | rejected |
| Auditor substitutes the name in `la_objsearch` and copies the file itself | needs the expected hash in the auditor, or a check after constructors run | auditor-side | rejected |
| Re-hash after `dlopen` | no: N03 constructors run inside `dlopen` | — | rejected |
| Host checks the source path against the fixed runtime path before opening | — | addresses finding 2 only for opens; needs the auditor's prefix in the host (explicit-loader root limitation, #582) | option for Phase 2 (D-N03-5) |
| fs-verity, immutable attribute, read-only mount | yes, if the deployment provides it | path | later hardening; needs root or filesystem support |
| Link N03 into the entrypoint | yes | ownership changes | out of scope (operator bytes change) |
| **P-N03 revision 3b** | **I1** | **P2 as main (audit-time), I2, I3** | **proposed** |

## Failure semantics and contract matrix

**Catchable** (`DetectorLoadError` → exit 2, run/load verification `failed` with null byte scope,
Gate A identity retained):

- open, regular-file, size, read or hash failures;
- `memfd_create`, write, seal or readback failures that return an error;
- `dlopen` returning NULL;
- post-check or registry refusal.

**Uncatchable** (journal stays `running` with null load verification):

- every auditor refusal (`_exit(127)`), including policy-caused failures inside the auditor's own
  `/proc` reads;
- SIGKILL;
- a seccomp policy that kills instead of returning an errno.

| Requirement | P-N03 behavior | Evidence (this phase) | Phase 2 must show |
|:--|:--|:--|:--|
| Loaded bytes = verified bytes (D5, MB-41 N03) | sealed memfd + readback hash; mapping identity post-check | I1: C01/C08 (170,457 compared bytes), F1–F6; B1–B3 contrast | in `DetectorHost`, under the real CLI |
| N03 path/source auditing kept | `d_path` + device/inode + byte equality to the fixed path | C02–C06, C14, C15, G4 | auditor pytest cases mirroring C/E series |
| Memfd/FD/alias paths cannot bypass | mapped-object identity; descriptor names fail closed | C11–C13, C16, C17, E3, E6 (E8: compatibility restriction) | same, plus installed launcher |
| Path mode | main classification for identified objects | C18, C19, 14/14 regressions; WSL overlay libcuda passes | full existing suite |
| N03 RUNPATH exception and dependencies | RUNPATH absolute, no `$ORIGIN` | D1/D2: same search-path entries and same 30 mapped object paths (11 vendor); only the RUNPATH attribution label and init name change to `/proc/self/fd/<M>` | required refusal of an N03 with `$ORIGIN`; entrypoint closure |
| Catchable vs uncatchable | split as above | C07, C09, C10, F3–F5 (exit 2); C- and E-series 127 | native CLI MB-58 controls, SIGKILL |
| Gate A identity | unchanged; N03 stays in the runtime package | not touched | identity unchanged in journal |
| Real GPU operator | registered op runs; float32/float16 outputs bit-equal across load modes; graph capture + 3 replays = eager | GPU smoke r5 G1–G3, G5, G6 (output hashes, no memory comparison) | full detector/graph/serial/double-buffer replays |
| Report scope | aggregate stays `hashed_before_load` | — | per-object N03 label per owner decision D-N03-2 |

## Experiments

Raw evidence is under `results/549_s2_2_n03/` (gitignored, local-only); the evidence JSON lists the
`SHA256SUMS` digest and every source, binary and summary hash.

**Setup.** Three evidence-only prefixes:

- `prefix_installed`: the untouched installed auditor;
- `prefix_proto_r1`: the revision 1 prototype;
- `prefix_proto`: the revision 3 prototype.

Each has a hardlinked `lib/vendor` from the #536 evidence tree and a fresh N03 copy per case. Every
run uses `/lib64/ld-linux-x86-64.so.2 --library-path <prefix>/lib/vendor --audit <auditor>` with
`PATH=/usr/bin:/bin`, as the launcher does. A shell hook at a fixed stage (`after_open`,
`after_hash`, `after_load`) places mutations inside the race window deterministically.

**Matrix r9** (45/45; inputs unchanged in every case; r7 and r8 with earlier probes also 45/45).
Revision 3b rows:

| Case | Exit | Auditor or host message |
|:--|--:|:--|
| C01 proposed, pinned | 0 | readback SHA = plan, seals `0x2f`, post-check passes; compared segments equal |
| C02 same bytes, foreign path | 127 | `operator library mapped from another path` |
| C03 hardlink under a foreign name | 127 | `operator library mapped from another path` |
| C04 symlink alias resolving to the fixed path | 0 | same acceptance as main |
| C05 rename-replace after hash | 127 | source `d_path` ends `(deleted)` |
| C06 in-place write after hash | 127 | `operator memfd differs from the operator library` |
| C07 in-place write after open, before read | 2 | host SHA-256 mismatch |
| C08 in-place write after load | 0 | compared segments still equal |
| C09 wrong SHA / C10 directory | 2 / 2 | host refusal |
| C11/C12 unlabeled memfd | 127 | `descriptor-backed object is not the operator memfd` |
| C13 without `F_SEAL_WRITE` | 127 | `operator memfd is not sealed` |
| C14 label names the fixed source, bytes mutated | 127 | `operator memfd differs …` |
| C15 label names a closed descriptor | 127 | `operator source descriptor is not open` |
| C16 deleted copy / C17 regular file via `/proc/self/fd` | 127 / 127 | `descriptor-backed object is not the operator memfd` |
| C18/C19 path mode, pinned/foreign | 0 / 127 | as main |
| C20 `MFD_EXEC` variant | 0 | loads |
| E3 relative alias → memfd | 127 | `mapped object is not the object its name resolves to` |
| E6 alias → deleted file, planted `… (deleted)` | 127 | same |
| F1 second load, revision 1 host | 3 | same handle returned; second plan's bytes not loaded |
| F2 second load, memfd kept open, no registry | 127 | new load, foreign source refused by the auditor |
| F3 second load, different SHA, registry | 2 | `operator already loaded with different bytes` |
| F4 second load, same SHA, foreign path, registry | 2 | `operator already loaded from another source path` |
| F5 second load, descriptor number reused, post-check | 2 | `loaded object is not the verified memfd` |
| F6 second load, same SHA, same path, registry | 0 | verified operator reused |
| E8 ordinary file named `libplugin.so (deleted)`, path mode | 127 | compatibility restriction (E7: main accepts) |

**GPU smoke r5** (6/6; r3 and r4 also 6/6) runs under `tools/resctl.py run --wait gpu0` on an RTX 5070 Ti Laptop,
sm_120, driver 617.42. G1 (installed, path), G2 (revision 3, path), G3 (revision 3, proposed,
device+inode post-check), G5 (installed, proposed) and G6 (revision 1, proposed) all produce:

- eager float32 output `03f283f1…0be9` and float16 output `7a5a3331…bc2b`;
- three CUDA graph replays equal to eager.

G4 (proposed, foreign same-byte source) exits 127. This covers the operator alone, not detector,
MOT or qualified parity.

**Retained failed predictions:**

- **r1 C05.** The refusal came from the path rule, not the predicted device/inode rule.
- **r2 C20.** A `MFD_NOEXEC_SEAL` memfd loads (the flag blocks `execve`, not `mmap(PROT_EXEC)`),
  so the design uses it.
- **r4.** The first revision 2 auditor compared devices and refused WSL's overlayfs
  `libcuda.so.1` (11/41 failures), so devices are no longer compared.
- **r5 F2.** The prediction was a host post-check refusal; the auditor refused first, because a
  kept-open `M` makes the second load genuine. F5 isolates the post-check.

r6/GPU r2 (revision 2) passed 42/42 and 6/6 and are retained, as are r7/r8 and GPU r3/r4
(revisions 3 and 3a). The revision 3b auditor binary is byte-identical to revision 3
(`6d0ded18…`).

**Parser self-tests** (`results/549_s2_2_n03/parser_selftest_r1/`), 12 synthetic cases for each of
the auditor, host post-check and GPU smoke parsers, all passing:

- short records with and without a final newline;
- path lengths 4095 and 4096;
- records of 8191, 8192 and 65535 bytes without a newline;
- a 70,000-byte target record;
- 8191- and 70,000-byte earlier records with header-shaped decoy text at chunk boundaries;
- no target.

The decoys are never parsed as records. The auditor identifies targets up to a path length of
4095 and refuses longer ones; the `getline` parsers identify every length. Runner and probe bytes of
r1/r2/r4/r5 are partly not retained; their summaries record the probe and auditor hashes.

## Known limits

- **New availability requirement.** Hosts without `memfd_create`, `/proc`, or executable memfd
  mappings are refused. Such hosts (seccomp, SELinux, AppArmor, `vm.memfd_noexec=2`) are
  **UNVERIFIED**; this host has `vm.memfd_noexec=0`. The refusal is catchable only when the
  failing call returns an error.
- The auditor now reads `/proc/self/maps` for every object. Compatibility restrictions (fail
  closed):
  - a mapped path with characters the kernel escapes does not equal its `realpath`;
  - an ordinary file whose name ends in ` (deleted)` is refused (E8);
  - an object whose `l_ld` is not in a file-backed mapping is refused (untested; the frozen N03's
    `PT_DYNAMIC` is file-backed);
  - an object whose mapped path is at least `PATH_MAX` bytes is refused (parser self-test).
- Ordinary path-classified objects are identified by mapped path plus inode, without the device
  (overlayfs). Path plus inode is not a universal identifier across filesystems; no in-scope
  bypass is demonstrated. The operator memfd is identified by device plus inode.
- `link_map` and `dladdr` names for N03 become `/proc/self/fd/<M>`, and `/proc/self/maps` shows
  `/memfd:saccade-n03:src-fd=<S> (deleted)`. Readers that match N03 by its path (LD_DEBUG evidence
  readers, report fields that echo the mapped name) must adapt.
- Residuals of P2/I2 (round 1 findings 2 and 4, round 2 finding A) stay as stated. D-N03-5 can
  narrow finding 2 for opens; D-N03-6 records owner acceptance otherwise.
- The prototype auditor is a research build. Phase 2 changes `shipping/src/loader_audit.c` and
  `detector_host.cpp`, so the installed package becomes **UNVERIFIED** until a separately
  authorized re-pin, and runtime identity coordinates move.

## Phase 2 scope (only after acceptance)

- Change the N03 load in `DetectorHost` (one helper plus a process-wide registry, with a CPU unit
  test) and `loader_audit.c` as specified above.
- Keep Gate A, journal v3/report v5, the engine/head buffers and compatibility gates unchanged.
- Refuse an operator whose RUNPATH or RPATH contains `$ORIGIN` (required, since a memfd load would
  change its expansion).
- Tests:
  - auditor pytest real-loader cases mirroring C02–C17, E3, E6, E8 and the vDSO/overlay positives;
  - the parser self-test cases for the auditor and host post-check parsers (and the GPU smoke and
    any report reader that parses `/proc/self/maps`), including 8191-byte boundaries,
    oversized earlier and target records, and the `PATH_MAX` boundary;
  - native Gate B tests for host refusals, the registry (F3/F4/F6) and the post-check (F5);
  - native CLI controls for catchable exit 2/failed and uncatchable 127/SIGKILL running/null;
  - a GPU manifest and legacy positive;
  - serial, double-buffer, graph and seven-sequence replays;
  - full pre-push and CI.

## Owner decisions requested

- **D-N03-1.** Accept P-N03 revision 3b (host sequence, registry, post-check and auditor rules) as
  the Phase 2 design.
- **D-N03-2.** Per-object N03 scope after Phase 2 evidence. Recommendation: `loaded_buffer` (the
  loader maps the sealed copy of the verified buffer). The aggregate stays `hashed_before_load`
  until separately authorized.
- **D-N03-3.** Hosts without `memfd_create`, `/proc` or executable memfd. Recommendation: refuse,
  with no pathname fallback.
- **D-N03-4.** Accept the auditor tightening that refuses every object whose name does not
  identify its mapping, other than the verified operator memfd.
- **D-N03-5.** Optionally have the host refuse a source path that is not the runtime package's
  fixed operator path. This is defense in depth for finding 2; it depends on runtime-root
  derivation.
- **D-N03-6.** Without D-N03-5, accept the residual that provenance is an audit-time observation
  (as on main), not a record of the path used at open.

Not included: S2-3, allowlist approvals, model or weight or N03 byte changes, re-pin, #547
licensing, public release or signing, V5 qualification or benchmarks, and launcher/root policy.
`distribution.status=local-only`, `owner_confirmation=null`.
