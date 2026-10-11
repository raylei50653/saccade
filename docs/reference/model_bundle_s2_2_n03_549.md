<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-11 -->
<!-- doc-module: cross -->

# #549 S2-2 N03 TOCTOU: threat model and loading proposal

**Status: PROPOSAL (Phase 1).** N03 MB-41 remains **OPEN**, S2-2 remains **PARTIAL**, aggregate
`load_verification.byte_scope` remains `hashed_before_load`, and qualified V5 remains **UNRESOLVED**.
No production loader, auditor, launcher, N03 bytes, allowlist, installed package or pin is changed.
This document asks for independent review and owner acceptance of one design. Implementation
(Phase 2) starts only after both.

Base: `main@74f2f5cb1a3db8a1f42da48d533e3322f243d138` (#582 merged). Governing contracts:
[ADR 028 D5](../decisions/028-model-bundle-runtime-separation.md),
[model-bundle contract](../architecture/model_bundle_contract_549.md),
[CC-536-01-02](../architecture/ship_export_contracts_536.md#cc-536-01-02), the
[S2-2 bounded candidate](model_bundle_s2_2_549.md) and #550. Evidence bindings:
[model_bundle_s2_2_n03_549.evidence.json](model_bundle_s2_2_n03_549.evidence.json). Prototype
sources: [prototype auditor](model_bundle_s2_2_n03_549/loader_audit_prototype.c) (compare with `shipping/src/loader_audit.c`),
[host probe](model_bundle_s2_2_n03_549/n03_host_probe.c),
[GPU smoke](model_bundle_s2_2_n03_549/n03_gpu_smoke.cpp).

## Gate 1 result

The prototype keeps the auditor's N03 path and source rule, closes descriptor bypasses, and binds
the mapped bytes to the verified bytes. Each claim is backed by a real-loader control:

- Same N03 bytes from a foreign path: still exit 127 (C02; GPU G4).
- Hardlink under a foreign name: exit 127 (C03).
- Path-mode positives and negatives: unchanged (C18/C19, plus 14/14 existing auditor and
  launcher regressions).
- Unlabeled memfd, unsealed memfd, forged content, forged source descriptor, and deleted- or
  regular-file `/proc/self/fd` loads: all exit 127 (C11–C17).
- Rename or in-place modification after the hash: refused (C05/C06); before the hash: a
  catchable host refusal (C07).
- In-place modification after load: no longer reaches mapped memory (C08 vs B3).

Gate 1 is therefore **not** a stop. This is research evidence on one host, not an accepted
design. The verdict for this phase is **PARTIAL**: production implementation is gated on the review
and owner decisions below.

## Current path and observed gaps

On main, `DetectorHost` runs `check_sha(op_path)`, then `dlopen(op_path, RTLD_NOW | RTLD_LOCAL)`
([detector_host.cpp](../../shipping/src/detector_host.cpp)). The auditor's
[`la_objopen`](../../shipping/src/loader_audit.c) resolves the mapped name with `realpath`.
It classifies an object as N03 by mapped or real basename, or by SONAME family, and requires
`real == <prefix>/share/saccade/build/libsaccade_scan_torchop.so`. An unresolvable name passes
whenever neither the bundled nor the operator rule matches its own text.

The probe reproduces these gaps under the untouched installed auditor `1ee968ab…`. Exit 3 means
the loaded non-writable `PT_LOAD` bytes differ from the verified buffer.

| Case | Exit | Observation |
|:--|--:|:--|
| A1 pinned path | 0 | loads (#582 reproduced) |
| A2 same bytes, foreign path | 127 | `operator library mapped from another path` (#582) |
| A3 unlabeled sealed memfd | 0 | `realpath` unresolved; classification bypassed (#582) |
| B1 rename-replace after hash | 3 | auditor passes; mapped inode is the replacement |
| B2 in-place write after hash | 3 | auditor passes; loaded bytes ≠ verified |
| B3 in-place write after load | 3 (after) | clean `MAP_PRIVATE` pages follow the page cache |
| B4 `/proc/self/fd/N` of a deleted N03-named copy | 0 | **existing bypass newly observed**: unresolvable name, basename `N` |

## Threat model

Protected properties:

- **P1 (byte equality).** The bytes that the dynamic loader maps for N03 are the bytes whose
  SHA-256 matched the plan (D5).
- **P2 (provenance and ownership).** N03 is the runtime package's object at
  `<prefix>/share/saccade/build/libsaccade_scan_torchop.so`. It never comes from the model bundle
  or another path, even with identical bytes.
- **P3.** Existing vendor, Python-family, readiness and RUNPATH rules stay unchanged.

Adversaries in scope:

- A local writer who can replace, rename, hardlink or modify N03 or its directory around the load.
- Hostile or misconfigured manifest or legacy bindings that point at a foreign operator.
- Descriptor-backed objects (memfd, deleted or `O_TMPFILE` files, `/proc/self/fd`, `/dev/fd`) used
  to avoid path classification.

Out of scope and unchanged:

- Code already executing in the process. The host and its loaded libraries are trusted; such code
  can map arbitrary executable memory anyway.
- Kernel, root and ptrace compromise.
- A malicious auditor file at the prefix. The launcher's readiness probe covers only a missing or
  broken auditor.
- Sequence-input TOCTOU.
- Name-based classification: N03 bytes renamed to another basename and loaded by path are not
  operator-classified, on main and in this proposal alike.

## Proposal P-N03: host-sealed buffer, auditor-verified source

**Host (Gate B, replaces `check_sha` and `dlopen` for N03 only):**

1. Open the resolved operator path once (`O_RDONLY | O_CLOEXEC`), giving the source descriptor `S`.
   Require a regular file and the manifest size when available. Read the whole file (EOF checked)
   into an owned buffer and require SHA-256 equal to the plan's.
2. Create `M = memfd_create("saccade-n03:src-fd=<S>", MFD_CLOEXEC | MFD_ALLOW_SEALING |
   MFD_NOEXEC_SEAL)`. Retry without `MFD_NOEXEC_SEAL` only on `EINVAL` (a kernel before 6.3, where
   the flag does not exist).
3. Write the buffer, then add `F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL` and read
   them back.
4. Read `M` back completely and require its size and SHA-256 equal to the plan's.
5. Call `dlopen("/proc/self/fd/<M>", RTLD_NOW | RTLD_LOCAL)` while `S` and `M` are still open,
   then close both.
6. There is no fallback to pathname loading. Every step failure is a `DetectorLoadError`.

**Auditor (`la_objopen` additions; existing path rules unchanged):**

- A mapped name under `/proc/` or `/dev/fd/`, or any absolute name that `realpath` cannot resolve,
  is refused unless it is the verified operator memfd. The auditor does not trust `realpath` of a
  descriptor name, because a file planted at the readlink text would resolve.
- The operator memfd passes only if every check holds:
  1. The name is exactly `/proc/self/fd/<M>`, and its readlink is
     `/memfd:saccade-n03:src-fd=<S> (deleted)` with strict decimal `S ≠ M`.
  2. `M` carries all four seals.
  3. The readlink of `/proc/self/fd/<S>` (the kernel `d_path`, the analogue of the existing
     `realpath` equality) equals the fixed operator path.
  4. `S` is a regular file with the same `st_dev` and `st_ino` as the file the auditor opens at
     that path (`O_NOFOLLOW`).
  5. `M` has the same size and the same bytes as that file, compared now in 64 KiB chunks. The
     auditor needs no hash code.
- Any failure is the existing `fail()`: a message, then `_exit(127)`.

**Resulting invariants:**

- **I1.** Mapped N03 bytes = sealed `M` = bytes with the planned SHA-256 (host step 4; seals make
  `M` immutable after the check).
- **I2.** Those bytes equal the fixed-path file at audit time, and the host's source was opened at
  that path (auditor checks 3–5).
- **I3.** No descriptor-backed object passes unless I1 and I2 hold.
- **I4.** Path-loaded objects follow exactly the main rules.

The label carries a descriptor number, not a claimed path. The auditor checks the kernel objects
themselves. Honest labeling by the host is the same in-process trust that main already places in
the host's `dlopen` argument.

## Alternatives

| Alternative | Byte equality | Provenance | Disposition |
|:--|:--|:--|:--|
| Status quo: hash, then path `dlopen` | no (B1–B3) | path only | OPEN residual |
| #582 unlabeled sealed memfd | yes | **bypassed** (A3) | rejected |
| Pin the inode: `dlopen /proc/self/fd/<S>` of the regular file | no: in-place writes still reach the mapping | main auditor accepts even a deleted copy (B4) | rejected |
| Auditor compares the memfd with the fixed path only, no source descriptor | yes | **weakened**: a same-byte foreign source would pass | rejected |
| Auditor substitutes the name in `la_objsearch` and copies the file itself | needs the expected hash inside the auditor, or the host checks after constructors have run | auditor-side | rejected: cross-namespace channel or post-init check |
| Re-hash after `dlopen` | no: N03 constructors run inside `dlopen` | — | rejected |
| fs-verity, immutable attribute, read-only mount | yes, if the deployment provides it | path | possible later hardening; needs root or filesystem support, not portable |
| Link N03 into the entrypoint | yes | ownership changes | out of scope (operator bytes change) |
| **P-N03** | **yes (I1)** | **kept (I2, C02/C03) and tightened (I3)** | **proposed** |

## Failure semantics and contract matrix

| Requirement | P-N03 behavior | Evidence (this phase) | Phase 2 must show |
|:--|:--|:--|:--|
| Loaded bytes = verified bytes (D5, MB-41 N03) | sealed memfd, readback hash; mapped segments equal the buffer | C01, C08, G3 (memory compare); B1–B3 contrast | in `DetectorHost`, under the real CLI |
| N03 path/source auditing kept | `d_path` + dev/ino + byte equality to the fixed path | C02, C03, C04, C05, C14, C15, G4 | auditor pytest cases mirroring C-series |
| Memfd/FD paths cannot bypass | descriptor names fail closed | C11–C13, C16, C17 | same, plus installed launcher |
| Path mode unchanged | existing rules untouched | C18, C19, 14/14 regressions | full existing suite |
| N03 RUNPATH exception and dependencies | RUNPATH is absolute with no `$ORIGIN`, so the search is identical | D1/D2: LD_DEBUG search differs only in the init name; 30 mapped objects identical (11 vendor) | entrypoint closure; refuse an N03 with `$ORIGIN` |
| Catchable host failures | `DetectorLoadError` → exit 2, run/load verification `failed` with null byte scope, identity retained | C07, C09, C10 (probe exit 2) | native CLI MB-58 controls |
| Uncatchable failures | auditor `_exit(127)` / SIGKILL → journal `running`, null load verification | C-series 127 (probe) | native CLI auditor + SIGKILL controls |
| Gate A identity | unchanged; N03 stays in the runtime package | not touched | identity unchanged in journal |
| Real GPU operator | registered op runs; float32/float16 bit-equal across load modes; graph capture + 3 replays = eager | G1–G3, G5 | full detector/graph/serial/double-buffer replays |
| Report scope | aggregate stays `hashed_before_load` | — | per-object N03 label per owner decision D-N03-2 |

## Experiments

Raw evidence is under `results/549_s2_2_n03/` (gitignored, local-only). Its `SHA256SUMS` digest is
`c36d83e3…08ce`; the evidence JSON lists every source, binary and summary hash.

Setup: evidence-only prefixes `prefix_installed` (untouched installed auditor) and `prefix_proto`
(prototype). Both have hardlinked `lib/vendor` from the #536 evidence tree and a fresh N03 copy per
case.

All runs use `/lib64/ld-linux-x86-64.so.2 --library-path <prefix>/lib/vendor --audit <auditor>`
with `PATH=/usr/bin:/bin`, as the launcher does. A shell hook runs at a fixed stage
(`after_open`, `after_hash` or `after_load`) to place mutations inside the race window
deterministically.

Final matrix r3 (31/31; inputs unchanged in every case) for the prototype:

| Case | Exit | Auditor or host message |
|:--|--:|:--|
| C01 proposed, pinned | 0 | readback SHA = plan, seals `0x2f`; mapped segments = verified |
| C02 proposed, same bytes, foreign path | 127 | `operator library mapped from another path` |
| C03 proposed, hardlink under a foreign name | 127 | `operator library mapped from another path` |
| C04 proposed, symlink alias resolving to the fixed path | 0 | same acceptance as main's `realpath` rule |
| C05 rename-replace after hash | 127 | source `d_path` ends `(deleted)` |
| C06 in-place write after hash | 127 | `operator memfd differs from the operator library` |
| C07 in-place write after open, before read | 2 | host SHA-256 mismatch |
| C08 in-place write after load | 0 | mapped segments still equal verified |
| C09 wrong expected SHA / C10 directory | 2 / 2 | host refusal |
| C11/C12 unlabeled memfd (foreign/pinned) | 127 | `descriptor-backed object is not the operator memfd` |
| C13 labeled, without `F_SEAL_WRITE` | 127 | `operator memfd is not sealed` |
| C14 label names the fixed source, bytes mutated | 127 | `operator memfd differs …` |
| C15 label names a closed descriptor | 127 | `operator source descriptor is not open` |
| C16 deleted copy / C17 regular file via `/proc/self/fd` | 127 / 127 | `descriptor-backed object is not the operator memfd` |
| C18/C19 path mode, pinned/foreign | 0 / 127 | unchanged |
| C20 `MFD_EXEC` variant | 0 | loads |

The GPU smoke runs under `tools/resctl.py run --wait gpu0` on an RTX 5070 Ti Laptop, sm_120,
driver 617.42. G1 (installed auditor, path), G2 (prototype, path), G3 (prototype, proposed) and
G5 (installed auditor, proposed) all produce the same results:

- eager float32 output `03f283f1…0be9` and float16 output `7a5a3331…bc2b`;
- three CUDA graph replays equal to eager.

G4 (proposed, foreign same-byte source) exits 127. This is the operator alone, not detector, MOT
or qualified parity.

Retained failed predictions:

- **r1 C05.** The refusal came from the path rule, not the predicted dev/ino rule. The dev/ino
  check remains a defense for bind-mount aliasing and is not separately exercised.
- **r2 C20.** A `MFD_NOEXEC_SEAL` memfd *does* load: that flag blocks `execve`, not
  `mmap(PROT_EXEC)`. The design therefore uses `MFD_NOEXEC_SEAL`. `MFD_EXEC` would be refused on
  a `vm.memfd_noexec=2` host.

## Known limits

- Hosts are refused when `memfd_create`, `/proc` or executable memfd mappings are unavailable
  (seccomp, SELinux, AppArmor). The refusal is catchable (exit 2); there is no path fallback. This
  is a new availability requirement compared with main; such hosts are **UNVERIFIED**, and this
  host has `vm.memfd_noexec=0`.
- `link_map` and `dladdr` names for N03 become `/proc/self/fd/<M>`, and `/proc/self/maps` shows
  `/memfd:saccade-n03:src-fd=<S> (deleted)`. Readers that match N03 by path must adapt, including
  the LD_DEBUG-based evidence readers and the probe.
- The auditor refuses `/proc/` and `/dev/fd/` names even when they resolve to an acceptable file.
  This is a deliberate tightening; production never loads that way.
- The auditor checks descriptors `S` and `M` by number during `dlopen`. A same-process thread
  that closes or `dup2`s over them concurrently (including a `close_range` sweep) is in-process
  code and out of scope; it would cause a refusal or misattribution, not a bypass of I1.
- The prototype auditor is a research build. Phase 2 changes `shipping/src/loader_audit.c` and
  `detector_host.cpp`, so the installed package becomes **UNVERIFIED** until a separately
  authorized re-pin, and runtime identity coordinates move.

## Phase 2 scope (only after acceptance)

- Change `DetectorHost` N03 load (one helper with a CPU unit test) and `loader_audit.c` as
  specified above.
- Keep Gate A, journal v3/report v5, the engine/head buffers and compatibility gates unchanged.
- Tests:
  - auditor pytest real-loader cases mirroring C02–C17;
  - native Gate B tests for host refusals;
  - native CLI controls for catchable exit 2/failed and uncatchable 127/SIGKILL running/null;
  - a GPU manifest and legacy positive;
  - serial, double-buffer, graph and seven-sequence replays;
  - full pre-push and CI.
- Optionally assert in `check_shipping_bundle.py` that the operator RUNPATH carries no `$ORIGIN`.

## Owner decisions requested

- **D-N03-1.** Accept P-N03, specifically the auditor additions and the host sequence, as the
  Phase 2 design.
- **D-N03-2.** Per-object N03 scope after Phase 2 evidence. Recommendation: `loaded_buffer` (the
  loader maps the sealed copy of the verified buffer). The aggregate stays `hashed_before_load`
  until separately authorized.
- **D-N03-3.** Hosts without `memfd_create` or `/proc`. Recommendation: fail closed (exit 2), with
  no pathname fallback.
- **D-N03-4.** Accept the auditor tightening that refuses all descriptor-named objects other than
  the verified operator memfd.

Not included: S2-3, allowlist approvals, model or weight or N03 byte changes, re-pin, #547
licensing, public release or signing, V5 qualification or benchmarks, and launcher/root policy.
`distribution.status=local-only`, `owner_confirmation=null`.
