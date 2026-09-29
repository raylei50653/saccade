# #465 compiled-head reverse engineering — result

Status: **terminal = `BOTH_SCOPES`; stage H = `INDUCTOR_LOWERING_BOUNDARY`;
stage B = `UNRESOLVED` (first divergence at AOT, not sufficient)** (formal run,
one, valid).
Basis: [declaration](native_runtime_compiled_head_reverse_engineering_declaration.md)
(blob `565b4a1d8afc22714d33d9b39d4a5d2b199a6eca`, frozen at the #488 merge
`23d1736e`; no amendment). This document records the result under that frozen
declaration only. It adds no arm, no threshold and no new study design; it reads
no time quantity and changes no benchmark claim. PR-2L's frozen parity result
remains authoritative for the runtime-route decision (declaration §9).

---

## 1. This run

| Item | Value |
|:--|:--|
| declaration freeze | #488 merge `23d1736e5871abb6b3c0c71582c2899c449a25dc` |
| superseded execution freeze (r1) | #489 merge `20fddc198d9d0a94d5f2b25663ea05c017a4b93d`, annotated tag `freeze/465-compiled-head-re` (object `21f45421`). The tag is correct; the frozen runner parsed `git ls-remote` with the PR-2L tag, so V1 could never pass (packet-free V1 dry 23/24). No measurement was performed and no MOT17 frame was read. The tag is kept unchanged. |
| execution freeze (r2) | #490 merge `51e34c39ae2850dd590cd15c5b822c64aa5e1921` (parents `20fddc19` + `49c12f34`, on the `origin/main` first-parent chain); annotated tag `freeze/465-compiled-head-re-r2`, tag object `37779340ef6df318e33f6bf099e82b69e641e3e0`, peeled (local and `origin`) = `51e34c39`; freeze record = #465 comment 5889469444 |
| runner | `scripts/eval/diagnostics/compiled_head_reverse_engineering.py`, blob `c3c1cad3bc13f9ffe995f4fbda4c33734d697bd4` |
| execution | detached checkout `51e34c39`, clean tree; main CI green on this commit (run 36562878559, 6/6); packet-free V1 dry under `machine-bench` first (24/24); the formal run was the direct child of `tools/resctl.py run machine-bench`, 2026-09-29 11:52:58–11:58:19Z, same lease at start and end |
| packet | `results/compiled_head_reverse_engineering_465/20260929T115258Z/` (gitignored, kept locally); `packet.json` sha256 `3aa775e4…`, `MANIFEST.json` sha256 `fa1a419f…` |
| compile cache | per-run directories inside the packet (`compile_cache/r1/{inductor,triton}`), as fixed by the runner |
| backend policy | PR-2L L1: `matmul_allow_tf32=False`, `cudnn_allow_tf32=True`, `cudnn_benchmark=False`, graph-executor optimize off |

## 2. Validity

| Gate | Result |
|:--|:--|
| V1 | 24/24 in-run (frozen inputs, environment, driver/runtime, declaration/runner blobs, HEAD = tag peeled commit, lease) |
| V-anchor | exact: the E,C pair reproduces PR-2L `l1/l1.json` and `l1/l1_pairs.csv` |
| repeat identity | 140 frames re-run, 0 failures |
| input mutation | 0 |
| recompile | Dynamo counters equal before and after the frames; recompile limit never hit |
| non-finite | 0 in every arm |
| JIT fallbacks | 0 calls |
| graph construction | every compiled call attributed (0 unattributed), no compile during the attribution pass; R1 executed graphs equal the isolated R0 arm's call by call. C_H and C_B execute graphs compiled by C (`reused_from` C), as the gate allows |
| wrap sites | head 6 sites, block 3 sites (block layout `[1, 1, 1]`), identical to R0 |
| `unresolved_reasons` | none |

## 3. Decision result (§5)

**§5.2 scope cut** (R = C): C_H ≠ E and C_B ≠ E ⇒ **`BOTH_SCOPES`**. Each scope
is labeled separately.

**§5.3 stage ladder**:

| Scope | R | D_s | A_s | Stage label |
|:--|:--|:--|:--|:--|
| H | C_H | = E (5316/5316 frames bit-identical) | = E (5316/5316) | **`INDUCTOR_LOWERING_BOUNDARY`** |
| B | C_B | = E (5316/5316) | ≠ E, ∉ class(C_B) | **`UNRESOLVED`** (first divergence at AOT, not sufficient) |

No stage-arm is exact to its reference (X ≡ R does not hold for any arm).

In the words of declaration §8: the L1 C-vs-E drift is carried by **both
scopes**. Within scope **H**, the first divergence appears at the
**Inductor lowering boundary** under the fixed PR-2L L1 backend policy, and it
does not reproduce exactly. Within scope **B**, the stage is **`UNRESOLVED`**.
(Eager = LibTorch is PR-2L's result; this run does not re-measure L.)

## 4. Descriptive drift sizes (report only)

No row in §5 depends on the numbers below. D_s and K are as defined in
declaration §4 (anchors that differ from E; floor crossings at 0.05).

| Arm | score \|D\| | score max-abs | box \|D\| | box max-abs | crossings \|K\| |
|:--|--:|--:|--:|--:|--:|
| C | 578,847,912 | 1.3653e-3 | 550,586 | 0.4657 | 5 |
| C_H | 9,469,529 | 6.232e-4 | 7,397 | 0.0934 | 0 |
| C_B | 571,362,101 | 1.3653e-3 | 545,379 | 0.4657 | 5 |
| D_H, A_H, D_B | 0 | 0 | 0 | 0 | 0 |
| A_B | 46,223,554 | 1.3677e-3 | 92,921 | 0.2423 | 0 |

§5.1 class checks against the reference:

| X vs R | score recall / precision | box recall / precision | max-abs ratio (score / box) | K overlap | Class |
|:--|:--|:--|:--|:--|:--|
| C_H vs C | 0.016 / 0.967 | 0.012 / 0.926 | 0.456 / 0.201 | 0 of 5 | partial |
| C_B vs C | 0.987 / 0.9995 | 0.990 / 0.999 | 1.000 / 1.000 | 5 of 5 | ∈ class(C) |
| A_B vs C_B | 0.063 / 0.783 | 0.119 / 0.701 | 1.002 / 0.520 | 0 of 5 | partial |

Reading, limited to what these numbers show:

- C_B ∈ class(C) and C_H is partial relative to C. Most of C's numerical drift,
  including all 5 floor crossings, falls in the block scope. This is a
  descriptive observation. It does **not** say the block scope is the only
  cause: C_H ≠ E, and the scope cut is `BOTH_SCOPES`.
- Along the B ladder, D_B = E and A_B already differs from E, with about 8% of
  C_B's score differences and none of its crossings. About 78% of A_B's score
  differences fall on anchors where C_B also differs. How the AOT-stage partial
  drift becomes C_B's full class is not resolved by this declaration.

## 5. Allowed follow-up

- **H**: the stage label is resolved, so declaration §6 allows a separate
  stage-2 operator localization PR for the head scope. That PR must
  distinguish the first observed divergent boundary from a causal
  operator/kernel.
- **B**: the ladder result is `UNRESOLVED`, so this declaration **does not
  authorize** a §6 operator localization of the block scope. Studying B needs a
  new declaration.
- Owner decision (09-29): H stage-2 is **not** started now. H has a small drift
  footprint (about 9.5M score differences, 0 crossings), so it stays as a side
  branch that is already eligible for §6. The next research direction is a
  **new B-scope declaration** on how the partial drift produced at
  AOT/decomposition (A_B) grows into C_B's full drift class. That declaration
  is separate from this result document.
- #465's runtime-route status is unchanged; headline numbers are unchanged.

Refs #465.
