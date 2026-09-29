# #465 compiled-head reverse engineering declaration

Status: **draft / diagnostic only / no measurement authorized by this document**

## 0. Question

PR-2L's L1 stage fed the same backbone features to three head forms: the
LibTorch head (L), an eager PyTorch head (E), and a compiled control head (C).
Its findings:

- (L = E) bit-for-bit on 5,316 / 5,316 frames;
- (L,C) and (E,C) have the same L1 deltas:
  - score max-abs: 0.0013653;
  - box max-abs: 0.4657 px;
  - 5 score-floor crossings;
- the frozen PR-2L terminal is `HEAD_PARITY_WITHIN_TOLERANCE`.

**Which C this is.** The C that produced these deltas is the PR-2L **L1**
construction, and nothing else (runner
`scripts/eval/diagnostics/native_head_parity_libtorch.py`, blob `437bb97e` at
freeze commit `504ab0e2`):

1. `build_mamba_gated_detector(..., use_whole_graph=True)` builds the detector;
   E is a `copy.deepcopy` of `det.mamba_head`, taken before any compile toggle;
2. C is `det.mamba_head` with `set_head_compile(True)` and
   `set_block_compile(True)` applied (`mamba_head.py` blob `ddd4524c`):
   - each `cls_head[i]` / `reg_head[i]` `nn.Sequential` (3 + 3 sites) is
     wrapped individually in `torch.compile(m, mode="default")`;
   - each `MambaBlock` in `mamba_blocks[scale][j]` is wrapped individually in
     `torch.compile(b, mode="default")`;
3. both E and C are executed through `head._forward_eager(feats,
   return_embeddings=False)`, not through `forward`, CUDA-graph replay, or the
   whole-graph harness.

This is **not** the L2 whole-graph harness arm `A_C`: `A_C` additionally runs the
compiled modules under the harness's CUDA-graph capture, double buffering, and
the post-head pipeline. This study localizes the L1 C-vs-E drift only. It makes
no claim about `A_C`, and it does not claim that the L1 drift explains any
end-to-end difference between `A_C` and the other L2 arms.

This study asks a narrower question:

> For the L1 C construction above, which compile scope (head modules, Mamba
> blocks, or their combination) carries the C-vs-E drift, and, within that
> scope, at which compile stage does it first appear?

The purpose is explanatory. It does **not** reopen the PR-2L parity verdict and
does not block use of LibTorch as the current native-runtime candidate.

## 1. Non-goals

This study does not:

- change the detector, tracker, preset, thresholds, model weights, or benchmark claims;
- optimize performance;
- select a new production runtime;
- require byte-identical end-to-end MOT output;
- modify the frozen PR-2L packet or rerun that formal experiment;
- attribute a cause from a numerical correlation alone;
- treat Python-vs-C++ as a mechanism class. PR-2L already showed (L = E);
- explain the L2 `A_C` arm or any out-of-head end-to-end divergence (§2 item 7).

Any production change discovered from this work requires a separate PR.

## 2. Evidence carried in from PR-2L

The reverse-engineering study may use the following already-observed facts as
motivation, but must not silently treat them as new measurements. All numbers
come from the PR-2L packet
`results/native_head_parity_465_libtorch/20260928T152911Z/`
(`packet.json` sha256
`e614f517ab089c8e8295bcafd645c30e45ee0e89be838af549a3193bb3ffaddc`;
L1 detail in `l1/l1.json`).

1. PR-2L validity held: V1 39/39; V2/V3/V4/V5 held.
2. L and E were bit-identical on every L1 frame (5,316 / 5,316).
3. L,C and E,C had identical reported L1 summaries.
4. E,C nonzero-difference counts, read from the persisted L1 histograms
   (bin 0 = exact zero):
   - scores: 578,847,912 of 3,572,352,000 entries differ (16.2%). The score
     comparison is unmasked: all 8,400 anchors × 80 classes per frame;
   - boxes: 550,594 of 3,589,968 coordinate entries differ (15.3%). The box
     comparison covers the 897,492 anchors in the pairwise union mask
     (max score ≥ 0.05 in E or C) × 4 coordinates;
   - score p99.9 upper edge = `1e-10`, which is the histogram's lowest nonzero
     edge. The p99.9 quantity therefore cannot separate drift classes and is
     not used in §5.
5. In L2, A_L was within the predeclared tolerance of A_C:
   - IDF1: 0.0 delta;
   - MOTA: 0.0 delta;
   - HOTA: +0.000498;
   - IDs: 0 delta.
6. A_L is not byte-identical to A_C end-to-end on any of the 7 sequences.
7. **Out-of-head end-to-end divergence (out of scope).** A_N (`--no-compile`
   harness reference) and A_L have identical aggregate L2 metrics
   (IDF1/MOTA/HOTA/IDs and the report-only DetA/AssA/FP/FN) and the same first
   divergent frame vs A_C on every sequence. Their per-sequence output
   files nonetheless differ on 5 of 7 sequences; only MOT17-09-SDP and
   MOT17-10-SDP are byte-identical. Each arm was byte-identical across its own
   two repeats. Because L = E bit-for-bit in the L1 comparison, this A_N-vs-A_L
   divergence is **not explained by any head-value difference observed in
   PR-2L**. Its mechanism is unattributed. It is recorded here so that a later
   reading does not misattribute it to a head or compile mechanism, and it is
   **out of scope** for this head-level study.

These facts constrain the investigation: the numerical difference to localize
is between eager/LibTorch semantics and the L1 compiled construction, not
between Python and LibTorch.

## 3. Mechanism classes to separate

R1 first separates by **compile scope**, then by **compile stage** within the
scope that carries the drift.

Compile scope:

- **Head scope (H)**: the 6 `cls_head` / `reg_head` Sequentials (Conv2d → SiLU →
  Conv2d).
- **Block scope (B)**: the `MambaBlock` modules. Their `selective_scan_fwd`
  custom op is opaque to `torch.compile`, so only the surrounding PyTorch ops
  can change.

Compile stage, within one scope:

- **Dynamo / graph-capture boundary**: graph capture, guards, or graph breaks
  change the executed semantics before lowering.
- **AOT / decomposition boundary**: functionalization or decomposition changes
  the operation sequence while still using eager kernels.
- **Inductor lowering / fusion boundary**: generated lowering, fusion, layout,
  reduction order, or kernel selection introduces the first numerical drift.

These are labels for localization only. A label is not a production decision.

**Backend-state / execution-policy dependence is not tested by this study.**
R1 holds the backend policy fixed at the PR-2L L1 values (§4, R0) and declares
no toggle arm, so no terminal can conclude backend-state dependence. Testing it
needs its own declaration that names the exact variables and their arms.

## 4. Two-stage design

### R0: static provenance, no MOT17 frame read

Before any numerical replay, record the exact compiled-head construction path:

- source function / module identity and blob (`mamba_head.py`, the PR-2L runner);
- the exact list of compile wrap sites for each scope (6 head sites; the
  `(scale, j)` block sites) and the options at each (`mode="default"`, backend);
- PyTorch / CUDA / cuDNN / driver identity, pinned as in PR-2L V1;
- backend policy, held fixed at the PR-2L L1 values:
  `cudnn.benchmark=False`, `cudnn.allow_tf32=True`,
  `cuda.matmul.allow_tf32=False`;
- Dynamo/FX/AOT graph identity for each wrap site, where available;
- graph-break / guard summary for each wrap site;
- generated-backend (Inductor) code identity for each wrap site, where available;
- operator/decomposition inventory sufficient to compare the stages;
- compile cache policy (cleared or reused), fixed for all arms.

R0 must not read MOT17 frame data and must not emit a parity verdict.

### R1: scope cut, then staged compile ladder

**Inputs.** R1 uses the same frame path and the same 7 sequences / 5,316 frames
as PR-2L L1, and it feeds the same backbone features to every arm on every
frame. All arms are executed through `_forward_eager(feats,
return_embeddings=False)`, as in PR-2L L1. Every non-E arm is built from its own
`copy.deepcopy` of the same pre-compile head as E, so no instance is toggled
twice.

**Arms.** Every compile arm wraps exactly the same sites as the PR-2L setters,
one module per `torch.compile` call. Only the scope and the backend change:

| Arm | Head sites | Block sites | Backend |
| --- | --- | --- | --- |
| E | eager | eager | none |
| C | compiled | compiled | `inductor`, `mode="default"` (= PR-2L L1 C) |
| C_H | compiled | eager | `inductor`, `mode="default"` |
| C_B | eager | compiled | `inductor`, `mode="default"` |
| D_s | scope s | other scope eager | `eager` |
| A_s | scope s | other scope eager | `aot_eager_decomp_partition` |

`s ∈ {H, B}`. The Inductor stage of the ladder is the scope arm `C_s` itself;
there is no separate I arm, because an I arm would repeat the `C_s`
construction exactly. The ladder arms `D_s` / `A_s` run only for the scopes the
scope cut selects (§5.2). The runner PR may construct all ladder arms up
front, but only the selected ones count toward the result.

If the installed PyTorch (2.11.0+cu130) cannot build a `D_s` or `A_s` arm at
these sites without changing semantics beyond that stage, the runner PR must
say so **before** measurement and omit that arm. It must not substitute an
approximate arm after the fact. An omitted arm makes every row that needs it
unreachable, and the result then falls through to `UNRESOLVED`.

**Anchor validity (V-anchor).** R1's C-vs-E comparison, computed with PR-2L's
L1 formulas, must reproduce the carried-in PR-2L E,C results exactly: per
sequence score/box max-abs, floor crossings, masked-anchor count, and the
score/box histograms. Any mismatch makes the run `UNRESOLVED`, because the drift
being localized would then not be the PR-2L drift.

**Per arm X, record against E** (scores and boxes are reported separately and
never combined):

- `n_score(X)`: count of score entries (8,400 anchors × 80 classes per frame,
  unmasked) with X ≠ E bitwise;
- `n_box(X)`: count of box coordinate entries with X ≠ E bitwise, over the
  **fixed E-mask** (anchors whose max score in E is ≥ 0.05) × 4. The E-mask
  does not depend on X, so the sets are comparable across arms. These counts
  are therefore not comparable to the pairwise-union counts in §2 item 4;
- overlap with the reference arm R (§5.1): `|D_X ∩ D_R|`, where D_X is the set of
  entries at which X differs from E, for scores and boxes separately;
- score max-abs and box max-abs (px) vs E;
- score-floor crossing count vs E (anchors with exactly one of X, E at or above
  0.05);
- non-finite count;
- bit-identity vs E and vs R on every frame;
- compile graph / code identity;
- repeat identity (each arm bit-identical to itself on a re-run subset).

No MOT metric is needed. The target is the first head-level scope and stage at
which the L1 C-vs-E drift appears.

## 5. Decision rules

### 5.1 Drift class

For an arm X and a reference arm R (R ≠ E), with scores and boxes evaluated
separately:

- **X ≡ R (exact)**: X is bit-identical to R on every frame;
- **X ∈ class(R)**: all of the following hold for **both** scores and boxes:
  1. overlap recall `|D_X ∩ D_R| / |D_R| ≥ 0.9`;
  2. overlap precision `|D_X ∩ D_R| / |D_X| ≥ 0.9`;
  3. max-abs(X) is within `[0.5, 2] × max-abs(R)`;
  4. and, for scores only, the floor-crossing count satisfies
     `|c_X − c_R| ≤ max(2, 0.5·c_R)`;
- **X = E**: X is bit-identical to E on every frame;
- otherwise **X is partial**: X ≠ E and X ∉ class(R).

If `D_R` is empty for scores or for boxes, then X ∈ class(R) requires `D_X` to
be empty for that quantity too.

These thresholds are fixed here. They must not be re-chosen after reading
data.

### 5.2 Scope cut (reference R = C)

The first satisfied row is the scope result:

| Observation | Scope result | Ladder |
| --- | --- | --- |
| validity failure (V-anchor, repeat identity, input mutation, non-finite) | `UNRESOLVED` | none |
| C_H ∈ class(C) and C_B = E | `HEAD_SCOPE` | s = H, R = C_H |
| C_B ∈ class(C) and C_H = E | `BLOCK_SCOPE` | s = B, R = C_B |
| C_H ≠ E and C_B ≠ E | `BOTH_SCOPES` | s = H with R = C_H, and s = B with R = C_B, each labeled separately |
| C_H = E and C_B = E | `SCOPE_INTERACTION` | none (terminal) |
| any other pattern (for example, exactly one scope arm differs from E but is partial) | `UNRESOLVED` | none |

`SCOPE_INTERACTION` means that neither scope reproduces any drift alone, yet
compiling both does. This is the successor to the old "harness compile
configuration" row. It is terminal here, and explaining it needs a new
declaration.

### 5.3 Stage ladder (per selected scope s, reference R = C_s)

The first satisfied row is the stage label for scope s:

| Observation | Stage label |
| --- | --- |
| D_s ∈ class(R) | `DYNAMO_OR_CAPTURE_BOUNDARY` |
| D_s ≠ E and D_s ∉ class(R) | `UNRESOLVED` (first divergence at capture, not sufficient) |
| D_s = E, A_s ∈ class(R) | `AOT_OR_DECOMPOSITION_BOUNDARY` |
| D_s = E, A_s ≠ E and A_s ∉ class(R) | `UNRESOLVED` (first divergence at AOT, not sufficient) |
| D_s = E, A_s = E | `INDUCTOR_LOWERING_BOUNDARY` (C_s ≠ E by §5.2) |
| a required ladder arm was omitted (§4) | `UNRESOLVED` |

Every "differs from E but is not in the reference class" case is an explicit
`UNRESOLVED` row. There is no silent fallthrough. The result also reports the
exact qualifier (X ≡ R) when it holds, but no row depends on it.

## 6. Stage-2 operator localization

Only after R1 assigns a non-`UNRESOLVED` stage label to a scope may a second PR
localize the first divergent graph region/operator in that scope.

Preferred methods, in order:

1. graph/partition bisection while preserving the same inputs;
2. exact intermediate-tensor capture at stable FX/AOT boundaries;
3. generated-code / kernel metadata inspection for the first divergent region.

Do not instrument arbitrary Python module hooks if fusion or graph lowering makes
the hook boundary different from the compiled graph boundary.

A stage-2 result must distinguish:

- "first observed divergent boundary" from
- "causal operator / kernel".

The former may be reported without claiming the latter.

## 7. Validity requirements for the runner PR

The runner PR must be independently reviewable and must, before any formal run:

- hard-bind the source/frozen identities it depends on, including the PR-2L
  packet sha256 and `l1/l1.json` used by V-anchor;
- use one declared output root;
- verify a clean tree and expected commit/ref;
- record the exact runtime / driver / library identity;
- verify all compared arms receive identical backbone features;
- verify no input mutation;
- require repeat identity for each arm;
- fail closed on graph-construction or stage-identity mismatch, including a
  wrap-site list that differs from R0;
- include synthetic/unit tests that read no MOT17 frames, including tests of
  the §5.1 class predicate and every §5.2 / §5.3 row;
- state whether any compile cache is cleared or reused, and keep that policy fixed.

No formal numerical measurement is authorized until the runner and its
measurement declaration are frozen separately.

## 8. Expected deliverable

The useful output is a compact answer of the form:

> Eager and LibTorch remain identical. The L1 C-vs-E drift is carried by
> **<scope result>**. Within scope **<s>**, the first divergence appears at
> **<stage label>** under the fixed PR-2L L1 backend policy, and it
> does / does not reproduce exactly (X ≡ R).

If the evidence cannot isolate a scope or stage, report `UNRESOLVED`. Do not add
post-hoc arms.

## 9. Relationship to #465

PR-2L's frozen parity result remains authoritative for the runtime-route decision.
This reverse-engineering work is a diagnostic side branch intended to explain
why the PR-2L L1 compiled construction differs numerically from eager/LibTorch.

Refs #465.
