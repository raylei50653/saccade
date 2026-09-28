# #465 compiled-head reverse engineering declaration

Status: **draft / diagnostic only / no measurement authorized by this document**

## 0. Question

PR-2L established that the LibTorch head (L) and the eager PyTorch head (E)
are bit-identical on all 5,316 MOT17 frames, while the existing compiled control
head (C) differs from both by the same amount:

- (L = E) bit-for-bit on 5,316 / 5,316 frames;
- (L,C) and (E,C) have the same L1 deltas:
  - score max-abs: 0.0013653;
  - box max-abs: 0.4657 px;
  - 5 score-floor crossings;
- the frozen PR-2L terminal is `HEAD_PARITY_WITHIN_TOLERANCE`.

This study asks a narrower question:

> Where is the first execution layer at which the compiled control path (C)
> diverges from eager / LibTorch semantics, and which compile-stage mechanism
> class is sufficient to reproduce that divergence?

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
- treat Python-vs-C++ as a mechanism class. PR-2L already showed (L = E).

Any production change discovered from this work requires a separate PR.

## 2. Evidence carried in from PR-2L

The reverse-engineering study may use the following already-observed facts as
motivation, but must not silently treat them as new measurements:

1. PR-2L validity held: V1 39/39; V2/V3/V4/V5 held.
2. L and E were bit-identical on every L1 frame.
3. L,C and E,C had identical reported L1 deltas.
4. In L2, A_L was within the predeclared tolerance of A_C:
   - IDF1: 0.0 delta;
   - MOTA: 0.0 delta;
   - HOTA: +0.000498;
   - IDs: 0 delta.
5. A_L is not byte-identical to A_C end-to-end.
6. The PR-2L packet is
   `results/native_head_parity_465_libtorch/20260928T152911Z/`
   with `packet.json` sha256
   `e614f517ab089c8e8295bcafd645c30e45ee0e89be838af549a3193bb3ffaddc`.

These facts constrain the investigation: the unexplained numerical difference is
between the eager/LibTorch semantics and the compiled control path, not between
Python and LibTorch.

## 3. Mechanism classes to separate

The first runner should distinguish these classes without assuming which one is
responsible:

- **Dynamo / graph-capture boundary** — graph capture, guards, or graph breaks
  change the executed semantics before lowering.
- **AOT / decomposition boundary** — functionalization or decomposition changes
  the operation sequence while retaining eager kernels.
- **Inductor lowering / fusion boundary** — generated lowering, fusion, layout,
  reduction order, or kernel selection introduces the first numerical drift.
- **Harness-specific compile configuration** — the ordinary staged compile
  variants remain eager-equivalent but the exact shipping/control compile setup
  differs.
- **Backend-state / execution-policy boundary** — a compile-scoped setting
  (for example matmul / cuDNN policy, layout, or autotune state) is necessary for
  the drift.

These are labels for localization only. A label is not a production decision.

## 4. Two-stage design

### R0 — static provenance, no MOT17 frame read

Before any numerical replay, record the exact compiled-head construction path:

- source function / module identity and blob;
- compile entry point and options;
- relevant PyTorch/CUDA/cuDNN policy state;
- Dynamo/FX/AOT graph identity where available;
- graph-break / guard summary;
- generated-backend identity where available;
- operator/decomposition inventory sufficient to compare the compile stages.

R0 must not read MOT17 frame data and must not emit a parity verdict.

### R1 — staged compile localization

Feed the **same backbone features** to all compared head forms in each sampled
frame. The runner must define and hard-bind the exact construction of each arm.
The intended conceptual arms are:

- **E** — eager reference;
- **D** — graph-captured / Dynamo-level execution with eager backend semantics;
- **A** — AOT/decomposition-level execution with eager kernels where supported;
- **I** — Inductor-compiled execution using the declared compile settings;
- **C** — the existing harness compiled control, unchanged.

If the installed PyTorch version cannot expose one conceptual stage without
changing semantics beyond that stage, the declaration must say so **before**
measurement and omit that arm rather than substitute an approximate arm
post-hoc.

For every arm, record:

- score and box max-abs vs E;
- bit-identity vs E;
- non-finite count;
- floor-crossing count;
- compile graph / code identity;
- repeat identity.

No MOT metric is needed for the first localization pass. The target is the first
head-level boundary that creates the C-vs-E numerical drift.

## 5. Decision table

The first satisfied row is the localization result:

| Observation | Localization result |
| --- | --- |
| D differs from E | `DYNAMO_OR_CAPTURE_BOUNDARY` |
| D = E, A differs from E | `AOT_OR_DECOMPOSITION_BOUNDARY` |
| D = E, A = E, I differs from E and matches C's drift class | `INDUCTOR_LOWERING_BOUNDARY` |
| D/A/I = E but C differs | `HARNESS_COMPILE_CONFIGURATION_BOUNDARY` |
| one stage changes only when a declared backend-state variable is changed | `BACKEND_STATE_DEPENDENT` |
| validity failure or the staged arms cannot separate the boundary | `UNRESOLVED` |

"Matches C's drift class" must be defined numerically in the runner declaration;
it must not be decided by visual similarity after the run.

## 6. Stage-2 operator localization

Only after R1 identifies a compile-stage boundary may a second PR localize the
first divergent graph region/operator.

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

- hard-bind the source/frozen identities it depends on;
- use one declared output root;
- verify a clean tree and expected commit/ref;
- record the exact runtime / driver / library identity;
- verify all compared arms receive identical backbone features;
- verify no input mutation;
- require repeat identity for each arm;
- fail closed on graph-construction or stage-identity mismatch;
- include synthetic/unit tests that read no MOT17 frames;
- state whether any compile cache is cleared or reused, and keep that policy fixed.

No formal numerical measurement is authorized until the runner and its
measurement declaration are frozen separately.

## 8. Expected deliverable

The useful output is a compact answer of the form:

> Eager and LibTorch remain identical. The first reproducible numerical
> divergence appears at **<named compile stage>** under **<frozen execution
> policy>**. The earliest observed divergent graph region is **<region>**.
> This does / does not reproduce the C-vs-E drift class.

If the evidence cannot isolate a boundary, report `UNRESOLVED`; do not add
post-hoc arms.

## 9. Relationship to #465

PR-2L's frozen parity result remains authoritative for the runtime-route decision.
This reverse-engineering work is a diagnostic side branch intended to explain
why the existing compiled control differs numerically from eager/LibTorch.

Refs #465.
