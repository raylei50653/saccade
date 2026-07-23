---
doc-status: active
doc-promotion: none
doc-date: 2026-07-22
doc-module: semantic
owner-module: semantic
work-class: theory-model-specification
wip-role: sole-active
activation-gate: "H0 ordered terminal owner acceptance + owner scheduling — both satisfied 2026-07-22; see Activation record"
issue: 175
target-decision-layer: score-ranking
primary-intent: capability-map
output-class: diagnostic-only
mainline-transition: none
---

# Gap-conditioned stochastic transition model — task charter

## Status and authority

**ACTIVE (WP-A0, 2026-07-22).** This charter records the stable problem
boundary for
[Issue #175](https://github.com/raylei50653/saccade/issues/175) and, since
WP-A0, the activation contract. It is sole-active in semantic WIP
(projection: [`docs/modules/semantic/TODO.md`](../../modules/semantic/TODO.md)).

### Activation record

The activation gate required both:

1. owner acceptance of an H0 ordered terminal — **satisfied**: three
   owner-accepted ordered terminals `H0_PROVENANCE_INVALID` exist
   ([#209](https://github.com/raylei50653/saccade/issues/209) accepted
   2026-07-19; re-entry #2 and re-entry #3 accepted 2026-07-21; state
   fact-owner: [claim-state
   registry](../contracts/claim_state_registry.md)); and
2. a separate owner scheduling decision — **satisfied**: recorded by the
   owner on Issue #175, 2026-07-22 —
   [**GCTM ACTIVATION AUTHORIZED** (comment 5043900665)](https://github.com/raylei50653/saccade/issues/175#issuecomment-5043900665).

Activation authorizes **substrate-agnostic A-layer theory/specification work
only**. It does not change any H0 state: faithful capture = none, actual H0
guarantee = none, candidate/guarantee sets empty, Phase B forbidden, and no
H0 repair or re-entry is authorized by this activation.

**Activation ≠ active-declaration seal.** The active declaration was **not
sealed** by WP-A0. Pre-activation obligation 1 (the canonical
observation/time interface named below) is the active-declaration seal
condition and was deferred to WP-A1; WP-A0 resolved none of the four
obligations and created neither D1 nor D2.

**Active-declaration seal (WP-A1, 2026-07-22).** Obligation 1 is resolved by
WP-A1: the nine-field canonical observation/time interface and the
observation-mode/causal-availability rules are frozen in
[D1 §2–§3](../models/gap_conditioned_stochastic_transition_spec_v1.md). With
the WP-A0 freezes (scope, ordered terminal partition, decision procedure,
encoded obligations) this completes every pre-seal condition, and the active
declaration is **sealed as of the WP-A1 merge**. Obligations 2–4 remain
unresolved work before any `GCTM_MODEL_SPEC_SEALABLE` selection; the seal
grants no runtime, data, B1/O1, online, or production authority and changes
no H0 state.

### Bindings frozen at activation

- **Primary claim object = A** (latent state transition \(K_\Delta\)), per the
  owner-accepted primary-object decision (2026-07-22; planning memo:
  [GCTM primary object A — scope and plan](../../modules/semantic/research/gctm_primary_object_a_scope_and_plan_20260722.md)).
- **B / C / D are downstream constructible**, reached only through typed
  boundaries; they are not part of the A-layer claim identity.
- The **conditional scope qualification** of existing-online §§9.2–9.4
  (owner-accepted, landed in-force via PR #249 in
  [existing-online object analysis](../../modules/semantic/research/existing_online_object_analysis_for_gctm_alignment_20260718.md))
  governs which correspondence/competition obligations attach to which claim
  scope. An A-layer model-spec seal requires A, the observation/time
  interface, and typed B/C/D boundaries — not event competition.

This document remains a task charter / problem specification, **not** a
canonical mathematical specification. It selects no model or terminal and
claims no completed proof, identifiability result, checklist pass, or
`GCTM_MODEL_SPEC_SEALABLE` result. It authorizes no data, fitting, capture,
runtime, online, or production work.

Authority is intentionally split:

- this repository document owns the stable research question, conceptual
  decomposition, scope, activation prerequisites, pre-activation blockers,
  expected deliverables, and provisional terminal family;
- Issue #175 owns parked/active workflow, scheduling, discussion, and PR
  linkage;
- [`docs/modules/semantic/TODO.md`](../../modules/semantic/TODO.md) owns only a
  one-line navigation pointer and the module WIP projection.

## Research and evidence layers

The intended layers are separate claim objects:

```text
H0
  runtime observability / evidence fidelity

GCTM
  stochastic transition model specification

B1
  runtime-grounded offline attribution / mathematical claim

O1
  online intervention / system efficacy
```

- Mathematical feasibility at GCTM does not establish offline effectiveness at
  B1.
- Offline effectiveness at B1 does not establish online effectiveness at O1.
- A negative H0 terminal would not automatically invalidate substrate-agnostic
  mathematics.
- H0 acceptance is a project-scheduling and bridge-runtime-claim gate; it is
  not a logical prerequisite for substrate-agnostic transition mathematics.
- Without an accepted runtime substrate or separately accepted fidelity edge,
  no bridge-runtime claim may be made.
- No arrow is an automatic handoff. B1 and O1 require separately declared and
  authorized work.

## Stable research question

The motivating mismatch is:

\[
\boxed{
\text{deterministic motion mismatch}
\;\longrightarrow\;
\text{gap-conditioned transition likelihood}
}
\]

The future specification task asks whether a nested, observation-aware
\(M1\rightarrow M2\) transition family can give motion innovation a consistent
probabilistic meaning across gap lengths while keeping calibration,
candidate-local ranking, and eventual system efficacy as different claims.

The goal is not to replace the production CUDA bridge score `bdist`,
equivalently \(s_\theta(X_\ell,X_c,\Delta_{\rm on})\), with another
uncalibrated distance. Legacy midpoint `bridge_dist` naming and the offline
`s0` proxy are not the production baseline object. A future specification must
separately account for deterministic mean evolution, gap-accumulated process
uncertainty, exit-state estimation uncertainty, entry observation uncertainty,
and any context-conditioned drift.

## Conceptual model-family boundary

The equations below fix the intended family boundary only. They are
provisional interfaces, not sealed derivations or proof claims.

### M0 — deterministic comparison baseline

M0 is a constant-velocity comparison baseline:

\[
x_\Delta=x_0+v_0\Delta.
\]

Its raw mismatch quantities have no automatic calibrated-probability meaning.

### M1 — constant-velocity Gaussian / white acceleration

M1 retains the constant-velocity mean and admits gap-conditioned uncertainty:

\[
dx_t=v_t\,dt,
\qquad
dv_t=L\,dW_t,
\qquad D=LL^\top,
\]

\[
\Phi_{M1}(\Delta)=
\begin{bmatrix}
I & \Delta I\\
0 & I
\end{bmatrix},
\qquad
Q_{M1}(\Delta)=
\begin{bmatrix}
\frac{\Delta^3}{3}D & \frac{\Delta^2}{2}D\\
\frac{\Delta^2}{2}D & \Delta D
\end{bmatrix}.
\]

### M2 — integrated OU residual velocity

M2 is the candidate family in which velocity is decomposed as
\(v_t=\bar v(c)+u_t\) and residual velocity follows

\[
du_t=-\gamma u_t\,dt+L\,dW_t,
\qquad dx_t=v_t\,dt.
\]

For \(b=e^{-\gamma\Delta}\) and \(a=(1-b)/\gamma\), the future canonical
specification must express M2 as an affine transition on
\(z=[x;v]\), at least in the explicit shape

\[
z_\Delta=A_\Delta z_0+d_\Delta(c)+\eta_\Delta,
\qquad \eta_\Delta\sim\mathcal N(0,Q_\Delta),
\]

\[
A_\Delta=
\begin{bmatrix}
I & aI\\
0 & bI
\end{bmatrix},
\qquad
d_\Delta(c)=
\begin{bmatrix}
(\Delta-a)\bar v(c)\\
(1-b)\bar v(c)
\end{bmatrix}.
\]

Recording this required shape does not complete the affine-transition blocker
below: the canonical state, parameter domain, noise construction, covariance,
and dimensional conventions remain to be made explicit and reviewed.

M2 must nest to M1 as \(\gamma\rightarrow0\), in both transition mean and
covariance. This is a required future proof, not a result of this charter.

## Observation and uncertainty boundary

Position-only and joint position–velocity observations are distinct modes:

\[
H_x=\begin{bmatrix}I&0\end{bmatrix},
\qquad H_{xv}=I.
\]

Entry velocity may require lagged or future frames, so a future specification
must declare its causal availability rather than treating the joint mode as
interchangeable with position-only runtime observation.

The uncertainty objects must remain separate:

```text
P0   exit-state estimation uncertainty
QΔ   process uncertainty accumulated over the gap
R1   entry-observation uncertainty
SΔ   total innovation covariance
```

A common provisional prediction/innovation form is

\[
P^-_\Delta=\Phi P_0\Phi^\top+Q_\Delta,
\qquad
S_\Delta=H P^-_\Delta H^\top+R_1,
\]

but the expression for \(S_\Delta\) is valid only under the independence
declarations named in blocker 3. If prediction error and entry-observation
error are dependent, the future specification must define their
cross-covariance and use the corresponding expanded expression.

## Score and probability semantics

For innovation \(r_\Delta=y_1-Hm_\Delta\), the following are not synonyms:

- standardized innovation
  \(q=r_\Delta^\top S_\Delta^{-1}r_\Delta\) measures residual size relative to
  declared uncertainty;
- \(\log\det S_\Delta\) measures predictive uncertainty volume;
- Gaussian NLL
  \(E=\tfrac12q+\tfrac12\log\det S_\Delta+\tfrac{k}{2}\log(2\pi)\) combines
  residual fit and uncertainty volume;
- candidate-region probability integrates density over a declared region and
  therefore depends on region geometry/volume as well as alignment.

Calibration and candidate-local ranking must be evaluated as different
capabilities. In particular, if every candidate in one candidate event shares

\[
S_\Delta=\alpha_\Delta I,
\]

then gap-conditioned isotropic scaling cannot change candidate-local ordering;
it changes calibration only. More generally, when all candidates in an event
share the same \(S_\Delta\), dimension, gap, context, and observation mode,
\(q\) and NLL induce the same ordering. Candidate-specific covariance can alter
ordering only when its source and causal availability are explicitly declared.

## Activation-contract and model-seal obligations

### Pre-activation boundary-normalization obligations

The following interface obligations bound any future active GCTM declaration.
They do not select a model, time convention, score policy, or terminal. They
make the presently unresolved interfaces explicit so an active declaration
cannot silently choose them.

1. **Canonical observation/time interface.** The active declaration must name
   `coordinate_substrate_id`, `frame_time_unit`, `physical_gap_definition`,
   `online_horizon_definition`, `g_phys_to_delta_on_mapping`,
   `bridge_at_convention`, `continuous_dt_conversion`,
   `production_cv_null_offset`, and `null_offset_treatment`. In particular,
   it must not silently equate \(g_{\rm phys}\) with
   \(\Delta_{\rm on}\). GCTM owns the physical-to-runtime mapping; H0 may
   supply only a runtime field or fidelity edge.
2. **L2 score-insertion contract.** Any B1/O1 score-ranking path remains
   blocked until the registry-owned L2 contract exists. That contract, not
   this charter, must identify the cutoff, ranking, margin, and claim score
   sources and preserve or explicitly redefine the candidate universe. This
   charter neither selects those sources nor authorizes a second score policy.
3. **Typed cross-layer mapping.** A future declaration must bind its physical
   transition object, runtime-aligned observation/innovation, and score policy
   through declared `exact`, `derived`, `proxy`, `hypothesis`, or candidate
   relations. The shared mapping card is authoritative for the common B1/O1
   vocabulary; it grants no production authority.

For the bridge-runtime consumer registration that uses these obligations, see
[H0→GCTM consumer compatibility requirements](../../modules/semantic/research/h0_gctm_consumer_compatibility_requirements_20260718.md).
For the shared cross-layer mapping and L2 ownership boundary, see
[GCTM downstream tasks — B1 / O1 objectives and semantics](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md).

All four items below are unresolved mandatory activation-contract obligations,
and all must be resolved before any future model-spec seal. Activation may
authorize the work needed to resolve them; neither activation nor its active
declaration may assert that they are already solved.

The timing is split explicitly:

- **Before the active declaration is sealed:** freeze the scope, a mechanically
  exhaustive ordered terminal partition (including the specification-incomplete
  outcome), and its decision procedure; encode all four obligations in the
  active contract. Item 1 is therefore an active-declaration seal condition,
  while items 2–4 remain explicitly unresolved work.
- **After activation, before selecting `GCTM_MODEL_SPEC_SEALABLE`:** resolve
  items 2–4 and every remaining proof or specification obligation required for
  a model-spec seal.

Neither the charter PR nor the WP-A0 activation PR resolves any of the four
items; the Obligation-status table in the frozen terminal-partition section
records all four as unresolved at activation.

### 1. Exhaustive specification-incomplete outcome

The terminal partition must add an exhaustive, mechanically decidable outcome
such as `GCTM_SPECIFICATION_INCOMPLETE` for a reachable state in which no
observation-interface, well-posedness, or identifiability rejection applies but
required definitions or proofs remain incomplete.

### 2. Calibration-only gain versus candidate-local ranking gain

The future specification must define these as different claims with different
nulls, metrics, and terminal consequences. Better coverage or cross-gap
likelihood calibration does not imply better event-local ordering, and ranking
gain does not establish calibrated probability.

*Resolved by WP-A5 (2026-07-22):* frozen as
[D1 §6](../models/gap_conditioned_stochastic_transition_spec_v1.md) — two distinct
claims **CAL** (cross-event calibration) and **RANK** (candidate-local ranking),
each with its own null, metric family, evaluation unit, and consequence, over the
frozen §5 score quantities (\(q=r^\top S_\Delta^{-1}r\), \(\log\det S_\Delta\),
Gaussian NLL, candidate-region probability; invertible regime canonical
\(C=0\), \(R_1\succ0\)). The \(\chi^2_k\)/Gaussian distributional statements hold
under an explicitly declared **CAL Gaussian working null**
(\(\delta z_0\sim\mathcal N(0,P_0)\), with \(\eta_\Delta\perp\delta z_0\) (§5.2) and
\(e^-\perp\epsilon_1\) (§5.4) — the two independences already frozen in §5, so the
**only** addition is \(\delta z_0\)'s Gaussianity; §6.1) — beyond frozen §5, which
supplies only zero-mean and covariance. The separation is frozen both
directions: shared-\(S_\Delta\) / isotropic \(S=\alpha_\Delta I\) rescaling (a
non-trivial \(\alpha_\Delta\neq1\)) may change calibration but **not** candidate-local
order (calibration gain ⇏ ranking gain),
and ranking is invariant to a **common** strictly-increasing reparametrization
applied **uniformly** across candidates within an event (distinct per-candidate
\(\varphi_i\) can reorder), which generally breaks calibration (ranking gain ⇏
calibration gain); candidate-specific covariance may alter ordering **only** when
its source and causal availability are explicitly declared. Generic log/proper
score is a distribution-quality metric (calibration+sharpness) and is **excluded**
from the CAL metric family (coverage/PIT/conditional calibration error). The
shared-\(S_\Delta\) \(q\)/NLL ordering-equivalence **proof** is
[D2 §7 (Lemma L5)](../models/gap_conditioned_stochastic_transition_lemmas_v1.md).
D1 §6 measures no gain, selects no metric/threshold, and makes no ranking or
calibration data claim. This resolves the **last** of the four numbered
obligations; a sealable terminal still requires the D1 identifiability/leakage
matrix (§7), the schema-only B1 input interface (§8), and terminal review.
*(Status update: §7 was frozen by WP-A6 and §8 by WP-A7, so only terminal review
— WP-A8 — remains; see the obligation-status updates below, which are the status
source of record.)*

### 3. Independence or explicit cross-covariance

The future specification must declare independence between prediction error
and entry-observation error, including any required initial-state/process-noise
assumptions, or define the cross-covariance. For
\(C=\operatorname{Cov}(e^-,\epsilon_1)\), the dependent-error case must use an
expression of the form

\[
S_\Delta=H P^-_\Delta H^\top+R_1+HC+C^\top H^\top.
\]

*Resolved by WP-A4 (2026-07-22):* frozen as
[D1 §5](../models/gap_conditioned_stochastic_transition_spec_v1.md) — four
separate uncertainty objects (\(P_0\) exit-state, \(Q_\Delta\) §4.5, \(R_1\)
entry-observation, \(S_\Delta\) total innovation); prediction-error convention
\(e^-=z_\Delta-m^-_\Delta\Rightarrow P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta\)
(under the **declared** assumption \(\eta_\Delta\perp\delta z_0\), not a §4.7
consequence); innovation \(r=He^-+\epsilon_1\). The
**exactly-one decision is independence**: canonical A-layer declares
\(e^-\perp\epsilon_1\) (with the named initial-state/process-noise assumption),
giving \(S_\Delta=HP^-_\Delta H^\top+R_1\); the dependent-error case is a frozen,
explicitly-declared deviation using the expanded form above with
\(C=\operatorname{Cov}(e^-,\epsilon_1)\) constrained to genuine cross-covariances
(\([P^-_\Delta,C;C^\top,R_1]\succeq0\); signs tied to the frozen \(e^-\)
convention). Resolving this obligation discharges neither WP-A5 (obligation 2)
nor the later D2 \(q\)/NLL-ordering increment; D1 §5 computes no \(q\)/NLL and
makes no ranking claim.

### 4. Canonical-state affine M2 transition

M2 must be written as an explicit affine transition on the canonical state,
not only as a residual-state narrative. The state definition, \(A_\Delta\),
\(d_\Delta(c)\), \(Q_\Delta\), domains, and units must form one complete and
reviewable interface.

*Resolved by WP-A2 (2026-07-22):* frozen as
[D1 §4](../models/gap_conditioned_stochastic_transition_spec_v1.md) — canonical
state \(z=[x;v]\in\mathbb R^{2d}\) (coordinate dim \(d\); concretizes §2
substrate \(\mathbb R^k\), \(k=2d\)), M2 SDE, \(A_\Delta\), \(d_\Delta(c)\),
\(Q_\Delta\) (noise-integral + closed form), \(\gamma=0\) continuous extension,
parameter domains, causal assumptions, and units as one interface. Resolving
this obligation does **not** discharge WP-A3 (PSD/nesting/asymptotics proofs,
D2), WP-A4 (obligation 3), or WP-A5 (obligation 2); D1 §4 defers those and the
production operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\) is kept out of
canonical \(d_\Delta(c)\) and out of M2 drift.

## Non-scope

This task authorizes none of the following:

```text
no data execution
no parameter fitting
no GT/FP reveal
no frozen-pair-table analysis
no tracker implementation
no CUDA/runtime hook
no H0 ABI change
no capture or packet consumption
no online hook
no threshold or policy selection
no B1 execution
no O1 execution
no production claim
no mainline transition
```

It also makes no PSD proof, nesting proof, identifiability resolution,
acceptance-checklist pass, model seal, research claim-state transition, or
production-behavior change.

## Future deliverables — create only after activation

WP-A0 (activation) created none of these. **D1 exists as a seed since WP-A1**
([gap_conditioned_stochastic_transition_spec_v1.md](../models/gap_conditioned_stochastic_transition_spec_v1.md):
§2–§8 frozen through WP-A7 — all D1 deliverable items complete); D2 exists since WP-A3
([lemmas v1](../models/gap_conditioned_stochastic_transition_lemmas_v1.md)).

### D1 — canonical model specification

- equations and admitted parameter domains;
- position-only and joint observation interfaces;
- identifiability and leakage matrix;
- schema-only interface for a separately declared future B1 input.

Intended future path:
`docs/research/models/gap_conditioned_stochastic_transition_spec_v1.md`.

### D2 — lemma and proof appendix

- \(M2\rightarrow M1\) limit;
- positive-semidefinite covariance argument;
- \(q\)/NLL ranking equivalence under shared covariance;
- dimensional consistency;
- short- and long-gap asymptotics.

Intended future path:
`docs/research/models/gap_conditioned_stochastic_transition_lemmas_v1.md`.

### Terminal closure

Completion must write the selected terminal and task lifecycle back to this
task and Issue #175. It must not create an empty D5-equivalent peer document.

## Terminal partition — frozen at activation (WP-A0)

The former provisional terminal family is **frozen** by WP-A0 as the ordered,
mechanically exhaustive terminal partition of this task. No terminal is
selected by this charter; selection happens only at terminal review, and owner
acceptance records it.

| Order | Terminal | Predicate coverage |
|:--|:--|:--|
| 1 | `GCTM_OBSERVATION_INTERFACE_UNDEFINED` | state, observation, time, coordinate, covariance, or causal-availability interface remains undefined |
| 2 | `GCTM_TRANSITION_FAMILY_NOT_WELL_POSED` | the admitted transition family is dimensionally, probabilistically, or compositionally ill-posed |
| 3 | `GCTM_IDENTIFIABILITY_UNRESOLVED` | the intended claim cannot be identified under the declared observations or leakage boundary |
| 4 | `GCTM_SPECIFICATION_INCOMPLETE` | no earlier rejection applies, but required definitions, proofs, or decision items remain incomplete |
| 5 | `GCTM_MODEL_SPEC_SEALABLE` | every specification and proof obligation required for a diagnostic-only seal is satisfied |

### Decision procedure (frozen)

1. Terminal review evaluates the predicates **in the order above**; the first
   applicable predicate is the terminal. No later terminal may be substituted
   for an earlier applicable one.
2. Each predicate is decided against a **checklist artifact** committed in the
   terminal-review change: a table mapping every obligation of this contract
   (the four mandatory obligations, the D1/D2 deliverable items, and the typed
   B/C/D boundary) to its owning section or artifact, each marked exactly one
   of `complete` / `incomplete` / `rejection-established` (with the rejection
   argument linked). Every `rejection-established` row must additionally fill
   a mandatory `rejection_terminal` field constrained to `{1, 2, 3}` (the
   terminal its rejection establishes); rows with any other status leave the
   field empty. Selection is then mechanical: if any `rejection-established`
   row exists, the terminal is the **smallest** `rejection_terminal` value
   among them (first match in terminal order); otherwise any `incomplete` row
   selects terminal 4; otherwise terminal 5.
3. The partition is exhaustive by construction: terminals 4 and 5 cover every
   state not captured by rejections 1–3.
4. Owner acceptance of the recorded terminal closes the task; completion
   writes the terminal back to this charter and Issue #175 per the
   terminal-closure rule above.

### Obligation status at activation

| Obligation | Status at WP-A0 | Resolution point |
|:--|:--|:--|
| 1 · canonical observation/time interface (nine named fields) | **unresolved** | WP-A1 — active-declaration seal condition |
| 2 · calibration-only gain vs candidate-local ranking gain as distinct claims | **unresolved** | before `GCTM_MODEL_SPEC_SEALABLE` |
| 3 · independence or explicit cross-covariance \(C\) | **unresolved** | before `GCTM_MODEL_SPEC_SEALABLE` |
| 4 · canonical-state affine M2 transition (complete interface) | **unresolved** | before `GCTM_MODEL_SPEC_SEALABLE` |

*Update (WP-A1, 2026-07-22):* obligation 1 is **resolved** — frozen in
[D1 §2–§3](../models/gap_conditioned_stochastic_transition_spec_v1.md); the
table above is retained as the WP-A0 snapshot. Obligations 2–4 remain
unresolved.

*Update (WP-A2, 2026-07-22):* obligation 4 (canonical-state affine M2
transition) is **resolved** — frozen in
[D1 §4](../models/gap_conditioned_stochastic_transition_spec_v1.md). Obligations
2 and 3 remain **unresolved** (their owning packets WP-A5 / WP-A4); the
sealable terminal still requires them plus the WP-A3 proof appendix (D2).

*Update (WP-A4, 2026-07-22):* obligation 3 (independence vs explicit
cross-covariance \(C\)) is **resolved** — frozen in
[D1 §5](../models/gap_conditioned_stochastic_transition_spec_v1.md) with the
independence decision (\(C=0\)). Obligation 2 (calibration vs ranking) remains
**unresolved** (WP-A5). The WP-A3 proof appendix (D2) has landed; the sealable
terminal still requires obligation 2 plus the later D2 \(q\)/NLL-ordering
increment (D2 §7) and terminal review.

*Update (WP-A5, 2026-07-22):* obligation 2 (calibration-only gain vs
candidate-local ranking gain as distinct claims) is **resolved** — frozen in
[D1 §6](../models/gap_conditioned_stochastic_transition_spec_v1.md), with the
\(q\)/NLL ordering-equivalence proof filled in
[D2 §7 (L5)](../models/gap_conditioned_stochastic_transition_lemmas_v1.md).
**All four numbered activation-contract obligations are now resolved.** The
sealable terminal still requires the remaining D1 deliverables — the
identifiability/leakage matrix (D1 §7) and the schema-only B1 input interface
(D1 §8) — plus terminal review; these are **not** numbered obligations but are
charter D1 deliverable items and terminal-review checklist rows, so terminal
review cannot yet reach `GCTM_MODEL_SPEC_SEALABLE` (it would select terminal 4,
`GCTM_SPECIFICATION_INCOMPLETE`, on those incomplete rows). The lease below is
revised accordingly.

*Update (WP-A6, 2026-07-22):* the D1 identifiability/leakage matrix is **frozen**
in [D1 §7](../models/gap_conditioned_stochastic_transition_spec_v1.md) — the
**terminal-3 predicate object** (`GCTM_IDENTIFIABILITY_UNRESOLVED`). It specifies
the identifiability target set, observation/data-design regimes, the core
single-event confounding, the multi-gap **quotient** identifiability (**given
\(\gamma\) known**, only \(\{D,P_{vv},\mathrm{sym}(P_{xv}),P_{xx}+R_1\}\) under
\(H_x\), with two **structural** non-identifiable gauges — \(P_{xx}\!\leftrightarrow\!R_1\),
not broken even by \(H_{xv}\), and \(\mathrm{asym}(P_{xv})\), \(H_x\)-invisible —
under \(H_{xv}\) it becomes **observable** but its identification still needs
multi-gap + shared params + known \(\gamma\), not a single \(H_{xv}\) event; and
with \(\gamma\) **unknown**, 4 gaps leave a degree of freedom so \(\gamma\) is
unidentified — its identification needs \(>4\) gaps + a joint-map identifiability
condition (**global injectivity** ⇒ global id, or at least **full-Jacobian-rank** ⇒
local id; not equivalent), **not claimed proven**), the CAL/RANK
mutual-non-identifiability result, and a frozen leakage matrix; its verdict is
conditional (identifiability **specified**, not empirically established — no data
authorized; a claim is non-identifiable exactly when it must rely on an
unidentifiable component **or an unmet identifiability regime** — the two structural
gauges, or \(\gamma\) unknown without the \(>4\)-gap joint-map condition) and it
**selects no terminal**. This is a D1 deliverable, **not** a
numbered obligation (all four remain resolved). One D1 deliverable remains — the
schema-only B1 input interface (D1 §8, WP-A7) — plus terminal review (WP-A8), so
terminal review still could not yet reach the sealable terminal.

*Update (WP-A7, 2026-07-23):* the schema-only B1 input interface is **frozen** in
[D1 §8](../models/gap_conditioned_stochastic_transition_spec_v1.md) — the **last**
D1 deliverable item. It fixes only the *interface shape* a separately declared
future B1 input must have: the fail-closed consumption rule (an input instantiates
D1 **iff** it supplies every `required`/triggered `required-if` field, passes the
nine well-formedness predicates W1–W9, and keeps its claims inside the
claim-restriction map — otherwise it may cite no §6/§7 conclusion), the four field
blocks (declaration / event / pair / derived, with every derived quantity
**computed, never supplied**), and a claim-restriction map translating the §7.6
leakage-blocking conditions into per-field mechanical verdicts. It is **not** a
numbered obligation (all four remain resolved), and it **grants no B1 authority**:
the **B1-slot identity**, the **score-layer contract**, B1's activation gate and
frozen inputs, and all evaluation design remain owned by the
[B1 charter](gctm_b1_runtime_grounded_offline_attribution_task.md) — D1 §8 links
them and does not restate or pre-decide them. Fields are *requirements*, not
availability claims: no runtime observability is asserted and no fidelity edge is
created. **All D1 deliverable items are now complete**; the remaining step before
`GCTM_MODEL_SPEC_SEALABLE` is **WP-A8 terminal review** (checklist artifact +
mechanical terminal selection), which is where any terminal is selected — this
update selects none.

Even the sealable terminal would grant no automatic B1, O1, online, mainline,
or production authority.

## Activation boundary

```text
accepted H0 ordered terminal
→ separate owner scheduling decision
→ explicit activation of GCTM theory/specification work
```

No arrow is automatic. Both gates were satisfied on 2026-07-22 (see the
Activation record above). The accepted H0 terminals provide **no** runtime
substrate: the abstract mathematics is not thereby refuted, but no
bridge-runtime claim is admissible without a separately accepted substrate or
re-charter.

## Expected state (lease — replaceable, not accepted state)

Planned work-packet order after WP-A0; each packet is a separate PR, and this
lease may be replaced within this charter without a registry or WIP change:

```text
WP-A1  freeze canonical observation/time interface (obligation 1; D1 seed)
WP-A2  canonical-state affine M2 + Q_Δ interface (obligation 4)
WP-A3  nesting / PSD / asymptotics proofs (D2)
WP-A4  independence vs explicit cross-covariance C (obligation 3)
WP-A5  calibration vs ranking claim definitions (obligation 2; D1 §6)
       + D2 §7 q/NLL ordering-equivalence proof (L5)
WP-A6  identifiability & leakage matrix (D1 §7) — terminal-3 predicate object
WP-A7  schema-only B1 input interface (D1 §8)
WP-A8  terminal review (checklist artifact + mechanical terminal selection)
```

*Lease revision (WP-A5):* the earlier lease folded "terminal review" into WP-A5,
but terminal review's checklist requires **every** D1 deliverable complete,
including the identifiability/leakage matrix and the B1 schema, which remain
reserved (D1 §7/§8). With those incomplete the mechanical decision procedure
would select terminal 4 (`GCTM_SPECIFICATION_INCOMPLETE`), not the sealable
terminal. Terminal review is therefore split off to WP-A8, with WP-A6/WP-A7
resolving the two remaining D1 deliverables first. (This lease is replaceable
without a registry or WIP change.)

**Current step:** WP-A7 landed (D1 §8 frozen: schema-only B1 input interface —
the last D1 deliverable). **All four numbered obligations (1/WP-A1, 4/WP-A2,
3/WP-A4, 2/WP-A5) remain resolved**, D2 (WP-A3) has landed, and **every D1
deliverable item is complete** (§2–§8 frozen). Next = **WP-A8 (terminal review)**:
the checklist artifact plus the charter's mechanical terminal selection. No
terminal is selected until that review, and even the sealable terminal would grant
no B1/O1/online/production authority.

## History

- 2026-07-16 — parked task charter created (problem boundary; Issue #175).
- 2026-07-17 — B1/O1 split into their own proposed charters (PR #179).
- 2026-07-22 — owner accepted primary object = A and the existing-online
  §§9.2–9.4 conditional scope qualification (landed via PR #249).
- 2026-07-22 — **activated (WP-A0)**: owner scheduling decision on Issue
  #175 ([comment 5043900665](https://github.com/raylei50653/saccade/issues/175#issuecomment-5043900665));
  `doc-status: active`, `wip-role: sole-active`; terminal partition and
  decision procedure frozen; four obligations recorded unresolved; active
  declaration not sealed (obligation 1 → WP-A1); no D1/D2 created.
  Landed via PR #250 (merge `f5cda311`).
- 2026-07-22 — **WP-A1: active declaration sealed**. D1 seed created
  ([spec v1](../models/gap_conditioned_stochastic_transition_spec_v1.md));
  nine-field canonical observation/time interface + observation modes frozen
  (D1 §2–§3); obligation 1 resolved; obligations 2–4 remain unresolved; next
  packet WP-A2 (canonical-state affine M2 + \(Q_\Delta\)).
- 2026-07-22 — **WP-A2: canonical-state affine M2 + \(Q_\Delta\) frozen**.
  [D1 §4](../models/gap_conditioned_stochastic_transition_spec_v1.md) freezes
  the full affine transition interface
  \(K_\Delta(z_0,c)=\mathcal N(A_\Delta z_0+d_\Delta(c),Q_\Delta)\) at
  \(\Delta=g_{\mathrm{phys}}\): canonical state, M2 SDE, \(A_\Delta\),
  \(d_\Delta(c)\), \(Q_\Delta\) (noise-integral + closed form), \(\gamma=0\)
  continuous extension to M1, parameter domains, causal assumptions, units.
  Obligation 4 resolved; obligations 2–3 remain unresolved; canonical drift /
  M2 mean / production operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\)
  kept strictly separate; no WP-A3–A5 obligation claimed complete. Next packet
  WP-A3 (nesting / PSD / asymptotics proofs; D2).
- 2026-07-22 — **WP-A3: D2 proof appendix landed** (PR #253, merge
  `4b41b582`). Created
  [D2](../models/gap_conditioned_stochastic_transition_lemmas_v1.md) proving
  internal properties of the frozen D1 §4 kernel only: L1 (\(Q_\Delta\succeq0\);
  rank \(=2\operatorname{rank}D\) for \(\Delta>0\)), L2 (\(\gamma\to0\) mean +
  covariance nesting to M1, with proven \(\gamma\)-continuity), L3 (semigroup /
  Chapman–Kolmogorov covariance composition, valid for degenerate \(Q\)), L4
  (short-/long-gap asymptotics). Resolves no numbered obligation (proof
  appendix), selects no terminal, and left frozen D1 §2–§4 byte-frozen. Next
  packet WP-A4 (obligation 3).
- 2026-07-22 — **WP-A4: innovation composition frozen** (obligation 3).
  [D1 §5](../models/gap_conditioned_stochastic_transition_spec_v1.md) freezes
  \(P_0\)/\(P^-_\Delta\)/\(R_1\)/\(S_\Delta\) composition over the §4 kernel:
  prediction-error convention \(e^-=z_\Delta-m^-_\Delta\),
  \(P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta\), innovation
  \(r=He^-+\epsilon_1\). The exactly-one obligation-3 decision is **independence**
  (\(e^-\perp\epsilon_1\Rightarrow C=0\), \(S_\Delta=HP^-_\Delta H^\top+R_1\)),
  with the dependent-error expanded form
  \(S_\Delta=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top\) frozen as an
  explicitly-declared deviation. Obligation 3 resolved; obligation 2 remains
  unresolved. Four uncertainty objects kept separate; operator-layer offset
  \(\pm(\mathrm{bridge\_at}-1)v\) kept out of \(m^-_\Delta/e^-/S_\Delta\); no
  \(q\)/NLL/ranking claim; §2–§4 byte-frozen. Next packet WP-A5 (obligation 2 +
  terminal review).
- 2026-07-22 — **WP-A5: calibration vs candidate-local ranking claim space
  frozen** (obligation 2 — the **last** numbered obligation).
  [D1 §6](../models/gap_conditioned_stochastic_transition_spec_v1.md) freezes two
  distinct claims — **CAL** (cross-event calibration) and **RANK** (candidate-local
  ranking) — each with its own null, metric family, evaluation unit, and
  consequence, over the frozen §5 score quantities (\(q=r^\top S_\Delta^{-1}r\),
  \(\log\det S_\Delta\), Gaussian NLL, candidate-region probability; invertible
  regime canonical \(C=0\), \(R_1\succ0\)). Separation frozen both directions:
  shared-\(S_\Delta\)/isotropic rescaling is calibration-only (calibration gain ⇏
  ranking gain); ranking is invariant to a common strictly-increasing
  reparametrization applied uniformly across candidates in an event — distinct
  per-candidate \(\varphi_i\) can reorder — and such a reparametrization breaks
  calibration (ranking gain ⇏ calibration gain); candidate-region probability
  depends on region volume (non-congruent regions disagree with \(q\)-order);
  candidate-specific covariance may change order only with declared
  source/causal-availability. The \(\chi^2_k\)/Gaussian statements hold under a
  declared **CAL Gaussian working null** (\(\delta z_0\sim\mathcal N(0,P_0)\); the
  two independences \(\eta_\Delta\perp\delta z_0\) and \(e^-\perp\epsilon_1\) are the
  ones already frozen in §5.2/§5.4, so only \(\delta z_0\)'s Gaussianity is new;
  §6.1) beyond frozen §5; generic log/proper score excluded from the CAL metric
  family. The \(q\)/NLL ordering-equivalence **proof** is
  filled in [D2 §7 (Lemma L5, + L5.1/L5.2)](../models/gap_conditioned_stochastic_transition_lemmas_v1.md).
  D1 §6 measures no gain, selects no metric/threshold/data, selects no terminal.
  Reserved sections renumbered (identifiability/leakage → D1 §7, B1 schema → D1
  §8); frozen §5 kept byte-frozen (its two "§6" identifiability references
  superseded via an append-only renumber note). Obligation 2 resolved ⇒ all four
  numbered obligations resolved. **Lease revised:** terminal review split off
  (WP-A8) behind WP-A6 (identifiability, D1 §7) and WP-A7 (B1 schema, D1 §8),
  since terminal review's checklist can't reach the sealable terminal while those
  D1 deliverables are reserved. Also fixed the #253 owner-flagged non-blocking D2
  §5.2 nit (矩陣→向量) under review. No B1/O1/runtime/online/production authority
  granted.
- 2026-07-22 — **WP-A6: identifiability & leakage matrix frozen** (D1 deliverable;
  the terminal-3 predicate object, **not** a numbered obligation).
  [D1 §7](../models/gap_conditioned_stochastic_transition_spec_v1.md) specifies the
  identifiability target set (\(\{\gamma,D,\bar v,P_0,R_1\}\), the deterministic
  operator offset, and the CAL/RANK claims), the observation/data-design regime
  axes (\(H_x\)/\(H_{xv}\), single-event vs multi-gap population, context
  observability, labels), the core confounding (a single position-only event is
  non-identifiable — \(P_0\)-propagation, \(Q_\Delta\), \(R_1\) are additively
  conflated at one \(\Delta\)), the multi-gap separation as a **4-shape
  decomposition** (\(S_\Delta=(P_{xx}+R_1)+a\,\mathrm{sym}(P_{xv})+a^2P_{vv}+q_{xx}D\),
  reusing frozen §4.6 / D2 L2/L4) — so, **given \(\gamma\) known**, \(H_x\) multi-gap
  identifies only the **quotient** \(\{D,P_{vv},\mathrm{sym}(P_{xv}),P_{xx}+R_1\}\)
  (with \(\gamma\) **unknown**, 4 gaps leave one DOF ⇒ \(\gamma\) unidentified;
  identifying it needs \(>4\) gaps + a joint-map identifiability condition (**global
  injectivity** ⇒ global id, or at least **full-Jacobian-rank** ⇒ local id; not
  equivalent), **not claimed proven** — "\(\ge5\) gaps" is necessary not sufficient),
  leaving two **structural** non-identifiable gauges (\(P_{xx}\!\leftrightarrow\!R_1\),
  not broken even by \(H_{xv}\), fixable only by declared known \(R_1\)/\(P_{xx}\); and
  \(\mathrm{asym}(P_{xv})\), \(H_x\)-invisible — under \(H_{xv}\) **observable** but its
  identification still needs multi-gap + shared params + known \(\gamma\), not a single
  \(H_{xv}\) event) — the **CAL/RANK mutual-non-identifiability**
  result (ranking does not identify the calibration scale \(\alpha_\Delta\); a uniform
  monotone reparam leaves order fixed but breaks calibration), and a frozen
  **leakage matrix** (covariance-/mean-/claim-level, separating multi-gap-blockable
  leaks from the structural gauges, each with its blocking condition). Verdict is
  **conditional** — identifiability is *specified*, not empirically established (no
  data authorized); a claim is non-identifiable exactly when it must rely on an
  unidentifiable component **or an unmet identifiability regime** (the two gauges, or
  \(\gamma\) unknown without the \(>4\)-gap joint-map condition). §7 **selects no
  terminal** (that is WP-A8). The §7.4 draft that claimed
  the *full* \(\{P_0,\gamma,D,R_1\}\) generically identifiable under \(H_x\) was
  corrected pre-merge per #256 owner review — round 1 (the two structural
  \(P_{xx}\!\leftrightarrow\!R_1\) / \(\mathrm{asym}P_{xv}\) gauges), round 2 (the
  joint-\(\gamma\) 4-gap DOF ambiguity, making the quotient conditional on \(\gamma\)
  known), and round 3 (\(H_{xv}\) makes \(\mathrm{asym}P_{xv}\) observable-not-identifiable;
  honesty boundary widened to "unmet identifiability regime"; injectivity ≠
  full-Jacobian-rank). Frozen §2–§6 kept byte-frozen; stale
  `reserved`/`unresolved` identifiability references in §5.7/§6.6 superseded via an
  append-only status note. Also cleared the WP-A5 roll-forward nit (the obligation-2
  summary note now carries the non-trivial \(\alpha_\Delta\neq1\) qualifier). All
  four numbered obligations remain resolved; remaining before terminal review = WP-A7
  (B1 schema, D1 §8). No new file ⇒ no master_map regeneration. No
  B1/O1/runtime/online/production authority granted. Next packet WP-A7.
- 2026-07-23 — **WP-A7: schema-only B1 input interface frozen** (the **last** D1
  deliverable; **not** a numbered obligation — all four remain resolved).
  [D1 §8](../models/gap_conditioned_stochastic_transition_spec_v1.md) fixes the
  *interface shape* a separately declared future B1 input must have for the A-layer
  objects to be instantiated unambiguously: a fail-closed **consumption rule** (an
  input instantiates this D1 **iff** it supplies every `required` / triggered
  `required-if` field, passes all nine well-formedness predicates, and keeps its
  claims inside the claim-restriction map — otherwise it is **not** an instantiation
  and may cite **no** §6/§7 conclusion, since those conclusions are always bound to
  their regime); an explicit **authority split** — the **B1-slot identity**, the
  **score-layer contract**, B1's activation gate and frozen-inputs list, all
  evaluation design (folds, trial unit, minimum exposure, metrics, thresholds,
  blind/reveal) and the reserved-symbol rules stay owned by the
  [B1 charter](gctm_b1_runtime_grounded_offline_attribution_task.md) /
  [synthesis core](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md),
  linked and **not** restated, with same-named fields (`observation_mode`,
  `causal_availability`, `context_definition`, `P0_exit_cov_id`, `R1_obs_cov_id`, …)
  reused as the same objects; four field blocks (**D** declaration-level, **E**
  event-level, **P** pair-level, **X** derived) where every Block-X quantity
  (\(A_\Delta,d_\Delta(c),Q_\Delta,m^-_\Delta,e^-,P^-_\Delta,r,S_\Delta,q,\log\det
  S_\Delta,E,\Pi\)) is **computed, never supplied** — substituting an external value
  for the frozen formula leaves the instantiation; predicates **W1–W9** (anchor
  identity \(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}+(\mathrm{bridge\_at}-1)\), so the
  forbidden default \(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}\) is mechanically
  rejected; the operator offset \(\pm(\mathrm{bridge\_at}-1)v\) kept in its own field
  and never folded into \(d_\Delta/m^-_\Delta/e^-\); observation-mode consistency with
  the overlap ⇒ dependent-\(C\) rule; PSD + the \(S_\Delta\succ0\) regime for
  \(q/E/\Pi\); dimension/unit/derived consistency; the **§7 identifiability regime**
  made mechanically checkable; CAL/RANK evaluation-unit separation; conservation +
  **label isolation**; fail-closed missing values with no imputation); and a
  **claim-restriction map** turning each §7.6 blocking condition into a per-field
  verdict (union across absences; CAL and RANK never rescue each other). §8 is
  **schema-only**: it activates no B1, asserts no runtime availability (fields are
  *requirements*, not availability claims), takes no data, fits nothing, chooses no
  storage format, and selects no terminal. Frozen §2–§7 kept byte-frozen; the stale
  "§8 reserved" rows in §6.6/§7.9 superseded via an append-only status note.
  Predicate decidability was checked on synthetic records (conforming input passes
  all nine; 17 injected violations each caught by the intended predicate; union rule
  leaves the quotient claim admissible) — scratchpad only, no code committed. **All
  D1 deliverable items are now complete**; the only remaining step before a sealable
  terminal is **WP-A8 terminal review** (checklist artifact + mechanical terminal
  selection). No new file ⇒ no master_map regeneration. No B1/O1/runtime/online/
  production authority granted. Next packet WP-A8.
- 2026-07-23 — bounded corrections per #258 owner review (REQUEST CHANGES → fixed
  pre-merge; **D1 §8 not yet frozen**; §2–§7 byte-frozen; the WP-A7 entry above is
  superseded on the predicate count and sanity figures). Three blockers closed the
  gap between §8's promise and its fields: (1) **Block X was not computable from
  D/E/P** — no field supplied \(D\), \(\bar v(c)\), or the \(\gamma\) used for
  scoring under `gamma_status=unknown`, so a record could pass every check and
  still be unrecomputable; added `transition_diffusion_D_id`,
  `context_drift_model_id`, `gamma_estimate_id`, event-level `vbar_value`, and a
  new **closure predicate W5′** (binding completeness ⇒ otherwise **not an
  instantiation at all**, since CAL and RANK both need a computable score; plus
  same-artifact sharing of \(\gamma,D,\bar v\) inside `parameter_sharing_scope`,
  with \(P_0/R_1\) sharing additionally required for identification claims,
  W6(f)); (2) **\(e^-\) is latent** — Block X split into **X-a** (computable) and
  **X-b** (\(z_\Delta,\delta z_0,\eta_\Delta,e^-\), symbolic only), since the
  schema never supplies \(z_\Delta\); the observable counterpart is \(r\), and
  adding `z_delta_true`-style inputs is forbidden as a W8 label-isolation
  violation; (3) **missing transpose** — dependent-\(C\) row corrected to
  \(S_\Delta=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top\) per frozen §5.4. Plus a
  scope strengthening: **`identification_scope`** \(\in\{\)`local`,`global`\(\}\)
  — a `local_full_rank_established` condition supports only local claims and may
  not be restated as global identification (§7.4/§7.7). Predicate count is now
  **ten** (W1–W9 + W5′); re-checked on synthetic records (conforming declaration
  passes all ten; **27** injected violations each caught by the intended
  predicate; a `local` claim under a local-only condition correctly passes).
  Scratchpad only, no code committed. Authority boundary unchanged: still
  schema-only, no B1 activation, no runtime-availability claim, no terminal.
- 2026-07-23 — second bounded correction per #258 owner re-review (still
  pre-merge; D1 §8 not yet frozen; §2–§7 byte-frozen). Two blockers: (1) the
  \(\bar v(c)\) binding was **self-contradictory** (Block D required the drift
  model to be per-event evaluable while Block E's `vbar_value` was required-if it
  is not, making that branch unreachable) and W5′(b)'s "same value" reading would
  have outlawed the legal \(c_1\neq c_2\Rightarrow\bar v(c_1)\neq\bar v(c_2)\)
  case — fixed by separating **mapping identity** (`context_drift_model_id`,
  shared) from **evaluability** (new `context_drift_model_evaluable`) and
  per-event values (`vbar_value`), with new **W5′(a′)** requiring each event to
  resolve \(\bar v(c)\) by at least one of the two paths and the two to agree when
  both exist; (2) the **claim set was not machine-readable** — `claim_target`
  (CAL/RANK) cannot tell W6 which branch to fire, so the restriction map was
  complete in prose but not mechanically decidable; fixed by a required
  **`claim_dependency_set`** over a closed atom vocabulary (D1 §8.2.1) listing
  only §7 identifiability dependencies (no metric/threshold/score-ladder
  semantics, so no B1-charter authority is taken), with W6 restated per atom
  (branches (a)–(g), including the new `mean_bias_attribution` ⇒
  `operator_offset_declared` rule) and §8.7 judging atoms via a set-intersection
  test. Re-checked on synthetic records: **38** injected violations each caught,
  three legality cases correctly pass. Scratchpad only, no code committed.
  Boundary unchanged: schema-only, no B1 activation, no terminal.
