---
doc-status: parked
doc-promotion: none
doc-date: 2026-07-16
doc-module: semantic
owner-module: semantic
work-class: theory-model-specification
wip-role: parked
activation-gate: "H0 ordered terminal owner acceptance + owner scheduling"
issue: 175
target-decision-layer: score-ranking
primary-intent: capability-map
output-class: diagnostic-only
mainline-transition: none
---

# Gap-conditioned stochastic transition model — parked task charter

## Status and authority

**PROPOSED / PARKED.** This charter records the stable problem boundary for
[Issue #175](https://github.com/raylei50653/saccade/issues/175). It is **not
active**, **not sealed**, **not sole-active**, and does not occupy semantic WIP.
Issue #175 does not activate automatically.

Activation requires both:

1. owner acceptance of an H0 ordered terminal; and
2. a separate owner scheduling decision.

This gate states a prerequisite; it does not declare H0's instantaneous
lifecycle state. Satisfying the prerequisite would not itself activate this
task.

This document is a parked task charter / problem specification, **not** a
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

The goal is not to replace `bridge_dist` with another uncalibrated distance. A
future specification must separately account for deterministic mean evolution,
gap-accumulated process uncertainty, exit-state estimation uncertainty, entry
observation uncertainty, and any context-conditioned drift.

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

This PR resolves none of the four items.

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

### 3. Independence or explicit cross-covariance

The future specification must declare independence between prediction error
and entry-observation error, including any required initial-state/process-noise
assumptions, or define the cross-covariance. For
\(C=\operatorname{Cov}(e^-,\epsilon_1)\), the dependent-error case must use an
expression of the form

\[
S_\Delta=H P^-_\Delta H^\top+R_1+HC+C^\top H^\top.
\]

### 4. Canonical-state affine M2 transition

M2 must be written as an explicit affine transition on the canonical state,
not only as a residual-state narrative. The state definition, \(A_\Delta\),
\(d_\Delta(c)\), \(Q_\Delta\), domains, and units must form one complete and
reviewable interface.

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

The following files do not exist and must not be created by this parked task.

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

## Provisional terminal family

This is a **provisional, not sealed** terminal shape. Its exhaustive order,
decision procedure, and dispositions must be completed before the active
declaration is sealed. That active contract must also carry the four mandatory
obligations above without claiming that the post-activation work is complete.
No terminal is selected by this charter.

| Provisional terminal | Required future coverage |
|:--|:--|
| `GCTM_OBSERVATION_INTERFACE_UNDEFINED` | state, observation, time, coordinate, covariance, or causal-availability interface remains undefined |
| `GCTM_TRANSITION_FAMILY_NOT_WELL_POSED` | the admitted transition family is dimensionally, probabilistically, or compositionally ill-posed |
| `GCTM_IDENTIFIABILITY_UNRESOLVED` | the intended claim cannot be identified under the declared observations or leakage boundary |
| `GCTM_SPECIFICATION_INCOMPLETE` | no earlier rejection applies, but required definitions, proofs, or decision items remain incomplete |
| `GCTM_MODEL_SPEC_SEALABLE` | every future specification and proof obligation required for a diagnostic-only seal is satisfied |

Even the sealable terminal would grant no automatic B1, O1, online, mainline,
or production authority.

## Activation boundary

```text
accepted H0 ordered terminal
→ separate owner scheduling decision
→ explicit activation of GCTM theory/specification work
```

No arrow is automatic. If the accepted H0 terminal does not provide the needed
runtime substrate, the abstract mathematics is not thereby refuted, but no
bridge-runtime claim is admissible without a separately accepted substrate or
re-charter.
