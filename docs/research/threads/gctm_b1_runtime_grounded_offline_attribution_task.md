---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-17
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "eight frozen owner-accepted conditions + registry §7 score-layer contract + B1-slot identity resolution + separate owner scheduling"
target-decision-layer: score-ranking
primary-intent: design-evaluation
output-class: "design candidate | diagnostic result | unexplained residual set"
mainline-transition: "none from this charter; per-terminal transitions listed inside"
created: 2026-07-17
---

# GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation — proposed task charter

## Status and authority

**PROPOSED / non-WIP.** This charter is the B1 task object split out of the
[B1/O1 objectives-and-semantics synthesis](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
on 2026-07-17, executing that document's §37 repository split. It is **not
active**, **not sealed**, **not sole-active**, and does not occupy semantic WIP
(sole active is O0, per [semantic TODO](../../modules/semantic/TODO.md)). It
does not activate B1, select a model, freeze numerical thresholds, authorize
data access, modify runtime behavior, or promote a production policy.

Authority is intentionally split:

- **this charter** owns B1's task identity, activation prerequisites —
  including the **B1-slot identity question** and the **registry §7
  score-layer prerequisite** (sole owner; other surfaces link here) — scope,
  frozen degrees of freedom, evidence-space and validity semantics, provisional
  terminal family, deliverables, and the B1→O1 handoff object shape;
- the [synthesis core](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
  owns the **shared** cross-task semantics (§0 alignment boundary, including
  the GPU foot-bridge hook scope and reserved-symbol renames; evidence chain
  and claim ownership §1; shared semantic rules §2; outcome interpretation
  matrix §35; forbidden inference shortcuts §36; final semantic summary §39) —
  this charter consumes them and does not restate them as a second truth;
- the [H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
  owns runtime observability / evidence fidelity;
- the [GCTM parked charter](gap_conditioned_stochastic_transition_model_task.md)
  owns the transition-family specification boundary;
- [`docs/modules/semantic/TODO.md`](../../modules/semantic/TODO.md) owns only a
  one-line navigation pointer and the module WIP projection.

## Unresolved B1-slot identity (open owner decision)

H0 declaration §7 and the
[O0 routing charter](closed/bridge_frozen_evidence_o0_routing_20260716.md) (route 5)
name a "separately declared **B1 consumer-faithful operating-curve study**".
Whether the `GCTM_B1` study specified here occupies that same B1 slot, coexists
with it, or supersedes it is **not decided by this charter**. Resolving that
relation is an explicit activation prerequisite below. Until an owner resolves
it, `GCTM_B1_*` must not be cited as the B1 that H0 route 5 makes candidate.

## Hook scope: GPU foot-bridge only

The cross-task hook scope is **owned by the
[synthesis core §0](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)**:
this task is scoped end-to-end to the GPU foot-bridge two-stage winner, and
the association (auction) stage is out of scope. This charter consumes that
boundary and does not restate its full rules; see core §0 for the legal
insertion surface and the association-stage exclusion.

## Task identity

> **B1 — Runtime-grounded offline attribution and score-ranking evaluation of
> the gap-conditioned stochastic transition model**

The primary intent is **design evaluation** because only a successful B1 design
candidate can justify O1. Calibration and attribution are mandatory
diagnostics, not substitutes for ranking value. Terminal prefixes are
`GCTM_B1_*` and future evidence files are `gctm_b1_*`, kept precisely so they
cannot be confused with the closed `m_b1_*` (M-B1) line.

## Activation gate

B1 may become active only after all of the following are owner-accepted and
frozen:

1. an accepted runtime substrate or fidelity edge from H0;
2. a positive, sealable GCTM theory terminal;
3. exact H0 evidence identity and checksum;
4. exact GCTM specification and proof identities;
5. one observation mode;
6. one parameterization family;
7. a sealed B1 declaration satisfying the project seal bar;
8. a separate owner scheduling decision.

No condition auto-activates B1.

Additionally:

- **Score-layer contract is a hard prerequisite.** B1 is an `L2 score` object.
  Per [claim-state registry §7](../contracts/claim_state_registry.md), the
  score-layer evidence contract (rank/margin/top-1 semantics and claim ladder)
  does not exist yet, so no B1 object can enter the admissible candidate set
  (`transition_semantics: unavailable`) regardless of H0/GCTM outcomes.
  Writing that contract does not depend on H0. None of the eight conditions
  above substitutes for it.
- **H0→GCTM consumer compatibility is a hard prerequisite for the
  bridge-runtime path.** A sealed B1 declaration must bind a
  B1-declaration-owner-accepted compatibility verdict mapping every required
  runtime-observable GCTM object to its accepted H0 guarantee/fidelity edge.
  This does not block
  substrate-agnostic GCTM mathematics or alter H0's terminal; it blocks only a
  bridge-runtime B1 claim when the required observation mapping is absent.
- **B1-slot identity must be resolved** (previous section) before activation.

A negative H0 terminal blocks bridge-runtime B1 unless a new evidence substrate
or explicit substrate-agnostic re-charter is accepted.

## Research question

> On an accepted runtime-grounded evidence substrate, does one frozen GCTM
> instantiation provide stable, interpretable, event-local ranking improvement
> over the deterministic baseline, without obtaining apparent long-gap gains by
> destroying short-gap ordering, leaking held-out information, or changing the
> candidate universe?

B1 contains three separate questions.

### B1-Q1 — Calibration

Does the declared model make motion innovation more consistently interpretable
across gap lengths and contexts?

### B1-Q2 — Candidate-local ranking

Does the model improve relative ordering inside the same candidate event?

### B1-Q3 — Mechanism attribution

Is any gain attributable to:

```text
M0 → M1
  gap-conditioned uncertainty / anisotropic normalization

M1 → M2
  velocity-memory decay and/or leakage-free context drift
```

These answers must remain separate.

## Maximum supported claim

The strongest legal B1 conclusion is:

> On the declared offline runtime-grounded substrate, the frozen GCTM score
> policy is a stable and interpretable **score-ranking design candidate**.

B1 does not establish:

```text
online causal availability
online hook fidelity
online assignment retention
MOT metric improvement
latency acceptability
production safety
production default suitability
```

## Scope

### In scope

- consume one accepted H0 evidence substrate;
- instantiate one accepted GCTM parameterization family;
- fit parameters on declared training folds only;
- compute M0, M1, and M2 under one frozen observation interface;
- evaluate calibration and ranking as separate claim spaces;
- quantify M0→M1 and M1→M2 attribution;
- evaluate per-gap, per-context, per-sequence, and held-out stability;
- evaluate short-gap retention;
- evaluate event-local assignment-flip potential in offline replay;
- produce one bounded terminal.

### Out of scope

```text
no online hook
no tracker state mutation
no production preset change
no online threshold search
no post-reveal parameter refit
no held-out sequence refit
no H0 ABI amendment
no missing-observable retrofit into H0
no hard-reject gate
no claim that likelihood is a GT posterior
no system-efficacy claim
no production claim
```

## Frozen inputs

A sealed B1 declaration must pin at least:

```text
accepted_h0_terminal
h0_evidence_id
h0_packet_hash
h0_schema_version
runtime_instrumentation_identity

accepted_gctm_terminal
gctm_spec_hash
gctm_lemma_hash
coordinate_substrate_id
time_conversion
observation_mode
causal_availability

parameterization_family
parameter_domain
context_definition
context_fallback
missing_value_rule
P0_exit_cov_id      # exit-state covariance P0; field renamed to avoid the P0 study code
R1_obs_cov_id       # entry-observation covariance R1; field renamed to avoid the R1 study code
fit_objective
regularization
precision
tie_breaking

model_set
score_transform
score_sign
score_composition
score_weight_or_weight_selection_rule
candidate_universe
event_key
source_scope
folds
trial_unit
dependence_treatment
aggregation
minimum_exposure
minimum_effect
short_gap_retention_bar
futility_stop
```

A future declaration must not leave a legal choice between standalone ranking,
additive score correction, candidate-specific covariance, or alternative
normalizations.

The base score is written \(s_{\mathrm{base}}\), never `s0`: in this
repository `s0` is the reserved name of the offline proxy of `bdist`, and any
use of `s0` to stand for a production quantity is registry-inadmissible
(synthesis core §0/§2.1).

## Evidence spaces and reductions

### Source pair space

\[
\mathcal U_{\mathrm{pair}}
=
\{(e,i): i\in C_e\}.
\]

Each source record contains the frozen runtime-grounded state, candidate
observation, gap, context, and model outputs.

### Candidate-event space

\[
\mathcal U_{\mathrm{evt}}
=
\{e: C_e \text{ is one candidate set}\}.
\]

The pair-to-event reduction is:

\[
\rho_{\mathrm{pair}\rightarrow\mathrm{evt}}
:
\{s(e,i)\}_{i\in C_e}
\mapsto
\text{rank, margin, top-k, selected candidate}.
\]

Ranking claims are quantified over events, not pooled independent rows.

### Calibration space

\[
\mathcal U_{\mathrm{cal}}
=
\{(e,i^\star): i^\star \text{ is the declared correct transition}\}.
\]

Calibration statistics do not become ranking statistics through aggregation.

### Trial and domain space

The declaration must select a trial unit such as:

```text
candidate event
track
sequence
fold
```

and name residual clustering above that unit.

### Conservation identities

At minimum:

```text
every admitted pair belongs to exactly one event
every event belongs to exactly one sequence and one fold
pair counts reconcile with event candidate counts
GT-present / GT-absent event partitions reconcile
fit, validation, and test partitions are disjoint
all dropped records have one enumerated reason
```

## Model-comparison semantics

### M0

M0 is the frozen deterministic baseline. It has no automatic probability
interpretation.

### M1

M1 tests the value of gap-conditioned uncertainty under the constant-velocity
mean.

If:

\[
S_{\Delta,e}=\alpha_{\Delta,e}I
\]

is shared by all candidates in the event, M1 cannot change candidate-local
ordering. It may improve calibration only.

A legal M1 ranking gain must come from a predeclared ranking-active mechanism,
such as:

- anisotropic covariance;
- causally valid candidate-specific observation covariance;
- interaction with a frozen base score through a predeclared score mapping.

### M2

M2 tests the additional value of velocity-memory decay and/or context drift
under the same observation and covariance interface used by M1.

M2→M1 comparisons must not mix:

```text
observation mode
candidate universe
fitting folds
P0 / R1 semantics
missing-value behavior
score-composition rule
```

### q and NLL invariant

When all candidates in one event share covariance, dimension, gap, context, and
observation mode:

\[
\operatorname{rank}(q_i)
=
\operatorname{rank}(\mathrm{NLL}_i).
\]

Any implementation reporting different rankings in this condition fails an
implementation-validity invariant.

## Fitting, blind, and reveal boundary

A sealable B1 protocol must define:

1. which data estimate dynamics or observation noise;
2. whether GT-linked transitions are used for fitting;
3. which labels remain hidden during model and policy selection;
4. fold-specific fit identities;
5. no-refit behavior after held-out reveal;
6. blind artifact hashes;
7. reveal binding to the sealed runner and frozen artifacts.

Legal patterns:

```text
Pattern A
  one independent development set disjoint from every reported test domain

Pattern B
  nested per-fold selection where each held-out fold is excluded from all
  operations determining its evaluated policy
```

Illegal patterns:

```text
full-pool fitting before LOO
post-reveal model selection
post-reveal exit-covariance / observation-covariance / gamma / D / weight / fallback change
treating repeated candidate rows as independent evidence
```

## Headline metrics

The future declaration must choose one primary ranking metric.

### Primary candidates

- event-local pairwise ranking accuracy;
- GT reciprocal rank;
- top-k GT recall;
- GT–best-negative score margin;
- correct assignment rate under a frozen offline decision rule.

### Mandatory stability views

```text
per sequence
per held-out fold
per gap bin
per context stratum
short-gap retention
long-gap effect
eligible-event exposure
GT-present / GT-absent counts
M0→M1 attribution
M1→M2 attribution
```

### Calibration diagnostics

- \(q\) coverage versus the declared \(\chi^2_k\) reference;
- coverage by gap and context;
- held-out NLL or another proper score;
- residual direction and multimodality diagnostics;
- covariance-volume diagnostics preventing unrestricted diffusion from
  receiving free credit.

### Forbidden headline substitutions

```text
pooled AUC
overall pair accuracy
raw NLL improvement
lower mean bridge distance
more FP removed
one best sequence
one best threshold
```

## Positive design bar

A positive B1 design candidate must satisfy:

1. provenance, observation, identifiability, fit/reveal, dependence, and
   implementation validity;
2. purpose alignment with score-ranking;
3. mechanism interpretability;
4. structural simplicity;
5. stable minimum ranking gain;
6. short-gap retention;
7. cross-sequence and held-out retention;
8. candidate-universe invariance;
9. runtime-causal portability of fields intended for O1;
10. no post-hoc selection.

Selection order:

```text
purpose alignment
→ mechanism interpretability
→ structural simplicity
→ stability
→ utility threshold
```

The highest-utility model does not auto-win.

## Provisional terminal family

This is a **provisional, not sealed** terminal shape. A sealed B1 declaration
must fix the exhaustive order and decision procedure. No terminal is selected
by this charter.

### 1. `GCTM_B1_PROVENANCE_INVALID`

Accepted H0/GCTM identities, hashes, schema, runner, or frozen inputs do not
match.

```text
current B1 execution invalid
no model inference
no O1 eligibility
scientific path not exhausted
```

### 2. `GCTM_B1_OBSERVATION_OR_IDENTIFIABILITY_INVALID`

Required observables are unavailable, causal status is false, context leaks
held-out or future information, the selected parameter family is not
identifiable, or the declared dependence model is invalid.

```text
current B1 study unresolved
no efficacy verdict
repair requires a new declaration or identifying evidence
```

### 3. `GCTM_B1_EXECUTION_INVALID`

Build, fit, runner, serialization, reveal binding, aggregation, or packet
failure.

```text
current execution closes
no model inference
repair and reseal required
```

### 4. `GCTM_B1_MODEL_IMPLEMENTATION_INVALID`

Mathematical or accounting invariants fail, including:

```text
illegal covariance
undefined inverse/determinant semantics
M2→M1 implementation-limit failure
q/NLL ranking disagreement under shared covariance
count-conservation failure
fit/test contamination
```

### 5. `GCTM_B1_CALIBRATION_ONLY`

Calibration passes but event-local ranking does not.

Maximum supported conclusion:

```text
the representation improves offline probabilistic normalization
or gap-conditioned calibration on the tested substrate
```

Blocked:

```text
no score-ranking design candidate
no O1 authorization
```

### 6. `GCTM_B1_NO_STABLE_RANKING_GAIN`

The study is valid but the ranking, stability, short-gap retention, or
interpretability bar is not met.

```text
current GCTM score-ranking path closes for the declared model,
observation, context, and complexity class
upstream H0/GCTM claims remain intact
no O1
```

### 7. `GCTM_B1_SCORE_DESIGN_CANDIDATE`

Every design bar passes.

Maximum supported conclusion:

```text
one frozen runtime-grounded offline GCTM score policy is eligible
for owner consideration as input to a separately declared O1
```

Not established:

```text
online retention
system efficacy
latency suitability
production safety
```

## Mainline transitions

| Terminal class | Mainline transition |
|:--|:--|
| provenance / observation / execution / implementation invalid | none; current experiment closes unresolved |
| calibration-only | closes score-ranking promotion for tested class |
| no stable ranking gain | closes current GCTM offline score path |
| score design candidate | adds a candidate ranking capability and makes O1 declaration eligible after owner acceptance |

## Deliverables

1. sealed declaration and sidecar;
2. frozen source/evidence inventory;
3. frozen fit/reveal manifest;
4. model and parameter artifacts;
5. runner and verifier identities;
6. event-level output packet;
7. fold/sequence result tables;
8. calibration diagnostics;
9. ranking and mechanism-attribution report;
10. terminal packet with maximum and blocked claims.

B1 must not modify the production preset or create an online hook.

## Canonical conclusion form

```text
Within <accepted H0 evidence identity>,
using <accepted GCTM spec identity>,
observation mode <...>,
parameterization family <...>,
candidate universe <...>,
and frozen fit/reveal protocol <...>:

Calibration:
  <passed / failed / invalid>

Candidate-local ranking:
  <minimum effect met / not met / invalid>

Mechanism attribution:
  <M1 uncertainty contribution>
  <M2 memory/context contribution>

Short-gap retention:
  <passed / failed / invalid>

Ordered terminal:
  <GCTM_B1_...>

Maximum supported conclusion:
  <bounded offline claim>

Not established:
  online retention
  system efficacy
  production suitability

O1 eligibility:
  <eligible after owner acceptance / not eligible>
```

## B1 → O1 handoff object

A positive B1 handoff must be a frozen policy object, not prose.

Minimum contents:

```text
b1_terminal_packet_hash
accepted_h0_evidence_id
accepted_gctm_spec_id
model_id
parameter_fit_id
observation_mode
causal_availability
coordinate_substrate_id
context_definition
context_fallback
missing_value_rule
score_transform
score_sign
score_weight
clamp_rule
precision
tie_breaking
eligible_event_rule
candidate_universe
expected_mechanism
offline_effect_bar
short_gap_retention_result
known_failure_strata
blocked_claims
```

O1 cannot alter this object while claiming to test the same B1 design. B1 may
define O1 eligibility but must not authoritatively define O1 execution; the
[O1 task charter](gctm_o1_online_intervention_efficacy_task.md) consumes the
accepted B1 handoff and must not rewrite B1 semantics.

## Pre-activation checklist

- [ ] accepted H0 runtime substrate exists;
- [ ] B1 declaration owner has accepted and bound an H0→GCTM consumer
      compatibility verdict to the required runtime-observable objects;
- [ ] accepted GCTM model specification exists;
- [ ] score-layer evidence contract (registry §7) exists;
- [ ] B1-slot identity relation resolved by owner;
- [ ] exact observation mode selected;
- [ ] one parameterization family selected;
- [ ] source/decision spaces and reductions defined;
- [ ] fit, blind, reveal, and no-refit rules frozen;
- [ ] primary ranking metric and minimum effect frozen;
- [ ] calibration separated from ranking;
- [ ] short-gap retention and stability bars frozen;
- [ ] ordered terminals exhaustive;
- [ ] every terminal maps to a state transition or explicit none;
- [ ] owner separately schedules activation.

## Read first

- [Synthesis core — shared B1/O1 semantics](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
- [H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
- [H0→GCTM consumer compatibility requirements](../../modules/semantic/research/h0_gctm_consumer_compatibility_requirements_20260718.md)
- [GCTM parked task charter](gap_conditioned_stochastic_transition_model_task.md)
- [O0 routing charter](closed/bridge_frozen_evidence_o0_routing_20260716.md)
- [claim-state registry（§7 score layer / §8 候選集）](../contracts/claim_state_registry.md)
- [bridge decision semantics](../tracker-decision/relink_bridge.md)
- [O1 task charter](gctm_o1_online_intervention_efficacy_task.md)

## Artifacts

None. This charter authorizes no execution and produces no evidence artifact.

## Current step

none — proposed; waiting on activation prerequisites and separate owner
scheduling. No probe is authorized.

## Acceptance

Owner accepts exactly one ordered `GCTM_B1_*` terminal from a future sealed
declaration → close per threads README; or the charter is discarded/superseded
before activation (route: declined, no execution).

## Must not

- activate, execute, fit, or reveal anything from this charter; **proposed ≠
  scheduled**; PR merge ≠ research acceptance.
- cite `GCTM_B1_*` as the B1 that H0 route 5 makes candidate before the owner
  resolves the B1-slot identity.
- treat this charter as satisfying the registry §7 score-layer contract
  prerequisite.
- restate shared §0/§1/§2/§35/§36/§39 semantics as a second truth; link the
  synthesis core instead.
- use `s0` to stand for a production quantity (registry inadmissibility).
- reinterpret this task onto the association (auction) stage.
- modify the production preset, create an online hook, or claim any system
  efficacy.

## History

- 2026-07-17: Opened as a proposed task charter by splitting the B1/O1
  objectives-and-semantics synthesis (landed in PR #178) per its §37
  repository split. Content carried over without semantic change; the B1-slot
  identity question and the registry §7 score-layer prerequisite are recorded
  as unresolved activation prerequisites. No activation, no seal, no
  registry/sole-active change.
- 2026-07-17: Review fix (PR #179 blockers): single-owner dedup — this charter
  is now the sole owner of the B1-slot identity question and the registry §7
  prerequisite (core §0 links here); the GPU foot-bridge hook scope is owned
  by core §0 and only consumed here. Frontmatter aligned to the canonical
  thread schema (`doc-promotion: navigation-only; not evidence`,
  `work-class: mainline-study`).
