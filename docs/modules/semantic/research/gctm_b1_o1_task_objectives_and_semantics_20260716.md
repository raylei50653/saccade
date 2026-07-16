<!-- doc-status: draft -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-16 -->
<!-- doc-module: semantic -->

# GCTM downstream tasks — B1 / O1 objectives and semantics

> **Working synthesis · not active · not sealed · no execution authority**
>
> This document consolidates the intended task objectives, evidence semantics,
> activation gates, claim boundaries, validity rules, and provisional terminal
> families for:
>
> - **B1** — runtime-grounded offline attribution and score-ranking evaluation;
> - **O1** — online score intervention and system-efficacy evaluation.
>
> It is source material for future, separately reviewed task charters and sealed
> declarations. It does not activate B1 or O1, select a model, freeze numerical
> thresholds, authorize data access, modify runtime behavior, or promote a
> production policy.

---

## 0. Alignment boundary (anti-drift; landed 2026-07-17)

This synthesis is landed for **semantic alignment only**: it fixes the intended
B1/O1 vocabulary so future charters do not drift. It grants no lifecycle state
and re-decides nothing owned elsewhere.

- **Upstream owners (unchanged).**
  [H0 declaration](headline_bridge_full_decision_capture_declaration_20260713.md)
  owns runtime observability / evidence fidelity;
  the [GCTM parked charter](../../../research/threads/gap_conditioned_stochastic_transition_model_task.md)
  owns the transition-family specification boundary. This document consumes
  both and re-decides neither.
- **B1 identity is an open owner decision.** H0 declaration §7 and the
  [O0 routing charter](../../../research/threads/bridge_frozen_evidence_o0_routing_20260716.md)
  (route 5) name a "separately declared **B1 consumer-faithful operating-curve
  study**". Whether the `GCTM_B1` study specified here occupies that same B1
  slot, coexists with it, or supersedes it is **not decided by this document**.
  Until an owner resolves that relation at charter time, `GCTM_B1_*` must not
  be cited as the B1 that H0 route 5 makes candidate.
- **Score-layer contract is a hard prerequisite.** B1 is an `L2 score` object.
  Per [claim-state registry §7](../../../research/contracts/claim_state_registry.md),
  the score-layer evidence contract (rank/margin/top-1 semantics and claim
  ladder) does not exist yet, so no B1 object can enter the admissible
  candidate set (`transition_semantics: unavailable`) regardless of H0/GCTM
  outcomes. Writing that contract does not depend on H0.
- **One online hook, one contract.** "Association/relink hook" spans two
  distinct online contracts: the association (auction) stage and the GPU
  foot-bridge two-stage winner
  ([bridge decision semantics](../../../research/tracker-decision/relink_bridge.md)).
  A future O1 declaration must pin exactly one. If the target is the bridge,
  the legal insertion surface is the candidate-local (stage-1) lost ranking
  under fixed pair eligibility; claim arbitration, loser fallback, and commit
  mutation are separate online-contract problems.
- **Reserved-symbol renames applied at landing** (the 2026-07-16 working draft
  used the left-hand forms):

  | Draft form | Landed form | Collision avoided |
  |:--|:--|:--|
  | \(s_0(i\mid e)\) base score | \(s_{\mathrm{base}}(i\mid e)\) | repo `s0` = offline proxy of `bdist`; registry forbids "s0 represents production `bdist`" |
  | `P0_model_id` | `P0_exit_cov_id` | P0 = sealed identifiability study code |
  | `R1_model_id` | `R1_obs_cov_id` | R1 = sealed capture-replay study code |
  | "ambiguous band" (unqualified) | runtime-coordinate band, defined by the future declaration | Door 0 closed the s0-proxy band class (`T2_NO_USABLE_RANKING_POWER_IN_CLASS`, class-scoped) |

  The `GCTM_B1_*` / `GCTM_O1_*` terminal prefixes and the `gctm_b1_*` /
  `gctm_o1_*` future file names are kept precisely so they cannot be confused
  with the closed `m_b1_*` (M-B1) line.

---

## 1. Evidence chain

```text
H0
runtime observability / evidence fidelity

→ GCTM
transition-family mathematical specification

→ B1
runtime-grounded offline attribution and score-ranking evaluation

→ O1
online intervention and system-efficacy evaluation

→ production evaluation
deployment suitability and promotion
```

Every arrow is a separate evidence edge. No upstream result automatically
authorizes the next task.

### 1.1 Claim ownership

| Object | Owns | Does not establish |
|:--|:--|:--|
| H0 | Whether runtime quantities and decision events are faithfully observable | Whether a stochastic model is mathematically valid or useful |
| GCTM | Whether M0/M1/M2 and the observation interface are well-posed and sealable | Empirical calibration, ranking value, online value |
| B1 | Whether one frozen GCTM instantiation has stable runtime-grounded offline ranking value | Online retention or MOT improvement |
| O1 | Whether one frozen B1 policy survives causal online execution and improves the actual system | Production safety or default promotion |
| Production evaluation | Latency, rollback, deployment risk, broad operational acceptance | Upstream scientific claims beyond its scope |

### 1.2 Non-retroactivity

- A valid negative B1 result does not invalidate H0 or the abstract GCTM
  mathematics. It closes the declared offline score-ranking path for the tested
  model class.
- A valid negative O1 result does not invalidate a valid B1 offline claim. It
  closes the declared online transport/intervention path for the tested hook and
  policy.
- An invalid experiment closes the current execution, not the scientific
  hypothesis path.
- A downstream positive result cannot repair an upstream provenance,
  identifiability, or fidelity failure.
- Engineering merge does not equal research acceptance.
- O1 acceptance does not equal production promotion.

---

## 2. Shared semantic rules

### 2.1 Target layer

Both B1 and O1 target **score-ranking**, not a coarse reject gate.

The intended intervention has the form:

\[
s_{\mathrm{new}}(i\mid e)
=
s_{\mathrm{base}}(i\mid e)
+
\Delta s_{\mathrm{GCTM}}(i\mid e),
\]

where \(e\) is one candidate event and \(i\) is one candidate in that event.
The base score is written \(s_{\mathrm{base}}\), never `s0`: in this repository
`s0` is the reserved name of the offline proxy of `bdist`, and any use of `s0`
to stand for a production quantity is registry-inadmissible (§0).

The GCTM signal changes relative preference inside the retained ambiguous band
(a runtime-coordinate band that the future declaration must define; it is not
the closed Door 0 s0-proxy band class, §0).
It must not become a hard rejection rule unless a later task is explicitly
re-chartered for the gate layer.

### 2.2 Transition likelihood is not a GT posterior

The model quantity is of the form:

\[
p(y_1\mid z_0,\Delta,c,M),
\]

or a score derived from that conditional transition model.

It is not automatically:

\[
P(\mathrm{GT}\mid y_1,z_0,\Delta,c).
\]

A transition likelihood may rank candidates, calibrate expected motion
deviation, or expose a failure mode. It must not be described as a calibrated
identity probability without a separately specified discriminative model and
calibration claim.

### 2.3 Four distinct claim spaces

```text
calibration space
  Does q or NLL have stable probabilistic meaning across gaps and contexts?

candidate-ranking space
  Does the score order the GT candidate above competing candidates in one event?

assignment space
  Does the changed ordering alter the selected association or relink decision?

system space
  Do altered decisions improve track-level and sequence-level MOT outcomes?
```

Improvement in one space does not imply improvement in the next.

### 2.4 Output classes

Each result must be classified as exactly one of:

- **design candidate** — purpose-aligned, interpretable, structurally simple,
  stability-validated, and above a predeclared utility bar;
- **performance upper-bound candidate** — estimates capability but carries no
  design authority;
- **diagnostic result** — calibration map, attribution, identifiability verdict,
  failure mode, or exceptional-tail description;
- **unexplained residual set** — unresolved support that must not be force-fit
  with additional conditions.

Only a B1 **design candidate** may make O1 eligible.

### 2.5 Validity versus efficacy

Both tasks must distinguish:

```text
INVALID / UNRESOLVED
  The experiment could not answer the question.

VALID NEGATIVE
  The experiment answered the question and the minimum effect was not met.

VALID POSITIVE
  The predeclared effect, stability, and mechanism bars were met.
```

There is no residual “describe more and continue” outcome in a sealed
mainline-capable declaration.

### 2.6 Dual-space and reduction typing

Every future declaration must name:

1. source space;
2. decision space;
3. reduction \(\rho\);
4. aggregation rule;
5. dependence structure;
6. conservation identities;
7. type \(\kappa\) for every decidable claim:

\[
\kappa
=
(\text{quantification space},
 \text{comparison relation},
 \text{decision rule}).
\]

Fidelity, calibration, ranking, assignment, and system metrics require separate
\(\kappa\)-objects.

---

# Part I — B1

## 3. B1 identity

Recommended task name:

> **B1 — Runtime-grounded offline attribution and score-ranking evaluation of
> the gap-conditioned stochastic transition model**

Recommended lifecycle before activation:

```text
status: proposed / parked
target-decision-layer: score-ranking
primary-intent: design evaluation
secondary-intents:
  capability map
  calibration diagnostic
work-class: offline-model-evaluation
output-classes:
  design candidate
  diagnostic result
  unexplained residual set
```

The primary intent is **design evaluation** because only a successful B1 design
candidate can justify O1. Calibration and attribution are mandatory diagnostics,
not substitutes for ranking value.

---

## 4. B1 activation gate

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

Additionally, B1 is an `L2 score` object: the score-layer evidence contract
named by claim-state registry §7 must exist before any B1 object can enter the
admissible candidate set (§0). That contract is a separate prerequisite; none
of the eight conditions above substitutes for it.

A negative H0 terminal blocks bridge-runtime B1 unless a new evidence substrate
or explicit substrate-agnostic re-charter is accepted.

---

## 5. B1 research question

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

---

## 6. B1 maximum supported claim

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

---

## 7. B1 scope

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

---

## 8. B1 frozen inputs

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
P0_exit_cov_id      # exit-state covariance P0; field renamed to avoid the P0 study code (§0)
R1_obs_cov_id       # entry-observation covariance R1; field renamed to avoid the R1 study code (§0)
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

---

## 9. B1 evidence spaces and reductions

### 9.1 Source pair space

\[
\mathcal U_{\mathrm{pair}}
=
\{(e,i): i\in C_e\}.
\]

Each source record contains the frozen runtime-grounded state, candidate
observation, gap, context, and model outputs.

### 9.2 Candidate-event space

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

### 9.3 Calibration space

\[
\mathcal U_{\mathrm{cal}}
=
\{(e,i^\star): i^\star \text{ is the declared correct transition}\}.
\]

Calibration statistics do not become ranking statistics through aggregation.

### 9.4 Trial and domain space

The declaration must select a trial unit such as:

```text
candidate event
track
sequence
fold
```

and name residual clustering above that unit.

### 9.5 Conservation identities

At minimum:

```text
every admitted pair belongs to exactly one event
every event belongs to exactly one sequence and one fold
pair counts reconcile with event candidate counts
GT-present / GT-absent event partitions reconcile
fit, validation, and test partitions are disjoint
all dropped records have one enumerated reason
```

---

## 10. B1 model-comparison semantics

### 10.1 M0

M0 is the frozen deterministic baseline. It has no automatic probability
interpretation.

### 10.2 M1

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

### 10.3 M2

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

### 10.4 q and NLL invariant

When all candidates in one event share covariance, dimension, gap, context, and
observation mode:

\[
\operatorname{rank}(q_i)
=
\operatorname{rank}(\mathrm{NLL}_i).
\]

Any implementation reporting different rankings in this condition fails an
implementation-validity invariant.

---

## 11. B1 fitting, blind, and reveal boundary

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

---

## 12. B1 headline metrics

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

---

## 13. B1 positive design bar

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

---

## 14. Recommended B1 terminal family

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

---

## 15. B1 mainline transitions

| Terminal class | Mainline transition |
|:--|:--|
| provenance / observation / execution / implementation invalid | none; current experiment closes unresolved |
| calibration-only | closes score-ranking promotion for tested class |
| no stable ranking gain | closes current GCTM offline score path |
| score design candidate | adds a candidate ranking capability and makes O1 declaration eligible after owner acceptance |

---

## 16. B1 deliverables

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

---

## 17. B1 canonical conclusion form

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

---

# Part II — O1

## 18. O1 identity

Recommended task name:

> **O1 — Online GCTM score intervention and system-efficacy evaluation**

Recommended lifecycle before activation:

```text
status: proposed / parked
target-decision-layer: score-ranking
primary-intent: design evaluation
work-class: online-intervention-evaluation
output-classes:
  design candidate
  diagnostic result
  unexplained residual set
```

O1 is not a second offline ranking study. Its question is whether a frozen B1
policy survives actual causal execution and changes the system outcome.

---

## 19. O1 activation gate

O1 may become active only after:

1. owner acceptance of `GCTM_B1_SCORE_DESIGN_CANDIDATE`;
2. one frozen B1 policy identity;
3. one online-causal observation mode;
4. exact parameter, context, normalization, fallback, and score-composition
   artifacts;
5. exact online hook placement and execution ordering;
6. sealed default-off and comparison protocol;
7. a sealed O1 declaration;
8. separate owner scheduling.

If B1 relied on `offline-diagnostic-only` fields, O1 is blocked. A causal
variant must return to B1 and earn a positive design terminal under the
online-compatible observation interface.

No B1 terminal auto-activates O1.

---

## 20. O1 research question

> When the frozen B1 GCTM score policy is executed at the real online
> association/relink hook, does it retain the intended event-local mechanism
> and produce a stable improvement in system-level tracking outcomes, while the
> disabled path reproduces baseline behavior and runtime costs remain within
> predeclared limits?

O1 contains four nested questions.

### O1-Q1 — Structural equivalence

Does installing the hook while disabled preserve baseline behavior?

### O1-Q2 — Observation and score fidelity

Are online state, score inputs, and score outputs the same semantic objects
accepted by B1?

### O1-Q3 — Intervention retention

Does enabling the score change candidate ordering and assignments in the
predicted direction?

### O1-Q4 — System efficacy

Do those changes improve track-level and sequence-level outcomes without
violating latency or safety guards?

---

## 21. O1 maximum supported claim

The strongest legal O1 conclusion is:

> The frozen GCTM score policy is an **online-retained system-efficacy
> candidate** under the declared hook, preset, runtime, data scope, and
> operational limits.

O1 does not establish:

```text
production safety across undeclared deployments
rollback readiness unless separately evaluated
default-on authorization
broad detector/model portability
production promotion
```

A positive O1 may only make a separately declared production evaluation
eligible.

---

## 22. O1 scope

### In scope

- implement one frozen B1 score policy at one exact online hook;
- preserve default-off behavior;
- expose shadow and applied-action audit;
- compare baseline, disabled, shadow, and enabled execution;
- measure online candidate ordering and assignment flips;
- measure MOT system outcomes;
- measure latency, FPS, memory, determinism, and failure modes;
- analyze per-sequence and repeated-run retention;
- select one bounded terminal.

### Out of scope

```text
no online model refit
no threshold or weight search on test sequences
no candidate-universe redesign
no hard-gate conversion
no hidden fallback change
no B1 reinterpretation after online results
no production default change
no broad production claim
no unrelated tracker optimization
no simultaneous assignment-layer redesign
```

---

## 23. O1 frozen policy identity

O1 consumes exactly one B1 design object:

```text
b1_terminal
b1_terminal_packet_hash
b1_policy_id
b1_model_id
b1_parameter_fit_id
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
```

O1 must not replace an offline-only observable with a “close enough” online
proxy. That substitution is a new substrate edge and requires fidelity evidence
plus renewed B1 acceptance.

---

## 24. O1 intervention semantics

The default intervention is a score correction:

\[
s_{\mathrm{on}}(i\mid e)
=
s_{\mathrm{base}}(i\mid e)
+
\lambda\,g_{\mathrm{GCTM}}(i\mid e),
\]

with \(\lambda\), sign, clamping, missing-value behavior, and precision frozen
before execution.

The intervention must declare:

```text
hook file and function
position relative to candidate construction
position relative to proposal and commit
execution stream/order
state read set
state write set
eligible-event rule
candidate score read
GCTM score write
tie-breaking
fallback
default-off path
audit record
```

"Proposal" and "commit" above carry the meanings of the audited online
contract (candidate-local ranking → per-lost detection-score claim → commit);
see the [bridge decision semantics](../../../research/tracker-decision/relink_bridge.md)
and §0.

The hook must not silently:

```text
remove candidates
create candidates
alter track state before the declared decision
change the candidate universe
inspect GT/FP labels
use future frames
refit model parameters
select a new weight
```

---

## 25. O1 comparison arms

### R — reference baseline

Accepted baseline without the O1 intervention branch.

### D — hook installed, disabled

Integrated code path with intervention disabled.

```text
R vs D
  structural/default-off equivalence
```

### S — shadow computation

GCTM score is computed and logged but does not modify ranking.

```text
D vs S
  observation/computation/timing perturbation
```

### A — applied intervention

Frozen GCTM score correction is applied.

```text
S vs A
  policy-action effect

R vs A
  total online system effect
```

The declaration must freeze which arms are mandatory before execution.

---

## 26. O1 online evidence spaces and reductions

### Runtime candidate space

\[
\mathcal U_{\mathrm{pair}}^{\mathrm{on}}
=
\{(r,e,i)\}.
\]

### Assignment-decision reduction

\[
\rho_{\mathrm{pair}\rightarrow\mathrm{decision}}^{\mathrm{on}}
:
\{s(r,e,i)\}_{i\in C_e}
\mapsto
\text{selected / rejected / unchanged}.
\]

### State-transition chain

```text
candidate scores
→ candidate ordering
→ assignment/relink decision
→ track-state transition
→ frame/track outcomes
→ sequence metrics
→ run-level verdict
```

Event-level improvement alone cannot establish system efficacy.

### Dependence unit

System inference must treat sequence/run structure explicitly. Frame and event
rows must not be treated as independent system trials.

### Conservation and audit identities

```text
every eligible event has one audit outcome
applied + skipped + invalid = eligible events
every score change maps to one candidate/event key
every assignment flip maps to its pre/post selected candidate
future-state divergences trace to an earlier applied action
disabled-path candidate and decision counts reconcile with baseline
sequence/run metric tables reconcile with raw outputs
```

---

## 27. O1 validity gates

### 27.1 Provenance gate

- B1 terminal and policy hashes match;
- runtime head, build, preset, data, and evaluator identities match;
- no unsealed code or policy drift.

### 27.2 Default-off equivalence gate

The disabled path must reproduce baseline under the predeclared relation.

### 27.3 Shadow perturbation gate

Shadow computation must not change policy-visible state. Timing perturbation
must remain within the declared bound.

### 27.4 Online observation-fidelity gate

- every B1-required field is causally available;
- values match the accepted online semantic definition;
- no offline reconstruction silently replaces runtime state;
- missing/fallback behavior matches B1.

### 27.5 Exposure and headroom gate

- enough eligible events occur;
- candidate events contain ranking headroom;
- intervention is not zero by construction;
- the primary system metric has non-trivial headroom;
- repeated runs support the declared variability model.

### 27.6 Runtime gate

- no crash or invalid packet;
- repeatability meets the declared rule;
- latency, FPS, memory, and CUDA-capture behavior remain within the frozen
  execution contract.

Only after these gates pass may lack of gain be interpreted as a valid negative.

---

## 28. O1 headline metrics

### System metrics

- AssA;
- HOTA;
- IDF1;
- identity switches;
- MOTA;
- FP/FN;
- recoverable/irrecoverable track outcomes.

Association-sensitive metrics should normally be primary.

### Mechanism-retention metrics

```text
eligible online events
applied score corrections
ordering changes
GT rank changes
assignment/relink flips
predicted-beneficial / predicted-harmful flips
flip-to-track-outcome attribution
per-gap and per-context retention
B1-predicted direction agreement
```

### Runtime metrics

```text
FPS and frame latency
p50/p95/p99 stage latency
GPU memory
build/runtime failures
CUDA graph/capture compatibility
repeated-run variance
baseline-disabled equivalence
```

### Per-domain reporting

```text
sequence
run
gap stratum
context stratum
intervention exposure
```

---

## 29. O1 positive bar

A positive O1 candidate must satisfy:

1. all validity gates;
2. disabled-path equivalence;
3. online observation and score fidelity;
4. sufficient intervention exposure and system headroom;
5. B1 mechanism retention;
6. the predeclared minimum system effect;
7. no guardrail violation;
8. per-sequence and repeated-run stability;
9. runtime-cost limits;
10. mechanism-localized assignment/track attribution;
11. no post-hoc weight, threshold, or fallback change.

A total metric gain without mechanism retention is not evidence for the GCTM
intervention.

---

## 30. Recommended O1 terminal family

### 1. `GCTM_O1_PROVENANCE_INVALID`

B1 policy identity, runtime head, build, preset, data, runner, or artifact
binding is invalid.

### 2. `GCTM_O1_EXECUTION_INVALID`

Build failure, crash, serialization failure, incomplete run, non-reproducible
execution, or invalid evaluator output.

### 3. `GCTM_O1_BASELINE_EQUIVALENCE_INVALID`

The hook-disabled path does not reproduce the frozen reference baseline.

```text
no efficacy verdict
B1 not falsified
```

### 4. `GCTM_O1_ONLINE_SUBSTRATE_INVALID`

Online fields, temporal reduction, candidate universe, context, missing
behavior, score computation, or hook semantics do not match B1.

```text
B1→O1 transport edge fails
no online efficacy verdict
requires a new fidelity edge, B1 revalidation, or O1 re-charter
```

### 5. `GCTM_O1_EXPOSURE_OR_HEADROOM_INSUFFICIENT`

The valid run contains too little eligible exposure, ranking headroom, system
headroom, or independent run/domain support.

```text
current O1 study unresolved
does not exhaust the intervention hypothesis
```

### 6. `GCTM_O1_INTERVENTION_HARMFUL`

The valid intervention violates an association, identity, system, or runtime
guardrail.

```text
current online policy/hook path closes
must remain default-off
B1 may remain valid
no production eligibility
```

### 7. `GCTM_O1_ONLINE_RETENTION_WITHOUT_SYSTEM_GAIN`

The B1 event-local mechanism is retained online, but the minimum system effect
is not met.

```text
mechanism retained
system did not convert it into useful MOT gain
no production eligibility
```

A future assignment-layer diagnosis requires a separate task.

### 8. `GCTM_O1_EFFICACY_NOT_RETAINED`

The valid intervention does not retain the B1 mechanism or does not produce
stable online benefit, without crossing the harmful terminal.

```text
current B1→O1 transport/intervention path closes
B1 remains an offline result
no production eligibility
```

### 9. `GCTM_O1_SYSTEM_EFFICACY_CANDIDATE`

Every positive bar passes.

Maximum supported conclusion:

```text
one frozen GCTM score policy is retained online and is eligible
for owner consideration as input to a separately declared production evaluation
```

Not established:

```text
default-on authorization
deployment safety
broad substrate portability
rollback readiness
production promotion
```

---

## 31. O1 mainline transitions

| Terminal class | Mainline transition |
|:--|:--|
| provenance / execution / equivalence / substrate invalid | none; current experiment closes unresolved |
| insufficient exposure/headroom | none; current experiment could not answer |
| harmful | closes current online intervention path and keeps it default-off |
| retention without system gain | closes production path for this score hook; may expose a separately chartered assignment bottleneck |
| efficacy not retained | closes current B1→O1 transport/intervention path |
| system-efficacy candidate | adds an online-retained decision capability and makes production evaluation eligible after owner acceptance |

---

## 32. O1 deliverables

1. sealed O1 declaration and sidecar;
2. frozen B1 policy package;
3. runtime implementation and default-off hook;
4. build/runtime provenance manifest;
5. reference/disabled/shadow/applied run inventory;
6. event and assignment audit packet;
7. track/sequence/run outcome tables;
8. latency and failure-mode report;
9. baseline equivalence verifier;
10. terminal packet with maximum and blocked claims.

The production preset remains unchanged unless a later production declaration
explicitly authorizes a change.

---

## 33. O1 canonical conclusion form

```text
Using B1 policy <identity>,
at online hook <identity>,
under runtime/build/preset/data identities <...>:

Reference vs disabled:
  <equivalent / invalid>

Disabled vs shadow:
  <non-perturbing / perturbing / invalid>

Shadow vs applied:
  <mechanism retained / not retained / harmful / invalid>

System outcome:
  <minimum effect met / not met / harmful / unresolved>

Runtime guardrails:
  <passed / failed / invalid>

Ordered terminal:
  <GCTM_O1_...>

Maximum supported conclusion:
  <bounded online claim>

Not established:
  production safety
  default-on authority
  broad portability

Production-evaluation eligibility:
  <eligible after owner acceptance / not eligible>
```

---

# Part III — Handoff and closure

## 34. B1 → O1 handoff object

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

O1 cannot alter this object while claiming to test the same B1 design.

---

## 35. Outcome interpretation matrix

| B1 outcome | O1 status | Meaning |
|:--|:--|:--|
| invalid/unresolved | blocked | B1 did not answer the offline question |
| calibration-only | blocked | no ranking design candidate |
| no stable ranking gain | blocked | offline score path closed |
| score design candidate | eligible after owner acceptance | O1 may be separately declared |
| score design candidate using offline-only observation | blocked | causal variant must return through B1 |
| positive B1 + invalid O1 | unresolved online edge | B1 remains valid |
| positive B1 + harmful O1 | online path closed/default-off | B1 remains offline-valid |
| positive B1 + retention without system gain | mechanism exists but a system bottleneck remains | no production eligibility |
| positive B1 + positive O1 | production evaluation eligible | still no production promotion |

---

## 36. Forbidden inference shortcuts

\[
\text{GCTM theory sealable}
\not\Rightarrow
\text{B1 empirical value}
\]

\[
\text{better calibration}
\not\Rightarrow
\text{better candidate ranking}
\]

\[
\text{better candidate ranking}
\not\Rightarrow
\text{assignment change}
\]

\[
\text{assignment change}
\not\Rightarrow
\text{better MOT outcome}
\]

\[
\text{offline ranking gain}
\not\Rightarrow
\text{online causal availability}
\]

\[
\text{online mechanism retention}
\not\Rightarrow
\text{system gain}
\]

\[
\text{online system gain}
\not\Rightarrow
\text{production promotion}
\]

\[
\text{engineering merge}
\not\Rightarrow
\text{research acceptance}
\]

---

## 37. Future repository split

This synthesis should not become the sealed owner for both studies. When the
tasks are scheduled, split it into two independent task objects:

```text
docs/research/threads/
  gctm_b1_runtime_grounded_offline_attribution_task.md

docs/research/threads/
  gctm_o1_online_intervention_efficacy_task.md
```

Each task owns its own:

```text
activation state
declaration
frozen degrees of freedom
validity gates
ordered terminals
evidence artifacts
owner terminal acceptance
```

B1 may define O1 eligibility but must not authoritatively define O1 execution.
O1 consumes the accepted B1 handoff and must not rewrite B1 semantics.

---

## 38. Pre-activation checklist

### B1

- [ ] accepted H0 runtime substrate exists;
- [ ] accepted GCTM model specification exists;
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

### O1

- [ ] owner-accepted B1 score design candidate exists;
- [ ] policy fields are online-causal;
- [ ] frozen B1 policy package exists;
- [ ] exact hook and execution order selected;
- [ ] reference/disabled/shadow/applied arms frozen;
- [ ] default-off equivalence relation frozen;
- [ ] online fidelity and exposure gates frozen;
- [ ] primary system metric and guardrails frozen;
- [ ] runtime cost bars frozen;
- [ ] ordered terminals exhaustive;
- [ ] production preset remains unchanged;
- [ ] owner separately schedules activation.

---

## 39. Final semantic summary

```text
B1 asks:
  Is the stochastic transition representation a real,
  stable, interpretable offline ranking capability
  on a runtime-grounded substrate?

O1 asks:
  Does one frozen B1 ranking policy survive causal online execution
  and improve the actual tracking system?

B1 positive:
  O1 may be declared.

O1 positive:
  production evaluation may be declared.

No result auto-promotes.
No invalid experiment becomes a negative scientific verdict.
No offline claim is silently transported online.
No online candidate becomes production behavior without a new acceptance edge.
```
