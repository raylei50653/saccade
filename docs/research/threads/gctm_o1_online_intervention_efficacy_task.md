---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-17
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "owner-accepted GCTM_B1_SCORE_DESIGN_CANDIDATE + frozen B1 policy package + sealed O1 declaration + separate owner scheduling"
target-decision-layer: score-ranking
primary-intent: design-evaluation
output-class: "design candidate | diagnostic result | unexplained residual set"
mainline-transition: "none from this charter; per-terminal transitions listed inside"
created: 2026-07-17
---

# GCTM O1 — online score intervention and system-efficacy evaluation — proposed task charter

## Status and authority

**PROPOSED / non-WIP.** This charter is the O1 task object split out of the
[B1/O1 objectives-and-semantics synthesis](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
on 2026-07-17, executing that document's §37 repository split. It is **not
active**, **not sealed**, **not sole-active**, and does not occupy semantic WIP
(the WIP lock is empty, per [semantic TODO](../../modules/semantic/TODO.md)). It
does not activate O1, implement any hook, freeze numerical thresholds,
authorize data access, modify runtime behavior, or promote a production
policy.

Authority is intentionally split:

- **this charter** owns O1's task identity, activation prerequisites, scope,
  frozen policy-identity requirements, intervention and comparison semantics,
  validity gates, provisional terminal family, and deliverables;
- the [synthesis core](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
  owns the **shared** cross-task semantics (§0 alignment boundary, including
  the GPU foot-bridge hook scope and reserved-symbol renames; evidence chain
  and claim ownership §1; shared semantic rules §2; outcome interpretation
  matrix §35; forbidden inference shortcuts §36; final semantic summary §39) —
  this charter consumes them and does not restate them as a second truth;
- the [B1 task charter](gctm_b1_runtime_grounded_offline_attribution_task.md)
  owns the B1→O1 handoff object shape; O1 consumes the accepted B1 handoff and
  must not rewrite B1 semantics;
- [`docs/modules/semantic/TODO.md`](../../modules/semantic/TODO.md) owns only a
  one-line navigation pointer and the module WIP projection.

O1 is not a second offline ranking study. Its question is whether a frozen B1
policy survives actual causal execution and changes the system outcome.

## Hook scope: GPU foot-bridge only

The cross-task hook scope is **owned by the
[synthesis core §0](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)**:
O1's intervention contract is scoped end-to-end to the GPU foot-bridge
two-stage winner, and the association (auction) stage is out of scope. This
charter consumes that boundary and does not restate its full rules; see core
§0 for the legal insertion surface and the association-stage exclusion.

## Task identity

> **O1 — Online GCTM score intervention and system-efficacy evaluation**

Terminal prefixes are `GCTM_O1_*` and future evidence files are `gctm_o1_*`.

## Activation gate

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
No `GCTM_D1_*` diagnostic terminal is a B1 terminal, policy handoff, O1
prerequisite, or authority source. Diagnostic seal, bounded no-go, and
interface-ready outcomes leave O1 `proposed`.

## Research question

> When the frozen B1 GCTM score policy is executed at the real online bridge
> relink hook (hook scope above), does it retain the intended event-local
> mechanism and produce a stable improvement in system-level tracking
> outcomes, while the disabled path reproduces baseline behavior and runtime
> costs remain within predeclared limits?

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

## Maximum supported claim

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

## Scope

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

## Frozen policy identity

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

## Intervention semantics

The default intervention is a score correction:

\[
s_{\mathrm{on}}(i\mid e)
=
s_{\mathrm{base}}(i\mid e)
+
\lambda\,g_{\mathrm{GCTM}}(i\mid e),
\]

with \(\lambda\), sign, clamping, missing-value behavior, and precision frozen
before execution. The base score is written \(s_{\mathrm{base}}\), never `s0`
(registry inadmissibility; synthesis core §0/§2.1).

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
see the [bridge decision semantics](../tracker-decision/relink_bridge.md) and
the hook-scope section above.

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

## Comparison arms

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

## Online evidence spaces and reductions

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

## Validity gates

### Provenance gate

- B1 terminal and policy hashes match;
- runtime head, build, preset, data, and evaluator identities match;
- no unsealed code or policy drift.

### Default-off equivalence gate

The disabled path must reproduce baseline under the predeclared relation.

### Shadow perturbation gate

Shadow computation must not change policy-visible state. Timing perturbation
must remain within the declared bound.

### Online observation-fidelity gate

- every B1-required field is causally available;
- values match the accepted online semantic definition;
- no offline reconstruction silently replaces runtime state;
- missing/fallback behavior matches B1.

### Exposure and headroom gate

- enough eligible events occur;
- candidate events contain ranking headroom;
- intervention is not zero by construction;
- the primary system metric has non-trivial headroom;
- repeated runs support the declared variability model.

### Runtime gate

- no crash or invalid packet;
- repeatability meets the declared rule;
- latency, FPS, memory, and CUDA-capture behavior remain within the frozen
  execution contract.

Only after these gates pass may lack of gain be interpreted as a valid negative.

## Headline metrics

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

## Positive bar

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

## Provisional terminal family

This is a **provisional, not sealed** terminal shape. A sealed O1 declaration
must fix the exhaustive order and decision procedure. No terminal is selected
by this charter.

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

## Mainline transitions

| Terminal class | Mainline transition |
|:--|:--|
| provenance / execution / equivalence / substrate invalid | none; current experiment closes unresolved |
| insufficient exposure/headroom | none; current experiment could not answer |
| harmful | closes current online intervention path and keeps it default-off |
| retention without system gain | closes production path for this score hook; may expose a separately chartered assignment bottleneck |
| efficacy not retained | closes current B1→O1 transport/intervention path |
| system-efficacy candidate | adds an online-retained decision capability and makes production evaluation eligible after owner acceptance |

## Deliverables

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

## Canonical conclusion form

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

## Pre-activation checklist

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

## Read first

- [Synthesis core — shared B1/O1 semantics](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md)
- [B1 task charter（含 B1→O1 handoff object）](gctm_b1_runtime_grounded_offline_attribution_task.md)
- [bridge decision semantics](../tracker-decision/relink_bridge.md)
- [H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
- [claim-state registry](../contracts/claim_state_registry.md)

## Artifacts

None. This charter authorizes no execution and produces no evidence artifact.

## Current step

none — proposed; blocked behind B1 (which is itself proposed and blocked
behind its own activation prerequisites). No probe is authorized.

## Acceptance

Owner accepts exactly one ordered `GCTM_O1_*` terminal from a future sealed
declaration → close per threads README; or the charter is discarded/superseded
before activation (route: declined, no execution).

## Must not

- activate, implement, or execute anything from this charter; **proposed ≠
  scheduled**; PR merge ≠ research acceptance.
- start O1 from any B1 terminal other than owner-accepted
  `GCTM_B1_SCORE_DESIGN_CANDIDATE`.
- rewrite B1 semantics or alter the frozen B1 handoff object.
- restate shared §0/§1/§2/§35/§36/§39 semantics as a second truth; link the
  synthesis core instead.
- use `s0` to stand for a production quantity (registry inadmissibility).
- reinterpret this task onto the association (auction) stage.
- change the production preset or claim default-on / production promotion.

## History

- 2026-07-17: Opened as a proposed task charter by splitting the B1/O1
  objectives-and-semantics synthesis (landed in PR #178) per its §37
  repository split. Content carried over without semantic change. No
  activation, no seal, no registry/sole-active change.
- 2026-07-17: Review fix (PR #179 blockers): the GPU foot-bridge hook scope is
  owned by core §0 and only consumed here (full restatement removed).
  Frontmatter aligned to the canonical thread schema
  (`doc-promotion: navigation-only; not evidence`,
  `work-class: mainline-study`).
