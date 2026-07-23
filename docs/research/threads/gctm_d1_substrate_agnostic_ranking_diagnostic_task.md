---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-23
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "separate owner scheduling"
target-decision-layer: diagnostic-only
primary-intent: model-and-interface-diagnostic
output-class: "diagnostic seal | bounded no-go | interface-ready"
mainline-transition: "GCTM_D1 only; never H0_ROUTE5_B1, GCTM_B1, or GCTM_O1"
created: 2026-07-23
---

# GCTM D1 — substrate-agnostic ranking diagnostic — proposed task charter

## Status and authority

**PROPOSED / non-WIP / declaration owner-accepted / execution unscheduled.**
`GCTM_D1` is an independent, substrate-agnostic diagnostic slot. Its sealed
declaration is owner-accepted as a frozen execution contract
(`gctm_d1_declaration_owner_acceptance_20260723`). The charter itself is **not
active**, **not closed**, and does **not** occupy the semantic WIP lock.
Canonical registry `state` remains **`none`**. Creating or accepting the
declaration does not authorize data access, fitting, runtime capture, H0
re-entry, B1/O1 activation, or a production-facing claim.

```text
declaration accepted     = yes (execution contract frozen)
execution unscheduled    = yes
WIP not acquired         = yes
canonical state          = none
blocked_by               = owner_scheduling
```

The machine-readable identity and authority boundary is owned by
[`gctm_b1_slot_identity_decision_v1`](../contracts/gctm_b1_slot_identity_decision_v1.json):

```text
GCTM_B1 != H0_ROUTE5_B1
GCTM_B1 coexists with H0_ROUTE5_B1
GCTM_D1 is isolated from both runtime-grounded slots
```

`GCTM_D1` was chosen instead of a second `B1` name so its diagnostic-only
authority cannot be mistaken for runtime-grounded B1 activation authority.

## Research question

> Within a declared substrate-agnostic observation and parameterization
> family, which GCTM ranking properties are well-defined, invariant,
> identifiable, falsifiable, and interface-ready before any claim is made
> about H0 capture or runtime fidelity?

This question is about mathematical and diagnostic behavior. It is not the
runtime-grounded offline ranking question owned separately by
`H0_ROUTE5_B1` or `GCTM_B1`.

## Allowed scope

The charter may define and compare:

- theoretical observation families;
- parameterization families;
- event-local ranking invariants;
- monotonicity, ordering, scale, and dominance guards;
- bounded counterexamples and falsification cases;
- diagnostic evaluation on synthetic inputs or explicitly sealed non-runtime
  inputs;
- identifiability and observability limits;
- the consumer interface required from a future runtime substrate;
- H0 producer/consumer schema and compatibility requirements.

The charter may not freeze a runtime observation mode or runtime
parameterization. A diagnostic family definition is not a runtime freeze.

### Pre-activation seal-candidate generation (explicitly allowed)

Before charter activation and before any WIP/scheduling decision, the
repository may generate a **pre-activation synthetic seal-candidate package**:

```text
generation_kind = pre_activation_synthetic_seal_candidate
substrate        = synthetic fixtures only
status           = SEAL_CANDIDATE_GENERATED
authority        = owner-reviewable proposal, not charter execution
```

This generation:

- **may** run machine-checkable invariants on synthetic fixtures;
- **may** emit a provisional selected terminal string for owner review;
- **must not** be recorded as an owner-accepted charter execution;
- **must not** perform a canonical registry terminal state transition;
- **must not** acquire WIP or unlock B1/O1/H0.

Owner-accepted charter **execution** remains gated by:

1. ~~owner acceptance of the sealed D1 declaration and bounded terminal procedure~~
   **satisfied** — `gctm_d1_declaration_owner_acceptance_20260723`;
2. a separate owner scheduling decision that assigns WIP
   (**remaining gate**).

Seal-candidate generation, declaration acceptance, and owner-accepted execution
are distinct authorities.

## Allowed evidence

Only the following evidence classes are admissible:

```text
GCTM theory seal
accepted score_ranking_evidence_contract_v1
synthetic fixtures
sealed non-runtime datasets
machine-checkable invariants
bounded counterexamples
schema validation results
```

Every non-synthetic input must declare an immutable input identity, checksum,
schema, purpose, and explicit `non_runtime` substrate class. Sealing such an
input prevents within-diagnostic drift; it does not make the input an H0
runtime substrate.

## Claim firewall

No `GCTM_D1` declaration, result, terminal, fixture, dataset, or validation
report may claim or imply:

```text
runtime faithful
online-grounded
equivalent to H0 capture
H0→GCTM compatibility completed
eligible to activate H0_ROUTE5_B1 or GCTM_B1
decision-relevant transition for either runtime B1
replacement for H0 provenance, evidence identity, or checksum
automatic O1 eligibility or activation
automatic WIP acquisition
parked-thread return to mainline
```

Diagnostic evidence has no authority to satisfy any of these runtime gate
classes:

```text
runtime_substrate
runtime_provenance
runtime_evidence_identity
runtime_checksum
runtime_consumer_compatibility
runtime_activation_authority
```

Any ambiguous or missing classification fails closed as diagnostic-only.

## Diagnostic protocol requirements

The sealed D1 declaration freezes:

```text
diagnostic_id
accepted_gctm_theory_identity
accepted_score_contract_identity
input_substrate_class = synthetic | sealed_non_runtime
input_identity
input_checksum
input_schema
observation_family
parameterization_family
candidate_universe
event_key
score_orientation
score_transform
normalization
tie_rule
invariants
counterexample_search_space
identifiability_questions
terminal_order
```

At minimum, the diagnostic procedure must:

1. preserve event and candidate-universe identity across compared scores;
2. distinguish score calibration from within-event ordering;
3. test monotonicity under each declared transformation;
4. reject scale-dependent comparisons without a frozen normalization;
5. reject a model whose aggregate gain hides a declared protected-stratum
   ordering loss;
6. report constructive counterexamples rather than averaging them away;
7. state which parameters or latent quantities remain non-identifiable;
8. emit the exact future consumer fields and semantics that a runtime producer
   would need to supply.

## Future H0 compatibility gate

A future proposal to consume D1 output in either runtime-grounded B1 remains
inadmissible until one compatibility verdict binds all of:

1. canonical H0 evidence manifest;
2. stable H0 evidence identity;
3. canonical checksum verification;
4. producer/consumer schema compatibility;
5. observation-semantics compatibility;
6. parameterization compatibility;
7. score transformation and normalization compatibility;
8. ordering-preservation verdict.

The verdict must be separately owner-accepted and must bind exact producer,
consumer, schema, observation, parameterization, and transform identities.
Missing, partial, stale, mismatched, or rejected verdicts select
`reject_runtime_consumption`. Until acceptance, D1 and both runtime B1 slots
remain isolated.

Because `H0_ROUTE5_B1` and `GCTM_B1` are distinct consumers that do not share
activation authority, the machine record carries two independent gates:

```text
gctm_d1_to_h0_route5_b1_compatibility_v1
gctm_d1_to_gctm_b1_compatibility_v1
```

A future verdict artifact may be referenced by both gates only when it
explicitly binds both exact consumer identities. Acceptance on one gate never
implies acceptance on the other. Both gates remain `missing`.

## Ordered terminal family

A sealed D1 declaration must select exactly one bounded terminal:

### `GCTM_D1_DIAGNOSTIC_SEAL`

The declared diagnostic invariants and falsification obligations pass on the
declared diagnostic substrate. Maximum claim: the declared
substrate-agnostic diagnostic object is internally sealed.

### `GCTM_D1_BOUNDED_NO_GO`

One or more declared invariants, guards, counterexamples, or identifiability
requirements rule out the tested family within the declared diagnostic scope.
Maximum claim: a bounded diagnostic no-go.

### `GCTM_D1_INTERFACE_READY`

The diagnostic family has a machine-checkable consumer interface and complete
compatibility requirements suitable for a future, separate H0 verdict.
Maximum claim: interface-ready, not runtime-compatible.

These terminals may transition only `GCTM_D1`. They cannot alter
`H0_ROUTE5_B1`, `GCTM_B1`, or `GCTM_O1`; cannot create a decision-relevant
candidate; cannot acquire WIP; and cannot authorize H0 re-entry.

Mechanical selection order (frozen by declaration acceptance):

1. `GCTM_D1_BOUNDED_NO_GO`
2. `GCTM_D1_DIAGNOSTIC_SEAL`
3. `GCTM_D1_INTERFACE_READY`

## Activation and exit

This charter may become active only after:

1. owner acceptance of a sealed D1 declaration and its bounded terminal
   procedure — **satisfied**
   (`gctm_d1_declaration_owner_acceptance_20260723`); and
2. a separate owner scheduling decision that assigns WIP — **not satisfied**.

Declaration acceptance does not activate or close this charter. If a future
active D1 selects one of the three terminals, it exits and releases WIP. No
exit condition performs a cross-slot state transition.

## Prohibited actions

This charter never authorizes:

- H0 re-entry or exactly-once authorization;
- H0 evidence repair or reconstruction;
- B1 or O1 activation;
- runtime observation/parameterization freeze;
- a sealed runtime B1 declaration;
- registration of diagnostic evidence as runtime evidence;
- changes to the accepted score-ranking contract;
- supersession of H0 route-5 B1;
- reactivation of a parked research thread.

## Current verdict

`GCTM_D1` is a proposed, isolated diagnostic charter whose declaration is
owner-accepted as a frozen execution contract. Execution remains unscheduled;
canonical state remains `none`; runtime-consumption gates remain fail-closed
`missing`; candidate set and semantic WIP lock remain empty.

### Declaration acceptance status (2026-07-23)

```text
acceptance_terminal  = GCTM_D1_DECLARATION_ACCEPTED
owner_acceptance_id  = gctm_d1_declaration_owner_acceptance_20260723
declaration frozen   = yes
execution unscheduled = yes
state remains none   = yes
next gate            = owner_scheduling
```

### Seal-candidate package status (2026-07-23)

A **pre-activation synthetic seal-candidate** package has been generated
(`status: SEAL_CANDIDATE_GENERATED`). Provisional mechanical terminal string:
**`GCTM_D1_INTERFACE_READY`** (not a canonical registry state transition;
declaration acceptance does not promote it).

- declaration (owner-accepted):
  [`gctm_d1_ranking_diagnostic_declaration_20260723.md`](../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md)
- terminal (seal-candidate report):
  [`gctm_d1_ranking_diagnostic_terminal_20260723.md`](../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md)
- packet (immutable PR #265 identities):
  [`evidence/gctm_d1_substrate_agnostic_ranking_20260723/`](../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/)

This is **not** owner-accepted charter execution. It does **not** acquire WIP,
unlock B1/O1, satisfy any runtime compatibility gate, or move the canonical
registry `state` off `none`. Charter activation still requires separate owner
scheduling.
