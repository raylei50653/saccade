---
doc-status: active
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-23
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: sole-active
activation-gate: "declaration owner acceptance + owner scheduling — both satisfied 2026-07-23; see Activation record"
target-decision-layer: diagnostic-only
primary-intent: model-and-interface-diagnostic
output-class: "diagnostic seal | bounded no-go | interface-ready"
mainline-transition: "GCTM_D1 only; never H0_ROUTE5_B1, GCTM_B1, or GCTM_O1"
created: 2026-07-23
---

# GCTM D1 — substrate-agnostic ranking diagnostic — task charter

## Status and authority

**ACTIVE / sole-active / one canonical execution authorized / execution not
completed.** `GCTM_D1` is an independent, substrate-agnostic diagnostic slot.
Its sealed declaration is owner-accepted as a frozen execution contract
(`gctm_d1_declaration_owner_acceptance_20260723`). Owner scheduling
(`gctm_d1_owner_scheduling_20260723` /
`gctm_d1_activation_owner_acceptance_20260723`) activates the slot, acquires
semantic WIP, and authorizes **exactly one** canonical synthetic diagnostic
execution of those frozen identities. The charter is **not closed**. Canonical
registry `state` remains **`none`** until a later owner-accepted execution
closure. This activation does not authorize fixture/runner/invariant/interface
replacement, H0 re-entry, B1/O1 activation, or a production-facing claim.

```text
declaration accepted     = yes (execution contract frozen)
owner scheduling         = yes (one canonical execution authorized)
WIP acquired             = yes (diagnostic sole-active)
execution completed      = no
canonical state          = none
blocked_by               = []
```

### Activation record

The activation gate required both:

1. `declaration_owner_acceptance` — **satisfied**
   (`gctm_d1_declaration_owner_acceptance_20260723` /
   evidence_class `owner_accepted_governance`; PR #266); and
2. `owner_scheduling` — **satisfied**
   (`gctm_d1_owner_scheduling_20260723` /
   evidence_class `owner_scheduling_decision`;
   slot `owner_acceptance_id` =
   `gctm_d1_activation_owner_acceptance_20260723`).

Canonical scheduling record:

[`gctm_d1_owner_scheduling_20260723.json`](../../modules/semantic/research/gctm_d1_owner_scheduling_20260723.json)
(`artifact_sha256: 17c534c5dc25ad0318d26b0208a7f9dceedc603ec1f81dc29fe27b49bc2607c8`).

Owner decision text:

```text
schedule GCTM_D1 for one canonical execution
```

Scope:

```text
one canonical execution of the exact declaration and packet identities
accepted by gctm_d1_declaration_owner_acceptance_20260723
```

**Activation ≠ execution completed.** This transition authorizes WIP and one
frozen-runner execution. It does not run the diagnostic, select a terminal,
promote the provisional seal-candidate string, or close the charter. Those
remain a later independent PR.

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

### Frozen pre-activation seal-candidate package

Before activation, the repository generated a **pre-activation synthetic
seal-candidate package** whose identities are frozen by declaration acceptance:

```text
generation_kind = pre_activation_synthetic_seal_candidate
substrate        = synthetic fixtures only
status           = SEAL_CANDIDATE_GENERATED
authority        = owner-reviewable proposal, not charter execution
```

Owner-accepted charter **execution** is now scheduled. The frozen package may
be used by the authorized one-shot canonical execution; it is still **not** a
canonical registry terminal transition and still does not unlock B1/O1/H0.

Activation requirements (both satisfied):

1. ~~`declaration_owner_acceptance`~~ **satisfied** —
   evidence_class `owner_accepted_governance` bound to
   `gctm_d1_declaration_owner_acceptance_20260723`;
2. ~~`owner_scheduling`~~ **satisfied** —
   evidence_class `owner_scheduling_decision` bound to
   `gctm_d1_owner_scheduling_20260723`.

Seal-candidate generation, declaration acceptance, owner scheduling, and
owner-accepted execution closure remain distinct authorities.

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
automatic WIP acquisition without owner scheduling
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

## Relation to other slots

```text
GCTM_D1  --isolated-->  H0_ROUTE5_B1
GCTM_D1  --isolated-->  GCTM_B1
GCTM_B1  --coexist-->   H0_ROUTE5_B1
```

`GCTM_D1` does not alias, supersede, or share activation authority with either
runtime-grounded slot. Runtime slots remain isolated.

Because `H0_ROUTE5_B1` and `GCTM_B1` are distinct consumers that do not share
activation authority, the machine record carries two independent gates:

```text
gctm_d1_to_h0_route5_b1_compatibility_v1
gctm_d1_to_gctm_b1_compatibility_v1
```

A future verdict artifact may be referenced by both gates only when it
explicitly binds both exact consumer identities. Acceptance on one gate never
implies acceptance on the other. Both gates remain `missing` with fail-closed
`incompatible_behavior: reject_runtime_consumption`.

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
candidate; and cannot authorize H0 re-entry. Terminal selection on an active
charter **releases** WIP; it does not create WIP.

Mechanical selection order (frozen by declaration acceptance):

1. `GCTM_D1_BOUNDED_NO_GO`
2. `GCTM_D1_DIAGNOSTIC_SEAL`
3. `GCTM_D1_INTERFACE_READY`

## Activation and exit

This charter is active because both activation requirements are satisfied:

1. `declaration_owner_acceptance` — **satisfied**
   (`gctm_d1_declaration_owner_acceptance_20260723` /
   evidence_class `owner_accepted_governance`); and
2. `owner_scheduling` — **satisfied**
   (`gctm_d1_owner_scheduling_20260723` /
   evidence_class `owner_scheduling_decision`).

Slot-level `owner_acceptance_id` is
`gctm_d1_activation_owner_acceptance_20260723` (activation acceptance only;
not the declaration acceptance id). When the authorized one canonical
execution selects one of the three terminals under a later owner acceptance,
the charter exits and releases WIP. No exit condition performs a cross-slot
state transition.

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
- reactivation of a parked research thread;
- replacement of the frozen fixture, runner, invariants, consumer interface,
  parameterization, or terminal procedure.

## Current verdict

`GCTM_D1` is an **active**, isolated diagnostic charter. Declaration is
owner-accepted; owner scheduling is accepted; one canonical synthetic
execution is authorized; semantic WIP is held. Canonical state remains
`none`; runtime-consumption gates remain fail-closed `missing`; the slot is
**not** a decision-relevant runtime candidate.

### Owner scheduling status (2026-07-23)

```text
decision             = schedule GCTM_D1 for one canonical execution
scheduling_id        = gctm_d1_owner_scheduling_20260723
owner_acceptance_id  = gctm_d1_activation_owner_acceptance_20260723
slot state           = active
WIP acquired         = yes
execution completed  = no
canonical state      = none
next step            = separate PR: frozen runner execution + mechanical
                       terminal selection + owner-accepted closure
```

### Declaration acceptance status (2026-07-23)

```text
acceptance_terminal  = GCTM_D1_DECLARATION_ACCEPTED
owner_acceptance_id  = gctm_d1_declaration_owner_acceptance_20260723
declaration frozen   = yes
```

### Seal-candidate package status (2026-07-23)

A **pre-activation synthetic seal-candidate** package has been generated
(`status: SEAL_CANDIDATE_GENERATED`). Provisional mechanical terminal string:
**`GCTM_D1_INTERFACE_READY`** (not a canonical registry state transition;
activation does not promote it).

- scheduling (owner-accepted):
  [`gctm_d1_owner_scheduling_20260723.json`](../../modules/semantic/research/gctm_d1_owner_scheduling_20260723.json)
- declaration (owner-accepted):
  [`gctm_d1_ranking_diagnostic_declaration_20260723.md`](../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md)
- terminal (seal-candidate report):
  [`gctm_d1_ranking_diagnostic_terminal_20260723.md`](../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md)
- packet (immutable PR #265 identities):
  [`evidence/gctm_d1_substrate_agnostic_ranking_20260723/`](../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/)

Owner scheduling accepts WIP and authorizes one canonical execution of those
frozen identities. It does **not** unlock B1/O1, satisfy any runtime
compatibility gate, or move the canonical registry `state` off `none`.
