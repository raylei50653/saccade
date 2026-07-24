---
doc-status: closed
doc-promotion: navigation-only; evidence lives in the linked packet
doc-date: 2026-07-24
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
target-decision-layer: runtime-consumer-universe-contract
primary-intent: freeze-runtime-candidate-universe
output-class: sealable | unsealable | invalid
created: 2026-07-24
closed: 2026-07-24
closed-verdict: GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE
---

# GCTM runtime-native candidate universe and event-composition contract

> **CLOSED:** The ordered procedure selected
> **`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`**. A score-independent,
> event-local, candidate-conserving runtime-native universe is structurally
> defined as consumer identity `gctm_runtime_native_candidate_universe_v1`.
> This does **not** establish an H0 completeness guarantee, runtime substrate,
> compatibility verdict, or B1 activation. WIP was released at terminal.

## Final status

| Field | Value |
|:--|:--|
| Terminal | **`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`** |
| Frozen runtime universe | `gctm_runtime_native_candidate_universe_v1` |
| Event key | `gctm_runtime_event_key_v1` = `(seq, frame, lost_slot, lost_instance_uid, event_key_version)` |
| Candidate key | `gctm_runtime_candidate_key_v1` = `(event_key, cand_slot, cand_instance_uid)` |
| Inclusion stage | `pre_score_eligible_v1` / `pre_score_eligible_set` |
| Score-independent | **yes** |
| Completeness semantics | defined; **not** an H0 guarantee |
| Current trace-v2 sufficient to define consumer universe | **yes** |
| Runtime guarantee established | **no** |
| Mechanical packet | [gctm_runtime_native_candidate_universe_20260724](../../../modules/semantic/research/evidence/gctm_runtime_native_candidate_universe_20260724/) |
| Next owner decision | minimal H0 registration-v3 delta (or reconsider GCTM_B1 runtime hook if producer cannot meet requirements) |
| Direct handoff as H0 implementation authority | **no** |

Owner acceptance is represented by merge of this exact packet under
`gctm_runtime_universe_terminal_owner_acceptance_20260724`. Merge does not
authorize H0 capture, re-entry, guarantee registration, compatibility verdict,
or B1/O1 activation.

## Decision question

Without modifying H0 registration-v2, executing capture, or issuing a
compatibility verdict, determine whether trace-v2 admits one:

```text
score-independent
complete (as consumer semantics)
non-circular
event-local
candidate-conserving
```

native candidate-event boundary on which M0, M1, and M2 can be compared over
the identical event set and candidate set without depending on winner, claim,
commit, or reveal information.

## Prerequisite binding

This charter binds and retains:

```text
H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT
h0_gctm_static_audit_terminal_owner_acceptance_20260723
```

Retained conclusions:

```text
current ABI / registration-v2 is structurally insufficient
repeat capture under unchanged interface is forbidden
```

It closes only the **consumer-owned** gap named by that terminal:

```text
candidate_universe
event membership / event composition semantics
```

It does **not** close producer completeness registration or runtime
compatibility.

## Authority boundary

Allowed:

```text
define one runtime-native candidate-event space
define candidate-universe identity
define event composition and completeness requirements
define a later registration-v3 requirement surface
mechanically determine whether the runtime universe is sealable
```

Forbidden:

```text
reopen or alter GCTM_D1
replace synthetic_event_candidate_set_v1 inside the closed D1 packet
modify H0 trace-v2 ABI
modify registration-v2
register an H0 guarantee
execute H0 capture
authorize H0 re-entry
issue a runtime compatibility verdict
activate H0_ROUTE5_B1 / GCTM_B1 / O1
change production/runtime behavior
```

## Ownership

New runtime-consumer identity:

```text
gctm_runtime_native_candidate_universe_v1
```

It is the future `GCTM_B1` declaration-target for a runtime candidate universe.
It is not the D1 synthetic universe, not an H0 guarantee, not a runtime
evidence substrate, and not a B1 activation declaration.
`GCTM_D1_INTERFACE_READY` remains closed and read-only.

## Score-policy spaces

Per `score_ranking_evidence_contract_v1`:

| Member | Frozen identity |
|:--|:--|
| \(U_{\mathrm{src}}\) | `h0_bridge_pair_record_space_v2` (`pair_records`) |
| \(U_{\mathrm{evt}}\) | `h0_bridge_lost_event_space_v1` |
| \(\rho\) | `pair_to_lost_event_v1` (total, functional) |
| \(C_e\) | `pre_score_eligible_candidates_for_event_v1` |

Any member change is a different score policy.

## Inclusion-stage decision

Exactly one native stage is selected:

```text
pre_score_eligible_set
```

Proved from the frozen writer order in `tracker_gpu.cu`:

```text
structural admission
→ height / speed / spatial (non-score)
→ pre_score_passes++
→ bdist score
→ cutoff / occupancy / appearance / portable_tail
→ best/second / margin / proposal
→ claim / commit
```

Final-eligible and claim/commit-derived sets are rejected as circular for pure
score-ranking comparison.

## Registration-v3 requirement surface

Requirements-only artifact:

[`h0_native_universe_completeness_registration_requirements_v1.json`](../../../modules/semantic/research/evidence/gctm_runtime_native_candidate_universe_20260724/h0_native_universe_completeness_registration_requirements_v1.json)

It does not modify registration-v2 and does not claim that registration-v3
exists.

## Machine checks

- Schema: [`gctm_runtime_universe_schema_v1.json`](../../../../scripts/tools/gctm_runtime_universe_schema_v1.json)
- Validator: [`validate_gctm_runtime_universe.py`](../../../../scripts/tools/validate_gctm_runtime_universe.py)

Fixed outputs on every path:

```text
authority_verified: false
runtime_guarantee_established: false
runtime_compatibility_established: false
h0_reentry_authorized: false
b1_activation_eligible: false
```

## Ordered terminal

1. `GCTM_RUNTIME_UNIVERSE_AUDIT_INVALID`
2. `GCTM_RUNTIME_UNIVERSE_UNSEALABLE`
3. `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`

Selected: **`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`**.

## History

- 2026-07-24 — charter acquired sole semantic WIP as the consumer re-charter
  branch of `H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT`.
- 2026-07-24 — frozen inputs, identities, inclusion decision, composition/
  completeness contract, registration requirements, schema/validator,
  positive/negative fixtures, and mechanical terminal completed without H0
  capture or registration mutation.
- 2026-07-24 — terminal selected; WIP released; no direct H0 implementation
  handoff.
