<!-- doc-status: draft -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-18 -->
<!-- doc-module: semantic -->

# H0 → GCTM consumer compatibility requirements

> **Draft pre-activation compatibility contract · no H0 terminal selected · no
> GCTM/B1 execution authority.**

## Purpose and non-authority

This document defines how an accepted H0 runtime-evidence baseline is registered
and checked before a bridge-runtime GCTM/B1 consumer may rely on it. It turns
the reusable part of H0 closure into three bounded outputs:

\[
\boxed{
\text{baseline registration}
\;+\;
\text{guarantee envelope}
\;+\;
\text{consumer compatibility verdict}
}
\]

It does **not**:

```text
select an H0 terminal
change H0 capture/replay requirements
prove GCTM model adequacy or identifiability
select M1/M2 parameters or a score policy
define the L2 rank / margin / top-1 contract
activate GCTM, B1, or O1
authorize capture, labels, fitting, runtime changes, or production behavior
```

The H0 declaration owns observability and its ordered terminal. The GCTM
charter owns the transition-family specification boundary. The B1 charter owns
B1 activation and its score-ranking study. This document owns only the
**consumer-side registration and compatibility protocol** between those owners.

## Boundary rule

Let \(\Gamma_{\mathrm{H0}}\) be the accepted H0 guarantee envelope, and let
\(R_{\mathrm{GCTM/B1,obs}}\) be the runtime-observable inputs declared by one
future GCTM/B1 consumer. The compatibility question is only:

\[
R_{\mathrm{GCTM/B1,obs}}
\subseteq
\Gamma_{\mathrm{H0}}\;?
\]

It is deliberately **not** the stronger and invalid claim:

\[
R_{\mathrm{GCTM/B1}}
\subseteq
\Gamma_{\mathrm{H0}}.
\]

H0 may establish faithful runtime observability. It cannot establish a Markov
state, model-family correctness, covariance identifiability, calibrated
likelihood, event-local ranking value, online retention, or production benefit.

## Lifecycle position

```text
owner-accepted H0 terminal that supplies a runtime substrate/fidelity edge
  → H0 baseline + guarantee envelope become registrable
  → this consumer compatibility review
  → if compatible, it satisfies one B1 bridge-runtime prerequisite
  → all remaining GCTM/B1 gates still apply
```

This is **not** an H0 handoff. H0's positive terminal makes a separately
declared B1 consumer study a candidate only. The current GCTM B1-slot identity
remains an owner decision.

The compatibility review is:

| Path | Is this document a blocker? |
|:--|:--|
| H0 capture / H0 terminal | No — H0 must not wait for a downstream model consumer. |
| Substrate-agnostic GCTM mathematics | No — abstract M1/M2 feasibility is not an H0 consequence. |
| Bridge-runtime GCTM/B1 | Yes — no runtime-grounded B1 claim without an accepted compatible baseline or a separately accepted fidelity edge. |

## 0. Downstream guarantee envelope

This section fixes the **only** H0 boundary that a downstream GCTM consumer may
eventually use. It is a registration contract, not an H0 acceptance, a GCTM
model specification, or a claim that any candidate field is already covered.

For a named immutable H0 baseline \(R\), distinguish these two sets:

\[
\Gamma_{\mathrm{H0,candidate}}(R)
=
\{\text{native ABI sources that may be proposed for registration}\},
\]

\[
\Gamma_{\mathrm{H0,usable}}(R)
=
\{\text{candidate sources bound to an accepted guarantee record}\}.
\]

The protocol itself names no baseline \(R\), so it contributes no usable
guarantee: \(\Gamma_{\mathrm{H0,usable}}(R)=\varnothing\) until the H0 owner
accepts that baseline and records the field-level envelope. A capture-ABI
field, an implementation reading, or an offline export is therefore only a
**candidate source**, never an implicit guarantee.

### 0.1 Three consumer states

| Consumer state | Meaning | Legal GCTM treatment |
|:--|:--|:--|
| `registered-guarantee` | An accepted H0 baseline binds the exact field or semantic relation, its domain, replay/non-perturbation basis, and invalidation inputs. | The declared consumer may use it only within that bound domain. |
| `candidate-source` | The H0 capture ABI or native operator names a possible source, but no accepted field-level guarantee has been bound. | Record it as `available_unassured`; it cannot satisfy a bridge-runtime claim. |
| `outside-envelope` | H0 does not define, guarantee, or supply the required object. | Declare it latent, remove it, obtain a separately accepted fidelity edge, or block the bridge-runtime claim. |

`candidate-source` deliberately maps to neither `exact` nor `derived` in a
sealed consumer declaration. Its eventual relation remains undecided until
registration; a `proxy` is not a fallback for absent native semantics.

### 0.2 Candidate sources that may enter an accepted envelope

The following inventory is exhaustive for the H0-to-GCTM consumer boundary.
It is not a promise that every future H0 baseline covers every row.

| GCTM consumer object | Candidate H0 native source | Maximum relation after registration | Boundary / non-inference |
|:--|:--|:--|:--|
| Event and runtime-instance identity | `seq`, `frame`, schema version, `cand_instance_uid`, `lost_instance_uid`, and pair/candidate/claim/commit keys | `exact` identity only | Visible `track_id` is an observation, not a cross-record or trajectory identity. |
| Native exit / entry snapshot | Pair-record anchors, velocities, ring lengths, EMA heights, `h_ref`, directional inputs, and other declared raw state fields | `exact` raw-field observation or explicitly declared `derived` observation | This does not establish a sufficient Markov state, latent state, trajectory continuity, or a global identity. |
| Operational horizon and observation point | `la`, `bridge_at`, `seq`/`frame`, and the declared pair-record emission point | `exact` runtime timing only | `la` / \(\Delta_{\rm on}\) is not automatically physical endpoint gap \(g_{\rm phys}\), continuous time, or an observation-update model. |
| Candidate competition and pair-score context | Structural/pre-score/final-eligible counts and masks; pair `bdist`; native best/second values; margin and proposal fields | `exact` event/operator observation only | Candidate-local likelihood does not determine the full claim/commit outcome without the declared universe and composition. |
| Claim and commit audit boundary | Claim-record detection-score and claim-winner fields; commit identity/status fields | `exact` audit observation only | These are downstream decision observations, not a relink probability, feedback label, or authority to alter claim/commit behavior. |

For every registered row, the consumer must name whether it observes the
runtime value online, lagged, or offline-only. No field may be promoted from a
different capture point, schema, preset, or runtime identity without a new
registration.

### 0.3 Objects outside the default H0 guarantee envelope

The following are not omissions to be silently repaired by a consumer; they
are explicit GCTM boundary lines. A row can move only through a separately
declared and accepted H0 delta/fidelity edge, never through a consumer proxy.

| Required object or claim | H0 disposition | Required GCTM/B1 disposition |
|:--|:--|:--|
| Physical gap \(g_{\rm phys}\), continuous-time conversion, and production-CV null-offset treatment | `outside-envelope` | GCTM defines and validates the mapping to the registered operational horizon; no temporary proxy. |
| Delayed / continue / abstain transition substrate | `outside-envelope` | Treat as latent, remove it from the declared model, or seek a separately accepted fidelity edge. An ordered rejection reason is not this substrate. |
| Latent/Markov-state correctness, M1/M2 well-posedness, PSD, nesting, or parameter identifiability | `outside-envelope` | GCTM model-specification and proof obligations. |
| Feedback, GT/FP labels, or reveal protocol | `outside-envelope` | B1 fitting/blind/reveal protocol owns it. |
| Calibration, likelihood interpretation, or candidate-local ranking value | `outside-envelope` | GCTM may define the quantity; B1 evaluates the empirical claim. |
| Online causal retention, system efficacy, or production suitability | `outside-envelope` | O1 and production evaluation own these claims. |

### 0.4 Making a candidate source usable

The H0 owner may make one candidate-source row usable only by recording an
immutable baseline and a field-level guarantee record containing:

```text
baseline_id
guarantee_id
guarantee_class
covered field or semantic relation
schema and runtime / instrumentation identity
policy base, resolved preset, and event-key version
declared domain and causal availability
replay or shadow-nonperturbation basis
invalidation inputs
```

The GCTM/B1 consumer then binds those identities, selects `exact`, `derived`,
`proxy`, or `unavailable` for each required object, and records any constraints.
If a required object remains outside the envelope, the only valid compatibility
disposition is `unavailable` (or `extension` for a separately accepted fidelity
edge); it must never be represented as a covered H0 guarantee.

### 0.5 Static registration contract — identity example

The registration process has one machine-checked, non-evidence example for
event/runtime-instance identity. Identity-v1 accepts exactly one `identity`
guarantee for the sealed pair key, with `pair_record`, `exact`, and `online`
all fixed; it rejects every other guarantee class or consumer object. The
[schema](../../../../scripts/tools/h0_gctm_guarantee_registration_schema_v1.json)
and [validator](../../../../scripts/tools/verify_h0_gctm_guarantee_registration.py)
enforce the distinction above: a `candidate-source` fixture is valid as a
pre-registration object but must report `structurally_usable: false`; only an
`actual`, owner-accepted `registered-guarantee` record with complete binding,
basis, domain, causal availability, and the sealed identity-v1 invalidation set
can report `structurally_usable: true`.

That sealed set is exactly:

```text
runtime_instrumentation_identity
policy_base_id
resolved_preset_id
h0_schema_version
capture_mode
event_key_version
observation_state_semantics_version
identity_lifecycle
consumer_domain
causal_availability
```

`structurally_usable` is deliberately not an owner-attestation result. The
validator always reports `authority_verified: false` because it does not yet
bind to an external owner-controlled acceptance registry or packet. No caller
may treat this static result as an accepted H0 guarantee.

The checked-in [identity fixture](../../../../tests/contract/fixtures/h0_gctm_guarantee_registration_candidate_identity_v1.json)
therefore proves the fail-closed pre-registration path only. It is not an H0
baseline, guarantee record, or compatibility verdict. The companion contract
test exercises the positive `structurally_usable: true` branch only in memory,
preventing a synthetic fixture from becoming a false H0 guarantee.

### 0.6 Static registration contract v2 — full candidate-row inventory

Registration v2 extends the machine-checked contract from the identity row to
the complete section 0.2 inventory, without changing identity-v1 (that schema
and its validator path stay frozen). The
[v2 schema](../../../../scripts/tools/h0_gctm_guarantee_registration_schema_v2.json)
and the shared
[validator](../../../../scripts/tools/verify_h0_gctm_guarantee_registration.py)
accept one record carrying up to seven guarantees, one per
`(guarantee_class, stream)` coordinate:

| `guarantee_class` | Consumer object | Streams | Relation |
|:--|:--|:--|:--|
| `identity` | `event_runtime_instance_identity` | `pair_record` | `exact`, sealed pair instance key only (unchanged from v1) |
| `snapshot` | `native_exit_entry_snapshot` | `pair_record` | `exact` or `derived` |
| `timing` | `operational_horizon_observation_point` | `pair_record` | `exact` |
| `competition` | `candidate_competition_pair_score_context` | `pair_record`, `candidate_record` | `exact` |
| `audit` | `claim_commit_audit_boundary` | `claim_record`, `commit_record` | `exact` |

The v2 contract adds these sealed rules on top of the v1 machinery:

- **ABI anchoring.** Every non-identity `covered_fields` entry must come from
  a sealed per-class allowlist, and the validator re-checks each allowlist
  against `record_fields` in the
  [`h0_bridge_decision_trace_v2` schema](../../../../scripts/tools/h0_bridge_decision_trace_schema_v2.json),
  so registration fails closed if the capture ABI and the allowlists drift.
  The validator also verifies the ABI document's own `capture_schema_version`
  and requires `baseline.h0_schema_version` (and therefore every
  `declared_domain.schema_id`) to equal it exactly; a different capture schema
  needs its own registration contract, never a silent promotion.
- **Track-id ban scope.** `*track_id*` fields are registrable only inside the
  `audit` class, where section 0.2 names claim-winner and commit identity
  fields as audit observations; they remain banned as identities everywhere.
- **Record keys.** The `competition`/`candidate_record` and both `audit`
  allowlists carry their own record keys because identity-v1 binds only the
  `pair_record` stream.
- **Causal availability.** Non-identity guarantees must declare `online`,
  `lagged`, or `offline_only` (section 0.2); identity stays `online`-only.
- **Invalidation sets.** Every class requires the sealed shared set (the
  identity-v1 set minus `identity_lifecycle`); `identity` re-adds
  `identity_lifecycle`, and any `derived` relation additionally requires
  `derivation_definition`.
- **Derivation binding.** A `derived` relation must also carry an immutable
  `derivation` object (`definition_id`, `definition_version`, 64-hex
  `content_hash`) naming exactly which derivation produced the covered
  observation; `exact` guarantees must not carry one. The invalidation input
  above then has a concrete referent: the guarantee dies when that bound
  definition changes.

The checked-in [v2 candidate fixture](../../../../tests/contract/fixtures/h0_gctm_guarantee_registration_candidate_sources_v2.json)
enumerates all seven candidate-source coordinates and, like the identity
fixture, is valid only as a pre-registration object
(`structurally_usable: false`). Everything in section 0.5 about
`authority_verified: false`, owner acceptance, and in-memory-only positive
branches applies unchanged to v2.

## 1. Baseline registration

After an H0 terminal supplying a runtime substrate/fidelity edge is
owner-accepted, its H0 baseline registration must bind at least:

```text
h0_terminal
h0_evidence_id
h0_packet_hash
h0_schema_version
runtime / instrumentation identity
policy-base and resolved-preset identities
capture mode
event-key version
observation and state-semantics version
dataset / sequence domain
```

The registration names one immutable baseline, conceptually
\(\mathrm{H0}@R\), rather than the mutable phrase “the H0 capture.” A future
consumer binds this identity in its declaration; it may not silently follow a
new packet, schema, preset, or runtime head.

## 2. Guarantee envelope

The accepted H0 packet must expose, by stable field/semantic reference, which
of the following guarantees are available in its declared domain:

| Guarantee class | What a consumer may rely on when the accepted H0 packet names it |
|:--|:--|
| `identity` | Event, lost, candidate, claim, and commit identities are keyed under the declared H0 schema. |
| `timing` | Observation point and capture-time convention are fixed and replayed. |
| `shadow_nonperturbation` | Capture does not alter policy-visible runtime behavior under H0's declared comparison. |
| `runtime_state` | The named raw state field is emitted from the declared native decision path. |
| `replay` | The named pair/candidate/claim/commit quantity satisfies H0's declared replay relation. |
| `domain` | The guarantee applies only to the declared preset, path, schema, runtime identity, and capture/evidence domain. |

Each reusable guarantee must have a stable registration record; a consumer must
not cite “H0 generally.” The future acceptance packet must provide at least:

```text
guarantee_id
baseline_id
guarantee_class
covered fields / semantic relation
declared domain
replay or non-perturbation basis
invalidation inputs
```

The H0 acceptance packet, not this table, decides which fields are actually
covered and whether any relation is exact. A consumer must label every use as
`exact`, `derived`, `proxy`, or `unavailable`; `hypothesis` is reserved for the
physical-to-runtime model mapping and cannot be promoted by H0 alone.

### Not guaranteed by H0

At minimum, the compatibility record must state that H0 does not guarantee:

```text
GT / FP labels or their reveal protocol
the correctness or sufficiency of a latent/Markov state
physical gap semantics, continuous-time conversion, or CV null-offset treatment
M1/M2 well-posedness, PSD, nesting, or parameter identifiability
calibration, likelihood interpretation, or candidate-local ranking gain
online causal retention, MOT gain, or production suitability
```

GT/FP labels belong to B1's fitting/blind/reveal protocol. Likelihood and
uncertainty are GCTM outputs derived from registered inputs; they are not H0
capture fields.

### Invalidation triggers

The consumer registration must be re-reviewed if any bound semantic input
changes, including:

```text
runtime/policy base, preset, event key, schema, or capture mode
proposal/candidate/claim/commit timing or identity lifecycle
anchor, velocity, EMA, history-window, or reconstruction semantics
the GCTM observation/time interface or required-field relation type
the declared consumer domain or causal-availability rule
```

Pure formatting, generated rendering, or bit-identical non-semantic changes do
not by themselves invalidate a registration. H0's own provenance and admitted
projection rules remain authoritative for deciding whether an H0 execution
baseline is valid.

## 3. Existing online operator correspondence

The runtime object to which a bridge-runtime consumer must correspond is not a
single-track motion transition or an intrinsic stochastic kernel. At one bridge
fire event, it is the deterministic hybrid operator

\[
F_{\Delta\mid\mathcal C}
:
\mathcal X_{\mathrm{event}}
\longrightarrow
\mathcal Z_c\times\mathcal Z_d,
\]

with the production composition:

\[
R
\rightarrow
M_{\mathrm{pre}}
\rightarrow
G_\Delta
\rightarrow
M_{\mathrm{post}}
\rightarrow
S_{\mathrm{rank}}
\rightarrow
D_{\mathrm{margin/claim/commit}}.
\]

Here \(\Delta=\operatorname{age}[\mathrm{lost}]\) is the operational
lost-age horizon used by the score. Commit transfers identity to the candidate;
it does not merge the lost motion state into the candidate state. Fixed native
inputs therefore have deterministic continuous and discrete outputs. Any GCTM
stochasticity must name a separate source—latent physical transition,
observation, native-state, context, population, or model-residual uncertainty.

### Event and competition correspondence

A pair-local state alone cannot determine an online decision. The compatibility
map must distinguish:

| Runtime object | Required interpretation for GCTM/B1 |
|:--|:--|
| `R` native reduction | Lost exit / candidate entry anchors, velocity, height, history-window, and short-history behavior must retain native semantics or be labelled `derived`/`proxy`. |
| \(\mathcal I_{\mathrm{struct}}\) | Structural lost competitors scanned for one candidate; part of the event input. |
| \(\mathcal I_{\mathrm{pre}}\) | Result of pre-score structural gates; not interchangeable with the structural set. |
| \(\mathcal I_{\mathrm{rank}}\) | Final eligible set after score and post-score masks; it is an operator result, not a context prior. |
| \(\mathcal J_L\) | Actual proposers of one lost \(L\), produced after candidate-local proposal; it is distinct from \(\mathcal I\) and is not a predeclared event input. |
| detection/track score \(q_j\) | Claim-key input for each member of the full candidate universe. Claim arbitration consumes it with that candidate's proposal; it does not re-rank `bdist`. If outside a GCTM intervention, it must be declared unchanged rather than omitted. |

In particular:

\[
\mathcal I_{\mathrm{rank}}
=
M_{\mathrm{post}}
\circ
G_\Delta
\circ
M_{\mathrm{pre}}
(\mathcal I_{\mathrm{struct}}).
\]

Treating \(\mathcal I_{\mathrm{rank}}\) as a predeclared context silently
conditions on a gate/cutoff outcome that belongs to the operator itself.

The claim boundary has the same directionality. Let
\(\mathcal U_{\mathrm{event}}\) be the full candidate universe at the bridge
event. Candidate-local processing first produces one proposal (or no proposal)
per candidate:

\[
P_j:
\left(
x_{C_j},\{x_L^{(i)}\}_{i\in\mathcal I_{\mathrm{struct},j}}
\right)
\longrightarrow
\operatorname{proposal}_j.
\]

For one lost \(L\), the actual claim set is therefore the intermediate result

\[
\mathcal J_L
=
\{j\in\mathcal U_{\mathrm{event}}:
\operatorname{proposal}_j=L\},
\]

not an input condition. Claim arbitration is a separate operator

\[
A_{\mathrm{claim}}:
\{(\operatorname{proposal}_j,q_j,j)\}_{j\in\mathcal U_{\mathrm{event}}}
\longrightarrow
\text{claim/commit},
\qquad
F_{\mathrm{event}}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j.
\]

Thus a singular pair-local \(x_C\), or \(\{q_j\}_{j\in\mathcal J_L}\),
cannot by itself determine cross-candidate claim or commit. A consumer either
declares the full \(\mathcal U_{\mathrm{event}}\) and this composition, or
declares claim/commit explicitly unchanged or audit-only.

### Continuous and discrete output correspondence

The consumer must say which layer it models and which layers it leaves under
the unchanged online contract:

| Output layer | Examples | Required declaration |
|:--|:--|:--|
| Continuous \(\mathcal Z_c\) | pair `bdist`, best/second values, score margin | Whether GCTM calibrates, augments, replaces, or only diagnoses the named score quantity. |
| Discrete ranking | eligibility, rank winner, margin pass, proposal | How score/mask semantics compose to this result; pair likelihood alone is insufficient. |
| Claim / commit | cross-candidate claim winner, identity commit | Explicitly unchanged, separately modelled, or audited only. Claim order is not geometry-score order. |

No consumer may describe an event-local pair likelihood as a final relink or
commit probability without declaring the competitor, mask, margin, and claim
composition. Likewise, a population distribution over the operator is a new
population object, not an intrinsic probability law of the production bridge.

## 4. GCTM/B1 consumer requirement map

Before a bridge-runtime B1 declaration is sealed, its consumer must enumerate
the required runtime-observable objects. The canonical row format is:

| Required object | H0 field / semantic source | Relation | Causal availability | Status / disposition |
|:--|:--|:--|:--|:--|
| `<GCTM/B1 object>` | `<H0@R field or event relation>` | `exact \| derived \| proxy \| unavailable` | `<online / lagged / offline-only>` | `covered \| constrained \| extension \| unavailable` |

The consumer registration binds one baseline and records:

```text
baseline_id
required_guarantee_ids
required runtime-observable objects and their relation types
requested extension or latent/unavailable disposition
maximum claim layer
consumer-specific invalidation inputs
```

The map must include, where the declared model uses them:

```text
lost/candidate identities and event membership
exit state, entry observation, and history-window provenance
structural/pre-score/final-eligible competitor sets and mask provenance
full candidate-universe and detection-score-key semantics; actual
  claim-proposer set \(\mathcal J_L\) as a proposal-composition output, or an
  explicit unchanged-claim declaration
online horizon Δ_on and physical endpoint gap g_phys
bridge_at convention and g_phys ↔ Δ_on mapping
frame-time unit and any continuous-dt conversion
production-CV null offset and its declared treatment
context and missing-value semantics
continuous-score versus discrete-decision output correspondence
declared source of any GCTM stochasticity
the exact causal information set intended for a future O1 transport
```

The horizon, `bridge_at`, continuous-dt, and null-offset rows are especially
important: the production score uses its operational lost-age horizon, which is
not automatically the physical inter-endpoint gap. H0 may cover the runtime
timing fields; GCTM owns the physical-time mapping and the treatment of any
induced null offset.

### Required dispositions

| Map status | Required disposition |
|:--|:--|
| `covered` | Register and bind the named H0 guarantee/field identity. |
| `constrained` | Narrow the GCTM model, domain, causal claim, or B1 conclusion explicitly. |
| `extension` | A separately declared and accepted H0 delta/fidelity edge is required; do not retrofit H0 inside B1. |
| `unavailable` | Model the quantity as latent, remove it, or block the bridge-runtime claim. |

### Required composition declaration

The future B1 declaration must choose one named relationship to a **named
subcomponent** of the existing online operator:

```text
augment named pair score
calibrate named score quantity
replace pair score under a frozen L2 contract
replace ranking under a frozen L2 contract
shadow-only diagnostic
```

It must not re-describe or silently replace the production baseline operator
\(F_{\Delta\mid\mathcal C}\). A `replace` option means only an explicitly
named pair-score or ranking-subcomponent intervention under a frozen L2
contract; it must preserve or separately re-charter every other operator
stage. It must not silently turn a score intervention into an eligibility
change, claim-arbitration change, or commit redesign. The registry-owned L2
contract remains the sole authority for rank, margin, and top-1 semantics.

## 5. Compatibility verdict

The consumer review records one of the following **pre-activation dispositions**.
They are not H0 terminals and do not alter the H0 terminal partition.

| Verdict | Meaning | B1 consequence |
|:--|:--|:--|
| `GCTM_H0_COMPATIBLE` | Every declared runtime-observable requirement is covered with an admissible relation and causal-availability rule. | This blocker clears; all other B1 gates remain. |
| `GCTM_H0_COMPATIBLE_WITH_CONSTRAINTS` | The declared model/claim is valid only under explicitly bound domain, state, timing, or availability constraints. | This blocker clears only for the constrained declaration. |
| `GCTM_H0_GAP_REMAINS` | A required object has no adequate H0 guarantee or accepted fidelity edge. | Bridge-runtime B1 remains blocked; choose a latent/substrate-agnostic re-charter or a separately declared H0 delta. |

### Verdict fact ownership and lifecycle

A compatibility verdict is a **B1 preactivation record**, not an H0 terminal,
GCTM model seal, or free-floating semantic state. Its fact owner and accepting
authority are the owner of the future B1 declaration that consumes it (the
**B1 declaration owner**). The authoritative record is the declaration's
immutable `h0_gctm_compatibility` record; this protocol document defines its
schema but owns no particular verdict.

That record must include at least:

```text
compatibility_verdict_id
consumer_declaration_id
baseline_id and required_guarantee_ids
gctm_specification_id
required-object map and relation types
selected verdict, constraints, and maximum claim layer
accepted_by / accepted_at
supersedes (if any)
invalidation inputs and current disposition
```

The H0 owner remains the sole authority for the cited H0 terminal,
baseline, and guarantee envelope; it does not decide consumer sufficiency. A
change or invalidation of any cited H0 guarantee, GCTM specification, required
object mapping, or consumer causal/domain rule makes the B1 record stale. Only
the B1 declaration owner may accept a successor record, which must name the
record it supersedes. A later sealed B1 declaration binds that accepted record;
it references the verdict and does not re-adjudicate it.

A B1-declaration-owner-accepted compatibility record must be bound in the
sealed B1 declaration together with its H0 and GCTM identities. A positive
compatibility verdict does not select the B1 task, resolve the B1 slot,
satisfy the L2 score-layer contract, or grant online authority.

## 6. Registration and handoff procedure

```text
1. H0 owner accepts an H0 terminal that supplies a runtime substrate/fidelity
   edge and records the H0 baseline identity.
2. GCTM/B1 consumer declares its required runtime-observable objects.
3. Consumer maps each object to H0 using the required relation type.
4. The B1 declaration owner accepts one identified compatibility record and
   records its constraints, gaps, invalidation inputs, and any predecessor.
5. A later sealed B1 declaration binds that accepted record, if all independent
   B1 activation gates also pass.
```

The direct downstream target is therefore not “GCTM automatically starts.” It
is an owner-resolved future bridge-runtime B1 consumer with a frozen
registration. H0's closure remains valid even if no such consumer is scheduled.

## Read first

- [H0 declaration — observability and ordered terminal](headline_bridge_full_decision_capture_declaration_20260713.md)
- [GCTM parked charter — transition-family boundary](../../../research/threads/gap_conditioned_stochastic_transition_model_task.md)
- [GCTM B1 charter — activation and frozen inputs](../../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- [B1/O1 shared semantics — evidence chain and target layer](gctm_b1_o1_task_objectives_and_semantics_20260716.md)
- [Claim-state registry §7 — L2 score-layer gap](../../../research/contracts/claim_state_registry.md#7-架構缺口顯式化而不是假裝可編排)
- [Bridge decision semantics](../../../research/tracker-decision/relink_bridge.md)
- [Online mathematical model](../../../reference/math_model.md)
- [Existing online object analysis for GCTM alignment](existing_online_object_analysis_for_gctm_alignment_20260718.md)

## Acceptance

This draft is complete as a **boundary/registration protocol** when owner review
confirms that it does not change H0's terminal, GCTM model authority, B1 slot
identity, L2 score semantics, or execution authorization. The proposed B1
charter binds this protocol as a bridge-runtime prerequisite; a particular
future sealed B1 declaration must additionally bind one
B1-declaration-owner-accepted
compatibility verdict.
