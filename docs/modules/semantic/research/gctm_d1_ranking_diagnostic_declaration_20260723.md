<!-- doc-status: active -->
<!-- doc-promotion: owner-accepted frozen diagnostic execution contract; not charter execution; not runtime evidence -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: semantic -->
<!--
  Status vocabulary (fixed):
    doc-status: active
      = this declaration document is in force as a frozen execution contract.
      ≠ semantic WIP active, ≠ GCTM_D1 slot active, ≠ charter execution.
    slot.lifecycle_state / state (registry): proposed / none
      = diagnostic slot not activated; no WIP; no candidate.
    slot.owner_acceptance_id
      = slot *activation* acceptance only; remains null until activation.
      ≠ declaration_owner_acceptance (bound under activation_evidence_bindings).
-->

# GCTM D1 — substrate-agnostic ranking diagnostic declaration (v1)

## Status

**Owner-accepted / frozen execution contract** for the sealed D1 diagnostic
declaration. Canonical registry diagnostic `state` remains **`none`**.
Execution is **unscheduled**. WIP is **not** acquired.

```text
generation_kind              = pre_activation_synthetic_seal_candidate
sealed_packet_status         = SEAL_CANDIDATE_GENERATED
declaration_status           = GCTM_D1_DECLARATION_ACCEPTED
owner_acceptance_id          = gctm_d1_declaration_owner_acceptance_20260723
acceptance_date              = 2026-07-23
next_gate                    = owner_scheduling
canonical_registry_state     = none
charter_execution            = not authorized by this acceptance
doc-status:active            = document in force (not WIP / not slot active)
slot.owner_acceptance_id     = null until slot activation (distinct field)
```

This acceptance freezes the diagnostic family, runner, fixtures, consumer
interface, invariants, compatibility-requirements identity, and exhaustive
terminal procedure. It does **not** execute the charter, select a canonical
diagnostic terminal for registry transition, schedule WIP, or authorize any
runtime claim.

Machine gate (slot governance):

```text
activation_requirements =
  declaration_owner_acceptance  → evidence_class owner_accepted_governance
  owner_scheduling              → evidence_class owner_scheduling_decision
```

`declaration_owner_acceptance` is satisfied by this PR. `owner_scheduling` is
**not** satisfied by another generic governance acceptance; it requires a
distinct `owner_scheduling_decision` evidence binding.

Machine sidecar (immutable sealed packet; PR #265):
[`evidence/gctm_d1_substrate_agnostic_ranking_20260723/declaration_sidecar.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/declaration_sidecar.json)

Charter:
[`docs/research/threads/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md`](../../../research/threads/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md)

Slot identity:
[`gctm_b1_slot_identity_decision_v1`](../../../research/contracts/gctm_b1_slot_identity_decision_v1.json)

## Owner acceptance metadata (immutable)

| Field | Frozen value |
|:--|:--|
| `owner_acceptance_id` | `gctm_d1_declaration_owner_acceptance_20260723` |
| `acceptance_terminal` | `GCTM_D1_DECLARATION_ACCEPTED` |
| `acceptance_date` | `2026-07-23` |
| `accepted_declaration_hash` | `33dda22efffe3f1d8cfceec66400ef7549b1b9ba692bcffbb0eabe78dee355e0` |
| `accepted_declaration_sidecar_hash` | `6799e09a5cc39e6bc05e2970d55fe15c59081b814937bafcb6ab4fefc1e99809` |
| `accepted_packet_manifest_hash` | `9efb9005711b1f9f08c6ae06ffc03cad49e7489fda56c7129d04c844fecd9bc2` |
| `accepted_packet_id` | `gctm_d1_substrate_agnostic_ranking_20260723` |
| `accepted_runner_identity` | `scripts/tools/run_gctm_d1_diagnostic.py` |
| `accepted_core_library_identity` | `scripts/tools/gctm_d1/` |
| `accepted_terminal_procedure_id` | `gctm_d1_mechanical_three_way_terminal_v1` |
| `accepted_fixture_pack_id` | `gctm_d1_synthetic_fixture_pack_v1` |
| `accepted_fixture_sha256` | `92dafcd0acc33642b2eaddb1c6178d2d2d519b1e40ea3160069155c9150151bd` |
| `accepted_consumer_interface_hash` | `45c16a6c8cf50d098b12cc8e4f1acbdcc846d0d431ad77f9fbe64bdb60bb57ce` |
| `accepted_compatibility_matrix_hash` | `25a2ca49dbfd6d9d8985681967beb3f4df51e2fc51901444f246c94e49fa7ddd` |
| `accepted_invariant_report_hash` | `f9ca3e64f7c4937686e3efa2d2661c068803f1aa48883013212a84b6ed3f940d` |
| `accepted_terminal_report_hash` | `c31e774d3497c8729cb9f1f01c135251a3a8b1a043ec02dfa7edef1ce2c27e0c` |
| `accepted_identities_hash` | `c913f1d17f36acb54b08a209b4ecb62b540c0f55e54c994035bcdc1ebe39125e` |

`accepted_declaration_hash` is the SHA-256 of the PR #265 sealed declaration
body (pre-acceptance-metadata). All packet hashes bind the immutable PR #265
evidence packet; fresh runner emit must remain bit-identical to that packet.

## Authority boundary

```text
slot_id: GCTM_D1
authority_class: diagnostic_only
may_transition: GCTM_D1 only
must_not_alter:
  quantity.bridge_capture_provenance
  H0_ROUTE5_B1
  GCTM_B1
  GCTM_O1
  decision_relevant_candidate_set
```

This declaration freezes a **diagnostic family** and consumer interface. It does
**not** freeze a production runtime observation mode, authorize H0 re-entry,
activate either runtime B1, or unlock O1.

### Established by this acceptance

```text
owner acceptance of:
  sealed D1 declaration
  diagnostic policy identity
  synthetic input identity
  I1–I12 invariant set
  consumer-interface identity
  compatibility-requirements identity
  exhaustive terminal procedure
```

### Not established by this acceptance

```text
charter execution
GCTM_D1 terminal acceptance (canonical registry)
GCTM_D1_INTERFACE_READY as canonical diagnostic state
semantic WIP scheduling
H0 compatibility verdict
runtime substrate
H0_ROUTE5_B1 activation
GCTM_B1 activation
GCTM_O1 activation
production claim
```

## Frozen identities

| Field | Frozen value |
|:--|:--|
| `diagnostic_id` | `gctm_d1_substrate_agnostic_ranking_v1` |
| `accepted_gctm_theory_identity` | `docs/research/models/gap_conditioned_stochastic_transition_spec_v1.md` |
| `accepted_gctm_theory_sha256` | `8401c90d8fe2766eb314a0f4eb55cad86d9a1ca3bbfacde8535b8eb55bc3ff6e` |
| `accepted_gctm_lemmas_identity` | `docs/research/models/gap_conditioned_stochastic_transition_lemmas_v1.md` |
| `accepted_gctm_lemmas_sha256` | `0c880466e73e1d2b34af018113b7ec866895996fbeb03a4e62d0c5d712aba2bd` |
| `accepted_score_contract_identity` | `score_ranking_evidence_contract_v1` |
| `accepted_score_contract_sha256` | `7dbc2d965079fa3fc13f7802a4a083b1c4cbf49d658ffe3728b6c405364a13b4` |
| `input_substrate_class` | `synthetic` |
| `input_identity` | `gctm_d1_synthetic_fixture_pack_v1` |
| `input_schema` | `gctm_d1_fixture_pack_v1` |
| `observation_family` | `position_innovation_residual_v1` |
| `parameterization_family` | `gctm_affine_m0_m1_m2_shared_interface_v1` |
| `candidate_universe` | `synthetic_event_candidate_set_v1` |
| `event_key` | `event_id` |
| `score_orientation` | `lower_better` |
| `score_transform` | `identity_after_declared_score` |
| `normalization` | `frozen_identity_no_free_scale` |
| `tie_rule` | `stable_cand_id_asc` |
| `ordering_active_mechanism` | `anisotropic_shared_innovation_covariance` |
| `counterexample_search_space` | synthetic \(d=2\) controlled residuals and \(S\) structures |

Exact checksums and artifact digests are recorded in the evidence packet
[`identities.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/identities.json)
and [`manifest.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/manifest.json).

## Observation family

**`position_innovation_residual_v1`**

- Observation mode: \(H_x\) (position-only).
- Residual: \(r = y_1 - H m^-_\Delta \in \mathbb R^d\) at the entry endpoint.
- Canonical gap index: \(\Delta = g_{\mathrm{phys}}\) (exit → entry frames).
- No reverse-time / candidate-backward atom is part of this family.
- Operator-layer horizon mismatch offset is **not** absorbed into the residual
  mean unless separately declared (theory §2 rows 8–9).

## Parameterization family

**`gctm_affine_m0_m1_m2_shared_interface_v1`**

| Model | Definition | Held fixed from prior |
|:--|:--|:--|
| **M0** | Deterministic baseline: Euclidean residual energy \(\lVert r\rVert^2\) | — |
| **M1** | Gap-conditioned Mahalanobis \(q=r^\top S^{-1}r\) / Gaussian NLL under the **same** mean residual | observation interface, candidate universe, missing-value rule, normalization, score composition |
| **M2** | Optional leakage-free context drift: \(r' = r - H d_\Delta(c)\) with exit-causal \(c\) only | all of the above **and** fitting semantics; M1→M2 may not change those fields |

## Ranking-active vs calibration-only

| Mechanism | Class | Mechanical test |
|:--|:--|:--|
| Shared isotropic / scalar covariance \(S=\alpha I\) (or shared positive scale of isotropic \(S\)) | **calibration-only** | Within-event order identical to Euclidean M0; shared rescaling changes absolute \(q\) level only |
| Shared **anisotropic** SPD innovation covariance | **ranking-active** | Can reorder candidates vs Euclidean M0 (constructive CEX retained) |
| Causally declared **candidate-specific** observation / innovation covariance | **ranking-active** | Can change order; can diverge \(q\) vs NLL (L5.2-style CEX retained) |
| Interaction with a frozen base score via predeclared monotone map | admissible in principle | not required for the minimal family sealed here |

**Primary ordering-active mechanism sealed by this declaration:**
`anisotropic_shared_innovation_covariance`.

`shared_isotropic_scalar_covariance` is **calibration-only** and is not ranking
evidence. Candidate-specific covariance must not be used for ranking without a
declared source and causal availability (fail closed otherwise).

Not treated as ranking evidence: lower mean distance alone, pooled AUC, raw
cross-event NLL improvement, overall pooled pair accuracy.

## Mandatory invariants (I1–I12)

Implemented and executed by
`scripts/tools/gctm_d1/invariants.py` /
`scripts/tools/run_gctm_d1_diagnostic.py`:

1. Every candidate pair belongs to exactly one event.
2. Candidate identity and event membership unchanged across M0/M1/M2.
3. Pair counts reconcile with event candidate counts.
4. Calibration and event-local ranking are separate claim spaces.
5. Shared scalar covariance cannot change candidate-local ordering.
6. Under shared covariance (and matched dimension/gap/context/mode), \(q\) and NLL
   produce identical rankings.
7. Score transforms used for ranking are monotone under frozen orientation.
8. Scale-dependent comparisons rejected unless normalization is frozen.
9. Aggregate improvement cannot hide a protected-stratum loss (incl. short-gap).
10. Undefined inverse / det / covariance / missing / tie behavior fails closed.
11. Non-identifiable quantities are explicitly listed.
12. Constructive counterexamples are retained, not averaged away.

Machine binding of all twelve is recorded in
[`invariant_report.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/invariant_report.json)
(`n_invariants=12`, `n_passed=12`, `all_passed=true`).

## Identifiability limits (explicit)

- \(P_{xx}\) vs \(R_1\) split: structurally non-identifiable without gauge fixing.
- \(\operatorname{asym}(P_{xv})\) invisible under \(H_x\).
- \(\gamma\) unknown without joint-map regime: not established; D1 treats
  \(\gamma\) as a **declared** scoring parameter when used, not as identified.
- CAL scale \(\alpha_\Delta\) not identifiable from RANK order alone.
- Single position-only event cannot identify the full \(\{P_0,R_1,D\}\) quotient.

## Consumer interface

Machine-readable interface:
[`consumer_interface.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/consumer_interface.json)

Compatibility requirements matrix (requirements only; gates remain `missing`):
[`compatibility_requirements_matrix.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/compatibility_requirements_matrix.json)

This is **interface-ready only**. Any runtime consumption requires a separate
owner-accepted H0 compatibility verdict on the independent gates:

```text
gctm_d1_to_h0_route5_b1_compatibility_v1
gctm_d1_to_gctm_b1_compatibility_v1
```

Both gates remain **`missing`**. Missing / partial / rejected verdicts select
`reject_runtime_consumption`. The two runtime consumers are independent;
acceptance on one gate never implies acceptance on the other.

## Terminal procedure (three-way, mechanical)

`accepted_terminal_procedure_id` =
`gctm_d1_mechanical_three_way_terminal_v1`

Exactly one of (mechanical selection order):

1. `GCTM_D1_BOUNDED_NO_GO` — invariants fail **or** ranking-active/calibration
   distinction falsified
2. `GCTM_D1_DIAGNOSTIC_SEAL` — invariants + ranking-active pass, but consumer
   interface incomplete
3. `GCTM_D1_INTERFACE_READY` — invariants + ranking-active + complete validated
   interface

Selection is mechanical via `select_terminal(..., interface_complete=...)`.
A provisional terminal string from seal-candidate generation is **not** an
owner-accepted charter execution and does **not** move canonical registry
`state` off `none`. Declaration acceptance does **not** promote
`GCTM_D1_INTERFACE_READY` to a canonical diagnostic terminal.

Validators report structural validity only; they do not assert owner authority,
runtime compatibility, or activation eligibility.

## Prohibited actions (reaffirmed)

- H0 re-entry or historical H0 packet reinterpretation as runtime-faithful
- Activation of `H0_ROUTE5_B1`, `GCTM_B1`, or `GCTM_O1`
- Production preset / tracker state modification
- Fit after held-out reveal
- Candidate-universe change between M0/M1/M2
- Runtime B1 registry updates
- Treating declaration acceptance as charter execution or WIP acquisition

## Artifacts

| Deliverable | Path |
|:--|:--|
| Declaration (this file) | `docs/modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md` |
| Machine sidecar | `.../evidence/gctm_d1_substrate_agnostic_ranking_20260723/declaration_sidecar.json` |
| Synthetic fixture pack | `.../fixture_pack.json` (+ `.sha256`) |
| Invariant report | `.../invariant_report.json` |
| M0/M1/M2 event packet | `.../event_level_diagnostic_packet.json` |
| Consumer interface | `.../consumer_interface.json` |
| Compatibility matrix | `.../compatibility_requirements_matrix.json` |
| Terminal report | `.../terminal_report.json` |
| Identities | `.../identities.json` |
| Manifest | `.../manifest.json` |
| Runner | `scripts/tools/run_gctm_d1_diagnostic.py` |
| Core library | `scripts/tools/gctm_d1/` |
| Contract tests | `tests/contract/test_gctm_d1_ranking_diagnostic_v1.py` |
