<!-- doc-status: proposed -->
<!-- doc-promotion: owner-reviewable sealed diagnostic declaration; not runtime evidence -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: semantic -->

# GCTM D1 — substrate-agnostic ranking diagnostic declaration (v1)

## Status

**Pre-activation synthetic seal-candidate declaration** (owner-reviewable).
Not charter execution. Not canonical registry terminal transition. Not WIP.
Not runtime evidence. Not an H0 compatibility verdict.

```text
generation_kind = pre_activation_synthetic_seal_candidate
status          = SEAL_CANDIDATE_GENERATED
```

Machine sidecar:
[`evidence/gctm_d1_substrate_agnostic_ranking_20260723/declaration_sidecar.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/declaration_sidecar.json)

Charter:
[`docs/research/threads/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md`](../../../research/threads/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md)

Slot identity:
[`gctm_b1_slot_identity_decision_v1`](../../../research/contracts/gctm_b1_slot_identity_decision_v1.json)

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

## Frozen identities

| Field | Frozen value |
|:--|:--|
| `diagnostic_id` | `gctm_d1_substrate_agnostic_ranking_v1` |
| `accepted_gctm_theory_identity` | `docs/research/models/gap_conditioned_stochastic_transition_spec_v1.md` (+ sha in sidecar) |
| `accepted_gctm_lemmas_identity` | `docs/research/models/gap_conditioned_stochastic_transition_lemmas_v1.md` (+ sha in sidecar) |
| `accepted_score_contract_identity` | `score_ranking_evidence_contract_v1` (+ sha in sidecar) |
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

Missing / partial / rejected verdicts select `reject_runtime_consumption`.

## Terminal procedure (three-way, mechanical)

Exactly one of:

1. `GCTM_D1_BOUNDED_NO_GO` — invariants fail **or** ranking-active/calibration
   distinction falsified
2. `GCTM_D1_DIAGNOSTIC_SEAL` — invariants + ranking-active pass, but consumer
   interface incomplete
3. `GCTM_D1_INTERFACE_READY` — invariants + ranking-active + complete validated
   interface

Selection is mechanical via `select_terminal(..., interface_complete=...)`.
A provisional terminal string from seal-candidate generation is **not** an
owner-accepted charter execution and does **not** move canonical registry
`state` off `none`.

## Prohibited actions (reaffirmed)

- H0 re-entry or historical H0 packet reinterpretation as runtime-faithful
- Activation of `H0_ROUTE5_B1`, `GCTM_B1`, or `GCTM_O1`
- Production preset / tracker state modification
- Fit after held-out reveal
- Candidate-universe change between M0/M1/M2
- Runtime B1 registry updates

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
