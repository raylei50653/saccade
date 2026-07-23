<!-- doc-status: proposed -->
<!-- doc-promotion: seal-candidate terminal report for owner review; not charter execution -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: semantic -->

# GCTM D1 seal-candidate terminal report — 2026-07-23

## Authority

```text
record_scope     = diagnostic_seal_candidate
generation_kind  = pre_activation_synthetic_seal_candidate
status           = SEAL_CANDIDATE_GENERATED
not              = owner-accepted charter execution
not              = canonical registry state transition
```

## Provisional selected terminal

# `GCTM_D1_INTERFACE_READY`

Machine packet:
[`evidence/gctm_d1_substrate_agnostic_ranking_20260723/terminal_report.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/terminal_report.json)

Declaration:
[`gctm_d1_ranking_diagnostic_declaration_20260723.md`](gctm_d1_ranking_diagnostic_declaration_20260723.md)

## Execution summary (synthetic seal-candidate only)

| Item | Result |
|:--|:--|
| Substrate | synthetic fixture pack `gctm_d1_synthetic_fixture_pack_v1` |
| Invariants I1–I12 | **all passed** |
| Ranking-active mechanism | `anisotropic_shared_innovation_covariance` **demonstrated** |
| Candidate-specific mechanism | mechanical q vs NLL order difference **demonstrated** |
| Calibration-only mechanism | `shared_isotropic_scalar_covariance` **distinguished** |
| Consumer interface | complete machine-readable artifact |
| Compatibility gates | requirements only; both runtime gates remain `missing` |
| Canonical registry state | **`none`** (seal-candidate bookkeeping only) |
| H0 / runtime claims | **none** |

## Terminal selection rule (three-way)

1. Invariant or ranking-active falsification failure → `GCTM_D1_BOUNDED_NO_GO`
2. Invariants + ranking-active pass, interface incomplete → `GCTM_D1_DIAGNOSTIC_SEAL`
3. Invariants + ranking-active + complete interface → `GCTM_D1_INTERFACE_READY`

## Canonical conclusion

### Maximum claims (seal-candidate)

- Pre-activation synthetic seal-candidate is machine-checkable.
- Calibration-only and ranking-active mechanisms are mechanically distinguishable.
- Ordering-active mechanism is precisely specified.
- Future runtime consumer fields are complete enough for a **separate** H0
  compatibility verdict process (not completed here).

### Blocked claims

- Owner-accepted charter execution
- Canonical registry terminal transition
- Runtime fidelity or H0 capture equivalence
- Activation of `H0_ROUTE5_B1`, `GCTM_B1`, or `GCTM_O1`
- Decision-relevant candidate / WIP acquisition

## Exit line (provisional)

`GCTM_D1_INTERFACE_READY` — seal-candidate only
