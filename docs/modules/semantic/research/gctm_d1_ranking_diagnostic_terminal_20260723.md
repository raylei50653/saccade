<!-- doc-status: proposed -->
<!-- doc-promotion: diagnostic terminal report for owner review; not runtime evidence -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: semantic -->

# GCTM D1 terminal report — 2026-07-23

## Selected terminal

# `GCTM_D1_INTERFACE_READY`

Machine packet:
[`evidence/gctm_d1_substrate_agnostic_ranking_20260723/terminal_report.json`](evidence/gctm_d1_substrate_agnostic_ranking_20260723/terminal_report.json)

Declaration:
[`gctm_d1_ranking_diagnostic_declaration_20260723.md`](gctm_d1_ranking_diagnostic_declaration_20260723.md)

## Execution summary

| Item | Result |
|:--|:--|
| Substrate | synthetic fixture pack `gctm_d1_synthetic_fixture_pack_v1` |
| Invariants I1–I12 | **all passed** |
| Ranking-active mechanism | `anisotropic_shared_innovation_covariance` **demonstrated** |
| Calibration-only mechanism | `shared_isotropic_scalar_covariance` **distinguished** |
| Consumer interface | complete machine-readable artifact emitted |
| Compatibility gates | requirements matrix emitted; both runtime gates remain `missing` |
| H0 / runtime claims | **none** |

## Constructive counterexamples retained

1. **Shared anisotropic \(S\)** reorders vs Euclidean M0
   (`E_shared_aniso`: M0 prefers `c_false`, M1 prefers `c_true`).
2. **Candidate-specific \(S\)** diverges \(q\) vs NLL order (`E_cand_spec`).
3. **M2 context drift** reorders vs M1 under fixed interface (`E_m2_drift`).
4. **L5.2 numeric** \((r,S)=(1,1)\) vs \((1.2,4)\): \(q\) prefers the second,
   NLL prefers the first.

## Canonical conclusion

### Maximum claims

- The declared substrate-agnostic observation/parameterization family is
  machine-checkable on synthetic non-runtime inputs.
- Calibration-only and ranking-active mechanisms are mechanically
  distinguishable.
- The ordering-active mechanism is precisely specified:
  **anisotropic shared innovation covariance** (with candidate-specific
  covariance admissible when causally declared).
- Future runtime consumer fields and semantics are complete enough for a
  **separate** H0 compatibility verdict process.
- Non-identifiable quantities are explicitly listed; absence/singularity fails
  closed toward `reject_runtime_consumption`.
- Results do not depend on pooled-row independence.

### Blocked claims

- Runtime fidelity or H0 capture equivalence
- H0→GCTM consumer compatibility completed
- Activation of `H0_ROUTE5_B1`, `GCTM_B1`, or `GCTM_O1`
- Decision-relevant registry candidate
- WIP acquisition
- Production observation-mode freeze
- Pooled-row independence as ranking evidence
- Repair or re-entry of historical H0 packets

## Registry effect

On owner acceptance of this terminal:

- update **only** the `GCTM_D1` diagnostic state;
- close this diagnostic task;
- do **not** alter `quantity.bridge_capture_provenance`, `H0_ROUTE5_B1`,
  `GCTM_B1`, `GCTM_O1`, or the decision-relevant candidate set.

Charter activation and WIP still require separate owner acceptance of the
declaration **and** a separate scheduling decision.

## Exit line

`GCTM_D1_INTERFACE_READY`
