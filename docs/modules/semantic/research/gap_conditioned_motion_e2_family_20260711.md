<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E2 position-only family freeze

> **E2 research acceptance:** `ACCEPTED_WITH_LIMITS` · freeze status
> `FROZEN_ACCEPTED_WITH_LIMITS`. This packet freezes a reduced, global,
> position-only M1-P/M2-P transition marginal and its train-only fitting rules.
> It does not persist or analyze E3 headline fold outputs, authorize Phase B,
> or claim a V1–V5 verdict. E3 signal generation is authorized under the sealed
> family and LOO/output contracts only.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)

E0: [substrate audit](gap_conditioned_motion_e0_20260711.md)

E1: [deterministic baseline](gap_conditioned_motion_e1_m0_20260711.md)

Packet: [evidence/gap_conditioned_motion_e2_family_20260711/](evidence/gap_conditioned_motion_e2_family_20260711/manifest.json)

## 1. Claim boundary

The E0 table identifies only

\[
d_i = \frac{x^{cand}_{foot,i} - x^{lost}_{foot,i}}{h_{ref,i}} \in \mathbb{R}^2,
\qquad \Delta t_i = gap_i.
\]

Accordingly, E2 freezes a population transition **marginal** over endpoint
displacement. It is not the originally proposed joint
\(p(x_1,v_1\mid x_0,v_0,\Delta t,c)\), does not reconstruct velocity direction
from scalar speed/residual fields, and does not use sequence as a headline
context. Coordinates are image `x-right/y-down`, normalized by the pair's
stored `h_ref`; time is the integer frame gap in `[1, 300]`.

## 2. Frozen family

All members share a global mean \(E[d\mid t]=\beta t\). The fit universe is
`gt_valid AND gt_match` from training sequences only.

| Model ID | Covariance growth | Interpretation |
|:--|:--|:--|
| `M1P-GLOBAL-CV` | \(t^2\Sigma_v\) | random constant residual velocity marginal |
| `M2P-GLOBAL-OU-H30` | \(k_\gamma(t)\Sigma_u\), half-life 30 frames | integrated stationary residual OU |
| `M2P-GLOBAL-OU-H90` | same, half-life 90 frames | integrated stationary residual OU |
| `M2P-GLOBAL-OU-H270` | same, half-life 270 frames | integrated stationary residual OU |

For M2-P,

\[
\gamma=\log(2)/h,\qquad
k_\gamma(t)=\frac{2(\gamma t-1+e^{-\gamma t})}{\gamma^2}.
\]

This kernel approaches \(t^2\) as \(\gamma\to0\), making M1-P the
non-decaying limit and keeping the M1/M2 comparison structurally matched. The
three declared half-lives are the complete first-round family; E3 must not add
or pick an undeclared value.

## 3. Fit, regularization, and LOO firewall

For each member, weighted MLE estimates global \(\beta\), standardizes residuals
by \(\sqrt{k(t)}\), and fits one full `2x2` base covariance. Its eigenvalues are
floored identically for all rows at

```text
max(1e-8, 1e-6 * max(trace(Sigma) / 2, 0))
```

The parameter artifact must record the fit count, covariance, whether the
floor fired, and the applied floor. No GT/FP-specific regularization exists.

Per LOO fold, all four members fit on training-sequence GT transitions. The
member with minimum summed **training-GT** NLL is selected; numerical ties at
`1e-12` relative tolerance use the declared order `M1`, `M2-H270`, `M2-H90`,
`M2-H30` (simpler/slower-decay first). The held-out sequence contributes no
fit, covariance, calibration, fallback, or selection statistic. Full-pool fits
may be diagnostic only and must have distinct artifact/fold IDs.

E3 must build each fold through the lineage-aware fold builder, not call the
array-only numerical primitive as an artifact boundary. Fit-row lineage is the
SHA256 of the canonical compact JSON array of sorted
`[seq, lost_id, cand_id]` keys.

Every parameter artifact requires:

```text
freeze_id · model_id · parameter_artifact_id · fold_id
held_out_sequence · train_sequences · fit_row_count · fit_row_key_sha256
source_pairs_sha256 · dimension · drift_per_frame · base_covariance
regularization_applied · eigenvalue_floor · training_total_nll
```

Every fold-selection artifact requires:

```text
selection_artifact_id · fold_id · held_out_sequence · train_sequences
fit_row_count · fit_row_key_sha256 · source_pairs_sha256
training_nll_by_model · selected_model_id · selection_tolerance · model_order
```

Artifact IDs are SHA256 digests of canonical JSON payloads before the ID field
is added. This makes both parameter and selection records independently
rebuildable and prevents a free-form `fold_id` from serving as lineage proof.

## 4. Frozen E3 output contract

Every pair/model output must retain:

- `freeze_id`, `model_id`, `parameter_artifact_id`, and `fold_id`;
- parameter lineage through the required artifact above;
- \(q_{motion}\), `log_det_covariance`, Gaussian constant, and full NLL as
  separate fields;
- source pair identity, gap, sequence, label, and source SHA lineage.

E3 must emit pair scores for **all four** frozen members. The selected model is
an additional fold marker only; non-winner scores must not be filtered, because
A8 requires the complete matched M1/M2 surface.

The scoring identity is

\[
E_{motion}=\tfrac12(q_{motion}+\log\det\Sigma(t)+2\log(2\pi)).
\]

E3 may generate these fields from the frozen table. It may not run A1–A8,
calibrate on held-out data, modify the family, or declare Phase B open.

## 5. E2 gates and result

The runner rechecks source SHA, finite coordinates, positive `h_ref`, usable
gap range, frame-window identity, coordinate/time semantics, and provenance.
On the frozen seven-sequence table:

```text
source SHA:        0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17
finite/support:    PASS
GT fit rows:       340
LOO support:       PASS (minimum training fold = 183 rows)
LOO artifact:      PASS (per-fold parameter + selection lineage)
headline context:  global only
E3:                AUTHORIZED (signal generation only)
Phase B:           unauthorized
```

The sealed per-sequence support is:

| Held-out sequence | Held-out GT | Training GT | Fit-row hash prefix |
|:--|--:|--:|:--|
| MOT17-02-SDP | 72 | 268 | `9e7fe454…` |
| MOT17-04-SDP | 12 | 328 | `9caca30e…` |
| MOT17-05-SDP | 42 | 298 | `ada1a106…` |
| MOT17-09-SDP | 14 | 326 | `164034ec…` |
| MOT17-10-SDP | 157 | 183 | `8b1e67dd…` |
| MOT17-11-SDP | 20 | 320 | `0aa3abb4…` |
| MOT17-13-SDP | 23 | 317 | `a88c8ed7…` |

Full hashes and both count maps are sealed in `model_family.json`. Every fold
exceeds the numerical primitive's minimum support of three fit rows.

### 5.1 Review acceptance boundary

PR #110 research review (second pass) records:

```text
Engineering / reproducibility: PASS
E2 position-only family mathematics: ACCEPT
LOO lineage and selection contract: ACCEPT
E2 research acceptance: ACCEPTED_WITH_LIMITS
E3 signal generation: AUTHORIZED
Phase B / A1–A8: NONE
V1–V5 verdict: NOT_YET_EVALUATED
Production / hook authorization: NONE
```

Authorized E3 scope is only:

```text
rebuild 7 LOO folds
persist 28 parameter artifacts
persist 7 selection artifacts
emit all 4 model scores per pair × fold
```

The canonical verifier may temporarily rebuild seven-fold fitting to check
lineage. That rebuild is not E3 evidence and must not be treated as Phase B
analysis. E3 must not calibrate, run A1–A8, change the family, or issue a
V1–V5 verdict.

## 6. Reproduction

```bash
uv run python \
  docs/modules/semantic/research/evidence/gap_conditioned_motion_e2_family_20260711/run_e2_family_freeze.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify

uv run pytest tests/unit/test_gap_conditioned_motion_e2_family.py -q
```

The verifier regenerates `model_family.json`, `recorded_output.txt`, and the
manifest byte-for-byte. Production/default behavior and evidence promotion
remain unchanged.
