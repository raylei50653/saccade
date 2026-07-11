<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — E2 position-only family freeze

> **E2 delivery state:** `FROZEN_PENDING_RESEARCH_ACCEPTANCE`. This packet
> freezes a reduced, global, position-only M1-P/M2-P transition marginal and
> its train-only fitting rules. It does not fit headline folds, generate E3
> pair signals, authorize Phase B, or claim a V1–V5 verdict.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

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

## 4. Frozen E3 output contract

Every pair/model output must retain:

- `model_id`, `parameter_artifact_id`, `fold_id`, fit-row count;
- dimension `d=2` and regularization flag/floor provenance;
- \(q_{motion}\), `log_det_covariance`, Gaussian constant, and full NLL as
  separate fields;
- source pair identity, gap, sequence, label, and source SHA lineage.

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
headline context:  global only
Phase B:           unauthorized
```

This is an engineering-complete freeze candidate, not chat-side research
acceptance. E3 remains gated on owner acceptance of this family and packet.

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
