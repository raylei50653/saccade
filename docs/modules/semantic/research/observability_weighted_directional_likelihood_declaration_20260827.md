# Observability-weighted directional likelihood — pre-outcome declaration

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-08-27 -->
<!-- doc-module: semantic -->

> **One-line:** test whether direction becomes empirically unobservable at low
> velocity SNR, and whether conditioning a standalone direction score on that
> observability exposes held-out ranking information hidden by raw cosine
> direction. This is an offline score-ranking capability map, not a runtime or
> MOT-efficacy claim.

- Thread: [observability-weighted directional likelihood task](../../../research/threads/observability_weighted_directional_likelihood_task.md)
- Machine study spec: [study v1](observability_weighted_directional_likelihood_study_v1.json)
- Score-ranking record: [SR2 declaration](observability_weighted_directional_likelihood_declaration_20260827.score.json)
- Implementation: [`observability_weighted_directional_likelihood.py`](../../../../scripts/tools/observability_weighted_directional_likelihood.py)

## 1. Authority and seal

This document is a **pre-outcome declaration**. The current implementation may
run synthetic tests and `--check-only` identity validation only. Formal loading
of study outcome rows, fitting fold covariance, computing metrics, producing an
evidence packet, or selecting a terminal is forbidden until the research owner
seals this declaration at one exact merged head.

Seal review freezes this document, the two JSON records, implementation bytes,
tests, and exact source identities together. A later formal run must be a
separately authorized execution from that sealed head. Any change to input
identity, estimator, bins, boxes, tie policy, or terminal order voids the run and
requires a new pre-outcome declaration. No terminal auto-authorizes production
code, a preset change, or another experiment.

## 2. Decision question and claim ceiling

```text
Target decision layer   score-ranking
Study intent            capability map only
Primary comparison      standalone observability-weighted direction cost
                        versus standalone raw cosine direction cost
Claim ceiling           SR2 held-out ranking on the frozen offline B1 universe
Assignment/system       absent; no additive score, Hungarian, tracker, IDF1,
                        HOTA, latency, runtime-fidelity, or production claim
```

The study has two ordered questions:

1. **Phenomenon:** on true relink pairs, does angular residual concentration
   increase from low to high covariance-normalized velocity evidence?
2. **Gap:** if that phenomenon exists, does the normalized likelihood improve
   held-out event-local ordering over raw direction on the identical candidates?

The questions are not interchangeable. A phenomenon pass without a ranking
pass establishes calibration structure only. A ranking pass does not establish
that adding the score to the current bridge cost improves assignments. That
would require a separate design/effectiveness declaration.

## 3. Exact phenomenon and existing evidence gap

The intended phenomenon is not "slow objects should ignore motion." It is:

> When estimated velocity magnitude is small relative to effective positional
> uncertainty, its angle is weakly identifiable and should approach a uniform
> direction observation rather than a confident unit vector.

Existing artifacts leave two concrete gaps:

- the historical speed/turn sweep can stratify by raw speed, but has no vector
  covariance, pair identity, or candidate-event label needed for this question;
- the frozen B1 pair table has labels and `dir_cos`, but no two-dimensional
  velocity covariance; the associated frozen MOT trajectories do retain the
  four-point windows from which an effective covariance can be estimated;
- runtime-faithful R1 events retain exact windows but do not carry the GT
  candidate utility labels needed for an SR2 ranking claim.

Accordingly, v1 joins no new runtime source. It reconstructs four-point windows
from the frozen B1 trajectories and evaluates only the frozen offline pair
universe. The effective covariance absorbs bbox jitter, model residual, and
within-window curvature; this study cannot uniquely attribute it to detector
noise.

## 4. Frozen source universe

The normative file list, byte counts, and SHA-256 values are in the
[machine study spec](observability_weighted_directional_likelihood_study_v1.json),
which is itself checked against
[`observability_weighted_directional_likelihood_study_schema_v1.json`](../../../../scripts/tools/observability_weighted_directional_likelihood_study_schema_v1.json)
before any identity is read: every frozen box, bin, estimator rule, and terminal
in that record is pinned by `const`, so a box can move only by moving the schema
with it in the same reviewed change. It binds:

- pair table
  `out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv`, SHA-256
  `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17`;
- its source context, including `mamba_whole_graph_m`, SDP, relink off, and
  interpolation off;
- the seven exact MOT17 SDP trajectory files under
  `results/MOT17_eval_m_b1_substrate_20260709T092543Z/`.

The table is not a production-fidelity substrate. The source context itself
records its flags as provenance, not verified runtime facts. Therefore no H0,
runtime compatibility, or production-reachable claim may be imported from this
study.

### 4.1 Candidate and event space

- Include rows with `gt_valid == 1` and
  `0.60 <= h_lost_raw / h_cand_raw <= 1.70`.
- Event key: `(seq, cand_id)` — one newborn candidate compared across possible
  lost predecessors.
- Candidate key: `(seq, cand_id, lost_id)`.
- Rankable event: at least one `gt_match == 1` and one `gt_match == 0`.
- Exact duplicate candidate keys invalidate the study.
- Both policies must retain byte-identical ordered candidate keys. Every
  exclusion receives exactly one declared reason and all counts reconcile.

The height interval defines this offline analysis universe. It is not imported
as an accepted production support gate.

## 5. Frozen estimator

All positions are bbox bottom centers. The lost history is exactly the last four
points, and candidate displacement ends at the candidate's first point. Actual
frame values, not row indices, are used.

### 5.1 Fold-only effective position covariance

For each leave-one-sequence-out fold, use tracks from the other six sequences
only. For every predicted track with sufficient history, take its first and
last four-point windows, deduplicated by
`(seq, track_id, side, anchor_frame)`. Fit two-dimensional linear OLS motion and
pool the residual outer products after dividing each residual by its sample
height:

\[
\widehat\Omega_{-s}
=
\frac{\sum_w E_{w,h}^{\mathsf T}E_{w,h}}
{\sum_w(4-2)}.
\]

No GT label enters this fit. For a sample with height \(h_i\), position
covariance is \(\Sigma_i=h_i^2\widehat\Omega_{-s}\). A non-finite, asymmetric,
or non-positive-definite fold covariance makes the study invalid; adding a
ridge, diagonal floor, shrinkage, or post-hoc fallback is forbidden in v1.

### 5.2 Velocity, displacement, and shared endpoint

Let \(w_i\) be the OLS slope weights on the four lost points:

\[
\hat v=\sum_iw_i p_i,
\qquad
\Sigma_v=\sum_iw_i^2\Sigma_i.
\]

For frame gap \(g>0\),

\[
\hat d=\frac{p_c-p_l}{g},
\qquad
\Sigma_d=\frac{\Sigma_l+\Sigma_c}{g^2},
\qquad
\Sigma_{vd}=-\frac{w_l\Sigma_l}{g}.
\]

The cross-covariance is mandatory because velocity and displacement share the
lost endpoint. With angle gradients \(g_v\) and \(g_d\),

\[
\sigma_{\Delta\theta}^2
=g_v^{\mathsf T}\Sigma_vg_v
+g_d^{\mathsf T}\Sigma_dg_d
-2g_v^{\mathsf T}\Sigma_{vd}g_d.
\]

Define

\[
q_v=\hat v^{\mathsf T}\Sigma_v^{-1}\hat v,
\qquad
\kappa=1/\sigma_{\Delta\theta}^2.
\]

The delta-method concentration is the exact tested approximation class; an
exact projected-normal likelihood is outside v1.

### 5.3 Scores and zero-vector semantics

Baseline raw direction:

\[
C_{raw}=1-\cos\Delta\theta.
\]

Candidate score, expressed relative to the uniform angular density:

\[
C_{owdl}=\log I_0(\kappa)-\kappa\cos\Delta\theta.
\]

The normalizer is retained because \(\kappa\) varies by candidate. Lower is
better. If either vector is exactly zero, angle is undefined and v1 assigns
\(\kappa=0\), `C_owdl=0` (uniform-relative evidence), and `C_raw=1` (the
historical cosine convention `dir_cos=0`); the candidate is retained. There is
no near-zero threshold and no epsilon in the denominator.

## 6. Held-out protocol

Each of the seven sequences is held out once. Effective covariance is fitted on
the other six sequences without labels, then all quantities and metrics are
computed on the held-out sequence. There is no in-sample alternative, no
hyperparameter search, and no refit after reveal.

The fixed \(q_v\) bins are `[0,1)`, `[1,4)`, `[4,9)`, and `[9,inf)`. Raw speed
bins are descriptive only and cannot force a terminal.

### 6.1 Phenomenon metric

On held-out `gt_match == 1` pairs, calculate mean resultant length

\[
R_B=\left|\frac{1}{|B|}\sum_{i\in B}e^{j\Delta\theta_i}\right|
\]

within each \(q_v\) bin. The phenomenon box passes iff:

```text
P1  low q_v (<1) has at least 30 GT pairs
P2  high q_v (>=9) has at least 30 GT pairs
P3  R_high - R_low >= 0.15
```

P1/P2 are observation-validity requirements. P3 is the directional
observability effect. It supports only that this frozen empirical estimator
separates a less concentrated from a more concentrated regime.

### 6.2 Ranking metric

For each rankable event, compare every GT row with every FP row. A strictly
lower score contributes 1, an exact tie 0.5, and a worse score 0. Event PWA is
the mean over its GT–FP pairs; study PWA is the macro-average over events. The
primary effect is

```text
Delta_PWA = held-out event-macro PWA(C_owdl)
            - held-out event-macro PWA(C_raw)
```

The ranking box passes iff all of:

```text
R1  at least 100 rankable held-out events
R2  pooled Delta_PWA >= +0.02
R3  per-fold Delta_PWA >= 0 in at least 5/7 folds
    and > 0 in at least 4/7 folds
R4  10,000-replicate sequence-cluster percentile bootstrap, seed 20260827:
    lower endpoint of the 95% interval >= 0
R5  short gap (1..10) has at least 20 events and Delta_PWA >= 0
R6  exact pair/event/GT-partition/policy-candidate conservation passes
```

The bootstrap resamples sequence clusters, never candidate rows. No uncertainty
bound assuming pair or event independence is admissible.

## 7. Ordered validity and terminals

Validity is evaluated before phenomenon and ranking effects:

1. exact source hashes/bytes and declaration bindings pass;
2. all seven folds are disjoint and present;
3. four-point history, endpoint, height, and finite-value rules reconcile;
4. every fold covariance is SPD without repair;
5. P1/P2, R1, R5 exposure, candidate conservation, and partition conservation
   pass.

Then select exactly one terminal in this order:

| Terminal | Condition | Establishes | Does not establish / handoff |
|:--|:--|:--|:--|
| `OWDL_INVALID_STUDY` | any validity item fails | no effect conclusion | repair needs a new declaration; no automatic rerun |
| `OWDL_OBSERVABILITY_NOT_SUPPORTED` | valid; P3 fails | this exact covariance/delta-method phenomenon box did not pass | not proof that direction is always observable or useless |
| `OWDL_NO_HELDOUT_DIRECTIONAL_RANKING_POWER` | valid; P3 passes; any ranking effect item fails | this exact OWDL score did not clear SR2 ranking boxes | no score integration or runtime handoff |
| `OWDL_DIRECTION_SIGNAL_PRESENT_SR2` | valid; phenomenon and ranking boxes pass | offline SR2 directional-channel signal on the frozen universe | only permits proposing a separate integration-design declaration; no automatic continuation |

Every terminal must report all bins, all folds, policy-identical counts, drop
reasons, and protected short-gap results. A positive terminal must not be called
"MOT improvement," "production-ready," or "runtime-faithful."

## 8. Implementation phases

1. **Current, pre-seal:** declaration, machine identities, pure math core,
   synthetic contract tests, and a check-only preflight that reads zero formal
   outcome rows. The study spec and the SR2 record are both validated by tests
   that need no frozen source file, so drift is caught on any machine that holds
   only the repository.
2. **After owner seal only:** add the byte-bound formal runner and evidence
   schema without changing estimator or boxes; re-review exact head.
3. **Separate execution authority:** consume the single declared run, verify
   the packet independently, then record one ordered terminal in a follow-up.

## 9. Background references and audit limit

The motivating distribution families are von Mises directional likelihood and
projected normal direction distributions; circular resultant length motivates
the calibration summary. The URLs supplied with the research request are
background leads only. This implementation turn does not promote them into a
completed literature review or source-verified novelty claim.
