# Observability-weighted directional likelihood — pre-outcome declaration

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-08-27 -->
<!-- doc-module: semantic -->

> **One-line:** test whether direction becomes empirically less observable at low
> covariance-normalized velocity evidence, and whether conditioning a standalone
> direction score on that observability exposes held-out ranking information
> hidden by raw cosine direction. This is an offline, MOT17-internal
> score-ranking capability map, not a runtime or MOT-efficacy claim.

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
Claim ceiling           MOT17-internal SR2 held-out ranking on the frozen
                        offline B1 universe; not external validation
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

The intended phenomenon is not "slow objects should ignore motion." It also is
not a theorem about projected normals, and v1 keeps the two apart:

> **Empirical hypothesis (tested).** On held-out GT-positive pairs, angular
> residual concentration is lower where the covariance-normalized velocity
> evidence \(q_v\) is low than where it is high.
>
> **Modeling convention (chosen, not tested).** In the low-observability limit
> v1 *conservatively* decays directional evidence toward the uniform-relative
> score. This is a policy choice about what a weakly identified angle is allowed
> to assert; it is not a claim that the direction law of a general anisotropic
> two-dimensional Gaussian tends to uniform as its mean tends to zero. Under a
> general projected normal that limit is not uniform — only the isotropic case
> degenerates that way — so the convention is named here rather than derived.

Accordingly \(q_v\) is an **estimator-defined observability index**:
covariance-normalized velocity evidence under the frozen effective covariance of
§5.1. It is deliberately not called a velocity signal-to-noise ratio. That
covariance absorbs bbox jitter, within-window curvature and model residual
together, so \(q_v\) is a proxy that orders regimes, not an identified physical
quantity. A phenomenon pass therefore supports only that this proxy separates a
less concentrated from a more concentrated regime, and carries no causal
attribution to detector noise.

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

### 4.1 Byte identity is not relational identity

The nine hashes answer "are these the same bytes". They cannot answer whether a
`pairs.csv` row is reconstructible from exactly one trajectory support. The
ordered obligation is therefore

```text
byte identity  →  semantic relational integrity  →  metrics
```

and the middle step is frozen now as `source_relation_contract` in the machine
study spec: trajectory `(seq, id, frame)` uniqueness; each lost endpoint and
candidate first point resolving to exactly one trajectory row; the last-four
actual-frame window present for every retained lost id; `gap` equal to candidate
first frame minus lost last frame; bottom-center and height reconstructed from
trajectory bytes; each pair row joining its support exactly once; one enumerated
reason per unjoined row; and joined plus excluded reconciling to the included row
count.

**It is frozen pre-seal and executed only post-seal.** Running it now would read
the frozen pair table, which the current authority forbids — `formal_rows_read`
must stay `0` — and would spend the blind boundary that the seal exists to
protect. The post-seal runner executes it as its first stage, fails closed to
`OWDL_INVALID_STUDY`, and emits `joined_candidate_count`, exclusion counts by
reason, and a reconstructed-support digest **before any metric or label
aggregate is computed**.

### 4.2 Candidate and event space

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

### 5.1 Fold-only effective residual covariance

\(\widehat\Omega\) is an **effective residual covariance**, not a calibrated
measurement noise. It is what is left over from a local linear fit, and it
absorbs bbox jitter, within-window curvature and model residual together. No
step of this study identifies those components separately.

For each leave-one-sequence-out fold, use tracks from the other six sequences
only. For every predicted track with sufficient history, take its first and
last four-point windows, deduplicated by the **exact support**
`(seq, track_id, frame_tuple)`. Keying on `side` instead would count a short
track's first and last window twice when they are the same four samples. Windows
that merely *overlap* are retained, and the residual degrees of freedom are then
not fully independent; this is declared, not corrected. Fit two-dimensional
linear OLS motion and pool the residual outer products after dividing each
residual by its sample height:

\[
\widehat\Omega_{-s}
=
\frac{\sum_w E_{w,h}^{\mathsf T}E_{w,h}}
{\sum_w(4-2)}.
\]

No GT label enters this fit. For a sample with height \(h_i\), effective
position covariance is \(\Sigma_i=h_i^2\widehat\Omega_{-s}\). A non-finite,
asymmetric, or non-positive-definite fold covariance makes the study invalid; adding a
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

Define the observability index

\[
q_v=\hat v^{\mathsf T}\Sigma_v^{-1}\hat v.
\]

The concentration is **not** \(1/\sigma^2_{\Delta\theta}\). That identity is the
small-angle limit of the delta method, and the regime this study is about is
exactly where it is least trustworthy. v1 keeps the delta-method variance and
maps it to \(\kappa\) by **matching mean resultants** — wrapped normal to von
Mises:

\[
\exp\!\left(-\tfrac{1}{2}\sigma_{\Delta\theta}^2\right)
=
\frac{I_1(\kappa)}{I_0(\kappa)}.
\]

\(I_1/I_0\) is strictly increasing from \(0\) to \(1\), so \(\kappa\) is unique.
The map recovers \(\kappa\to1/\sigma^2_{\Delta\theta}\) as the variance goes to
zero, and decays smoothly to \(\kappa\to0\) as it grows, with no threshold in
between. \(\kappa\) is obtained by a declared deterministic search — double from
\(1\) until the resultant is reached, then exactly 100 bisections — so no library
optimizer, tolerance or seed enters the frozen procedure.

Two approximation classes are therefore tested, and both are named: the
delta-method propagation of \(\sigma^2_{\Delta\theta}\), and the resultant
matching that turns it into a von Mises concentration. An exact projected-normal
likelihood is outside v1.

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

Under resultant matching the zero case is the **limit** of the map rather than a
separate branch: an undefined angle carries an infinite \(\sigma^2_{\Delta\theta}\),
whose matched resultant is \(0\), whose \(\kappa\) is \(0\). The rule is stated
explicitly anyway so that the implementation and the declaration agree without
relying on a limit being taken. As in §3, uniform here is the conservative
convention v1 chose for an unidentified angle, not an asserted property of the
underlying anisotropic direction law.

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
R4  delete-one-sequence robustness: recomputing pooled Delta_PWA with each
    sequence removed gives Delta_PWA >= 0 in all 7 of 7 deletions
R5a at least 20 short-gap rankable events (§6.3)
R5b short-gap event-macro Delta_PWA >= 0
R6  exact pair/event/GT-partition/policy-candidate conservation passes
```

R5a is exposure and is a validity item; R5b is an effect and is a ranking-box
item. A study that cannot observe the protected stratum is invalid, not
negative.

R4 was a 10,000-replicate sequence-cluster percentile bootstrap whose 95% lower
endpoint had to be non-negative. It is no longer a gate. Ten thousand replicates
over **seven** clusters do not become ten thousand independent facts; the
information is still seven sequences, and reading a percentile interval off it
as a hard positive gate states more than the evidence carries. The question that
actually needs answering — *is the effect carried by one sequence?* — is answered
directly and deterministically by deleting each sequence in turn.

The bootstrap is still computed and still reported, as **descriptive uncertainty
only**. It resamples sequence clusters, never candidate rows, and it selects no
terminal. No uncertainty bound assuming pair or event independence is admissible
in either role.

### 6.3 Protected short-gap stratum

`gap` is a property of a candidate row `(cand_id, lost_id)`, not a single-valued
property of an event, so "the event's gap" has no natural answer and cannot be
the selector. What the protected stratum exists to ask is narrower and precise:

> when the correct predecessor is a recently lost track, is OWDL at least not
> worse than raw direction?

The stratum is therefore selected by the gap of the **GT-positive row**. For a
rankable event \(e=(seq,cand\_id)\) define

\[
G_s(e)=\{r:\ \mathit{gt\_match}(r)=1,\ 1\le \mathit{gap}(r)\le 10\},
\qquad
F(e)=\{r:\ \mathit{gt\_match}(r)=0\}.
\]

\(e\) is a **short-gap rankable event** iff \(|G_s(e)|\ge1\) and
\(|F(e)|\ge1\). Let \(E_s\) be the set of such events. For a policy \(p\),

\[
\mathrm{PWA}^{short}_p(e)
=
\frac{1}{|G_s(e)|\,|F(e)|}
\sum_{g\in G_s(e)}\sum_{f\in F(e)}
\operatorname{pairwin}_p(g,f),
\]

with `pairwin` exactly as in §6.2 — strictly lower score 1, exact tie 0.5, worse
0 — and

\[
\Delta\mathrm{PWA}_{short}
=
\operatorname*{macro}_{e\in E_s}\mathrm{PWA}^{short}_{OWDL}(e)
-
\operatorname*{macro}_{e\in E_s}\mathrm{PWA}^{short}_{raw}(e).
\]

Two consequences are deliberate:

- **Every FP row is retained**, whatever its gap. Filtering negatives to short
  gaps too would change the distractor competition set, and R5 would stop asking
  whether short true relinks are harmed and start asking a narrower, easier
  ranking question.
- **An event with both short and long GT-positive rows contributes only its short
  rows here.** Its long-gap GT comparisons still enter the full-study PWA of
  §6.2, but admitting the whole event into the protected slice would smuggle
  long-gap comparisons into short-gap protection.

## 7. Ordered validity and terminals

Validity is evaluated before phenomenon and ranking effects:

1. exact source hashes/bytes and declaration bindings pass;
2. the §4.1 `source_relation_contract` passes in full, before any metric or
   label aggregate is computed;
3. all seven folds are disjoint and present;
4. four-point history, endpoint, height, and finite-value rules reconcile;
5. every fold covariance is SPD without repair;
6. P1/P2, R1, R5a exposure, candidate conservation, and partition conservation
   pass.

An execution that produces no packet at all — runner crash, build failure,
serialization failure — is a validity failure by this list and takes
`OWDL_INVALID_STUDY`. There is no unmapped outcome and no "describe more and
continue".

Then select exactly one terminal in this order. Every effect terminal carries
`MOT17_INTERNAL` in its own name: the corpus and generalization scope belong in
the identifier, because a terminal is quoted far more often than the table row
beside it. `HELDOUT` is deliberately *not* in the names — that is a protocol
property, and it is stated in the Establishes column instead.

| Terminal | Condition | Establishes | Does not establish / handoff |
|:--|:--|:--|:--|
| `OWDL_INVALID_STUDY` | any validity item fails | no effect conclusion | repair needs a new declaration; no automatic rerun |
| `OWDL_MOT17_INTERNAL_OBSERVABILITY_NOT_SUPPORTED` | valid; P3 fails | on MOT17, held out by fold, this exact effective-covariance phenomenon box did not pass | not proof that direction is always observable or useless, and says nothing about any other corpus |
| `OWDL_MOT17_INTERNAL_NO_DIRECTIONAL_RANKING_POWER_SR2` | valid; P3 passes; any ranking effect item fails | on MOT17, held out by fold, this exact OWDL score did not clear SR2 ranking boxes | no score integration or runtime handoff; not a claim that held-out direction ranking is generally powerless |
| `OWDL_MOT17_INTERNAL_DIRECTION_SIGNAL_SR2` | valid; phenomenon and ranking boxes pass | on MOT17, held out by fold, SR2 directional-channel capability evidence on the frozen universe | only permits proposing a separate **cross-dataset confirmation** declaration; integration design is not yet proposable; no automatic continuation |

Every terminal must report all bins, all folds, policy-identical counts, drop
reasons, delete-one-sequence deltas, and protected short-gap results. A positive
terminal must not be called "MOT improvement," "production-ready," or
"runtime-faithful."

It must also not be called external validation. Outcomes are held out by fold,
but the *design* is not virgin to MOT17: the question, the pair universe, the
height band and the trajectories all descend from earlier MOT17 work on this
line. A positive terminal is therefore MOT17-internal held-out capability
evidence, and the next admissible step is confirmation of direction and sign on
an independent corpus — MOT20 or DanceTrack — not integration design. The
bridge-gate line already supplies the cautionary case: an effect that looked
real on MOT17 did not reproduce on either.

## 8. Implementation phases

1. **Current, pre-seal:** declaration, machine identities, pure math core,
   synthetic contract tests, and a check-only preflight that reads zero formal
   outcome rows. The study spec and the SR2 record are both validated by tests
   that need no frozen source file, so drift is caught on any machine that holds
   only the repository.
2. **After owner seal only:** add the formal runner and evidence schema without
   changing estimator or boxes; re-review exact head. "Byte-bound" is
   discharged concretely: the runner's own SHA-256, the nine source hashes, and
   the sealed head are recorded together in the evidence packet's blind→reveal
   binding before the first outcome row is read, so the reveal names the exact
   code that produced it rather than a policy id.
3. **Separate execution authority:** consume the single declared run, verify
   the packet independently, then record one ordered terminal in a follow-up.

## 9. Derivation provenance and audit limit

### 9.1 What is standard, and what this study is choosing

This is not a literature review, and v1 makes no novelty claim. It is the
minimum needed to tell borrowed mathematics from this study's own choices:

| Element | Status |
|:--|:--|
| von Mises negative log-likelihood and its \(I_0\) normalizer | standard directional statistics |
| von Mises mean resultant \(I_1(\kappa)/I_0(\kappa)\) | standard |
| wrapped-normal mean resultant \(\exp(-\sigma^2/2)\) | standard |
| matching resultants to approximate one circular law by another | standard technique |
| projected normal as the direction law of a bivariate normal | standard; the general anisotropic case does **not** reduce to uniform at zero mean |
| delta-method propagation of \(\sigma^2_{\Delta\theta}\) through two shared-endpoint estimates | **this study's tested approximation** |
| resultant matching applied to that particular variance | **this study's tested approximation** |
| uniform at exactly zero velocity or displacement | **this study's conservative policy choice**, not a projected-normal theorem |
| \(q_v\) as an observability index | **this study's estimator-defined proxy**, not an identified physical quantity |

The last four rows are what a negative terminal exhausts, and what a positive
terminal is conditional on.

### 9.2 Audit limit

The motivating distribution families are von Mises directional likelihood and
projected normal direction distributions; circular resultant length motivates
the calibration summary. The URLs supplied with the research request are
background leads only. This implementation turn does not promote them into a
completed literature review or source-verified novelty claim.
