<!-- doc-status: active -->
<!-- doc-promotion: none; predeclared analysis protocol, not evidence -->
<!-- doc-date: 2026-07-11 -->
<!-- doc-module: semantic -->

# Gap-conditioned probabilistic motion — Phase B (A1–A8) predeclared design

> **Design only.** Phase B execution remains **unauthorized**. This document
> predeclares the complete A1–A8 analysis protocol — statistics, support
> layers, numeric criteria, and the V1–V5 decision rule — **before any E3
> pair score exists**. It is frozen at merge; the merge commit is the
> predeclaration seal that E1's acceptance limit said was missing for
> post-hoc criteria. Phase B may run only after (a) this design is
> review-accepted, and (b) E3 signals are sealed under the E2 output
> contract. It authorizes no calibration change, family change, production
> change, or verdict by itself.

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)

Inputs: [E0 substrate](gap_conditioned_motion_e0_20260711.md) ·
[E1 baseline](gap_conditioned_motion_e1_m0_20260711.md) ·
[E2 family freeze](gap_conditioned_motion_e2_family_20260711.md) ·
[production substrate mapping](production_substrate_mapping_20260711.md) (binding §5/§6/§8/§9) ·
[D0 estimator-fidelity gate — Issue #112](https://github.com/raylei50653/saccade/issues/112)

## 1. Ordering and authorization rule

```text
1. This design merges (review-accepted)            ← predeclaration seal
2. E3 runs: 7 LOO folds · 28 parameter + 7 selection artifacts ·
   all 4 model scores per pair × fold (sealed E2 contract §4/§5.1)
   E3 emits signals ONLY; no table below may be computed in the E3 PR
3. Owner records Phase B authorization in the thread
4. A1–A8 run as ONE reproduction entrypoint over the sealed E3 table
5. Exactly one V1–V5 bounded verdict per §6
```

Any criterion change after step 2 requires a new design revision with its
own seal and must be reported as a deviation, not silently applied.

## 2. Inputs and evaluation layers

**Score table.** The sealed E3 pair×fold×model table (source pair table SHA
`0ae38967…`). Every pair in sequence *s* is analyzed under fold-*s*
held-out parameters only; full-pool fits are diagnostic-only with distinct
artifact IDs (E2 §3). Scores: `q_motion` (df = 2), `log_det_covariance`,
and `E_motion` (full NLL), kept as separate fields. Members: all four of
`GCM-E2-POSITION-ONLY-v1`; the per-fold selection winner is a marker, never
a filter.

**Support layers** (substrate mapping §5/§8; all cuts row-level):

| Layer | Support | Role in Phase B |
|:--|:--|:--|
| **Primary** | \(S_A\): gap ∈ [1, 26] | drives the verdict-facing statistics |
| Secondary | \(S_{C2}\): gap ∈ [1, 60] · \(S_B\): gap ∈ [2, 45] | reported, non-verdict |
| Exploratory | all-gap [1, 300] | representation-level only; may inform V-boxes 2/3 below but no deployment narrative |

**Primary gap cells.** Canonical bins stay frozen (1–10 · 11–30 · 31–60 ·
61–150 · 151–300). \(S_A\) intersects only two: `1–10` and `11–26∩S_A`
(bin 11–30 truncated at the row level). These two cells × member are the
primary cell set. No post-hoc rebinning.

**Support floors (frozen).**

```text
LOW_SUPPORT cell:   GT rows < 15   → diagnostic only, excluded from criteria
qualifying fold:    held-out GT rows ≥ 20 on the evaluated support
                    (all-gap qualifiers per E2 §5: 02/05/10/11/13;
                     04 and 09 are diagnostic-only at every layer)
```

## 3. D0 coupling (parallel precondition gate for threshold transfer)

A1–A8 consume **offline-builder atoms** and the E2 observation
\(d=\Delta\text{foot}/h_{ref}\) only; they never cite consumer-A kernel
quantities and do **not** wait on D0 execution. D0 (Issue #112) gates the
**claim ceiling** of the outputs, per the substrate-mapping claim ladder:

| D0 verdict | Ceiling for A6 / verdict language |
|:--|:--|
| `threshold_transfer_supported` | A6 threshold schedules may be phrased as consumer-A predicate candidates (claim level 2) |
| `rank_only_transfer_supported` | rank/shape/morphology statements only; numeric thresholds do not port |
| `not_fidelity_aligned` (or D0 not yet run) | signal-level regularity only (claim level 1) |

The V1–V5 verdict itself is a **representation** verdict and is independent
of D0. Any follow-up hook proposal inherits both this verdict and the D0
ceiling separately.

## 4. A1–A8 specifications

Common conventions: AUC is tie-aware GT AUC of the negated score (higher =
better GT separation), identical to E1. The q90 tail is the within-cell
pooled 90th percentile, inclusive on ties, identical to E1. All criteria
below apply per member unless stated.

### A1 — gap-bin calibration

GT rows only, per (gap cell × member), on primary cells plus all-gap bins
as exploratory. Under the model, `q_motion` ~ χ²(df=2). Record dispersion
ratio `r = mean(q)/2` and empirical coverage `c50/c90/c95` at the χ²₂
50/90/95% points (4.605 at 90%). Frozen classification on `c90`:

```text
approximately calibrated   c90 ∈ [0.85, 0.95]
over-dispersed model       c90 > 0.95   (Σ too wide; over-covers)
under-dispersed model      c90 < 0.85   (Σ too narrow)
```

`r`, `c50`, `c95` are reported diagnostics, not gates.

### A2 — role-reversal rate

Apply E1's frozen descriptive criterion (`AUC < 0.5` **and** q90-tail GT
enrichment `> 1`) to `E_motion` and `q_motion` per (gap bin × member),
all-gap layer, with per-cell sequence attribution of tail GT.
`R_flip` = count of reversal cells. E1's M0 baseline is 0/20.
**Criterion: `R_flip = 0` new aggregate reversal cells.** Descriptive
comparability with E1 is the point; this is not a hypothesis test.

### A3 — short-gap retention

On the two primary cells, per member, recompute `bridge_dist` AUC on the
identical row set (no reuse of E1's all-support numbers) and compare:

```text
retention(member) := AUC(−E_motion) ≥ AUC(−bridge_dist) − 0.05
                     in BOTH primary cells (non-LOW_SUPPORT)
```

`q_motion` AUC is reported alongside; retention is judged on `E_motion`.

### A4 — escape-tail audit

Cohort = the four sealed far-Hamming GT tracks (all MOT17-10-SDP) from the
escape-tail forensic packet, keyed by `[seq, lost_id, cand_id]`. All four
are scored under the genuinely held-out fold-10 parameters. Per track ×
member: within-native-gap-bin percentile of `E_motion` among that bin's FP
rows, and whether the pair sits inside the pooled q90 tail.

```text
addressed(track, member) := E_motion below the within-bin pooled q90 tail
tail-reduction box       := addressed for ≥ 3 of 4 tracks
anti-over-diffusion tie  := no native cohort bin classified over-dispersed
                            (A1) for that member
```

Cohort gaps are long-gap → this analysis lives on the **exploratory**
layer (substrate mapping §8): it may support the representation verdict
but never a deployment narrative.

### A5 — separability (descriptive only)

GT AUC per (gap cell × member × score ∈ {`E_motion`, `q_motion`}) on all
three layers, next to the four M0 atoms recomputed on identical rows.
No gate; feeds no criterion except through A3.

### A6 — conditional safe region, S_old vs S_new

Anchor form: \(\max_D P_{FP}(D)\ \text{s.t.}\ \operatorname{UCB}[P_{GT}(D)]\le\epsilon\),
ε = 0.05, with the **identical cluster-aware UCB estimator and clustering
unit as the `gt_support_morphology_step0_20260711` packet**. Regions are
upper cuts `D = {score > τ}` per primary gap cell; τ grid = train-fold
pooled FP quantiles {50, 60, 70, 75, 80, 85, 90, 95}%. Threshold selection
uses **train-fold rows only**; evaluation is held-out (LOO), calibration
frozen per fold (feeds A7).

```text
S_old: score = bridge_dist (offline builder), same grid/UCB/rows
S_new: score = E_motion per member
no-thinner(member) := pooled held-out retained-FP(S_new) ≥ retained-FP(S_old)
                      on S_A, AND no qualifying fold with
                      retained-FP(S_new) < 0.8 × retained-FP(S_old)
```

A6 numeric τ values are offline-builder quantities; their mapping onto
consumer-A `bdist` (production 0.4 / gap-conditioned schedule) is governed
by the D0 ceiling of §3.

### A7 — LOO sequence transfer

Per qualifying fold (held-out GT ≥ 20 on the evaluated support), report
held-out A1 calibration metrics and pooled-primary AUC with fold-frozen
parameters and A6 thresholds.

```text
transfer(member) := pooled S_A AUC(−E_motion) > 0.5 in EVERY qualifying fold
```

Folds 04/09 are always reported, never criterion-bearing.

### A8 — M1 vs M2 attribution

All four members on all pairs×folds (E2 forbids winner-only output).
Report per-fold training-NLL selection winner vs held-out total NLL
(sum over held-out GT rows, S_A), matched `log det Σ(t)` growth curves,
and per-cell calibration classes.

```text
M2 ≻ M1 (dominance, frozen) :=
  retention(M2)                                   (A3)
  AND held-out total NLL(M2) < held-out total NLL(M1)   (S_A, GT)
  AND no primary cell where M2 is over-dispersed while M1 is
      approximately calibrated                     (matched uncertainty)
If multiple M2 members dominate: declared order H270 → H90 → H30.
```

## 5. Success boxes → analysis mapping

| Thread success box (all required for V1/V2) | Decided by |
|:--|:--|
| short-gap discrimination retained | A3 |
| long-gap GT tail concentration reduced | A4 (≥3/4, exploratory layer) |
| effect direction more consistent across gap bins | A2 (`R_flip = 0`) + A7 (every qualifying fold) |
| LOO conditional safe region no thinner | A6 |
| improvement not explained by unrestricted diffusion | A1 (winning member never over-dispersed in a primary cell) + A4 tie + A8 matched uncertainty |

## 6. Verdict decision rule (exactly one)

```text
V4  if > 50% of primary GT cells are LOW_SUPPORT, or the E3 lineage audit
    fails, or an E0-style identifiability blocker is documented.
    V4 is reachable ONLY through these predeclared routes — never as a
    post-hoc soft landing for a failed criterion.

boxes(member) := all five §5 rows hold for that member

V5  if boxes(member) fails for every member (and V4 route not triggered)
V1  if boxes(M1P-GLOBAL-CV) holds and no M2 member dominates M1 (A8)
V2  if boxes(M2 member) holds and that member dominates M1 (A8)

V3  is UNREACHABLE this round and is predeclared as such: the frozen table
    never identified velocity/joint observations (E0), so "joint velocity
    too noisy" cannot be concluded from a family that never fit a joint
    member. Absence of a joint fit is not evidence of joint noise.
```

The verdict is representation-level. Production-facing phrasing is
additionally capped by the D0 ceiling (§3) and the claim ladder
(substrate mapping §9); no level may be skipped.

## 7. Multiplicity and reporting discipline

- The verdict is driven **only** by the frozen criteria above on the
  primary cell set plus the two named exploratory boxes. Every other
  table/figure is descriptive and must be labeled so.
- All cells are reported (including LOW_SUPPORT and folds 04/09) — support
  exclusions remove criterion weight, never rows from the report.
- No per-sequence pick-best, no post-hoc rebinning, no member added or
  dropped, no held-out retune (E2 firewall unchanged).
- Every headline number carries its layer tag (`S_A` / secondary /
  exploratory) and member ID.

## 8. Deliverables

| Artifact | Content |
|:--|:--|
| E3 packet `evidence/gap_conditioned_motion_e3_signals_20260711/` | sealed pair×fold×model score table · 28 parameter + 7 selection artifacts · manifest (signals only) |
| Phase B packet `evidence/gap_conditioned_motion_phase_b_20260711/` | A1–A8 tables (12-table set from the thread) · figures · verdict record |
| Phase B research note | protocol reference (this doc) · results · limits · exactly one V1–V5 |
| Single reproduction entrypoint | rebuilds E3 fold artifacts and every A1–A8 table + verdict from the frozen pair table |
| Unit tests | criterion functions pure/deterministic; mutation checks that held-out contamination and tail/cell edits flip the recorded outputs |

## 9. Must not

Everything in the thread's Must-not list, plus:

- compute any §4 table inside the E3 signal-generation PR
- change a numeric criterion after E3 signals exist without a sealed
  design revision + deviation note
- cite consumer-A kernel quantities or claim threshold transfer without
  the D0 verdict (Issue #112)
- use exploratory-layer (all-gap) results to drive a deployment narrative
  (substrate mapping §8)
- declare V3, or reach V4 outside its predeclared routes
