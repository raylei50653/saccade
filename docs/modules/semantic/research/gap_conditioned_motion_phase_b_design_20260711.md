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

Thread: [gap-conditioned probabilistic motion probe](../../../research/threads/closed/gap_conditioned_probabilistic_motion_probe_20260711.md)

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

## 3. D0 coupling (parallel gate; scope-limited to bridge atoms)

A1–A8 consume **offline-builder atoms** and the E2 observation
\(d=\Delta\text{foot}/h_{ref}\) only; they never cite consumer-A kernel
quantities and do **not** wait on D0 execution.

D0 (Issue #112) certifies estimator fidelity for the **bridge atoms
only**: `bdist`/`score_m_bridge`, `dist_h`, `fwd_r`/`bwd_r`.
\(E_{motion}\) is classified **research-only** for consumer A (substrate
mapping §7) — it has no kernel counterpart — so **no D0 outcome can
upgrade an \(E_{motion}\) statement**:

| D0 verdict | Ceiling it sets |
|:--|:--|
| `threshold_transfer_supported` | numeric statements about the **`bridge_dist` baseline (A6 `S_old`)** may be phrased as consumer-A `bdist` schedule candidates (claim level 2) |
| `rank_only_transfer_supported` | rank/shape statements about `bridge_dist` only; numeric thresholds do not port |
| `not_fidelity_aligned` (or D0 not yet run) | signal-level regularity only (claim level 1) |

\(S_{new}=\{E_{motion}>\tau\}\) and every A6 / verdict statement about it
remain **representation-level regardless of the D0 verdict**. Upgrading
them to a level-2 predicate candidate requires a separate, future
contract: a named production consumer that actually computes the
reconstructable observation \(d\) (substrate mapping §7), plus its own
reconstruction-fidelity gate. That contract is out of scope for Phase B.

The V1–V5 verdict itself is a **representation** verdict and is
independent of D0. Any follow-up hook proposal inherits this verdict, the
D0 ceiling, and the missing \(E_{motion}\) consumer contract separately.

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

Because the M0 baseline is already 0/20, A2 can only establish **absence
of regression** — it cannot establish "more consistent than M0", and no
relative-consistency claim may appear in the verdict record. The
corresponding success box is accordingly named "no new aggregate reversal
+ positive held-out direction" (§5), with the direction half supplied by
A7.

### A3 — short-gap retention

On the two primary cells, per member, recompute `bridge_dist` AUC on the
identical row set (no reuse of E1's all-support numbers) and compare:

```text
retention(member) := AUC(−E_motion) ≥ AUC(−bridge_dist) − 0.05
                     in BOTH primary cells (non-LOW_SUPPORT)
```

`q_motion` AUC is reported alongside; retention is judged on `E_motion`.

### A4 — escape-tail audit

Cohort = **exactly four pairs**, one per sealed far-Hamming GT lost track
(all MOT17-10-SDP): for each track, the min-\(d_H\) representative — the
GT-matched pair with the **minimal** Hamming distance \(d_H\) recorded in
the escape-tail forensic packet (the track's best-case pair; the cohort is
"escape" precisely because even this pair is far);
ties broken by smaller gap, then smaller `cand_id`. The runner must
resolve the four exact `[seq, lost_id, cand_id]` keys from the sealed
packet and record them (with the packet hash) in the Phase B packet
**before** any pair is scored. All four are scored under the genuinely
held-out fold-10 parameters.

Reference population (single, frozen): the within-native-gap-bin **pooled
(GT + FP) q90 tail** of `E_motion`, ties inclusive — the E1 tail
definition. No other population defines the criterion.

```text
not-high-energy(pair, member) := E_motion below the within-bin pooled
                                 q90 tail
escape-cohort box             := not-high-energy for ≥ 3 of 4 pairs
anti-over-diffusion tie       := no native cohort bin classified
                                 over-dispersed (A1) for that member
```

This box is deliberately **weakened**: it claims only that the escape
cohort is not high-energy under `E_motion`. It is **not** a "tail
reduction relative to M0" claim — E1 already showed the cohort need not
occupy any single M0 score's pooled q90 tail, so an improvement-over-M0
statement is not identifiable here. The within-bin FP-referenced
percentile and the same pairs' `bridge_dist` / `resid_mean` pooled-q90
membership are reported as descriptive diagnostics only.

Cohort gaps are long-gap → this analysis lives on the **exploratory**
layer (substrate mapping §8): it may support the representation verdict
but never a deployment narrative.

### A5 — separability (descriptive only)

GT AUC per (gap cell × member × score ∈ {`E_motion`, `q_motion`}) on all
three layers, next to the four M0 atoms recomputed on identical rows.
No gate; feeds no criterion except through A3.

### A6 — conditional safe region, S_old vs S_new

Anchor form: \(\max_D P_{FP}(D)\ \text{s.t.}\ \operatorname{UCB}[P_{GT}(D)]\le\epsilon\),
ε = 0.05. **The bound is frozen here, not inherited**: the morphology
step-0 packet's own terminal states that a cluster-aware bound was *not*
established (its track-level CP was a nominal diagnostic only), so there
is no existing estimator to reuse.

Frozen A6 estimator:

```text
bound        one-sided exact binomial (Clopper–Pearson) upper confidence
             bound at 95% (one-sided α = 0.05) on cluster-level GT
             containment. This is a MODEL-BASED CP bound valid under the
             track-cluster independence assumption below; it must NOT be
             described as a cluster-robust 95% population bound
cluster unit (sequence, lost_id); a cluster is "contained" if ANY of its
             GT pairs falls in D. Independence is ASSUMED across clusters
             within a fold; sequence-level residual clustering remains a
             DECLARED LIMITATION (undischarged since morphology step-0) —
             per-fold tables are the primary reading, pooled numbers are
             supplements
region       upper cut D = {score > τ} per primary gap cell
grid         τ = train-fold pooled FP quantiles {50,60,70,75,80,85,90,95}%
selection    per fold × cell, τ = grid value maximizing training FP_removed
             subject to the CP bound ≤ ε on TRAINING clusters only;
             held-out rows contribute nothing to selection (LOO firewall)
metric       FP_removed := P_FP(D), the fraction of the cell's FP rows
             captured by D (rename of "retained-FP"; same object)
```

Terminal semantics and non-vacuity (frozen):

```text
NO_FEASIBLE_THRESHOLD  cell×score where no grid τ satisfies the bound
                       on training clusters
S_new infeasible where S_old feasible  → no-thinner FAILS for that member
both infeasible in a cell              → cell recorded BOTH_EMPTY;
                                         excluded from the criterion and
                                         can NEVER support a pass (no
                                         vacuous 0 ≥ 0)

held-out safety precondition (evaluated BEFORE any FP_removed comparison):
  for EVERY qualifying fold × non-BOTH_EMPTY primary cell, the held-out
  cluster-level empirical GT leakage of S_new must be ≤ ε = 0.05
  (leakage = fraction of held-out (sequence, lost_id) clusters contained
  in D — same unit as the bound); any violation → no-thinner FAILS for
  that member. S_old held-out leakage is reported alongside as a
  descriptive diagnostic (the baseline is not the object under test).

no-thinner(member) := held-out safety precondition holds,
                      AND pooled held-out FP_removed(S_new) ≥ FP_removed(S_old)
                      on S_A, AND pooled held-out FP_removed(S_new) > 0,
                      AND no qualifying fold with
                      FP_removed(S_new) < 0.8 × FP_removed(S_old)
```

`S_old` uses `bridge_dist` (offline builder); `S_new` uses `E_motion` per
member; identical grid, bound, and rows. Held-out evaluation reports
realized FP_removed and GT leakage per fold (feeds A7).

Claim ceilings differ by side (§3): `S_old` τ values map onto consumer-A
`bdist` (production 0.4 / gap-conditioned schedule) only under the D0
ceiling; `S_new` τ values are representation-level regardless of D0.

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

| Success box (all required for V1/V2) | Decided by |
|:--|:--|
| short-gap discrimination retained | A3 |
| escape cohort not high-energy under `E_motion` (weakened; no M0-relative reduction claim) | A4 (≥3/4, exploratory layer) |
| no new aggregate reversal + positive held-out direction (weakened; not "more consistent than M0") | A2 (`R_flip = 0`) + A7 (every qualifying fold) |
| LOO conditional safe region no thinner (held-out GT-leakage ≤ ε precondition; non-vacuous `FP_removed`) | A6 |
| improvement not explained by unrestricted diffusion | A1 (winning member never over-dispersed in a primary cell) + A4 tie + A8 matched uncertainty |

The second and third boxes are deliberately weaker than the thread's
original mother-line phrasing ("tail concentration reduced" / "more
consistent"); the thread's box list is updated to match in the same PR.
The stronger readings are not identifiable on this substrate (A2/A4
rationale above) and must not be claimed.

## 6. Verdict decision rule (exactly one)

```text
boxes(member)  := all five §5 rows hold for that member
dominates(M2)  := the A8 dominance rule over M1P-GLOBAL-CV

V4 routes (exhaustive): > 50% of primary GT cells are LOW_SUPPORT, or the
E3 lineage audit fails, or an E0-style identifiability blocker is
documented. V4 is reachable ONLY through these routes — never as a
post-hoc soft landing for a failed criterion.

Priority partition (evaluate top-down; exactly one fires):

  if any V4 route triggered:                            V4
  elif exists M2 with boxes(M2) AND dominates(M2):      V2
       (multiple qualifying M2: declared order H270 → H90 → H30)
  elif boxes(M1P-GLOBAL-CV):                            V1
  else:                                                 V5

V5 semantics (redefined; the thread's verdict table is updated to match
in this PR): V5 = "representation + attribution contract NOT
ESTABLISHED" — no member both passes all success boxes AND holds a
claimable verdict slot. V5 does NOT assert "no real fix / only
over-diffusion"; that stronger reading was the pre-partition wording and
is retired. The residual case — some M2 passes boxes but fails dominance
while M1 fails boxes — falls INSIDE this definition by construction and
must still carry an explicit anomaly note ("a member passed all success
boxes without a claimable verdict slot") in the verdict record.

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
- cite consumer-A kernel quantities or claim `bridge_dist` threshold
  transfer without the D0 verdict (Issue #112)
- phrase \(S_{new}\)/\(E_{motion}\) as a consumer-A predicate candidate
  under **any** D0 outcome — no consumer-A counterpart exists (§3);
  level 2 requires a future \(E_{motion}\)/\(d\) consumer contract
- let an empty region pass A6 vacuously (`BOTH_EMPTY` cells never
  support a pass)
- use exploratory-layer (all-gap) results to drive a deployment narrative
  (substrate mapping §8)
- declare V3, or reach V4 outside its predeclared routes
