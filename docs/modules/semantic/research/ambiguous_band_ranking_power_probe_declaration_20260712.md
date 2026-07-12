# Door 0 — ambiguous-band ranking-power probe (predeclaration)

<!-- doc-status: active -->
<!-- doc-promotion: predeclared study contract; seal = owner acceptance via PR merge -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

> **One-line:** mainline realignment step ③ — a cheap, read-only, early-stopping
> probe that decides whether the frozen signal family has **any usable,
> stable, interpretable ranking power** over GT vs FP candidates inside the
> gate-retained ambiguous band, before any score-layer interaction interface
> (step ④) is built. Every terminal is a mainline state transition (§20.7).

Thread: [ambiguous_band_ranking_power_probe_20260712.md](../../../research/threads/ambiguous_band_ranking_power_probe_20260712.md) ·
Contract: [framework §20](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) (v1, PR #133) ·
Substrate precondition: [production substrate mapping](production_substrate_mapping_20260711.md)

**Seal semantics:** this document is the study's §20.2 declaration. Merge of
the PR carrying it = research-owner seal. Execution (one run) is authorized
only after seal; results land in a follow-up PR with a committed evidence
packet. Any deviation from this declaration (atoms, thresholds, metrics,
boxes, terminals) voids the run and requires a new declaration.

**Revision 2 (2026-07-12, pre-seal — owner seal-review fixes; no outcome
information involved):** ① H/T3 headroom made joint (PWA **and** top-1) and
the false "arithmetically unavailable" rationale removed, with explicit
H-over-boxes precedence; ② tie policy for ranks and the quantile estimator
frozen; B3 clarified as metric-level re-aggregation with **no** threshold
re-fit; ③ B6 replaced by a mechanically decidable construction-invariant
check (fire-rate direction + flip decomposition); ④ T2 scoped to the Door-0
complexity class, and the reachable-slice limit made mandatory for **all**
terminals, not only T1.

**Revision 3 (2026-07-12, pre-seal — round-2 owner review, two scope
closures; no outcome information involved):** ① the Door-0 class is now
**exactly the 12 enumerated §6 candidates** (frozen q85/q15 + λ→∞
lexicographic transform) — not "up to second-order AND", which would have
exhausted the 9 untested AND pairs; T2 closes only these 12 members; ② the
reachable-slice caveat is now **unconditional** — every recorded terminal
(T0–T3) carries the §3 scope clause verbatim, with no directional trigger to
evaluate, so no implementer choice can decide its presence.

**Revision 4 (2026-07-12, pre-seal — final review, one clause fix):** the
universal caveat's opening ("established on the gate-retained band")
contradicted T0, which establishes nothing; the clause is now neutral
("study scope is the gate-retained band; this terminal establishes no claim
inside the production-reachable set …") and holds identically for T0–T3.
No terminal, box, candidate, or protocol change.

---

## 1. §20.2 declaration block

```text
Target decision layer   score-ranking
Study intent            capability map (primary); no secondary intents
Design objective        n/a (not a design evaluation). Role-legal diagnostic
                        instrument per §20.3: establish presence/absence of
                        stable, interpretable improvement of the GT-vs-FP
                        relative ordering inside the retained ambiguous band,
                        measured by event-local ranking metrics, at the
                        minimum effects predeclared in §7.
Selection rule          none over candidates — all 12 predeclared candidates
                        are reported in full (no cherry-pick); the terminal is
                        decided by the §7 boxes, not by best-performer choice.
Validity gate           §8 (V1–V5), separating UNRESOLVED from futility.
Stop condition          §9 — sufficiency: ≥1 candidate passes all boxes;
                        futility: none passes; no third door (§20.7).
Output class            diagnostic result (capability map); on futility the
                        band is additionally recorded as an unexplained
                        residual set with respect to the 12-member Door-0
                        tested class (§10). No design candidate may be
                        claimed; §20.5 applies to every number in the
                        results.
Mainline transition     §10 — T1 opens step ④; T2 closes the 12-member
                        tested class (step ⑤, class-scoped); T3 closes
                        step ④ as unnecessary at Door-0 resolution;
                        T0 = UNRESOLVED/INVALID-STUDY closes the experiment
                        only. Every terminal carries the §3 unconditional
                        reachable-set scope caveat.
```

## 2. Substrate (frozen)

- Pairs table: `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv`,
  SHA256 `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17`
  (identical to the E0-frozen builder output; 7-seq MOT17 SDP,
  `mamba_whole_graph_m` B1 offline).
- Pool: `gt_valid == 1` rows only (21,789 rows; 340 GT / 21,449 FP),
  loaded via `audit_relink_safe_reject.load_gt_valid_pool`.
- Baseline score: `s0 = score_m_bridge` **offline production proxy**
  (`ensure_prod_proxy_scores`):
  `w = sqrt(clip(lost_exit_speed/0.12, 0, 1))`,
  `s0 = w·½(fwd_resid+bwd_resid) + (1−w)·dist_h` — the offline counterpart of
  the consumer-A `bdist` aggregate (estimator-shifted; see substrate mapping
  §2.2; no deployment-facing numeric claim is made from it).
- Missing values: rows with NaN in any declared source column are dropped
  before event construction; if that removes > 5 % of band rows the study is
  invalid (V5). Counts-only recon observed 0 such rows.

## 3. Coarse gate and pools

**Coarse gate (predeclared, production-native):** retain rows with
`h_lost_raw/h_cand_raw ∈ [0.60, 1.70]` — the offline proxy of the consumer-A
EMA scale pre-gate, the only production hard pre-gate with a faithful offline
proxy on this table. `score_m_bridge ≤ 0.4` is the production **decision
boundary**, not a coarse gate, and is deliberately **not** applied to the
pool.

Pools (per owner Door-0 spec):

- **P1** — all gate-retained band rows, event-agnostic. Descriptive
  diagnostics only (per-atom pooled separability); no terminal force.
- **P2** — **primary**: within-event GT–FP pairs (event definition §4).
- **P3** — hard subset: rankable events whose baseline `s0` ranking does not
  place a GT row at rank 1 (top-1 miss). Reported per candidate
  (ΔPWA, Δtop-1 restricted to P3); localizes headroom; no separate boxes.
- **Reachable slice (descriptive only):** P2 restricted to rows with
  `s0 ≤ 0.4` (production-reachable candidates). Counts-only recon: 34
  rankable events, median size 2 — predeclared as **too thin for terminal
  force**. **Unconditional scope caveat (all terminals, no trigger
  condition):** the results PR must report the slice for the baseline and
  every candidate, and the recorded terminal statement — whichever of T0–T3
  — **always** carries the following clause verbatim, with no directional
  test deciding its presence: *"study scope is the gate-retained band; this
  terminal establishes no claim inside the production-reachable set
  (s0 ≤ 0.4; 34 events, descriptive only); any step-④ decision must treat
  the decision surface (threshold/margin interplay), not assume in-place
  reranking behavior."*

## 4. Event and trial semantics

- **Decision event** = `(seq, cand_id)`: one newborn candidate choosing among
  live lost tracks — the consumer-A decision shape (newborn fires once at
  `hit_streak == 4`).
- **Rankable event** = ≥ 1 GT row and ≥ 1 FP row after the gate.
- Events with multiple GT rows: all GT rows enter pairwise counts; MRR and
  top-1 use the best-ranked GT row.
- **Ranking direction:** lower score = better; candidates within an event are
  sorted ascending.
- **Clustering declaration (Gate B posture):** events share lost tracks
  across events and share scene/pipeline state within a sequence; the
  independence unit is **not** resolved below the sequence level. Therefore
  **no Clopper–Pearson or other formal confidence bound is claimed and none
  is used to cross any §7 boundary**; the terminals rest on predeclared
  effect sizes, per-sequence direction consistency, concentration and LOO
  retention. Output claim ceiling: L1 in-sample + LOO-retention diagnostic.

## 5. Metrics (event-local, all predeclared)

For a ranking r of an event's candidates:

**Ranking keys (frozen):** baseline key = `s0` (scalar); candidate key =
the lexicographic tuple `(c, s0)` (§6). Two rows are **tied** iff their keys
are exactly equal under the applicable key.

**Tie policy (frozen):**

- **PWA pairs:** GT strictly better = 1; tied = 0.5; worse = 0.
- **Ranks (MRR / top-1):** pessimistic against GT —
  `rank(GT row) = 1 + #{FP rows with strictly better key} + #{FP rows with
  tied key}`. A GT row therefore holds rank 1 only if it is strictly better
  than every FP row in the event. The same formula applies to baseline and
  candidate keys; no random or index-order tie-breaking anywhere.

Metrics:

- **PWA** (pairwise ranking accuracy): over all (GT, FP) pairs in the event,
  scored per the tie policy. Event PWA = mean over its pairs; **pooled PWA =
  macro-average over rankable events** (equal event weight). Per-seq PWA =
  macro over the sequence's rankable events.
- **MRR**: reciprocal of the best (smallest) GT rank under the pessimistic
  rank formula, macro-averaged over rankable events.
- **top-1**: fraction of rankable events whose best GT rank = 1 (i.e. a GT
  row strictly beats all FP rows).
- **margin** (descriptive): `s0(best FP) − s0(best GT)` per event; reported,
  no box.

Baseline metrics use the baseline key. Candidate metrics use the candidate
key. Δmetric = candidate − baseline on the identical event set.

## 6. Probe candidate family (frozen; 12 candidates)

Atom sources (builder columns / derivations already fixed by the frozen
family): `dist_h`; `log_h_ratio = |log(h_cand_raw/h_lost_raw)|`;
`resid_mean = ½(fwd_resid+bwd_resid)`;
`speed_mismatch = |lost_exit_speed − cand_entry_speed|`; `dir_cos`; `gap`.

**Binarization (fixed, no search):** condition fires on the unsafe tail —
`c_a = 1[v_a ≥ q85(v_a | band rows)]` for lower-is-better atoms;
`c_dir = 1[dir_cos ≤ q15(dir_cos | band rows)]`. Quantiles are computed on
gate-retained band rows (GT+FP) of the fitting fold set (§6.1) with the
**frozen estimator `numpy.quantile(values, q, method="linear")`** on
float64 values (integer-valued atoms such as `gap` are cast to float64 and
use the same linear interpolation; no alternative `method` may be used).
Condition comparisons use `≥` / `≤` exactly as written. No other quantiles,
thresholds, atoms, or operators may be evaluated under this seal; OR-of-AND
and any third-order form are forbidden.

**Singles (6):** `c_dist_h`, `c_log_h_ratio`, `c_resid_mean`,
`c_speed_mismatch`, `c_gap`, `c_dir_cos_low`.

**Second-order ANDs (6):**
`c_dist_h ∧ c_gap` · `c_dist_h ∧ c_speed_mismatch` ·
`c_gap ∧ c_speed_mismatch` · `c_gap ∧ c_dir_cos_low` ·
`c_dist_h ∧ c_log_h_ratio` · `c_resid_mean ∧ c_gap`.

**Scoring transform (parameter-free):** lexicographic demotion — sort key
`(c, s0)` ascending: all `c = 0` candidates rank ahead of all `c = 1`
candidates; `s0` orders within each group. This is the λ→∞ limit of
`s = s0 + λ·c` and introduces **no tunable weight**; Door 0 deliberately does
not fit λ.

**A-priori mechanism direction (construction invariant, feeds B6):** every
candidate demotes geometrically/temporally strained pairs — larger
height-normalized displacement (`dist_h`), larger height inconsistency
(`log_h_ratio`), larger midpoint residual (`resid_mean`), larger
entry/exit speed mismatch (`speed_mismatch`), longer occlusion (`gap`),
reversed direction (`dir_cos` low). Under lexicographic demotion with this
fixed direction, no reversed-direction variant exists in the run, and B6 is
decided **mechanically** by the two §7 decomposition inequalities — no
post-hoc reviewer judgment of "mechanism consistency" may enter the
terminal. Known prior risk (recorded, not disqualifying for a diagnostic
probe): `gap`-family conditions were sequence-specific in the ε=0 repair
line; B3/B5 are the guards.

### 6.1 Threshold fitting protocol

- **In-sample:** quantiles fit on all 7-seq band rows.
- **LOO:** for each held-out sequence, quantiles fit on the other 6
  sequences' band rows; all metrics evaluated on the held-out sequence's
  rankable events; LOO-pooled = macro over all events, each under its fold's
  thresholds.

## 7. Headroom check and minimum-effect boxes

**H (headroom, evaluated first; joint condition):** terminal = T3 iff, on
the in-sample baseline (key = `s0`, pessimistic ranks):

```text
H1  baseline pooled PWA ≥ 0.98
H2  baseline pooled top-1 ≥ 0.98
    (equivalently: P3 top-1-miss events ≤ 2 % of rankable events)
```

Both must hold. Rationale: high PWA alone does not establish that the
decision-relevant ordering is solved — an event can carry near-perfect PWA
while a single FP still outranks the GT (top-1 miss); T3's "step ④
unnecessary" claim therefore requires the pairwise **and** the top-1 ceiling
jointly. **Precedence (predeclared):** if H holds, the 12 candidates are
still computed and reported descriptively, but the terminal is T3 regardless
of box outcomes — within ≤ 2 % residual headroom, a box pass is
indistinguishable from selection noise at this probe's resolution, and
Door 0 does not adjudicate it.

A candidate **passes** iff all of:

```text
B1  in-sample pooled ΔPWA ≥ +0.02
B2  per-seq in-sample ΔPWA ≥ 0 in ≥ 5/7 seqs AND > 0 in ≥ 4/7
B3  concentration: min over s of pooled ΔPWA re-aggregated over all
    rankable events excluding sequence s ≥ +0.01. Metric-level
    re-aggregation ONLY — the in-sample thresholds (fit on all 7 seqs)
    are held fixed; no threshold re-fit occurs in B3. Threshold-refit
    robustness is exclusively B5's axis.
B4  in-sample pooled ΔMRR ≥ 0 AND Δtop-1 ≥ 0
B5  LOO: LOO-pooled ΔPWA ≥ +0.01 AND per-fold ΔPWA ≥ 0 in ≥ 5/7 folds
    (thresholds re-fit per fold per §6.1)
B6  mechanical direction decomposition (in-sample, pooled over rankable
    events; no judgment call):
      B6a  fire-rate direction: P(c=1 | FP rows) > P(c=1 | GT rows),
           computed on rankable-event rows
      B6b  flip decomposition: n_good > n_bad, where over all in-event
           (GT, FP) pairs, n_good = pairs whose PWA contribution
           increased under the candidate key vs the baseline key, and
           n_bad = pairs whose contribution decreased
    Both inequalities strict. B6 is a construction-invariant check —
    the demotion direction is fixed a priori (§6) and no reversed
    variant exists in the run.
```

**Multiplicity guard:** 12 candidates are evaluated; no p-values are claimed
(§4), so the guards against selection noise are structural — B2/B3/B5 must
all hold on the same candidate, and all 12 results are reported in full.

## 8. Validity gate (V — separates UNRESOLVED from futility)

```text
V1  pairs.csv SHA256 matches §2 exactly
V2  pooled rankable events after gate ≥ 150
V3  ≥ 10 rankable events in ≥ 6/7 sequences
V4  gate retains ≥ 90 % of GT rows
V5  NaN-dropped rows ≤ 5 % of band rows
```

Any V failure ⇒ terminal T0. Disclosure per §19.6-analog: V2–V5 were
calibrated on a counts-only recon of this table (observed: 205 rankable
events; per-seq min 12; GT retention 329/340 = 96.8 %; 0 NaN rows). The
recon computed **no ranking metric, no label-vs-score ordering**; outcome
information did not feed back into this declaration.

## 9. Stop conditions

- **Sufficiency:** ≥ 1 candidate passes B1–B6 → stop, report all passers and
  the full 12-candidate capability map.
- **Futility (mandatory):** no candidate passes → stop. No widening of the
  family, no threshold re-search, no "describe more and continue" (§20.7).
- §20.6 futility symptoms (boundary-hugging gains, single-sequence tails,
  LOO non-retention) are covered by B3/B5 and must be named in the results if
  they blocked a candidate.

## 10. Terminals → mainline transitions (§20.7)

The **Door-0 tested class** is: **exactly the 12 enumerated candidates of
§6** (the 6 singles and the 6 listed ANDs), with the frozen q85/q15
thresholds (§6 estimator) and the λ→∞ lexicographic transform. It is **not**
"all second-order ANDs over the atom family" — 9 of the 15 possible pairs
are untested and are **not** exhausted by any terminal below. No terminal
claim extends beyond these 12 members; none extends to untested AND pairs,
other quantiles, continuous signals, finite-λ weightings, learned scores, or
other transform families.

```text
T0  UNRESOLVED / INVALID-STUDY (validity failure)
    → closes this experiment only; hypothesis path stays open;
      not reportable as any form of exhaustion.
T1  RANKING_SIGNAL_PRESENT (H not triggered; ≥1 candidate passes B1–B6)
    → transition: authorizes step ④ — a design evaluation declared at
      target layer = score-ranking (interaction interface). T1 itself
      remains a diagnostic result; no design candidate exists until
      step ④ passes §20.4 selection.
T2  NO_USABLE_RANKING_POWER_IN_CLASS
    (validity pass, H not triggered, no passer)
    → transition: closes the core unknown FOR THE 12-MEMBER DOOR-0
      TESTED CLASS on this substrate (step ⑤, class-scoped); the band
      is recorded as an unexplained residual set WITH RESPECT TO THESE
      12 MEMBERS; re-running any of the 12 on this family/substrate is
      blocked. T2 does NOT establish anything about the 9 untested AND
      pairs, other quantiles, continuous signals, or finite weightings;
      probing any wider or different set is not an auto-continuation
      (§20.7 no-third-door applies to this study's path) — it requires
      a new §20.2 declaration and an explicit owner charter decision.
T3  NO_HEADROOM (H1 ∧ H2 triggered)
    → transition: closes step ④ as unnecessary AT DOOR-0 RESOLUTION
      (+2 pp / 2 %) on this pool — the band is already ordered by
      baseline geometry to within the probe's resolution; residual IDs
      pain must be attributed outside this pool or below this
      resolution.
```

Exactly one terminal must be recorded. **The §3 unconditional reachable-set
scope caveat attaches to every recorded terminal** — always, with no
directional trigger to evaluate.

## 11. Execution and evidence order

```text
this PR merge = declaration seal
→ single execution run (read-only; no preset/hook/production change)
→ results PR: committed evidence packet (runner script + all-12 table +
  per-seq/LOO/P3/reachable-slice tables + SHA manifest) + terminal
→ owner research acceptance
```

The runner script is written **after** seal, against this declaration
verbatim; it must recompute V1 and refuse to run on SHA mismatch.

## 12. Must not

- 不得改 preset / evidence_ledger / production hook / closed gates；
- 不得在本 seal 下新增 atom、換 quantile、fit λ、或引入 OR-of-AND；
- 不得以 P1 pooled 分離度或 reachable slice 取代 P2 boxes 判 terminal；
- 不得對任何 §7 界線使用未解決 clustering 下的 CP bound（§19.5 類推）；
- 不得把 T1 重貼標籤為 design candidate（§20.5）；
- 不得在 futility 後擴家族續跑（§20.7 無第三扇門）；
- 不得把 T2 解讀為超出 **12-member tested class** 的耗盡宣稱（9 個未測 AND pair／其他 quantile／連續訊號／有限 λ 權重／learned score 均不在本 probe 測試範圍；任何擴集＝新 §20.2 宣告＋owner charter 決定）；
- 不得省略或條件化 reachable-set scope caveat（§3 無條件規則：每個 recorded terminal 一律附 verbatim clause，無方向判斷）。
