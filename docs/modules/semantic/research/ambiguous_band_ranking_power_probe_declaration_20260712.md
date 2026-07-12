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
                        residual set. No design candidate may be claimed;
                        §20.5 applies to every number in the results.
Mainline transition     §10 — T1 opens step ④; T2 closes the signal-family
                        ranking path (step ⑤); T3 closes step ④ as
                        unnecessary; T0 = UNRESOLVED/INVALID-STUDY closes the
                        experiment only.
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
  force**. Interpretation rule: if T1 passes on P2 while this slice shows
  pooled ΔPWA ≤ 0, the T1 statement must carry the limit *"ranking power not
  demonstrated inside the production-reachable set; step ④ must treat the
  decision surface (threshold/margin interplay), not assume in-place
  reranking gain."*

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

- **PWA** (pairwise ranking accuracy): over all (GT, FP) pairs in the event,
  fraction where the GT row ranks strictly better; ties count 0.5.
  Event PWA = mean over its pairs; **pooled PWA = macro-average over
  rankable events** (equal event weight). Per-seq PWA = macro over the
  sequence's rankable events.
- **MRR**: reciprocal of the best GT rank, macro-averaged.
- **top-1**: fraction of rankable events with a GT row at rank 1.
- **margin** (descriptive): `s0(best FP) − s0(best GT)` per event; reported,
  no box.

Baseline metrics use `s0` alone. Candidate metrics use the §6 transform.
Δmetric = candidate − baseline on the identical event set.

## 6. Probe candidate family (frozen; 12 candidates)

Atom sources (builder columns / derivations already fixed by the frozen
family): `dist_h`; `log_h_ratio = |log(h_cand_raw/h_lost_raw)|`;
`resid_mean = ½(fwd_resid+bwd_resid)`;
`speed_mismatch = |lost_exit_speed − cand_entry_speed|`; `dir_cos`; `gap`.

**Binarization (fixed, no search):** condition fires on the unsafe tail —
`c_a = 1[v_a ≥ q85(v_a | band rows)]` for lower-is-better atoms;
`c_dir = 1[dir_cos ≤ q15(dir_cos | band rows)]`. Quantiles are computed on
gate-retained band rows (GT+FP) of the fitting fold set (§6.1). No other
quantiles, thresholds, atoms, or operators may be evaluated under this seal;
OR-of-AND and any third-order form are forbidden.

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

**A-priori mechanism direction (B6, written before execution):** every
candidate demotes geometrically/temporally strained pairs — larger
height-normalized displacement (`dist_h`), larger height inconsistency
(`log_h_ratio`), larger midpoint residual (`resid_mean`), larger
entry/exit speed mismatch (`speed_mismatch`), longer occlusion (`gap`),
reversed direction (`dir_cos` low). A candidate whose observed improvement
depends on the opposite direction fails B6 regardless of magnitude. Known
prior risk (recorded, not disqualifying for a diagnostic probe): `gap`-family
conditions were sequence-specific in the ε=0 repair line; B3/B5 are the
guards.

### 6.1 Threshold fitting protocol

- **In-sample:** quantiles fit on all 7-seq band rows.
- **LOO:** for each held-out sequence, quantiles fit on the other 6
  sequences' band rows; all metrics evaluated on the held-out sequence's
  rankable events; LOO-pooled = macro over all events, each under its fold's
  thresholds.

## 7. Headroom check and minimum-effect boxes

**H (headroom, evaluated first):** if baseline pooled PWA ≥ 0.98 in-sample,
terminal = T3 (no candidate evaluation can override; +2 pp is arithmetically
unavailable).

A candidate **passes** iff all of:

```text
B1  in-sample pooled ΔPWA ≥ +0.02
B2  per-seq in-sample ΔPWA ≥ 0 in ≥ 5/7 seqs AND > 0 in ≥ 4/7
B3  concentration: min over s of pooled ΔPWA recomputed excluding
    sequence s ≥ +0.01  (leave-one-seq-out of the improvement)
B4  in-sample pooled ΔMRR ≥ 0 AND Δtop-1 ≥ 0
B5  LOO: LOO-pooled ΔPWA ≥ +0.01 AND per-fold ΔPWA ≥ 0 in ≥ 5/7 folds
B6  observed improvement direction consistent with the §6 a-priori
    mechanism note (no post-hoc flip)
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

```text
T0  UNRESOLVED / INVALID-STUDY (validity failure)
    → closes this experiment only; hypothesis path stays open;
      not reportable as signal-family exhaustion.
T1  RANKING_SIGNAL_PRESENT (≥1 candidate passes B1–B6)
    → transition: authorizes step ④ — a design evaluation declared at
      target layer = score-ranking (interaction interface); carries the
      §3 reachable-slice limit if triggered. T1 itself remains a
      diagnostic result; no design candidate exists until step ④ passes
      §20.4 selection.
T2  NO_USABLE_RANKING_POWER (validity pass, H not triggered, no passer)
    → transition: closes the core unknown — the current frozen signal
      family has no usable ranking power in the retained ambiguous band
      (step ⑤); the band is recorded as an unexplained residual set;
      further Boolean ranking studies on this family/substrate are
      blocked pending genuinely new signals.
T3  NO_HEADROOM (H triggered)
    → transition: closes step ④ as unnecessary on this substrate — the
      band is already ordered by baseline geometry; residual IDs pain
      must be attributed outside this pool.
```

Exactly one terminal must be recorded.

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
- 不得在 futility 後擴家族續跑（§20.7 無第三扇門）。
