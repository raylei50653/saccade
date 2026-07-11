---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# Escape-Tail Forensic — Four-Track PR-C (issue #102)

> **One-line:** Sealed Step-0 far-Hamming descriptive tail（4 tracks，`k=8`，`d_H >= 3`，**4/4 MOT17-10-SDP**）逐件落入預宣告五類別：**3× `TRUE_LONG_GAP_REENTRY` + 1× `UNRESOLVED`**；聚合終端 **`ROLE_REVERSAL_SUPPORTED`**。授權後續 partial-order audit 將 motion atoms 視為 `conditional_orderable` / `context_only` 候選；**不**授權全域 closure arcs、tail veto、production / preset / ledger 變更。

Thread: [gt_support_morphology_20260711.md](../../../research/threads/gt_support_morphology_20260711.md) ·
Procedure: [framework §19](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) ·
Research line: [boolean_closure_domain_line_20260711.md](boolean_closure_domain_line_20260711.md) ·
Step-0: [gt_support_morphology_step0_20260711.md](gt_support_morphology_step0_20260711.md) ·
Packet: [evidence/escape_tail_forensic_20260711/](evidence/escape_tail_forensic_20260711/manifest.json)

## 0. Scope and claim ceiling

- **Read-only / offline / evidence-only**（issue #102 scope guards）。
- Frozen cohort 僅來自 sealed Step-0 `tail_tracks.json`；檢視結果後不增不刪 tracks。
- 二值化 = Step-0 已宣告的 pool-median（audit-only）；本 forensic **不**重設 sealed thresholds，只做敏感性診斷。
- Claim ceiling = **L1 forensic** on a single sequence cluster。Nested held-out（PR-E）仍是唯一可超 L1 的 confirmatory unit。
- Explicitly **not** done: gate-rule search, MWC, min-cut, closure arcs, orderability promotion into a sealed partial order, production/preset/ledger changes.

## 1. Frozen cohort

| track_key | sealed min `d_H` | n `gt_match` rows | primary GT id |
|:--|--:|--:|--:|
| `MOT17-10-SDP\|455` | 4 | 1 | 41 |
| `MOT17-10-SDP\|459` | 3 | 1 | 29 |
| `MOT17-10-SDP\|467` | 3 | 7 | 71 |
| `MOT17-10-SDP\|503` | 6 | 2 | 59 |

Provenance: Step-0 packet SHA seal on `pairs.csv` =
`0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17`；runner 拒絕 SHA 不符的來源。

## 2. Method (predeclared)

Runner: [`run_escape_tail_forensic.py`](evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py) ·
Rules: [`classification_rules.json`](evidence/escape_tail_forensic_20260711/classification_rules.json)

Per track, evidence always includes:

1. sequence / lost-track identity + source ledger rows;
2. disappearance / gap / re-entry timeline;
3. box / height / scale at exit and re-entry (pairs + MOT17 GT);
4. values and threshold sides for all 8 atoms, with motion / height / geometry decomposition;
5. annotation continuity on the primary GT id;
6. signal recomputation from raw pair columns;
7. threshold-sensitivity (near-median flips + p40/median/p60 membership);
8. gap-window GT visibility (occlusion_strong criterion);
9. exactly one of the five predeclared categories.

**Occlusion_strong (fixed):** gap-window `vis_mean ≤ 0.35` **or** `frac(vis=0) ≥ 0.25`.

**TRUE_LONG_GAP_REENTRY (fixed):** annotation OK ∧ signal OK ∧ not threshold-dominated ∧ height safe on min-`d_H` row ∧ ≥2 motion violations ∧ occlusion_strong.

**Aggregate `ROLE_REVERSAL_SUPPORTED` (fixed):** ≥3 TRUE and zero annotation/signal/threshold artifact categories.

Reproduce:

```bash
uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify
```

`--verify` rebuilds in a temp dir and compares the committed packet.

## 3. Per-track cards

### 3.1 `MOT17-10-SDP|455` → `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | lost last f=114 → cand 468 first f=171 · gap=57 · GT id 41 |
| Height | h_raw 60.35 → 60.75 · `log_h_ratio=0.0066` **SAFE** (margin ≫ thr) |
| Motion / geom on min-`d_H` | **VIOL** `speed_mismatch`, `resid_mean`, `bridge_dist`, `dist_h` · `dir_cos=0.96` SAFE |
| `d_H` | 4 |
| Gap visibility | mean **0.19** · frac(vis=0)=0.36 · longest full-invis=8 · **occlusion_strong** |
| Annotation | same GT id; continuous 10–200; no internal gaps |
| Signal recompute | PASS |
| Threshold | `resid_mean` near median (+0.3%); `speed_mismatch` strong (+228% of thr) → not artifact-dominated |
| Competing | near-median residual; membership drops under p60 binarization |

**Reading:** classic short-to-medium occlusion fragmentation on MOT17-10 — height/scale preserved, motion continuity and bridge geometry break. Supports motion role-reversal as a **conditional** story, not a global ordering dimension.

### 3.2 `MOT17-10-SDP|459` → `UNRESOLVED`

| item | value |
|:--|:--|
| Timeline | f=160 → cand 498 f=276 · gap=116 · GT id 29 |
| Height | h_raw **95.0 → 163.2** · `log_h_ratio=0.541` SAFE but only **0.026** under thr 0.567 |
| Motion on min-`d_H` | **VIOL** all three motion atoms (`resid_mean`, `dir_cos`, `speed_mismatch`) · pure motion triple |
| `d_H` | 3 |
| Gap visibility | mean **0.65** · frac(vis=0)=**0** · longest full-invis=**0** · **not occlusion_strong** |
| Annotation | same GT id; continuous 90–351 |
| Signal recompute | PASS |
| Threshold | `resid_mean` near-median; leaves tail under p60 |

**Reading:** long tracker gap on a **still-visible, approaching** pedestrian (scale nearly doubles; direction reverses). Motion atoms fail, but this is **not** clean long-occlusion re-entry, and height is only barely “safe” under the exploratory median. Competing THRESHOLD_ARTIFACT and fragmentation-without-occlusion hypotheses remain open → **`UNRESOLVED`**.

### 3.3 `MOT17-10-SDP|467` → `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | exit f=168; **7** gt_match re-entries (gaps 72…237) · GT id 71 |
| min-`d_H` row | cand 489 · gap=72 · f=168→240 |
| Height | h_raw 59.91 → 56.74 · `log_h_ratio=0.054` **SAFE** on all 7 rows |
| Motion / geom on min-`d_H` | **VIOL** `dir_cos`, `speed_mismatch`, `dist_h` |
| `d_H` | min 3 (other fragments up to 7) |
| Gap visibility (min row) | mean **0.03** · frac(vis=0)=0.80 · longest full-invis=44 · **occlusion_strong** |
| Annotation | same GT id; continuous 143–562 |
| Signal recompute | PASS |

**Reading:** strongest case — deep occlusion, multi-fragment reappearance, height stable, motion reverse + speed mismatch. Textbook long-gap re-entry under crowd occlusion on a moving-camera street sequence.

### 3.4 `MOT17-10-SDP|503` → `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | exit f=293; cand 536/541 · gaps 136, 165 · GT id 59 |
| min-`d_H` row | cand 536 · gap=136 · f=293→429 |
| Height | h_raw 52.01 → 62.55 · `log_h_ratio=0.185` **SAFE** |
| Motion / geom on min-`d_H` | **VIOL** all three motion + `score_m_bridge`, `bridge_dist`, `gap` |
| `d_H` | **6** (stays in tail even under p60) |
| Gap visibility | mean 0.48 · frac(vis=0)=0.27 · longest full-invis=20 · **occlusion_strong** |
| Annotation | same GT id; continuous 235–530 |
| Signal recompute | PASS |

**Reading:** long gap with partial occlusion, height preserved, multi-atom motion/geometry collapse. Membership is **not** a pure median-split accident (survives p60).

## 4. Cross-track checks

| check | result |
|:--|:--|
| Annotation issues | **0/4** — all gt_match rows keep `gt_lost == gt_cand`; no GT id internal gaps |
| Signal computation issues | **0/4** — formulas recompute; values match Step-0 `gt_rows.csv` within `.6g` rounding |
| Height atom on min-`d_H` | **0/4 violate `log_h_ratio`** (matches Step-0 descriptive profile) |
| Motion enrichment | min-`d_H` motion violations: 455→2, 459→3, 467→2, 503→3 |
| Threshold domination | **0/4** under the predeclared rule (strong violations remain on every track) |
| Sequence clustering | **4/4 MOT17-10-SDP** — independent-trial count is not 4 |

Threshold membership (diagnostic only):

| binarization | frozen tracks still with min `d_H≥3` |
|:--|:--|
| pool p40 | 455, 459, 467, 503 |
| pool median (sealed exploratory) | 455, 459, 467, 503 |
| pool p60 | **503 only** |

Membership is median-sensitive for 455/459/467, but the mechanism evidence (occlusion + multi-atom motion break + height safe) is not reducible to a single near-median flip.

## 5. Aggregate bounded terminal

```text
Aggregate terminal: ROLE_REVERSAL_SUPPORTED

counts:
  TRUE_LONG_GAP_REENTRY     3
  UNRESOLVED                1
  ANNOTATION_ISSUE          0
  SIGNAL_COMPUTATION_ISSUE  0
  THRESHOLD_ARTIFACT        0
```

### What this authorizes

- A **separate** partial-order audit may treat motion atoms (`speed_mismatch`, `dir_cos`, `resid_mean`) as candidates for `conditional_orderable` or `context_only` (research-line Phase B / PR-D prep).
- Height/scale (`log_h_ratio`) remains the cleanest global-orderable candidate from this tail (0/4 min-row violations).

### What remains blocked

- global closure arcs on motion atoms;
- veto against the protected escape tail;
- production rules, presets, default tracker decisions, closed OR-tail gate changes;
- evidence_ledger promotion beyond this bounded forensic result;
- interpreting zero-exposure cells as unsafe;
- L2+ morphology claims without nested held-out confirmation (PR-E);
- treating MOT17-10 as a multi-sequence population (clustering is fact).

### Downstream routing (issue #102)

`ROLE_REVERSAL_SUPPORTED` → open a **separate** partial-order audit task before any MWC prototype (PR-D). Conditional closure probe stays contingent on that audit, not on this forensic alone.

## 6. Must not (reaffirmed)

- change atom definitions, directions, or sealed thresholds;
- run gate-rule search / MWC / min-cut / closure compression / policy optimization in this unit;
- revise framework §19 morphology terminals from these four cases;
- promote the aggregate terminal into production or the evidence ledger.

## 7. Engineering vs research acceptance

| layer | status |
|:--|:--|
| Engineering | deterministic runner + committed packet + `--verify` reproduction |
| Research acceptance | chat/owner review of categories + aggregate terminal (separate from merge) |
| Next PR | partial-order audit / restricted closure prototype (PR-D), not automatic |
