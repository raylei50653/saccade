---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization R1.1 — Transfer Failure Attribution Pack

> **One-line:** **DOWNGRADED to external diagnostic overlay** (2026-07-10 chat review). Net contribution: **2 unique harmful AND events located + 3 descriptive symptom classes** (role reversal, weak holdout retention, margin contraction). "**Primary F3**" as a causal mechanism ranking is **rejected** — see downgrade section. Terminal B retained (it never rested on this note). R2 still unauthorized.

## 2026-07-10 chat-review downgrade (normative — read before the rest)

Verified against code/artifacts (details: [A1 audit note](safe_region_a1_audit_20260711.md)):

1. **Post-hoc scoring floors.** F1/F3/F4 are hard-floored to 40/55/50 (`safe_region_assetization_r11.py:584,598,605`); the rules were not pre-registered in parent R1, so the F-ranking is descriptive taxonomy, not a robust mechanism ranking.
2. **The F3 ≥3-reversal floor trigger double-counts.** The four L3 reversals in `basis_role_reversal.csv` are **2 unique (basis, holdout-event) pairs** — `b2:07de9243…`→MOT17-05 `f4:c8:l1:i32` and `b2:afd594d1…`→MOT17-11 `f4:c0:l4:i5` — each counted under K2 and K5. Deduplicated evidence = 2 < 3, so the 55 floor would not fire.
3. **Predicate identity is not unique.** `b2:07de9243…` has **12 observed-mask aliases** (six `abs_log_h↑ ∧ score_m_bridge↓`, six `abs_ratio_m1↑ ∧ score_m_bridge↓`); naming one alias as "the" mechanism violates the Boolean contract (observed-mask equality ≠ logical/semantic identity).
4. This study **does not consume the A0 conversion pack** (inputs: raw Q4.5 registry + events), so it contributes no A1 pack-consumption evidence.

**Acceptable residue:** role reversal, weak retention, and margin contraction *exist* as observed symptoms on this heuristic. **Not acceptable:** "F3 identified as primary causal mechanism", transfer-null claims, or reading the taxonomy scores as calibrated.

## Authorization

| Item | Value |
|:--|:--|
| Task | `R1.1 — Transfer Failure Attribution Pack` |
| Status | **AUTHORIZED** · sole active |
| Parent | R1 V-C ([safe_region_assetization_r1_20260710.md](safe_region_assetization_r1_20260710.md)) |
| Study | `out/signal_study/safe_region_assetization_r11_20260710/` |
| Code | `src/saccade/perception/eval/safe_region_assetization_r11.py` · `scripts/tools/run_safe_region_assetization_r11.py` |
| Tests | `tests/unit/test_safe_region_assetization_r11.py` |
| LOO protocol | `transductive_globally_registered_basis_LOO` (unchanged naming) |

### Forbidden (enforced)

```text
no new model family · no new grammar · no new signals
no optimizer tuning for better LOO · no hook/preset/production
no grammar distillation · no evidence_ledger promotion · no reopen terminal A
```

---

## Scope

**Only goal:** explain why L2/L3 have pooled hard-safe productivity but hurt held-out GT under sequence LOO.

Focus models (fixed, not LOO-tuned):

| Model | Pooled n_neg | LOO hold GT hurt (sum) | Mean pairwise active Jaccard |
|:--|--:|--:|--:|
| L2 K2 | 1 | 2 | 0.86 |
| L2 K5 | 1 | 3 | 0.81 |
| L3 K2 | 5 | 2 | 0.52 |
| L3 K5 | 8 | 2 | 0.59 |

---

## Fixed outputs (emitted)

| File | Content |
|:--|:--|
| `fold_summary.csv` | per-fold active set, τ, train/hold capture, dominance |
| `basis_overlap_jaccard.csv` | pairwise fold active-set Jaccard |
| `basis_selection_stability.csv` | basis selection frequency across folds |
| `holdout_event_attribution.csv` | hold GT-hurt / neg-captured events + firing bases |
| `basis_role_reversal.csv` | train-neg vs hold-GT support per active basis |
| `margin_contraction.csv` | train GT margin → hold GT margin |
| `registry_global_vs_train.csv` | global label-free pure-safe vs train-only pure-safe |
| `model_attribution_summary.csv` | per-model aggregates |
| `failure_taxonomy.json` | primary + secondary + scores + decision mapping |
| `summary.json` / `manifest.json` | study authority |

---

## Key empirical findings

### 1. Role reversal (F3 core)

Bases selected as **train productive (neg support)** fire on **held-out GT**:

| Basis (abbrev) | Kind | Train role | Holdout harm folds |
|:--|:--|:--|:--|
| `score_m_bridge` low_tail thr~0.053 | G1 | train neg | MOT17-05 (L2) |
| `dist_h` high_tail thr~0.47 | G1 | train neg | MOT17-05 (L2 K5) |
| `abs_log_h`↑ ∧ `score_m_bridge`↓ | G2 AND | train neg | MOT17-05 (L3) |
| `resid_mean`↑ ∧ `score_m_bridge`↓ | G2 AND | train neg | MOT17-11 (L3) |

L3 holdout GT-hurt events (complete list in `holdout_event_attribution.csv`):

- `MOT17-05-SDP:f4:c8:l1:i32` ← `b2:07de9243…` (AND abs_log_h high ∧ score_m_bridge low)
- `MOT17-11-SDP:f4:c0:l4:i5` ← `b2:afd594d1…` (AND resid_mean high ∧ score_m_bridge low)

Same predicate polarity that marks removable negatives on train marks GT-consistent pairs on holdout → **cross-sequence semantic/sign conflict**, not mere threshold drift.

### 2. Margin contraction (F1 secondary)

Folds with train GT margin > 0 and hold GT margin < 0: **4/14** (L3: MOT17-05, MOT17-11 for both K2 and K5).

```text
example L3:K5 / MOT17-05:
  train_gt_safety_margin = +0.10
  hold_gt_margin         = −0.10
  hold_gt_hurt           = 1
```

Transport of the decision boundary fails where role-reversed bases fire.

### 3. Productive islands / weak holdout retention (F4 secondary)

- L3 train sequence dominance mean ≈ **0.30** (not single-seq monopoly, but still skewed).
- Train-productive-basis → holdout-neg retention: L3 K2 **0.29**, K5 **0.11** (most productive bases do **not** re-fire on holdout negatives).
- Holdout productive capture is sparse (L3 total hold neg capture sum = 4 across 7 folds).

Pooled multi-seq support (5–6 seq) does **not** imply per-fold transferable productive geometry.

### 4. Basis stability (F2 not primary)

| Signal | Value |
|:--|:--|
| L3 mean pairwise active Jaccard | **0.52–0.59** (moderate instability) |
| L2 mean pairwise active Jaccard | **0.81–0.86** (more stable, still hurts) |
| Global vs train pure-safe Jaccard | **~0.70** |

Active-set churn exists (especially L3) but L2 is relatively stable and **still** produces hold GT hurt via role reversal → F2 is real but not the binding mechanism.

### 5. Registry layer (diagnostic only)

```text
global label-free pure-safe registry
  ≠ train-only pure-safe registry   (Jaccard ~0.70)

LOO remains:
  global basis coordinates + train-only supervised selection
  + held-out label isolation
```

Train-only pure-safe sets gain ~7 extra AND bases per L3 fold on average (`n_only_train`), i.e. labels reshuffle which registered masks look “pure” — consistent with sequence-conditioned semantics, not a separate bug.

### 6. Support (F5 rejected as primary)

Max pooled n_neg = **8** under hard safety → not an empty-cohort null. F5 score low (10).

---

## Failure taxonomy (accepted structure)

```text
primary:   F3  cross-sequence semantic / sign conflict      [REJECTED as "primary" — see downgrade section]
secondary: F4  single-sequence / weak-transfer productive islands
           F1  coordinate / margin transport failure
```

Scores (L3-emphasis aggregate): F3=55 · F4=50 · F1=40 · F2=39.9 · F5=10 — **post-hoc floored values** (F3=55 and F4=50 are floor hits, not measured magnitudes; the F3 floor fires only via K2/K5 double-counting).

### Decision mapping

| Code | Mapping | R1.1 application |
|:--|:--|:--|
| **F3** | frozen signals lack invariant semantics → conditional applicability **or** new signal family | **Primary path** |
| **F4** | sequence-conditioned islands; no global gate | Reinforces: do not ship global threshold gate |
| **F1** | relative/normalized coordinate transport spec | **Narrow optional** only if F3 addressed; not sufficient alone |
| F2 | close fixed-basis grammar path | Not primary (stability moderate) |
| F5 | inconclusive | Rejected |

### What this closes

```text
✗ “more expressive second-order sparse linear grammar will transfer”
✗ global threshold safe-region gate on this frozen signal pack
✗ R2 grammar distillation (still requires V-A; V-C+F3 forbids)
```

### What this does **not** close

```text
· all linear models forever
· all coordinate normalizations (F1 secondary leaves a narrow door)
· conditional / sequence-scoped applicability research
· new signal family research
```

---

## Relation to terminal B

Q4.5/R1: thin residual safe points; no thick portable region.

R1.1 mechanism:

> Productive predicates on train reverse role on holdout sequences (especially MOT17-05 / MOT17-11), collapsing GT margin. Pooled multi-seq capacity was an aggregate of sequence-conditioned islands, not an invariant safe region.

Terminal **B retained and mechanistically strengthened**.

---

## Reproduce

```bash
.venv/bin/python scripts/tools/run_safe_region_assetization_r11.py \
  --study-id safe_region_assetization_r11_20260710 \
  --out out/signal_study/safe_region_assetization_r11_20260710

.venv/bin/python -m pytest tests/unit/test_safe_region_assetization_r11.py -q --no-cov
```

---

## Next-stage gate (owner only)

**2026-07-10 review supersedes this ordering:** do **not** pick one of the four lines yet — the A1 terminal on the conversion pack (`1a180620bc…`) comes first; the S0/S1/Q1/N1 audit passed 2026-07-11 and the terminal awaits the owner. Only after the terminal, choose **one** narrow line (not all):

1. **Conditional applicability** research (F3+F4) — sequence/context-gated use of residual predicates; still no global gate.
2. **New signal family** (F3) — only with explicit owner auth and new cohort contract.
3. **Relative/normalized coordinate transport spec** (F1 only) — diagnostic design doc, not implementation, and only if still pursuing threshold geometry.
4. **Formal close** of threshold safe-region mainline on frozen-5 absolute coordinates.

Default fail-closed: **no eng. start** until owner picks exactly one of the above.
