---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# Escape-Tail Forensic — Four-Track PR-C (issue #102)

> **One-line:** Sealed Step-0 far-Hamming descriptive tail（4 tracks，`k=8`，`d_H >= 3`，**4/4 MOT17-10-SDP**）→ **3× `TRUE_LONG_GAP_REENTRY` + 1× `UNRESOLVED`** · aggregate **`ROLE_REVERSAL_SUPPORTED`**. Research acceptance = **`ACCEPTED_WITH_LIMITS`**（[PR #104](https://github.com/raylei50653/saccade/pull/104) review）：L1 单序列 forensic only；仅授权后续独立 partial-order audit。Numerical cutoffs remain **PR-C operationalizations** (not sealed in #102).

Thread: [gt_support_morphology_20260711.md](../../../research/threads/gt_support_morphology_20260711.md) ·
Procedure: [framework §19](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) ·
Research line: [boolean_closure_domain_line_20260711.md](boolean_closure_domain_line_20260711.md) ·
Step-0: [gt_support_morphology_step0_20260711.md](gt_support_morphology_step0_20260711.md) ·
Packet: [evidence/escape_tail_forensic_20260711/](evidence/escape_tail_forensic_20260711/manifest.json)

## 0. Scope and claim ceiling

- **Read-only / offline / evidence-only**（issue #102 scope guards）。
- Frozen cohort 仅来自 sealed Step-0 `tail_tracks.json`；不增不删 tracks。
- 二值化 = Step-0 已宣告的 pool-median（audit-only）；本 forensic **不**重设 sealed thresholds，只做敏感性诊断。
- Claim ceiling = **L1 forensic** on a single sequence cluster。Nested held-out（PR-E）仍是唯一可超 L1 的 confirmatory unit。
- Explicitly **not** done: gate-rule search, MWC, min-cut, closure arcs, orderability promotion into a sealed partial order, production/preset/ledger changes.

### 0.1 What #102 predeclared vs what PR-C operationalized

| Layer | Status |
|:--|:--|
| Per-track terminal **names** (5) | predeclared in #102 |
| Aggregate terminal **names** (3) | predeclared in #102 |
| Required evidence checklist | predeclared in #102 |
| `occlusion_strong` numerical cutoffs (`vis_mean ≤ 0.35` / `frac(vis=0) ≥ 0.25`) | **PR-C operationalization** (not sealed in #102) |
| `≥ 2` motion violations for TRUE | **PR-C operationalization** |
| Aggregate `≥ 3 TRUE + 0 artifact → ROLE_REVERSAL_SUPPORTED` | **PR-C operationalization** |
| `THRESHOLD_REL_FLIP_MAX = 5%` | **PR-C operationalization** |

Do **not** describe the numerical cutoffs as “predeclared” or “fixed before viewing outcomes.” They are recorded in [`classification_rules.json`](evidence/escape_tail_forensic_20260711/classification_rules.json) for reproducibility and remain subject to research-owner interpretation.

## 1. Frozen cohort

| track_key | sealed min `d_H` | n `gt_match` rows | primary GT id |
|:--|--:|--:|--:|
| `MOT17-10-SDP\|455` | 4 | 1 | 41 |
| `MOT17-10-SDP\|459` | 3 | 1 | 29 |
| `MOT17-10-SDP\|467` | 3 | 7 | 71 |
| `MOT17-10-SDP\|503` | 6 | 2 | 59 |

Provenance: Step-0 packet SHA seal on `pairs.csv` =
`0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17`；runner 拒绝 SHA 不符的来源。

## 2. Method

Runner: [`run_escape_tail_forensic.py`](evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py) ·
Rules: [`classification_rules.json`](evidence/escape_tail_forensic_20260711/classification_rules.json) ·
Scene: [`scene_evidence.json`](evidence/escape_tail_forensic_20260711/scene_evidence.json) + [`scene_sheets/`](evidence/escape_tail_forensic_20260711/scene_sheets/)

### 2.1 Signal-computation check (non-tautological)

Derived atoms are **recomputed from pairs.csv raw columns** under declared formulas and compared to the **independent sealed** Step-0 `gt_rows.csv` values for the same `(track_key, gap)`. Additional checks:

- `gap == cand_first_frame - lost_last_frame`
- builder field domain: `|dir_cos| ≤ 1`, non-negative distances / heights / gap

**Explicitly untested (declared residual risk):** pixel/trajectory replay of builder-emitted `fwd_resid` / `bwd_resid` / `dir_cos` / speeds. Aggregate terminals therefore **cannot** claim that substrate-level signal bugs are fully ruled out—only that formula-vs-sealed-ledger and frame-gap consistency pass on this packet.

### 2.2 Scene-level evidence (issue #102 checklist)

Per min-`d_H` row the packet records:

| Requirement | Implementation |
|:--|:--|
| Occlusion | GT visibility in gap window + operational `occlusion_strong` flag + mid-gap frames on contact sheet |
| Nearby identities | top-5 other GT ids by foot distance + IoU at exit and re-entry frames |
| Truncation | box-vs-frame-border flags (margin 8 px) at exit/re-entry |
| Camera / scene motion | median foot displacement of all GT ids present at both exit and re-entry frames (proxy, not GMC) |
| Re-entry geometry | feet delta, endpoint boxes, contact sheet |
| Auditable frames | JPEG contact sheet: exit · mid-gap samples · re-entry, with primary GT (green) and others (orange) |

Contact sheets:

- [`scene_sheets/MOT17-10-SDP_455_min_dh.jpg`](evidence/escape_tail_forensic_20260711/scene_sheets/MOT17-10-SDP_455_min_dh.jpg)
- [`scene_sheets/MOT17-10-SDP_459_min_dh.jpg`](evidence/escape_tail_forensic_20260711/scene_sheets/MOT17-10-SDP_459_min_dh.jpg)
- [`scene_sheets/MOT17-10-SDP_467_min_dh.jpg`](evidence/escape_tail_forensic_20260711/scene_sheets/MOT17-10-SDP_467_min_dh.jpg)
- [`scene_sheets/MOT17-10-SDP_503_min_dh.jpg`](evidence/escape_tail_forensic_20260711/scene_sheets/MOT17-10-SDP_503_min_dh.jpg)

### 2.3 Operational TRUE rule (not #102-sealed)

```text
TRUE_LONG_GAP_REENTRY (operational) iff
  annotation OK
  ∧ independent signal checks OK
  ∧ not threshold_artifact_dominates
  ∧ height safe on min-d_H row
  ∧ n(motion violations) >= 2
  ∧ occlusion_strong_operational
  ∧ contact sheet available
  ∧ scene_supports_occlusion_or_crowd
```

Reproduce:

```bash
uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

uv run python docs/modules/semantic/research/evidence/escape_tail_forensic_20260711/run_escape_tail_forensic.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --verify
```

## 3. Per-track cards (operational categories)

### 3.1 `MOT17-10-SDP|455` → operational `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | f=114 → cand 468 f=171 · gap=57 · GT id 41 |
| Height | h_raw 60.35 → 60.75 · `log_h_ratio=0.0066` SAFE |
| Motion / geom | VIOL `speed_mismatch`, `resid_mean`, `bridge_dist`, `dist_h` · `dir_cos≈0.96` SAFE |
| `d_H` | 4 |
| Gap visibility | mean ≈0.19 · frac(vis=0)≈0.36 · occlusion_strong_operational **true** |
| Scene | contact sheet present; nearby GT at re-entry (e.g. id 71 ~34 px); camera-motion proxy median foot disp recorded |
| Annotation | same GT id; continuous |
| Signal check | PASS vs sealed Step-0 + gap/frame + domain |
| Competing | near-median `resid_mean`; p60 membership sensitivity; nearby-id residual |

### 3.2 `MOT17-10-SDP|459` → `UNRESOLVED`

| item | value |
|:--|:--|
| Timeline | f=160 → cand 498 f=276 · gap=116 · GT id 29 |
| Height | h_raw **95 → 163** · `log_h_ratio=0.541` SAFE but near thr |
| Motion | all three motion atoms VIOL |
| Gap visibility | mean ≈0.65 · frac(vis=0)=**0** · occlusion_strong_operational **false** |
| Scene | contact sheet shows a still-visible approaching person; scale change dominates |
| Signal check | PASS |
| Why not TRUE | fails operational occlusion_strong; not cleanly long-occlusion re-entry |

### 3.3 `MOT17-10-SDP|467` → operational `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | exit f=168; 7 gt_match re-entries · GT id 71 |
| min-`d_H` | cand 489 · gap=72 · f=168→240 |
| Height | SAFE on all 7 rows |
| Motion / geom | VIOL `dir_cos`, `speed_mismatch`, `dist_h` |
| Gap visibility | mean ≈0.03 · frac(vis=0)≈0.80 · occlusion_strong_operational **true** |
| Scene | contact sheet + nearby identities at re-entry; deep mid-gap invisibility on sheet |
| Signal check | PASS |

### 3.4 `MOT17-10-SDP|503` → operational `TRUE_LONG_GAP_REENTRY`

| item | value |
|:--|:--|
| Timeline | exit f=293; gaps 136 / 165 · GT id 59 |
| min-`d_H` | cand 536 · gap=136 · `d_H=6` (survives p60) |
| Height | SAFE |
| Motion / geom | multi-atom motion + geometry VIOL |
| Gap visibility | mean ≈0.48 · frac(vis=0)≈0.27 · occlusion_strong_operational **true** |
| Scene | contact sheet + nearby cluster at re-entry |
| Signal check | PASS |

## 4. Cross-track checks

| check | result |
|:--|:--|
| Annotation issues | **0/4** |
| Independent signal checks (step0 sealed + gap/frame + domain) | **0/4 fail** |
| Builder pixel-replay of residuals/speeds | **untested** (declared) |
| Height atom on min-`d_H` | **0/4 violate `log_h_ratio`** |
| Contact sheets | **4/4 present** |
| Sequence clustering | **4/4 MOT17-10-SDP** |

Threshold membership (diagnostic only):

| binarization | frozen tracks still with min `d_H≥3` |
|:--|:--|
| pool p40 | 455, 459, 467, 503 |
| pool median | 455, 459, 467, 503 |
| pool p60 | **503 only** |

## 5. Aggregate (operational)

```text
Aggregate terminal: ROLE_REVERSAL_SUPPORTED
research_acceptance: ACCEPTED_WITH_LIMITS
authority: PR #104 research-owner review (2026-07-11)
claim_ceiling: L1 single-sequence forensic (MOT17-10-SDP only)

counts:
  TRUE_LONG_GAP_REENTRY     3
  UNRESOLVED                1
  ANNOTATION_ISSUE          0
  SIGNAL_COMPUTATION_ISSUE  0
  THRESHOLD_ARTIFACT        0
```

### What this authorizes (within limits)

- A **separate** partial-order audit may treat motion atoms as candidates for `conditional_orderable` or `context_only`.

### What remains blocked

- global closure arcs on motion atoms;
- MWC conclusions from this forensic alone;
- veto against the protected escape tail;
- production rules / presets / ledger promotion;
- claiming pixel-level signal bugs are fully ruled out;
- multi-sequence generalization from MOT17-10 alone;
- L2+ morphology claims without nested confirmation (PR-E).

### Downstream routing

`ROLE_REVERSAL_SUPPORTED` under `ACCEPTED_WITH_LIMITS` → open partial-order audit (PR-D prep) before any MWC prototype.

## 6. Response to PR #104 research-owner review

| Blocking issue | Fix in this revision |
|:--|:--|
| 1. Signal check tautology | Recompute from pairs raw **vs sealed Step-0 `gt_rows`**; plus gap/frame + domain checks. `SIGNAL_COMPUTATION_ISSUE` is reachable on sealed mismatch. Builder pixel-replay declared untested. |
| 2. Missing scene evidence | Contact sheets + nearby IDs + truncation + camera-motion proxy in packet; note no longer asserts unaudited “crowd/moving-camera” without sheet/proxy fields. |
| 3. Numerical rules not predeclared | Relabeled as **PR-C implementation-time operationalization**; #102 only owns terminal vocabulary. |

## 7. Must not (reaffirmed)

- change atom definitions, directions, or sealed thresholds;
- run gate-rule search / MWC / min-cut / closure compression in this unit;
- revise framework §19 morphology terminals from these four cases;
- promote beyond the bounded forensic result (no production / ledger / global motion closure).

## 8. Engineering vs research acceptance

| layer | status |
|:--|:--|
| Engineering | deterministic runner + committed packet + `--verify` + scene sheets |
| Research acceptance | **`ACCEPTED_WITH_LIMITS`** ([PR #104](https://github.com/raylei50653/saccade/pull/104) research-owner review) |
| Next | PR-D partial-order audit (motion = conditional/context candidates only) |
