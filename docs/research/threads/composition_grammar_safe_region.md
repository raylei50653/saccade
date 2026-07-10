---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# composition grammar × safe-region geometry

> **One-line:** T0-B-R1 **DONE** · awaiting **PR review**. Terminal B unchanged. Multi-seq = **12 coords → 8 per-grid masks / 4 global diagnostic**; bidirectional per-seq PASS; radius≥1 = **0**. Bounded verdict **not self-accepted**.

## Status

| Item | Status |
|:--|:--|
| Coverage audit (recon) | **CLOSED** |
| Q4.5 terminal | **B** `isolated_safe_points_only` (unchanged) |
| Production preset | **unchanged** |
| Signal family expansion | **not authorized** |
| Unrestricted Boolean / 3+ atoms | **forbidden** |
| T0-A artifact preflight | **ACCEPTED** |
| T0-A report | [composition_grammar_t0_artifact_preflight_20260710.md](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md) |
| T0-B interpretation pack | **EXECUTED** · R1 corrections applied |
| T0-B-R1 | **DONE** — unit + reconciliation fixes |
| T0-B note | [composition_grammar_t0_region_interpretation_20260710.md](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md) |
| T0-B evidence | [evidence/m_b1_5_t0_region_interpretation_20260710/](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/) |
| Current authorized work | **none pending implementation** — hold for PR review |
| Bounded verdict | **not yet accepted** (do not self-accept) |
| G7 item | **contract-gap only** |
| `evidence_ledger` | **not promoted** |

```text
Central judgment (held):
  Primary Q4.5 gap = region observation + validation contraction
  ≠ missing signals
  ≠ arbitrary Boolean grammar
```

## Current boundary

- Fixed frozen 5-signal family; locked \(D_{\text{online}}\) decision cohort.
- Runtime full atlases required; committed Q4.5 pack alone insufficient.
- G1 `S::` ⟂ G2/G3 `P::`; `mask_sha256` primary scope = per registered grid.
- G7 cannot be audited without inventing NOT / operand-role semantics.
- R1 closed: multi-seq units, bidirectional per-seq equality, mask invariance, capacity concentration, naming.

## Read first

1. **[T0-B interpretation note (R1)](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)** — review input
2. **[T0-A preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)**
3. **[Coverage audit](../../modules/semantic/research/composition_grammar_safe_region_coverage_audit_20260710.md)**
4. Stage 2 canonical: [m_b1_5_stage2_d_online_final_20260710.md](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md)

## Truth base

```text
Q4.5 evaluator:  PR #89 head 6df1739b · merge 234f9f59
Docs stack:      PR #90 consolidation 51b9c73e
Integration:     PR #93 merge 8f7a3700
T0-A accepted:   f8cfff56 + f1981c12
T0-B executed:   7b54f5c2
T0-B-R1 dispatch: 32ecd242
Machine study:   out/signal_study/m_b1_5_stage2_q45_20260710/
```

## Current step — awaiting PR review

**Branch:** `research/m-b1-5-t0-region-interpretation`  
**Script:** `scripts/tools/analyze_m_b1_5_t0_region_interpretation.py`  
**Evidence revision:** `T0-B-R1`

### R1 acceptance checklist (execution)

```text
[x] original input hashes unchanged
[x] no evaluator rerun or modification
[x] headline 154 = 1 + 153 + 0
[x] bidirectional per-sequence equality PASS
[x] per-grid mask invariance PASS
[x] multi-seq: 12 coords → 8 primary per-grid masks / 4 global diagnostic
[x] capacity concentration top-1/3/5 reported (∑mask_n_neg denom)
[x] dual-margin headline reproduced (radius≥1 = 0/154)
[x] G7 remains contract-gap
[x] Terminal B / production / ledger unchanged
```

### Reconciled headline (post-R1)

```text
154 PS = 1 G1 + 153 G2 + 0 G3
142 single-seq · 12 multi-seq (AND)
12 multi-seq coords → 8 per-registered-grid masks · 4 global strings (diagnostic)
34 productive per-grid mask units · ∑mask_n_neg=48 · top5 share≈33%
143/154 coords on multi-coord per-grid mask plateaus
26 components · 12 single-cell-width strips · 0 genuine 2D-thick
full_neighborhood_safe_radius≥1: 0/154
nearest_unsafe_distance=1 on all PS
G7: not_derivable_from_current_artifact_contract
Terminal B / production / ledger: unchanged
```

### Bounded verdict candidate (still provisional — human accept only)

> Productivity among the 154 PS cells is explained by threshold-coordinate mask plateaus, single-sequence support, and axis-degenerate / edge-touching components. No registered full-neighborhood thickness under conservative edge policy. Multi-seq body is small (12 coords / 8 per-grid masks). Not a portable region, production candidate, or G7 result.

### Review gate

```text
T0-B execution ✓
→ R1 evidence correction ✓
→ PR review  ← current (open PR only when ready; not auto)
→ bounded research acceptance
→ only then: close line / minimal emit / region-LOO / restricted G7
```

## Acceptance boundary (held)

T0-B may conclude only descriptive geometry of the existing registered atlas.

It must not claim:

```text
formal or portable safe region
online parameter-region retention
productive reject policy
production candidate
G7 equivalence
new grammar necessity
threshold-path global falsification
```

## Must not (held)

- Rewrite or rerun the Q4.5 evaluator.
- Rebuild missing runtime atlases.
- Expand research scope beyond R1 corrections.
- Self-accept the bounded verdict.
- Auto-open PR without explicit request.
- Promote evidence or change terminal B.

## History

- 2026-07-10: coverage audit closed; thread opened.
- 2026-07-10: T0-A completed/accepted; T0-B authorized.
- 2026-07-10: T0-B executed at `7b54f5c2`.
- 2026-07-10: review CHANGES REQUESTED (R1) at `32ecd242`.
- 2026-07-10: T0-B-R1 applied; status → awaiting PR review (verdict unaccepted).
