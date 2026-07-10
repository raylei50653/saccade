---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# composition grammar × safe-region geometry

> **One-line:** T0-B executed; review = **CHANGES REQUESTED (R1)**. Terminal B unchanged. Current work is a bounded evidence-correction pass only: fix multi-sequence mask units, strengthen fail-closed reconciliation, and close naming/reporting gaps. No evaluator rerun or research expansion.

## Status

| Item | Status |
|:--|:--|
| Coverage audit (recon) | **CLOSED** — fact note linked below |
| Q4.5 terminal | **B** `isolated_safe_points_only` (unchanged) |
| Production preset | **unchanged** |
| Signal family expansion | **not authorized** by this thread |
| Unrestricted Boolean / 3+ atoms | **forbidden** |
| T0-A artifact preflight | **ACCEPTED** |
| T0-A report | [composition_grammar_t0_artifact_preflight_20260710.md](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md) |
| T0-B interpretation pack | **EXECUTED** — review changes requested |
| T0-B note | [composition_grammar_t0_region_interpretation_20260710.md](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md) |
| T0-B evidence | [evidence/m_b1_5_t0_region_interpretation_20260710/](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/) |
| Current authorized work | **T0-B-R1 correction only** |
| Bounded verdict | **not yet accepted** |
| G7 item | **contract-gap report only** (`not_derivable…`) — not equivalence |
| `evidence_ledger` | **not promoted** |

```text
Central judgment (held):
  Primary Q4.5 gap = region observation + validation contraction
  ≠ missing signals
  ≠ arbitrary Boolean grammar
```

## Current boundary

- Fixed frozen 5-signal family; locked \(D_{\text{online}}\) decision cohort.
- G1–G3 are complete only within their registered Q4.5 lattices.
- Runtime full atlases are required; committed evidence pack alone is insufficient.
- G1 `S::` unique-boundary atoms and G2/G3 `P::` quantile atoms are disjoint namespaces.
- `mask_sha256` is not a global primary key; primary quotient scope is per registered grid.
- G4–G6 remain deferred.
- G7 cannot be audited from current artifacts without inventing NOT / operand-role semantics.
- Online selected-policy retention remains distinct from parameter-region retention.
- R1 may only correct derived analysis, reconciliation, labels, note, and evidence pack.

## Read first

1. **[T0-B interpretation note](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)** — current correction target
2. **[T0-A preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)** — accepted schema/key/derivability contract
3. **[Coverage audit](../../modules/semantic/research/composition_grammar_safe_region_coverage_audit_20260710.md)** — T0 research questions and bounded verdicts
4. Stage 2 canonical: [m_b1_5_stage2_d_online_final_20260710.md](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md) · [entry contract](../../modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md)
5. Closed Stage 1/2 history: [m_b1_online_hook_20260709.md](m_b1_online_hook_20260709.md)
6. Signal table contract: [signal_table_schema.md](../eval/signal_table_schema.md)

## Truth base

```text
Q4.5 evaluator:  PR #89 head 6df1739b · merge 234f9f59
Docs stack:      PR #90 consolidation 51b9c73e
Integration:     PR #93 merge 8f7a3700
T0-A accepted:   f8cfff56 + f1981c12
T0-B executed:   7b54f5c2
Machine study:   out/signal_study/m_b1_5_stage2_q45_20260710/
```

## T0-A review verdict

```text
T0-A: ACCEPTED

Accepted facts:
  runtime atlases hash-match manifest;
  committed pack is a subset and cannot substitute for full atlases;
  outputs 1–5 are directly/deterministically derivable;
  dual margin is derivable only with an explicit conservative edge policy;
  G7 form is not derivable because NOT / necessary / support roles are absent.

Contract correction:
  T0-B item 7 is not a G7 equivalence result.
  It must report `not_derivable_from_current_artifact_contract`,
  with optional G1/G2 mask-string overlap clearly labeled non-G7.
```

## T0-B review verdict — CHANGES REQUESTED (R1)

The overall geometry headline remains plausible and no evaluator rerun is needed, but the current evidence packet is not yet accepted because two review findings affect required T0 answers.

### R1.1 Multi-sequence mask unit correction — blocking

Current code/evidence labels `4` as the number of per-grid unique masks among the 12 multi-sequence AND coordinates, but that value is the **global mask-string diagnostic**.

The emitted per-grid counts are:

```text
1 + 3 + 1 + 3 = 8 per-grid mask units
```

R1 must report both without collapsing them:

```text
12 multi-sequence AND coordinates
→ 8 primary per-registered-grid unique masks
→ 4 global mask strings (diagnostic only)
```

Rename machine fields so no `per_grid_*` field contains the global-string count. Update summary, reconciliation, research note, and thread headline where applicable.

### R1.2 Bidirectional per-sequence reconciliation — blocking

The current PASS checks atlas-positive JSON entries against `per_sequence.csv`, but does not fully prove equality.

For every productive-safe cell, compare the canonical atlas embedded maps against all positive rows in `per_sequence.csv` and fail closed on:

```text
missing positive sequence
extra positive sequence
n_neg mismatch
n_gt mismatch
sum(per-sequence n_neg) != n_neg_captured
sum(per-sequence n_gt) != n_gt_captured
n_sequences_with_neg mismatch
```

Sparse absence may represent zero, but an extra positive row must never pass silently.

### R1.3 Per-grid mask invariance assertion — required

Before using `.first()` or union/max aggregation, assert that all coordinates sharing `(grammar, grid_id, mask_sha256)` agree on:

```text
n_neg_captured
n_gt_captured
n_sequences_with_neg
per_sequence_neg map
per_sequence_gt map
```

Any disagreement is `reconciliation_failed`, not a value to resolve by first/max.

### R1.4 Worst-sequence naming — required

The current metric takes the minimum only among sequences with positive support. Do not call that unqualified `worst_sequence_productive_capacity`.

Use explicit names such as:

```text
min_positive_sequence_n_neg
min_positive_sequence
```

Optionally also emit the all-seven-sequence minimum, but keep the distinction explicit.

### R1.5 Capacity concentration answer — required

Directly answer whether productive capacity is concentrated in a few **per-grid mask units**. Add deterministic concentration summaries using `mask_n_neg`, not duplicate-inflated `total_n_neg`, at least:

```text
number of productive per-grid mask units
distribution of mask_n_neg
top-1 / top-3 / top-5 share of summed per-grid-mask capacity
maximum mask_n_neg
```

State the denominator and retain global mask-string results as diagnostic only.

### R1.6 Machine-table completeness — minor

Populate `max_full_neighborhood_safe_radius=0` for the G1 component instead of leaving the component CSV field blank.

## R1 acceptance

```text
- original input hashes unchanged
- no evaluator rerun or modification
- headline 154 = 1 + 153 + 0 still reconciles
- bidirectional per-sequence equality PASS
- per-grid mask invariance PASS
- multi-seq answer explicitly reports 8 per-grid / 4 global diagnostic
- capacity concentration explicitly reported
- dual-margin results either reproduce exactly or any difference is explained
- G7 remains contract-gap only
- Terminal B / production / ledger unchanged
```

After R1, update this thread to `awaiting PR review`; do not self-accept the bounded verdict and do not open the PR automatically.

## Current descriptive result — provisional until R1

```text
154 PS = 1 G1 + 153 G2 + 0 G3
142 single-seq · 12 multi-seq (AND)
143/154 coords on multi-coord per-grid mask plateaus
26 components · 12 single-cell-width strips · 0 genuine 2D-thick
full_neighborhood_safe_radius≥1: 0/154
nearest_unsafe_distance=1 on all PS
G7: not_derivable_from_current_artifact_contract
```

These values are provisional only in the sense that the evidence packet still needs the R1 reconciliation and unit corrections; the review has not identified evidence that currently overturns the zero-thickness headline.

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

Terminal B, production defaults/presets, and evidence-ledger status remain unchanged.

## Must not (held)

- Rewrite or rerun the Q4.5 evaluator.
- Rebuild missing runtime atlases.
- Change lattices, thresholds, signals, labels, or unresolved policy.
- Join `P::` operands to the `S::` atom atlas by ID.
- Treat `mask_sha256` as a global primary key.
- Use sparse `per_sequence.csv` absence as zero support without explicit map reconciliation.
- Use quotient `region_id` as coordinate identity.
- Add G4–G7 enumeration or a generic Boolean engine.
- Start region-LOO, ranking/assignment, online sweeps, hook, or production work.
- Promote evidence or change terminal B.

## History

- 2026-07-10: coverage audit reconnaissance closed; thread opened; next = T0.
- 2026-07-10: PR #93 merged; T0 split into preflight then execution.
- 2026-07-10: T0-A completed at `f1981c12`; schema/derivability reviewed.
- 2026-07-10: T0-A accepted; T0-B authorized with outputs 1–6 plus fail-closed G7 contract-gap report.
- 2026-07-10: T0-B executed at `7b54f5c2`; evidence + note landed.
- 2026-07-10: T0-B review requested bounded R1 corrections; verdict remains unaccepted and no PR is authorized yet.
