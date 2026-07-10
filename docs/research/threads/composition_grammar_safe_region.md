---
doc-status: closed
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
closed: 2026-07-10
---

# composition grammar × safe-region geometry

> **One-line:** T0-A/T0-B/R1 **closed**. Bounded verdict accepted: existing registered Q4.5 atlas has **0/154** productive-safe coordinates with full-neighborhood radius ≥1. Terminal B retained. PR #94 is engineering-only merge work; no further task is authorized from this thread.

## Final status

| Item | Status |
|:--|:--|
| Coverage audit | **CLOSED** |
| T0-A preflight | **ACCEPTED** |
| T0-B interpretation | **ACCEPTED after R1** |
| Q4.5 terminal | **B** `isolated_safe_points_only` |
| Production preset | **unchanged** |
| Evidence ledger | **not promoted** |
| PR | **#94 open / mergeable**; engineering merge separate |
| Next authorized work | **none** |

## Read first

1. [T0-B final interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)
2. [T0-A artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
3. [Coverage audit](../../modules/semantic/research/composition_grammar_safe_region_coverage_audit_20260710.md)
4. [Stage 2 final](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md)

## Final evidence state

```text
154 productive-safe coordinates
  = 1 G1 atom
  + 153 G2 AND
  + 0 G3 OR

142 single-sequence
12 multi-sequence
12 multi-seq coordinates
  → 8 primary per-registered-grid masks
  → 4 global mask strings (diagnostic only)

34 productive per-grid mask units
sum(mask_n_neg) = 48
top-1 / top-3 / top-5 capacity shares
  = 8.3% / 22.9% / 33.3%

143/154 coordinates on multi-coordinate mask plateaus
26 components
12 single-cell-width strips
0 genuine 2D-thick components

nearest_unsafe_distance = 1 for all 154
full_neighborhood_safe_radius >= 1 = 0/154

G7 = not_derivable_from_current_artifact_contract
```

Validation gates:

```text
input hashes unchanged
headline reconciliation PASS
bidirectional per-sequence equality PASS
per-grid mask invariance PASS
synthetic dual-margin checks PASS
no evaluator rerun or modification
```

## Accepted bounded verdict

> Within the existing registered Q4.5 G1–G3 lattices, productive-safe support is predominantly explained by threshold-coordinate mask plateaus, single-sequence support, and thin or edge-touching components. Under the declared conservative dual-margin policy, no registered full-neighborhood thickness is observed. The atlas therefore remains `isolated_safe_points_only`, and Stage 2 terminal B is retained.

Maximum promotion:

```text
accepted bounded descriptive closure of the existing-atlas region question
```

Not promoted to:

```text
formal or portable safe region
online parameter-region retention
productive reject policy
production candidate
G7 equivalence
new grammar necessity
global threshold-path falsification
```

## Research decision

Close the current registered threshold-region line.

Do **not** authorize from this thread:

```text
T1 evaluator emit
region-level LOO
restricted G7 implementation
online region sweep
hook or production preset change
```

Why:

- no full-neighborhood thickness exists to justify region-transfer work;
- the bounded descriptive evidence is already committed;
- G7 is a missing semantic contract, not evidence that the grammar is necessary;
- more infrastructure would not close a stronger claim on this substrate.

## Reopen conditions

A future reopen requires a separate explicit contract backed by at least one of:

```text
new signal-family evidence with a declared falsifier
new hook placement or decision substrate
valid NOT / necessary / support role semantics for G7
a newly registered atlas with nonzero multi-sequence thickness
```

No reopen is authorized here.

## PR boundary

PR #94 contains the preflight, derivation script, evidence pack, R1 corrections, and accepted bounded documentation state.

```text
research acceptance: complete
engineering merge: separate repository action
production promotion: blocked
```

## History

- 2026-07-10: coverage audit closed; T0 thread opened.
- 2026-07-10: T0-A completed and accepted.
- 2026-07-10: T0-B executed at `7b54f5c2`.
- 2026-07-10: review requested R1 at `32ecd242`.
- 2026-07-10: R1 completed at `c0bac5cc`; all correction gates passed.
- 2026-07-10: PR #94 opened; bounded verdict manually accepted; thread closed.
