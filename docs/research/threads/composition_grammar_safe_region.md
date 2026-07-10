---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# composition grammar × safe-region geometry

> **One-line:** T0-A preflight **ACCEPTED**. Terminal B **unchanged**. Current authorized work = **T0-B Existing Atlas Region Interpretation Pack** on outputs 1–6 plus a fail-closed G7 contract-gap report; no evaluator rerun or grammar expansion.

## Status

| Item | Status |
|:--|:--|
| Coverage audit (recon) | **CLOSED** — fact note linked below |
| Q4.5 terminal | **B** `isolated_safe_points_only` (unchanged) |
| Production preset | **unchanged** |
| Signal family expansion | **not authorized** by this thread |
| Unrestricted Boolean / 3+ atoms | **forbidden** |
| T0-A artifact preflight | **ACCEPTED** — schema/derivability boundary reviewed |
| T0-A report | [composition_grammar_t0_artifact_preflight_20260710.md](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md) |
| Current authorized work | **T0-B** bounded derived interpretation on confirmed runtime artifacts |
| G7 item | **observation-gap output only** — true `¬N∧P` audit is not derivable |
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

## Read first

1. **[T0-A preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)** — accepted schema/key/derivability contract
2. **[Coverage audit](../../modules/semantic/research/composition_grammar_safe_region_coverage_audit_20260710.md)** — T0 research questions and bounded verdicts
3. Stage 2 canonical: [m_b1_5_stage2_d_online_final_20260710.md](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md) · [entry contract](../../modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md)
4. Closed Stage 1/2 history: [m_b1_online_hook_20260709.md](m_b1_online_hook_20260709.md)
5. Signal table contract: [signal_table_schema.md](../eval/signal_table_schema.md)

Truth base:

```text
Q4.5 evaluator:  PR #89 head 6df1739b · merge 234f9f59
Docs stack:      PR #90 consolidation 51b9c73e
Integration:     PR #93 merge 8f7a3700
T0-A accepted:   f8cfff56 + f1981c12
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

## Current step — T0-B Existing Atlas Region Interpretation Pack

**Branch:** `research/m-b1-5-t0-region-interpretation`  
**Base:** T0-A reviewed tip `f1981c12`

### Objective

Interpret the existing 154 productive-safe coordinates using the confirmed artifact schema, without rerunning the Q4.5 evaluator or changing the registered search space.

Answer whether the observed productivity is explained by:

```text
threshold-coordinate duplication
single-sequence support
axis-degenerate components / thin strips
edge-censored apparent distance
or genuine full-neighborhood thickness
```

### Required inputs

Read-only runtime inputs:

```text
out/signal_study/m_b1_5_stage2_q45_20260710/
  atom_atlas.parquet
  pairwise_and_atlas.parquet
  pairwise_or_atlas.parquet
  region_stability.csv          # quotient/topology cross-check only
  per_sequence.csv              # productive-cell cross-check only
  threshold_registry.json
  summary.json
  manifest.json
```

Fail closed if the full atlases are missing or their hashes do not match the manifest. Do not rebuild them.

### Fixed derivation units

```text
G1 grid:
  one (feature, direction) registered 1D lattice
  coordinate = thr_index 0..86

G2/G3 grid:
  one registered axis-pair
  coordinate = (thr_index_a, thr_index_b), each 0..20

Connectivity:
  G1 = immediate bilateral 1D adjacency
  G2/G3 = 4-neighbor adjacency
  never connect coordinates across registered grids

Mask quotient:
  primary scope = per registered grid
  cross-grid equal mask strings may be described separately,
  but must not replace the primary denominator or merge components
```

## Seven T0-B deliverables

### 1. Raw-coordinate area

Emit safe and productive-safe coordinate counts and ratios:

```text
per registered grid
per grammar as a coordinate-weighted aggregate
```

Denominator is the complete registered coordinate count for that grid. Do not compare raw G1 and G2 cell counts without normalization context.

### 2. Unique-mask area

Emit per-grid unique-mask safe/productive counts and ratios.

Grammar aggregate may use the sum of per-grid unique-mask denominators (`grid_scoped_micro`), but must not globally deduplicate mask strings. Optional global mask-string overlap is diagnostic only.

### 3. Productive capacity distribution

For the 154 productive-safe coordinates and their per-grid mask quotients, report:

```text
n_neg_captured distribution
negative capture rate
capacity by grammar / grid / component
capacity by sequence-support count
coordinates per productive mask
capacity concentration by productive mask
```

Do not multiply capacity merely because one prediction mask appears at several threshold coordinates.

### 4. Cross-sequence productive-support geometry

Use atlas embedded per-sequence JSON as the canonical cell-level source. Cross-check productive cells against `per_sequence.csv`; fail closed on mismatches.

Report coordinate and per-grid-mask units separately:

```text
n_sequences_with_productive_support
productive sequences and counts
single-sequence vs multi-sequence support
worst-sequence productive capacity
max-sequence share
12 known multi-seq AND coordinates → number of per-grid unique masks
```

Do not use pooled GT0 intersection as the headline.

### 5. Component shape / axis degeneracy

Reconstruct coordinate components from complete atlas grids, not from `region_stability.region_id`.

Report at least:

```text
component coordinate size
per-grid unique-mask count
axis spans / bounding-box dimensions
active-axis count
isolated / 1D interval / row strip / column strip
single-cell-width strip / genuine 2D-thick component
```

Use `region_stability.csv` only as a quotient-level reconciliation check.

### 6. Dual boundary margin

Emit both metrics; never collapse them into one `margin`.

#### `nearest_unsafe_distance`

Shortest registered-lattice graph distance from a productive-safe coordinate to a registered non-productive-safe coordinate in the same grid.

Also emit:

```text
distance_to_lattice_edge
nearest_unsafe_edge_censored
```

An edge-censored value must not be interpreted as robust thickness.

#### `full_neighborhood_safe_radius`

Largest integer radius whose entire required neighborhood:

```text
G1: bilateral interval
G2: Manhattan / repeated 4-neighbor erosion
```

exists inside the registered lattice and remains productive-safe.

**Conservative edge policy:** missing off-lattice neighbors fail the full-neighborhood condition. A coordinate touching a lattice edge therefore has radius 0. This prevents lattice truncation from creating artificial thickness.

Required synthetic checks:

```text
isolated point → radius 0
one-cell-wide strip → radius 0
3×3 filled block center → radius >= 1
diagonal-only cells are disconnected
edge-touching strip does not gain artificial radius
nearest unsafe may be > 0 while full radius = 0
```

### 7. G7 contract-gap report

Do not implement or claim a G7 semantic-equivalence audit.

Emit a bounded machine-readable result:

```text
status: not_derivable_from_current_artifact_contract
missing:
  - logical NOT / complement predicate identity
  - necessary-envelope operand role
  - support operand role
  - parameterization linking N and P
maximum_claim:
  existing G1/G2 mask-string overlap only; not G7 equivalence
```

An optional overlap table may compare existing mask strings, but its filename and labels must state `non_g7_mask_overlap`.

## Machine outputs

Create a new derived study directory:

```text
out/signal_study/m_b1_5_t0_region_interpretation_<timestamp>/
  summary.json
  grammar_area_summary.csv
  unique_mask_summary.csv
  productive_capacity.csv
  cross_sequence_productive_support.csv
  component_geometry.csv
  boundary_margin.csv
  g7_contract_gap.json
  artifact_reconciliation.json
  manifest.json
```

Suggested implementation:

```text
scripts/tools/analyze_m_b1_5_t0_region_interpretation.py
```

The script must be deterministic, take explicit input/output paths, never mutate inputs, record input SHA256 values, and fail closed on missing required fields or headline reconciliation failure.

## Reconciliation gates

Required before interpretation:

```text
productive-safe total = 154
G1 singleton = 1
G2 AND = 153
G3 OR = 0
registered G1 rows = 870
registered G2 rows = 17640
registered G3 rows = 17640
```

Any mismatch yields `reconciliation_failed`; do not replace the canonical headline silently.

## Research note and thread update

Create:

```text
docs/modules/semantic/research/
composition_grammar_t0_region_interpretation_20260710.md
```

The note must answer the original T0 acceptance questions, clearly separate coordinate and mask units, declare edge censoring, preserve the unresolved firewall, and provide only a bounded verdict candidate.

After execution, update this thread with:

```text
T0-B status
artifact/note pointers
reconciled headline
observation gaps
bounded verdict candidate
next state = awaiting T0-B review / PR review
```

Do not authorize the next task in the execution commit.

## Acceptance boundary

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

Terminal B, production defaults/presets, and evidence-ledger status remain unchanged during execution.

## Must not

- Rewrite or rerun the Q4.5 evaluator.
- Rebuild missing runtime atlases.
- Change lattices, thresholds, signals, labels, or unresolved policy.
- Join `P::` operands to the `S::` atom atlas by ID.
- Treat `mask_sha256` as a global primary key.
- Use sparse `per_sequence.csv` absence as zero support.
- Use quotient `region_id` as coordinate identity.
- Add G4–G7 enumeration or a generic Boolean engine.
- Start region-LOO, ranking/assignment, online sweeps, hook, or production work.
- Promote evidence or change terminal B.

## Next gate

```text
T0-B execution
→ evidence packet + research note + thread headline
→ engineering review / PR review
→ bounded research acceptance
→ only then decide: close line / minimal emit / region-LOO design / restricted G7 contract
```

## History

- 2026-07-10: coverage audit reconnaissance closed; thread opened; next = T0.
- 2026-07-10: PR #93 merged; T0 split into preflight then execution.
- 2026-07-10: T0-A completed at `f1981c12`; schema/derivability reviewed.
- 2026-07-10: T0-A accepted; T0-B authorized on `research/m-b1-5-t0-region-interpretation` with outputs 1–6 plus fail-closed G7 contract-gap report.
