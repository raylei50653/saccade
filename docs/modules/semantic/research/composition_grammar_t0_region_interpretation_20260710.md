# T0-B Existing Atlas Region Interpretation Pack

<!-- doc-status: closed -->
<!-- doc-promotion: none; not evidence_ledger; bounded descriptive geometry only -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for T0-B derived geometry interpretation -->

**Task:** T0-B · revision **T0-B-R1**  
**Branch:** `research/m-b1-5-t0-region-interpretation`  
**Authorize:** T0-B `4c347281` · R1 dispatch `32ecd242` · base T0-A tip `f1981c12`  
**Execution:** `7b54f5c2` · R1 `c0bac5cc`  
**PR:** #94 — engineering merge remains separate from research acceptance  
**Thread:** [composition_grammar_safe_region.md](../../../research/threads/closed/composition_grammar_safe_region.md)  
**Preflight:** [composition_grammar_t0_artifact_preflight_20260710.md](composition_grammar_t0_artifact_preflight_20260710.md)

> Descriptive geometry of the **existing registered Q4.5 atlas**.  
> **Accepted bounded conclusion:** no registered full-neighborhood thickness was observed.  
> **Not** portable safe-region proof · **not** production candidate · **not** G7 equivalence · **not** global threshold-path falsification.

---

## 1. Provenance and gates

| Item | Value |
|:--|:--|
| Input study | `out/signal_study/m_b1_5_stage2_q45_20260710/` |
| Derivation script | `scripts/tools/analyze_m_b1_5_t0_region_interpretation.py` |
| Runtime outputs | `out/signal_study/m_b1_5_t0_region_interpretation_20260710/` |
| Committed evidence | [evidence/m_b1_5_t0_region_interpretation_20260710/](evidence/m_b1_5_t0_region_interpretation_20260710/) |
| Input hashes | match Q4.5 manifest; unchanged through R1 |
| Headline reconciliation | **PASS** — 154 = 1 G1 + 153 G2 + 0 G3 |
| Per-sequence equality | **PASS** — bidirectional n_neg/n_gt/maps/totals |
| Per-grid mask invariance | **PASS** before quotient collapse |
| Synthetic dual-margin checks | **PASS** |
| Evaluator rerun / modification | **none** |

```text
Terminal B:      isolated_safe_points_only (retained)
Production:      unchanged
evidence_ledger: not promoted
```

### Fixed units

```text
coordinate = registered threshold-index cell within one grid
mask       = mask_sha256 within one registered grid (primary quotient)
component  = bilateral G1 / 4-neighbor G2 connected PS set within one grid
```

Never join `S::` and `P::` by atom ID. Never use global mask-string collapse as the primary denominator.

### Fixed dual-margin policy

| Metric | Definition |
|:--|:--|
| `nearest_unsafe_distance` | same-grid graph distance to a registered non-productive-safe coordinate |
| `distance_to_lattice_edge` | minimum steps to registered lattice boundary |
| `full_neighborhood_safe_radius` | G1 bilateral interval; G2 Manhattan / repeated 4-neighbor erosion |
| edge policy | missing off-lattice neighbor fails the full-neighborhood condition |

Edge-censored distance is not region thickness.

---

## 2. Deliverables

| # | Output | Artifact |
|:--|:--|:--|
| 1 | Raw-coordinate area | `grammar_area_summary.csv` |
| 2 | Per-grid unique-mask area | `unique_mask_summary.csv` |
| 3 | Productive capacity | `productive_capacity.csv` · `productive_capacity_by_per_grid_mask.csv` |
| 4 | Cross-sequence support | `cross_sequence_productive_support.csv` |
| 5 | Component geometry | `component_geometry.csv` |
| 6 | Dual boundary margin | `boundary_margin.csv` |
| 7 | G7 contract gap | `g7_contract_gap.json` · diagnostic `non_g7_mask_overlap.json` |

---

## 3. Accepted descriptive facts

| Quantity | Value |
|:--|:--|
| Productive-safe coordinates | **154** = 1 G1 + 153 G2 AND + 0 G3 OR |
| Single-sequence support | **142** |
| Multi-sequence support | **12** |
| Multi-seq primary per-grid masks | **8** across four registered grids |
| Multi-seq global mask strings | **4**, diagnostic only |
| Productive per-grid mask units | **34** |
| Sum of per-grid `mask_n_neg` | **48** |
| Capacity top-1 / top-3 / top-5 shares | **8.3% / 22.9% / 33.3%** |
| Coordinates on multi-coordinate mask plateaus | **143 / 154** |
| Plateau masks with width > 1 | **23** |
| Coordinate components | **26** |
| Single-cell-width strips | **12** |
| Genuine 2D-thick components | **0** |
| Coordinates with `full_neighborhood_safe_radius ≥ 1` | **0 / 154** |
| Coordinates with nearest-unsafe > 0 and radius 0 | **154 / 154** |
| Coordinates touching lattice edge | **85 / 154** |

### Area

| Grammar | Registered coordinates | Productive-safe | Ratio |
|:--|--:|--:|--:|
| G1 atom | 870 | 1 | 0.001149 |
| G2 AND | 17640 | 153 | 0.008673 |
| G3 OR | 17640 | 0 | 0 |

Raw counts across G1 and G2 are not directly comparable without lattice normalization.

### Capacity

- Coordinate `n_neg_captured`: 142 cells capture one negative; maximum is four.
- Per-grid mask capacity distribution: 1→26, 2→4, 3→2, 4→2.
- No single per-grid mask unit dominates capacity; top-1 share is 8.3%.
- Sequence minima are explicitly named `min_positive_sequence*`; they are not all-seven-sequence worst cases.

### Components and margins

Largest components are axis-degenerate or thin, including 1×19 and 18×1 strips. Some components span both axes but have no full 4-neighbor interior.

```text
for every productive-safe coordinate:
  nearest_unsafe_distance = 1
  full_neighborhood_safe_radius = 0
```

Thus all productive-safe coordinates are immediately adjacent to non-PS support, and none has a complete safe neighborhood of radius one.

---

## 4. G7 contract gap

```text
status: not_derivable_from_current_artifact_contract
missing:
  - logical NOT / complement identity
  - necessary-envelope operand role
  - support operand role
  - N/P parameterization
maximum_claim:
  existing G1/G2 mask-string overlap only; not G7 equivalence
```

No G7 implementation or semantic-equivalence result is authorized by this study.

---

## 5. Accepted bounded verdict

**Accepted at the PR #94 review gate on 2026-07-10.**

> Within the existing registered Q4.5 G1–G3 lattices, the 154 productive-safe coordinates are predominantly explained by threshold-coordinate mask plateaus, single-sequence support, and thin or edge-touching components. Under the declared conservative dual-margin policy, no registered full-neighborhood thickness is observed (`full_neighborhood_safe_radius ≥ 1`: 0/154). The atlas therefore remains `isolated_safe_points_only`, and Stage 2 terminal B is retained.

Maximum promotion:

```text
accepted bounded descriptive closure of the existing-atlas region question
```

This does **not** establish:

```text
formal or portable safe region
online parameter-region retention
productive reject policy
production candidate
G7 equivalence
new-grammar necessity
global threshold-path falsification
```

---

## 6. Research decision

The current registered threshold-region line is **closed**.

Not authorized from this evidence:

```text
T1 evaluator emit work
region-level LOO design
restricted G7 implementation
online region sweep
production hook or preset change
```

Reasoning:

- no non-trivial full-neighborhood thickness exists to motivate region-transfer validation;
- the required descriptive outputs already exist in a committed evidence pack;
- G7 remains a contract/schema gap rather than an observed necessary grammar extension;
- additional tooling would not close a stronger hypothesis on the current substrate.

Possible future reopen requires a new explicit contract, such as:

```text
new signal-family evidence with declared falsifier
new hook placement / decision substrate
valid G7 NOT and operand-role semantics
or a newly registered atlas exhibiting nonzero multi-sequence thickness
```

No such reopen is authorized here.

---

## 7. Reproduce

```bash
uv run python scripts/tools/analyze_m_b1_5_t0_region_interpretation.py \
  --study out/signal_study/m_b1_5_stage2_q45_20260710 \
  --out out/signal_study/m_b1_5_t0_region_interpretation_20260710
```

Fails closed if full atlases are missing or hashes differ from the Q4.5 manifest.

---

## 8. R1 correction record

| Finding | Resolution |
|:--|:--|
| Multi-seq unit mislabeled | **8** primary per-grid masks; **4** global strings diagnostic |
| One-way sequence check | bidirectional n_neg/n_gt/maps/totals equality |
| Silent mask collapse | invariance assertion before collapse |
| Ambiguous worst naming | renamed `min_positive_sequence*` |
| Missing concentration | top-1/top-3/top-5 on per-grid `mask_n_neg` |
| Blank G1 radius | filled zero |

Engineering merge of PR #94 remains a separate repository action from this accepted bounded research conclusion.
