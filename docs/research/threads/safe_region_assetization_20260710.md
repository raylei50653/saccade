---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A-R1 contract correction delivered against review blockers CR1–CR5. Current sole task is **R0-A-R1 completed — awaiting chat-side re-review**. R0-B, R1 asset generation, grammar extension, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-A-R1 completed — awaiting chat-side re-review** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A packet | original at `762adf9a`; **R0-A-R1 correction delivered** (same note path) |
| Chat-side review | **R0-A was CHANGES_REQUESTED**; R0-A-R1 **awaiting re-review** (not self-accepted) |
| R0-B final contract | **NOT AUTHORIZED** |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| R2 grammar asset extension | **CONDITIONAL** after accepted R0/R1 and an asset-increment hypothesis |
| R3 transfer qualification | **CONDITIONAL** on selected A1 assets |
| R4 intervention qualification | **CONDITIONAL** on selected A2 assets |
| Current maturity | **A0 retained** |
| Occ-exit conditional modeling | **PARKED** future RegionAsset producer/consumer |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

The research object is a reusable `RegionAsset`, not a rule and not a coverage-table cell:

```text
producer semantics
→ parameter coordinates
→ stable region identity
→ geometry
→ productive capacity
→ applicability geometry
→ transfer qualification
→ action contract
```

Hard separation remains:

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

No assetization step may manufacture causal, transfer, or online evidence absent from the source artifacts.

## Read first

1. [R0-A / R0-A-R1 preflight packet](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
2. [Closed A0/T0 thread](composition_grammar_safe_region.md)
3. [T0 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
4. [T0 region interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)
5. [Superseded grammar-coverage design](composition_grammar_coverage_program_20260710.md)

## Accepted A0 baseline

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
142 single-sequence · 12 multi-sequence
34 productive per-grid mask units
26 coordinate components
full_neighborhood_safe_radius >= 1: 0/154
G3 Hard OR: registered-lattice NULL_RESULT
terminal B: isolated_safe_points_only
```

This baseline is descriptive only. It is not transferable, actionable, or production-ready.

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight
  R0-A-R1 correction  ← delivered; awaiting re-review
  R0-B final accepted contract

R1 G1–G3 deterministic asset conversion
R2 grammar-by-grammar asset extension
R3 transfer qualification toward A2
R4 intervention qualification toward A3
A4 separate production approval
```

## R0-A review result (historical)

### Pass (retained)

- sealed Q4.5/T0 provenance and live-tree drift handling;
- distinct grains; ordinal component IDs invalid; per-grid mask primary;
- semantic ≠ mask equivalence; G3 first-class null required;
- A0–A4 maturity and action firewalls;
- no evaluator rerun required for derivable A1 core.

### Blocking corrections (CR1–CR5) — addressed in R0-A-R1

| ID | Issue | R0-A-R1 fix |
|:--|:--|:--|
| CR1 | Content IDs scoped through whole-pack `asset_set_id` | Split `truth_context_id` / `pack_id` / content IDs / optional `evidence_record_id` |
| CR2 | Additive component capacity | Distribution over members; sum retracted; event-union `BLOCKED_BY_ARTIFACT` |
| CR3 | Sequence union as applicability | Incidence + union + intersection + min/max; union ≠ A2 |
| CR4 | G3 null fake concrete semantic tree | Search-domain null; `semantic_definition_id` nullable; count names split |
| CR5 | Region↔mask JSON list authority | Coordinate FKs authoritative; optional derived `region_mask_link` |

## Current step — R0-A-R1 completed — awaiting chat-side re-review

**Working branch:** `research/composition-grammar-coverage-program`

### Deliverable (same path, revised in place)

```text
docs/modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md
```

### Correction summary

```text
identity: content IDs independent of pack_id / unrelated grammar / schema version
capacity: non-additive member distributions; event-union blocked
sequence: incidence + union + intersection; descriptive only at A1
G3 null: search_domain_id; semantic_definition_id NULL; n_non_null vs n_null_records
region↔mask: coordinate FKs authoritative
R1 still: conditional deterministic conversion after R0-B only
maturity: A0
```

### Not authorized

```text
R0-A / R0-A-R1 research acceptance (chat-side only)
R0-B final contract
R1 asset file generation
A0→A1 maturity promotion
evaluator rerun / modification
G4–G7, LOO, shadow, hooks, presets, ledger promotion
```

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset pack files;
- implement G4–G7;
- implement generic runtime/framework code;
- infer G7 roles;
- add LOO, shadow, hook, preset, or production work;
- change terminal B or maturity A0;
- promote to evidence ledger;
- open an R1 implementation PR;
- self-accept R0-A-R1 or authorize R0-B/R1 from this thread alone.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94 merged; G1–G3 retained as A0 baseline.
- 2026-07-10: Safe-Region Assetization Program opened; R0-A authorized.
- 2026-07-10: R0-A delivered at `762adf9a`; artifact hashes and document checks passed.
- 2026-07-10: chat-side review **CHANGES_REQUESTED** (CR1–CR5); sole active → R0-A-R1.
- 2026-07-10: R0-A-R1 contract correction delivered in place; **awaiting chat-side re-review** (not accepted; R0-B/R1 unauthorized; A0 retained).
