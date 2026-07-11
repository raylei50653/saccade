# Composition Grammar × Safe-Region Property Coverage Audit

<!-- doc-status: closed -->
<!-- doc-promotion: none; not evidence_ledger until promoted -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for coverage/task-design conclusions -->

**Task type:** research-design reconnaissance only  
**No implementation performed**  
**Date:** 2026-07-10  
**Revision:** post-review decision-gate edition (branch truth, dual area, dual margin, G7 gate, online split)  
**Thread (navigation only):** [composition_grammar_safe_region.md](../../../research/threads/closed/composition_grammar_safe_region.md)

> Coverage / task-design conclusion for Q4.5 region geometry.  
> **Not** rule search · **not** new Boolean framework · **not** production change.  
> Q4.5 terminal B unchanged. **Do not** auto-promote to `evidence_ledger` (see thread).

---

## Document truth base (do not collapse)

| Layer | Identity | Role |
|---|---|---|
| **Q4.5 evaluator truth** | PR **#89** final head `6df1739b` · merge commit `234f9f59` | Evaluator v4 + evidence pack refresh; machine study SHAs / terminal B |
| **Canonical consolidated documentation truth** | stacked docs consolidation merge `51b9c73e` (PR **#90**) | Entry contract + Stage 2 final narrative on top of Q4.5 v4 truth |
| **Machine study** | `out/signal_study/m_b1_5_stage2_q45_20260710/` | Artifacts used for geometry inventory |
| **Workspace `main` at audit time** | `e8f30e67` (as of audit, 2026-07-10) | **Historical:** Stage 2 Q1–Q4.5 sources/docs were not yet on main; truth lived on research stack |

```text
Do not treat 51b9c73e as the original Q4.5 evaluator branch tip.
6df1739b is the Q4.5 evaluator/evidence truth.
51b9c73e is the later stacked documentation consolidation merge
that fast-forwards/repoints branch narrative onto Q4.5 v4 truth.
```

**Historical note (audit-time):** Stage 2 Q1–Q4.5 evaluator/docs were **not on `main`** when this audit was written; machine artifacts already existed on disk under `out/signal_study/`.  
**Integration (PR #93):** the accepted stack lands on `main` without changing evaluator or documentation **truth identities** (still #89 / #90 / coverage-gate conclusions above).

---

## 11.1 Research Question Restatement

Under a **fixed** frozen signal family and the **locked** Stage 2 \(D_{\text{online}}\) decision cohort, map:

\[
\text{Composition Grammar}\times\text{Parameterized Predicates}
\rightarrow
\text{GT-safe / Productive Region Geometry}
\]

then measure contraction under:

\[
\text{V1 pooled offline}
\rightarrow
\text{V2 per-sequence geometry}
\rightarrow
\text{V3 nested LOO transfer}
\rightarrow
\text{V4 online intervention}
\]

The object of study is **region geometry** (points, ridges, thick intervals, sequence islands, non-transferable / online-collapsed regions), **not** best-rule search or FP maximization.

**Central judgment of this audit (held after review):**

> The primary Q4.5 gap is **region observation** and **validation contraction**,  
> not missing signals and not unrestricted Boolean grammar.

---

## 11.2 Fixed Boundary Confirmation

| Boundary | Status | Evidence |
|---|---|---|
| Signal family | **Fixed** — 5 frozen: `score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean` | `ORDERED_SIGNALS` / `threshold_registry.json` |
| No new signals / ranking features | **Held** in Q4.5 mainline | competition columns demoted |
| Competition / assignment features | **Untrusted** | `assignment_group_key_status=invalid_frame_provenance` |
| Primary substrate | **Locked** `resolved ∧ baseline_selected` | neg=23, GT-protect=64, total=87 |
| Unresolved selected | **21** — blocks candidacy, not defaulted negative | `cohort_summary.json` |
| Hook / production preset | **Unchanged** | Stage 1 closed; Stage 2 terminal B |
| Grammar scope | Restricted: plan §18.1–18.7; Q4.5 implements only singleton + pairwise AND/OR | `three_plus_atom_combos_forbidden` |

---

## 11.3 Existing Data and Ownership Map

| Layer | Owner | Input | Output | Trusted? | Limitation |
|---|---|---|---|---|---|
| Event facts | Stage 1 B-audit + Q1–Q3 join (`d_online_stage2.py`) | hook events + A1 MOT + GT | `d_online_events.parquet` (244) | **Yes** (with incomplete-label policy) | 39 unresolved / 4 ambiguous; frame≠MOT |
| Decision cohort | Q4 lock (`d_online_stage2_q4.py`) | events | primary 87 + selected unresolved 21 | **Yes** | primary excludes unresolved |
| Predicate lattice | Q4.5 registry | primary signals | unique-boundary single; q05 pairwise | **Yes** for frozen signals | lattices not cross-grammar comparable |
| Grammar enumeration | Q4.5 atlas | lattice | atom / AND / OR atlases | **Yes** for G1–G3 **within registered lattices** | G4–G7 absent; not continuous domain |
| Region topology | `classify_region_stability` v4 | productive-safe cells | `region_stability.csv` | **Yes** for interior boolean | dual margin metrics not emitted |
| Per-seq audit | `region_metrics` + `per_sequence.csv` | masks × seq | counts / JSON | **Partial** | GT0∩≈pooled; productive multi-seq under-expressed |
| LOO audit | nested LOSO v3/v4 | train rebuild | clause summary | **Yes** for exact-absolute clause | not region/quantile transfer |
| Online audit | Stage 1 A/B | freeze OR-tail policy | baudit + e2e | **Yes** for **selected-policy** freeze only | parameter-region retention not measured |

---

## Phase E0 — Contract Reconstruction (definition table)

| Term | Canonical definition | Existing owner | Ambiguity / fixed dual view |
|---|---|---|---|
| **Safe point** | Lattice coordinate \(\theta\) with `gt_hurt==0` on resolved primary **and** `unknown_capture==0` → `observed_safe_point` | `region_metrics` | Unit is **threshold coordinate + prediction mask**, not event subset alone. Same mask can span many thr (`mask_sha256`). |
| **Productive-safe** | Safe point with `n_neg>0` → `productive_safe_point` (unknown must be 0) | same | On primary labels, support = neg+gt ⇒ **safe-with-support ≡ productive**. Vacuous empty cells are `empty_region`, not observed_safe. Floor is effectively `FP_removed>0` with unknown block. |
| **Safe area** | Set of GT0 coordinates in declared lattice \(\Theta_G\) | counts only | **Emit dual ratios (not owner either/or):** `coordinate_safe_area_ratio` (thickness under thr perturbation) **and** `unique_mask_safe_ratio` (distinct event-selection outcomes). Denominators are lattice-specific; never compare raw cell counts across grammar dimensions. |
| **Interior** | Coord in productive-safe **coordinate union** with full bilateral (1D) / 4-neigh (2D) neighbors also productive-safe | `classify_region_stability` v4 | Binary interior flag. Same-mask plateau is diagnostic only. |
| **Boundary margin** | Tolerance to thr perturbation — **two metrics, not one** | neighbor flags, plateau/component sizes | (1) `nearest_unsafe_distance` — min grid steps to any unsafe coord; (2) `full_neighborhood_safe_radius` — max radius that remains safe in **all** required neighbor directions (true thickness; aligns with interior erosion). Thin strip can have (1)>0 while (2)=0. |
| **Per-seq / cross-seq geometry** | For **resolved labeled GT-hurt only**, at the **same registered coordinate**: pooled GT0 iff every sequence-specific GT_hurt is zero | per-seq counts | That identity is **not** productive overlap, unresolved safety overlap, multi-seq negative capacity, LOO transfer, or equal per-seq lattices. **Primary missing R4 name:** **Cross-sequence productive-support geometry** (`n_sequences_with_productive_support`, productive-support intersection, sequence-specific productive islands, worst-sequence productive capacity). |
| **LOO transfer** | Train rebuild lattice → select productive-safe → freeze absolute clause → holdout | `nested_loso_portability_audit` | Explicitly **exact absolute thr@12dp clause repeatability**, not quantile/region portability. Deletion LOO / fixed partition **forbidden** as portability. |
| **Online retention** | Split into two questions | Stage 1 freeze arm | **A. Selected-policy online retention** (freeze OR-tail): supported, 0 triggered. **B. Parameter-region online retention** (atlas \(\Theta\)): **not measured**. Do not read A as “whole G3 region online-collapsed.” |

---

## Phase E1 — Capability Inventory (condensed)

Legend: I=implemented · E=emitted · D=derivable · T=tested · C=used in terminal claim

These five columns **must not be collapsed**. Code presence ≠ research question answered.

| Capability | I | E | D | T | C | Limitation |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Singleton lattice (G1) | ✓ | ✓ | — | ✓ | ✓ | unique boundaries; 870 rows **within registered lattice** |
| Pairwise AND (G2) | ✓ | ✓ | — | ✓ | ✓ | q05 lattice; 17640 |
| Pairwise OR (G3) | ✓ | ✓ | — | ✓ | ✓ | 0 productive-safe |
| Atom-count / k-of-n (G4) | ✗ | ✗ | ✗ | ✗ | ✗ | no evaluator |
| Family-count (G5) | ✗ | ✗ | ✗ | ✗ | ✗ | no family ownership |
| Extreme∨moderate consensus (G6) | ✗ | ✗ | partial* | ✗ | ✗ | *needs composed enum |
| ¬N∧P (G7) as new enum | ✗ | ✗ | — | ✗ | ✗ | **First:** G1/G2 mask equivalence audit |
| G7 as semantic relabel of G1/G2 | ✗ | ✗ | **likely partial** | ✗ | ✗ | envelope role may already exist as masks |
| Point GT hurt / neg capture | ✓ | ✓ | — | ✓ | ✓ | |
| Unresolved contamination | ✓ | ✓ | — | ✓ | ✓ | blocks candidacy |
| Safe / productive **counts** | ✓ | ✓ | — | ✓ | ✓ | |
| Coordinate area ratio | ✗ | ✗ | **yes** | ✗ | ✗ | denom known, not emitted |
| Unique-mask area ratio | ✗ | ✗ | **yes** | ✗ | ✗ | 15 unique AND prod masks known |
| Semantic-duplicate awareness | ✓ | ✓ | — | ✓ | partial | 122/153 prod AND are dups |
| Component topology | ✓ | ✓ | — | ✓ | ✓ | strips; max size 19 |
| Interior (coord-union) | ✓ | ✓ | — | ✓ | ✓ | **0** interiors |
| `nearest_unsafe_distance` | partial | partial | partial | partial | ✗ | neighbor GT hurts only |
| `full_neighborhood_safe_radius` | partial | partial | partial | ✓ (interior tests) | ✓ as interior=0 | binary radius∈{0,≥1} via interior; continuous radius not emitted |
| Per-seq counts | ✓ | ✓ | — | ✓ | partial | |
| Cross-seq productive-support geometry | ✗ | ✗ | **yes** | ✗ | ✗ | 12 multi-seq AND countable |
| Nested LOSO exact clause | ✓ | ✓ | — | ✓ | ✓ | 0 portable |
| Quantile/region LOO transfer | ✗ | ✗ | ✗ | ✗ | ✗ | needs new audit |
| Selected-policy online retention | ✓ | ✓ | — | ✓ | ✓ | freeze OR-tail, 0 fire |
| Parameter-region online retention | ✗ | ✗ | ✗ | ✗ | ✗ | multi-θ / parameterized hook absent |

\*G6 could be sketched from G1 extreme + G2 moderate cells but is not a registered grammar enumeration.

---

## 11.4 Grammar Coverage Matrix (G1–G7)

Status codes: `SUPPORTED` · `PARTIAL` · `DERIVABLE_FROM_EXISTING_ARTIFACTS` · `NOT_MEASURED` · `BLOCKED_BY_PROVENANCE` · `BLOCKED_BY_SUBSTRATE` · `NOT_APPLICABLE`

| Grammar | Evaluator | V1 pooled geometry | V2 cross-seq productive geometry | V3 LOO | V4 online | Research necessity now |
|---|---|---|---|---|---|---|
| **G1 Singleton** | SUPPORTED **within registered lattices** | PARTIAL (counts + 1 prod cell; dual area not emitted) | PARTIAL | SUPPORTED as exact-clause (0 portable) | selected-policy: PARTIAL (freeze thr only, 0 fire); region: NOT_MEASURED | Covered; observation gaps only |
| **G2 Pairwise AND** | SUPPORTED **within registered lattices** | PARTIAL (153 prod; thin strips; 0 interior) | PARTIAL (12 multi-seq) | same | region: NOT_MEASURED (hook has no AND) | Covered; geometry views missing |
| **G3 Hard OR** | SUPPORTED **within registered lattices** | SUPPORTED as null result (0 prod) | PARTIAL | same | selected-policy: SUPPORTED for freeze hard-OR; region: NOT_MEASURED | Covered; null is a real result |
| **G4 Atom-count** \(\sum P_i\ge k\) | NOT_MEASURED | — | — | — | NOT_APPLICABLE to current hook | **Low** — expansion; OR already null |
| **G5 Family-count** | NOT_MEASURED | — | — | — | NOT_APPLICABLE | **Deferred** — needs mechanism-family ownership |
| **G6 Extreme∨Moderate** | NOT_MEASURED | — | — | — | NOT_APPLICABLE | **Medium-low** after T0 |
| **G7 ¬N∧P** | NOT_MEASURED as enum | — | — | — | NOT_APPLICABLE | **Conditional** — only if T0/T1 show G1/G2 reinterpretation insufficient; **first step = G7 equivalence audit** |

### Scope of “fully enumerated”

```text
G1–G3 are fully enumerated WITHIN the registered Q4.5 lattices,
directions, frozen signal pairs, combinators, and unresolved firewall.

This does NOT mean:
  - continuous parameter domain coverage
  - all equivalent discretizations
  - secondary/untrusted competition features
  - 3+ atom formulas
  - G4–G7
```

### G1–G3 headline facts (canonical)

| | Single | AND | OR |
|---|--:|--:|--:|
| Lattice cells (registered) | 870 | 17640 | 17640 |
| Productive-safe coordinates | **1** | **153** | **0** |
| Unique productive masks (AND) | — | **15** | — |
| Multi-seq productive (`n_seq_neg≥2`) | 0 | **12** | 0 |
| Coord-union interior | 0 | 0 | 0 |
| Region candidates | 0 | 0 | 0 |
| Nested exact-absolute portable clauses | 0 (shared audit) | | |

G1 sole productive cell: extreme `resid_mean` high-tail (`thr_index=86`, n_neg=1, single sequence).  
G2: 141/153 productive cells are **single-sequence** neg support; heavy semantic duplication (122/153 `semantic_duplicate_mask=1`). Largest components are thin strips (e.g. 1×19, 18×1) → no 4-neigh interior.  
G3: OR always pulls GT mass under this lattice — productive-safe empty.

---

## 11.5 Region Property Coverage Matrix (R1–R6)

| Property | G1 | G2 | G3 | Evidence pointer | Missing semantics |
|---|---|---|---|---|---|
| **R1 Safe Area** | PARTIAL | PARTIAL | PARTIAL | atlas counts; registry lattice sizes | Dual ratios not emitted: `coordinate_*` and `unique_mask_*` |
| **R2 Productive Area** | PARTIAL | PARTIAL | SUPPORTED (0) | `productive_safe_point`; n_neg | Same dual ratios; capacity distribution not first-class |
| **R3 Boundary Margin** | PARTIAL | PARTIAL | NOT_APPLICABLE (empty) | interior flag; neighbors; component sizes | Must split **distance** vs **thickness**; see §R3 dual metrics |
| **R4 Cross-seq productive-support geometry** | PARTIAL | PARTIAL | PARTIAL | `per_sequence.csv`, shares | Not GT0∩ re-report; need productive multi-seq views |
| **R5 LOO Transfer** | PARTIAL | PARTIAL | PARTIAL | nested LOSO exact-absolute | **Not** region/quantile/boundary transfer |
| **R6 Online Retention** | split | split | split | Stage 1 baudit freeze | See §R6 A/B split |

### R3 dual metrics (locked language for T0)

| Metric | Definition | Thin 1-wide strip | Thick block with interior |
|---|---|---|---|
| `nearest_unsafe_distance` | Min grid distance from a safe coord to any unsafe coord | often **>0** | **>0**, usually larger |
| `full_neighborhood_safe_radius` | Max \(r\) s.t. all required lattice neighbors within radius \(r\) are safe (bilateral / 4-neigh erosion depth) | **=0** (matches `interior=0`) | **≥1** (matches interior) |

```text
Do NOT use nearest_unsafe_distance alone as “margin quality.”
A thin strip can look healthy on distance while having zero thickness.
full_neighborhood_safe_radius is the thickness metric aligned with interior.
```

### R4 precise missing observation

```text
For resolved labeled GT-hurt only, at the same registered coordinate:
  pooled GT0  ⇔  every sequence-specific GT_hurt is zero.

That is NOT new research content for R4.

Primary missing R4 = Cross-sequence productive-support geometry:
  n_sequences_with_productive_support
  productive_support intersection (multi-seq productive coordinates)
  sequence-specific productive islands
  worst-sequence productive capacity
```

### R6 split (locked)

| Question | Status | Evidence |
|---|---|---|
| **A. Selected-policy online retention** | **SUPPORTED** for freeze portable OR-tail | Stage 1 close: eligible=244, triggered=0, rejected=0 |
| **B. Parameter-region online retention** | **NOT_MEASURED** | Would need multi-θ policies or parameterized hook replay over atlas coordinates |

```text
Freeze null-fire ≠ “entire G3 / all OR lattice online-collapsed.”
It is null retention of one offline-selected absolute thr vector
outside D_online support.
```

### Validation layer status

| Layer | Status | What it answers today | Claim boundary |
|---|---|---|---|
| **V1 Pooled offline** | SUPPORTED for G1–G3 point atlas **within registered lattices** | Existence of sample-zero-GT / productive cells; null OR | Points ≠ rules; unresolved blocks candidacy |
| **V2 Cross-seq productive geometry** | PARTIAL | shares, n_seq_neg, per-seq counts | Productive multi-seq under-expressed as product |
| **V3 Nested LOO** | SUPPORTED for exact-absolute clause | Train-select-freeze-holdout clause repeatability | **Not** region portability |
| **V4 Online** | A supported (freeze); B not measured | freeze thr outside support → 0 fire | No atlas-region online productivity from offline mass |

---

## 11.6 Validation Contraction Map

```text
V1: 154 productive-safe coordinates (1 + 153 + 0)
      ↓  unique-mask collapse (AND)
V1′: ~15 distinct productive masks (AND)
      ↓  multi-seq productive filter (n_seq_neg ≥ 2)
V2: 12 AND cells (still no interior; often axis-degenerate strips)
      ↓  full-neighborhood thickness / coord-union interior
V2′: 0 region candidates
      ↓  exact-absolute nested LOSO
V3: 0 portable clauses
      ↓  selected-policy online (freeze OR-tail — different parameterization)
V4-A: 0 triggered / 0 rejected  (support mismatch)
V4-B: parameter-region retention  NOT_MEASURED
```

**Interpretation:** Contraction to empty under current gates is **already decisive for promotion**. Terminal B does **not** wait on G4–G7. Remaining research value is **interpreting the thin residual** (duplicates vs islands vs axis-degenerate strips vs any real thickness), not inventing Boolean search space.

---

## Phase E2–E3 — Property × Grammar detail (high-signal only)

### R1 / R2 — Dual area (fixed dual output; not owner either/or)

T0 **must** report both:

```text
coordinate_safe_area_ratio
unique_mask_safe_ratio

coordinate_productive_area_ratio
unique_mask_productive_ratio
```

| Ratio | Answers |
|---|---|
| coordinate_* | How thick is the safe set under threshold perturbation on the registered lattice? |
| unique_mask_* | How many distinct event-selection outcomes are safe/productive? |

- Coordinate-only inflates area via semantic-duplicate plateaus.  
- Mask-only loses threshold thickness.  
- Lattice denominators stay grammar-specific; never compare raw counts across G1 vs G2 dimensions.

**Derivable without rerun** from existing atlases + `mask_sha256` / `semantic_duplicate_mask`.

### R3 — Distance vs thickness

- Current terminal uses **interior boolean** ≈ `full_neighborhood_safe_radius ≥ 1` existence.  
- `0 interior` does **not** report continuous radius or nearest-unsafe distance series.  
- T0 must compute both metrics on productive-safe coordinate unions (derivable / light offline analysis on existing coordinates).

### R4 — Cross-sequence productive-support geometry

- GT0 intersection under additive resolved GT-hurt is **not** the headline gap.  
- Headline gap: multi-seq productive support (already countable: **12** AND cells with `n_seq_neg≥2`; 141 single-seq).

### R5 — LOO

- Nested LOSO = **exact absolute clause** only.  
- Region / quantile / component transfer = design-only until T0 shows remaining promotable structure.

### R6 — Online

- Policy retention (freeze): measured null.  
- Region retention: not measured; hook ABI = hard-OR of fixed singleton tails only.

### G7 authorization gate

```text
Only authorize G7 evaluator extension if T0/T1 show that the missing
research question cannot be answered by reinterpreting existing
singleton and AND surfaces.

First deliverable: G7 equivalence audit
  - Does ¬N_a already appear as a G1 upper/lower tail mask?
  - Does ¬N_a ∧ P_b already appear as a G2 AND mask?
  - What is NEW: operand role, GT-envelope boundary,
    safety-anchor vs support-evidence semantics,
    envelope-relative parameterization — not necessarily new masks.

Only if required masks or parameter coordinates are missing
→ restricted G7 atlas slice / evaluator change.
```

For a single signal, \(\neg N_a\) is often numerically a tail singleton; \(\neg N_a \land P_b\) may already live in pairwise AND cells. G7’s research value is **role-structured envelope geometry**, not free Boolean expansion.

---

## 11.7 Gap Classification

| ID | Gap | Class | Severity |
|---|---|---|---|
| C1 | Dual safe/productive area ratios (coordinate + unique-mask) | **Derived view** (T0) | Medium |
| C2 | Cross-seq productive-support geometry | **Derived view** (T0) | High |
| C3 | Dual margin: nearest_unsafe vs full-neighborhood radius | **Derived view** first; light emit later if useful | Medium |
| C4 | Component shape / axis degeneracy report | **Derived view** (T0) | Medium |
| C5 | G7 semantic equivalence audit vs G1/G2 masks | **Derived view** (T0) | Medium (gate for any grammar work) |
| C6 | Region-level LOO | **Evaluator change** — only after T0 | High if portability reopened; else deferred |
| C7 | Parameter-region online retention | **Blocked by hook grammar / multi-θ tooling** | High for V4-B; not needed to keep terminal B |
| C8 | Selected-policy online retention | **Already supported** (freeze null) | — |
| C9 | G4–G6 grammar expansion | Deferred / low | Low |
| C10 | G7 enum extension | Conditional on C5 fail | Conditional |

---

## 11.8 Recommended Minimal Landing

### Primary recommendation (single path)

> **T0 — Existing Atlas Region Interpretation Pack**  
> Offline interpretation of G1–G3 artifacts only.  
> **No new signal family · no Boolean framework rewrite · no hook/production change · terminal B unchanged.**

T0 answers:

> Of the **154** productive-safe cells, how many are threshold duplicates, single-sequence event islands, axis-degenerate thin strips — and is there any real full-neighborhood thickness the headline (`0 interior`) under-states?

**Only after T0** decide:

- whether T1 evaluator emit is needed;  
- whether region-level LOO is needed;  
- whether any G7 grammar extension is needed.

### Why not a new Boolean engine / signal family

- G1–G3 already exhaust authorized Q4.5 combinators **within registered lattices**.  
- Terminal B is driven by geometry gates (0 interior, thin strips, 0 exact-LOSO portable), not missing DNF power.  
- Plan §19 forbids unrestricted Boolean mining.  
- Entry terminal after Q4 is already B (weak separation); ranking path is separately blocked by invalid assignment-group key.

### What to reuse

- Full `atom_atlas` / `pairwise_*_atlas` / `region_stability` / `per_sequence` / nested LOSO artifacts.  
- Cohort lock, unresolved firewall, coord-union topology, exact-LOSO naming.

---

## 11.9 Phased Follow-up Tasks

### T0 — Existing Atlas Region Interpretation Pack (**do first**)

```text
objective:
  Interpret existing Q4.5 G1–G3 artifacts as region geometry;
  close observation gaps without evaluator rewrite.

inputs:
  out/signal_study/m_b1_5_stage2_q45_20260710/*
  Q4.5 evaluator truth: 6df1739b / merge 234f9f59
  consolidated docs truth: 51b9c73e

required outputs (all seven):
  1. raw-coordinate safe/productive area ratios
     (declared lattice denominators per grammar)
  2. unique-mask safe/productive area ratios
  3. productive capacity distribution
     (n_neg histogram; single- vs multi-seq)
  4. multi-sequence productive-support geometry
     (intersection-style multi-seq productive cells;
      sequence islands; worst-seq capacity)
  5. component shape / axis degeneracy
     (strip vs block; plateau widths; max component sizes)
  6. two boundary metrics:
       nearest_unsafe_distance
       full_neighborhood_safe_radius
  7. G7 semantic equivalence audit
       (which ¬N / ¬N∧P roles already exist as G1/G2 masks)

acceptance:
  - dual area always reported together
  - dual margin always reported together; strip may have distance>0 and radius=0
  - R4 framed as productive-support geometry, not GT0∩ re-derivation
  - no cross-grammar raw cell-count comparison
  - G7 audit states mask-equivalence vs true missing coordinates
  - terminal B retained; no production/hook claims

non-goals:
  new evaluator; new grammar enum; best-rule selection;
  region LOO implementation; hook change; signal family expansion

dependencies: none (artifacts present)
claim boundary:
  descriptive geometry only; not portable safe-region
```

### T1 — Minimal emit extension (**only if T0 shows persistent need**)

```text
objective:
  Promote T0 dual-area / dual-margin / multi-seq productive flags
  into evaluator CSV columns if manual derived tables are insufficient
  for reproducibility or claim hygiene.

inputs: existing Q4.5 evaluator + locked events + T0 findings
outputs: extended columns + unit tests
acceptance:
  interior still coord-union;
  tests: thin strip → full_neighborhood_safe_radius==0, nearest_unsafe_distance may be >0;
         thick block → radius≥1 and interior
non-goals: G4–G7 enum; quantile LOO; hook; new signals
dependencies: T0 completed; residual gap documented
claim boundary: still not formal_safe_region without interior+nested LOSO gates
```

### T2 — Region-Transfer LOO Spec (**design-only until authorized**)

```text
objective:
  Specify selection unit (point vs component vs boundary) and transfer kind
  (absolute thr vs quantile rank) only if T0 shows non-trivial thickness
  or multi-seq productive structure worth transferring.

inputs: nested_loso_summary definition; T0 geometry
outputs: design note + acceptance outline
acceptance: explicitly distinct from exact-absolute clause audit
non-goals: implementation by default; reopening terminal A without data
dependencies: T0; owner decision that threshold path remains live
claim boundary: design only
```

### T3 — G7 (**only if equivalence audit fails and threshold path reopened**)

```text
objective:
  If T0 G7 audit finds missing masks/coordinates or missing envelope
  parameterization, formalize GT envelope N_a and restricted ¬N_a / ¬N_a∧P_b.

inputs: T0 equivalence audit; d_online_events + primary lock
outputs: envelope table + only-if-needed restricted G7 slice
acceptance:
  - authorized only if T0/T1 cannot answer via G1/G2 reinterpretation
  - envelope falsifiers declared; unresolved policy unchanged
  - no 3+ free Boolean; no production claim
non-goals: unrestricted DNF; signal mining; automatic hook mapping
dependencies: T0 item 7 must complete first
claim boundary: descriptive role-structured geometry only
```

### T4 — Online retention map (documentation)

```text
objective:
  Keep A/B split permanent in docs:
    selected-policy retention (freeze) vs parameter-region retention.
inputs: portable_or_tail ABI; Stage 1 close study
outputs: grammar→hook support matrix + retention taxonomy
acceptance:
  freeze null ≠ G3 region online collapse;
  AND/G7 region retention labeled NOT_MEASURED without hook change
non-goals: hook change; e2e thr search
dependencies: none
claim boundary: exploration only
```

---

## 11.10 Owner Decisions Required

Still open (not inventable from code alone):

1. **Productivity floor for “interesting” region views** — descriptive tables may keep `FP_removed>0`; multi-seq remains a separate filter (already used for region_candidate via `MIN_SEQS_FOR_REGION=2`).
2. **Whether threshold-path geometry work continues after T0** — T0 may prove further grammar expansion is not worth it.
3. **LOO selection unit if region transfer is pursued** — point, component, or boundary.
4. **Mechanism family ownership** before any G5 work.
5. **Whether parameter-region online retention is ever required** — would imply multi-θ tooling or hook ABI expansion (out of exploration scope).

**Closed / fixed by this revision (no longer owner forks):**

- Dual area ratios: **both** coordinate and unique-mask.  
- Dual margin metrics: **both** nearest_unsafe_distance and full_neighborhood_safe_radius.  
- R4 primary name: **cross-sequence productive-support geometry**.  
- Online: **policy retention vs region retention** split.  
- G7: **equivalence audit before enum**.  
- G1–G3 “full enumeration” only **within registered lattices**.  
- Commit truth: **#89 `6df1739b` / `234f9f59` vs docs `51b9c73e`**.  
- Cohort, unresolved non-defaulting, exact-LOSO ≠ region portability, competition untrusted, production unchanged, 3+ atoms forbidden.

---

## Reviewer checklist (§12)

| # | Answer |
|---|---|
| 1 | **G1:** full registered lattice + 1 productive-safe extreme point. **G2:** 153 productive-safe coords / 15 unique masks / 12 multi-seq, thin strips, 0 interior. **G3:** complete registered OR atlas with 0 productive-safe. Unresolved firewall + nested exact-LOSO. |
| 2 | `0 interior` ≈ no positive `full_neighborhood_safe_radius`. Does **not** replace dual area ratios, multi-seq productive maps, nearest-unsafe series, quantile LOO, or parameter-region online retention. |
| 3 | **No formal emitted dual denominators yet**; lattice sizes known — ratios **derivable** and T0-mandatory. |
| 4 | Interior + neighbors today; T0 must emit **distance vs thickness** separately. |
| 5 | No first-class productive-support intersection product; GT0∩ ≡ pooled for resolved labeled hurt; multi-seq productive **derivable** (12 AND). |
| 6 | Nested LOSO = **exact absolute clause** only. |
| 7 | Stage 1 credibly measures **selected-policy** freeze retention (null). **Parameter-region** retention not measured. |
| 8 | G7 only after equivalence audit; G4/G5/G6 deferred expansion. |
| 9 | Dual areas, multi-seq productive tables, strip topology, dual margins, G7 mask equivalence — **without rerun**. |
| 10 | Yes: T0 interpretation pack; optional T1 only if needed. |
| 11 | **Yes — Q4.5 terminal `isolated_safe_points_only` (B) unchanged.** |
| 12 | Distinctions held: rule search forbidden; point audit done; region geometry partial; transfer = exact-clause; online = policy A vs region B. |

---

## Bounded verdict

```text
Current atlas coverage:
  G1–G3 fully enumerated within the registered Q4.5 lattices,
  directions, frozen signal pairs, combinators, and unresolved firewall;
  R1–R2 as point counts (dual area ratios derivable, not emitted);
  R3 as interior boolean + thin-strip topology (0 interior);
  R4 as per-seq counts/shares — missing cross-seq productive-support geometry;
  R5 as exact-absolute nested LOSO (0 portable);
  R6-A selected-policy online retention: freeze OR-tail supported (0 fire);
  R6-B parameter-region online retention: not measured;
  G4–G7 not measured as grammars.

Primary missing observation:
  Existing Atlas Region Interpretation Pack (T0):
  dual area + multi-seq productive-support geometry
  + dual margin (distance vs thickness)
  + component degeneracy
  + G7 mask equivalence audit
  — not a new Boolean search engine.

Primary missing grammar, if any:
  none by default;
  G7 only if T0 equivalence audit shows G1/G2 reinterpretation insufficient
  and threshold path is deliberately reopened;
  G4/G5/G6 deferred.

Can be derived without rerun:
  partial (T0 almost entirely)

Requires evaluator change:
  no for T0;
  yes (minimal) for T1 only if T0 shows emit need;
  yes (larger) for region-LOO or G7 enum only if gated in

Requires new signal family:
  no

Requires online hook change:
  no for exploration

Recommended next task:
  T0 Existing Atlas Region Interpretation Pack
  (seven outputs above). Keep terminal B.
  Defer T1 / region-LOO / G7 until T0 closes or proves need.

Existing Q4.5 terminal:
  unchanged

Truth base:
  Q4.5 evaluator: 6df1739b (PR #89 head); merge 234f9f59
  Docs consolidation: 51b9c73e (PR #90)
  Machine study: out/signal_study/m_b1_5_stage2_q45_20260710
```

**One-line conclusion:**  
Q4.5 already answers whether restricted singleton/AND/OR geometry yields thick, multi-seq, portable safe regions (**no** under current gates). The coverage holes are **region observation layers** (dual area, productive multi-seq geometry, distance vs thickness margin, honest policy-vs-region online mapping, G7 role equivalence), not unrestricted Boolean power. Minimal next step is **T0 interpretation of existing artifacts**; that pack is the decision gate for any later emit, LOO, or grammar work.

---

## Appendix A — Key artifact paths

```text
# Machine studies (on disk)
out/signal_study/m_b1_5_stage2_q1q3_20260710/
out/signal_study/m_b1_5_stage2_q4_20260710/
out/signal_study/m_b1_5_stage2_q45_20260710/
out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/

# Q4.5 headline files
out/signal_study/m_b1_5_stage2_q45_20260710/summary.json
out/signal_study/m_b1_5_stage2_q45_20260710/atom_atlas.csv
out/signal_study/m_b1_5_stage2_q45_20260710/pairwise_and_atlas.csv
out/signal_study/m_b1_5_stage2_q45_20260710/pairwise_or_atlas.csv
out/signal_study/m_b1_5_stage2_q45_20260710/region_stability.csv
out/signal_study/m_b1_5_stage2_q45_20260710/nested_loso_summary.json
out/signal_study/m_b1_5_stage2_q45_20260710/threshold_registry.json
out/signal_study/m_b1_5_stage2_q45_20260710/per_sequence.csv

# Source (PR #89 evaluator truth @ 6df1739b; integrated via PR #93)
src/saccade/perception/eval/d_online_stage2.py
src/saccade/perception/eval/d_online_stage2_q4.py
src/saccade/perception/eval/d_online_stage2_q45_atlas.py
scripts/tools/run_m_b1_5_stage2_q45_atlas.py
tests/unit/test_d_online_stage2_q45_atlas.py

# Contracts (consolidated docs truth @ 51b9c73e; integrated via PR #93)
docs/modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md
docs/modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md
docs/modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md
docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md
```

## Appendix B — Canonical headline numbers

```text
stage2_q45_terminal: isolated_safe_points_only  (terminal_letter B)
taxonomy_version: stage2_q45_atlas_v4
evaluator_truth: 6df1739b (PR #89 head) / merge 234f9f59
docs_truth: 51b9c73e (PR #90 consolidation)

n_primary: 87 (neg=23, pos=64)
n_selected_unresolved: 21

productive_safe coordinates:
  singleton = 1
  pairwise AND = 153
  pairwise OR = 0
  total = 154

AND unique productive masks: 15
AND multi-seq productive (n_seq_neg >= 2): 12
AND single-seq productive: 141
coordinate-union interior = 0
region candidates = 0
exact-absolute nested-LOSO portable clauses = 0
max component_size_coordinates: 19  (thin strips)

online:
  selected-policy freeze OR-tail: triggered=0 rejected=0
  parameter-region retention: NOT_MEASURED
```

## Appendix C — Revision log (this edition)

```text
1. Split truth base: Q4.5 evaluator 6df1739b/234f9f59 vs docs 51b9c73e
2. Qualify G1–G3 enumeration to registered lattices only
3. Lock dual area ratios (coordinate + unique-mask); not owner either/or
4. Rename R4 primary gap to cross-sequence productive-support geometry
5. Split margin into nearest_unsafe_distance vs full_neighborhood_safe_radius
6. Gate G7 behind G1/G2 equivalence audit before any enum
7. Split R6 into selected-policy vs parameter-region online retention
8. Collapse recommended next step to T0 interpretation pack (7 outputs);
   demote T1 / LOO / G7 until T0 proves need
```
