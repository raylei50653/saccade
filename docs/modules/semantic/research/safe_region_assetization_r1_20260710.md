---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization R1 — G1–G3 Region Asset Pack + Linearized Feasibility Probe

> **One-line:** **DOWNGRADED to external diagnostic overlay** (2026-07-10 chat review; see below). Phase B V-C survives only as *heuristic-specific descriptive failure* — the LOO candidate pool is global-label-screened, so no inductive class null. This note's tables are **not** A1 acceptance objects; the A1 unit is the conversion pack `1a180620bc…`. Terminal B retained. R2 not authorized.

## 2026-07-10 chat-review downgrade (normative — read before the rest)

Verified against code/artifacts (details: [A1 audit note](safe_region_a1_audit_20260711.md)):

1. **LOO is not held-out-label-isolated at the screening layer.** Basis `n_neg`/`n_gt` are computed on full-cohort labels (`safe_region_assetization_r1.py:1010-1011`); the top-48 candidate pool is ranked by them (`:1334-1348`, `:1625`); the same pool is passed into every fold and truncated to top-24 (`:1372`). The §"LOO naming boundary" claim that *labels do not select registry* is **false for the candidate pool**. Defensible residue: (a) pooled in-sample 4→8 compositional witness; (b) this global-label-screened, top-24, equal-weight heuristic hurts GT on MOT17-05/11 fold refits; (c) no V-A positive evidence. **Not defensible:** falsifying the entire non-negative sparse K≤5 class, or calling the transductive failure a stronger inductive null.
2. **`A1_region_asset` maturity tags in this study's metadata are overlay labels**, not pack maturity. Phase A "ACCEPTED" below means *descriptively reconciled*, not A1-accepted.
3. **Known semantic defect:** `linear_probe_models.csv` L0 rows write `multi_sequence_productive_coordinates` (=12) into the `n_productive_sequences` column (`r1.py:1574`) — impossible in a 7-sequence cohort. Do not consume probe tables as semantic-fidelity assets.
4. This study **never reads the conversion pack** (inputs: sealed Q4.5 registry + raw events), so it provides no pack-consumption / query-utility / reusable-abstraction evidence for A1.

## Research acceptance (chat-side)

| Gate | Status |
|:--|:--|
| Phase A asset conversion | **ACCEPTED** (mask-count reconciliation applied) |
| Phase B feasibility probe | **ACCEPTED as V-C** |
| Engineering acceptance | pending PR/code/artifact review (if landing) |
| Terminal | **B retained and strengthened** |
| R2 grammar distillation | **not authorized** |
| G4–G7 expansion | **not authorized by this result** |
| Hook / preset / production | **unchanged** |
| Evidence ledger | **no automatic promotion** |

### Verdict refinement (accepted wording)

```text
in-sample grammar-limited
cross-sequence invariance-limited
```

Not blanket `signal-limited` for all future model classes. Closed class only:

```text
frozen 5 signals
× registered predicate basis
× interaction order ≤ 2
× non-negative sparse weights K ≤ 5
× hard GT / unknown constraints
```

---

## Authorization / scope

| Item | Value |
|:--|:--|
| Task | G1–G3 Region Asset Pack + Linearized Feasibility Probe |
| Type | offline research tooling + derived evidence |
| Level | D1 |
| Thread | [safe_region_assetization_20260710.md](../../../research/threads/closed/safe_region_assetization_20260710.md) |
| Study | `out/signal_study/safe_region_assetization_r1_20260710/` |
| Code | `src/saccade/perception/eval/safe_region_assetization_r1.py` · `scripts/tools/run_safe_region_assetization_r1.py` |
| Tests | `tests/unit/test_safe_region_assetization_r1.py` (T1–T10) |

---

## Fixed research truth (not rewritten)

```text
signal family: fixed frozen 5
primary: resolved ∧ baseline_selected · neg=23 · GT=64 · n=87
selected unresolved: 21 (never defaulted negative)
terminal B: isolated_safe_points_only
G1 PS coords=1 · G2 PS coords=153 · G3 PS=0
G2 unique productive prediction masks=15
multi-seq coords=12 · coord-union interior=0 · nested portable=0
```

---

## Identity inventory (claim hygiene)

**Do not call different asset identities “unique masks.”**

Authoritative split (`identity_inventory.json` / `summary.json`):

| Field | Value | Meaning |
|:--|--:|:--|
| `n_unique_prediction_masks_productive_safe` | **15** | Distinct `mask_sha256` over G1∪G2∪G3 PS (G1⊂G2) |
| `n_unique_prediction_masks_g2` | **15** | Matches historical G2 unique productive masks |
| `n_unique_prediction_masks_g1` | **1** | Singleton PS mask (also in G2 set) |
| `n_grid_local_mask_assets` | **34** | Rows in `region_masks.csv` = **grid-local placements** |
| `n_productive_safe_components` | **26** | 1 G1 + 25 G2 components |
| `n_coordinate_instances_productive_safe` | **154** | 1+153+0 lattice cells |
| `n_semantic_role_assets` | **27** | Component assets + G3 domain-null |

### 34 vs 15 reconciliation

```text
34  ≠  distinct prediction masks
34  =  grid-local mask placements
       (same prediction mask under multiple pairwise feature×direction grids)

15  =  distinct productive-safe prediction masks (historical lock)

1+15+0 is wrong as a sum: G1 mask is already one of the G2 15
       → union = 15, not 16
```

Earlier prose that said “34 unique productive mask units” was **wrong naming**. Correct: **34 grid-local mask assets**, **15 unique prediction masks**.

`region_id` for mask assets is now grid-scoped (`…:g<gridhash>`) so 34 rows have 34 distinct IDs (no cross-grid collision).

---

## Phase A — Region asset pack (ACCEPTED)

| Grammar | lattice_n | coord productive ratio | unique prediction masks (PS) | components |
|:--|--:|--:|--:|--:|
| G1 | 870 | 0.00115 | 1 | 1 |
| G2 AND | 17640 | 0.00867 | **15** | 25 |
| G3 OR | 17640 | 0.0 | 0 | domain-null |

- Dual area: coordinate ratios ≠ unique-mask ratios; grammar-specific denominators.
- Dual margin: **all 26 components** have `nearest_unsafe_distance > 0` and `full_neighborhood_safe_radius = 0` (thin strips).
- Separates “looks like a plateau” from “has thickness.”
- Claim contract: `A1_region_asset` · `observation_only` · `promotion_status=forbidden`.

### Two reusable asset classes

1. **Descriptive region assets** — identity, topology, capacity, sequence support, dual margin, claim boundary.
2. **Non-transferability evidence asset** — Phase B LOO failure under a strictly larger model class than G1–G3 (blocks re-discovering sequence-conditioned fit via another grammar search).

### Q1

**Yes** — stable A1 research assets emitted. No A2/A3.

---

## Phase B — Linearized feasibility probe (ACCEPTED as V-C)

### Basis

| Item | Count |
|:--|--:|
| Non-constant collapsed basis | 7136 |
| Order-1 / Order-2 | 752 / 6384 |
| Pure-safe productive (GT=0, unk=0, neg>0) | 14 |

Collapse by prediction mask; aliases retained. No 3+ interaction.

### Results

| Family | Hard-safe pooled |
|:--|:--|
| L0 G2 | max n_neg = **4** |
| L1 equal-weight count | **invalid** (GT+unknown captured) |
| L2 sparse singleton K≤5 | n_neg=1; single-seq |
| L3 sparse + AND K=2..5 | n_neg=**5→8**; multi-seq 5–6 **in-sample** |

L3 improves pooled capacity over G1–G3 cells, but **nested sequence LOO holds GT hurt** on multiple folds.

### LOO naming boundary (required)

```text
protocol kind:
  transductive / globally registered basis LOO

NOT claimed:
  fully inductive train-only threshold transport

Layers (must keep separate):
  1. global label-free basis registry
     — sealed Q4.5 threshold_registry / atlas lattice
     — may use all-sequence feature values; labels do not select registry
  2. train-only supervised model selection
     — weights / support / τ from train-sequence labels only
  3. held-out label isolation
     — holdout labels never select basis, K, or threshold
```

~~This does **not** weaken the null result: the model already receives a looser held-out covariate channel via the global registry and still fails LOO → stronger non-transfer claim.~~ **Retracted (2026-07-10 review):** the candidate pool is selected by full-cohort *labels* (not just label-free covariates), so the "stronger null" reading is invalid — see the downgrade section at the top.

### Q2–Q4 (accepted interpretation)

| Q | Answer |
|:--|:--|
| Q2 Grammar capacity | **In-sample yes** — linear OR-of-AND exceeds single G2 cell (8>4) |
| Q3 Signal capacity under this probe | Hard-safe multi-seq solutions exist **in-sample** |
| Q4 Robustness / transfer | **Fail** — LOO GT hurt; not transferable |

---

## ~~Stronger null than terminal B alone~~ (downgraded: descriptive only)

**2026-07-10 review:** the "closed class null" below is retracted as a class-level claim — the search only ever evaluated the global-label-screened top-24 pool with equal weights (see downgrade section). What survives is heuristic-specific: *this* screening+fit recipe fails fold refits. Original text kept for the record:

**Q4.5:** G1–G3 registered grammar finds no thick, multi-seq, portable safe region.

**R1 adds:** under fixed 5 signals, registered predicates, ≤2-order interaction, non-negative sparse K≤5, hard GT/unknown — even a more expressive linearized model only improves **pooled fit**, not **sequence-LOO transferable** safe regions.

Closed class null. Does **not** claim:

- all linear models impossible;
- all signal transforms fail;
- new signal families useless;
- any normalization / relative coordinate useless.

---

## Bounded verdict: **V-C**

```text
Linearized models improve pooled fit but collapse under
per-sequence or LOO validation.

The added capacity produces overfit, not transferable safe regions.
```

Refined:

```text
in-sample grammar-limited
cross-sequence invariance-limited
```

**Implications:**

- No grammar expansion for promotion.
- No gate / hook from this probe.
- Terminal **B** retained **and** strengthened.
- R2 distillation **not authorized** (needs V-A).

---

## Next authorized direction (not this note’s deliverable)

**R1.1 — Transfer Failure Attribution Pack** (no new model, no grammar expansion):

- fold basis overlap / Jaccard;
- weight & τ stability;
- which bases trigger held-out GT hurt;
- train productive support on holdout;
- boundary vs support vs sign conflict;
- train-only vs global registry gap;
- failure taxonomy F1–F5.

Goal: decide relative/normalized coordinates vs new signal family vs conditional applicability vs close threshold safe-region line.

**Do not** restart grammar search. R1 already answered: *can fit better, cannot stably transfer.*

---

## Reproduce

```bash
.venv/bin/python scripts/tools/run_safe_region_assetization_r1.py \
  --study-id safe_region_assetization_r1_20260710 \
  --out out/signal_study/safe_region_assetization_r1_20260710

.venv/bin/python -m pytest tests/unit/test_safe_region_assetization_r1.py -q --no-cov
```

Key artifacts: `identity_inventory.json` · `summary.json` · `region_*.csv` · `linear_probe_*.csv` · `basis_*.csv`.

---

## Relation to prior pack conversion

A0 packaging ([safe_region_asset_r1_conversion_20260710.md](safe_region_asset_r1_conversion_20260710.md), PR #95) remains valid. This note is the **capacity-probe R1** + research acceptance write-back. No ledger promotion.
