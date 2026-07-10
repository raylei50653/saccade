---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# Safe-Region A1 — Acceptance-Unit Lock + Read-Only S0/S1/Q1/N1 Audit

> **One-line:** A1 acceptance unit **locked** to conversion pack `1a180620bc…`. Read-only audit **26/26 PASS** (S0 unit lock · S1 semantic crosswalk · Q1 pack-only battery · N1 negative controls) + mutation sensitivity **5/5**. R1/R1.1 downgraded to **external diagnostic overlay**. Outcome: **`A1_ACCEPTED_WITH_LIMITS` recorded 2026-07-11 → maturity A1, gate CLOSED** (see Terminal outcome).

## Context

Chat-side review (2026-07-10) rejected the four-part A1 framing (semantic fidelity · query utility · decision utility · reusable abstraction) as an acceptance test with a non-unique acceptance object, and prescribed a minimal takeover: lock one acceptance unit, run a read-only S1+Q1+N1 audit, downgrade R1/R1.1 to overlay. This note records that takeover. All seven review findings were **verified against code/artifacts before write-back** (see below).

## Acceptance unit (S0 — locked)

| Item | Value |
|:--|:--|
| A1 acceptance unit | conversion pack `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` |
| `pack_id` | `1a180620bc050e70dd4a673f991ab24410e78ab02d2cbe49346414800ca631a7` |
| Declared maturity | **A0** (`review_status=A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW`) |
| Probe root (`safe_region_assetization_r1_20260710/`) | **external diagnostic overlay** — its `A1_region_asset` metadata tags are overlay labels, **not** A1 acceptance objects |
| R1.1 root (`safe_region_assetization_r11_20260710/`) | external diagnostic overlay |

Neither probe consumes the pack: both read sealed Q4.5 registry + raw events directly (`safe_region_assetization_r1.py`, `safe_region_assetization_r11.py` take `q45_dir`/`events_path`; zero references to the pack root). They therefore provide **no** pack-consumption, query-utility, or reusable-abstraction evidence.

## Audit (read-only; 2026-07-11)

| Item | Value |
|:--|:--|
| Runner | `scripts/tools/run_safe_region_a1_audit.py` |
| Report | `out/signal_study/safe_region_a1_audit_20260711/` (`a1_audit_report.json` · `a1_audit_checks.csv`) |
| Verdict | **AUDIT_PASS — 26/26 checks** |
| Goldens | predeclared accepted A0 baseline in the thread doc (recorded before this audit) |
| Oracle readback | one-time S1 only (sealed T0 + Q4.5 evidence dirs); Q1/N1 are pack-only |

### S1 semantic crosswalk (11/11 PASS)

- 154 membership rows, all productive-safe, `gt_hurt` sum 0; regions 26 = 1 G1 + 25 G2; G2 claim levels 6×L0 (isolated) + 19×L1 (multi-coord, L1⇔`n_coords>1`).
- 34 grid-local mask units → **15 unique `mask_sha256`**, G1's mask ⊂ G2 set (union 15, not 16); pack mask-sha set == sealed T0 set.
- 26 regions ↔ 26 T0 components via `t0_component_id_alias`: `n_coords`/`shape_class`/radius mismatches **0**; all 154 coords ↔ T0 `productive_capacity` rows: capture counts + per-sequence JSON mismatches **0**.
- 1 G3 domain-null (17640 registered coords, 0 productive, L0). Thin-strip geometry everywhere (radius 0, nearest-unsafe > 0).
- Threshold registry: pack envelope seals the Q4.5 file (`source_file_sha256` match), shared keys equal, all **1080** entry rows reconstruct sealed `thr_value`/coords exactly.
- 7-sequence cohort; membership per-sequence keys ⊆ cohort (6 sequences carry productive support).

### Q1 pack-only battery (5/5 PASS)

topology (1+153+0 coords; 1+25 components) · dual capacity (denominators 870/17640/17640; cohort 23 neg / 64 GT) · duplicate-mask grain (34 units vs 15 masks distinguishable pack-only) · G3 null precise — these four against predeclared goldens. Sequence union/intersection is verified as pack-only **computable** with closure (∩ ⊆ ∪ ⊆ cohort, all 26 regions), **not** as exact-answer against per-region goldens (none were predeclared; claim capability, not exactness).

### N1 negative controls (6/6 PASS)

`forbidden_promotions` covers A0→A1 self-accept, ceiling→per-object L1, safe-region→production, GT0→population-zero, family→instance, thr_index→value; capacity is declared non-additive (Σ`n_neg`=176 ≫ cohort 23, so the sum→event-mass promotion is both forbidden and demonstrably wrong); `observation_only` everywhere; 154 policy instances ≠ 34 mask units ≠ 15 masks.

### Not audited (stays open for the terminal)

- **D1 decision trace** — no pre-declared bounded decision rule exists yet; running one post-hoc would repeat the R1.1 error. Binding: a pre-declared D1 is required **only** for an `A1_ACCEPTED` that claims decision utility; `A1_ACCEPTED_WITH_LIMITS` may record the absence of D1 as an explicit `acceptance_limit` — **no post-hoc D1 is required to close A1**.
- **Reusable abstraction** — usage-based; requires a second independent consumer adopting RegionAsset IDs/relations. Not verifiable today.

## Verified review findings (basis for the overlay downgrade)

All checked against code/artifacts, not taken on faith:

1. **Acceptance object was not unique** — probe metadata writes `maturity_level=A1_region_asset` (`safe_region_assetization_r1.py:537,762,2048,2147`) while the pack declares A0 and the thread held A1 open. → resolved by S0 lock.
2. **R1/R1.1 are not pack consumers** — both read raw Q4.5/events; zero pack references. → no A1-Q/abstraction evidence.
3. **LOO is not held-out-label-isolated at the screening layer** — basis `n_neg`/`n_gt` computed on full-cohort labels (`r1.py:1010-1011`), top-48 candidate pool ranked by them (`r1.py:1334-1348,1625`), same pool reused in every fold, truncated to top-24 (`r1.py:1372`). The notes' claim "labels do not select registry" is false for the candidate pool. Defensible residue: pooled in-sample 4→8 compositional witness; **this global-label-screened, top-24, equal-weight heuristic** hurts GT on MOT17-05/11 fold refits; no V-A evidence. **Not** an inductive falsification of the whole non-negative sparse K≤5 class.
4. **F1/F3/F4 scores are post-hoc hard-floored** to 40/55/50 (`r11.py:584,598,605`); rules not pre-registered in R1.
5. **F3's ≥3-reversal floor trigger double-counts** — the 4 L3 reversals are 2 unique (basis, holdout-event) pairs counted under K2 and K5 (`basis_role_reversal.csv`); deduplicated evidence = 2 < 3.
6. **F3 predicate identity is not unique** — harmful basis `b2:07de9243…` has 12 observed-mask aliases (`abs_log_h↑∧score↓` and `abs_ratio_m1↑∧score↓` families); naming one alias as the mechanism violates the Boolean contract (observed-mask equality ≠ semantic identity).
7. **Probe artifact semantic defect** — `linear_probe_models.csv` L0 rows fill `n_productive_sequences` with `multi_sequence_productive_coordinates` (=12) in a 7-sequence cohort (`r1.py:1574`). Defect is in the **probe overlay**, not the conversion pack (pack passed S1 with 0 mismatches).

## Downgraded overlay readings

- **R1 probe**: V-C stands only as *heuristic-specific descriptive failure* under a global-label-screened transductive protocol; "in-sample grammar-limited + cross-seq invariance-limited" survives as description, not as an inductive class null.
- **R1.1**: net contribution = **2 observed harmful AND events located** (`b2:07de9243…`→MOT17-05 `f4:c8:l1:i32`; `b2:afd594d1…`→MOT17-11 `f4:c0:l4:i5`) + three descriptive symptom classes (role reversal, weak holdout retention, margin contraction). **Not** a primary-causal-mechanism ranking, not transfer-null evidence, not pack-consumption evidence.
- Terminal B unaffected either way (it never rested on the probe overlay).

## State after this note

```text
A1 acceptance unit: conversion pack 1a180620bc… (locked)
A1 state: A1_PENDING_VALIDATION → S0/S1/Q1/N1 audit PASS (26/26)
maturity: A0 retained (terminal not yet recorded)
terminal decision: research-owner only —
  A1_ACCEPTED_WITH_LIMITS is now supportable if limits enumerate:
    · decision utility not demonstrated (no pre-declared D1 trace)
    · reusable abstraction unproven (usage-based; no second consumer)
    · event-mass / non-productive-cell / predicate-alias queries need raw artifacts
R1/R1.1: external diagnostic overlay (descriptive only)
R2–R4: unauthorized (fail-closed; unchanged)
production / ledger / terminal B: unchanged
```

## Terminal outcome (write-back 2026-07-11)

The block above is the pre-terminal state, kept for the record. After PR #97 merged (CI green) and mutation sensitivity passed 5/5 on `main` (`tests/unit/test_safe_region_a1_audit.py`: alias tamper → S1.7 FAIL · per-sequence tamper → S1.8 FAIL · promotion removal → N1 FAIL · self-promoted maturity → S0.3 FAIL), the research owner recorded:

```text
A1_ACCEPTED_WITH_LIMITS → maturity A1 — gate CLOSED
```

Authoritative record with full accepted scope / acceptance_limits / non-authorizations: [thread § A1 terminal record](../../../research/threads/safe_region_assetization_20260710.md#a1-terminal-record-2026-07-11). No post-hoc D1 was run; its absence is an enumerated `acceptance_limit`.

## Reproduce

```bash
.venv/bin/python scripts/tools/run_safe_region_a1_audit.py \
  --pack out/signal_study/m_b1_5_safe_region_asset_r1_20260710 \
  --t0-evidence docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710 \
  --q45-evidence docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710 \
  --out out/signal_study/safe_region_a1_audit_20260711
```
