---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# Boolean-Atom Partial-Order Audit — PR-D Gate (issue #106)

> **One-line:** Sealed 8-atom substrate → **`global_orderable`** = `{dist_h, log_h_ratio}` · **`conditional_orderable`** = `{bridge_dist, speed_mismatch, dir_cos, resid_mean}` (short-gap only) · **`context_only`** = `{score_m_bridge, gap}` · **`unresolved`** = ∅. Aggregate terminal **`GLOBAL_PARTIAL_ORDER_READY`**. Research acceptance = **`ACCEPTED_WITH_LIMITS`** on [PR #107](https://github.com/raylei50653/saccade/pull/107). Authorizes a **separate** restricted-closure prototype on the accepted global pair only.

Thread: [gt_support_morphology_20260711.md](../../../research/threads/gt_support_morphology_20260711.md) ·
Procedure: [framework §19](../../../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) ·
Research line: [boolean_closure_domain_line_20260711.md](boolean_closure_domain_line_20260711.md) ·
Step-0: [gt_support_morphology_step0_20260711.md](gt_support_morphology_step0_20260711.md) ·
PR-C: [escape_tail_forensic_20260711.md](escape_tail_forensic_20260711.md) ·
Packet: [evidence/boolean_atom_partial_order_20260711/](evidence/boolean_atom_partial_order_20260711/manifest.json)

## Research acceptance status

```text
PR #107 / issue #106: ACCEPTED_WITH_LIMITS
Aggregate terminal: GLOBAL_PARTIAL_ORDER_READY

Accepted map:
  global_orderable:     dist_h, log_h_ratio
  conditional_orderable: bridge_dist, speed_mismatch, dir_cos, resid_mean
  context_only:         score_m_bridge, gap
  unresolved:           ∅

Review history (not erased):
  Initial operational terminal = GLOBAL_PARTIAL_ORDER_READY
    with global = {bridge_dist, dist_h, log_h_ratio}
  Research-owner review found bridge_dist provenance misclassification
    and incorrect score_m_bridge unit claim; revisions accepted above.

Restricted closure prototype:
  AUTHORIZED as a separate post-merge task only
  global solve atoms ONLY = {dist_h, log_h_ratio}
  bridge_dist and other conditional atoms MUST NOT enter the global solve
```

### Amendment (2026-07-12, append-only) — the accepted axes are **offline-coordinate** axes

`GLOBAL_PARTIAL_ORDER_READY` stands **on its own substrate**; this amendment
retracts nothing. It records a limit that was not visible when the terminal was
accepted.

The atoms were derived through `audit_relink_safe_reject.ensure_prod_proxy_scores`,
which builds **offline proxies** of the live quantities — its own docstring states
"Height ratio uses raw endpoint heights (ema proxy)". So both accepted global axes
are offline-coordinate axes:

| Axis | Accepted (offline) | Runtime (kernel) |
|---|---|---|
| `dist_h` | rebuilt from the offline pair table | \(\|a_{\mathrm{lost}}-a_{\mathrm{cand}}\|/h_{\mathrm{ref}}\), kernel `bridge_anchor4` anchors |
| `log_h_ratio` | \(\log(h^{\mathrm{raw}}_{\mathrm{lost}}/h^{\mathrm{raw}}_{\mathrm{cand}})\) — **raw box heights** | \(\log(e_{\mathrm{lost}}/e_{\mathrm{cand}})\) — **EMA state** |

[D0](d0_runtime_shadow_fidelity_results_20260712.md) subsequently certified that
this offline substrate is **`T2 PROXY_UNFAITHFUL`**, with a **distorted GT
boundary** (7.03 % offline-safe-but-online-unsafe). A GT-retention guarantee
proved in offline coordinates therefore does **not** transfer to runtime
coordinates by formula shape or field name (see the
[runtime-quantity fidelity protocol](../../../research/contracts/runtime_quantity_fidelity_protocol.md)).

**Binding consequence:** the authorized restricted-closure prototype **may not be
solved on these axes** until their runtime transfer is audited. That audit is
[S0](safe_domain_runtime_transfer_declaration_20260712.md)
([thread](../../../research/threads/closed/runtime_faithful_safe_domain_20260712.md)).

## 0. Scope and claim ceiling

- **Read-only / offline / evidence-and-contract** ([issue #106](https://github.com/raylei50653/saccade/issues/106)).
- Frozen atoms, safer directions, pool-median descriptive split, source SHA, and GT trial unit match Step-0 / PR-C.
- Explicitly **not** done: MWC, min-cut, graph-cut, parametric closure, rule search, DNF compression, weight optimization, production/preset/ledger changes, escape-tail veto.
- Claim ceiling = **L1 order contract** under `ACCEPTED_WITH_LIMITS`. Nested held-out (PR-E) remains the only confirmatory unit for L2+.
- PR-C motion role-reversal evidence is **binding negative evidence** against unqualified global promotion of `speed_mismatch` / `dir_cos` / `resid_mean`.
- Restricted-closure is a **separate post-merge task**; not part of this audit PR.

### 0.1 Terminal vocabulary (closed)

| Terminal | Meaning |
|:--|:--|
| `GLOBAL_PARTIAL_ORDER_READY` | ≥1 nontrivial global-orderable dimension; all eight atoms bounded; allowed/forbidden contract complete → **separate** restricted-closure prototype only **after research acceptance** |
| `CONDITIONAL_STRUCTURE_ONLY` | No defensible global partial order; conditional/context structure only |
| `ORDERABILITY_UNRESOLVED` | Role map not defensible; closure work blocked |

Accepted terminal: **`GLOBAL_PARTIAL_ORDER_READY`** under **`ACCEPTED_WITH_LIMITS`**.

## 1. Frozen substrate

| Item | Value |
|:--|:--|
| Pool | `m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv` · SHA `0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17` |
| Trial unit | `(seq, lost_id)` · 209 GT tracks · descriptive min-d_H representative |
| Atoms (8) | `score_m_bridge`↓ · `bridge_dist`↓ · `dist_h`↓ · `log_h_ratio`↓ · `resid_mean`↓ · `dir_cos`↑ · `speed_mismatch`↓ · `gap`↓ |
| Binarization | pool median (audit-only; sensitivity at p40 / median / p60) |
| PR-C binding | aggregate `ROLE_REVERSAL_SUPPORTED` · research acceptance `ACCEPTED_WITH_LIMITS` (L1 single-seq MOT17-10) |

Runner: [`run_partial_order_audit.py`](evidence/boolean_atom_partial_order_20260711/run_partial_order_audit.py) ·
Reproduce:

```bash
uv run python docs/modules/semantic/research/evidence/boolean_atom_partial_order_20260711/run_partial_order_audit.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv --verify
```

## 2. Aggregate terminal

```text
Accepted terminal: GLOBAL_PARTIAL_ORDER_READY
Research acceptance: ACCEPTED_WITH_LIMITS

global_orderable:     dist_h, log_h_ratio
conditional_orderable: bridge_dist, dir_cos, resid_mean, speed_mismatch
                       context = short_gap_continuous_association (gap ≤ 60)
context_only:         score_m_bridge, gap
unresolved:           (none)

→ authorizes SEPARATE restricted-closure on {dist_h, log_h_ratio} after merge
→ does NOT authorize motion / bridge_dist global arcs, tail veto, or production
```

**Role-assignment self-declaration:** roles are research judgment under closed rules, not a fitted classifier. `V_i` / shells / threshold flips are descriptive. Executable `global_admissibility_check` may only **block** `global_orderable` (PR-C motion, motion-extrapolation composites, weighted composites with motion parents, regime descriptors).

Routing detail: [`aggregate.json`](evidence/boolean_atom_partial_order_20260711/aggregate.json).

## 3. Atom cards

Metrics at frozen median split on min-d_H representatives (n=209). Full cards: [`atom_roles.json`](evidence/boolean_atom_partial_order_20260711/atom_roles.json) · table: [`atom_metrics.csv`](evidence/boolean_atom_partial_order_20260711/atom_metrics.csv).

### 3.1 `log_h_ratio` — `global_orderable`

| Field | Value |
|:--|:--|
| Safer | lower (height consistency) |
| Provenance | derived `abs(log(h_cand/h_lost))` |
| V_i | 2/209 (2 sequences) |
| Tail d_H≥3 | **0/4** |
| p40→p60 flips | 4/209 |
| PR-C reversal | no |

Height is preserved on the protected escape tail. Direction is mechanism-stable and not a pure exploratory split artifact. Strongest structural global dimension on this substrate.

### 3.2 `dist_h` — `global_orderable`

| Field | Value |
|:--|:--|
| Safer | lower (geometry) |
| Provenance | builder raw — endpoint foot distance / `h_ref` only |
| V_i | 2/209 (MOT17-10 only) |
| Tail d_H≥3 | 2/4 |
| p40→p60 flips | 7/209 |
| PR-C reversal | no |
| Global admissibility | **pass** |

Pure structural leaf (no velocity parents). Sequence clustering of the few violations is a competing explanation; no accepted role-reversal mechanism blocks global promotion.

### 3.3 `bridge_dist` — `conditional_orderable` (demoted; PR #107 review)

| Field | Value |
|:--|:--|
| Safer (declared) | lower |
| Provenance | **motion-extrapolation composite** (not parentless geometry) |
| Formula | \(m_l=x_l+v_l\frac{gap}{2}\), \(m_c=x_c-v_c\frac{gap}{2}\), \(\mathrm{bridge\_dist}=\|m_l-m_c\|/h_{\mathrm{ref}}\) |
| Parents | lost/cand foot xy · exit/entry **velocities** · **gap** · `h_ref` |
| V_i | 2/209 (MOT17-10 only) |
| Tail d_H≥3 | 2/4 |
| p40→p60 flips | 1/209 |
| Global admissibility | **blocked** (motion-extrapolation composite) |

**PR-C relation:** long-gap re-entry (`TRUE_LONG_GAP_REENTRY`) is exactly where constant-velocity mid-gap extrapolation is least defensible; velocity parents share the motion substrate that carries accepted role-reversal evidence. Without independent multi-seq mechanism evidence, **not** `global_orderable`. Short-gap CV regime remains a **proposal-only** conditional context.

### 3.4 `score_m_bridge` — `context_only`

| Field | Value |
|:--|:--|
| Safer (declared) | lower |
| Provenance | **weighted composite**: `w·resid_mean + (1-w)·dist_h`, `w=√clip(exit_speed/0.12)` |
| V_i | 1/209 |
| Tail d_H≥3 | 1/4 |
| Scale-guard | **blocks global** (not for unit mismatch) |

Scale-guard ([`scale_guard.json`](evidence/boolean_atom_partial_order_20260711/scale_guard.json)):

- **Units are compatible:** `fwd_resid`, `bwd_resid`, `dist_h`, and `bridge_dist` are all height-normalized dimensionless (`/h_ref`). There is **no** px-vs-h unit incompatibility block.
- parent `resid_mean` carries PR-C role-reversal evidence → parent role conflict;
- GT corr(score, resid_mean) ≈ **0.977** vs corr(score, dist_h) ≈ **0.658** → residual dominance within a shared unit;
- high-w regime (w≥0.9, ~10% of pool) has mean residual-term fraction ≈ **0.99**;
- speed-dependent mixing injects hidden context dependence.

Blocks: parent role conflict · residual dominance · regime-dependent `w`. Ranking utility as a production proxy is not order legitimacy.

### 3.5 `resid_mean` — `conditional_orderable`

| Field | Value |
|:--|:--|
| Safer | lower (short-gap motion fit) |
| Context | `short_gap_continuous_association` (gap ≤ 60 frames; observable without GT outcome) |
| V_i | 3/209 · tail **3/4** |
| PR-C | **role-reversal supported** (long-gap re-entry) |

Must **not** enter global arcs. Conditional arc is **proposal-only** for a later conditional-closure study.

### 3.6 `dir_cos` — `conditional_orderable`

| Field | Value |
|:--|:--|
| Safer | higher (direction continuity) |
| Context | same short-gap continuous regime |
| V_i | 39/209 (7 sequences) · tail 3/4 |
| PR-C | role-reversal supported |

Multi-sequence violations under median split plus PR-C long-gap reversal block global promotion. Conditional proposal only.

### 3.7 `speed_mismatch` — `conditional_orderable`

| Field | Value |
|:--|:--|
| Safer | lower (speed continuity) |
| Context | same short-gap continuous regime |
| V_i | 57/209 · tail **4/4** |
| PR-C | role-reversal supported |

Strongest motion violation concentration on the protected tail. Global promotion forbidden by PR-C binding.

### 3.8 `gap` — `context_only`

| Field | Value |
|:--|:--|
| Declared safer | lower (audit convention) |
| Role | regime descriptor |
| V_i | 9/209 · p40→p60 flips 15/209 |

Long gap is not intrinsically unsafe (true re-entries exist). `gap` defines the observable context for motion conditioning; it is not a valid global monotone order dimension.

## 4. Pairwise / shell notes

- Pairwise `V_ij`: [`pairwise_violation_profile.csv`](evidence/boolean_atom_partial_order_20260711/pairwise_violation_profile.csv). Shared violations identify coupling candidates only (e.g. motion pairs co-violate); they do **not** estimate a cell-risk landscape.
- Shell contribution + protected-tail membership: in [`threshold_sensitivity.json`](evidence/boolean_atom_partial_order_20260711/threshold_sensitivity.json) → `shell_contribution`.
- Far-Hamming tail remains **protected GT mass** (never a veto).

## 5. Dependency DAG (composites)

[`atom_dependency_graph.json`](evidence/boolean_atom_partial_order_20260711/atom_dependency_graph.json):

```text
h_lost_raw, h_cand_raw  →  log_h_ratio
lost foot, cand foot, lost_exit_velocity, gap, h_ref → fwd_resid
lost foot, cand foot, cand_entry_velocity, gap, h_ref → bwd_resid
fwd_resid, bwd_resid    →  resid_mean
lost_exit_velocity, lost foot, cand foot → dir_cos
  (cos(v_lost_exit, displacement); NOT cand entry velocity)
lost_exit_speed, cand_entry_speed → speed_mismatch
resid_mean, dist_h, lost_exit_speed → score_m_bridge   (blocked weighted composite)
lost/cand foot, velocities, gap, h_ref → bridge_dist   (motion-extrapolation; blocked)
lost/cand foot, h_ref → dist_h                         (pure geometry leaf)
```

Edges are provenance only — not closure arcs and not weights. Formula-level parents for `dir_cos` / residuals match `build_relink_candidates.pair_features`.

## 6. Allowed / forbidden global order contract

### Allowed ([`allowed_global_order.json`](evidence/boolean_atom_partial_order_20260711/allowed_global_order.json))

```text
z_i = 1  ⇔  declared safer side
global dimensions: dist_h↓, log_h_ratio↓
reject domains in a later study must be downward-closed on this orientation
graph vertices (next task only): Boolean cells on the global_orderable subcube
optimization weights / MWC solve: forbidden here
```

### Forbidden ([`forbidden_order.json`](evidence/boolean_atom_partial_order_20260711/forbidden_order.json))

| Dimension | Why forbidden globally |
|:--|:--|
| `speed_mismatch`, `dir_cos`, `resid_mean` | PR-C role-reversal; conditional only |
| `bridge_dist` | motion-extrapolation composite; related to PR-C long-gap regime |
| `score_m_bridge` | parent role conflict + residual dominance (units OK) |
| `gap` | regime descriptor |

Conditional arc proposals for the three motion atoms are marked **proposal-only**.

## 7. Identifiability boundary (restated)

```text
not observed ≠ unsafe
zero exposure ≠ an ordering proof
per-cell risk field remains non-identifiable (Step-0)
```

This audit uses only GT placement morphology, PR-C mechanism evidence, threshold sensitivity diagnostics, and composite scale-guard — not barrier heights, merge trees, or fitted risk surfaces.

## 8. Downstream routing

| If terminal | Action |
|:--|:--|
| `GLOBAL_PARTIAL_ORDER_READY` + `ACCEPTED_WITH_LIMITS` | After merge, open a **new** restricted global-closure prototype using **only** `{dist_h, log_h_ratio}`; vs frozen OR-tail under exact GT-UCB; candidate-only |
| Conditional proposals | May seed a **separately reviewed** conditional-representation task (`bridge_dist` + motion); not part of the global prototype |
| Must not | Put `bridge_dist` or other conditional atoms in the global solve; combine audit with MWC in the same PR; promote motion / `score_m_bridge` / `gap` to global arcs without independent evidence + owner override |

## 9. Must not

- Treat this note as evidence_ledger promotion;
- Run MWC / min-cut / rule search in this unit;
- Promote motion atoms, `bridge_dist`, or `score_m_bridge` to global order from the same pooled substrate without resolving the documented contradictions;
- Veto the protected escape tail;
- Change production presets, defaults, tracker behavior, or closed OR-tail policy;
- Claim L2+ or multi-sequence confirmation from this L1 contract.
