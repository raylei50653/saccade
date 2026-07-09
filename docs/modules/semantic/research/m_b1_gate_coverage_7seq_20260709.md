# m B1 — production-shaped L0 gate coverage (7-seq)

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Date:** 2026-07-09  
**Status:** **RESEARCH D1** — gate *coverage* map; not production GO  
**Ledger:** `m.prod_shaped.bulk_cover` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)（批量地圖；單訊號深度另列）  
**Study (numbers master):** [`out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/`](../../../../out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/)  
**Pairs substrate:** reuses `m_b1_smoke_20260709T092543Z` 7-seq pairs (bridge/interp off)  
**Contract:** [signal_table_schema §0.4–0.5](../../../research/eval/signal_table_schema.md) · tool `audit_relink_safe_reject.py`  
**Sibling:** [m_b1_bridge_discriminability](m_b1_bridge_discriminability_20260709.md) (AUC/thr) · [m_b2 reconnect](m_b2_reconnect_bridge_ab_20260709.md)


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

> **Ops note:** full 7-seq MOT on this host is ~30s — treat as cheap. Re-run substrate freely; do not gate experiments on “save the eval.”

---

## TL;DR

**Layer = L0 Support Gate coverage** (who fails production-shaped cut), **not** L1 score strength.

On the **full offline** `U_relink_pair` pool (wide gap enum, not live attempt set):

| Gate (m-shaped offline proxy) | GT_hurt | FP_removed | Read |
|:--|--:|--:|:--|
| `score_m_bridge > 0.4` (`relink_bridge_px`) | **70%** | **99%** | Defines operating region; **not** a safe-reject rule |
| `h_ratio ∉ [0.6, 1.7]` | **3.2%** | **54%** | Real scale gate; mild GT tax, large FP cut |
| composite px ∨ h | **70%** | **99%** | Dominated by px |
| s-shaped px 0.25 / h [0.75,1.33] on same m pairs | **82%** / **13%** h | higher | Confirms m’s relaxed gates recover more true pairs |

**By gap:** px GT_hurt rises with gap (1–10 ≈ **21%** → 151–300 ≈ **98%**). Long-gap true bridges are mostly *outside* live px — expected, not a bug.

**By seq (`prod_m_px_or_h_fail`):** 10 and 02 dominate GT mass hurt; 04 almost none (few pos). See study `prod_m_combo_by_seq`.

**ε=0 research headroom (full pool, still only ceilings):**  
`log_h` ceiling ~31% FP · `score_m_bridge` 1D ε=0 thr≈9.9 → ~20% FP · mid-point `bridge_dist` ceiling ~14% FP.  
**No interpretable multi-feature probe hit ε=0 with FP>0** in this round (same as prior smoke).

**Not claimed:** change production px/h; live fire counts; B2 reconnect.

---

## 1. What “coverage” means here

```text
reject = fail support gate  (pair would not pass production cut)
GT_hurt = true offline relink pair rejected
FP_removed = false pair rejected   (soft upper bound only)
```

Production bridge **accept** requires (live): score ≤ `relink_bridge_px` **and** height ratio in `[h_lo, h_hi]`.  
Offline we use:

- `score_m_bridge = w·½(fwd+bwd) + (1−w)·dist_h`, `w=√clip(s_lost/0.12)` — matches kernel formula shape  
- `h_ratio = h_lost_raw / h_cand_raw` — **raw endpoint**, not `ema_h` (proxy; not bit-exact)

Full offline pool ≠ live candidate set (live also has min_lost / ttl / hit_streak / uniqueness).  
⇒ **full-pool GT_hurt of px is mostly “outside operating region,”** not “bridge is broken.”

---

## 2. How to read the tables

Open study files; do not treat this markdown as fact-owner.

1. **`metrics_safe_reject_audit.csv`** — every rule × `all` / `gap_*` / `seq_*`  
2. **`metrics_safe_reject_summary.json`** — `prod_m_combo_by_seq`, 1D frontiers, all rows  
3. Prefer **gap-stratified** rows for px; all-pool px GT_hurt is a blunt instrument  

### Orientation (as-of study; verify in CSV)

**m px=0.4 by gap** (`prod_m_score_gt_px0.4`):

| gap bin | GT_hurt% (orient.) | FP_rm% |
|:--|--:|--:|
| 1–10 | ~21% | ~95% |
| 11–30 | ~59% | ~97% |
| 31–60 | ~76% | ~99% |
| 61–150 | ~87% | ~100% |
| 151–300 | ~98% | ~100% |

**m h-gate alone** (`prod_m_h_ratio_out_0.6_1.7`): all-pool GT_hurt ~3%, FP_rm ~54%.  
In gap 11–30 this rule is **ε=0** on that bin (no GT hurt in-bin) with ~50% FP cut — interesting **local** safe context; not yet a production rule (needs LOO / multi-bin and B2).

**s-shaped on m pairs:** tighter px/h → more GT_hurt (m relaxation is doing real coverage work on true pairs).

---

## 3. Implications for “慢慢補 gate 覆蓋面”

| Done this stamp | Still open |
|:--|:--|
| m production-shaped px + h on 7-seq pairs | Live `bridge_attempts/accepts` CSV |
| gap + **seq** coverage bins | Hard-pool-only production coverage table (optional column) |
| s-shaped contrast on same pairs | Interpretable ε=0 **context** rules beyond 1D ceilings |
| score_m_bridge 1D ε-frontier | U_cand IoU/Maha/cost-cap gates |
| | Other doors (depth / occ / handover) only when failure mode prioritised |

**Next cheap steps (no ceremony):**

1. Expand probe rules aimed at **ε=0** using h + extreme residual / long-gap far (bin-aware).  
2. Optional hard-pool (`bridge_dist≤1` or `score_m_bridge≤1`) re-audit of prod gates.  
3. Live 7-seq bridge ON → dump attempt stats when ready for B2-side coverage.

---

## 4. Reproduce

```bash
uv run python scripts/tools/audit_relink_safe_reject.py \
  --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv \
  --study-dir out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z \
  --write-study --by-gap --by-seq
```

Full substrate rebuild (when pairs stale): study `README.md` — 7-seq MOT ~30s.

---

## Related

- Tool: `scripts/tools/audit_relink_safe_reject.py` (`production_shaped_rules`, `--by-seq`)  
- Config surface: [config_surface §5](../../../research/tracker-decision/audit/config_surface.md)  
- Preset: `configs/presets/mamba_whole_graph_m.yaml` (`relink_bridge_px: 0.4`, h `[0.6, 1.7]`)  
