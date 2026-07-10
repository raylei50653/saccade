# M-B1 Stage 1 wire — default-off portable OR-tail hook

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 wire inventory (closed with e2e).  
**Close:** [m_b1_hook_stage1_e2e_20260710.md](m_b1_hook_stage1_e2e_20260710.md)  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Eng contract:** [m_b1_portable_or_tail_hook_contract_20260709.md](m_b1_portable_or_tail_hook_contract_20260709.md)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

---

## Status

```text
Stage 1: CLOSED
composite: online_hook_wired__e2e_safe_vacuous

online_hook:              wired_default_off
e2e_validation:           done (A1≡B, Δ=0)
e2e_safe_for_default_off: yes
classification:           online_effect_neutral_but_safe__vacuous_online_thr
production_preset:        unchanged

Stage 1 code:        WIRED (default-off)
offline event table: PASS (n_rejected=8721 = freeze FP)
                     = offline pairs replay only
                     ≠ online B-audit / runtime event table
online/e2e A/B:      DONE — study m_b1_hook_ab_20260710T062345Z
                     online eligible=244 rejected=0
                     (support mismatch vs prod gates — not eng failure)
Stage 2:             NOT STARTED — first do online B-audit, not thr re-fit
```

---

## What landed

| Piece | Location |
|:--|:--|
| Two-stage plan | `docs/modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md` |
| Policy loader (fail-closed) | `src/saccade/perception/eval/portable_or_tail.py` |
| Unit tests | `tests/unit/test_portable_or_tail.py` |
| CLI (default-off) | `--research-portable-or-tail-policy` · `--research-portable-or-tail-audit` · env `SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY` |
| Pipeline wire | `src/saccade/perception/eval/pipeline.py` |
| Native thr inject | `tracker_gpu.cu` `set_research_portable_or_tail` · propose-kernel OR-tail reject |
| A/B runner | `scripts/tools/run_m_b1_hook_ab.py` |

### Flag semantics

```text
unset / empty  → hook OFF (no thr load, no kernel policy path)
path set       → load portable_policy.json fail-closed (freeze thr+hash+op='>' lock);
                 reject if ANY of 5 frozen tails fire
--research-portable-or-tail-audit
               → RESERVED / NOT IMPLEMENTED (fail-closed if set)
               → online B-audit event export still PENDING
offline pairs tables
               → scripts/tools/run_m_b1_hook_ab.py --offline-events-only only
```

### Native counters

Device `d_relink_dbg_[i]` is copied into host `get_relink_debug()` at **index i+1**
(host[0] = archived cursor from `d_relink_cursor_`).

| meaning | native `d_relink_dbg_` slot | host `get_relink_debug()` index |
|:--|--:|--:|
| births | 0 | 1 |
| revived | 1 | 2 |
| bridge_attempts | 2 | 3 |
| bridge_accepts | 3 | 4 |
| hook_eligible | 4 | 5 |
| hook_rejected | 5 | 6 |
| atom0 score_m_bridge | 6 | 7 |
| atom1 abs_log_h | 7 | 8 |
| atom2 dist_h | 8 | 9 |
| atom3 abs_ratio_m1 | 9 | 10 |
| app_veto | 10 | 11 |
| atom4 resid_mean | 11 | 12 |

Do **not** use native slot numbers as host vector indices.

---

## Offline validation (as-of wire)

Study example: `out/signal_study/m_b1_hook_ab_*` from:

```bash
uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --offline-events-only
```

| check | result |
|:--|:--|
| n_rejected | **8721** (= freeze FP_removed) |
| recon (eligible = zero+singleton+cofire) | pass |
| GT_hurt offline | 0 (freeze-aligned) |
| full table | `hook_candidate_events.parquet` (+ csv) |

---

## Still required

**Stage 1 eng milestone — CLOSED** (see [e2e close](m_b1_hook_stage1_e2e_20260710.md)).

**Ordered next (post–Stage 1; separate PR):**

1. Online B-audit full event table for the 244 baseline-ok pairs.
2. Only then redefine Stage 2 domain on \(D_{\text{online}}\) (not offline q85 re-fit).
3. If FP mass insufficient → placement / ranking alternatives (plan + thread).

**Stop** — no thr re-fit / no preset in Stage 1 PRs.

---

## Must not

- threshold sweep / rule search / zone-gap atoms
- silent default-on
- production preset change
- Stage 2 in the same PR
