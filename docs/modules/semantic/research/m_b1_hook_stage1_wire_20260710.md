# M-B1 Stage 1 wire — default-off portable OR-tail hook

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 wire inventory.  
**Evidence:** [m_b1_hook_stage1_e2e_20260710.md](m_b1_hook_stage1_e2e_20260710.md)  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

---

## Status

```text
Stage 1 overall: OPEN
Stage 1a evaluation-entry: PASSED
Stage 1b action-path (controls): PASSED
frozen online relevance: NULL (support mismatch)

online_hook:              wired_default_off
freeze e2e B:             eligible=244 rejected=0 A1≡B
control P/F:              atom fire + reject + decision change proven
e2e_safe_for_default_off: yes (freeze null-effect only)
production_preset:        unchanged
online B-audit:           PENDING
strict A0 identity:       NOT MET (soft 6/7)
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
| A/B + Stage 1b control runner | `scripts/tools/run_m_b1_hook_ab.py` (`--run-action-path-controls`) |
| Control fixtures | `scripts/tools/fixtures/m_b1_stage1/` (activation thr=0.2, force_reject thr=-1) |

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

**Done:** Stage 1a entry + Stage 1b plumbing controls (see [e2e](m_b1_hook_stage1_e2e_20260710.md)).

**Still pending for Stage 1 overall CLOSED:**

1. Online B-audit full event table (zero/singleton/cofire/rejected/decision-changed).
2. Strict A0 identity or rebased A0 stamp.
3. Hook-on repeated-run hashes + honest runtime overhead rows.

**After that / Stage 2:** redefine safe-negative on \(D_{\text{online}}\) — not offline q85 re-fit first.

**Stop** — no production thr search / no preset in this PR.

---

## Must not

- threshold sweep / rule search / zone-gap atoms
- silent default-on
- production preset change
- Stage 2 in the same PR
