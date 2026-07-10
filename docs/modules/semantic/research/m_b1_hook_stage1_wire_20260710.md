# M-B1 Stage 1 wire — default-off portable OR-tail hook

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 implementation status note (not e2e results).  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Eng contract:** [m_b1_portable_or_tail_hook_contract_20260709.md](m_b1_portable_or_tail_hook_contract_20260709.md)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)

---

## Status

```text
composite: online_hook_wired__e2e_pending

online_hook:              wired_default_off
e2e_validation:           pending
e2e_safe_for_default_off: no
production_preset:        unchanged

Stage 1 code:        WIRED (default-off)
offline event table: PASS (n_rejected=8721 = freeze FP)
                     = offline pairs replay only
                     ≠ online B-audit / runtime event table
online/e2e A/B:      PENDING
Stage 2:             NOT STARTED (separate PR)
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
path set       → load portable_policy.json fail-closed; reject if ANY of 5 tails fire
audit flag     → separate from policy; for full-table export (offline table always via runner)
```

### Native counters (`get_relink_debug`)

After host cursor:

| dbg index | meaning |
|--:|:--|
| 2 | bridge_attempts |
| 3 | bridge_accepts |
| 4 | hook_eligible |
| 5 | hook_rejected |
| 6–9 | atom0–3 fires |
| 10 | app_veto |
| 11 | atom4 fires |

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

**Minimum eng milestone (PR #87 if kept narrow):**

1. Rebuild/load `saccade_tracking_ext` with `set_research_portable_or_tail` (built under `build/`).
2. A1: hook-off e2e vs trusted B2 baseline identity.
3. B: hook-on e2e; metrics / reconnect / runtime / determinism + native counters.
4. Publish `e2e_safe_for_default_off: yes/no` + classification.

**Pending for full Stage 1 artifact freeze (do not imply done by offline table):**

5. Online B-audit full event table (candidate universe at hook decision point).
6. Online fired-atom / singleton vs co-fire / per-seq rejected on online events.
7. Decision-change / reconnect-change joins where required by plan §7–8.

**Stop** — no Stage 2 / no preset.

---

## Must not

- threshold sweep / rule search / zone-gap atoms
- silent default-on
- production preset change
- Stage 2 in the same PR
