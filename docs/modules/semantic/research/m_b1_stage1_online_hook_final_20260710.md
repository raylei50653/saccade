# M-B1 Stage 1 online hook — final (CLOSED)

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: stage1-online-final = this file; hook ABI contract = m_b1_portable_or_tail_hook_contract_20260709.md -->

**Role:** Canonical Stage 1 eng + evidence close (wire + e2e + B-audit + A0 + det + runtime).  
**Hook ABI contract (retain separately):** [m_b1_portable_or_tail_hook_contract_20260709.md](m_b1_portable_or_tail_hook_contract_20260709.md)  
**Plan (retain):** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)  
**Offline freeze:** [m_b1_repaired_eps0_loo_pass_candidate_20260709.md](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)  
**Stage 2 final:** [m_b1_5_stage2_d_online_final_20260710.md](m_b1_5_stage2_d_online_final_20260710.md)  
**Consolidation:** [m_b1_doc_consolidation_report_20260710.md](m_b1_doc_consolidation_report_20260710.md)

| Study | Arms |
|:--|:--|
| [`m_b1_hook_ab_20260710T062345Z`](../../../../out/signal_study/m_b1_hook_ab_20260710T062345Z/) | A1 + freeze B only |
| [`m_b1_hook_ab_20260710T064657Z_stage1b`](../../../../out/signal_study/m_b1_hook_ab_20260710T064657Z_stage1b/) | A1 + B + **P** + **F** controls |
| [`m_b1_hook_ab_20260710T071001Z_stage1_close`](../../../../out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/) | A1 + **B-audit** + B-repeat + rebased A0 |

---

# Part A — Evidence (from Stage 1 e2e close)

## Milestone status (locked language)

```text
stage1a_evaluation_entry:     PASSED
frozen_policy_online_relevance: NULL_support_mismatch
stage1b_action_path:          PASSED   (plumbing controls P/F)
stage1b_eng_milestone:        PASSED
online_baudit:                PASSED   (244-row full table; recon ok)
a0_identity:                  strict_pass (rebased A0 stamp)
determinism_repeated_run:     PASSED
runtime_overhead:             PASSED   (wall-clock named; pure kernel NOT_MEASURED)
stage1_overall:               CLOSED
e2e_safe_for_default_off:     yes  (freeze B null-effect)
production_preset:            unchanged
```

### Allowed claim

> Stage 1 engineering closure is complete: **trusted observation + intervention  
> substrate** (policy load, evaluation-entry, action-path under controls, full  
> B-audit table, rebased strict A0, determinism, named runtime).  
> Hook **mechanism** validated. Frozen offline thr remains  
> **null online relevance** (`triggered=0`, `policy_effect_supported: no`).  
> `e2e_safe_for_default_off=yes` = null-effect mount is safe — **not** online power.

### Forbidden claim

> “Frozen offline thr is online-effective” / production preset / default-on /  
> “pure kernel overhead = A1−B wall-clock” /  
> Stage 1 null-effect ⇒ Stage 2 thr already justified.

**Stage 2 entry:** [m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md).

---

## Stage 1a — evaluation-entry (freeze A1 / B)

| | IDF1 | AssA | HOTA | MOTA | IDs | FP |
|:--|--:|--:|--:|--:|--:|--:|
| A1 hook-off | 80.3 | 72.8 | 74.3 | 81.9 | 359 | 2067 |
| B freeze on | 80.3 | 72.8 | 74.3 | 81.9 | 359 | 2067 |
| Δ | 0 | 0 | 0 | 0 | 0 | 0 |

| counter | A1 | B |
|:--|--:|--:|
| hook_eligible | **0** | **244** |
| hook_rejected | 0 | **0** |
| atom0..4 | 0 | **0** |

- A1≡B MOT hashes (all 7 seq)
- Offline pairs replay n_rejected=8721 ≠ online pruning power

**Proved:** candidate reaches evaluator; freeze thr never fires on \(D_{\text{online}}\).

### Support mismatch

```text
D_online = D_offline ∩ {bdist≤0.4} ∩ {0.6≤hr≤1.7} ∩ {other gates}
```

Freeze q85 thr lives outside online support → vacuous thr is domain error, not eng failure.

---

## Stage 1b — action-path plumbing controls

Pre-specified (not metric-picked; not GT-safe; not candidate/preset):

| Arm | control_arm | atom0 thr | others |
|:--|:--|--:|:--|
| **P** activation | `activation` | **0.2** | disabled (1e9) |
| **F** force-reject | `force_reject` | **−1** | disabled |

Fixtures: `scripts/tools/fixtures/m_b1_stage1/`

### Results (`m_b1_hook_ab_20260710T064657Z_stage1b`)

| check | P activation | F force-reject |
|:--|--:|--:|
| hook_eligible | 265 | 305 |
| atom0 | **168** | **305** |
| hook_rejected | **168** | **305** |
| rejected == atom0 / eligible | yes | yes (rej==elig) |
| result differs from A1 (required) | **yes** | **yes** |
| pass | **yes** | **yes** |

**Proved under controls:**

```text
signal > thr → atom counter ↑ → hook_rejected ↑ → candidate suppressed → MOT hash ≠ A1
```

---

## Online B-audit full event table (`…T071001Z_stage1_close`)

Native audit ring exports every baseline-ok pair at the hook evaluation point.

| field family | content |
|:--|:--|
| identity | sequence, frame, event_id, cand/lost slot + track ids, join_key |
| signals | score_m_bridge, abs_log_h, dist_h, abs_ratio_m1, resid_mean |
| margins | thr_margin_0..4 = signal − thr |
| fire | atom_bitmask, fired_atom_ids, n_atoms_fired, fire_class |
| decision | rejected_by_hook; host ranking baseline vs hook (lower bdist wins) |

| count | value |
|:--|--:|
| n_hook_eligible | **244** |
| n_zero_fire | **244** |
| n_singleton / n_cofire | 0 / 0 |
| n_rejected | **0** |
| native hook_eligible align | **yes** |
| reconciliation errors | **[]** |

Artifacts:

```text
hook_candidate_events.{csv,parquet}   # full table (zero-fire retained)
rejected_events.{csv,parquet}         # empty (derived)
atom_summary.csv
per_sequence_summary.csv
baudit_summary.json
```

Derived summaries are **group-by only** from the full table; counters fail-closed if misaligned.

---

## Strict A0 identity

| stamp | result |
|:--|:--|
| Legacy `results/MOT17_eval_m_b2_bridge_on_20260709T094646Z` | soft 6/7 (MOT17-04 float box drift when keys match; multi-seq global ids) |
| **Rebased** `…/a0_rebased_from_A1/` | **strict_pass** vs this study’s A1 |

Provenance: `A0_STAMP.json` records source arm, git commit, legacy compare, and forbids upgrading soft legacy identity to strict without rebase.

---

## Determinism

| run | aggregate MOT hash |
|:--|:--|
| B_audit | `14976756f55256ba…` |
| B_repeat | `14976756f55256ba…` (identical) |

Per-seq result hashes match on all 7 sequences. Event-table hash recorded. Empty repeated-hash arrays are **not** treated as pass.

Also: B_audit ≡ B_repeat implies audit ring is decision-neutral (export-only).

---

## Runtime contract

| name | meaning (wall-clock arm seconds) |
|:--|:--|
| hook_disabled | A1 full e2e |
| hook_enabled_policy | B (with audit when baudit-on) |
| hook_enabled_policy_no_audit | B_repeat |
| audit_enabled | B with audit ring |
| pure_policy_kernel_overhead | **NOT_MEASURED** |

Must **not** claim A1/B wall-clock delta as pure policy kernel overhead.

---

## Reproduce

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp>_stage1_close \
  --run-e2e --run-baudit --run-determinism --rebase-a0 --prior-controls-ok
```

Stage 1b controls (if re-running P/F):

```bash
... --run-e2e --run-action-path-controls
```

---

## Must not

- claim Stage 1 CLOSED from freeze vacuous A1/B alone
- thr search as Stage 1 fix
- soft A0 → silent strict identity
- production preset / default-on
- treat control thr=0.2 as production candidate
- treat wall-clock A1/B delta as kernel overhead

---

# Part B — Wire inventory (from Stage 1 wire note)

## Status

```text
Stage 1 overall: CLOSED
Stage 1a evaluation-entry: PASSED
Stage 1b action-path (controls): PASSED
frozen online relevance: NULL (support mismatch)

online_hook:              wired_default_off
freeze e2e B:             eligible=244 rejected=0 A1≡B
control P/F:              atom fire + reject + decision change proven
online B-audit:           PASSED (244-row full table; recon ok)
strict A0 identity:       PASSED (rebased stamp)
determinism_repeated_run: PASSED
runtime_contract:         PASSED (wall-clock named; pure kernel NOT_MEASURED)
e2e_safe_for_default_off: yes (freeze null-effect only)
production_preset:        unchanged
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
| Native thr inject + audit ring | `tracker_gpu.cu` `set_research_portable_or_tail` · propose-kernel OR-tail + event write |
| Audit drain API | `get_portable_or_tail_audit_events` / overflow / clear |
| A/B + B-audit + det runner | `scripts/tools/run_m_b1_hook_ab.py` (`--run-baudit --run-determinism --rebase-a0`) |
| Control fixtures | `scripts/tools/fixtures/m_b1_stage1/` (activation thr=0.2, force_reject thr=-1) |

### Flag semantics

```text
unset / empty  → hook OFF (no thr load, no kernel policy path)
path set       → load portable_policy.json fail-closed (freeze thr+hash+op='>' lock);
                 reject if ANY of 5 frozen tails fire
--research-portable-or-tail-audit
               → online B-audit event ring (export-only; no decision change)
               → per-seq _portable_or_tail_audit_<seq>.json
offline pairs tables
               → scripts/tools/run_m_b1_hook_ab.py --offline-events-only
                 (writes offline_hook_candidate_events.*)
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
| full table | `offline_hook_candidate_events.parquet` (+ csv) |

---

## Stage 1 close study

`out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/`

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp>_stage1_close \
  --run-e2e --run-baudit --run-determinism --rebase-a0 --prior-controls-ok
```

| check | result |
|:--|:--|
| online events | **244** (all zero-fire) |
| recon + native counter align | pass |
| rebased A0 strict | pass |
| B_audit ≡ B_repeat hashes | pass |
| runtime contract named | pass |
| stage1_overall | **CLOSED** |

**Next (Stage 2):** redefine safe-negative on \(D_{\text{online}}\) using the B-audit table — not offline q85 re-fit first.

**Stop** — no production thr search / no preset in Stage 1 PRs.

---

---

# Part C — Consolidation note

Absorbed files (claims_and_evidence_preserved; see consolidation report):

- `m_b1_hook_stage1_e2e_20260710.md` — blob `25d770fbe43a878feac901e9076690c39e9a5911` (Part A)
- `m_b1_hook_stage1_wire_20260710.md` — blob `37aed64990cc69c609ae74e667e3e2ff731ff4c9` (Part B)

**Not absorbed (standalone contracts still living):**

- `m_b1_portable_or_tail_hook_contract_20260709.md` (code/doc ABI)
- `m_b1_to_m_b1_5_two_stage_plan_20260710.md` (runner/plan body)

```text
research_claims: unchanged from absorbed sources
production_preset: unchanged
stage1_overall: CLOSED
frozen_policy_online_relevance: NULL_support_mismatch
```
