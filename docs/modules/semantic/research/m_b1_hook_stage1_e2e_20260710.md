# M-B1 Stage 1 e2e — portable OR-tail hook (1a entry + 1b controls)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 evidence with **honest milestone split** (not production GO).  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)

| Study | Arms |
|:--|:--|
| [`m_b1_hook_ab_20260710T062345Z`](../../../../out/signal_study/m_b1_hook_ab_20260710T062345Z/) | A1 + freeze B only |
| [`m_b1_hook_ab_20260710T064657Z_stage1b`](../../../../out/signal_study/m_b1_hook_ab_20260710T064657Z_stage1b/) | A1 + B + **P** + **F** controls |

---

## Milestone status (locked language)

```text
stage1a_evaluation_entry:     PASSED
frozen_policy_online_relevance: NULL_support_mismatch
stage1b_action_path:          PASSED   (plumbing controls P/F)
stage1b_eng_milestone:        PASSED
online_baudit:                PENDING
a0_identity:                  soft_pass / strict unresolved (6/7)
determinism_repeated_run:     PENDING
runtime_overhead:             PENDING
stage1_overall:               OPEN
e2e_safe_for_default_off:     yes  (freeze B null-effect only)
production_preset:            unchanged
```

### Allowed claim

> Policy loading and **evaluation-entry** wiring are valid;  
> online **rejection/action** chain is proven **under Stage 1b control arms**  
> (not by freeze B alone).

### Forbidden claim

> “The full hook engineering chain is valid” / “Stage 1 CLOSED”  
> from freeze A1/B vacuous results alone.

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
- A0: **strict fail** (MOT17-04 differs); **soft 6/7**
- Offline pairs replay n_rejected=8721 ≠ online pruning power

**Proved:** candidate reaches evaluator; freeze thr never fires on \(D_{\text{online}}\).  
**Not proved by B alone:** atom→reject→suppress→decision change.

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
| **P** activation | `activation` | **0.2** (midpoint of bridge_px=0.4) | disabled (1e9) |
| **F** force-reject | `force_reject` | **−1** | disabled |

Fixtures: `scripts/tools/fixtures/m_b1_stage1/`

### Results (`m_b1_hook_ab_20260710T064657Z_stage1b`)

| check | P activation | F force-reject |
|:--|--:|--:|
| hook_eligible | 265 | 305 |
| atom0 | **168** | **305** |
| hook_rejected | **168** | **305** |
| rejected == atom0 / eligible | yes | yes (rej==elig) |
| result differs from A1 | **yes** | **yes** |
| pass | **yes** | **yes** |

Eligible counts differ from freeze B (244) because rejects change track state — further evidence the action path is live.

**Proved under controls:**

```text
signal > thr → atom counter ↑ → hook_rejected ↑ → candidate suppressed → MOT hash ≠ A1
```

---

## Still open (why stage1_overall = OPEN)

| Contract item | Status |
|:--|:--|
| Online full event table (zero/singleton/cofire/rejected/decision-changed) | PENDING |
| Strict A1==A0 result-file identity | NOT MET (soft only) |
| Hook-on repeated-run hashes | PENDING |
| Hook-disabled / policy / audit runtime overhead (pure) | PENDING |

PR may merge Stage 1a+1b eng evidence while overall stays **OPEN**.

---

## Reproduce

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp>_stage1b \
  --run-e2e --run-action-path-controls
```

---

## Must not

- claim Stage 1 CLOSED from freeze vacuous A1/B
- thr search as Stage 1 fix
- soft A0 → silent strict identity
- production preset / default-on
- treat control thr=0.2 as production candidate
