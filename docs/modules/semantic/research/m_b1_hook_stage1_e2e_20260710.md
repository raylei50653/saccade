# M-B1 Stage 1 e2e — portable OR-tail hook A1 vs B

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 e2e result note (not production GO).  
**Study:** [`out/signal_study/m_b1_hook_ab_20260710T062345Z/`](../../../../out/signal_study/m_b1_hook_ab_20260710T062345Z/)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Wire:** [m_b1_hook_stage1_wire_20260710.md](m_b1_hook_stage1_wire_20260710.md)

---

## Headline

```text
e2e_safe_for_default_off: yes
classification: online_effect_neutral_but_safe__vacuous_online_thr
production_preset: unchanged
```

Default-off research hook is **safe** on MOT17-SDP 7-seq (`mamba_whole_graph_m`): A1≡B metrics and result hashes, zero online rejects.  
It is also **null-effect online**: frozen offline q85 thr never fires inside the production bridge/height gates.

---

## Arms

| Arm | Policy | Preset |
|:--|:--|:--|
| A1 | hook **off** | `mamba_whole_graph_m` + SDP + double-buffer |
| B | frozen `portable_policy.json` **on** | same |
| A0 ref | `results/MOT17_eval_m_b2_bridge_on_20260709T094646Z` | trusted B2 stamp |

Policy: `out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json`  
`candidate_id = m_b1_repaired_eps0_loo_pass_20260709` · freeze hash lock enforced.

---

## Metrics (7-seq SDP)

| | IDF1 | AssA | HOTA | MOTA | IDs | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A1 (hook off) | 80.3 | 72.8 | 74.3 | 81.9 | 359 | 2067 | 17895 |
| B (hook on) | 80.3 | 72.8 | 74.3 | 81.9 | 359 | 2067 | 17895 |
| Δ(B−A1) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Matches published B2 bridge-on headline (IDF1≈80.3 · AssA≈72.8 · IDs 359 · FP≈2067).

**Result-file identity**

- A1≡B aggregate MOT hash: **yes** (all 7 seq)
- A0 strict hash: **no** (only `MOT17-04-SDP.txt` differs)
- A0 soft identity: **yes** (6/7 seq hash match; metrics match B2 stamp)

---

## Native counters (`get_relink_debug`)

| | bridge_attempts | bridge_accepts | hook_eligible | hook_rejected | atom0..4 |
|:--|--:|--:|--:|--:|--:|
| A1 | 878 | 189 | **0** | 0 | 0 |
| B | 878 | 189 | **244** | **0** | **0** |

Hook-off path does not enter policy eval (eligible=0).  
Hook-on evaluates 244 baseline-ok pairs and rejects **none**.

Offline pairs replay (same study): `n_rejected=8721` on the offline universe — **not** interchangeable with online counters.

---

## Why online thr is vacuous (Stage 1 finding)

Online propose kernel only reaches the OR-tail for pairs that already pass production gates, including:

```text
bdist <= bridge_px          # preset relink_bridge_px = 0.4 (h-normalized)
hr ∈ [bridge_h_lo, bridge_h_hi]  # [0.6, 1.7]
```

Frozen offline thr (q85 of **all** offline pairs, including pairs online would never accept):

| atom | thr | vs online gate |
|:--|--:|:--|
| `score_m_bridge` / bdist | **11.91** | must already be ≤ **0.4** → atom0 impossible |
| `abs_log_h` | **1.35** | hr∈[0.6,1.7] ⇒ max\|log hr\|≈0.51 → atom1 impossible |
| `dist_h` | **6.73** | constrained by bdist≤0.4 blend → practically unreachable |
| `abs_ratio_m1` | **2.09** | max\|hr−1\|≈0.7 under height gate → atom3 impossible |
| `resid_mean` | **14.04** | same residual scale as bdist → unreachable |

So offline FP mass (8721) lives **outside** the online baseline-ok set. Stage 1 correctly proves:

1. Wiring works (default-off invisible; default-on path counts eligible).
2. Frozen application is safe (A1≡B).
3. Frozen thr has **no online reject power** under current production gates.

Stage 1 **must not** re-fit thr (forbidden). Any thr re-calibration on the online candidate universe is **Stage 2 / separate PR** if pursued.

---

## Acceptance checklist (Stage 1 minimum eng milestone)

| Item | Result |
|:--|:--|
| Extension exposes `set_research_portable_or_tail` | yes |
| A1 hook-off metrics ≈ B2 / soft A0 identity | yes (6/7 hash + metrics) |
| B hook-on metrics + native counters | yes |
| `e2e_safe_for_default_off` published | **yes** |
| Online full B-audit event table | **still pending** (CLI audit fail-closed) |
| Production preset change | **no** |
| Stage 2 | **not started** |

---

## Must not (reaffirmed)

- treat offline `n_rejected=8721` as online effect
- silent default-on / preset flip
- thr sweep / rule search in Stage 1
- claim production GO from this note

---

## Reproduce

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --run-e2e
```
