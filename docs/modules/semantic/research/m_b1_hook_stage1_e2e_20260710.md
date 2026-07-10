# M-B1 Stage 1 e2e — portable OR-tail hook A1 vs B

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 1 **formal close** + e2e evidence (not production GO).  
**Study:** [`out/signal_study/m_b1_hook_ab_20260710T062345Z/`](../../../../out/signal_study/m_b1_hook_ab_20260710T062345Z/)  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)  
**Plan:** [m_b1_to_m_b1_5_two_stage_plan_20260710.md](m_b1_to_m_b1_5_two_stage_plan_20260710.md)  
**Wire:** [m_b1_hook_stage1_wire_20260710.md](m_b1_hook_stage1_wire_20260710.md)

---

## Stage 1 formal close

```text
Stage 1 engineering milestone: PASSED / CLOSED
e2e_safe_for_default_off: yes
classification: online_effect_neutral_but_safe__vacuous_online_thr
production_preset: unchanged
```

> Result is **not** “policy had no effect therefore failed.”  
> It cleanly shows: **the hook engineering chain is valid**, but the **discriminating domain of the offline policy has almost no intersection with the actual online candidate domain**.

Evidence chain (complete):

| check | result |
|:--|:--|
| default-off does not enter policy eval | A1 `hook_eligible=0` |
| default-on evaluates candidates | B `hook_eligible=244` |
| A1/B MOT result hashes | **identical** (all 7 seq) |
| metrics Δ(B−A1) | **0** on IDF1/AssA/HOTA/MOTA/IDs/FP/FN |
| production preset | unchanged |

Therefore `e2e_safe_for_default_off=yes` **holds**.

### Three locked conclusions

1. **Engineering safety** — hook-off invisible; hook-on does not damage baseline.
2. **Online wiring** — eligible=244 proves policy path is live (not dead code / mis-indexed counters).
3. **Offline hypothesis lacks online portability** — offline 8721 rejects mostly sit where production gates already exclude pairs; offline pruning power must not be read as online pruning power.

### Conditional-domain / support mismatch

```text
D_offline = all recorded pairs

D_online  = D_offline
            ∩ { bdist ≤ 0.4 }
            ∩ { 0.6 ≤ hr ≤ 1.7 }
            ∩ { other baseline gates pass }
```

q85 thresholds were estimated on \(D_{\text{offline}}\) then applied on the truncated \(D_{\text{online}}\). Thresholds falling outside online support is the **natural** outcome of that domain error — not merely “thr too big.”

### What Stage 1 does **not** authorize

- Immediate thr re-fit on offline pairs
- Claiming Stage 2 success from this null online reject rate
- Production preset / default-on

**PR boundary (this close):** wire + A/B runner + counters + Stage 1 evidence only — **no** online thr calibration.

### Next (ordered)

1. **Online full B-audit event table** for the 244 baseline-ok pairs (signals, GT/FP, atom margins, final association) — still pending; do this **before** thr re-fit.
2. Stage 2 study object becomes:

```text
max_C  FP_removed(C | D_online)
s.t.   GT_hurt(C | D_online) ≤ ε
```

   i.e. safe-negative mass **inside** the production-accepted conditional domain — not another all-pairs q85.

3. If audit shows insufficient FP mass among the 244 → placement may be **too late** (safe negatives already consumed upstream). Then compare: keep placement + fine cal · move hook earlier · ranking/margin instead of reject.

---

## Headline (numeric)

```text
e2e_safe_for_default_off: yes
classification: online_effect_neutral_but_safe__vacuous_online_thr
production_preset: unchanged
```

Default-off research hook is **safe** on MOT17-SDP 7-seq (`mamba_whole_graph_m`): A1≡B metrics and result hashes, zero online rejects.  
Null online effect is explained by **domain support mismatch**, not a broken hook.

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

## Why online thr is vacuous (support detail)

Online propose kernel only reaches the OR-tail for pairs that already pass production gates:

```text
bdist <= bridge_px          # preset relink_bridge_px = 0.4 (h-normalized)
hr ∈ [bridge_h_lo, bridge_h_hi]  # [0.6, 1.7]
```

Frozen offline thr (q85 of **all** offline pairs):

| atom | thr | vs online gate |
|:--|--:|:--|
| `score_m_bridge` / bdist | **11.91** | already ≤ **0.4** → atom0 impossible |
| `abs_log_h` | **1.35** | hr∈[0.6,1.7] ⇒ max\|log hr\|≈0.51 → atom1 impossible |
| `dist_h` | **6.73** | constrained by bdist≤0.4 blend → unreachable |
| `abs_ratio_m1` | **2.09** | max\|hr−1\|≈0.7 under height gate → atom3 impossible |
| `resid_mean` | **14.04** | residual scale gated with bdist → unreachable |

Offline FP mass (8721) lives **outside** \(D_{\text{online}}\). That is Stage 1’s scientific result, not a wiring failure.

---

## Acceptance checklist (Stage 1 — CLOSED)

| Item | Result |
|:--|:--|
| Extension exposes `set_research_portable_or_tail` | yes |
| A1 hook-off metrics ≈ B2 / soft A0 identity | yes (6/7 hash + metrics) |
| B hook-on metrics + native counters | yes |
| `e2e_safe_for_default_off` published | **yes** |
| Domain diagnosis (support mismatch) recorded | **yes** |
| Online full B-audit event table | **pending** (next ordered step; not Stage 1 blocker) |
| Production preset change | **no** |
| Stage 2 thr / domain remodel | **not started** (separate PR) |

---

## Must not (reaffirmed)

- treat offline `n_rejected=8721` as online effect
- silent default-on / preset flip
- thr sweep / rule search as Stage 1 “fix”
- claim production GO from this note
- skip B-audit and jump to offline-style q85 re-fit

---

## Reproduce

```bash
bash scratch/ab_env.sh uv run python scripts/tools/run_m_b1_hook_ab.py \
  --policy out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_hook_ab_<stamp> \
  --run-e2e
```
