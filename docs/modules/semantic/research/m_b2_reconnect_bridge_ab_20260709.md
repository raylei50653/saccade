# m B2 — reconnect state: bridge ON vs OFF (`mamba_whole_graph_m`)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Date:** 2026-07-09  
**Status:** **RESEARCH live note (D3)** — B2 IDs-state; pairs with B1 signal note  
**Study (numbers master):** [`out/signal_study/m_b2_bridge_ab_20260709T094646Z/`](../../../../out/signal_study/m_b2_bridge_ab_20260709T094646Z/)  
`context.json` · `metrics_reconnect.json` · `events_bridge_{on,off}.csv`  
**Sibling B1 (signal, not e2e):** [m_b1_research_history_20260709_20260710.md](m_b1_research_history_20260709_20260710.md)  
**Contract:** [signal_table_schema.md](../../../research/contracts/signal_table_schema.md) §0.1 — **B1 ≠ B2**  
**Tool:** `scripts/eval/diagnostics/reconnect_rate.py` (`--json-out` / `--events-out`)

> Master rates / e2e live in study_dir. This file is pointer + leverage reading.  
> Re-run → new stamp; change pointer only.

---

## TL;DR（as-of study）

On production-like **m** (interp **ON**, double-buffer, SDP), ablating only `relink_bridge`:

| Side | e2e (OVERALL) | reconnect success rate |
|:--|:--|:--|
| **bridge OFF** | IDF1≈78.7 · AssA≈71.4 · IDs **423** · FP≈2794 | **~31.5%** (515 opps) |
| **bridge ON** | IDF1≈80.3 · AssA≈72.8 · IDs **359** · FP≈2067 | **~39.2%** (469 opps) |
| **Δ (on − off)** | IDF1 **+1.6** · AssA **+1.4** · IDs **−64** · FP **−727** | rate **+~7.8 pp** |

**Leverage (read with B1):**

1. **B2 moves:** bridge ON raises same-pred-id resume rate and improves e2e IDs/AssA/IDF1 on this m stamp — not a null.  
2. **B1 explained the ceiling, not the on/off delta:** offline hard-pool AUC ~0.76 says geometry is a mid-strength ranker among near pairs; B2 shows the **online mechanism still recovers a non-trivial fraction of lost→resume events**. Do not collapse “AUC mid” into “bridge useless”.  
3. **Short gaps dominate success** (1–10 frames ~57% on / ~45% off); long gaps stay hard (~0–10%). Geometry/time still decay with gap — consistent with B1 gap story.  
4. **Fewer opportunities under bridge ON** (515→469): track structure changes (earlier reconnect / different birth-death), so raw opp count is not a pure “same event set” A/B — always report rate **and** n.  
5. **Not a GO to retune gates from this alone:** RESEARCH ablate; production m already defaults bridge ON. Noise band for tiny IDF1 knobs still applies to *further* tweaks, not to denying the observed +1.6 here.

**Not claimed:** B1 thr tables predict B2 rates 1:1; s offline numbers are m; appearance not needed forever.

---

## 1. Question & definition (B2)

**Question:** After a GT person is uncovered again following a gap, does the tracker resume the **same pred id**?

| Term | Definition (`reconnect_rate.py`) |
|:--|:--|
| Match | greedy IoU ≥ 0.5 per frame |
| Opportunity | coverage resumes after ≥ `min_gap` (default 1) lost frames |
| Success | `pred_id(resume) == pred_id(before gap)` |

This is **state / mechanism**, not offline pair ranking (B1).

---

## 2. Substrate recipe (matched ablate)

Both arms: `mamba_whole_graph_m` · SDP · double-buffer · **interpolate ON** (production default).  
Only knob: `--relink-bridge-enabled` vs `--no-relink-bridge-enabled`.

```bash
STAMP=...
# bridge OFF
uv run python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --detect-barrier event --no-relink-bridge-enabled \
  --output results/MOT17_eval_m_b2_bridge_off_${STAMP}
# bridge ON
uv run python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --detect-barrier event --relink-bridge-enabled \
  --output results/MOT17_eval_m_b2_bridge_on_${STAMP}

uv run python scripts/eval/diagnostics/reconnect_rate.py \
  --pred-dir results/MOT17_eval_m_b2_bridge_on_${STAMP} \
  --baseline-dir results/MOT17_eval_m_b2_bridge_off_${STAMP} \
  --label bridge_on --baseline-label bridge_off \
  --json-out out/signal_study/m_b2_bridge_ab_${STAMP}/metrics_reconnect.json \
  --events-out out/signal_study/m_b2_bridge_ab_${STAMP}/events_bridge_on.csv \
  --baseline-events-out out/signal_study/m_b2_bridge_ab_${STAMP}/events_bridge_off.csv
```

**Paths this stamp:**

| Role | Path |
|:--|:--|
| off MOT | `results/MOT17_eval_m_b2_bridge_off_20260709T094646Z/` |
| on MOT | `results/MOT17_eval_m_b2_bridge_on_20260709T094646Z/` |
| study | `out/signal_study/m_b2_bridge_ab_20260709T094646Z/` |

**Why not reuse B1 MOT dump:** B1 substrate is **interp-off** (raw death/birth for pairs). B2 production-like needs **interp-on** so reconnect opportunities match the shipped tracker path.

---

## 3. Gap stratification (orientation)

Open `metrics_reconnect.json` for cells. Shape as-of:

- **1–10:** largest opp mass; bridge ON lifts success most clearly.  
- **11–30:** still positive but lower absolute rate.  
- **31+:** sparse; rates noisy; do not headline long-gap alone.

Events CSVs support custom slices (disp, seq) without re-running MOT.

---

## 4. Side-by-side with B1

| Line | Asks | This round (m) |
|:--|:--|:--|
| **B1** | Can `bridge_dist` rank true offline pairs? | full AUC ~0.87 / hard ~0.76; base-rate wall on thr |
| **B2** | Does online bridge keep pred id across gaps? | rate ~31.5%→~39.2%; e2e IDF1 +1.6 / IDs −64 |

**Joint reading:** geometry is good at rejecting far junk (B1 full) and mid at hard near pairs (B1 hard); the live bridge still converts a slice of gaps into correct resumes (B2) with measurable e2e identity gain. Next research doors (appearance / depth / long-gap) should target **where B2 still fails** (long gaps, hard near collisions), not re-litigate “is full-pool AUC high”.

---

## 5. Verdicts

| Claim | Verdict |
|:--|:--|
| B2 tooling + study_dir export works on m | **GO (D3 smoke)** |
| Bridge ON improves reconnect rate vs OFF (this stamp) | **GO (signal)** |
| Bridge ON improves e2e IDs/AssA/IDF1 vs OFF (this stamp) | **GO (e2e, matched ablate)** |
| Change production bridge defaults from this note | **N/A** — m already ON; no silent retune |
| B1 AUC substitutes for B2 | **NO** |

---

## 6. Next

- **D4 (optional):** `meta.json` auto, recipe lock in schema, ledger promote if desired.  
- Optional: reconnect under **interp-off** (align death set with B1) — separate study_id.  
- Optional: m production hard gate as second B1 hard def (still B1).  
- Cross-check false reconnects (success that are wrong person) needs GT id continuity analysis beyond current binary success — out of scope here.
