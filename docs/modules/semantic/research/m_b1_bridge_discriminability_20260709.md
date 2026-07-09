# m B1 — bridge_dist discriminability on `mamba_whole_graph_m`

**Date:** 2026-07-09  
**Status:** **RESEARCH live note (D2)** — first m-line B1 read; not a production GO  
**Study (numbers master):** [`out/signal_study/m_b1_smoke_20260709T092543Z/`](../../../../out/signal_study/m_b1_smoke_20260709T092543Z/)  
`context.json` · `metrics_auc.json` · `metrics_thr.csv` · `pairs.csv` · study `README.md`  
**Thr / gap 延伸：** `metrics_thr_fine.csv` · `metrics_thr_gt_hurt.json` · `metrics_thr_gap_schedules.csv` · `metrics_thr_gap_nonlinear.json` · `metrics_thr_gap_hybrid.json`（§3b–3c）  
**Substrate MOT:** `results/MOT17_eval_m_b1_substrate_20260709T092543Z/`  
**D1 smoke (tools only):** [m_b1_substrate_smoke_20260709.md](../../../research/eval/m_b1_substrate_smoke_20260709.md)  
**Method hub (s historical):** [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) — style only; **do not** treat its thr/AUC tables as m  
**Contract:** [signal_table_schema.md](../../../research/eval/signal_table_schema.md) §0.1–0.3 · `U_relink_pair`

> **重測：** 新 `study_id` → 改本檔 pointer，不改 s 文內嵌表、不改舊 study_dir。  
> 下文 as-of 數字是 **當次 study 的摘要**；裁決以 study 檔為準。

---

## TL;DR（as-of study）

Same **qualitative** story as offline_relink (s): `bridge_dist` is a **strong full-pool ranker** and a **weaker hard-pool ranker**; precision is still **base-rate limited**.

| Pool | as-of (see `metrics_auc.json`) | How to read |
|:--|:--|:--|
| **full** (`gt_valid`) | n≈22k · base≈**1.6%** · AUC≈**0.87** | Mostly rejects far impossibles (neg median ≫ pos median) |
| **hard** (`bridge_dist≤1`) | n≈1.8k · base≈**12%** · AUC≈**0.76** | Operating region; geometry still helps but is not a free precision win |

**Leverage (not “AUC high → ship bridge”):**

1. Full-pool AUC ≈ 0.87 ⇒ geometry is useful for **negative-pool reduction / coarse reachability**.  
2. Hard-pool AUC ≈ 0.76 ⇒ among spatially plausible pairs, ranking is **better than s-historical ~0.65**, but still far from a closed identity channel — and the pool is a **different death/birth set** (m detector + m tracker noise), so this is **not** a clean A/B of the bridge formula alone.  
3. At thr `bridge_dist≤1` (full), recall of true relinks is only ~**62%** with precision ≈ **base rate of the hard pool (~12%)** — see `metrics_thr.csv`. Useful recall still floods FPs unless the negative pool is cut first.  
4. **B1 ≠ B2:** this says nothing about online reconnect success or e2e IDs with bridge **on**.

**Not claimed:** production preset change; m “beats” s thr tables; AssA ceiling solved.

---

## 1. Substrate (B1 recipe)

Production `mamba_whole_graph_m` defaults **bridge ON + interpolate ON**. B1 requires raw death/birth:

```text
--preset mamba_whole_graph_m --detector SDP
--double-buffer --detect-barrier event
--no-interpolate-tracklets --no-relink-bridge-enabled
```

| Field | This study |
|:--|:--|
| preset / detector | `mamba_whole_graph_m` / SDP |
| relink / interpolate | off / off |
| double-buffer | true |
| e2e substrate (context.e2e) | IDF1≈78.1 · IDs **786** · FP **475** · FN≈21.5k |
| builder | `build_relink_candidates.py` (same rules as offline §2) |
| score | `bridge_dist` (lower-is-better; AUC on `-score`) |
| hard rule | `bridge_dist<=1` (offline-aligned; **not** m production px/h gate) |
| git at summarize | `e560d81b` (see context; main has later merge) |

Full CLI: study `README.md` · schema §0.2.

---

## 2. Reading full vs hard (offline style)

**Full pool.** Pos median `bridge_dist` ≈ 0.60 vs neg ≈ 5.0 (`context.score_dist`) — nearly disjoint tails. That is why full AUC lands ~0.87: easy far negatives. Gap mass is heavy at long gaps (see `context.gap_bins`); velocity extrapolation is expected to degrade with gap (s hub quantified that; m remeasure of per-bin AUC is optional follow-up).

**Hard pool.** Conditioning on `bridge_dist≤1` lifts base rate ~1.6% → ~12% and drops AUC ~0.87 → ~0.76. Geometry still ranks above chance among near candidates, but:

- Precision at the hard-pool edge thr (=1) **is** the hard base rate (~12%) at full recall of hard positives.  
- Tighter thr (0.15 / 0.30) buy precision at large recall cost — same **base-rate wall** structure as offline §3; exact cells live in `metrics_thr.csv` only.

**vs s hub (method only, not scoreboard):**

| | s offline (as-of 2026-06-09) | m this study |
|:--|:--|:--|
| substrate preset | `mamba_whole_graph` | `mamba_whole_graph_m` |
| full AUC (story) | ~0.90 | ~0.87 |
| hard AUC (≤1) | ~0.65–0.68 | ~0.76 |
| full base rate | ~1.3% | ~1.6% |
| n_pos (gt_valid) | 256 | 340 |

Use this table only as **shape comparison**. Different detector capacity, track fragmentation, and IDs count mean **hard AUC is not a pure bridge-ablation**. Do not rewrite s tables as m.

---

## 3. Working-point discipline

Prefer opening `metrics_thr.csv` over copying cells here. As-of orientation (full pool):

| thr | role (narrative) |
|:--|:--|
| 0.15 | high prec / low recall corner |
| 0.30 | mid — offline often quotes |
| 0.50 | more recall, prec falls |
| 1.00 | hard-pool boundary; prec ≈ hard base rate |

Greedy uniqueness tags (`accepted` / `already_linked` on pairs) mirror online propose/commit **semantics** for offline analysis only — not an e2e reconnect metric (B2).

---

## 3b. 門檻誰不傷 GT？（thr 掃 as-of study）

**定義：** 接受規則 `bridge_dist ≤ thr`；**GT_hurt** = 真 relink 對（`gt_match=1`）被 thr 拒掉的比例 = `fn / n_pos`。  
**Master：** `metrics_thr_fine.csv` · `metrics_thr_gt_hurt.json`（同 study_dir）。  
n_pos=340（gt_valid）。

### 真 GT 對的 `bridge_dist` 分位

| 分位 | as-of |
|:--|:--|
| p50 | ~0.60 |
| p75 | ~1.83 |
| p90 | ~4.20 |
| p95 | ~6.29 |
| p99 | ~9.29 |
| max | ~13.1 |

長尾很重：很多真重連在幾何上並不「近」。

### 關鍵 thr → 傷 GT 多少（full pool）

| thr | GT_hurt | rec | prec | 解讀 |
|:--|:--|:--|:--|:--|
| **0.15** | **~83%** | ~17% | ~43% | 幾乎只撿最近真對；**重傷 GT** |
| **0.30** | **~68%** | ~32% | ~33% | offline 常引 thr；仍傷大半真對 |
| **0.40**（近 m `bridge_px` 量級，**≠同一旋鈕**） | **~60%** | ~40% | ~28% | 緊 gate 以丟真對換 prec |
| **0.50** | **~55%** | ~45% | ~23% | 半數以上真對在外 |
| **1.00**（hard 邊界） | **~38%** | ~62% | ~12% | 難池內全收；**池外 38% 真對仍傷** |
| **~2.5**（≈ rec 80%） | ~19% | ~81% | ~5% | 少傷 GT → FP 爆 |
| **~6.3**（≈ pos p95） | ~5% | ~95% | ~2–3% | **幾乎不傷 GT** 但 prec≈base rate 級 |
| **~9.3**（≈ pos p99） | ~1% | ~99% | ~更低 | 實質「全收真對」→ 無排序槓桿 |

### 哪個值「不傷 GT」？

| 目標 | 答案（offline full pool） |
|:--|:--|
| **幾乎不傷**（hurt ≲ 5%） | thr ≳ **真對 p95 ≈ 6.3**（as-of） |
| **幾乎零傷**（hurt ≲ 1%） | thr ≳ **p99 ≈ 9.3** |
| **hard 池內不傷**（已限定 d≤1 的 212 真對） | thr=**1.0** 對 hard-pos hurt=**0%**；thr≥**0.85** 約 hurt≲4% |
| **短 gap（1–10）不傷** | 真對幾乎都 d≤1；thr=**1.0** 對 gap1–10 hurt=**0%** |
| **緊 thr（0.15–0.5）** | **一律重傷 GT**（hurt 55–83%）；只適合作「高 prec 候選過濾」，不能當「保住真重連」門檻 |

**一句：**  
全池上 **沒有**「又緊又幾乎不傷 GT」的 thr——不傷 GT 就要 thr 拉到 **p95+（≳6）**，prec 塌到不可用；  
**不傷 hard 內真對** 的邊界是 **thr=1.0**（定義使然）；短 gap 可放心用 ≤1，長 gap 真對大量在 1 外。

**Caveat：** 這是 **B1 offline pair thr**，不是 e2e IDF1；也 **不是** 直接改 `relink_bridge_px`。不傷 offline 真對 ≠ online 無假 merge（見 B2）。

### 3c. 非線性 `thr(gap)` 試算（as-of same study）

Master：`metrics_thr_gap_schedules.csv` · `metrics_thr_gap_nonlinear.json` · `metrics_thr_gap_hybrid.json`。

在 full pool 上比較 **固定 thr / 分段 p90·p95 / 線性 / √gap / log / 二次 / 飽和指數 / 幂律**，以及 **hybrid floor + sigmoid blend**。

| 方案族 | 形式（精神） | 全池 hurt≈ | 短 gap 1–10 | 長 gap 行為 | 評語 |
|:--|:--|:--|:--|:--|:--|
| `fixed_1` | thr≡1 | ~38% | 不傷 | **重傷**（61+ 半數以上） | 基線 |
| `piecewise_p95` | 每 bin 真對 p95 | ~5–6% | ~6% | 各段均衡 ~5–6% | **最穩、可解釋** |
| `sat_p95` | thr0+(max−thr0)(1−e^{−g/τ}) | ~6% | 略鬆 | 平滑放寬 | 好用連續式 |
| `power_mul_p95` | k(a+b·g^p), p≈0.95 | ~5% | **好** | 放寬 | 近線性、覆蓋均衡 |
| `sqrt/log_mul_p95` | k·(a+b√g) 等 | ~5% 總 | **差（~30% hurt）** | 過鬆 | 截距為負 → 短 gap thr 太緊；**勿單用 mul** |
| `sqrt/log_add_p95` | pred+k | ~5% | 尚可 | 長尾 | 優於 mul |
| `*_floor` hybrid | max(adaptive, 短gap p95) | ~5% | **修好** | 同 adaptive | 推薦修法 |
| `sigmoid_blend_1→pw95` | (1−σ)·1 + σ·pw95 | 低 hurt | 近似 1 | 漸進到 pw95 | 實作友善 |

**非線性有沒有用？**

- **有用（相對 fixed_1）：** 任意合理的 **隨 gap 放寬**（分段 / 飽和 / 幂律 / sigmoid blend）都能把 **長 gap GT_hurt 從 50–80% 拉到 ~5–10%**，同時短 gap 不必開到 thr=6。  
- **沒有免費 prec：** 全池 prec 仍被 base rate 壓在 **~2–4%**（hurt≲10% 時）；非線性改善的是 **hurt 在 gap 上的分配**，不是創造鑑別力。  
- **坑：** 純 **乘性** √/log 擬合（截距常為負）會在 **短 gap 過度收緊**——總 hurt 好看，短遮擋真對反而被砍。應用 **additive、分段、飽和、或 max(·, short_floor)**。

**實作優先序（offline 形狀 → 再 B2 驗）：**

1. `piecewise_p95` 或 `sat` / `sigmoid_blend`（連續、少參數）  
2. `power` / `sqrt_add` + **short_floor**  
3. 不要單獨上 `sqrt_mul` / `log_mul` 而不看分 gap hurt  

---

## 4. Verdicts & non-goals

| Claim | Verdict |
|:--|:--|
| B1 data path on m is valid | **GO (D1)** |
| Full-pool geometry useful for neg reduction | **GO (signal)** |
| Hard-pool geometry alone carries precision / identity | **NO** — base-rate + mid AUC |
| Ship or retune production bridge from this note | **NO** — RESEARCH; needs B2 + e2e |
| Replace s offline tables with these numbers | **NO** — s remains historical hub |

**Optional next (not done here):**

- **Hard def #2:** pool by m production gate envelope (px / h_lo / h_hi) vs `≤1` — new study_id, report both.  
- **D3 B2 (done):** [m_b2_reconnect_bridge_ab_20260709.md](m_b2_reconnect_bridge_ab_20260709.md) — bridge on/off reconnect + e2e.  
- Per-gap-bin AUC on m (s-style curve).  
- Appearance / depth first-gate before geometry (offline lever, not this note).

---

## 5. Reproduction

```bash
# numbers already at study_dir; re-run only if substrate/code changes → new stamp
uv run python scripts/tools/summarize_relink_pairs.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/<new_id> \
  --hard-dist 1.0 \
  --preset mamba_whole_graph_m \
  --mot-dir results/MOT17_eval_m_b1_substrate_20260709T092543Z \
  --relink off --interpolate off --double-buffer true
```

Substrate rebuild recipe: study `README.md` or schema §0.2.
