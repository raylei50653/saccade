# Weight methods × GT-safe productive region（非 best FP）

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/weight_method_safe_region.py`  
**Study:** [`out/signal_study/m_weight_safe_20260709T142000Z/`](../../../../out/signal_study/m_weight_safe_20260709T142000Z/)  

> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

`weight_method_table.csv` · `rank_eps0_by_productive_region.csv` · `summary.json`  
**Ledger:** `m.weight.safe_region`  
**Pairs:** 7-seq B1 offline（`m_b1_smoke_20260709T092543Z`）

---

## 0. 目標改寫

```text
Old: maximize best_FP_removed  s.t. GT_hurt ≤ ε
New: maximize productive plateau thickness
     s.t. GT_hurt ≤ ε, per-seq stable, LOO 不崩
```

**禁止：** unconstrained weights · raw linear sum · per-seq fit · MLP · 只追 best FP 的 black-box search。

**CPU：** pack 級 `ProcessPool`（`--jobs`）+ LOO fold threads（`--loo-workers`）。  
本工具是 pairs.csv offline → **GPU 閒置是預期**。

---

## 1. 五種方法族

| # | method | transform / form |
|--:|:--|:--|
| 1 | `gt_cdf_linear_*` | evidence = \(F_{GT}(x)\)；equal / max-prod80 simplex |
| 2 | `soft_and_{min,geometric,harmonic}` | consensus of \(F_{GT}\) |
| 3 | `clipped_logz_linear_*` | \(\mathrm{clip}(z(\log1p(x)),\pm3)\) |
| 4 | `sparse_monotone_logistic*` | nonneg + L1 logistic on GT-CDF or log-z |
| 5 | `cvar_gt_cdf_linear` | simplex weight max worst-seq productive proxy |
| — | `raw_rank_linear_equal_BASELINE` | pool rank equal（對照，非候選） |

Packs：`score∩abs_log_h`、`bridge∩abs_log_h`、`score∩abs_ratio`、`dist_h∩abs_log_h`、三元 `+neg_dir` / `+resid`。

---

## 2. 1D 指標契約（重要）

對 **fused 1D score**，若 thr 用同一 score 的 GT-quantile 參數化，則

```text
GT_hurt_rate(thr(u)) ≈ u
```

⇒ **1D GT-tail `safe_area_ratio` 近乎恆等式**，不能當方法比較主指標。

本工具改報：

| 指標 | 含義 | 角色 |
|:--|:--|:--|
| **frontier_FP_rate @ ε** | 安全 thr 上最大 FP 率 | 產能 |
| **productive_plateau_w@80** | 安全且 FP≥0.8·best 的 thr 帶寬 / \(\sigma_{GT}\) | **厚度主指標** |
| **robust_FP_rate** | 每 seq 皆 ≤ε 的 thr 上 max FP 率 | 跨 seq |
| **boundary margin** | best thr 到 unsafe 的 \(\sigma\) 距離 | 貼邊風險 |
| **LOO_hurt@train_best** | train 最佳 thr 套 held-out | 過擬合 |

2D AND 的 **GT-tail 面積** 仍用 [`gt_safe_region_area.py`](m_b1_gt_safe_region_area_20260709.md)（多軸 intersection 非恆等）。

---

## 3. 主結果（as-of study）

### 3.1 ε=0：結構性 isolated

所有方法在 ε=0 的 **plateau ≈ 0σ**：

```text
ε=0 thr 必須 ≥ max(score|GT)
⇒ 安全 thr 帶退化成刀刃
⇒ productive_plateau_w@80 = 0（結構，非某方法獨有）
```

與 2D AND 的「ε0 safe% &lt;1% isolated」一致。

**ε=0 若只看 LOO-hurt 再比 frontier FP**（`score∩abs_log_h`）：

| method | FPr | FP | LOOh max | marg σ |
|:--|--:|--:|--:|--:|
| gt_cdf_linear_max_prod80 / cvar | 21.4% | 4592 | **0.006** | 0.003 |
| soft_and geo/harm | 16.0% | 3438 | **0.006** | ~0.02 |
| soft_and min | 15.2% | 3263 | **0.006** | 0.026 |
| clipped_logz（他 pack） | 可到 42% | 高 | 0.01–0.05 | 薄 |

→ **GT-CDF / soft-AND 較「保守可轉」**；clipped log-z 產能高但 LOO 較髒。

### 3.2 ε=0.01：厚度排序（主表）

Top by **plateau_w@80 → robust FP → frontier FP**：

| rank | method | pack | plat σ | FPr | rob FPr | FP | LOOh | class |
|--:|:--|:--|--:|--:|--:|--:|--:|:--|
| 1 | clipped_logz_max_prod80 | bridge∩abs_log_h | **0.66** | 47.9% | 37.1% | 10270 | 0.051 | seq_unstable |
| 2 | clipped_logz_max_prod80 | score∩abs_log_h | **0.64** | 51.4% | 46.2% | 11035 | **0.025** | seq_unstable |
| 3 | clipped_logz_equal | dist_h∩abs_log_h | 0.59 | 57.0% | 40.2% | 12224 | 0.071 | seq_unstable |
| 6 | sparse_logistic_logz | score∩abs_log_h | 0.53 | 54.8% | **52.2%** | 11752 | 0.038 | seq_unstable |
| — | soft_and / gt_cdf | （族均值） | ~0.05–0.08 | ~35% | — | — | ~0.03–0.04 | 更薄 |

**Family mean @ ε=0.01：**

| family | mean plat | mean FPr | mean LOOh |
|:--|--:|--:|--:|
| **clipped_logz** | **0.45** | **50.8%** | 0.045 |
| raw_rank baseline | 0.39 | 48.7% | 0.064 |
| sparse_logistic | 0.29 | 44.9% | 0.041 |
| soft_and | 0.08 | 34.8% | **0.033** |
| gt_cdf linear | 0.07 | 36.6% | 0.038 |
| cvar gt_cdf | 0.05 | 36.1% | 0.037 |

---

## 4. 解讀

```text
1) 目標已從「單點 max FP」換成「厚 + 穩」。
   ε=0 在 1D fused 上幾乎沒有厚平台 → 不要把 ε=0 單 thr 當 production region。

2) clipped log-z 在 ε=0.01 最能把 productive thr 帶「撐厚」、FP 產能也最高；
   但 LOO hurt max 仍 >0，class=seq_unstable → 不是可上線權重。

3) GT-CDF + soft-AND 符合 gate 語義、LOO 常較乾淨，但 plateau 薄、產能中等。
   適合當「保守 evidence 座標」+ 與 2D AND / rule-search 對齊，而不是單靠 1D sum 吃 FP。

4) sparse logistic 多選 log-z 特徵、權重非負；與 clipped_logz 同方向，
   可用來確認「哪些 signal 反覆有獨立貢獻」——尚未當 production。

5) CVaR 在現有 simplex 上與 max-prod80 常選到相近 w（資料裡 worst-seq 與 aggregate
   相關）；要真正拉開需更強 per-seq 目標或 2D region 上的 CVaR。
```

**一句話：**  
加權主線仍是 **GT-CDF evidence + soft-AND consensus**；工程上 **clipped log-z** 是最強的「厚度/產能」對照，但跨 seq / LOO 仍不穩。兩者都還沒通過 production-promising 清單。

---

## 5. 與前序鏈結

| 前序 | 關係 |
|:--|:--|
| [GT safe region area](m_b1_gt_safe_region_area_20260709.md) | 2D AND 面積；本 note = 1D fused 厚度 |
| [energy transform](m_b1_energy_transform_separability_20260709.md) | log 族 d′ 更好 → clipped log-z 產能領先一致 |
| [LOO rule search](m_b1_gate_rule_search_loo_20260709.md) | 仍 loo_partial；權重沒有自動修好 transfer |
| [policy card](m_b1_policy_card_eps0_or5_20260709.md) | in-sample OR-5 仍 candidate only |

---

## 6. Reproduce

```bash
uv run python scripts/tools/weight_method_safe_region.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_weight_safe_<stamp> \
  --n-grid 36 --weight-steps 9 --do-loo \
  --jobs 6 --loo-workers 5
```

約 **~80s** on 32-thread host（6 pack processes）；單核會慢 3–4×。

---

## 7. Next（仍非 preset）

```text
1. 2D soft-AND / GT-CDF axes 上直接 maximize productive_safe_area（非 1D thr）
2. atom repair（禁 gap、zone 緊）後 re-LOO rule-search
3. 僅 LOO clean + e2e 後才碰 B2 / preset
```
