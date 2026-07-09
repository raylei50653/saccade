# Energy transform separability audit（raw / log / sqrt / rank）

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/energy_transform_separability.py`  
**Study:** [`out/signal_study/m_energy_xform_20260709T123727Z/`](../../../../out/signal_study/m_energy_xform_20260709T123727Z/)  
`energy_transform_separability.csv` · `summary.json` · `rows.json`  
**Ledger:** `m.energy.transform_separability` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Pairs:** 7-seq B1 offline (`m_b1_smoke_*`, bridge/interp off)

---

## 0. 陷阱（寫進契約）

```text
單一 scalar + 只做 threshold / ranking
  ⇒  AUC(energy) == AUC(log(energy))   （嚴格單調）
```

| 問什麼 | 用什麼 |
|:--|:--|
| 有沒有分離度 / 排序訊號？ | **AUC · AP · KS · quantile gap · pruning curve** |
| raw 還是 log 更像**線性 margin**？ | **d′ · Fisher · logistic logloss · Brier · ECE · coef 跨 seq** |
| 能不能當固定 thr 的 safe gate？ | 分布位置 + 跨 seq hurt（另見 [dist stability](m_b1_signal_distribution_stability_20260709.md)） |

**禁止用 AUC 比 raw vs log。**

Pruning `energy < T` ≡ `log(energy) < log(T)` → 曲線本質不變。  
Transform **只在** 進加權 / 線性模型 / 校準時重要：

```text
score = w1·bridge + w2·energy + …   ← raw 尾部會吃掉矩陣
score = w1·log1p(bridge) + …       ← 才可能是穩定 additive evidence
```

---

## 1. 三層 audit（工具已實作）

| Layer | 指標 | 對 log 敏感？ |
|:--|:--|:--|
| **1 排序** | AUC, AP, KS, quantile gap, pruning curve | 否（單調族） |
| **2 線性 margin** | d′, Fisher, Gaussian overlap, logloss, Brier, ECE, coef_std_across_seq | **是** |
| **3 診斷** | auto `diagnosis` + `best_transform` | 綜合 |

Transforms：`raw` · `log1p(e/scale)` · `sqrt` · `rank`  
Slices：`global` · `hard_pool` · gap bins  

`diagnosis`：

```text
rank_signal_only   | raw_linear_good | log_linear_good
piecewise_needed   | slice_only_signal | no_signal
```

---

## 2. 7-seq 主結果（global；as-of study）

驗證：**AUC 跨 raw/log1p/sqrt 位元級相同**（例 score_m_bridge 全為 0.8674）。

| signal | AUC | d′ raw | d′ log1p | Fisher raw | Fisher log | logloss raw→log | diagnosis |
|:--|--:|--:|--:|--:|--:|:--|:--|
| score_m_bridge | 0.867 | 1.04 | **1.42** | 0.54 | **1.00** | 0.489→0.467 | **log_linear_good** (best≈sqrt) |
| bridge_dist | 0.870 | 1.03 | **1.42** | 0.53 | **1.01** | 0.488→0.467 | **log_linear_good** |
| resid_mean | 0.863 | 1.06 | **1.41** | 0.56 | **1.00** | 0.493→0.472 | **log_linear_good** |
| dist_h | 0.842 | 1.09 | **1.34** | 0.60 | **0.90** | 0.521→0.508 | **log_linear_good** |
| abs_ratio_m1 | 0.865 | 0.70 | **1.31** | 0.24 | **0.86** | ~持平 | **log_linear_good**（raw 尾部最兇） |
| abs_log_h | 0.863 | 1.34 | 1.58 | 0.90 | 1.25 | ~持平 | **log_linear_good**（已是 log 族） |
| neg_dir_cos | 0.683 | **0.70** | 0.70 | 0.25 | 0.24 | ~持平 | **raw_linear_good**（有界 [-1,1]） |
| speed_mismatch | 0.610 | 0.41 | 0.43 | 0.08 | 0.09 | ~0.67 | **rank_signal_only**；hard **no_signal** |

### 解讀（對上你的模板）

```text
AUC raw == AUC log，且 log/sqrt 的 d′ / Fisher / logloss 明顯更好
  ⇒  幾何 energy 是「有排序的倍率/重尾訊號」，不是 raw 線性距離 margin。
  ⇒  進綜合 score 必須 log1p 或 sqrt（或 z-score），禁止 raw 直接加權。

dir_cos：AUC 中等偏弱，raw≈log（已有界）
  ⇒  不靠 transform 救命；hard 池近 no_signal → context / 勿當全域主 term。

speed：弱排序 + 難池死
  ⇒  不是 transform 問題；不配全域 score。
```

**hard_pool**（`bridge_dist≤1`）：幾何族 AUC 降到 ~0.78–0.80，但 **log 相對 raw 的 d′ 優勢仍在**（score：d′ 0.72→1.08）。  
操作區內仍是 log-linear 更乾淨，不是「hard 了就可以 raw」。

---

## 3. 與 safe-negative-pruning / 加權

| 用法 | transform 重要？ |
|:--|:--|
| `reject if energy > T` | **否**（raw/log 等價） |
| `score = Σ w_i · energy_i` | **是** — raw 尾部支配 |
| logistic / 校準 / margin | **是** — 用 d′/logloss 選空間 |

與 [dist stability](m_b1_signal_distribution_stability_20260709.md) 合流：

- linear 重尾 kurt~11 → 與此 audit 的 **log_linear_good** 一致  
- 固定 thr 跨 seq 不穩 → 是 **閘位置** 問題，不是「該不該 log thr」  
- **內部**算子：log/sqrt；**對外** px knob 仍可 linear 契約

---

## 4. 實作含義（default-off RESEARCH）

1. **Relink 綜合分 / cost 融合：** 距離族一律 `log1p` 或 `sqrt` 後再加權；h 用 `|log ratio|`（已 log）。  
2. **不要** 用 AUC(raw) vs AUC(log) 做 transform 決策。  
3. **新 energy 上線前** 跑本工具，必過：

```text
global 表: AUC | d'_raw | d'_log | Fisher | logloss | diagnosis
hard_pool 同表
diagnosis ∈ {raw_linear_good, log_linear_good, ...} 寫進 study
```

4. `abs_ratio_m1` raw d′ 差、log 翻倍 → 最典型「linear 偏離尺度不適合 raw 線性規則」。

---

## 5. Reproduce

```bash
uv run python scripts/tools/energy_transform_separability.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_energy_xform_<stamp> \
  --all

# 單訊號
uv run python scripts/tools/energy_transform_separability.py \
  --pairs ... --signal score_m_bridge
```

CSV 欄位對齊提案：`signal_name, transform, slice, n_gt, n_fp, auc, ap, dprime, fisher, …, coef_std_across_seq, best_transform, diagnosis`。

---

## Related

- [scale linear/log](m_b1_signal_scale_linear_log_20260709.md) — AUC 不變性 + band  
- [dist stability](m_b1_signal_distribution_stability_20260709.md) — 跨 seq thr 穩定性  
- [signal mine batch](m_b1_signal_mine_batch_20260709.md) — hard AUC 排名  
- schema §0.5 Gate vs Score  
