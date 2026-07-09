# B1 signals — linear vs log（尺度分析）

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Ledger:** `m.scale.linear_vs_log` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Study:** [`out/signal_study/m_b1_scale_compare_20260709T123000Z/scale_compare.json`](../../../../out/signal_study/m_b1_scale_compare_20260709T123000Z/scale_compare.json)  
**Pairs:** 7-seq B1 substrate (same as mine batch)


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

---

## 1. 先前有沒有做？

| 之前 auto-mine | 尺度 |
|:--|:--|
| 幾何距離族 | **linear** only（`bridge_dist`, `score_m_bridge`, …） |
| h-ratio | 閘用 **linear ratio band**；AUC 用 **\|log h\|**（混用、未系統對照） |
| dir_cos | linear \([-1,1]\) |

→ **沒有**正式的「同一物理量 × linear/log/sqrt 尺度表」。本 note 補上。

---

## 2. 原則（避免白做）

**ROC AUC 只看排序。** 對分數做**嚴格單調**變換（`log1p`、`sqrt`、`1/(x+ε)` 若 x>0 且單調）→ **AUC 完全不變**。

本批驗證（`bridge_dist`）：

```text
linear / log1p / sqrt  AUC full = 0.8699950361  (Δ = 0)
ε=0 時 FP_removed 相同 (3001)；只 thr 數字單位不同
  linear thr≈13.12   log1p thr≈2.65   sqrt thr≈3.62
```

**尺度仍重要的場合：**

1. **thr / production knob 語義**（px=0.4 是 linear 空間）  
2. **ε-frontier 的 thr 數值**（不是 FP 數）  
3. **band 幾何**（h 比：linear 區間 vs log-symmetric）  
4. **多 term 加權**（混 linear 距離 + log 尺度會尺度失控）  
5. **分布形狀 / 校準 / 可視化**（skew：linear 重尾 → log1p 較對稱）

---

## 3. 距離族：linear vs log1p vs sqrt

| signal | AUC (任一單調尺度) | pos/neg median **linear** | pos/neg median **log1p** | pos skew lin→log |
|:--|--:|:--|:--|:--|
| score_m_bridge | 0.867 / hard 0.802 | 0.76 / 4.86 | 0.56 / 1.77 | 1.9 → 1.0 |
| bridge_dist | 0.870 / 0.763 | 0.60 / 4.98 | 0.47 / 1.79 | 2.3 → 1.1 |
| resid_mean | 0.863 / 0.792 | 0.88 / 6.03 | 0.63 / 1.95 | 1.9 → 0.9 |
| dist_h | 0.842 / 0.748 | 0.59 / 2.54 | 0.47 / 1.26 | 2.2 → 1.0 |
| speed_mismatch | 0.610 / 0.535 | 0.016 / 0.026 | 同序 | 弱訊號，尺度救不了 |

**結論：** 換 log **不會抬 AUC**；只讓分布好看、thr 單位變。難池弱的 `speed`/`dir` 不是「沒取 log」的問題。

---

## 4. h_ratio：這是唯一「尺度敘事」真有分叉的

### 4.1 當 ranker（離 1 的距離）

| 距離定義 | full AUC | hard AUC |
|:--|--:|--:|
| \|ratio − 1\|（linear 偏離） | **0.865** | **0.784** |
| \|log ratio\| | 0.863 | 0.784 |
| ½(r + 1/r) − 1（log 友好對稱） | 0.863 | 0.784 |
| **raw ratio 當「愈大愈好」** | **0.41** | — | ← **錯 ranker**（真對在 1 附近，不是大 ratio） |

\|ratio−1\| 與 \|log\| 幾乎同分；**不可**用 raw ratio 直接當 AUC 分數。

### 4.2 當 L0 band（這裡 linear vs log 才像「不同閘」）

| band | 類型 | GT_hurt | FP_rm |
|:--|:--|--:|--:|
| **[0.6, 1.7] m prod** | 近 log 對稱（asym≈0.02） | 3.24% | 54.1% |
| [1/1.7, 1.7] 嚴格 log-sym | log | 2.94% | 53.4% |
| [e^{-0.5}, e^{0.5}]≈[0.61,1.65] | log | 3.53% | 55.8% |
| [0.5, 2.0] | log-sym 寬 | 1.18% | 42.4% |
| [0.75, 1.33] s | 近 log-sym | 12.7% | 72.3% |

m 的 `[0.6,1.7]` 本質上已是 **log 對稱高度比閘**（1/0.6≈1.67≈1.7）。改成嚴格 `±log(1.7)` 幾乎無差（hurt 11→10）。

---

## 5. dir_cos

`linear` vs Fisher `z=arctanh`：**AUC 同 0.691 / hard 0.538**。難池近隨機與尺度無關。

---

## 6. 對「深度訊號分析」的操作建議

| 用途 | 建議尺度 |
|:--|:--|
| **報 AUC / 排訊號** | 任意單調；固定一種寫進 meta 即可 |
| **production thr** | 與 code 一致：**linear**（px、raw h ratio band） |
| **可視化 / 合取前標準化** | 距離用 **log1p** 降 skew；h 用 **log ratio** |
| **多 term 加權 (L2)** | 先各 term 校準到同尺度，**禁止** raw linear 距離 + raw ratio 直接加 |
| **想「換尺度救訊號」** | 對單調族 **無效**；應換物理量或 universe |

---

## Reproduce

```bash
# numbers in:
# out/signal_study/m_b1_scale_compare_20260709T123000Z/scale_compare.json
```
