# GT-safe region area（GT-CDF / tail-mass 空間）

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/gt_safe_region_area.py`  
**Study:** [`out/signal_study/m_gt_safe_area_20260709T125933Z/`](../../../../out/signal_study/m_gt_safe_area_20260709T125933Z/)  
**Ledger:** `m.gt.safe_region_area`  
**Pairs:** 7-seq B1 offline

```text
coordinate_space = GT_tail_mass
u = P_GT(score > thr) ∈ (0,1)
thr(u) = quantile_GT(1 − u)
S_ε = { (u_a,u_b) | GT_hurt(AND) ≤ ε }
```

**禁止**在 raw thr 空間量面積（單位不可比）。

---

## 1. 主指標（優先序）

| 指標 | 含義 |
|:--|:--|
| **productive_safe_area@80** | 安全且 FP≥0.8·best_FP 的面積比 | **主** |
| **safe_area_ratio** | 全安全域面積（可能含零產能） |
| **robust_safe_area_ratio** | 每條 seq 都 ≤ε 的交集 |
| **best_point_boundary_distance** | best 點到 unsafe 的 u-空間距離 |
| **plateau_width_min** | 產能平台寬度 |

分級（u-空間）：

| safe_area_ratio | 標籤 |
|:--|:--|
| <1% | isolated / overfit risk |
| 1–5% | thin — 必須 LOO |
| 5–15% | usable |
| >15% | broad（還要看 productive@80） |

---

## 2. 7-seq 結果（AND，25×25 GT-tail grid）

| pair | ε | safe% | **p80%** | robust% | best FP | bdist | class |
|:--|--:|--:|--:|--:|--:|--:|:--|
| score_m_bridge ∩ abs_log_h | **0** | **0.35%** | **0.17%** | 0.35% | 4009 | 0.04 | **isolated_sweet_spot** |
| score_m_bridge ∩ abs_log_h | 0.01 | 3.12% | 0.52% | 1.04% | 8610 | 0.04 | thin_but_promising |
| bridge_dist ∩ abs_log_h | **0** | **0.00%** | 0 | 0 | 0 | — | **isolated**（ε=0 空） |
| bridge_dist ∩ abs_log_h | 0.01 | 3.47% | 1.04% | 1.39% | 10390 | 0.04 | thin_but_promising |
| dist_h ∩ abs_log_h | 0 | 0.87% | 0.69% | 0.87% | 5127 | 0.04 | isolated |
| dist_h ∩ abs_log_h | 0.01 | 2.08% | 0.69% | 0.87% | 7739 | 0.04 | thin |
| score ∩ abs_ratio_m1 | 0 | 0.52% | 0.35% | 0.52% | 4367 | 0.04 | isolated |
| score ∩ neg_dir_cos | 0 | 0.69% | 0.17% | 0.69% | 2631 | 0.04 | isolated |

全表：`safe_area_table.csv`。

---

## 3. 解讀（對齊 LOO）

```text
ε=0 safe_area ≪ 1%  對所有試過的 2D AND
  ⇒  「GT_hurt=0 的點」可以存在，但「可維護安全域」幾乎沒有
  ⇒  與 rule-search in-sample GT0 + LOO partial 一致：
     不是 broad production region，是 outlier-sensitive 薄殼 / 孤立甜點

ε=0 → ε=0.01：safe% 從 ~0.3% 跳到 ~3%
  ⇒  少量 GT outlier 鎖死 ε=0；域在放寬後連續打開（不是隨機噪聲）
  ⇒  仍屬 thin band，要 LOO / robust，不可當 broad_safe_productive
```

**best_FP 高 + safe_area 極小 + bdist 僅 1 grid step (0.04)**  
= 高風險甜點特徵，不是可維護 gate。

---

## 4. Production-promising 清單（本批：全未過）

```text
1. aggregate GT_hurt ≤ ε          — 點上可以
2. per-seq ≤ ε (robust area)      — robust ≈ safe，但面積仍 <1% @ε0
3. productive_safe_area@80 非平凡 — ε0 下 ≤0.7%，失敗
4. best 點有 unsafe margin        — bdist~0.04 僅一格，偏薄
5. LOO 不崩                       — 已見 partial
6. FP 不集中單 seq                — 另表
```

中文收束：

> **不是找到一個 GT_hurt=0 的點就算安全；**  
> 要在 **GT-CDF 空間**有一片安全且有產能的區域，跨 seq 不崩，LOO 不消失，best 點不能貼 unsafe 邊界。

本輪 2D AND：**不滿足 production-promising**。

---

## 5. 與 raw thr 面積的對照

先前 `combo_gate_safe_region` 在 raw 分位格上報 `AND_n_valid` 可達 90%。  
那是 **raw/pool 分位格點數**，不是 GT-CDF 面積：

| 量法 | 風險 |
|:--|:--|
| raw thr / pool quantile cells | 單位不可比；大量格點在「對 GT 極嚴 thr」上虛增 safe |
| **GT_tail_mass u-space** | 直接表示「切到 GT 分布哪裡」 |

兩表都保留：raw cell count 看 recoverability 敘事；**面積裁決用本工具**。

---

## 6. Reproduce

```bash
uv run python scripts/tools/gt_safe_region_area.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --all-default-pairs --n-grid 25
```

---

## Related

- [combo safe region](m_b1_combo_gate_safe_region_20260709.md) — raw-grid recoverability  
- [policy card](m_b1_policy_card_eps0_or5_20260709.md) — in-sample OR-5  
- [LOO](m_b1_gate_rule_search_loo_20260709.md) — transfer partial  
- [dist stability](m_b1_signal_distribution_stability_20260709.md) — thr 位置  
