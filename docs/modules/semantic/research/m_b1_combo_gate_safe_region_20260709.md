# Combo gate safe region（2D thr surface · AND 撈門檻）

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/combo_gate_safe_region.py`  
**Study:** [`out/signal_study/m_combo_safe_20260709T124215Z/`](../../../../out/signal_study/m_combo_safe_20260709T124215Z/)  
**Ledger:** `m.combo.safe_region` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Pairs:** 7-seq B1 offline substrate  
**Contract:** GT_hurt hard / FP soft（schema §0.4）


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

---

## 0. 為什麼要 2D surface，不是固定 pair

單訊號 safe thr 很窄（甚至只有尾部 ceiling）。交集：

```text
reject if A > ta AND B > tb
```

可以讓 **ta、tb 都比單用時更鬆**，仍 GT_hurt≤ε，因為：

```text
FP 常多訊號一起壞；GT 多半只在一個維度看起來壞
```

兩種收益（都要報，缺一不可）：

| 收益 | 定義 |
|:--|:--|
| **marginal FP gain** | best_AND_FP > best_single_FP（同 ε） |
| **threshold recoverability** | safe region 格子數 ≫ 單軸 valid thr；**原本單軸不可用的寬 thr 變可用** |

Production 穩不穩看 **safe region 面積 / plateau**，不是孤立 best 點。

```text
safe_region(ε) = {(ta,tb) | GT_hurt(ta,tb) ≤ ε}
報：n_valid · valid_area_ratio · max/mean FP · plateau span · isolated?
```

---

## 1. 語義（px ∩ h-ratio）

```text
px / score_m_bridge   ≠ safe gate
                      = operation-zone / condition energy

abs_log_h             = support evidence（尾部較乾淨）

AND                   = consensus reject candidate
  reject if in hard energy zone AND scale-tail evidence
```

不是「兩個 safe gate 相乘」，是 **條件區 ∩ 支持證據**。

---

## 2. 主結果（ε=0，AND，22×22 分位格）

| pair | 1A best FP | 1B best FP | AND best FP | dFP | nvA | nvB | **nv&** | **&/max(1)** | recov | mFP |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| **score_m_bridge ∩ abs_log_h** | 4218 | **6272** | 3879 | −2393 | 9 | 13 | **436** | **33.5×** | **Y** | n |
| bridge_dist ∩ abs_log_h | 2677 | 6272 | 4195 | −2077 | 6 | 13 | 430 | 33× | Y | n |
| score_m_bridge ∩ abs_ratio_m1 | 4218 | 5245 | 4196 | −1049 | 9 | 11 | 456 | 41× | Y | n |
| resid_mean ∩ abs_log_h | 3191 | 6272 | 4023 | −2249 | 7 | 13 | 442 | 34× | Y | n |
| dist_h ∩ abs_log_h | 4732 | 6272 | 4079 | −2193 | 10 | 13 | 459 | 35× | Y | n |
| score_m_bridge ∩ neg_dir_cos | 4218 | 109 | 4105 | −113 | 9 | 1 | 377 | 42× | Y | n |
| abs_log_h ∩ speed_mismatch | 6272 | 623 | 4749 | −1523 | 13 | 2 | 416 | 32× | Y | n |

**結構（本基板）：**

1. **幾乎沒有 marginal FP gain**（best AND < best single）— 單軸 `abs_log_h` ceiling 已很能砍 FP。  
2. **全面 threshold recoverability**（valid cells **25–40×** 單軸）— 這才是 AND 的主收益。  
3. best plateau **非孤立**（`isolated=false`），但 near-best plateau 仍偏窄 → 可維護的是**整片 safe region**，不是單點調到 best。

---

## 3. 深讀：`score_m_bridge ∩ abs_log_h`（px 條件 × h 證據）

### 3.1 單軸 ε=0 多窄

| 軸 | n_valid thr | 可用 thr 下界（最鬆仍 ε=0） | best FP thr |
|:--|--:|--:|--:|
| A `score_m_bridge` | **9** | **ta ≥ 10.06** | 10.06 → 4218 FP |
| B `abs_log_h` | **13** | **tb ≥ 0.952** | 0.952 → 6272 FP |

單獨 bridge energy：thr 必須 ≥~10 才 ε=0（遠尾 ceiling）。  
`px=0.4` 遠在此下界之外 → **單軸不可能當 safe reject**（與 dist-stability 一致）。

### 3.2 AND 撈回的 thr

| | |
|:--|:--|
| AND ε=0 safe cells | **436**（grid 的 90%） |
| ta 可低至 | **4.79**（vs 單軸 10.06） |
| **ta 低於單軸 min 的 cells** | **238** |
| best AND | ta=4.79, tb=0.79 → **3879 FP**, GT_hurt=0, seq_hurt_std=0 |

```text
threshold recoverability 實例：
  單軸不可用的 ta∈[4.8, 10) 在 AND 下變可用
  （只要同時 abs_log_h 也過對應 tb）
```

這就是「交集把各自 threshold 放寬」的量化版：  
**放寬 = 允許更低的 reject thr（更積極）且仍不傷 GT。**

### 3.3 不要只報 best FP

| 只報 best AND FP=3879 | 完整敘事 |
|:--|:--|
| 看起來輸 single h=6272 | 換來 **33×** 可調 (ta,tb) 面積 |
| 像「AND 更弱」 | 像「bridge 條件 thr 從不可用變可維護」 |
| 易 overfit 一點 | plateau 非孤立；seq_hurt_std=0 @ best |

---

## 4. 保留判準（工具自動旗標）

`worth_keep` 若：

```text
marginal_FP_gain:
  best_AND_FP > best_single_FP 且 plateau 非孤立

OR

threshold_recoverability:
  AND_n_valid ≥ 8
  AND_n_valid / max(single_n) ≥ 3
  best_AND_FP ≥ 0.5 × best_single_FP   # 產能不可崩
```

本批 7 對在 ε=0 **全部 recoverability=Y、marginal=n、keep=Y**。

---

## 5. 需要避免的錯誤

1. 只報 `best_AND_FP` 不報 `n_valid` / `valid_area_ratio`  
2. 把 px 單軸 thr=0.4 當 safe（單軸 ε=0 下界 ~10）  
3. 把 AND best 當唯一上線 thr，不看 region  
4. 不做 per-seq hurt std（本批 best 點為 0，仍需每次查）  
5. 以為 2D overfit 不會發生 — **thin plateau + 高 FP** 才是危險形  

---

## 6. 實作含義（RESEARCH / default-off）

```text
L0a  條件能量（bridge score）— 定義 zone，非 safe 單閘
L0b  support（|log h|）— 可單用 ceiling，或作 AND 證據
L0c  consensus AND — 恢復條件軸的寬 thr 帶；監控 safe_region_area
```

上線候選敘事應是：

```text
reject if score_m_bridge > ta AND abs_log_h > tb
  with (ta,tb) inside measured safe_region(ε=0)
```

而不是：

```text
reject if score_m_bridge > 0.4
```

調 thr 時優化目標建議：

```text
maximize  min_FP_over_neighborhood(ta,tb)     # 區域產能
s.t.      neighborhood ⊂ safe_region(ε)
          seq_hurt_std ≤ τ
```

---

## 7. Reproduce

```bash
uv run python scripts/tools/combo_gate_safe_region.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_combo_safe_<stamp> \
  --all-default-pairs --n-grid 22 --modes AND

# 單對
uv run python scripts/tools/combo_gate_safe_region.py \
  --pairs ... --pair score_m_bridge,abs_log_h
```

產物：`pairs/<A>__<B>/surface.csv` · `single_a.csv` · `single_b.csv` · `summary.json`

---

## Related

- [dist stability](m_b1_signal_distribution_stability_20260709.md) — 為何單軸 px 不穩  
- [energy transform](m_b1_energy_transform_separability_20260709.md) — 加權前 log；AND 是閘不是權重  
- [h_ratio depth](m_gate_h_ratio_signal_7seq_20260709.md) — 單軸 support 稅比  
- schema §0.4 constrained FP pruning  
