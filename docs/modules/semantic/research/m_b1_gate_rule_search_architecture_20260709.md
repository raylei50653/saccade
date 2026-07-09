# Gate rule search architecture（受約束搜尋，非排列組合）

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Tool:** `scripts/tools/gate_rule_search.py`  
**Study:** [`out/signal_study/m_gate_rule_search_20260709T124534Z/`](../../../../out/signal_study/m_gate_rule_search_20260709T124534Z/)  
**Ledger:** `m.gate.rule_search` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Pairs:** 7-seq B1 offline · RESEARCH / default-off

---

## 0. 問題形式（不是 brute force）

```text
maximize   FP_removed(C)
s.t.       GT_hurt(C) ≤ ε          # hard
           per_seq_GT_hurt_std ≤ τ   # stability
           complexity ≤ k            # interpretability
```

工具棧：

| 工具 | 用途 |
|:--|:--|
| **Pareto / dominance** | 丟掉被支配 rule |
| **ε-constrained opt** | 安全契約，不追無界 score |
| **monotone AND prune** | A∩B∩C ⊆ A∩B；FP support 太小剪掉後代 |
| **submodular greedy OR** | 互補 coverage，處理重疊 |
| **sparse atoms** | 連續 thr 平面 → 少量分位 atoms |

**不**掃 `px × h × gap × speed × …` 全排列。

---

## 1. 四層架構

```text
L0  role + transform
    signal 分類：condition | support | fusion | diagnostic
    transform audit 已在 energy_transform_separability

L1  atom generation
    continuous → 可解釋 boolean
    tail_q{85,90,95,99} · hard_zone_q{50,70} · gap_bin

L2  conjunction mining
    找 FP-heavy / GT-rare 的 AND
    單調剪枝 + Pareto

L3  submodular greedy OR
    reject if clause1 OR clause2 OR …
    每步 max ΔFP − λΔhurt − μ complexity，且保持 ε
```

最終 policy 長得像：

```text
reject if
  (condition_zone_A AND condition_zone_B)
  OR (support_tail_h)
  OR (support_tail_resid)
  …
```

---

## 2. Role 分類（降維的第一步）

| signal | role | 含義 |
|:--|:--|:--|
| score_m_bridge / bridge_dist / dist_h / resid | **condition** | operation-zone / 何時懷疑 |
| abs_log_h / abs_ratio_m1 | **support** | safe-reject 證據 |
| neg_dir_cos | support（弱） | 可進 mining，難池弱 |
| speed_mismatch | **diagnostic** | 幾乎不進 policy |
| gap | **condition** | bin，不作單軸 safe |
| log1p(·) | **fusion**（外部） | 加權用，不靠 log thr 穩固定閘 |
| boundary_mass / seq_std | **diagnostic** | 約束與報表 |

避免：

```text
px × h × gap × speed × turn × crowd × …
```

變成：

```text
condition(zone) ∩ support(tail)  →  AND clauses
union of complementary clauses     →  sparse OR
```

---

## 3. 7-seq 結果（as-of study）

### ε = 0

| 階段 | 規模 | 結果 |
|:--|:--|:--|
| atoms | 45 → mine 38 | Pareto **5** |
| clauses | ≤200 | Pareto **6** |
| best atom | `score_m_bridge:tail_q85` | FP **3269**, hurt **0** |
| best AND clause | `dist_h:zone_q70 ∧ score_m_bridge:zone_q70` | FP **4174**, hurt **0** |
| **greedy OR (5)** | 見下 | FP **9130**, hurt **0**, seq_std **0** |

```text
reject if
  (dist_h:zone_q70 AND score_m_bridge:zone_q70)   # condition consensus
  OR (abs_log_h:tail_q85)                         # support h
  OR (resid_mean:tail_q85)
  OR (abs_ratio_m1:tail_q85)
  OR (dist_h:tail_q85)
```

| vs | ΔFP |
|:--|--:|
| best atom | **+5861** |
| best single clause | **+4956** |

Greedy 軌跡（diminishing returns 清晰）：

```text
+ zone AND zone     → 4174 FP
+ h tail            → 7125  (+2951)
+ resid tail        → 8104  (+979)
+ |ratio-1| tail    → 8811  (+707)
+ dist_h tail       → 9130  (+319)
```

### ε = 0.01

| | |
|:--|:--|
| best clause | `dist_h:zone_q70 ∧ score_m_bridge:zone_q50` FP 6235 hurt 3 |
| greedy OR (4) | FP **10184**, hurt **1** (0.29%), seq_std 0.015 |
| 停在 4 條 | 下一步 score≤0 被拒（正 Δ 約束） |

---

## 4. 和前序工具的關係

| 工具 | 層 | 回答 |
|:--|:--|:--|
| energy_transform_separability | L0 | raw vs log **線性 margin**（非 AUC） |
| dist_stability | diagnostic | thr 坐在分布哪、跨 seq 漂 |
| combo_gate_safe_region | L2 特化 2D | **safe region area** / 撈門檻 |
| **gate_rule_search** | L1–L3 總控 | atoms → AND → OR policy |

combo surface 回答「兩個連續 thr 的 region」；  
rule search 回答「多 atoms 的可解釋 policy 在 ε 下能蓋多少 FP」。

---

## 5. 為何這不是 overfit 排列組合

1. **Atoms 少**：每 signal 固定幾個 quantile，不是連續網格笛卡爾。  
2. **Role 過濾**：condition / support 分開，gap 不單飛。  
3. **AND 單調剪枝**：FP support 不足不長子節點。  
4. **Pareto**：同 ε 下被支配的 clause 丟掉。  
5. **Greedy OR**：只加有 **正** marginal score 的 clause；重疊自動貶值。  
6. **硬約束**：每步 union 必須 GT_hurt≤ε。

仍需人工：LOO seq、production 語義審查、B2/e2e。數字 master 在 study_dir。

---

## 6. Reproduce

```bash
uv run python scripts/tools/gate_rule_search.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_gate_rule_search_<stamp> \
  --eps-grid 0.0,0.01 \
  --max-and-size 3 --max-or-rules 5 --min-fp-support 100
```

```text
eps_0p0/atoms.csv
eps_0p0/clauses.csv
eps_0p0/summary.json   # policy + Pareto ids
summary.json
```

---

## 7. 一句

> **連續訊號 → 少量有角色的 atoms → Pareto + itemset AND → submodular greedy OR。**  
> 多維系統變成可審計的 ε-constrained rule search，而不是排列組合調參。
