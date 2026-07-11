---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# GT-Support Morphology — Step-0 Identifiability + Placement Audit

> **One-line:** 7-seq relink pair pool 上，Boolean atom lattice 的 **per-cell risk field 不可識別**（任何 k 下僅 1 個 cell 達 ε≤0.05 支撐），但 **GT placement distribution 可識別**。**Procedure verdict: `UNRESOLVED`**（cluster-aware UCB 未建立、core closure 未求解、forensic 未跑、nested chain 未重跑）。**Descriptive morphology hypothesis（無 terminal 效力）**：corner-concentrated placement（k=5 median-split 下 M₀=97.1%）+ 一條 far-Hamming、motion-violation-enriched 的 descriptive tail（4/209 tracks；log_h_ratio 0/4 違反）。Track-level CP UCB x=4→4.33% 僅為 **nominal diagnostic（not cluster-adjusted）**，不得用於跨 ε_morph=5% 界線。

Procedure: [framework §19（v1 — PROPOSED, awaiting owner seal）](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
**Reproduction packet（committed, auditable from PR branch）**: [evidence/gt_support_morphology_step0_20260711/](evidence/gt_support_morphology_step0_20260711/manifest.json) — gt_rows.csv（340 GT rows × atom bits/values/d_H）· cell_occupancy_k{4,5,6,8}.csv · tail_tracks.json（4 tracks 全列 + per-seq 分布）· cp_ucb.json（numerator 出處 + nominal 標記）· audit scripts + recorded outputs · manifest 含 pairs.csv SHA256 seal
Working dir（not tracked）: `out/signal_study/gt_support_morphology_step0_20260711/`

## 0. Scope and claim ceiling

- Read-only counting probe。無 gate、無 rule search、無 threshold 掃描、不碰 Q4.5 evaluator、不碰任何 closed gate（A1 / terminal B / R2–R4 均不變）。
- Pooled、in-sample、選在既有 8 條已挖掘訊號上 → **§8.3 選擇偏誤全額適用；claim ceiling = L1 descriptive**。
- 二值化 = **pool median（探索性、佔用率最佳情形上界）**：真 sealed threshold 只會讓佔用更差。corner 集中的**形狀**對此穩健（各 atom P(z=1|GT) ∈ 0.66–0.98），tail 的**成員名單**不穩健。
- Procedure v1 的 class boundaries 由本 audit 校準（framework §19.6 允許但須明說）：confirmatory unit = 未來 nested per-fold rerun，非本 audit。

## 1. Declared choices

| Item | Value |
|:--|:--|
| Pool | `m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv`，gt_valid only：21,789 rows（GT 340 / FP 21,449） |
| Trial unit（§8.1） | lost track `(seq, lost_id)` → **209 GT tracks**；殘餘 clustering = sequence |
| Atoms（8，mining-AUC 序） | score_m_bridge · bridge_dist · dist_h · log_h_ratio · resid_mean · dir_cos · speed_mismatch · gap（定向 z=1 = 較安全側；衍生式 = `audit_relink_safe_reject.ensure_prod_proxy_scores`） |
| Binarization | pool median per signal（宣告為 audit-only） |
| GT cell 映射 | **descriptive layer**：track 取其 gt_match rows 的最小 Hamming cell（min-d_H representative）。正式 closure 驗證須用 §19.4 set-valued semantics（Z_u 全集 + H_C(u)），本 audit 未做 |

Pooled GT0 的 95% UCB（rule of three, track 單位）= **3/209 ≈ 1.4%**，**nominal（not cluster-adjusted）**：209 tracks 之上仍有宣告的 sequence-level residual clustering，此值只是 pooled 精度的樂觀下限示意，不得直接作 boundary 判定（framework §19.5 UCB validity）。

## 2. Per-cell identifiability（class-6 gate）— FAIL

| k | cells | ≥1 GT track | n≥30（UCB≤10%） | n≥59（UCB≤5%） |
|--:|--:|--:|--:|--:|
| 4 | 16 | 11 | 1 | 1 |
| 5 | 32 | 15 | 1 | 1 |
| 6 | 64 | 23 | 2 | 1 |
| 8 | 256 | 44 | 3 | 1 |

唯一達 ε≤0.05 支撐的 cell 恆為 all-safe corner 本身。Per-fold 更強：**04 / 09 各 12 tracks 全落 corner**，其餘 31 cells GT exposure = 0 → per-fold morphology 不可判定，僅 pooled 可作描述。

**結論：cell-level 風險場（merge tree / barrier / per-cell UCB）在本資料密度下不進主線（framework §19.1）。研究物件 = μ_GT placement distribution。**

## 3. GT placement morphology

Hamming distance to all-safe corner（track = min-d cell）：

| | d=0 | d=1 | d=2 | d≥3 |
|--:|--:|--:|--:|--:|
| k=5 | **203（97.1%）** | 4（1.9%） | 0 | 2（1.0%） |
| k=8 | 117（56.0%） | 77（36.8%） | 11（5.3%） | **4（1.9%）** |

FP 質量分佈相反：k=8 時 **67.7% FP rows 位於 d≥3**（k=5：51.4%）——該區僅 4 個 GT tracks。coverage–risk 交換在遠區結構性有利。

**Far-Hamming descriptive tail 違反側寫（k=8，d≥3 的 4 tracks；min-d_H representative）：**

> ⚠ **Per-sequence 分布：4/4 全在 MOT17-10-SDP**（packet `tail_tracks.json`）。tail 的「4 個 trial」共享同一場景與 pipeline state —— sequence-level clustering 在此不是理論疑慮而是實據，任何把 4 當獨立試驗數的 bound 都不成立；這也是 verdict 停在 UNRESOLVED 的直接證物，並且是 PR-C forensic 的關鍵 context（單場景機制 vs 通用機制待分辨）。

| atom | violated |
|:--|--:|
| speed_mismatch | 4/4 |
| dir_cos | 3/4 |
| resid_mean | 3/4 |
| bridge_dist / dist_h | 2/4 |
| score_m_bridge / gap | 1/4 |
| **log_h_ratio** | **0/4** |

機制解讀（候選）：長遮擋重入的真 relink **保高度、破運動連續性** — 與 R1.1 的 role-reversal 描述症狀及長 gap ReID 族群一致。含義：**motion 類 atoms（speed_mismatch / dir_cos / resid_mean）非全域單調**，不得直接作為全域 closure 維度；height 類（log_h_ratio）是目前唯一 0 違反的全域單調候選。

## 4. Verdict

```text
Procedure verdict: UNRESOLVED（framework §19.5）

觸發的 UNRESOLVED 條件：
  - 無 valid UCB：209 tracks 之上有宣告的 sequence-level residual
    clustering（§8.1），且 tail 4/4 集中於 MOT17-10-SDP 單一序列
    （clustering 為實據非假設）；plain CP 只算 nominal diagnostic，
    不得跨 epsilon_morph 界線；cluster-aware bound 未建立
  - core closure C* 未在宣告偏序下求解（out-of-core mass 無法計算）
  - 4 條 far-Hamming tail 未 forensic
  - nested chain（atom 發現/定向/二值化/verdict）未重跑

Descriptive morphology hypothesis（bounded；無 terminal 效力）:
  corner-concentrated placement with a small far-Hamming,
  motion-violation-enriched tail
  - M0 = 97.1%（k=5 median-split lattice；min-d_H representative，
    descriptive layer only）
  - far-Hamming descriptive tail: 4/209 tracks（k=8, d_H>=3）；
    violation 集中 motion group（speed_mismatch 4/4 · dir_cos 3/4 ·
    resid_mean 3/4），log_h_ratio 0/4 -> mechanism-consistent（候選）
  - nominal track-level CP diagnostic（not cluster-adjusted）:
      x=0 -> 1.42%   x=4 -> 4.33%（n=209）
    距 5% 界線僅 0.67pp，分類完全依賴未滿足的獨立性假設 -> 不採用
  - 此 tail 是 Hamming 描述量，「out-of-core GT mass」一詞保留給
    C* 求解後的 {u : H_C*(u)=1}（framework §19.4）
  - 其餘 cells: UNRESOLVED support（not barriers; framework §19.2）
```

尾巴仍是 **protected GT mass 候選，不是 unsafe hole**：介入方向 = 移除或 regime-條件化被違反的 motion atoms，**不得 veto**（此保護規則不依賴 terminal 判定）。

## 5. Next（evidence order，framework §19.6）

1. Owner seal procedure v1 = **PR 1（Draft）review + merge**（framework §19；ε_morph/UCB 界線已入草案；forensic 結果不得進此 PR）。
2. Escape-tail forensics：4 條 tracks 逐件，僅得落入預宣告類別（true long-occlusion re-entry / annotation issue / signal computation issue / threshold artifact / unresolved）。
3. Nested per-fold rerun 整條鏈（atom 發現、定向、二值化、verdict）→ 才可能升 L2+。

## Must not

- 把本 note 的 verdict 當 sealed procedure 的輸出引用；
- 把零 exposure cells 解讀為高風險 / 障壁；
- 對 escape tail 加 veto；
- 以本 audit 直接授權任何 gate / preset / ledger 變更；
- 重啟 grammar search 或碰 safe-region assetization 的 closed gates。
