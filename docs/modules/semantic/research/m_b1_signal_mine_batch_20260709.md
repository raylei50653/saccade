# m B1 — auto batch signal mine (7-seq)

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**What this is:** 一次把 **offline `U_relink_pair` 上可自動挖的連續訊號** 跑完 §2 checklist（分布 / full+hard AUC / thr / gap·seq / ε-frontier / 可選 prod gate）。  
**What this is not:** production GO；U_cand / live fire / B2；人工升格。

**Study master:** [`out/signal_study/m_b1_signal_mine_20260709T122534Z/`](../../../../out/signal_study/m_b1_signal_mine_20260709T122534Z/)  
**Tool:** `scripts/tools/mine_relink_signals.py`  
**Ledger:** [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Pairs:** existing 7-seq B1 substrate (bridge/interp off)


> **As-of / closed method note.** Numbers live in `out/signal_study/`. Status for the freeze candidate → [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md); phase nav → [hub](m_b1_offline_safe_region_phase_20260709.md). **Do not churn status here.**

---

## Capability boundary（自動能挖完什麼）

| 能自動 | 不能只靠自動 |
|:--|:--|
| B1 pairs 上 catalog 內每個 score 的深度數字包 | 改 preset / 寫 GO |
| hard-pool 排名、ε=0 headroom | `U_cand`（要 cand dump） |
| prod-shaped gate 覆蓋（若有 reject 定義） | live `bridge_attempts` |
| 機器 `auto_verdict` 一句 | 跨訊號因果、policy 權重（Score Audit L2） |
| 重跑：`--all` 一鍵 | B2 reconnect（另 tool） |

**完整「訊號宇宙」** = B1 offline catalog（本批）+ 幀內 + live + B2。本批只閉合 **B1 offline**。

---

## Rank（hard AUC）

| # | signal_id | full | hard | 訊號強度 | L0 備註 |
|--:|:--|--:|--:|:--|:--|
| 1 | `m.score_m_bridge.px` | 0.867 | **0.802** | strong/mid | prod px=0.4：GT_hurt **70%** / FP_rm 99%（操作區，非 safe-reject） |
| 2 | `m.fwd_bwd_resid` | 0.863 | 0.792 | strong/mid | 與 score 同族；無獨立 prod thr 本批 |
| 3 | `m.h_ratio.scale` | 0.863 | 0.784 | strong/mid | prod band：hurt **3.2%** / FP_rm **54%**（最佳 L0 稅比） |
| 4 | `m.bridge_dist.midpoint` | 0.870 | 0.763 | strong/mid | thr=1 邊緣：hurt 38% / FP 93% |
| 5 | `m.dist_h` | 0.842 | 0.748 | mid/mid | 純空間；弱於 residual blend |
| 6 | `m.gap` | 0.729 | 0.651 | mid/weak | **context** 非 identity |
| 7 | `m.dir_cos` | 0.691 | **0.538** | weak / ~random hard | 全池有方向；**難池死** |
| 8 | `m.speed_mismatch` | 0.610 | **0.535** | weak / ~random hard | ε0 只砍 ~3% FP |

**結構結論（自動可下、仍 RESEARCH）：**

1. **幾何家族**（score_m_bridge / resid / mid-bridge / dist_h / h_ratio）主導可分性；hard AUC 全在 0.75–0.80，**無閉合 ID 通道**。  
2. **唯一低稅大砍 FP 的 prod gate 形狀**仍是 **h-ratio band**（見 Gate 1 深度 note）。  
3. **px 閘**是操作區定義（砍掉 bulk 遠負例 + 大量長 gap 真橋），不要當 safe-reject 敘事。  
4. **dir_cos / speed_mismatch** 難池 ≈ 隨機 → 不配做 hard-pool 主 ranker；合取需 ε=0 約束另測。  
5. **gap** 是先驗 context（真對更短），不是外觀/幾何 identity。

細部 by-gap / by-seq / thr / frontier → 各 `signals/*.json`。

---

## Reproduce

```bash
uv run python scripts/tools/mine_relink_signals.py \
  --pairs out/signal_study/m_b1_smoke_20260709T092543Z/pairs.csv \
  --study-dir out/signal_study/m_b1_signal_mine_$(date -u +%Y%m%dT%H%M%SZ) \
  --all

# 單訊號
uv run python scripts/tools/mine_relink_signals.py --pairs ... --signal m.dir_cos
```

---

## Next（自動外）

- Score Audit：term margin / 加權（schema §0.5；工具待建）  
- `U_cand` mine（幀內 IoU/Maha/cost）  
- Live fire coverage  
- ε=0 **合取** rule 搜尋（在 ceiling 之上）  
