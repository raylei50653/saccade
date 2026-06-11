# Relink Normalization & Gate Analysis — 實測數據驗證

> 基於 `mamba_whole_graph` / `no-interp` 基底 (MOT17 SDP train 7 sequences)，對 relink
> pipeline math verification report §6 的公式做實測驗證，並分析多種正規化與門控策略的
> 區分力。最後更新 2026-06-10 (加入新正規化下的 scale gate 分析)。

---

## 1. 數據規模

| 項目 | 數值 |
|------|------|
| 基底 | `results/MOT17_mamba_whole_graph_nointerp` |
| 候選對 | 16,411 (CSV: `scripts/tools/out/relink_candidates_mamba_nointerp.csv`) |
| GT-valid 對 | 15,293 (184 pos / 15,109 neg) |
| Hard pool (bridge_dist ≤ 1) | 2,694 對 (133 pos) |

---

## 2. 線性插值精確度 (within-track)

合成缺口實驗 (2.1M 樣本)：從 confirmed tracks 中移除中間幀，用首尾幀線性插值預測，與實際值比較。

| Gap | n | mean err (h) | p50 | p90 |
|-----|---|-------------|-----|-----|
| 1 | 83,905 | 0.007 | 0.004 | 0.017 |
| 5 | 78,835 | 0.013 | 0.008 | 0.029 |
| 10 | 73,764 | 0.020 | 0.012 | 0.046 |
| 20 | 65,828 | 0.032 | 0.017 | 0.079 |
| 30 | 59,663 | 0.042 | 0.021 | 0.106 |

**結論**：gap ≤ 20 的插值誤差僅 ~3% 身高，極可靠。目前 `interpolate_max_gap=20` 保守但安全，
可推到 30。gap ≤ 20 已被插值解決，relink 不需處理。

---

## 3. 速度估計方法比較

閉式回歸 `v = (3x₃+x₂−x₁−3x₀)/10` vs 平均速度 (mean per-frame velocity)，
各用於橋接距離計算。使用 mean-vel 重算 (配對 CS V)。

| 方法 | Full AUC | Hard AUC |
|------|----------|----------|
| CSV 原始 (mean-vel) | 0.901 | — |
| Recomputed mean-vel | 0.870 | 0.712 |
| Closed-form regression | 0.838 | 0.614 |
| **dist_h (純空間)** | **0.851** | **0.763** |
| Speed-weighted blend | 0.856 | 0.772 |

**結論**：閉式回歸不比 mean-vel 好 (Δ = −0.032)。純空間距離在 hard-pool (0.763)
優於兩種速度外推。速度加權 blend 在 hard-pool 最佳 (0.772)，但增益有限 (+0.009)。

---

## 4. 方向信號 (速度方向 vs 位移方向)

分析 true relink 的出現位置與丟失前速度方向的夾角。

| Speed (h/f) | n_pos | cos_pos | P<30° | Δ cos (true−false) |
|-------------|-------|---------|-------|---------------------|
| <0.01 | 29 (16%) | +0.07 | 24% | +0.15 |
| 0.01–0.02 | 42 (24%) | −0.10 | 17% | 0.00 (死區) |
| 0.02–0.05 | 54 (30%) | +0.07 | 20% | +0.40 |
| 0.05–0.10 | 27 (15%) | +0.12 | 22% | +0.65 |
| ≥0.10 | 27 (15%) | +0.44 | 52% | +0.82 |

**結論**：方向只在 speed ≥ 0.02 h/f 有效。speed ≥ 0.05 時極強 (Δ=+0.7)，
但只覆蓋 30% 的 true relink。低速 (<0.01, 40%) 應關閉方向閘。建議用速度加權軟融合，非硬閘。

---

## 5. 框高正規化策略

分析多種高度參考對 foot-distance 正規化的區分力影響。

### 5.1 策略比較

| Strategy | Full AUC | Hard AUC | Δ hard |
|----------|----------|----------|--------|
| **L_med (單側 lost median)** | **0.8499** | **0.6991** | **+0.019** |
| L_last (單側 lost last) | 0.8329 | 0.6899 | +0.010 |
| D_0,0 (雙側 median avg) | 0.8567 | 0.6871 | +0.007 |
| current (avg lost_last + cand_first) | 0.8508 | 0.6802 | baseline |
| EMA N=30 + cand | 0.8563 | 0.6756 | −0.005 |

### 5.2 關鍵發現

1. **和 candidate 平均永遠是負向操作**：所有雙側策略的 hard AUC 都低於單側。
   候選首幀的尺度變異 (中位 130 vs lost 122) 疊加噪聲，降低區分力。
2. **丟失尾幀不可靠**：naive last-frame (0.690) < median (0.699)。
   尾幀可能被遮擋截斷，跳過 1–3 幀取中位數更穩。
3. **EMA 改善 full-pool 但傷害 hard-pool**：遠處負樣本受益於平滑，
   但近距離操作區需要精確的瞬時尺度。
4. **對稱跳幀無益**：lost 跳 3 + cand 跳 3 的 hard AUC 全低於僅 lost 側處理。

### 5.3 建議

**將 `h_ref` 從 `(h_lost_last + h_cand_first) / 2` 改為 `median(h_lost[:-3])`**

一行改動，零新參數。full AUC 保持 0.850，hard AUC 從 0.680 → 0.699 (+2.8%)。

---

## 6. 框高比率閘 (Scale Gate)

分析 lost 最後框與 cand 首框的高度比率作為獨立門控信號。

### 6.1 舊正規化 (last / first)

| Gate | Recall | False Rejected | Precision |
|------|--------|---------------|-----------|
| [0.5, 2.0] | 97.8% | 42.0% | 1.2% → 2.0% |
| **[0.7, 1.4]** | **85.3%** | **66.1%** | **1.2% → 3.0%** |
| [0.8, 1.25] | 70.1% | 76.9% | 1.2% → 3.6% |

### 6.2 新正規化 (medL_skip3 / medC_skip3)

採用 §5.3 建議的正規化後，比率分布整體右移且變寬：
true median 1.045→1.135, p90 1.33→1.62。舊 gate 閾值不再適用。

| Gate | Recall | False Rejected | Precision |
|------|--------|---------------|-----------|
| [0.5, 2.0] | 91.8% | 39.5% | 1.2% → 1.8% |
| [0.7, 1.4] | 77.2% | 65.4% | 1.2% → 2.6% |
| **[0.8, 1.8]** | **83.7%** | **56.0%** | **1.2% → 2.3%** |
| [0.8, 1.25] | 60.3% | 76.5% | 1.2% → 3.0% |

### 6.3 組合效果：Gate + L_med 距離正規化

先以 scale gate 過濾，再以 L_med 單側距離排名：

| Gate | 剩餘對 | pos | Full AUC | Hard AUC |
|------|--------|-----|----------|----------|
| 無 gate + L_med | 15,293 | 184 | 0.850 | **0.699** |
| [0.7, 1.4] + L_med | 5,371 | 142 | 0.835 | 0.667 |
| [0.8, 1.25] + L_med | 3,658 | 111 | 0.856 | 0.684 |

**結論**：scale gate 是獨立過濾器，和距離正規化正交。gate 後 hard AUC 略降 (0.699→0.684)，
但候選池從 15k 砍到 3.7k (76%)。實際部署建議用 [0.8, 1.8] 配合新正規化 (recall 84%)，
或用 [0.7, 1.4] 配合舊正規化 (recall 85%)。兩者效果相近，選較簡單的舊 gate + L_med。

---

## 7. 綜合建議 (優先級排序)

| 優先級 | 改動 | 預期效果 | 成本 |
|--------|------|----------|------|
| P0 | 框高正規化改為 `median(h_lost[:-3])` | hard AUC +0.019 | 一行 |
| P1 | 加入 scale gate `h_ratio ∈ [0.7, 1.4]` (舊 gate + L_med 組合) | 砍 65% 負樣本，precision 2.2× | 數行 |
| P2 | 若採用新正規化，scale gate 調為 `[0.8, 1.8]` | recall 84% vs 85% | 一行 |
| P3 | 速度加權 blend `w(s)·sym_fb + (1-w)·dist_h` | hard AUC +0.009 | 中等 |
| P4 | 方向軟權重 (speed ≥ 0.02 才啟用) | 高速 case 改善 | 中等 |

---

## 8. Live-Accept 模擬與 P0 復核 (2026-06-11)

腳本：`scratch/pipeline_math_validation/scripts/validate_p0p1_gate.py`
（從 results txt 重建每軌高度史後重算，hard pool 定義 bridge_dist ≤ 1 得 1,032 對 / 102 pos，
與 §1 的 2,694 不同 — 早期分析的 pool 切法不可考，以下數字以本腳本可重現版本為準。）

### 8.1 P0 復核：L_med 增益無法重現

| Strategy | Full AUC | Hard AUC |
|----------|----------|----------|
| avg(last,first)（= CSV dist_h） | 0.8508 | **0.7121** |
| L_med skip3 (P0) | 0.8499 | 0.6995 |
| lost_last only | 0.8329 | 0.6977 |

**L_med 在重現分析中反而低於雙側平均** (−0.013 hard)，與 §5.1 結論相反。
P0 降級為「未驗證」，不建議實作；已 commit 的 lost-only h_ref 改動（b605ac3e）
同樣顯示離線 ranking 略差，需在 MOT17 ablation 中確認是否保留。

### 8.2 P1 模擬：對線上已接受的 relink 套 scale gate

對 live scorer 實際 accept 的 53 TP / 388 FP（pair precision 僅 12%）套用
`h_lost_last / h_cand_first` 比率閘：

| Gate | TP | FP | Precision |
|------|----|----|-----------|
| 無 gate | 53 | 388 | 0.120 |
| [0.7, 1.4] | 47 | 203 | 0.188 |
| **[0.75, 1.33]** | **47** | **182** | **0.205** |
| [0.8, 1.25] | 45 | 155 | 0.225 |

被殺掉的 6 個 TP 全部是 gap ≥ 37 幀的長缺口（37/40/62/90/194/236），其中 4 個
ratio 剛好壓在邊界外 (1.41, 0.67, 0.67, 0.69)。**短缺口 (<37) 的 TP 零損失**。
後續可考慮 gap-adaptive band（長缺口放寬），但先驗證固定 gate。

**建議操作點：[0.75, 1.33] — 殺掉 53% 錯誤 relink，TP 損失與 [0.7,1.4] 相同。**

---

## 9. 產出圖表

所有圖表位於 `scratch/pipeline_math_validation/output/`：

| 檔案 | 內容 |
|------|------|
| `relink_roc_full_vs_hard.png` | ROC: full vs hard pool, 4 methods |
| `relink_pr_full_vs_hard.png` | PR: full vs hard pool, base rate annotation |
| `relink_auc_by_gap.png` | bridge_dist AUC by gap bin |
| `relink_auc_by_speed.png` | AUC by speed: velocity contribution Δ |
| `relink_speed_distributions.png` | Speed histogram: lost/cand/min, gap dist |
| `relink_speed_bar_chart.png` | Pair distribution by speed bin |
| `relink_bridge_vs_disth_scatter.png` | Scatter: bridge vs dist_h by speed |
| `relink_auc_by_speed_with_samples.png` | AUC + sample count overlay |
| `relink_direction_polar.png` | Polar rose: true vs false direction |
| `relink_direction_by_speed.png` | cos(θ) bar + P<30° by speed |
| `relink_angle_vs_speed.png` | Scatter: angle vs speed with zone labels |
| `relink_direction_cumulative.png` | Cumulative: forward fraction vs speed cutoff |
| `relink_height_ratio.png` | Height ratio distribution + by gap |
| `relink_height_gate.png` | Gate recall/precision trade-off |
| `interp_error_analysis.png` | Interpolation error: boxplot, CDF, scatter |
| `interp_error_growth.png` | Error growth: mean, histogram, by speed |

### 分析腳本

| 腳本 | 用途 |
|------|------|
| `scratch/pipeline_math_validation/scripts/validate_relink_math.py` | 公式代數驗證 + 實測 AUC |
| `scratch/pipeline_math_validation/scripts/plot_relink_auc.py` | ROC/PR/AUC 圖表 |
| `scratch/pipeline_math_validation/scripts/plot_relink_dist.py` | 速度分布圖 |
| `scratch/pipeline_math_validation/scripts/plot_relink_direction.py` | 方向分析圖 |
| `scratch/pipeline_math_validation/scripts/plot_relink_height.py` | 框高比率圖 |
| `scratch/pipeline_math_validation/scripts/analyze_interp.py` | 線性插值誤差分析 |
