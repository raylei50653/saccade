# B1 訊號分布 → 實作穩定性

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Focus:** 分布形態（尾部 / 偏斜 / 跨 seq 漂移）如何影響 **thr、權重、ε-gate 實作穩定性**。  
不是 AUC 排行（單調變換 AUC 不變，見 [scale note](m_b1_signal_scale_linear_log_20260709.md)）。

**Study:** [`out/signal_study/m_b1_dist_stability_20260709T124000Z/dist_stability.json`](../../../../out/signal_study/m_b1_dist_stability_20260709T124000Z/dist_stability.json)  
**Ledger:** `m.dist.stability` → [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)  
**Pairs:** 7-seq B1 offline substrate

---

## 1. 問題定義（實作視角）

| 分布症狀 | 實作後果 |
|:--|:--|
| 重尾 + 高偏斜 | 固定 thr 被少數 outlier 拉；ε=0 ceiling 被單點 max(GT) 綁架 |
| 跨 seq 尺度漂移 | 同一 `px=0.4` 在 04 vs 10 的 GT_hurt 差一個數量級 |
| thr 落在 GT 密度陡坡 | thr 微抖 → 大量真對進出 |
| 異質尺度直接加權 | `w·dist + (1−w)·log_h` 的 w 無物理意義，掃參不穩 |
| 校準 / 合取 | linear 空間「差 1.0」在不同 seq 含義不同 |

---

## 2. 形態：linear 很髒，log1p 接近可用

全池（距離族，as-of study）：

| signal | scale | skew | excess kurt | p95/med | % >5×med |
|:--|:--|--:|--:|--:|--:|
| score_m_bridge | **linear** | **2.77** | **11.3** | 4.2 | 3.2% |
| score_m_bridge | **log1p** | **0.28** | −0.2 | 1.7 | 0% |
| bridge_dist | linear | 2.75 | 10.9 | 4.4 | 3.7% |
| bridge_dist | log1p | 0.20 | −0.3 | 1.8 | 0% |
| resid_mean | linear | 2.62 | 9.7 | 4.0 | 3.0% |
| resid_mean | log1p | 0.13 | −0.2 | 1.7 | 0% |
| abs_ratio−1 | linear | **4.5** | **26** | 9.4 | 11% |
| abs_log_h | （已是 log） | 1.18 | 1.2 | 3.4 | 0.5% |

**含義：** 在 **linear 距離空間**做固定 thr / 線性加權 / 方差估計，預設就不穩。  
log1p 幾乎消掉超峰與 5×median 尖刺 → **內部算子**（融合、校準、統計 thr）應優先 log1p 或 z-score；**對外 production knob** 仍可暴露 linear（與 kernel 一致），但內部不要 raw linear 混料。

`speed_mismatch`：linear 與 log1p 都仍偏（值域本來就小、近 0）→ 尺度救不了弱訊號。

---

## 3. Production thr 坐在哪（密度位置）

| thr | pos CDF（≤thr 的真對%） | 若 reject(score>thr) 的 GT_hurt | 含義 |
|:--|--:|--:|:--|
| **px=0.4** (`score_m_bridge`) | **30%** | **70%** | thr 在真對分布**左側陡坡**；大量真對在 thr 外 |
| bridge hard=1.0 | 62% | 38% | 仍偏緊，但比 0.4 穩一點 |
| ε0 附近 thr≈9.9 | ~100% | ~0% | 尾部；穩砍遠 FP，不碰操作區 |
| \|log h\|≈log(1.7) | **97%** | **~3%** | thr 在真對**右尾**；低稅、位置合理 |

**px=0.4 不穩的根因不是「沒用 log thr」**，而是：  
固定 linear thr 切在 **真對主體內部**（只有 30% 真對落在 accept 側）。任何跨 seq 尺度平移都會劇烈改 hurt。

GT 落在 thr 鄰域（jitter 敏感）：

| | ±10% thr 帶內的 GT | ±50% 帶內 |
|:--|--:|--:|
| score thr=0.4 | 5.3% | **29%** |
| bridge thr=1.0 | 3.8% | **27%** |

→ 約三成真對距離 thr 在半個 thr 以內；**微調 px 會換一批人**，跨資料集必漂。

---

## 4. 跨 seq：固定 thr 的 hurt 方差（核心）

`score_m_bridge` **fixed thr=0.4**（7-seq）：

| 指標 | 值 |
|:--|:--|
| GT_hurt **mean** | 57% |
| GT_hurt **std** | **20 pp** |
| GT_hurt **range** | **17% – 85%** |
| FP_rm | ~99% ±1%（幾乎總是砍光遠 FP） |

**同一 production knob，序列間 hurt 差 5×。**  
`log1p(thr)` 的固定閘與 linear 固定閘 **hurt 曲線相同**（單調等價）→ **只把 thr 改寫成 log 單位，不改善跨 seq 穩定性。**

對照：

| 策略 | hurt std（跨 seq） | 註 |
|:--|--:|:--|
| 固定 thr=0.4 | **20 pp** | 現 production 形狀 |
| 固定 thr=1.0 (bridge) | 15 pp | 仍大 |
| 固定 \|log h\|≈0.53 | **2.9 pp** | 已在尾部，穩得多 |
| per-seq q90（參考） | ~0.5 pp | 穩但 **不是** 現 live 語義；且 FP 只砍 ~10% |

**實作結論：**  
要穩，必須改 **thr 所在分布位置**（尾部 / 分位 / 條件化），不是改 thr 的單位制。

---

## 5. ε=0 ceiling：max(GT) 被 outlier 綁架

Leave-one-seq 上 `thr = max(train GT score)`：

| signal | linear thr **CV** | log1p thr **CV** | holdout FP_rm 仍 |
|:--|--:|--:|:--|
| score_m_bridge | 0.027 | **0.010** | 均值 15%，**std 19 pp** |
| bridge_dist | 0.118 | **0.048** | 11% ±16 pp |
| resid_mean | 0.082 | **0.031** | 13% ±18 pp |
| abs_log_h | 0.045 | 0.034 | 29% ±15 pp |

- log1p 讓 **thr 數值的 CV 變小**（紀錄/監控更穩）。  
- **holdout FP_rm 方差仍大**（同一 ε0 規則，不同 holdout seq 可砍 FP 差很多）→ 尾部質量跨 seq 不均，**safe-reject 產能不穩定**。  
- 實作若用 `max(GT)` 當 ε0：建議 **winsorize / p99.5 of GT** 或 **LOO 分位**，避免單條 outlier track 鎖死 ceiling。

---

## 6. 多 term 加權：尺度決定「w 是否可調」

`score = w·score_m_bridge + (1−w)·|log h|`，掃 w∈{0,0.25,0.5,0.75,1} 的 full AUC span：

| 預處理 | AUC span（w 敏感度） |
|:--|--:|
| **raw linear 混** | 0.050（且 w 含義偏） |
| log1p(dist)+log_h | 0.070 |
| z-score both | 0.066 |
| z(log1p dist)+z(log_h) | 0.069 |

- raw linear：距離 median~5、log_h median~0.1 → **w 幾乎只動距離項**。  
- z-score / log1p 後 w 才是可解釋旋鈕；span 變大代表「真的在權衡兩路」，不是 bug。  
- **實作硬約束：** 進加權 / auction cost 前，term 必須 **各自校準**（z-score、分位 rank、或 learned scale），禁止 raw linear 距離直接進 Π。

---

## 7. 實作方案建議（按穩定性優先）

### 7.1 分層用不同尺度

```text
L0 production knob (對外 / CUDA 現狀)
  → 保持 linear px、linear h-ratio band（bit 契約）

L0 研究 / offline-reject / ε 掃描（內部）
  → log1p(distance)、|log h|；thr 用分位或 winsorized max

L1 term 融合 / 權重掃
  → per-term z-score 或 rank-Gauss；再加權

L2 margin / 校準
  → 同空間；禁止混 c 與 p、混 linear dist 與 raw ratio
```

### 7.2 固定 thr 的穩定性閘門（上線前自測）

對任何候選 thr，7-seq 必報：

```text
per-seq GT_hurt: mean, std, min, max
pos_CDF(thr)          # 建議 >0.9 才當「尾部 safe 閘」
local GT mass ±20% thr
```

經驗門檻（本基板）：

| 角色 | pos_CDF(thr) | 跨 seq hurt std |
|:--|:--|:--|
| 操作區 accept（如 px） | 可低（0.3） | **預期不穩**；勿當 safe-reject |
| 低稅 support gate（如 h-band） | ≳0.95 | std ≲ 5 pp 較可接受 |
| ε=0 FP prune | ~1.0 | 報 holdout FP_rm std；大則 cap 保守 |

### 7.3 不要做的事

1. 以為 `thr_log = log(px)` 會讓跨 seq 更穩 → **不會**（單調等價）。  
2. raw `bridge_dist + h_ratio` 進同一 cost → 尺度失控。  
3. 用全池 max(GT) 當唯一 ε0 thr 且不 winsorize → outlier 綁架。  
4. 只看 overall IDF1 調 px → 掩蓋 04 vs 10 的 hurt 五倍差。

### 7.4 可做的穩向改動（仍 RESEARCH，不改 default）

| 方向 | 動機 |
|:--|:--|
| **條件 thr**：`thr(gap)` / thr(seq 統計) | 固定 0.4 跨 seq 不穩的直接對策 |
| **h-gate 保留 linear band** | 已在尾部、hurt std 小；與分布診斷一致 |
| **內部 score 用 log1p** 再 map 回 linear accept | 融合穩；對外仍 linear |
| **ε0 用 p99.5(GT) 或 LOO** | 降 ceiling CV |
| **權重前 z-score** | 多 term 可調 |

---

## 8. 與前序分析的關係

| 文 | 回答什麼 |
|:--|:--|
| [scale linear/log](m_b1_signal_scale_linear_log_20260709.md) | AUC 不變；band 幾何 |
| **本文** | **分布 → thr/權重穩定性**（實作） |
| [signal mine batch](m_b1_signal_mine_batch_20260709.md) | 誰強誰弱（hard AUC） |
| [h_ratio depth](m_gate_h_ratio_signal_7seq_20260709.md) | 為何 h-gate 稅比好（對上本文：thr 在尾部） |

**一句收束：**  
幾何分數在 linear 空間是**重尾、高偏斜**的；production `px=0.4` 切在真對主體內 → **跨 seq 本質不穩**。log 不能修固定 thr 的跨 seq hurt，但 **必須**用於內部融合與尾部統計；要穩只能把閘挪到尾部、條件化，或分位/校準 thr。
