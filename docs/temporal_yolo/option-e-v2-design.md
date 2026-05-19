# Option E-v2：Quality-Gated Temporal Feature Fusion — 設計

> **狀態**：設計階段（2026-05-19）。基於 Option E（`runs/gated_det_v1`，IDF1 57.2%）延伸。
> 目標：直接利用 t-1 的 FPN 特徵，補強遮擋恢復與時序一致性。

---

## 動機

Option E 的 gate 機制只透過 tracker state 間接利用時序資訊（boxes + Kalman velocity），
但**沒有使用 Frame_{t-1} 本身的 FPN 特徵**。理論上：

- 遮擋中目標：t-1 的特徵保留了「人應該長這樣」的訊號，當前幀 FPN 衰減時可以補強
- 背景一致性：靜態背景的特徵在 t-1 / t 高度一致，可協助 FP 抑制
- 新出現目標：tracker 還沒抓到，但 t-1 的偵測 heatmap 可能已有微弱訊號

---

## 核心公式

```
P_t_fused = P_t + α_tier × Q_spatial × warp(P_{t-1}.detach(), GMC)
```

### 各項定義

**Q_spatial**：每像素質量圖，融合多個來源
```python
Q = max(
    tracker_quality_heatmap,    # 來自 TrackSpatialGate（已有）
    detector_score_heatmap_t-1  # 來自上幀 YOLO raw score map（新增）
)
```

**α_tier**：依軌跡狀態分層的混合強度（不是單一純量）

| 軌跡狀態 | α | 理由 |
|---------|---|------|
| Tentative / 新進區域 | 0.05 | 弱輔助，避免 FP |
| 確認 + age 1–10 | 0.15 | 剛遮擋過，受益最大 |
| 確認 + age >10 | 0.05 | 已穩定，避免 lock-in |
| det_idx=-1（純預測）| 0.20 | 遮擋中最需要記憶 |

**warp(·, GMC)**：用 GMC homography 對齊 t-1 特徵到 t 的座標系

**.detach()**：t-1 不參與當幀梯度反向傳播

---

## 必須處理的五個失敗模式

### 1. FP 強化迴路（最大風險）

靜態 FP（招牌、影子）被 confirm → Q 高 → t-1 特徵反覆注入 → FP 越來越穩固。

**對策**：
- α_tier 對 age >10 的軌跡降權至 0.05
- 軌跡 age 規範化後乘入 Q：`Q' = Q × (1 - age/100)`
- 每 N 幀對該軌跡的特徵記憶做衰減

### 2. 空間錯位（FPN 解析度問題）

P3 = 80×80（每格 8 像素），P5 = 20×20（每格 32 像素）。Kalman 預測偏 1 格 → 實際偏 8–32 像素 → 特徵混到錯誤位置。

**對策**：
- 用 GMC homography warp t-1 特徵
- P5 解析度低，warp 誤差自然被吸收
- P3 在訓練時加入 jitter augmentation 增強對位移的容忍

### 3. 新出現目標 Q=0 的盲區

目標已在 t-1 出現但 tracker 未抓到（首次偵測延遲）→ 我們完全丟掉了 t-1 的有用特徵。

**對策**：
- Q_spatial 加入 detector score heatmap：`Q = max(tracker_Q, detector_score_t-1)`
- 讓未 track 但有偵測訊號的區域也保留 t-1 貢獻

### 4. 梯度流問題

若 P_{t-1} 不 detach，BPTT 經過多幀會爆炸或消失，且訓練成本暴增。

**對策**：
- `P_{t-1}.detach()`，t-1 只作為「凍結記憶」
- 失去跨幀表徵學習，但訓練穩定性大幅提升
- 這是 SOTA video detection（如 SELSA / MEGA）的標準做法

### 5. 鏡頭運動

Camera pan → t-1 像素位置 ≠ t 像素位置 → 背景特徵完全錯位。

**對策**：
- 直接套用現有 GMC 的 homography
- Warp 在 FPN 出口（每尺度獨立 warp）執行
- 已有 `GMC` 模組（C++ + Python），無新基礎建設成本

---

## 與 Option D 的差異

| 面向 | Option D | Option E-v2 |
|------|----------|-------------|
| Frame_{t-1} 接入方式 | Tracker 狀態（boxes + vel）| **FPN 特徵直接快取** |
| Decoder | Cross-Attention（100 queries）| **無（標準 YOLO head）** |
| Q_spatial 用途 | 空間 gate（乘 FPN）| **雙用：gate + 時序融合權重** |
| GMC 角色 | 不使用 | **特徵 warp 對齊** |
| 訓練穩定性 | 中（gt_ratio curriculum）| **高（detach + α=0 熱啟動）** |
| 從 gated_det_v1 熱啟動 | 否（架構不同）| **是（α=0 等效）** |

---

## 實作計畫

### Phase 0：Sanity Check（無風險驗證）

**目標**：確認 feature cache + α=0 推論結果與 `gated_det_v1` 完全相同。

**改動**：
- `GatedYOLODetector` 加 `prev_feats` 屬性，每幀推論後快取 P3/P4/P5
- 加 `α_p3/p4/p5` 參數，預設 0
- 推論流程加 fusion：`P_fused = P + α × Q × warp(P_prev)`

**驗收**：α=0 時 IDF1 = 57.2%（誤差 <0.05pp）

### Phase 1：固定 α，不 warp，不 detector heatmap

**目標**：純粹看 baseline gain（無 warp = 接受空間錯位）。

**參數**：
- α 統一 = 0.1
- Q_spatial 只用 tracker heatmap
- 無 GMC warp

**驗收**：FP、IDs、Rcll 三個方向至少一個改善（或全部不退步）。

### Phase 2：加 GMC warp

**目標**：看背景 FP 是否進一步下降。

**參數**：
- 同 Phase 1，但加 `warp(P_prev, GMC)`
- 用既有 `GMC.estimate()` 取得 homography

**驗收**：相對 Phase 1，FP 進一步 -3% 或 MOT17-13（鏡頭運動序列）IDF1 +1pp。

### Phase 3：α_tier 分層 + detector heatmap

**目標**：精細化各軌跡狀態的最佳 α。

**參數**：
- α_tier 表（見上方）
- Q_spatial 加入 detector score heatmap

**驗收**：相對 Phase 2，IDs -10% 或 IDF1 +0.5pp。

### Phase 4：Lock-in 失敗模式檢測

**目標**：確認長序列無 FP 累積問題。

**檢查**：
- MOT17-04（1050 幀，最長）逐幀 FP 累積曲線
- 比較 baseline 與 Option E-v2 的 FP 增長率
- 若 FP 增長率 > baseline × 1.2 → 啟用 age 衰減

---

## 預期收益與風險

| 指標 | 樂觀 | 中性 | 悲觀 |
|------|------|------|------|
| FP | -10% | -3% | +5%（lock-in 主導）|
| IDs | -15% | -5% | +0% |
| Rcll | +1pp | +0.3pp | -0.5pp |
| FPS | -5% | -8% | -12% |

**最大不確定性**：FP 方向。Phase 4 是 GO/NO-GO 的決定點。

---

## 訓練策略

### 從 gated_det_v1 熱啟動（推薦）

1. Phase 1 直接推論測試（無需重訓）：α=0.1 hardcoded
   - 如果 Phase 1 已有正向訊號 → 進 Phase 2
   - 如果 Phase 1 退步 → 訓練 learnable α

2. 若需訓練：
   - 凍結 backbone，只訓 `α_tier` 參數（Phase 1，5 epochs）
   - 解凍 backbone fine-tune（Phase 2，10 epochs）
   - lr_α = 1e-3，lr_backbone = 1e-5

### 訓練資料

- 從 `MOT17TemporalClip` 取 T=2 幀片段（t-1, t）
- t-1 跑 forward + cache feats
- t 跑 forward + fusion + loss
- 標準 YOLO detection loss（無 Auction matcher）

---

## 程式碼預估改動

| 檔案 | 改動 | 行數估計 |
|------|------|---------|
| `src/saccade/perception/temporal_yolo/yolo_gated_detector.py` | 加 feature cache + fusion 邏輯 | +80 |
| `src/saccade/perception/temporal_yolo/temporal_fusion.py`（新）| `TemporalFeatureFusion` 模組 | +120 |
| `scripts/eval/eval_gated_bytetrack.py` | 加 `--temporal-fusion` flag | +30 |
| `train/temporal_yolo/train_e_v2.py`（新）| Phase 1+2 訓練腳本 | +200 |

**總計**：~430 行新增；不改動 C++ 程式碼。

---

## 開放問題

1. **Detector heatmap 從哪取？**
   YOLO 原始 output 是 (B, 300, 6)（已 NMS），需從 raw `model.0~22` 中途的 detect head 前一層取 score map。
   或者：用 t-1 final detections 反向投影成 heatmap（簡單但解析度低）。

2. **Q_spatial 是否需要 per-scale 不同？**
   P3（小目標）vs P5（大目標）可能需要不同的 sigma scaling。先用統一，消融確認。

3. **α_tier 該硬編碼還是 learnable？**
   建議：v1 硬編碼快速驗證；若有正向訊號，v2 改 learnable per-scale per-tier。

4. **FPS 損失能不能在 C++ side 攔截？**
   Feature cache + warp 都在 Python；若進 prod 需移到 C++ SequenceRunner。
   暫不考慮，等 Python 路徑驗證有效再說。

---

## 與其他方向的關係

- **Option D NO-GO**：證明 decoder 路線不通，本方案跳過 decoder 完全避開該瓶頸
- **ROI ReID NO-GO**：raw FPN 特徵做 cross-frame 比較不穩；本方案用 same-position 融合而非 cross-position 比較，避開穩定性問題
- **Motion relink NO-GO**：相同症狀（時序資訊未進入特徵）；本方案從 feature level 而非 association level 介入

---

## 下一步

如果 Phase 0 sanity check 通過，建議**先做 Phase 1 純推論測試**（無訓練成本），
快速判斷是否值得進入訓練階段。
