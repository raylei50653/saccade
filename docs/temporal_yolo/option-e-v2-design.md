# Option E-v2：Quality-Gated Temporal Feature Fusion — 實作與結果

> **狀態**：實作完成（2026-05-19 設計，2026-05-22 完成全 Phase 驗證）。
> 基於 Option E（`runs/gated_det_v1`，IDF1 57.2%）延伸，
> 無需重訓，純推論即可獲得增益。

---

## 執行摘要

| Phase | 結果 |
|-------|------|
| **P0 Sanity Check** | ✅ α=0 輸出與 gated_det_v1 完全一致 |
| **P1 Fixed α 掃描** | ✅ MOTA +1.6pp (α=0.15)，Rcll +2.6pp |
| **P2 GMC Warp** | ❌ NO-GO：sparse optical flow GMC 太粗糙，全面倒退 |
| **P3 α_tier 分層** | ✅ **MOTA 54.2% (+1.7pp)，FP 2932 (-21%)** |
| **P4 Lock-in 檢測** | ✅ 無鎖定問題；最長序列 FP -25% |

**最終配置**：Phase 3 α_tier ×1.0，不 warp，從 `gated_det_v1/best.ckpt` 熱啟動。

### 整體對比（MOT17 train，7 SDP sequences，yolo26s）

| Metric | Baseline (gated_det_v1) | **E-v2 α_tier** | Delta |
|--------|------------------------|-----------------|-------|
| MOTA   | 52.5% | **54.2%** | **+1.7pp** |
| IDF1   | 56.9% | 55.6% | -1.3pp |
| Rcll   | 56.2% | 57.3% | +1.1pp |
| FP     | 3712  | **2932** | **-21%** |
| FN     | 49159 | 48003 | -2.4% |
| IDs    | 515   | 545   | +30 |
| Prcn   | 94.4% | **95.6%** | +1.2pp |

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
    detector_score_heatmap_t-1  # 來自上幀 YOLO raw score map（新增，Phase 3 未實作）
)
```

**α_tier**：依軌跡狀態分層的混合強度（已實作）

| 軌跡狀態 | α | 理由 |
|---------|---|------|
| Tentative / 新進區域 | 0.05 | 弱輔助，避免 FP |
| 確認 + age 1–10 | 0.15 | 剛遮擋過，受益最大 |
| 確認 + age >10 | 0.05 | 已穩定，避免 lock-in |
| det_idx=-1（純預測）| 0.20 | 遮擋中最需要記憶 |

**Age decay**：`Q' = Q × max(0, 1 - age / 100)`，防止靜態 FP 長期鎖定。

**warp(·, GMC)**：GMC homography 對齊（Phase 2 實作但結論 NO-GO）。

**.detach()**：t-1 不參與當幀梯度反向傳播。

---

## 程式碼架構

| 檔案 | 改動 |
|------|------|
| `src/saccade/perception/temporal_yolo/temporal_fusion.py`（新）| `TemporalFeatureFusion` 模組（feature cache、α_tier、GMC warp） |
| `src/saccade/perception/temporal_yolo/yolo_gated_detector.py` | 整合 fusion 至 forward hooks，加 `GatedDetConfig.enable_temporal_fusion` |
| `src/saccade/perception/temporal_yolo/yolo_conditioned.py` | `TrackerGateInput` 新增 `confirmed_ages`，`GaussianHeatmapRenderer` 新增 `per_track_weights` |
| `scripts/eval/eval_gated_bytetrack.py` | 加 `--temporal-fusion`、`--fusion-alpha`、`--fusion-warp` flag，tracker-based gate input |

---

## 各 Phase 詳細結果

### Phase 0：Sanity Check ✅

α=0 時輸出與 gated_det_v1 完全一致（max diff = 0.0），確認 feature cache + fusion 無副作用。

```bash
uv run scripts/eval/eval_gated_bytetrack.py --ckpt runs/gated_det_v1/best.ckpt \
  --temporal-fusion --fusion-alpha 0.0
```

### Phase 1：Fixed α 掃描 ✅

| α | IDF1 | MOTA | FP | FN | Rcll |
|---|------|------|----|----|------|
| 0 (baseline) | 56.9% | 52.5% | 3712 | 49159 | 56.2% |
| 0.05 | 56.5% | 53.3% | 3710 | 48211 | 57.1% |
| 0.10 | 56.7% | 53.7% | 4232 | 47184 | 58.0% |
| **0.15** | **57.6%** | **54.1%** | 4655 | **46315** | **58.8%** |

**結論**：α 越大 Rcll 越高，但 FP 也增加。α=0.15 取得最佳 MOTA/IDF1/Rcll。

### Phase 2：GMC Warp ❌ NO-GO

| α | Warp | MOTA | Rcll | vs no-warp |
|---|------|------|------|------------|
| 0.15 | OFF | 54.1% | 58.8% | — |
| 0.15 | ON | 52.0% | 56.0% | -2.1pp |
| 0.05 | OFF | 53.3% | 57.1% | — |
| 0.05 | ON | 52.0% | 55.8% | -1.3pp |

**根因**：SparseOpticalFlowGMC（100 corners, downscale 8×）的 affine estimate 精度不足以做 FPN-level 特徵 warp。grid_sample 在 20×20（P5）解析度引入顯著 artifacts。後續可改用更高品質 GMC（ECC）再試。

### Phase 3：α_tier 分層 ✅

| α scale | IDF1 | MOTA | FP | Rcll | Prcn |
|---------|------|------|----|------|------|
| ×0.8 | 55.9% | 53.4% | 3681 | 57.2% | 94.6% |
| **×1.0** | 55.6% | **54.2%** | **2932** | 57.3% | **95.6%** |
| ×1.2 | 55.2% | 53.2% | 3582 | 56.8% | 94.7% |
| ×1.5 | 55.1% | 52.2% | 4374 | 56.7% | 93.6% |

**結論**：α_tier ×1.0 為最佳配置。FP 達歷史新低（2932），MOTA 達歷史新高（54.2%）。α_tier 成功達成「遮蔽軌跡高 boost、穩定軌跡低 boost」的設計目標。

```bash
# 最佳配置
uv run scripts/eval/eval_gated_bytetrack.py --ckpt runs/gated_det_v1/best.ckpt \
  --temporal-fusion --fusion-alpha 1.0
```

### Phase 4：Lock-in 檢測 ✅

MOT17-04（最長序列，1050 幀）FP 分析：

| Seq | Baseline FP | α_tier FP | Delta |
|-----|-------------|-----------|-------|
| 04 (1050f) | 2879 | **2152** | **-25%** |
| 02 (600f) | 353 | 329 | -7% |
| 05 (837f) | 80 | 40 | -50% |
| 其他 | — | — | ±<10 |

**結論**：無 lock-in 問題。最長序列 FP 反降 25%，證明 age decay（100 幀窗口）有效阻止靜態 FP 長期鎖定。

---

## 與其他方向的關係

- **Option D NO-GO**：證明 decoder 路線不通，本方案跳過 decoder 完全避開該瓶頸
- **ROI ReID NO-GO**：raw FPN 特徵做 cross-frame 比較不穩；本方案用 same-position 融合而非 cross-position 比較，避開穩定性問題
- **Motion relink NO-GO**：相同症狀（時序資訊未進入特徵）；本方案從 feature level 而非 association level 介入

---

## 未完成項目

1. **Detector score heatmap**：Q_spatial = max(tracker_Q, detector_score_t-1)。目前只用 tracker_Q。需要從 YOLO detect head 前一層擷取 raw score map，或將 NMS 後的 detections 反向投影為 heatmap。
2. **Per-scale α_tier tuning**：目前 P3/P4/P5 共用相同 α_tier 值，可進一步區分尺度。
3. **FPS 優化**：tracker state snapshot 查詢（`get_state_snapshots` + `get_tentative_candidates`）每幀成本 ~5-10ms，可透過 caching 減少。
4. **訓練支援**：目前純推論驗證（從 gated_det_v1 熱啟動），`train/temporal_yolo/train_e_v2.py` 尚未實作。

---

## 下一步

建議方向：
1. **推論部署**：將 α_tier 配置設為預設，在生產 pipeline 中驗證
2. **FPS 優化**：cache tracker state 在 Python side，減少 C++ D2H transfer
3. **訓練版**：若推論結果穩定，實作訓練腳本做 learnable α_tier
