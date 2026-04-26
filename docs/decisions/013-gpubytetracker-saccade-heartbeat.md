# ADR 013: GPUByteTracker 整合與 Saccade Heartbeat 機制

## 背景 (Background)
在 MOT17 基準測試中，原本的追蹤系統在面對 YOLO26m 中型模型與極端環境（夜間、快速運動、擁擠人群）時，面臨 IDs (ID Switches) 頻繁與召回率 (Recall) 不穩的問題。此外，全量提取 SigLIP 2 特徵雖然精準，但在多路影像併發時會產生嚴重的 GPU 負載壓力。

## 決策 (Decision)
我們決定定稿全新的 C++ 高階追蹤引擎 **`GPUByteTracker`**，並實作 **"Saccade Heartbeat" (智慧眼跳心跳)** 更新機制。

### 1. 核心追蹤演算法：Appearance-First ByteTrack
- **雙階關聯**：保留 ByteTrack 的高/低分偵測框二次匹配邏輯。
- **外觀優先融合**：代價矩陣採用 `(1-w)*IoU + w*ReID`。
- **強特徵跳轉 (Strong ReID Gate)**：當餘弦相似度 > 0.75 時，忽略空間物理限制，強行連結軌跡，對抗相機劇烈晃動。
- **密度感應 (Density-Aware)**：在人群擁擠處自動提升 ReID 權重至 0.8，依賴外觀而非不穩定的空間重疊。

### 2. Saccade Heartbeat (定期更新機制)
- **決策**：放棄每影格全量提取 ReID，改為每 **10 幀** 執行一次原生解析度 (1080p) 提取。
- **技術發現**：實驗證明，稀疏更新能避免 EMA 記憶被影格模糊「污染」，真 ID 錯誤 (IDt) 從 34 大幅削減至 12 (-64%)。
- **負載優化**：ReID 運算開銷降低 **90%**，大幅提升系統併發能力。

### 3. 動態環境補償
- **GMC (全域運動補償)**：實作狀態向量與協方差矩陣的同步仿射變換。
- **光線自適應 (Light Compensation)**：
    - **前處理**：偵測亮度並自動進行對比度拉伸（Contrast Stretching）。
    - **追蹤層**：根據 `light_factor` 動態調整 Kalman Filter 的 $R$ 矩陣（測量雜訊），穩定夜間軌跡。

### 4. 數據管線：原生解析度裁切 (Native Crop)
- **路徑**：YOLO 在 1280 空間偵測，ReID 直接在 1080p 原始顯存空間執行 RoI Align。
- **優勢**：保留微小紋理特徵，IDF1 提升約 2%。

## 後果 (Consequences)
### 正面影響 (Pros)
- **工業級精準度**：Precision 穩定在 **90-95%**，極少產生虛假軌跡。
- **極致穩定性**：ID Switches 在長時序序列中表現優異。
- **效能王炸**：在 RTX 5070 Ti 上維持 6ms 延遲，支援超多路併發。

### 負面影響 (Cons)
- **Recall 受限於模型**：目前的 MOTA 瓶頸完全轉移到了 YOLO 偵測器的召回能力。
- **代碼複雜度**：C++ 核心邏輯變得精密，後續維護需依賴完善的消融實驗腳本。

### 5. Farewell Embedding（告別特徵，2026-04-26）

Heartbeat 間隔 10 幀意味著軌跡消失時，FeatureBank 中存放的最後一筆特徵最多可以是 10 幀前的快照，在遮擋或跨鏡切換場景下恢復準確率受限。

**決策**：在 `SmartTracker` Python 層實作輕量的 Farewell ReID 補救機制：

- **5 幀滑動緩衝** (`_farewell_buffer: deque(maxlen=5)`)：每幀追蹤完成後，將 `(frame_tensor, tracked_ids, tracked_boxes)` 壓入緩衝（僅持有 Python ref，不額外配置 GPU 記憶體）。
- **消失偵測**：`lost_ids = prev_tracked_ids − current_tracked_ids`，只在有軌跡消失時才啟動，不影響正常幀。
- **3 幀補提**：對每個 `lost_id` 倒序掃緩衝，最多取最近 3 幀的 bbox；每幀獨立 crop → SigLIP2 → embedding `[1, 768]`。
- **L2-normalized Average**：`normalize(mean(normalize(embs, dim=1), dim=0))` — 逐幀歸一化消除尺度差異，平均後再次歸一化恢復單位球約束，抵銷特徵震盪。
- **批量寫入 FeatureBank**：所有消失軌跡的告別 embedding 一次 `update_batch`，取代舊的過期特徵。

**效果**：正常幀維持 150+ FPS 純 IoU 追蹤；只有「生離死別」瞬間才以 K≤3 次 SigLIP2 推理換取最新鮮的特徵，直接對抗跨鏡 IDF1 瓶頸。純 Python 層改動，不需重新編譯 C++。

## 狀態 (Status)
**已定稿 (Finalized)** 並通過 MOT17 全測試集驗證。Farewell Embedding 於 2026-04-26 實裝，待下次 MOT17 全集驗證。
