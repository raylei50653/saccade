# Perception & Extraction (L1-L2) 進度核對

> 架構說明見 `docs/layers/L1_perception.md` 與 `docs/layers/L2_vector_path.md`。

## 模組狀態

- [x] **L1: Perception (YOLO26)** — 已完成
- [x] **L2: Deduplication (SigLIP 2)** — 已完成

## L1: Perception

- [x] YOLO26 TensorRT 推理（NMS-Free）
- [x] Zero-Copy 管線（GStreamer nvh264dec + 5-Buffer Pool）
- [x] 極速裁切（`torchvision.ops.roi_align`）
- [x] GPUByteTracker：雙階段匹配 + Sinkhorn + GPU Kalman
- [x] GMC 全域運動補償（OpenCV optical flow → 仿射矩陣）
- [x] 光線自適應（`light_factor` → 動態 R 矩陣）
- [x] Strong ReID Gate（CosSim > 0.75 強制連結）
- [x] Density-Aware 代價矩陣（ReID 權重 0.5 / crowded 0.8）

## L2: Deduplication

- [x] SigLIP 2 (ViT-B/16) TensorRT 特徵提取
- [x] Saccade Heartbeat（每 10 幀更新，避免 EMA 被模糊幀污染）
- [x] 語義漂移處理（EMA 質心，閾值 0.95）
- [x] AsyncEmbeddingDispatcher（獨立 CUDA Stream 非同步推理）
- [x] FeatureBank（768-dim 向量化矩陣檢索）
- [x] Farewell ReID（軌跡消失瞬間從最近 3 幀補提 embedding，L2-normalized average 寫入 FeatureBank）

## 待處理

- [x] **Tentative/Confirmed 狀態機**（詳見 `/docs/experiments/tracking/tentative_confirmed_state.md`）
  - 三層 detection 分級：High(≥0.5) / Mid(0.25~0.5) / Low(0.1~0.25)
  - Mid-conf detection 可建立 Tentative track，連續 3 幀匹配後升 Confirmed
  - 已完成 MOT17 full SDP A/B 驗證，`mid=0.40 / confirm_score=0.50` 為目前預設
- [ ] 跨鏡頭 Re-ID（多路共享 FeatureBank）
- [ ] 極端光照動態曝光補償優化

## 已完成里程碑

- [x] **極致效能突破**: 10 路串流 3000 FPS 聚合吞吐量，單路均值 300 FPS。
- [x] **雜訊消除機制實裝**: 語義（EMA）與時序（In-filling）雜訊消除，過濾 80% 冗餘數據。
- [x] **純 C++ 感知層**: GPUByteTracker / SmartTracker 全面遷移至 C++/CUDA。
- [x] **硬體加速解碼**: GStreamer nvh264dec + TensorRT 全流程對接。
- [x] **GPUByteTracker 核心強化 (ADR 013)**: ReID 融合、Strong ReID Gate、GMC、光線自適應、Heartbeat 修正為 10 幀。
- [x] **全 GPU 混合關聯 (ADR 015)**: GPU Sinkhorn-Auction Kernel + Top-K 稀疏化已實裝；post-association 採 CPU 二階段 Greedy IoU Matching（ByteTracker 標準架構），MOTA 從 -6.6% 提升至 +30.6%（MOT17-04-SDP, 150 幀）。
- [x] **Farewell ReID（ADR 013 延伸）**: 軌跡消失時從 5 幀緩衝中取最近 3 幀，補提 SigLIP2 embedding，L2-normalized average 後更新 FeatureBank。解決 heartbeat 間隔最長 10 幀的特徵過期問題，在不影響正常幀 FPS 的前提下提升跨鏡頭 IDF1。
- [x] **Lazy ReID Arbiter 入口**: 已暴露 Tentative snapshot API，並完成 candidate / embedding / arbiter dry-run profiling；正式 control path 仍保留為實驗開關。

最後更新：2026-04-26（Tentative/Confirmed 狀態機與 Lazy ReID Arbiter 入口已完成）
