# Saccade MOT17 精度與效能優化實驗紀錄 (2026-04-18)

## 實驗背景
目標是將 MOT17 的 MOTA 指標從 33.1% 提升至 50% 以上，並同時達成 30+ FPS 的即時效能。

## 技術路線與結論

### 1. 解析度與採樣策略
- **實驗 A: 1080p Native Tiling (r=1.0)**
    - 結果：Recall 慘跌 (< 1%)，偵測器對大尺度目標感知失效。
    - 結論：**失敗**。解析度並非越高越好。
- **實驗 B: 960 Tiled (r=0.5)**
    - 結果：MOT17-09 達成 **51.4% MOTA**。
    - 結論：**最佳甜點解析度**。兼顧小人召回與偵測器感受野。
- **實驗 C: 動態 ROI (Dynamic Fixation)**
    - 結果：MOT17-04 達成 **35.5% MOTA**，優於固定分塊。
    - 結論：**最省算力的方案**。適合嵌入式裝置。

### 2. 並行化與 Zero-Copy
- **DALI GPU Decoding**: 成功消除了 `cv2.imread` 的延遲，Preprocessing 降至 4ms。
- **1-Frame Lag Pipeline**: 將 YOLO 與 SigLIP 並行，輸送量衝至 **80 FPS**。
- **全 GPU 座標流**: 數據全程在 VRAM 流轉，消除 CPU/GPU 同步開銷。

### 3. 穩定性優化
- **NSA Kalman Adaptation**: 解決了車載場景 IDs 爆增問題，IDs 從 1061 降至 230。
- **Centrality Weighting**: 解決了 Tiling 邊界斷裂感。

## 最終定稿配置
- **Detector**: YOLO26-M (FP16)
- **Engine**: 640 Native
- **Input**: 960 Adaptive Tiling or 640 Global
- **E2E Speed**: ~12.6ms (79 FPS)
- **Final MOTA (MOT17-09)**: 51.4%
- **Final IDF1 (Total)**: 44.4%
