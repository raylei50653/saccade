# L1: 感知層 (Perception Layer)

## 1. 定義與目標
L1 是 Saccade 的「視網膜與視覺中樞」，負責處理最即時、高頻率的視覺數據。目標是在極低延遲下完成物件偵測與持續追蹤，並過濾出具備「語義價值」的區域供 L2 處理。

## 2. 核心組件
- **解碼器 (GstClient)**: 透過 GStreamer (nvh264dec) 進行硬體加速解碼，輸出 NV12 格式。
- **預處理器 (Preprocessor)**: 在 GPU 內執行 NV12 到 RGB 的轉換、Resize 與 Normalize。
- **偵測器 (Detector)**: 使用 YOLO26 (TensorRT, NMS-Free) 進行物件偵測。
- **追蹤器 (GPUByteTracker + SmartTracker)**:
    - **GPUByteTracker (C++/CUDA)**: 雙階段匹配（high/low score + Sinkhorn）、ReID 融合代價矩陣、Strong ReID Gate、GPU Kalman Filter。
    - **SmartTracker (Python)**: 協調層，負責 GMC（全域運動補償）、光線自適應係數計算、Saccade Heartbeat（每 10 幀更新一次 SigLIP 2 特徵）。
    - **自適應壽命 (Adaptive TTL)**: 在 L6 指令下，可將 `track_buffer` 從 30 幀縮減至 10 幀。
    - **目標清理 (Target Culling)**: 緊急模式下自動銷毀低置信度目標以釋放 VRAM。
- **重排緩衝區 (ReorderingBuffer)**: 解決並行處理導致的時序錯亂，提供 150ms 排序窗口。

## 3. 資料流向
- **Input**: RTSP/WebRTC H.264 原始串流。
- **Output**: 偵測 BBox、追蹤 ID、原始影格 Tensor (GPU)。

## 4. 關鍵優化 (Industrial V2)
- **Zero-Copy**: 資料解碼後直接進入 GPU 5-Buffer Pool，全程不回傳 CPU。
- **Parallel Streams**: 偵測與搬運在獨立 CUDA Streams 執行。
- **In-filling**: 影格跳躍（>40ms）時自動生成虛擬 BBox，確保追蹤連續性。
- **GMC**: 使用 OpenCV optical flow 計算仿射矩陣，補償相機運動導致的 Kalman 狀態偏移。
- **Light Compensation**: 根據幀亮度動態調整 Kalman R 矩陣，穩定夜間軌跡。
- **Saccade Heartbeat**: 每 10 幀觸發一次原生解析度 SigLIP 2 特徵提取，避免 EMA 被模糊幀污染。

## 5. 效能調優指標 (Verified via 5000-frame Benchmark)
- **E2E Latency**: 平均 **6.68 ms** (P99 < 9.3 ms)。
- **Throughput**: 實測可達 **149 FPS** 單路（WSL2 / RTX 5070 Ti），10 路聚合 3000 FPS。
- **Preprocessing (NPP)**: < 0.13 ms。
- **YOLO Inference**: ~3.12 ms。
- **Drop Frame Rate**: 在高負載下自動 Drop，優先保證「最新幀」實時性。
