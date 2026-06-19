# L1: 感知層 (Perception Layer)

> **Legacy architecture snapshot**：本文保留早期 L1 / industrial streaming 架構脈絡。現行 MOT17 eval baseline 與 stage 名稱請以 [DATAFLOW.md](../../DATAFLOW.md)、[PIPELINE.md](../../PIPELINE.md)、[mot17_default_config.md](../../reference/mot17_default_config.md) 為準；其中 headline preset 是 `mamba_whole_graph`（native_640），不是下方 2026-05 tiling comparison。

## 1. 定義與目標
L1 是 Saccade 的「視網膜與視覺中樞」，負責處理最即時、高頻率的視覺數據。目標是在極低延遲下完成物件偵測與持續追蹤，並過濾出具備「語義價值」的區域供 L2 處理。

## 2. 核心組件
- **解碼器 (GstClient)**: 透過 GStreamer (nvh264dec) 進行硬體加速解碼，輸出 NV12 格式。
- **預處理器 (Preprocessor)**: 在 GPU 內執行 NV12 到 RGB 的轉換、Resize 與 Normalize。
- **偵測器 (Detector)**: 使用 YOLO26 (TensorRT, NMS-Free) 進行物件偵測。
- **追蹤器 (GPUByteTracker + SmartTracker)**:
    - **GPUByteTracker (C++/CUDA)**: 雙階段匹配（high/low score + Sinkhorn）、ReID 融合代價矩陣、Strong ReID Gate、GPU Kalman Filter。[詳細技術解析請見 GPUByteTracker 專頁](tracker_deep_dive.md)
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

## 5. 評估套件 (perception/eval/)

MOT17/MOT20 評估邏輯統一放在 `perception/eval/` package，`scripts/eval/mot17.py` 僅作為 CLI entry point。

| 模組 | 職責 |
| :--- | :--- |
| `evaluator.py` | `run_eval()` 主流程（序列迴圈、profiling、結果輸出）；`runner.py` 只是 compatibility shim |
| `detection.py` | `detect_adaptive_960_tiled`、`detect_native_960`、tiled NMS / cross-tile duplicate merge / tile diagnostics 工具函式 |
| `pool.py` | `AdaptiveFramePool` GPU buffer 管理 |
| `preprocess.py` | `parse_preprocess`、`geometry_mid_thresh_scale` |
| `relink.py` | `SemanticRelinker`（embedding-based ID recovery） |
| `streaming.py` | `DALIStreamerStream` JPEG 序列讀取 |
| `tracking.py` | `GlobalTrackIdMapper`（跨序列 ID 統一） |

## 6. 效能調優指標 (Verified via 5000-frame Benchmark)
- **E2E Latency**: 平均 **6.68 ms** (P99 < 9.3 ms)。
- **Throughput**: 實測可達 **149 FPS** 單路（WSL2 / RTX 5070 Ti），10 路聚合 3000 FPS。
- **Preprocessing (NPP)**: < 0.13 ms。
- **YOLO Inference**: ~3.12 ms。
- **Drop Frame Rate**: 在高負載下自動 Drop，優先保證「最新幀」實時性。

## 7. 2026-05 Detection / Tiling Legacy Snapshot

- 2026-05 時 `scripts/eval/mot17.py` 支援多條 detector evaluation 路徑；目前 headline baseline 改為 `mamba_whole_graph` / `native_640`，下列內容只作 tiling NO-GO 追溯：
  - `--tiling 960p_2x2`
  - `--tiling 960p_3x2`
  - `--tiling native_960`
- 另有實驗中的 `--tiling sahi_960p_2x2`：
  - 與 `960p_2x2` 使用相同 `960 -> 4x640` slice geometry
  - merge 改由 SAHI default sliced postprocess 負責，不走 repo 的 cross-tile merge
- `native_960` 是單張 `960x960` letterbox / resize 推論；tiled 路徑則會先做跨 tile duplicate merge，再送進 tracker。
- `960p_2x2` 的 cross-tile merge 已改成 seam-aware：
  - seam-near pair 使用較寬鬆的 duplicate gate
  - cluster representative 不再只保留單一 best box，而是偏向非 seam 候選的加權融合框
- evaluation path 也新增 tile diagnostics，可輸出：
  - `pre_merge_seam`
  - `post_merge_seam`
  - `merged_clusters`
  - `compression`
- 目前實驗結論不是「tiled 已追上 native」。
  - 在 `MOT17-04-SDP / MOT17-10-SDP` 上，`native_960` 仍明顯優於 `960p_2x2 tiled`。
  - seam-aware merge 能降低一部分 tile seam 汙染，但仍未追回 `native_960` 的 `FN / MOTA`。
