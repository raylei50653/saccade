# Media & Streaming (Media 模組) 進度核對

## 模組狀態：高效能工業級 (Industrial Grade)

## 1. C++ 零拷貝解碼器 (GstClient)
- [x] **RTSP/WebRTC 對接**: 透過 GStreamer `appsink` 實作高效能接取。
- [x] **[重大更新] 工業級零拷貝 V2 (2026-04-14)**:
    - [x] **5-Buffer 狀態機緩衝池**: 引入 `EMPTY`, `WRITING`, `READY`, `PROCESSING` 狀態，徹底解決多執行緒資料競爭。
    - [x] **Per-buffer CUDA Streams**: 為每個緩衝區分配獨立 Stream，實現並行 H2D 搬運與計算。
    - [x] **PyTorch 深度整合**: 支援 `ExternalStream` 指標傳遞，實現全 GPU 非同步推理。
    - [x] **RAII 自動資源釋放**: 透過 Python GC 與 C++ 綁定，自動回歸緩衝區狀態。
- [x] **自動丟幀機制 (Drop Frame)**: 當分析速度慢於採集速度時自動丟幀，防止累積延遲。

## 2. 硬件加速工具 (ffmpeg_utils.py & DALI)
- [x] **[重大更新] NVIDIA DALI GPU 預處理 (2026-04-15)**:
    - [x] **全 GPU 管線**: 整合 DALI 實現影像解碼、縮放與歸一化全在 GPU 執行。
    - [x] **效能突破**: 預處理延遲從 ~8ms 降至 <1.5ms。
    - [x] **DALI Pipeline**: 實作於 `media/dali_pipeline.py`，支援多串流非同步併發處理。
- [x] **NVENC/NVDEC**: 已封裝 NVIDIA 硬體編解碼工具，支援 L1 感測管線。
- [x] **GStreamer 整合**: 支援 OpenCV 透過 GStreamer 高效能接入。

## 3. 性能監測
- [x] **CUDA Stream 並行監控**: 已確認支援並發搬運。
- [x] **多路串流支援**: 成功達成 10 路串流 3000 FPS 聚合吞吐量。

## 待處理
- [ ] 整合 DALI 至實時 RTSP 串流 (目前優先於文件輸入優化)。
- [ ] 實作針對 L3 緩衝的影格抽樣策略。
- [ ] RTSP 斷線自動恢復監聽 (Watchdog)。

最後更新：2026-04-15
