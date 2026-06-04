# Media & Streaming Module (媒體接入解碼)

## 📐 模組職責
負責工業級 RTSP 媒體流接入、GStreamer 零拷貝解碼狀態機維護與 DALI GPU 影像預處理。

## 🟢 目前現況
* **GStreamer GPU 零拷貝解碼 (GstZeroCopyDecoder)**：
  * 使用 NVDEC (nvh264dec) 進行 GPU 硬件解碼，直接輸出 `NV12` (YUV420) 格式以節省 PCIe 頻寬。
  * **GPU 顏色轉換 (`_nv12_to_rgb_gpu`)**：直接在 GPU (PyTorch) 內將 NV12 數據轉為 RGB（YUV 通道切片 ➡️ 雙線性插值 upsample UV ➡️ 套用 BT.601 YUV2RGB 公式 ➡️ clamp(0, 255) ➡️ transpose(1,2,0)），實現真正的 100% GPU-side 端到端解碼與色彩處理，完全消除了 CPU 端數據拷貝。
* **C++ GstClient 5-Buffer 狀態機**：
  * C++ 端 `GstClient` 管理固定大小 (POOL_SIZE=5) 的 GPU 緩衝池，由 `EMPTY`, `WRITING`, `READY`, `PROCESSING` 原子狀態機控制。當緩衝區排滿時，C++ 自動丟棄幀（Drop Frame）。
  * 每個 Buffer 分配獨立的 `cudaStream_t`，並透過 pybind 暴露指標，供 Python `torch.cuda.ExternalStream` 附掛以進行無阻塞並行計算。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `[1] fetch` + `[2] ingest_preprocess`（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | RTSP H.264 流（NVDEC 硬解） |
| **輸出** | GPU RGB tensor（zero-copy）→ `AdaptiveFramePool` → detector |
| **上游 → 下游** | `RTSP → NVDEC(NV12) → _nv12_to_rgb_gpu → GstClient 5-buffer pool → [2] preprocess → [3] detect` |

## ⚖️ GO / NO-GO 決策

🟢 工業級落地，無 active ablation。RTSP 合約見 [runbooks/rtsp_contract.md](runbooks/rtsp_contract.md)。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
