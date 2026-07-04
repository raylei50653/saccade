# Media & Streaming Module (媒體接入解碼)

## 📐 模組職責
負責工業級 RTSP 媒體流接入、GStreamer 零拷貝解碼狀態機維護與 DALI GPU 影像預處理。

## 🟢 目前現況
* **GStreamer GPU 零拷貝解碼 (GstZeroCopyDecoder)**：
  * 使用 NVDEC (nvh264dec) 進行 GPU 硬件解碼，直接輸出 `NV12` (YUV420) 格式以節省 PCIe 頻寬。
  * **GPU 顏色轉換 (`_nv12_to_rgb_gpu`)**：直接在 GPU (PyTorch) 內將 NV12 數據轉為 RGB（YUV 通道切片 ➡️ 雙線性插值 upsample UV ➡️ 套用 BT.601 YUV2RGB 公式 ➡️ clamp(0, 255) ➡️ transpose(1,2,0)），實現真正的 100% GPU-side 端到端解碼與色彩處理，完全消除了 CPU 端數據拷貝。
* **C++ GstClient 5-Buffer 狀態機** (race-fixed, ADR-009 已實作):
  * C++ 端 `GstClient` 透過獨立的 `BufferPool` 類 (`include/media/buffer_pool.hpp`) 管理 POOL_SIZE=5 的 GPU 緩衝池,由 `EMPTY→WRITING→READY→PROCESSING→EMPTY` 原子狀態機 (CAS) 控制。當緩衝區排滿時自動丟棄幀(Drop Frame)。
  * 每個 Buffer 分配獨立的 `cudaStream_t`,`acquire_empty_slot` 以 CAS 取得 slot,`ensure_grow` 只成長 EMPTY 槽不釋放 in-use 槽,`~BufferPool` 先 sync 全部 stream 再 free。
  * Python 端 `_on_cpp_frame` 實作完整契約: `with frame_data` (RAII mark_processing/release) + `sync_buffer` (等 H2D) + `torch.cuda.ExternalStream` (stream-ordered 接軌) + `clone()` (脫離 pool 生命週期)。
  * 單元測試: `tests/native/test_gst_buffer_pool.cpp` (8 race 場景, `ctest -R gst_buffer_pool`)。
  * 啟用: `SACCADE_MEDIA_USE_CPP=1` (opt-in)。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `fetch` + `ingest_preprocess`（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | RTSP H.264 流（NVDEC 硬解） |
| **輸出** | GPU RGB tensor（zero-copy）→ `AdaptiveFramePool` → detector |
| **上游 → 下游** | `RTSP → NVDEC(NV12) → _nv12_to_rgb_gpu → GstClient 5-buffer pool → ingest_preprocess → detect` |

## ⚖️ GO / NO-GO 決策

🟢 工業級落地，無 active ablation。RTSP 合約見 [runbooks/rtsp_contract.md](runbooks/rtsp_contract.md)。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
