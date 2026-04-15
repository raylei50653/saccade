# ADR 010: NVIDIA DALI GPU 預處理 (NVIDIA DALI GPU Preprocessing)

## 狀態
**已通過 (Accepted)** - 2026-04-15

## 背景 (Background)
在 Saccade 達到 10 路串流 3000 FPS 的過程中，傳統的 CPU 預處理（使用 OpenCV 或 torchvision）成為了主要瓶頸：
1. **CPU 負載過重**：多路串流的解碼、縮放與歸一化佔用了大量 CPU 週期，導致 Orchestrator 調度延遲。
2. **PCIe 傳輸瓶頸**：將影像從 CPU 搬運至 GPU 增加了額外的延遲。
3. **處理能力限制**：OpenCV 難以在維持低延遲的同時支撐每秒數千影格的預處理。

## 決策 (Decision)
我們決定引入 **NVIDIA DALI (Data Loading and Augmentation Library)** 作為核心預處理管線：
1. **全 GPU 管線**：利用 `nvdec` 在 GPU 上直接解碼 H.264/H.265，隨後在 VRAM 內完成 `Resize` 與 `Normalize`。
2. **非同步執行**：DALI 管線與 TensorRT 推理管線並行運作。
3. **零拷貝對接**：DALI 輸出的 Tensor 直接轉換為 `torch.Tensor`，無需經過 CPU 中轉。

## 實作細節
- **Pipeline 定義**：實作於 `media/dali_pipeline.py`。
- **輸入類型**：初期針對文件輸入 (`ExternalSource`) 進行優化，後續將擴展至 RTSP 串流。
- **配置參數**：
    - `batch_size`: 根據串流數量動態調整。
    - `device`: `gpu`。
    - `output_dtype`: `DALIDataType.FLOAT`。

## 後果與折衷 (Consequences & Trade-offs)
- **優點**：
    - 預處理延遲從 ~8ms 降低至 <1.5ms。
    - 釋放大量 CPU 資源給系統監控與 L5 認知層。
    - 支援極高吞吐量的批次處理。
- **缺點**：
    - 增加約 200-300MB 的 VRAM 佔用（取決於 batch size）。
    - 引入了對 `nvidia-dali-cuda120` 的額外依賴。

## 驗證指標
- 通過 `scripts/latency_breakdown.py` 驗證預處理階段的耗時。
- 32 路串流穩定運行且 CPU 使用率維持在 20% 以下。
