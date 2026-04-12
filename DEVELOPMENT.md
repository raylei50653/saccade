# YOLO-LLM 開發指南

本專案旨在建構一個高效、低延遲的視覺推理系統，結合即時目標偵測與深度認知推理，並極大化 NVIDIA GPU 的運算效率。

## 1. 系統架構核心 (The 5 Pillars)

本系統採用雙軌非同步管線，以確保在有限的 VRAM 資源下，達到即時回應與深度分析的平衡。

1. **雙軌非同步管線 (Bifurcated Pipeline)**
   - **快路徑 (感知)：** 使用 YOLO26 與傳統 CV 追蹤技術，負責基礎偵測與資訊熵評估。
   - **慢路徑 (認知)：** LLM / VLM 僅在偵測到高價值事件時，從 MediaMTX 拉取關鍵幀進行深入分析。

2. **變範圍自動分配 (Dynamic Compute Provisioning)**
   - 在 12GB VRAM 限制下，利用 **llama.cpp** 的彈性調度能力，動態調整模型上下文 (`-c`) 與 Offload 層數 (`-ngl`)，必要時將計算負載平滑遷移至 64GB 系統主記憶體。

3. **防禦性熱切換 (Systemd + NVML Hot Swapping)**
   - 透過 Systemd 管理進程，配合 MediaMTX 處理串流緩衝，確保在 AI 模型切換或重啟時，視訊串流不中斷。

4. **底層算力優化 (Pure NVIDIA Native Zero-Copy)**
   - 實現 MediaMTX -> NVDEC -> NVMM -> CUDA Tensor 的零拷貝數據路徑，最小化 CPU 與 GPU 間的數據搬移。

5. **環境與狀態管理 (Unified Environment)**
   - **Nix Flakes：** 鎖定 CUDA、GStreamer 等系統級依賴。
   - **uv：** 進行 Python 環境隔離，實現秒級依賴安裝與重現性。

## 2. 開發環境與建構 (DevOps)

本專案依賴 Nix 與 uv 進行宣告式環境管理。

- **初始化環境：**
  ```bash
  # 進入開發環境
  nix develop
  # 安裝 Python 依賴
  uv sync
  ```

## 3. 媒體與串流管理 (Media Gateway)

使用 **MediaMTX** 作為核心媒介閘道，實現推理單元與影像源的解耦。

- **媒體路徑：** 攝像頭/FFmpeg -> [RTSP/RTMP] -> **MediaMTX** -> [RTSP/WebRTC] -> 推理引擎
- **壓力測試：** 使用 `ffmpeg -re -i source.mp4 -c copy -f rtsp rtsp://localhost:8554/stream` 進行多路串流壓力模擬。

## 4. 關鍵技術堆疊

| 層級 | 技術 |
| :--- | :--- |
| **算法** | YOLO26, Qwen-3.5 (GGUF), CLIP/SigLIP, **llama.cpp (推理後端)** |
| **媒體** | MediaMTX, FFmpeg (NVDEC/NVENC), GStreamer |
| **計算與資源** | TensorRT, PagedAttention, NVML |
| **環境維運** | Nix Flakes, uv (Rust-based) |

## 5. 開發約定

- **效能優先：** 所有新的 Python 算子需優先考慮是否有對應的 CUDA/TensorRT 加速實現。
- **無狀態推理：** 推理單元應設計為可隨時重啟的服務，狀態應儲存於 ChromaDB 或 Redis 中。
- **類型安全：** 程式碼應嚴格遵守 Type Hinting 約定。
