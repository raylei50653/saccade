# YOLO-LLM 開發指南

本專案旨在建構一個高效、低延遲的視覺推理系統，結合即時目標偵測與深度認知推理，並極大化 NVIDIA GPU 的運算效率。

## 1. 系統架構核心 (The 5 Pillars)

本系統採用純視覺與向量檢索管線，以確保在極低 VRAM (1.5GB) 佔用下，達到毫秒級即時回應與精準檢索。

1. **純視覺向量管線 (Vision-Vector Pipeline)**
   - **感知層 (Perception)：** 使用 YOLO11 與 TensorRT 引擎，負責即時物件追蹤與偵測。
   - **特徵提取 (Extraction)：** 運用 Zero-Copy Cropper 與 **Jina-CLIP-v2** TRT 引擎，將物件裁切為 512x512 格式並提取 1024 維高品質特徵向量。

2. **語義漂移去重 (Semantic Drift Handling)**
   - 透過 GPU 內的 `Cosine Similarity`，將新特徵與快取進行比對。避免連續幀重複存儲，僅當物體姿態或特徵產生「漂移 (Drift)」時寫入。

3. **防禦性熱切換 (Systemd + NVML Hot Swapping)**
   - 透過 Systemd `--user` 管理進程，配合 MediaMTX 處理串流緩衝，確保在模組切換或重啟時，視訊串流不中斷。

4. **底層算力優化 (Pure NVIDIA Native Zero-Copy)**
   - 實現 `MediaMTX -> NVDEC -> CUDA Tensor -> TensorRT (YOLO & SigLIP)` 的 100% 零拷貝數據路徑，全程無 CPU 記憶體搬運。

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
  
- **編譯 TensorRT 模型：**
  ```bash
  # 首次啟動前需將 ONNX 轉為 TRT Engine
  uv run python scripts/build_engine.py
  ```

## 3. 媒體與串流管理 (Media Gateway)

使用 **MediaMTX** 作為核心媒介閘道，並以 GStreamer `nvh264dec` 作為解碼前端。

- **啟動所有服務：**
  ```bash
  ./scripts/saccade up
  ```

## 4. 關鍵技術堆疊

| 層級 | 技術 |
| :--- | :--- |
| **算法** | YOLO11 (TRT), Jina-CLIP-v2 (TRT FP16, 512x512), `torchvision.ops.roi_align` |
| **媒體** | MediaMTX, FFmpeg (NVDEC), GStreamer (`appsink`) |
| **計算與資源** | TensorRT, CUDA Streams, Pynvml |
| **環境維運** | Nix Flakes, uv (Rust-based) |

## 5. 開發約定

- **效能優先：** 所有新的 Python 算子需優先考慮是否有對應的 CUDA/TensorRT 加速實現。
- **無狀態推理：** 推理單元應設計為可隨時重啟的服務，狀態應儲存於 ChromaDB 或 Redis 中。
- **類型安全：** 程式碼應嚴格遵守 Type Hinting 約定。
