# Cognition 模組進度核對 (2026-04-12)

## 模組狀態：已棄用 / 架構轉型 (Deprecated)

> 💡 **重要通知：** 
> 為了在邊緣設備 (12GB VRAM) 上榨出極限吞吐量並確保毫秒級延遲，Saccade 系統已將架構從「雙軌 VLM 認知」轉移至「純視覺向量 (Pure Vision-Vector) 管線」。
> 原本由 `llama-server` (VLM) 承擔的認知與分析工作，現在已完全轉由 `perception/` 模組內的 **SigLIP TensorRT 引擎** 配合 **Zero-Copy 特徵提取** 所取代。

## 已停用組件
- [x] **`llm_engine.py`**: VLM HTTP 請求與 60 秒超時機制 (已從管線移除)。
- [x] **`yolo-vlm-backend.service`**: Systemd 後台服務 (已停用，釋放約 6.5GB VRAM)。
- [x] **`resource_manager.py`**: 由於不再需要動態調配 `-c` 與 `-np`，VRAM 管理器轉為備用。

## 歷史里程碑 (存檔)
- 曾成功實作基於 VRAM 狀態的模型動態載入。
- 曾成功達成 16-Slot 並發處理與 `[image_0]` 多模態提示詞的閉環分析。

## 架構演進
- 有關「特徵提取 (Feature Extraction)」的最新進展，請參閱 [perception.md](perception.md) 的 Phase 1 ~ 4 紀錄。
- 有關「時空記憶 (Memory)」的進展，請參閱 [storage.md](storage.md)。
