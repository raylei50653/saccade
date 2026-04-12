# ADR 001: 選擇 llama.cpp 作為推理後端

## 狀態
已通過

## 背景 (Context)
需要在 VRAM 有限的環境下動態調整推理任務。vLLM 偏向高吞吐伺服器，對記憶體 offload 與頻繁切換權重支援較重。

## 決策 (Decision)
優先選用 `llama.cpp` 配搭 `llama-cpp-python` 進行快速實作與驗證。

## 未來優化 (Future Optimizations)
在系統穩定後，若推理延遲仍為瓶頸，考慮遷移至 **TensorRT-LLM** 以獲得更極致的 NVIDIA GPU 加速效能。

## 取捨 (Consequences / Trade-offs)
- **優點**: 支援 GGUF 與精細的 `-ngl` 控制，適合邊緣運算。
- **缺點**: 比起 TensorRT-LLM，吞吐量可能稍低。
