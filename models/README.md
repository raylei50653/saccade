# 模型權重管理 (Weights Management)

本目錄存放所有二進位模型檔案，這些檔案**不會**提交至 Git 倉庫。

## 下載清單與版本參考

| 模組 | 檔案 | 來源 (HF/Repo) | 版本 | 備註 |
| :--- | :--- | :--- | :--- | :--- |
| **Perception** | `yolo26-n.pt` | [ultralytics/yolo](https://github.com/ultralytics/ultralytics) | v26.0 | Nano 基礎權重 |
| **Cognition** | `qwen3.5-Q4_K_M.gguf` | [HuggingFace](https://huggingface.co/Qwen) | 2025-03 | LLM 推理引擎 |
| **VLM** | `siglip-so400m.bin` | [HuggingFace/google](https://huggingface.co/google/siglip-so400m-patch14-384) | v1.0 | 視覺特徵向量生成 |

## 快速編譯 (TensorRT)

在 `scripts/` 中有對應的編譯指令，建議在本地環境將 `.pt` 轉換為 `.engine` 以獲得最佳效能。
