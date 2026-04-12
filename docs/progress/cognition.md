# Cognition 模組進度核對 (2026-04-12)

## 模組狀態：進行中 (llama-cpp 優先策略)

## 1. LLM 引擎 (llm_engine.py)
- [x] **基礎實作**: llama-cpp-python 類別封裝。
- [x] **GPU 加速**: 支援 `-ngl` (n_gpu_layers) 硬體 offload。
- [ ] **動態參數**: 支援根據 resource_manager 反饋動態調整上下文長度 (n_ctx)。
- [ ] **本地模型緩存**: 自動偵測並載入指定的 GGUF 權重。

## 2. 視覺推理 (vlm_engine.py)
- [ ] **模型選型**: 確定期望使用的 Qwen-VL / SigLIP 權重格式。
- [ ] **視覺提示詞**: 實作從 OpenCV 影像轉為 VLM 可接受的 Base64 或 Tensor 格式。
- [ ] **推理循環**: 整合視覺特徵與 LLM 文字推理。

## 3. 幀選取器 (frame_selector.py)
- [ ] **熵值介面**: 接收來自 `perception/entropy.py` 的觸發信號。
- [ ] **MediaMTX 接取**: 從 RTSP 串流抓取關鍵幀 (NVDEC 優先)。
- [ ] **多幀批處理**: 同時選取多個關鍵幀以增加視覺理解深度。

## 4. 資源調度器 (resource_manager.py)
- [ ] **VRAM 監控**: 實作實時 GPU 記憶體監測 (pynvml)。
- [ ] **權重切換**: 當 VRAM 不足時，主動釋放或減小 LLM 的 GPU 層數。
- [ ] **Offload 策略**: 定義系統在 Perception (快路徑) 與 Cognition (慢路徑) 之間的權重分配。

## 已知問題
- ⚠️ 目前所有模型載入為阻塞式 (Blocking)，需考慮非同步加載以防慢路徑卡死系統。
- ⚠️ 缺乏 GGUF 模型清單紀錄。

## 優化與未來計畫
- [ ] **TensorRT-LLM**: 完成 llama-cpp MVP 後，評估 TensorRT-LLM 的導入以降低慢路徑延遲。
