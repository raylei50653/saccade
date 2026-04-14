# Runbook: VRAM OOM 處理指南

Saccade 的感知快路徑 (L1) 必須保持運作。當 VRAM 負擔過重時，系統應按照以下優先級進行調整。

## 1. 症狀識別
- `nvidia-smi` 顯示 VRAM 使用率超過 90%。
- `Detector` 拋出 `RuntimeError: CUDA out of memory`。
- 推流延遲顯著增加 (> 500ms)。

## 2. 緊急處置流程
1. **停止認知擴展 (L5)**: 若有啟動 LlamaIndex 或 LLM 推理，優先將其關閉。
2. **降低去重頻率 (L2)**: 提高 `SemanticDriftHandler` 的相似度閾值 ($\epsilon$)，減少 CLIP 提取次數。
3. **增加跳幀 (L1)**: 修改 `main.py` 的 sleep 時間或設定為 `frame_id % 2 == 0` 才進行偵測。

## 3. 自動恢復機制
- `ResourceManager` 會監控 `vram_limit`，若低於閾值，系統會自動切換至 `minimal` profile 並卸載不必要的 VRAM 緩存。
