# Perception 模組進度

## 狀態：進行中

## 已完成
- [ ] detector.py — YOLO26 基礎推理封裝
- [x] zero_copy.py — NVDEC → CUDA Tensor 路徑驗證

## 進行中
- [ ] entropy.py — 資訊熵閾值調校

## 待處理
- [ ] 與 cognition 觸發介面對接
- [ ] TensorRT 加速實測

## 已知問題
- VRAM 在多路串流下峰值超出預期，待 resource_manager 協調

## 最後更新
2026-04-12
