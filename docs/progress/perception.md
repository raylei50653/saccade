# Perception 模組進度 (2026-04-12)

## 模組狀態：穩定運行 (追蹤功能已開啟)

## 已完成
- [x] **detector.py**: YOLO26 基礎推理封裝。
- [x] **追蹤系統**: 使用 `model.track` 實作跨影格標籤一致性。
- [x] **標籤翻譯**: 實作 `get_actionable_labels` 將 Class ID 轉換為易讀標籤。

## 進行中
- [ ] **entropy.py**: 資訊熵閾值動態調校，避免事件頻率過高。
- [x] **zero_copy.py**: 已實作 OpenCV + NVDEC 硬體加速路徑 (穩定)，NVIDIA DALI 作為實驗性高效能路徑保留。

## 已完成里程碑
- [x] **硬體加速解碼**: 成功整合 GStreamer `nvh264dec` 實現 GPU 硬體解碼。
- [x] **效能基準數據**: 完成 OpenCV 與 DALI 路徑的對比測試 (1080p @ 30FPS 達成)。

## 待處理
- [ ] **TensorRT**: 遷移至 TensorRT 以進一步降低延遲。

## 最後更新
2026-04-12
