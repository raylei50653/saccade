# Perception & Extraction 模組進度 (2026-04-12)

## 模組狀態：極速穩定運行 (140+ FPS, Zero-Copy 實作完成)

## 已完成
- [x] **detector.py**: YOLO11 基礎推理封裝，支援 TensorRT (.engine) 優先載入，延遲 <10ms。
- [x] **追蹤系統**: 使用 `model.track` 實作跨影格標籤一致性。
- [x] **zero_copy.py**: 實作 OpenCV + NVDEC 硬體加速路徑，並透過 `media/mediamtx_client.py` 提供純 CUDA Tensor 輸出。
- [x] **cropper.py (Phase 1)**: 實作 ZeroCopyCropper，透過 `torchvision.ops.roi_align` 達成微秒級 (60µs) 零拷貝目標裁切。
- [x] **feature_extractor.py (Phase 2)**: 實作 TRTFeatureExtractor，載入 SigLIP SO400M TensorRT 引擎，達成 <30ms 之多目標並發語義特徵提取。
- [x] **tracker.py (Phase 3)**: 實作 SmartTracker，建立 IoU 與移動向量之非同步事件鉤子 (Event Hooks)，並在獨立 CUDA Stream 執行特徵提取，不阻塞 YOLO。
- [x] **drift_handler.py (Phase 4)**: 實作 SemanticDriftHandler，在 GPU VRAM 中計算 Cosine Similarity，精準過濾冗餘特徵，僅保留「語義漂移」狀態。

## 進行中
- [ ] 跨鏡頭對齊機制 (Re-ID) 研究。

## 已完成里程碑
- [x] **硬體加速解碼**: 成功整合 GStreamer `nvh264dec` 實現 GPU 硬體解碼與 TensorRT 全流程對接。
- [x] **純視覺向量管線**: 成功移除 VLM 依賴，以 YOLO 標籤與 SigLIP 特徵建立結構化記憶。

## 最後更新
2026-04-12
