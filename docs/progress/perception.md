# Perception & Extraction (L1-L2) 進度核對

## 系統架構層次
- [x] **L1: Perception (極速感測層 - YOLO26)** - 升級中
- [x] **L2: Deduplication (語義去重層 - SigLIP 2)** - 規劃中

## 模組狀態：架構升級中 (目標：NMS-Free & High Semantic Precision)

## L1: Perception (感測實作)
- [x] **YOLO26 推理**: 核心偵測器已切換至 YOLO26。支援 TensorRT (.engine) 與 NMS-Free 推理。
- [x] **Zero-Copy 管線**: 整合 GStreamer `nvh264dec` 與 CUDA Tensor 直接映射。
- [x] **極速裁切 (Cropper)**: 透過 `torchvision.ops.roi_align` 達成微秒級零拷貝目標裁切。
- [x] **跨影格追蹤**: 支援 YOLO26 格式之跨影格標籤一致性。

## L2: Deduplication (去重實作)
- [ ] **SigLIP 2 特徵提取**: 規劃從 Jina-CLIP-v2 遷移至 SigLIP 2 (ViT-B/16)。
- [x] **Jina-CLIP (Legacy)**: 現有 TensorRT 引擎可作為備選方案。
- [x] **語義漂移處理 (Drift Handler)**: 實作基於 GPU Cosine Similarity 的動態過濾機制。
- [x] **Smart Tracker (C++ Native)**: 完成智能追蹤與特徵提取排程的 C++ 實作，利用 pybind11 綁定 Python 3.12 虛擬環境，達成 100% GPU Zero-Copy 特徵觸發。

## 待處理
- [ ] 跨鏡頭對齊機制 (Re-ID) 研究與實作。
- [ ] 針對極端光照環境的動態曝光補償優化。

## 已完成里程碑
- [x] **純 C++ 感知層**: 成功將 Tracker (GPUByteTracker, SmartTracker) 遷移至底層 C++/CUDA 執行。
- [x] **硬體加速解碼**: 成功整合 GStreamer `nvh264dec` 實現 GPU 硬體解碼與 TensorRT 全流程對接。
- [x] **純視覺向量管線**: 成功建立以 YOLO 標籤與 Jina-CLIP 特徵為核心的結構化數據流。

最後更新：2024-05-23
