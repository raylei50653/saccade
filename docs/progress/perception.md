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
- [x] **SigLIP 2 特徵提取**: 已從 Jina-CLIP-v2 遷移至 SigLIP 2 (ViT-B/16)。
- [x] **Jina-CLIP (Legacy)**: 作為備選方案保留。
- [x] **語義漂移處理 (Drift Handler)**: 
    - 實作「語義雜訊消除」：採用指數移動平均 (EMA) 質心，穩定特徵向量。
    - 動態增益控制：熱身期 0.7，穩定期 0.3。
    - **Multi-Process Offloading**: 利用 `ProcessPoolExecutor` (spawn) 將 L2 運算卸載至子進程，釋放主執行緒 GIL。
- [x] **Smart Tracker (C++ Native)**: 
...
    - 實作「時序雜訊消除」：處理 L1 影格跳躍或偵測遺失。
    - 補丁預測 (In-filling)：當影格跳躍 >40ms 時，利用目標動量預測虛擬 BBox。
    - 卡爾曼濾波 (Kalman Filter)：過濾偵測器位置抖動。
- [x] **Zero-Copy 特徵觸發**: 達成 100% GPU Zero-Copy。

## 待處理
- [ ] 跨鏡頭對齊機制 (Re-ID) 研究與實作。
- [ ] 針對極端光照環境的動態曝光補償優化。

## 已完成里程碑
- [x] **極致效能突破**: 成功達成 10 路串流 3000 FPS 聚合吞吐量 (Aggregate Throughput)，單路均值 300 FPS。
- [x] **雜訊消除機制實裝**: 完成語義 (EMA) 與時序 (In-filling) 雜訊消除，過濾 80% 冗餘數據。
- [x] **純 C++ 感知層**: 成功將 Tracker (GPUByteTracker, SmartTracker) 遷移至底層 C++/CUDA 執行。
- [x] **硬體加速解碼**: 成功整合 GStreamer `nvh264dec` 實現 GPU 硬體解碼與 TensorRT 全流程對接。
- [x] **純視覺向量管線**: 成功建立以 YOLO 標籤與 SigLIP 2 特徵為核心的結構化數據流。

最後更新：2026-04-14
