# GPUByteTracker 深入解析

**GPUByteTracker** 是 Saccade 系統感知層 (L1) 最核心的多目標追蹤 (MOT) 引擎。為了徹底消除 CPU 與 GPU 之間的資料傳輸瓶頸 (PCIe Overhead)，整個追蹤演算法從底層使用 C++ 與 CUDA 重新實作，實現了「輸入 Tensor → 內部狀態更新 → 輸出追蹤結果」的完全零拷貝 (Zero-Copy) 與零同步 (Zero-Sync) 流程。

## 1. 核心架構特點

### 1.1 全 CUDA 實作與 Zero-Sync
傳統的 ByteTrack 依賴 CPU 執行 Kalman Filter 與匈牙利演算法 (Hungarian Algorithm)，這迫使 GPU 偵測出的 Bounding Box 必須先 `cudaMemcpy` 搬回 CPU，造成顯著的延遲。
GPUByteTracker 將所有的預測 (Predict)、匹配成本計算 (Cost Matrix Calculation) 與更新 (Update) 均寫為 CUDA Kernel。透過 `pybind11` 封裝，Python 層只需傳入 BBox 的 GPU 記憶體指標 (`uintptr_t`) 與 CUDA Stream，演算法便能以非同步的方式在 GPU 內執行完畢。

### 1.2 雙階段 Sinkhorn 匹配 (Dual-stage Sinkhorn)
為了因應高密度的工業場景，系統摒棄了無法在 GPU 上高效平行化的傳統匈牙利演算法，改採 **Sinkhorn 演算法** 來逼近最佳匹配。
- **第一階段 (高分框匹配)**：將信心度高於閾值的偵測框與現存軌跡進行匹配。
- **第二階段 (低分框匹配)**：將剩餘未匹配的軌跡與低分框進行匹配，以挽救被部分遮擋的目標 (繼承 ByteTrack 的核心精神)。

### 1.3 混合成本矩陣 (ReID Fusion Cost)
在計算匹配代價時，GPUByteTracker 融合了空間與語義資訊：
`Cost = (1 - w) * IoU_Distance + w * Cosine_Distance`
並內建 **Strong ReID Gate**：當語義相似度 (CosSim) 大於 `0.75` 時，無視空間距離的限制強制配對，這能極大程度地抵抗相機的劇烈晃動或目標的短暫消失。

## 2. 物理與環境補償機制

### 2.1 全域運動補償 (GMC - Global Motion Compensation)
攝影機的移動或震動會導致 Kalman Filter 的預測產生嚴重偏差。
系統在 Python 層利用 OpenCV 計算相鄰影格的光流 (Optical Flow) 產生 2x3 的仿射矩陣 (Affine Matrix)，並將此矩陣指標傳入 `update()` 函數。底層的 `gmc_kernel` 會在預測階段直接對 Kalman 的狀態向量 $(cx, cy, a, h, vx, vy)$ 進行仿射變換修正。

### 2.2 動態光線補償 (Light Compensation)
工業現場的光線變化 (如夜間或強光直射) 會影響偵測框的穩定性 (BBox Jitter)。
透過傳入 `light_factor`，GPUByteTracker 會動態調整 Kalman Filter 的測量雜訊共變異數矩陣 (Measurement Noise Covariance Matrix, $R$ 矩陣)。光線條件越差，信任預測模型的比重就越高，從而平滑夜間的軌跡。

## 3. 狀態與生命週期管理

GPUByteTracker 內建了完整的軌跡狀態機 (Track State Machine)：
- **Tentative (暫態)**: 新生成的軌跡，必須連續被偵測到 `confirm_streak` 次 (預設 3 次) 才會轉為正式軌跡。支援基於分數的自適應確認 (Adaptive Confirmation)。
- **Tracked (追蹤中)**: 穩定匹配的軌跡，提供可靠的追蹤 ID。
- **Lost (丟失)**: 無法匹配的軌跡，其特徵會暫存 `track_buffer` 幀。在此期間，若透過 ReID 再次匹配成功，則恢復為 Tracked。
- **Removed (移除)**: 超過緩衝時間的軌跡將被徹底釋放。在資源吃緊時 (VRAM > 96%)，資源管理器可觸發 Target Culling，主動降低 `track_buffer` 以清理記憶體。

## 4. 介面與整合

在 C++ 原始碼 (`include/tracking/tracker_gpu.hpp`) 中，核心介面定義如下：

```cpp
std::vector<TrackResult> update(
    float* boxes_ptr,       // GPU 偵測框指標 [x1, y1, x2, y2]
    float* scores_ptr,      // GPU 信心度指標
    int* classes_ptr,       // GPU 類別指標
    int num_dets,           // 偵測數量
    cudaStream_t stream,    // 綁定的 CUDA Stream
    float* embeddings_ptr,  // (可選) GPU SigLIP 2 特徵指標
    float* gmc_ptr,         // (可選) 仿射矩陣指標
    float light_factor,     // 光線補償係數
    float mid_thresh_scale  // 自適應降級係數
);
```

透過這些機制的結合，GPUByteTracker 在複雜場景下不僅能維持高達數百 FPS 的推論速度，同時具備對抗遮擋、晃動與光線變化的強大韌性。
