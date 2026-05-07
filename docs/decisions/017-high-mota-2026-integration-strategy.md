# ADR 017: 2025-2026 高 MOTA 技術整合策略分析

## 1. 背景與動機 (Background)
根據 2025-2026 年多目標追蹤 (MOT) 領域的最新開源研究報告，MOT 技術已從單純的外觀/特徵比對 (ReID) 轉向更深層次的運動建模 (Motion Modeling)、端到端預測與品質感知關聯 (Quality-Aware Association)。

目前 Saccade 專案已具備：
- **ADR 013**: 基於 C++/CUDA 的 `GPUByteTracker`，採用 Saccade Heartbeat (每 10 幀提取一次 SigLIP2 特徵) 以降低負載，並具備初步的 GMC 機制。
- **ADR 015**: 全 GPU 原生的 Sinkhorn-Auction Hybrid 混合關聯算子，解決了 CPU-GPU 同步瓶頸，達成了 Zero-Copy 追蹤。

為了進一步突破在高度遮擋、非線性運動及高密度場景下的 MOTA (Multiple Object Tracking Accuracy) 上限，我們對最新研究進行了可行性分析，總結出以下兩大整合路徑。

## 2. 整合路徑一：運動補償升級 (參考 UCMCTrack)

### 2.1 現狀挑戰
目前的 `GPUByteTracker` 依賴卡爾曼濾波器進行線性運動預測，並且其 `gmc.hpp` 雖然實作了基於 OpenCV Lucas-Kanade 的全局運動補償 (GMC)，但這依賴於 CPU 運算，部分破壞了純 GPU 的 Zero-Copy 管線。

### 2.2 升級策略
參考 UCMCTrack（無外觀線索的高效追蹤）的核心理念：
- **2D 地平面映射馬氏距離 (MMD)**：將 3D 世界的運動透過透視變換投射到 2D 影像地平面，以此計算距離矩陣。這可以在 `Phase 0: Geometric Gating` (幾何粗篩) 階段，對 Bbox 的底部中點 (Bottom-center) 進行透視變換調整。
- **純 GPU 均勻相機補償 (Uniform CMC)**：將目前的 OpenCV 依賴替換為純 CUDA 實作的輕量級相機補償算子。這將徹底免除特徵點匹配的 CPU/GPU 數據傳輸 (D2H)，提升高負載下的實時追蹤極限。

### 2.3 預期效益
能在不依賴高成本 ReID 特徵的情況下，單憑運動幾何與補償機制，大幅提升追蹤穩定度，並在 16 路以上串流時維持極高的 FPS。

## 3. 整合路徑二：品質感知遮擋處理 (參考 SelectMOT)

### 3.1 現狀挑戰
根據 ADR 015，我們的 Sinkhorn-Auction 混合分配引擎目前基於 `(1-w)*IoU + w*ReID` 構建代價矩陣 (Cost Matrix)。然而，如果輸入的 YOLO26 檢測框本身因為嚴重遮擋而品質不佳，再好的分配演算法也會產生誤判 (ID Switch)。

### 3.2 升級策略
參考 SelectMOT 的「兩階段選擇匹配」(TSSM) 與品質感知 (Quality-Aware) 框架：
- **動態檢測選擇模組 (DSM in Phase 0)**：在 `Phase 0: Geometric Gating` 插入品質評估邏輯。結合 YOLO 的 Confidence Score、Bounding Box 幾何異常變化率，為每個檢測框動態生成一個「Quality Score」。
- **品質加權代價矩陣 (Quality-Weighted Sinkhorn)**：修改 `fused_sinkhorn_topk_kernel`，將「Quality Score」作為 Sinkhorn 邊際分佈的先驗權重。低品質框將在並行拍賣 (Auction) 階段被抑制其競價權重。
- **狀態選擇更新模組 (SSM in Post-Update)**：在 `track_state_update_pre_kernel` 階段，若目標關聯到低品質框，則動態放大卡爾曼濾波器 $R$ 矩陣 (觀測噪聲)，強制系統更信任預測而非有缺陷的觀測結果。

### 3.3 預期效益
極大強化系統在擁擠人群或頻繁遮擋場景下的表現。透過演算法層面的品質過濾，最大化 GPU Sinkhorn 算子的分配精準度，降低虛假軌跡生成。

## 4. 下一步行動計劃 (Action Plan)

1. **優先切入方向**：修改 `src/tracking/` 下的 CUDA Kernel，實作 **方向二 (Cost Matrix 的品質加權)**。這不需要改動外部模型，且能直接在現有 Sinkhorn 架構上發揮效益。
2. **具體實作**：
   - 擴充 `Track` 與 `Detection` 結構體以包含 Quality metrics。
   - 更新 `fused_sinkhorn_topk_kernel` 接受品質分數，並將其整合到 cost 計算或 probability 正規化中。
3. **後續實驗**：在 MOT17 等標竿資料集上驗證修改後的 GPU 算子效能與 MOTA 提升幅度。
