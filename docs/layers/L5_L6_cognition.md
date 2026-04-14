# L5: 檢索層 (Retrieval Layer)

## 1. 定義與目標
L5 是系統的「聯網介面與語義查詢入口」。負責將使用者的自然語言查詢轉換為向量，並從 L4 中檢索相關事件。目標是提供即時、低延遲的語義搜尋與錄影回放連動。

## 2. 核心組件
- **API 伺服器 (FastAPI)**: 提供 RESTful 介面。
- **語義查詢轉碼器 (Query Encoder)**: 使用 CLIP/SigLIP 將 Text 轉為 Vector。
- **時空過濾器 (Spatiotemporal Filter)**: 支援時間範圍、物件類別、置信度等過濾。

## 3. 資料流向
- **Input**: User Query (Text/Image)、Time Range。
- **Output**: 排序後的事件列表、錄影截圖、即時串流連結。

---

# L6: 認知與資源層 (Cognition Layer)

## 1. 定義與目標
L6 是 Saccade 的「決策大腦 (Decision Maker)」，負責高層級的資源管控與系統平衡。目標是確保系統在資源極限環境下依然能透過自適應策略存活。

## 2. 核心組件
- **資源管理器 (ResourceManager)**: 實時監測 VRAM 與算力負載。
- **降級決策引擎 (Degradation Logic)**: 決定系統是否進入 Stepped Degradation。
- **幀率調度器 (FrameSelector)**: 基於 L2 的漂移分數動態調整 L1 的偵測頻率。

## 3. 資料流向
- **Input**: VRAM Stats (NVML)、Latency Spike (L1)、Drift Score (L2)。
- **Output**: 降級指令 (Level 0-3)、POOL_SIZE 調整指令、Resolution Switch 指令。

## 4. 關鍵優化
- **Stepped Degradation (階梯式降級)**: 
    - **REDUCED (85%)**: 縮小 5-Buffer Pool，優化影格緩衝。
    - **FAST_PATH (92%)**: 關閉 L2 特徵提取，暫停語義漂移檢測。
    - **EMERGENCY (96%)**: 
        - **熱切換解析度**: 降至 320p 以降低 Tensor 佔用。
        - **目標清理 (Target Culling)**: 強制銷毀 Confidence < 0.4 或長時間靜止 (Low Drift) 的目標。
        - **緩衝區縮減 (Buffer Pruning)**: 將 Tracker 的 `lost_buffer` (追蹤遺失容忍度) 從 30 幀縮減至 5 幀，立即釋放物件歷史狀態與快取特徵。
- **Hysteresis (遲滯保護)**: 防止在臨界點反覆切換模式。
