# L6: 資源層 (Resource - Resource Management)

## 1. 定義與目標
L6 是 Saccade 的「決策大腦」，負責高層級的 VRAM 資源監控與系統平衡。目標是確保系統在資源極限環境下透過自適應策略優雅降級，優先保證核心感知（L1）。

## 2. 核心組件
- **ResourceManager** (`src/saccade/resource/resource_manager.py`): 透過 NVML 實時監測 VRAM 負載，輸出降級指令。
- **FrameSelector** (`src/saccade/resource/frame_selector.py`): 基於 L2 漂移分數動態調整 L1 偵測頻率。

## 3. 資料流向
- **Input**: VRAM Stats (NVML)、Latency Spike (L1)、Drift Score (L2)。
- **Output**: DegradationLevel（NORMAL / REDUCED / FAST_PATH / EMERGENCY）。

## 4. 階梯式降級策略

| Level | VRAM 閾值 | 動作 |
|---|---|---|
| NORMAL | < 85% | 正常運行 |
| REDUCED | > 85% | 縮減 5-Buffer Pool 大小 |
| FAST_PATH | > 92% | 暫停 L2（SigLIP 2）與 L5（RAG） |
| EMERGENCY | > 96% | 解析度 640→320、Target Culling（Confidence < 0.4）、track_buffer 30→10 |

## 5. Hysteresis（遲滯保護）
- 升級門檻：85% / 92% / 96%
- 降級恢復：需降至觸發點 **-5%** 才恢復上一級，防止臨界點頻繁切換（Thrashing）。
