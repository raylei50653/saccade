# ADR 009: 工業級零拷貝管線 V2 (Industrial Zero-Copy V2)

## 狀態
**已通過 (Accepted)** - 2026-04-14

## 背景 (Background)
在 Saccade 的高併發場景下，原有的單緩衝區/單 Stream 零拷貝機制存在兩個核心風險：
1. **競爭冒險 (Race Condition)**：當 Python 的 Cognition 分析速度慢於 GStreamer 採集速度時，緩衝區會被覆寫，導致分析殘缺影像。
2. **全域阻塞 (Global Blocking)**：使用單一 CUDA Stream 導致 H2D 搬運與推理計算串行化，無法充分利用 GPU 的並行硬體特性。

## 決策 (Decision)
我們決定重構 `GstClient` 與 Python 綁定層，引入以下機制：

### 1. 循環緩衝池與狀態機 (Buffer Pool & State Machine)
- 實作固定大小 (POOL_SIZE=5) 的 GPU 緩衝池。
- 引入原子狀態陣列 `buffer_states`：
    - `EMPTY`: 可供 C++ 寫入。
    - `WRITING`: 正在進行非同步 H2D 搬運。
    - `READY`: 搬運指令已發出，等待處理。
    - `PROCESSING`: Python 正在持有指標進行推理（鎖定狀態）。
- **Drop Frame 機制**：若所有緩衝區皆非 `EMPTY`，C++ 採集端將主動丟棄影格以維持實時性。

### 2. 並行傳輸 (Per-buffer CUDA Streams)
- 為每個緩衝區分配獨立的 `cudaStream_t`。
- 允許影格 $N$ 的搬運與影格 $N-1$ 的計算在 GPU 內部並行執行。

### 3. PyTorch 深度整合 (ExternalStream Integration)
- 透過 `pybind11` 暴露 `stream_ptr` 與 `cuda_ptr`。
- Python 端使用 `torch.cuda.ExternalStream(ptr)` 綁定，將推理算子直接排入緩衝區專屬流，消除 CPU 端同步阻塞。

### 4. RAII 生命週期管理 (Resource Guard)
- 在 Python 綁定層封裝 `FrameData` 物件。
- 利用 Python GC 或手動 `release()` 自動觸發 C++ 端的 `releaseBuffer`，確保狀態機自動回歸 `EMPTY`。

## 後果與折衷 (Consequences & Trade-offs)
- **優點**：
    - 徹底消除 Race Condition。
    - 推理延遲抖動 (Jitter) 顯著降低。
    - GStreamer 管線與 Python 邏輯完全解耦。
- **缺點**：
    - 增加 5 倍的影格緩衝 GPU 記憶體佔用（約 $5 \times 6MB = 30MB$ @1080p）。
    - 實作複雜度提高。

## 驗證指標
- 通過 `tests/benchmarks/bench_core_transport.py` 驗證影格流失率與同步正確性。
