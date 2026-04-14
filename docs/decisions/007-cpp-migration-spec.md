# ADR 007: C++ Migration & Dynamic Linking Specification

## 1. Context (背景)
為了消除 Python 的性能瓶頸 (GIL, Interpreter Overhead)，Saccade 核心感官管線將遷往 C++。此遷移必須保證模組化與可擴展性。

## 2. Decision (決策)

### 2.1 程式風格與標準
- **標準**：C++17 (平衡現代化功能與編譯器兼容性)。
- **命名**：類別 `PascalCase`, 函式 `camelCase`, 變數 `snake_case`, 私有成員 `snake_case_`。
- **管理**：強制使用 RAII (Resource Acquisition Is Initialization)，嚴禁裸指針 (`new`/`delete`)，優先使用 `std::unique_ptr`。

### 2.2 動態連結架構 (Shared Library)
- **解耦**：所有核心組件 (Perception, Media, Tracking) 必須編譯為獨立的動態連結庫 (`.so`)。
- **導出控制**：使用 `__attribute__((visibility("default")))` 僅暴露公共接口，縮小符號表，加速加載。
- **依賴管理**：利用 RPATH 確保 `.so` 在運行時能正確定位彼此。

### 2.3 接口設計 (Interface-First)
- **抽象層**：定義抽象基類 (如 `IPerceptionEngine`)，具體實現 (如 `TensorRTDetector`) 隱藏在私有標頭中。
- **Zero-Copy 契約**：接口必須支持直接傳遞 `cudaPtr` (Raw Pointer 或 `std::shared_ptr<CudaBuffer>`)。

### 2.4 異常與錯誤處理
- **內部**：使用 `std::runtime_error` 拋出嚴重錯誤。
- **邊界**：C API 封裝層必須捕獲所有異常並轉化為 `int` 錯誤碼，確保跨語言兼容性。

## 3. Architecture Layering (分層架構)
1. **Infrastructure**: `libcommon.so` (Logging, CudaHelpers).
2. **Perception**: `libperception.so` (TRTEngine, Detectors, Extractors).
3. **Tracking**: `libtracker.so` (GPUByteTracker).
4. **Media**: `libmedia.so` (GStreamer Wrappers).
5. **Node**: `saccade_node` (Executable, Pipeline Orchestrator).

## 4. Trade-offs (權衡)
- **優點**：極致延遲、模組化熱插拔、資源精確控制。
- **缺點**：開發難度增加、編譯時間變長、需精確管理 ABI 兼容性。
