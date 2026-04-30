# Native Scripts

本目錄集中放置 C++ / CUDA build 與 coverage 相關腳本。

## 目前腳本

- `rebuild.sh`
  - 重建 native extension 與相關 `.so` 連結。
- `coverage_native.sh`
  - 建立 coverage build、執行 native tests、輸出 gcov 摘要。

## 說明

- `scripts/test_native.sh` 仍保留在 `/scripts/` 根層，作為文件與日常驗證的穩定入口。
