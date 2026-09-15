# Runbooks

本目錄提供常見維運與故障處理流程。

## 文件索引

- [native_extension_install.md](native_extension_install.md): 第三方安裝 `saccade_tracking_ext`（ADR 025）：支援矩陣、`pip install 'saccade[native-build]'` → `cmake -DPYTHON_EXECUTABLE=<venv>` → `saccade_build.pth` / `SACCADE_BUILD_PATH`、驗證腳本、故障排除。
- [runtime_identity_republication.md](runtime_identity_republication.md): runtime coordinate 候選捕捉（座標落後也能跑）、archive 與 promote 流程、math-model scoped re-audit。
- [nsys_profiling.md](nsys_profiling.md): nsys profiling 工作流（compile+CUDA graph 全開）、injection 死鎖禁忌 flag、hang 簽名、開銷校準。
- [stream_recovery.md](../../modules/streaming/runbooks/stream_recovery.md): 串流斷線恢復流程。
- [vram_oom.md](../../modules/resource/runbooks/vram_oom.md): VRAM OOM 緊急處置與降級策略。

## 維護原則

- Runbook 應聚焦操作步驟、症狀判斷、驗證方式與回滾手段。
- 若流程依賴腳本或 systemd 服務，請寫明實際檔案或命令。
