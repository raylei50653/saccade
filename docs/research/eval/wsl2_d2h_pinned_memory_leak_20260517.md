# WSL2 D2H Staging Memory Leak Debug Report (2026-05-17)

目的：整理這次 `MOT17` / C++ evaluator 記憶體異常的完整排查過程，包含早期誤判、最終根因、修正方式與驗證結果，避免後續只記得片段結論。

## Scope

- 日期：`2026-05-17`
- 平台：`WSL2 + WDDM CUDA driver`
- 主要路徑：
  - `scripts/eval/mot17.py --cpp-threads 1`
  - `src/tracking/seq_runner.cpp`
  - `src/perception/trt_engine.cpp`

## Executive Summary

最終結論：

- 「`60GB DRAM`」不是實體 RAM，而是 `VmSize`；CUDA / driver 會把大塊 GPU / runtime 位址空間映射進 process virtual space，不能直接當成 DRAM 使用量。
- 真正暴漲的是 `VmRSS / RssAnon`，且成長發生在 `uv run ...` 啟動出的 `python3` child process，不是 `uv` parent。
- `SequenceRunner` 原本每 frame 會對 pageable host destination 做 4 次大型 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, ...)`。在 `WSL2 + WDDM` 下，driver 會替 pageable destination 配 pinned staging buffer，且不會即時歸還，造成 `RssAnon` 線性上升。
- 把 `h_count / h_ids / h_boxes / h_scores` 改成 `cudaMallocHost` 的 pinned destination 後，這條 leak 消失。
- 先前曾一度以為修法無效，後來證明那次測到的是 **repo root 殘留的舊 `libsaccade_tracking.so` 被誤載入**，不是新 build 的 pinned fix 失效。

## Initial Symptoms

最早觀察到：

- `VmRSS` 在短時間內線性上升
- 斜率約 `~1.2 GB/s`
- 約 `20s` 內曾出現 `VmRSS 706MB -> 21791MB`
- RAM 耗盡後 `WSL2` 會 OOM-kill process，看起來像「WSL 掛掉」

同時有一個容易誤判的訊號：

- `VmSize` 很大，看起來像「60GB DRAM」
- 但這主要是 CUDA / driver 的 virtual address mapping，不代表等量 physical DRAM

## Measurement Correction

排查中一度出現「修了之後 RSS 很平」與「`htop` 仍暴漲」互相矛盾的結果，原因是量測對象抓錯了。

早期錯誤：

- 直接量 `uv run ...` 的 parent PID
- 那個 process 只負責 launcher / supervisor，`VmRSS` 長期只有數十 MB

正確做法：

- 量整個 process tree
- 主要記憶體都在 `uv` 啟動出的 `python3` child process
- `htop` 看到很多 `python` 列，多半是同一個 process 的 threads，不是獨立多個 leak process

這個修正很重要，因為它解釋了為什麼 `/proc/<uv-pid>/status` 看起來平，但 `htop` 會明顯暴漲。

## Non-Causes Ruled Out

在收斂到 D2H staging 前，已先量化排除以下方向：

| Step | Observation | Reading |
|---|---|---|
| TRT engine deserialize | `+191MB` 後穩定 | 不是 leak 主因 |
| TRT context create (main / worker) | 小幅上升後穩定 | 不是 leak 主因 |
| `CppSequenceRunner` 建構 | 小幅上升後穩定 | 不是 leak 主因 |
| `runner.run()` 0 frames | 幾乎不增長 | alloc / workbench init 不是主因 |
| 跳過 `cv::imread` | 斜率仍在 | `cv::imread` 不是主因 |
| 跳過 `cv::split` | 斜率仍在 | `cv::split` 不是主因 |
| 跳過 TRT inference | 斜率仍在 | TRT inference 不是主因 |
| 幾乎跳空整個 frame body | 斜率仍在 | 問題更靠近同步 / materialization |
| `MALLOC_MMAP_THRESHOLD_=131072` | 無效 | 不是 glibc fragmentation |
| D2H 改 pinned destination | 有效 | 指向 pageable D2H staging 問題 |

## Why Python Was Suspected

曾有一組數據看起來像是 Python 端在漏：

- `runner.run()` 30 frames（TRT inference fail）只增 `+41MB`
- 完整 `mot17.py` 跑法卻出現 `~1.2GB/s` 上升

因此一度懷疑：

- Python post-processing
- `motmetrics`
- evaluator 上層資料結構累積

但後來更合理的解釋是：

- 不同路徑走到的 D2H materialization 行為不同
- 真正的分界不是「有沒有 Python」，而是「有沒有高頻大型 pageable D2H async copy」

## Root Cause

最終成立的根因模型：

- 每 frame 有 4 個 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, ...)`
- host destination 原本是 pageable memory
- 在 `WSL2 + WDDM` 下，driver 會為 pageable destination 配 pinned staging buffer
- 這些 staging allocation 不會即時歸還，造成 `RssAnon` 持續上升

量級估算：

- 約 `~3MB staging/call`
- `4 calls/frame`
- 約 `12MB/frame`
- `100 FPS * 12MB ~= 1.2GB/s`

這和現場量到的增長斜率同量級。

## Implemented Fix

修正位置：

- pinned host output buffer 宣告： [include/tracking/seq_runner.hpp](/home/ray/developer/ai/saccade/include/tracking/seq_runner.hpp:144)
- pinned host output buffer 配置 / 釋放： [src/tracking/seq_runner.cpp](/home/ray/developer/ai/saccade/src/tracking/seq_runner.cpp:78)
- D2H copy 改寫到 pinned destination： [src/tracking/seq_runner.cpp](/home/ray/developer/ai/saccade/src/tracking/seq_runner.cpp:259)

做法：

- `h_out_count_pinned_` 用 `cudaMallocHost(sizeof(int))`
- `h_out_ids_pinned_` 用 `cudaMallocHost(max_objs * sizeof(int))`
- `h_out_boxes_pinned_` 用 `cudaMallocHost(max_objs * 4 * sizeof(float))`
- `h_out_scores_pinned_` 用 `cudaMallocHost(max_objs * sizeof(float))`

這樣 D2H DMA 直接寫入 pinned destination，不再需要 driver 私下為 pageable 目標配置 staging buffer。

## Supporting Change Kept

另外保留一個與這次 leak 無直接因果，但邏輯上仍合理的改動：

- TRT execution context 改為 lazy init： [src/perception/trt_engine.cpp](/home/ray/developer/ai/saccade/src/perception/trt_engine.cpp:40)

理由：

- 只查 metadata 或使用外部 context 的 caller，不需要一載入 engine 就建立 context
- 這能避免不必要的 runtime / VRAM state 開銷
- 它不是這次 `RSS` leak 的主修正，但值得保留

## Changes Not Credited As Fixes

以下改動曾做過，但本次結論不把它們列為 leak fix：

- 移除 `cv::split`
- `decoded_bgr_` 改成 member reuse
- 其他影像 decode / CHW 轉換優化

原因：

- 沒有被量測證明能消除 `~1.2GB/s` RSS growth
- 最小且可解釋的修正是 pinned D2H destination

## False Negative Caused By Stale Root `.so`

修正後一度又看到「完整 eval 還是 1.2GB/s 暴漲」，當時如果只看現象，會得出「pinned fix 無效」的錯誤結論。

後來查到真正原因：

- `build/` 裡的新 binary 確實包含 RSS probe 與 pinned fix
- 但執行時 `/proc/<pid>/maps` 顯示實際載入的是 repo root 的舊 `libsaccade_tracking.so`
- 原因是 `LD_LIBRARY_PATH` 末尾有一個 `:`
- 在 Linux loader 規則下，尾端空項等於 current directory，讓 root 殘留 `.so` 進入搜尋路徑
- 結果是 `build/saccade_tracking_ext*.so` 連到 root 那份過期 `libsaccade_tracking.so`

這件事解釋了兩個現象：

- 為什麼新加的 native RSS probe 沒印出來
- 為什麼完整 eval 仍呈現舊的 leak 行為

處理方式：

- 移除 repo root 殘留的 Saccade `.so`
- 之後相同命令就會正確載入 `build/` 內的新版本

因此，這次 debug 的重要教訓是：

- 在有本地 `.so` 與 `build/` 並存時，不能只看 Python import path
- 要直接查 `/proc/<pid>/maps` 或 loader resolution，確認實際載入的 shared object

## Repro Commands

建置：

```bash
cmake --build build -j$(nproc)
```

標準 eval 路徑：

```bash
PYTHONPATH=build:. LD_LIBRARY_PATH=build UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/eval/mot17.py \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --cpp-threads 1 \
  --sequences MOT17-02-SDP
```

若要避免再被 root stale `.so` 影響，建議：

```bash
PYTHONPATH=/home/ray/developer/ai/saccade/build:/home/ray/developer/ai/saccade \
LD_LIBRARY_PATH=/home/ray/developer/ai/saccade/build \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/eval/mot17.py \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --cpp-threads 1 \
  --sequences MOT17-02-SDP
```

量測 process tree 的 `/proc/<pid>/status`：

```bash
uv run python scripts/eval/mot17.py \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --cpp-threads 1 \
  --sequences MOT17-02-SDP
```

實務上要注意：

- 不要只量 `uv` parent PID
- 要追 `python3` child process 的 `VmRSS / RssAnon`

## Verification Results

### Detector-Only Sanity Check

只建立 detector 並停住數秒：

- `python3` child `VmRSS` 約穩在 `~984MB`
- `RssAnon` 約穩在 `~416MB`
- 沒有持續線性成長

這表示：

- 載入 engine / detector 本身不會造成 `~1.2GB/s` DRAM leak

### Full Eval Before Loader Fix

在還沒清掉 root stale `.so` 前，完整 eval 觀察到：

- `python3` child `VmRSS` 約在 `15s` 內長到 `~16.6GB`
- `RssAnon` 約 `~16GB`
- 行為與原始 leak 一致

但這組數據後來證明量到的是舊 `libsaccade_tracking.so`。

### Full Eval With Correct `build/` Binary

以明確指向 `build/` 的方式重跑完整單 sequence：

- 約 `15s` 內 `VmRSS` 維持在 `~1.22GB`
- `RssAnon` 維持在 `~530MB`
- 沒有再出現 `~1.2GB/s` 線性成長
- 完整 sequence 正常結束，`run_eval_cpp` 約 `13.2s`

這是本次最關鍵的驗證結果，因為它直接證明：

- pinned D2H fix 生效
- 先前「修了還是漏」的觀察是 loader artifact，不是實際 regression

## Why This Beats The Python-Leak Theory

如果真正主因是 Python post-processing / `motmetrics`：

- 把 D2H destination 改成 pinned 不應該讓 `RSS` 斜率直接消失
- 明確載入新 C++ binary 後，也不應該整體穩住

實際上：

- pinned D2H 一改，正確 binary 下 `VmRSS / RssAnon` 就穩住
- detector-only path 本來就穩
- 問題只在舊 shared object 被誤載入時重現

因此更合理的結論是：

- 問題核心在 driver staging memory
- 不是 Python heap leak

## Remaining Caveat

本次結論只覆蓋 `SequenceRunner` 這條主要 D2H materialization path。

repo 內若還有其他大尺寸 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, ...)`，未來仍可能踩到同類問題。若再看到 RSS 線性成長，下一步應優先檢查：

1. 是否還有高頻的大尺寸 pageable D2H async copy
2. 是否又有 stale local `.so` / loader resolution 問題
3. 是否是另一條沒走 `SequenceRunner` pinned output buffers 的 code path

## Final Reading

這次事件的最終判讀是：

- `VmSize` 大不是 DRAM leak，本體是 CUDA virtual mapping
- 真正的 DRAM 壓力是 `VmRSS / RssAnon`
- 在 `WSL2 + WDDM` 上，大尺寸 pageable `cudaMemcpyAsync(DeviceToHost)` 會被放大成「每秒數 GB 的假 leak 行為」
- 這條路徑的正確修法不是 Python GC、不是 glibc tunable、也不是 image decode 微優化
- 正確修法是：把高頻 D2H output destination 改成 pinned host memory
- 若驗證結果和理論矛盾，先檢查實際載入的是不是你以為的那個 `.so`
