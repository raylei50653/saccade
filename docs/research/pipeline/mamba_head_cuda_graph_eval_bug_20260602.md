# Mamba Head CUDA Graph — eval 整合 bug 調查與修復紀錄 (2026-06-02)

> **✅ 已修復(2026-06-02)。** 真因不是 capture 寫法,而是 **custom
> `saccade::selective_scan_fwd` CUDA op 跑在 legacy default stream(stream 0)**:pybind
> binding 預設 `void* stream = nullptr`,SSM scan kernel 永遠在 stream 0 launch,而
> CUDA-graph capture 只錄 capture stream 上的工作 → scan 沒被錄進 graph → replay 時 `y`
> 從未被填 → cls 飽和 → ~10× FP → MOTA 崩。修法:(1) pybind binding 加 `stream_ptr`
> 並傳進 kernel launcher;(2) Python op 傳 `torch.cuda.current_stream(u.device).cuda_stream`;
> (3) graph path 改用 `torch.cuda.make_graphed_callables`。隔離 bit-exact(diff 0.0)、
> full-SDP eval 與 eager 在 run-to-run 噪音帶內 parity、FPS 95.5→110.2(+15%)。preset 已設
> `use_cuda_graph: true`。下方為原始調查紀錄(保留脈絡)。

---

## 0. 修復摘要(2026-06-02)

- **真因**:`selective_scan_fwd` kernel launch 在 legacy default stream,CUDA-graph capture
  不錄該 stream → replay 缺少 scan kernel → 下游讀到未填的 `y` → cls 飽和。
  之所以隔離測試「bit-correct」是因為當時沒在 graph 下跑 SSM op,或 microbench 沒驗 MOTA;
  一旦在真 graph capture 下跑 CUDA op(本次 `make_graphed_callables`)就立刻重現(隔離 diff ~140)。
- **修法**:
  - `src/tracking/tracker_gpu_python.cpp` — `selective_scan_fwd` binding 新增 `uintptr_t stream_ptr`,
    傳給 `selective_scan_fwd` / `selective_scan_fwd_half`(header 本就有 `void* stream`)。
  - `src/saccade/perception/temporal_yolo/mamba_head.py` — custom_op 與 fallback 兩處均傳
    `torch.cuda.current_stream(u.device).cuda_stream`;`forward` 的 graph 分支改走
    `_graphed_forward` → `torch.cuda.make_graphed_callables`(只蓋單幀、非 temporal/flow 路徑)。
- **驗收**:隔離 distinct 輸入 bit-exact(0.0);full SDP 4 次跑(off×2/on×2)精度互相重疊於噪音內
  (on#2 = IDF1 73.8/MOTA 77.8/FP 5392,落在 off 帶 73.8–74.0 / 77.8–78.0 / 5158–5343);
  FPS 95.5→110.2(+15%);nsys graph-on `cudaLaunchKernel` ~172/幀(head 收成單一 graph launch)。

相關:[detect 歸因 memory](../../../) · `mamba_head.py` · `mamba_gated_detector.py` ·
ADR/research [eval 並發](../../reference/concurrent_eval.md)

---

## 1. 背景與動機

nsys 量到單流 `mot17.py --preset mamba_optimal --detector SDP` 的 detect 是 **launch-bound**:
- GPU 利用率 **17%(83% idle)**;每幀發 **~278 個 kernel**,`cudaLaunchKernel` CPU 開銷
  **4.25 ms/幀 > GPU 真正運算 3.4 ms/幀**;同步(`cudaStreamSynchronize` 等)僅佔 ~5% wall。
- 正解是 **CUDA graph**:把每幀數百個小 launch 收成一次 replay。mamba_head 是最大單一成本
  (44%),且輸入 shape 固定(P3 80²/128, P4 40²/256, P5 20²/512)、gate=identity、v14 單幀
  (`temporal_T=0`)→ 完全符合 graph 靜態前提。

## 2. 兩個 bug

### Bug A — graph 在 eval 從未啟用(grad fallback)
- `mamba_head.forward`(`mamba_head.py:629-635`)在 `torch.is_grad_enabled()` 為真時走
  eager fallback。
- eval 路徑(`evaluator.py` / `detection.py:detect_native_640 → detect_single_patch_640
  → detector.detect_raw → MambaGatedDetector.forward`)**從來沒有 `no_grad`/`inference_mode`**
  → grad 預設開啟 → 永遠 eager。
- 所以 preset 寫 `use_cuda_graph: true` 是**死的**;過往「CUDA-graph 1.80× / 省 1.33ms」只是
  從未進 eval 的 microbench。線上一直是 278 launch/幀、83% idle。
- **修法**:已在 `MambaGatedDetector.forward` / `_detect_from_feats` / `_detect_batch` 加
  `@torch.no_grad()`(正確、略快、且是 graph 生效前提;數值不變)。

### Bug B — graph 啟用後在完整 eval 下輸出損毀
加 `no_grad` 後 graph 確實 capture(印出 "Capturing new CUDA Graph"),但 MOTA 崩潰。

逐幀對照(MOT17-05-SDP,同 60 幀,`SACCADE_DBG` 印 head 輸入/輸出):

| | eager | graph |
|---|---|---|
| head **輸入** p3 mean/std/max | 0.1254 / 0.5869 / 4.668 | **逐位元相同** |
| head **輸出** cls max | ~0.98 | **1.0000(飽和)** |
| n(score>0.5) / 幀 | 21–37 | **245–452** |
| 偵測框 / 幀(MOT 輸出) | 4–7 | 17–30 |
| MOTA(60 幀) | **3.9%** | **−28.9%** |

→ **head 輸入完全相同,但 graph 的 cls 輸出飽和到 1.0**,造成 ~10× false positive。
框座標仍合理,純粹是分數被污染。

## 3. 隔離測試矩陣(全部 bit-correct,無法重現 Bug B)

用同一 v14 ckpt + 真 TRT backbone,逐項排除:

| 測試情境 | eager vs graph 最大差 | 結論 |
|---|---|---|
| N=1 `_detect_batch` vs `_detect_from_feats` | 0(bit-equal) | layout/索引正確 |
| 多 distinct 輸入(head 直餵) | ≤1.4e-3 | graph 對輸入正確反應 |
| 真 backbone + `detect_raw`,逐幀 clone | ≤6e-5 | backbone 整合正確 |
| 重用同一 canvas buffer + 真實幀 | ≤1.5e-3 | buffer 重用無關 |
| 交錯 `tracker.update` | ≤1.5e-3 | 單純 tracker 無關 |
| 幀間大量記憶體 churn(40×4M tensor) | ≤1.4e-3 | torch graph private pool 有保護 |
| **不**逐幀同步(只在最後 sync) | ≤1.5e-3 | 非 copy_/replay race |
| fresh tensor/幀(複刻 `detect_single_patch_640` 非 letterbox 路徑) | ≤1.8e-3 | 輸入分配模式無關 |
| 返回前 clone static_outputs | 仍崩(−27.5%) | 非輸出別名 |

→ 隔離下一切正確,**只有完整 eval pipeline 會壞**。

## 4. 根因判定

手寫 `torch.cuda.CUDAGraph` capture(`mamba_head.py:637-678`,自管 `_cuda_graphs` /
`_static_inputs` / `_static_outputs` / warmup+capture)**對 eval runtime 環境不 robust**:
head 輸入正確、輸出損毀 → 是**捕獲後的 graph 工作緩衝/內部狀態在 eval 的其他 GPU 工作下被破壞**。
排除了輸入、輸出別名、torch 端記憶體 churn、copy/replay race。最可能的剩餘來源是 eval 完整
pipeline 才有的 **GMC / `tracker.update_into` 的 C++ GPU kernel(`saccade_tracking_ext`)或
stream/mempool 狀態**與 graph 私有 pool 的互動(尚未坐實到單一元件)。

## 5. 目前狀態(repo 已留安全)

- `configs/presets/mamba_optimal.yaml` → `use_cuda_graph: false`(含註解)。eval 已驗證回到
  eager 基準(60 幀 MOTA 3.9%)。
- `@torch.no_grad()` 保留(正確,且為日後修圖前提)。
- 無殘留 debug。
- 既有工具保留:`scripts/benchmarks/mamba_head_cudagraph.py`(eager vs graph replay)、
  `mamba_head_breakdown.py`、`mamba_detect_breakdown.py`。

## 6. 下次怎麼修(backlog)

1. **改用 `torch.cuda.make_graphed_callables` 包 `mamba_head`**(它正確管理 graph pool +
   stream + 靜態 I/O,對外部 GPU 工作 robust),取代 `mamba_head.py` 內手寫 capture。
   或:把 detection 釘在**專屬 CUDA stream** 上 capture/replay,並隔離 GMC/tracker 的分配。
2. **務必跑 full-eval MOTA parity**(本次 bug 潛伏正是因為過去只測 microbench 速度、從沒驗
   eval MOTA)。驗收門檻:per-seq MOTA/IDF1 與 eager 基準 ±0.3pp。
3. 預期收益:detect 7.45→~6.1 ms、production ~98→~110 FPS(+~12%),數值應不變。
4. 重新定位 Bug B 的確切元件:在 eval graph-on 下,逐步停用 GMC / `tracker.update_into` /
   interpolation,觀察 head 輸出何時恢復正常,坐實污染來源。

## 7. 重現指令

```bash
# Bug A(grad fallback):無 Capturing 印出
SACCADE_DBG=1 uv run python scripts/eval/mot17.py --preset mamba_optimal --detector SDP \
  --sequences MOT17-05-SDP --max-frames 6      # (在 no_grad 修正前)

# Bug B(啟用後崩):暫時把 preset use_cuda_graph 改 true 再跑
uv run python scripts/eval/mot17.py --preset mamba_optimal --detector SDP \
  --sequences MOT17-05-SDP --max-frames 60     # MOTA -28.9% vs eager 3.9%

# nsys launch-bound 佐證
nsys profile --trace=cuda,nvtx -o /tmp/ss uv run python scripts/eval/mot17.py \
  --preset mamba_optimal --detector SDP --sequences MOT17-05-SDP --max-frames 150
nsys stats --report cuda_api_sum /tmp/ss.nsys-rep   # cudaLaunchKernel 佔大頭,sync ~5%
```
