# nsys Profiling Runbook（compile + CUDA graph 全開路徑）

Date: 2026-07-07（取代 2026-06-20「只能 `--no-compile`」的舊工作流）

## 可用命令

```bash
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/wg \
  .venv/bin/python scripts/eval/mot17.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer \
  --sequences MOT17-04-SDP --output /tmp/wg_out

.venv/bin/python scripts/benchmarks/nsys_frame_attribution.py /tmp/wg.nsys-rep
```

torch.compile、`use_cuda_graph`、Triton cache 熱載入全部保持 production 狀態，**不需要 `--no-compile`**。

## 禁忌 flag（會 host 端死鎖）

| flag 組合 | 結果 |
|---|---|
| `--trace=cuda --sample=none --cpuctxsw=none` | ✅ 正常跑完 |
| 預設 flag（OSRT + CPU sampling + cpuctxsw 全開） | ❌ hang 在載入直到被殺 |
| `--trace=cuda,nvtx --sample=none --cpuctxsw=none` | ❌ hang（同款） |

歸因（2026-07-07 三組對照實驗）：CUPTI 的 CUDA trace、graph capture、Triton cache load **同時開是安全的**；致死的是 **OSRT tracing / CPU sampling** 與 **NVTX injection** 這兩層額外的 host 端 interposition（pthread / dlopen / file-lock 攔截），與 compile 路徑 host 端初始化（Triton cache materialize 的 temp + `os.replace`、fork、CUDA module load）交界處互鎖。

代價：**NVTX stage 標記不可用**。stage 歸因改用 kernel 名 / `graphId` / `streamId` 結構（`nsys_frame_attribution.py` 即是這條路，夠用）。

## Hang 判斷簽名（hang vs 只是慢）

- log byte 數凍住（mtime 不動）
- GPU util 0% 但仍佔記憶體（context 活著閒置）
- 主進程**全部** thread 都是 `S`、瞬時 CPU≈0、無一條 `R`
- 無 compiler 子進程（cc1plus / ptxas / cicc）
- `.nsys-rep` 沒在寫

hang 不會觸發任何完成事件——跑 nsys 一律掛 `timeout`。被殺後 nsys 的 `--start-agent` agent 與 `nsys-launcher` 會 linger，要 `kill -9`。

## 開銷校準（讀數規則）

node-mode tracing 會把 wall/frame 膨脹在 **host 側**（實測 seq09 3.37→4.04 ms、seq04 3.34→5.03 ms）；**kernel / graph span 是硬體時戳，可信**，GPU busy 不受影響。

- ✅ production bubble = production wall/frame（無 nsys 跑一次拿 FPS）− trace 的 GPU union busy/frame
- ❌ 直接讀 trace 裡的 device-idle gap（會被 host 膨脹放大 2–4×）

另注意：NVJPG 硬體引擎的 JPEG decode **不會出現在 kernel 表**——tail 裡的「idle」可能是 decode 在跑（線索：`rgba2rgb` 出現在 tail、JPEG bitstream HtoD、`cudaStreamSynchronize` 佔 tail 大宗）。

## 相關

- 歸因結果：[perf_attribution_whole_graph_m.md](../../research/pipeline/perf_attribution_whole_graph_m.md)
- kernel 層細節（s preset 時代，方法仍有效）：[whole-graph-kernel-fragmentation.md](../../modules/detection/research/whole-graph-kernel-fragmentation.md)
