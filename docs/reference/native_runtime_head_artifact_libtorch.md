# Native runtime head artifact — LibTorch（#465 Phase B PR-1L／U1 redesign）

> 狀態：PR-1L artifact 端完成；**不改** eval harness、preset、threshold、weights、benchmark claim，也沒有量 parity（那需要自己的預宣告，見 §6）。
> 路線依據：[failure-localization r2 結果](native_runtime_head_failure_localization_r2_result.md) = `EAGER_NUMERICS_WITHIN` ⇒ 下一個候選 head 形式優先 LibTorch；該 terminal **不接受任何形式**，本 artifact 仍須通過自己的 parity 宣告。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B1／B5（B1 的 shipping 責任明列「LibTorch TorchScript」為可用形式）。TRT 形式見 [PR-1](native_runtime_head_artifact.md)／[PR-1R](native_runtime_head_artifact_tf32_off.md)（皆被 parity 否決）。
> 工具：`scripts/model/export_headline_mamba_head_torchscript.py`（`developer_build_debug`）；operator：`src/tracking/mamba_scan_torchop.cpp` → `build/libsaccade_scan_torchop.so`。

---

## 1. PR-1L 做的選擇

| 軸 | 選擇 | 理由 |
|:--|:--|:--|
| head 形式 | **TorchScript（`torch.jit.trace`）＋ `libsaccade_scan_torchop.so`**，執行時 graph executor optimize 關 | r2 的 `R_E`（eager head 數值＋compiled S2）在容差內，而 optimize 關的 TorchScript interpreter 呼叫的正是 eager 的 aten kernel（§3 結構檢查逐位元相等）。AOTInductor（`torch.export`）跑 Inductor 生成的 kernel，數值不是 `R_E` 測到的那一種，本 PR 不採用 |
| selective scan | 新的 C++ operator **`saccade_native::selective_scan_fwd`**（`TORCH_LIBRARY`），與 Python custom op `saccade::selective_scan_fwd` 同參數、呼叫同一個 `selective_scan_fwd` launcher、在 current CUDA stream 上啟動 | 既有的 `saccade::` op 只在 Python 註冊（`mamba_head.py` 的 `torch.library.custom_op`），C++ 載入時會回呼 Python，不符 G2。namespace 分開，所以同一個 process 同時載入兩者（parity harness）不會重複註冊，traced graph 也只會指向一個實作 |
| artifact 範圍 | **只有 head**：`p3/p4/p5 → cls_p3..p5, reg_p3..p5`（同 PR-1） | S2 留給 U3b；`conf_thr`／`max_det` 只有 B2 resolved config 一個來源 |
| 精度與形狀 | **FP32**；batch 靜態 1；輸入形狀靜態（`(1,128,80,80)`／`(1,256,40,40)`／`(1,512,20,20)`） | 與 oracle 相同。trace 只對這組形狀成立（§2 第 6 項） |

head 由**與 oracle 相同的建構路徑**產生（重用 `export_headline_mamba_head.build_head` → `build_mamba_gated_detector(..., trt_backbone_engine=<preset>, use_whole_graph=True)`），trace 的是 `_whole_graph_fn` 實際呼叫的 `MambaDetectionHead._forward_eager`（T=1、`return_embeddings=False`）。trace 期間只把 `mamba_head._saccade_selective_scan_op` 暫時換成 native operator，結束即還原；不修改 `src/saccade/`。

## 2. Lineage 與 fail-closed 規則

與 PR-1 相同的輸入檢查（重用同一份程式）：preset 的 `use_whole_graph`、preset ckpt 路徑 == inventory `s.t3t1_phase_b`、ckpt sha256 == inventory、head 以自己的 loader probe 載入乾淨（missing／unexpected 皆空）。另外：

1. operator library 必須存在，其 `DT_NEEDED` 不得含 `libpython*`、`libtorch_python*`；
2. traced graph（inlined）不得含 `prim::PythonOp`、不得呼叫 `saccade::selective_scan_fwd`，且必須至少呼叫一次 `saccade_native::selective_scan_fwd`；
3. trace 期間的每一個 `TracerWarning` 都必須來自 allowlist 的 source line。allowlist 只有 `_selective_scan_cuda` 裡三個**對 shape 的** Python 分支（state 數檢查、shared／per-channel 判定、rank-1 C broadcast），它們由權重與靜態輸入形狀決定，不由資料決定。其他 TracerWarning（可能把資料相依的分支烤成常數）一律拒絕；
4. **身分**：`torch.jit.save` 的檔案 bytes 每次都不同（`.data/serialization_id` 是隨機值，`*.debug_pkl` 記錄 trace 的呼叫堆疊與絕對路徑），兩者都不影響執行。manifest 記錄檔案 `sha256`（只識別這一份檔案）與可攜的 **`content_sha256`**（其餘 entry 名稱＋bytes 排序後的 sha256）；`--check` 以 `content_sha256` 比對重新 trace 的結果；
5. **runtime requirements**（consumer 必須遵守，記錄在 manifest）：graph executor optimize 關、cuDNN benchmark 關、cuDNN TF32 允許、matmul TF32 關。後三項是 oracle harness 實際使用的 torch 預設值，也是 LibTorch C++ 的預設值；
6. 形狀：artifact 只對 §1 的靜態輸入形狀成立（第 3 項的 shape 分支已固定）；
7. **operator 邊界**（shipping runtime 的 C++ boundary）：`saccade_native::selective_scan_fwd` 在任何 raw pointer 進 CUDA launcher 之前，依 Python op 被 `_selective_scan_cuda` 呼叫時的 contract 檢查所有輸入：非空 tensor 皆在 `u` 的 CUDA device、dtype 與 `u` 相同；`u` 為非空 `(B,L,D)` 且可 32-bit 索引；`delta` 形狀同 `u`；`a_per_channel` ∈ {0,1}，為 1 時 `A` 是 `(D,N)`、為 0 時是 `(N,)` 或 `(1,N)`；`B`、`C` 是 `(B,L,N)`（C 已由呼叫端 broadcast）；`D` 為空或 `(D,)`；N 為 [1,32] 的 2 的冪。違反時丟 C++ error，不會變成 illegal access 或錯誤的 pointer 解讀；`CUDAGuard` 固定在 `u` 的 device。

第 2–5 項的規則由 `tests/unit/test_headline_head_torchscript_export.py` 在 CI 檢查（包括 allowlist 的每一行仍逐字存在於 `mamba_head.py`，以及 content hash 只忽略那兩類 entry）；有 CUDA 與已 build 的 library 時，同一個測試另外檢查 native operator 與 Python custom op 在三組形狀／參數上逐位元相同，以及第 7 項：6 種合法形式（per-channel、shared 1-D／2-D、無 D、half、non-contiguous）通過，27 種 contract 違反都以 operator 自己的 error 拒絕、且之後 CUDA context 仍正常。跨 GPU（同型但不同 device index）的情形在單 GPU 機器上無法測，由同一個 device 相等檢查涵蓋。

輸出（gitignored 的 `models/yolo/`）：

| 檔案 | 身分 |
|:--|:--|
| `mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt` | **可攜身分 = `content_sha256`**；同一組輸入重新 trace 時 content 相同 |
| `mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json` | `saccade.head_artifact_lineage_torchscript/v1`：preset／inventory、ckpt、`mamba_args`、head 載入描述、artifact 範圍、TorchScript 檔案與 content hash、native scan 呼叫數、TracerWarning 位置、operator library（hash、`DT_NEEDED`）、runtime requirements、結構檢查、環境、backbone engine hash（只供 B5 驗證） |
| `build/libsaccade_scan_torchop.so` | 這台機器的 build（綁 CUDA／torch 版本），sha256 只識別這次 build |

## 3. 這台機器上的紀錄

manifest 由 clean tree 產生（tool commit `37d4d21d`，`git_dirty=false`；operator 已含 §2 第 7 項的 contract 檢查）。

| 項目 | 值 |
|:--|:--|
| checkpoint | `runs/mamba_gt_v14replica_t3_t1/best.ckpt`，sha256 `c161c88e…0876`（== inventory `s.t3t1_phase_b`），epoch 15 |
| head 載入 | loader probe：missing 0、unexpected 0；`upsample_loaded=true`；in_channels `(128, 256, 512)`；temporal blocks 存在但 bypass |
| TorchScript | 檔案 sha256 `1663ec97022879e8078fb5b9dc3b65f4d2b76e4f8df6d5848300c2b3c999d879`（45,656,246 bytes）；**content_sha256 `f6a540edfecd6e421c533c5c2280fc26351b887006c1384bb61a567ce2f39487`**；native scan 呼叫 3 次（P3／P4／P5 各一） |
| TracerWarning | 只有 allowlist 的三個位置（`mamba_head.py:30`、`:254`、`:258`） |
| operator library | `build/libsaccade_scan_torchop.so` sha256 `cfea782f…aa43`；`DT_NEEDED` = libtorch、libc10、libnvrtc、libc10_cuda、libcudart、libtorch_cpu、libtorch_cuda、libstdc++、libgcc_s、libc |
| 環境 | torch 2.11.0+cu130；cuDNN 9.19.0；RTX 5070 Ti Laptop（SM 12.0） |
| 可重建 | `--check`：重新 trace 的 content_sha256 與 manifest 相同、on-disk 檔案未被改動、結構檢查逐位元相同 → `OK` |
| 結構檢查 | 在 runtime requirements 下載入存檔的 artifact，對 **合成** `torch.randn` 特徵（seed 0／1／2，不是 MOT17 frame）與同 process 的 eager head（Python scan op）比較：6 個輸出 × 3 個 seed **全部逐位元相同**（max abs diff 0.0） |
| backbone engine | sha256 `2ef3d4d4…9ce4`（只記錄，供 B5 驗證） |

**不讀 Python 的載入（一次性確認，未 commit 為工具）**：用 scratch 的 C++ 程式（只連 `libtorch`／`libtorch_cpu`／`libtorch_cuda`／`libc10`，`ldd` closure 無 `libpython*`／`libtorch_python*`）`dlopen` operator library、`torch::jit::setGraphExecutorOptimize(false)`、`torch::jit::load` 這份 artifact，以零輸入執行，6 個輸出形狀正確。這說明 artifact 在 G2 條件下可以載入與執行；它不是數值證據，正式的 native loader 屬於後續 PR（PR-8 的對應物）。

## 4. PR-2L 之前已經看過的東西（不是 parity 證據）

- §3 的結構檢查：只用合成特徵，比較對象是 **eager** head（`E`），不是 oracle 的 compiled head（`C`）。r2 已記錄 E 與 C 不是 bit-exact（`R_E` 的 txt 與 `R_C` 不同，metric 在容差內），所以本 artifact 對 oracle 的關係**預期**與 `R_E` 同類，但這需要 parity 宣告與正式 run 才能成立。
- 沒有對任何 MOT17 frame 執行本 artifact。

## 5. 已知限制

- `torch.jit.trace`／`torch.jit.script` 在 torch 2.11 已標示 deprecated；本 artifact 綁定目前的 torch 版本（manifest 記錄）。版本升級時需重新匯出並重新檢查；長期是否改用其他 LibTorch 形式是 owner 決策，不在本 PR。
- artifact 是靜態形狀；operator library 是 per-machine build；跨機器分發與 bundling 是 Phase C／owner 決策。
- backbone engine 的 provenance 與 PR-1 相同，仍是 inventory 記錄的狀態；本 PR 只記錄 hash。
- LibTorch C++ 預設的 cuDNN／TF32 設定與 oracle 相同，但 consumer 仍須主動設定 graph executor optimize 關（C++：`torch::jit::setGraphExecutorOptimize(false)`），否則 profiling executor 可能融合 kernel、改變數值。

## 6. 下一步（不在本 PR）

- **PR-2L parity 預宣告**：先宣告、後量測。必須先決定 harness 如何把這個 artifact 放進 `_whole_graph_fn` 的 head 位置：(a) runner 端替換（與 localization runner 同樣的注入方式，不改 harness），或 (b) 新增 harness 旗標（改 `src/saccade/` ⇒ 觸發 runtime-identity attestation）。容差與 terminal 結構可沿用 PR-2 的 V／L 結構，由宣告決定。
- PR-3 以後依 PR-2L 的 terminal 恢復。

## 7. 重現

```bash
cmake --build build --target saccade_scan_torchop
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head_torchscript.py            # 產生到 *_candidate stem（#536：staging 檢查後才發布；frozen stem 另需 --replace-frozen-stem）
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/model/export_headline_mamba_head_torchscript.py --check    # 重新 trace 並比對紀錄
```
