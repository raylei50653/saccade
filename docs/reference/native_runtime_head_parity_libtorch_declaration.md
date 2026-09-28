# Native runtime head parity (LibTorch) — 預宣告（#465 Phase B PR-2L／U1 redesign）

> 狀態：**預宣告，於 PR-2L 的第一個量測 run 之前凍結。** **declaration 凍結點** = 本文所在 PR 以 merge commit（不是 squash）併入 `main`；review 期間的修訂列在 §11，凍結之後只能以 §11 的 append-only amendment 修訂，不得 inline 編輯。**執行凍結點**見 §2 V1（runner PR 的 merge commit＋annotated tag）。
> 這是**新的**宣告，不是 [PR-2](native_runtime_head_parity_declaration.md)（blob `0941a010`）或 [PR-2R](native_runtime_head_parity_tf32_off_declaration.md)（blob `2f48ddfd`）的 amendment。兩者的 terminal `HEAD_PARITY_OUT_OF_TOLERANCE` 永久保留，描述的是 TRT 形式。
> 權威 seal bar：[experiment contract §20.8](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（引用，不複述）。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B1；受測 artifact：[native_runtime_head_artifact_libtorch.md](native_runtime_head_artifact_libtorch.md)（PR-1L，#482）；路線依據：[failure-localization r2 結果](native_runtime_head_failure_localization_r2_result.md) = `EAGER_NUMERICS_WITHIN`（該 terminal 不接受任何形式）。

---

## 0. 這個 gate 回答什麼

> PR-1L 匯出的 LibTorch head（TorchScript＋`saccade_native::selective_scan_fwd`，graph executor optimize 關，head-only），放進 headline 設定下 oracle head 的位置，能不能取代 oracle 的 PyTorch head？

與 PR-2 的關係：oracle、reference arm、資料、L1／L2 的量、門檻、容差政策、terminal 集合與判定順序**全部沿用 PR-2 宣告**（在看到任何 MOT 結果之前凍結，本文不重新校準）。實質差異只有兩項：受測 head 換成 L（§3），以及 L2 的 L 不是由 harness 旗標載入，而是由 runner 在 harness **既有的 head 替換槽位**注入（§5.1）。

| | 狀態 |
|:--|:--|
| L 與 oracle head 在**同一組 backbone 特徵**上的 tensor 差異 | **回答**（L1） |
| 只替換 head 呼叫後，7-seq MOT 輸出與 metric 的差異 | **回答**（L2） |
| 替換是否可接受 | 由 §6 的 terminal 決定；非 bit-exact 時**交 owner 判定**（boundary §5 B1） |
| L 是否在 G2（不讀 Python）的 native loader 下得到同樣輸出 | **不回答**：本 study 的 L 由 Python harness 內的 `torch.jit.load` 執行；native loader 屬 terminal 3／4 之後的 PR |
| L 在 CUDA graph capture 下能否執行 | 只作為執行前提（capture 失敗 ⇒ `UNRESOLVED`，§6），不另作 claim |
| FPS／latency、benchmark claim | 不回答、不改 |
| TRT 形式、S2 artifact、其他 LibTorch 形式（AOTInductor、script、optimize 開） | 不回答（§20.8 item 4） |
| PR-2／PR-2R 出界的原因 | 不回答 |

target decision layer = `none (cross-layer substrate work)`；κ 見 §4、§5。

## 1. 宣告前已看過的資料

全部是 **archived reference**，**不參與**任何門檻或容差的設定（§4、§5 的數值是 PR-2 宣告在 2026-09-27、看到任何 MOT 結果之前凍結的政策，本文原樣引用）：

- PR-2 宣告 §1 與 PR-2R 宣告 §1 列出的全部內容；PR-2 與 PR-2R 正式 run 的全部結果（L1、L2、per-sequence、`A_N` 與 `A_C` 在 IDF1／MOTA／IDs 上相同）。
- failure-localization r1 的 `UNRESOLVED` 紀錄與 r2 正式 run 的全部結果，包括 `R_E`（eager head 數值＋compiled S2，在 replay 系統內）對 `R_C`：IDF1 0、IDs 0、HOTA +5e-4，以及 r2 結果文件 §6 的 row→anchor 一致性比例。
- PR-1L 的結構檢查：**合成** `torch.randn` 特徵（seed 0／1／2）上，L 與同 process 的 eager head 6 個輸出逐位元相同；以及 scratch C++ loader 的零輸入執行。**沒有**在任何 MOT17 frame 上執行過 L。
- PR-2R packet `results/native_head_parity_465_tf32_off/20260927T151100Z/` 中 `A_C_1` 的 7 個 txt 的 sha256（§5 V4 使用）。只讀 hash；這些檔案與 PR-2 `A_C`、r2 `R_C` byte-identical 已是公開紀錄。

**r2 的 `R_E` 在容差內不是 `A_L` 的預測**：(1) `R_E` 的 head 在 replay detector 內、whole-graph CUDA graph 之外執行，`A_L` 的 L 在 harness 的 captured whole graph 之內；(2) `R_E` 的 scan 是 Python custom op `saccade::selective_scan_fwd`，L 是 native op `saccade_native::selective_scan_fwd`（同一個 launcher，但這是 PR-1L 的程式結構，不是在真實 frame 上觀察到的數值關係）。

本文撰寫時**沒有**執行任何新的 frame、MOT 或 instrumentation 量測。

## 2. 凍結輸入與執行凍結點（validity gate V1）

量測開始前，runner 逐項比對，任何一項不符 ⇒ `UNRESOLVED`。識別 study 的輸入全部 hard-code 在 runner，**不是** CLI 選項；正式 invocation 不帶任何 flag。

| 項目 | 凍結值 |
|:--|:--|
| TorchScript artifact | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt`；**檔案 sha256 `1663ec97022879e8078fb5b9dc3b65f4d2b76e4f8df6d5848300c2b3c999d879`**（凍結這一份檔案，不只凍結可重建的內容）且 `content_sha256` = `f6a540edfecd6e421c533c5c2280fc26351b887006c1384bb61a567ce2f39487` |
| artifact lineage manifest | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json`，**檔案 sha256 `677bc82320657a9ae289e78699c7a6f86e890fc629d1bb8189c08d35b57a277a`**；內容：`torchscript.sha256`／`content_sha256` == 上列、`op_library.sha256` == 下列、`runtime_requirements` == `{graph_executor_optimize: false, cudnn_benchmark: false, cudnn_allow_tf32: true, matmul_allow_tf32: false}`、`tool.git_dirty == false` |
| 可重建 | `export_headline_mamba_head_torchscript.py --check` 回 `OK`（重新 trace 的 `content_sha256` 與 manifest 相同） |
| operator library | `build/libsaccade_scan_torchop.so`，**sha256 `cfea782f320f641a5ca7752dfc54467909afcca96b6d6d9f884a40cbbe81aa43`**（per-machine build，凍結的是 manifest 綁定的這一份）；`DT_NEEDED` 不含 `libpython*`／`libtorch_python*`；HEAD 的 `src/tracking/mamba_scan_torchop.cpp` blob == `6b19e9a359fd04f344b92e7d2c596fea4fcbb505`（== manifest `tool.git_commit` `37d4d21d` 上的 blob，即 library 的來源） |
| traced graph | 載入後的 inlined graph：`saccade_native::selective_scan_fwd` 恰 3 次、無 `saccade::selective_scan_fwd`、無 `prim::PythonOp` |
| checkpoint | `runs/mamba_gt_v14replica_t3_t1/best.ckpt`，sha256 `c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876` |
| backbone engine | `models/yolo/yolo26s_backbone_640_best.engine`，sha256 `2ef3d4d40dfb670982cbbb98e6ed7d07e5b1a590cfa126d5ccf7342ee1579ce4`（== manifest `companions.backbone_engine.sha256`） |
| preset | `configs/presets/mamba_whole_graph.yaml`，sha256 `093b66ed124063f035ae9cf2a76e4f5426743cd819fb66e3e54994c97ea42cd1`（== manifest `preset.sha256`）；preset 不設定 head engine |
| 環境 | torch `2.11.0+cu130`、cuDNN `91900`、GPU `NVIDIA GeForce RTX 5070 Ti Laptop GPU`（SM 12.0）、host `DESKTOP-0FLA6SQ`，皆 == manifest `environment` |
| NVIDIA driver 與 CUDA runtime | L 的 custom CUDA op 在 CUDA graph capture 內執行，driver 是執行 substrate 的一部分，以 exact 比對凍結：NVIDIA driver `nvidia-smi --query-gpu=driver_version` == `616.92`（字串相等）；CUDA driver API `cuDriverGetVersion` == `13040`；CUDA runtime `cudaRuntimeGetVersion` == `13000`；process 內映射的 `libcudart.so.13` 恰一份，realpath == `.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13`、sha256 `96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0`（operator library 的 RUNPATH 解析到的 `build/cuda_devlink/libcudart.so.13` 是指向同一檔案的 symlink）。runner 在 V1 讀一次；L1 process 與每個 L2 child（`A_L` 在載入 operator library 之後）在 sidecar 中各讀一次，**全部**必須 exact 等於上列。user-mode driver `libcuda.so.1` 的 realpath 與 sha256 只記錄（driver version 已涵蓋） |
| env hatch | caller 不得設定任何 `SACCADE_*`；每個 arm 記錄 `resolved_env_overrides()`，三個 arm 必須相同 |
| 資料 | `datasets/MOT17/train`，序列固定為 `MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP`（共 5316 frames），依此順序 |
| 程式碼 | clean tree；`scripts/eval/mot17.py`、`src/saccade/**` 不因本 study 修改 |
| declaration blob | runner 內釘的本文凍結 blob == HEAD 與 worktree 上本文的 blob |
| **執行凍結點** | PR-2L runner PR 的 merge commit，由 annotated tag **`freeze/465-pr2l-libtorch-parity`** 標示。runner 檢查：HEAD 是 2-parent commit、在 `origin/main` 的 first-parent 鏈上；`git cat-file -t refs/tags/freeze/465-pr2l-libtorch-parity` = `tag`；本地 `…^{commit}` = HEAD；`git ls-remote origin refs/tags/freeze/465-pr2l-libtorch-parity^{}` = HEAD；比較一律用 commit SHA。本文所在 PR 的 merge commit 不含 runner，所以不是執行凍結點 |
| runtime identity | packet 記錄 HEAD 的 `docs/reference/runtime_identity.generated.json` blob 與其 `coordinate`；正式 run 只在執行凍結點的 CI（含 `runtime_identity` workflow）全綠之後執行，由 freeze record 記錄。本 study **不**把 L 或 operator library 納入 runtime identity，也不 republish |
| 資源 | 全部量測在同一個 `tools/resctl.py run machine-bench` 租約內執行，runner 是租約 owner 的直接 child，開始與結束時為同一份租約 |

## 3. 三種 head 實作

| 代號 | 實作 | 角色 |
|:--|:--|:--|
| **C** | PyTorch head，`set_head_compile(True)`＋`set_block_compile(True)`（oracle 預設） | **主 oracle**（headline claim 由它產生） |
| **E** | PyTorch head，eager（不 compile），scan = Python custom op | 次 oracle／reference |
| **L** | PR-1L TorchScript artifact（§2），`torch.jit.load(..., map_location="cuda")`，graph executor optimize 關，scan = native op | 受測 |

**L 的 runtime requirements**（manifest `runtime_requirements`）：graph executor optimize = false、`torch.backends.cudnn.benchmark` = false、`torch.backends.cudnn.allow_tf32` = true、`torch.backends.cuda.matmul.allow_tf32` = false。runner 在載入 L 之前設定 optimize = false（其餘三項是 harness 的 torch 預設值，runner 不改寫，只檢查）；**每一次** L 被呼叫時讀回四項，任一不等於上列 ⇒ fail-closed（`UNRESOLVED`）。

graph executor optimize 是 process-global 的設定。`src/saccade/` 中另外只有兩個 TorchScript 函數：`mamba_head.py` 的 scan fallback `_selective_scan_jit` 與 `_selective_scan_legacy_n1_jit`，推論路徑只在 CUDA 輸入但 extension 無法載入（`ImportError`）或輸入不在 CUDA 時才會用到。runner 在 L1 process 與 `A_L` child 中把這兩個 module attribute 換成被呼叫即 raise 的 guard（V2(c)、V5），所以 optimize 設定只作用在 L 上；`A_C`／`A_N` 不安裝 guard（它們不改 optimize 設定）。

## 4. L1 — head tensor（gross-error screen）

**用途**：抓「算的不是同一個函數」（權重錯位、trace 烤進錯的分支、native op 讀錯 layout）。它**不是**接受準則，接受與否在 L2。

**同輸入條件**：在同一個 process 內，每張 frame：GPU `decode_jpeg` → `float()/255` → `F.interpolate(640×640, bilinear, align_corners=False)` → 同一個 TRT backbone `infer` → **同一組 `p3/p4/p5` tensor** 依 C、E、L 的固定順序餵給三個 head。C 與 E 是兩個獨立的 head instance（同一份 state dict，由同一個 `build_mamba_gated_detector` 建構路徑產生），避免 compile 切換。三者都在 CUDA graph 之外執行；graph 內的組態由 L2 涵蓋。

**量**（每個 pair ∈ {(L,C), (L,E), (E,C)}，所有 frame 與 anchor 取最大值）與 PR-2 §4 完全相同：

- `score_maxabs`：`sigmoid(cls)` 逐元素 |Δ| 的最大值（全部 80 類、全部 anchor）；
- `box_maxabs_px`：以 `_postprocess_mamba_fixed_eager` 相同的 anchor／stride 解碼成 xyxy，乘 `(W_orig/640, H_orig/640)` 換到原圖座標後，逐座標 |Δ| 的最大值；計入的 anchor 是該 pair 兩邊的聯集，對 (L,C) 即 `(score_C ≥ 0.05) | (score_L ≥ 0.05)`（score = 80 類 sigmoid 的最大值）。0.05 = headline 的 `base_score_floor`。
- 跨越 0.05 本身不是 gross error；它的後果交給 L2。

**κ_L1** =（空間：5316 frames × 全部 anchor；關係：(L,C) 的上述兩個最大值；規則：`score_maxabs ≤ 0.05` **且** `box_maxabs_px ≤ 4.0` ⇒ `L1_PASS`，否則 `L1_GROSS_ERROR`）。
(L,E)、(E,C) 只報告，不參與判定；另報各 pair 的 per-sequence max 與 99.9 percentile，以及 **(L,E) 6 個輸出逐位元相同的 frame 數**（PR-1L 在合成特徵上觀察到逐位元相同；真實 frame 上是否仍成立只報告，不判定）。

**L1 validity（V2）**：
- (a) 每個序列的前 20 frames，C、E、L 各重跑一次，輸出必須與第一次逐位元相同；
- (b) 每張 frame，三個 head 執行完之後 `p3/p4/p5` 必須與餵給 C 之前的 clone 逐位元相同（沒有任何 head 改寫共同輸入）。
- (c) 整個 L1 process 中，§3 的兩個 JIT scan fallback guard 的呼叫次數 = 0（C、E 都經 CUDA custom op `saccade::selective_scan_fwd`）。

任一不成立 ⇒ `UNRESOLVED`。

## 5. L2 — 7-seq MOT

### 5.1 Runner-side injection（替換哪一層）

harness 的 whole-graph detect（`MambaGatedDetector._whole_graph_fn` 與 `_whole_graph_fn_preprocessed`）已經有一個 head 替換槽位：`self._trt_head` 不為 `None` 時呼叫 `self._trt_head.infer_graph(p3, p4, p5)`，否則呼叫 `self.mamba_head._forward_eager([p3, p4, p5], return_embeddings=False)`。PR-2／PR-2R 的 `A_T` 就是經 `--mamba-head-engine` 把 TRT head 放進這個槽位。`_trt_head` 在 `src/saccade/` 與 `scripts/eval/` 中只有這兩處讀取。

**被替換的層恰為**：`p3/p4/p5 → (cls_preds[3], reg_preds[3])` 這一次呼叫。它之前的 frame decode、640 resize、TRT backbone，以及之後的 compiled `_postprocess_mamba_fixed`（S2）、座標縮放、NMS、tracker、輸出，全部是未修改的 harness 程式，與 `A_C` 相同（`A_C` 的 postprocess 也是 compiled）。

**入口**：三個 arm 都由 runner 以同一個 child 入口啟動：child 以 `runpy.run_path("scripts/eval/mot17.py", run_name="__main__")` 執行**未修改**的 `mot17.py`（相對路徑，讓 harness manifest 的 cmdline 與直接執行相同），`sys.argv` 為該 arm 的凍結參數。`A_C`／`A_N` 的 child **不取代任何函數或 attribute、不載入 operator library 或 artifact**（只註冊一個寫 sidecar 的 `atexit` 觀察）；只有 `A_L` 的 child 在 `runpy` 之前依序：

1. 驗證並以 `torch.ops.load_library` 載入 §2 的 operator library；
2. 設定 graph executor optimize = false；
3. `torch.jit.load` §2 的 artifact，重新驗證檔案 sha256、`content_sha256` 與 traced graph 條件，`.eval()`；
4. 以 wrapper 取代 module attribute `mamba_gated_detector.build_mamba_gated_detector`（`mot17.py` 在呼叫當下才 import 這個名稱）。wrapper 以**原封不動的參數**呼叫原函數，然後斷言：`trt_head_engine == ""`、回傳物件的 `_trt_head is None`、`use_whole_graph is True`、`use_detail_fusion is False`；接著設定 `detector._trt_head = LibTorchHeadAdapter(L)`，並把 `detector.mamba_head` 的 `forward` 與 `_forward_eager` 都換成被呼叫即 raise 的 guard（安裝之後 PyTorch head 不得再經任何路徑執行，包括非 whole-graph 的 `_detect_from_feats`）。wrapper 必須恰被呼叫一次；
5. `LibTorchHeadAdapter.infer_graph(p3, p4, p5)`（`infer` 同）：讀回 §3 的四項 runtime requirement；斷言輸入為 CUDA、float32、shape `(1,128,80,80)`／`(1,256,40,40)`／`(1,512,20,20)`；在 current stream 上呼叫 `L(p3, p4, p5)`，回傳 `([cls_p3, cls_p4, cls_p5], [reg_p3, reg_p4, reg_p5])`；不做任何 copy、dtype 轉換、`.contiguous()` 或其他運算；斷言 6 個輸出為 CUDA float32、shape 與 `_forward_eager` 相同。

`mot17.py` 之後照常呼叫 `set_postprocess_compile(True)`、`set_head_compile(True)`、`set_block_compile(True)`（與 `A_T` 相同：compiled PyTorch head 存在但不被 whole graph 呼叫）。whole graph 的 warmup 與每個輸入 shape 的 capture 會經過 adapter，之後的 replay 執行 captured kernels。

**L2 的同輸入條件**：`A_L` 與 `A_C` 是兩個 process，各自計算 backbone 特徵；兩者使用同一份 frame、同一個 decode／resize 程式與同一個 backbone engine。**L2 不直接觀察兩個 arm 的 `p3/p4/p5` 相同**（replay 在 captured graph 內，adapter 看不到）；同輸入的 head 比較由 L1 負責，L2 的前提由 V3（每 arm 重複 byte-identical）與 V4（oracle 錨定）支撐。

### 5.2 Arms 與執行

皆為 `scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer --sequences <§2> --output <新目錄>`（經 §5.1 的 child 入口）：

| arm | 額外參數 | 注入 | 意義 |
|:--|:--|:--|:--|
| `A_C` | （無） | 無 | oracle |
| `A_L` | （無） | §5.1 | 只把 head 呼叫換成 L；其餘同 oracle |
| `A_N` | `--no-compile` | 無 | reference：與 PR-2 §5 相同的 **composite nuisance reference**（同時關掉 head／block compile 與 postprocess compile），`|Δ_N|` 不是 head-only 的估計，只是容差的參考尺度 |

**執行順序**：`A_C#1, A_L#1, A_N#1, A_C#2, A_L#2, A_N#2`，同一 session、同一租約。任何一個 run 失敗（exit ≠ 0、缺 txt、缺 manifest、V5 不成立）即停止其餘 run，packet 仍寫出，terminal = `UNRESOLVED`。

### 5.3 Validity

| gate | 條件 | 不成立 ⇒ |
|:--|:--|:--|
| V3 確定性 | 每個 arm 的兩次 run，7 個 `MOT17-*-SDP.txt` 逐位元相同 | `UNRESOLVED` |
| V4 oracle 錨定 | `A_C#1` 的 7 個 txt sha256 等於 PR-2R packet `A_C_1`：02 `d426ca1b61ae1441b94cc3269a6fee90f01ca7ec3a649755f170c767b7515328`、04 `cb55746fdc059fae12a6efcbce5cfa00b40fdb4f23b8766ee021a8b5b1602d7b`、05 `ea46b483046879ea28bae14f25f7ec251fea6359e6d850592504ee09f3f4d8dc`、09 `da0a74836843293da0e45923905bf25f31d68145b71771a1da043cdf1e576df4`、10 `587f2f05bfe9caacf57fba2cd5a6828b1b548ddcaef3d155c96055844bf92705`、11 `38dd77309e98a88b3abf827d18b087b6f025bef176b47c280a4961f3013247b0`、13 `4c93f75e6e162e7ab6007a746fc7da97250d54b01d8bd63500a7b07a5af73137`（確認 child 入口是透明的，且本 study 的 oracle 就是 PR-2／PR-2R／r2 的同一個 oracle） | `UNRESOLVED` |
| V5 注入正確 | `A_L` 每個 run 的 child sidecar：§5.1 步驟 1–4 全部成立且 wrapper 恰呼叫一次；adapter 被呼叫 ≥ 1 次，每次的 runtime requirement 讀回與輸入／輸出斷言都成立；`forward`／`_forward_eager` guard 與 §3 的兩個 JIT scan fallback guard 的呼叫次數皆 = 0；`A_C`／`A_N` 每個 run 的 sidecar：未取代任何函數或 attribute、process 結束前的 `/proc/self/maps` 不含 operator library（sidecar 只由 `atexit` 觀察寫出，不改變執行）。每個 child（三個 arm）與 L1 process 的 §2 driver／CUDA runtime 讀回 exact 等於凍結值。三個 arm 的 harness `run_manifest.json` 的 `cmdline` 等於該 arm 的凍結參數，且不含 `--mamba-head-engine`、`--mamba-trt` | `UNRESOLVED` |

（V3 兩次相同不證明確定性；它只是讓 arm 之間的差異可歸因於 head 的必要條件。）

### 5.4 Metric 與判定（與 PR-2 §5 相同）

**Metric**：以 `saccade.perception.eval.metrics` 對每個 arm 的 #1 輸出計算 7-seq combined，**不經四捨五入**：IDF1、MOTA 由 `_evaluate_single_sequence` 的 counts 直接算，HOTA 用 `_calculate_hota` 的原始 float，以百分點表示；IDs = `num_switches` 總和。

1. 若 `A_L` 與 `A_C` 的 7 個 txt 全部逐位元相同 ⇒ `L2_EXACT`。
2. 否則，對 m ∈ {IDF1, HOTA, MOTA, IDs}：Δ_L,m = m(A_L) − m(A_C)，Δ_N,m = m(A_N) − m(A_C)；容差 b_m = min(max(|Δ_N,m|, floor_m), cap_m)；floor = 0.20 pt（IDF1／HOTA／MOTA）、5（IDs）；cap = 1.00 pt、30（IDs）。
   **κ_L2** =（空間：7-seq combined 的四個 metric；關係：|Δ_L,m| 對 b_m；規則：四個都 `|Δ_L,m| ≤ b_m` ⇒ `L2_WITHIN`，任一超過 ⇒ `L2_OUT`）。雙向：變好與變差同樣算偏差。

floor、cap 的理由與限制見 PR-2 §5：**cap 不是「允許退化 1 pt／30 IDs」**，它是非 exact 結果最多還能進入 owner 判定（terminal 4）的範圍；`WITHIN_TOLERANCE` 本身不自動接受。

**只報告、不判定**：DetA、AssA、FP、FN、per-sequence 的四個 metric、每個序列第一個分歧的 frame、`A_N` 與 `A_C` 是否 byte-identical、`A_N#1` 是否與 PR-2R `A_N_1` byte-identical、`A_L` 各序列 txt 是否與 r2 `R_E` byte-identical。

## 6. Terminal（窮盡，依序判定）

| # | terminal | 條件 | 主線轉移 |
|:--|:--|:--|:--|
| 1 | `UNRESOLVED` | V1–V5 任一不成立；或任何 runner／harness／artifact 載入錯誤、CUDA graph capture 失敗、例外、缺 packet（execution-invalid 一律歸此，fail-closed） | 無。只關閉這一次量測；**不得原樣重跑**。重跑需要 owner 同意的 append-only amendment（§11）說明原因與修正，且該 amendment **不得**修改 §4、§5 的門檻、容差政策、arms 或本表 |
| 2 | `HEAD_PARITY_GROSS_ERROR` | L1 = `L1_GROSS_ERROR` | PR-1L 形式被否決；#465 維持 `requires_runtime_redesign`。下一步 = **預宣告的 failure-localization study**（先宣告、後量測）；不回頭調整門檻或容差，不重跑 PR-2L |
| 3 | `HEAD_PARITY_EXACT` | L1 PASS 且 `L2_EXACT` | U1 關閉，無 named limit；PR-1L 成為 Phase B 的 head 形式。之後才做 shipping／native loader 整合與 runtime identity（各自的 PR）；native loader 須證明與 `A_L` byte-identical |
| 4 | `HEAD_PARITY_WITHIN_TOLERANCE` | L1 PASS 且 `L2_WITHIN` | **待 owner 判定**：<br>• `ACCEPT` ⇒ U1 關閉，named limit：native shipping 的 MOT 輸出不與 headline byte-identical。PR-1L 成為 Phase B 的 head 形式；之後才做 shipping／native loader 整合與 runtime identity。PR-9／PR-10 的 parity oracle 改為 `A_L` 組態（本文 §5.1 的注入；日後由 native loader 或 harness 旗標取代注入時，須先證明與 `A_L` byte-identical），使其餘單元仍能以 byte-identity 驗收；shipping 產物的任何數字只能引用 native／`A_L` 的量測，不得沿用 headline 數字<br>• `REJECT` ⇒ 同 terminal 5 |
| 5 | `HEAD_PARITY_OUT_OF_TOLERANCE` | L1 PASS 且 `L2_OUT` | 同 terminal 2 |

任何 terminal 都不撤銷 PR-2、PR-2R、r1、r2 的結果。

## 7. Packet

runner 把以下內容寫到非 scratch 的 timestamped 目錄 `results/native_head_parity_465_libtorch/<UTC>/`（以 `open_run` claim）並附 manifest：本文與 runner 的 git blob sha、執行凍結點（commit SHA、tag object）、V1 每一項的實測值、runtime identity blob 與 coordinate、L1 的全部 pair 統計（CSV）與 V2 結果、6 個 run 的完整輸出目錄、stdout 與 child sidecar（V5）、metric 計算的 counts 與未四捨五入值、判定過程與 terminal。PR-2L 的結果文件引用這個 packet，只寫 terminal 與支撐它的數字。

## 8. 本文刻意沒有做的事

- 沒有執行任何 PR-1L artifact 在真實 frame 上的量測；runner PR 也不在 MOT17 frame 上跑 smoke（只做不讀 MOT17 frame 的結構檢查與 unit test）。
- 沒有依 PR-2、PR-2R、r2 的任何數字設定或調整門檻、容差或 reference arm；沒有加入 TRT 或 `R_E` 的 arm。
- 沒有修改 `scripts/eval/mot17.py`、`src/saccade/**`、preset 或 artifact；沒有新增 harness 旗標（owner 決定採 runner-side injection）。
- 沒有預先決定 owner 在 terminal 4 的判定；沒有宣告 native loader、G2、shipping 或 runtime identity 的任何事。

## 9. 執行順序

1. 本文 review → 修訂（§11）→ 以 merge commit 併入 `main` = **declaration 凍結**；在 #465 貼出 freeze record（merge SHA、本文 blob）。
2. PR-2L runner 另開 PR（`scripts/eval/diagnostics/native_head_parity_libtorch.py`＋unit tests＋artifact producer registry entry）：hard-bind 本文 blob 與 §2 的輸入、輸出與 terminal；與 PR-2 runner 相同的判定邏輯（`tolerance`、`l1_verdict`、metric 計算、terminal 順序）以測試釘住。runner PR 只做 unit test 與不讀 MOT17 frame 的結構檢查。
3. runner PR 以 merge commit 併入 → 建立 annotated tag `freeze/465-pr2l-libtorch-parity` 並 push → 確認 CI 全綠 → 在 #465 貼出執行 freeze record（merge SHA、tag object、runner blob）。
4. 從該 SHA（detached checkout 可）在 `machine-bench` 租約下做 V1 dry check（不寫 packet）→ **一次**正式 run。
5. 依 §6 的 terminal 寫結果文件（另一個 PR）；不在結果文件中改寫本文。

## 10. Owner 決定

- 2026-09-28：PR-2L 採 **(a) runner-side injection**（harness 不改）；hard-bind artifact 檔案／content sha、operator library sha、ckpt／backbone／preset、torch／CUDA／host、optimize=false、cuDNN／TF32；L1 = C／E／L 共用 backbone 特徵；L2 = 未修改的 harness，只換 head 呼叫；容差 = PR-2 凍結政策，不重新校準；只有通過之後才做 shipping／native loader 整合與 runtime identity。

## 11. Review 修訂與 amendments

**凍結前的 review 修訂**（#485 owner review，任何量測之前，沒有看過新資料）：

- R1：§2 原本 pin 了 torch、cuDNN、GPU／SM、host 與 operator library sha，但沒有 pin NVIDIA driver 與 CUDA driver／runtime。L 的 custom CUDA op 在 CUDA graph capture 內執行，driver 是執行 substrate 的一部分。§2 新增「NVIDIA driver 與 CUDA runtime」一列：driver `616.92`、`cuDriverGetVersion` `13040`、`cudaRuntimeGetVersion` `13000`、process 內唯一的 `libcudart.so.13`（realpath 與 sha256），在 V1 與每個 process 的 sidecar 中 exact 比對。值由宣告撰寫時在同一台機器上讀取（不涉及任何 frame 或 MOT 資料）。
- 維持不變：arms、門檻、容差政策、validity 的其餘條件、terminal。

**凍結後的 amendments**（append-only）：（無）
