# Native runtime shipping boundary（#465 Phase A.5）

> 狀態：Phase A.5 範圍凍結；**不改** runtime 行為、preset、threshold、weights、benchmark claim、packaging／release CI，也**不決定** CUDA／TensorRT bundling policy。
> 基準：[native_runtime_packaging_audit.md](native_runtime_packaging_audit.md)（#466，source `7be2f51f`）。該文是 Phase A 的歷史調查，本文不改寫它、不重述它的證據，只引用其編號：§5 的 11 項 Python 擁有語義記為 **S1–S11**，§10 的後續單元記為 **U1–U6**。
> 本文只定義 Phase B 的**邊界與順序**；沒有實作任何單元、沒有執行 inference、沒有發任何數字。

本文回答：**Phase B 可以動哪些 Python、不可以動哪些 Python。** 目的是讓「shipping runtime 不含 Python」不會擴張成整個 repository 的 Python 重寫。

---

## 0. 兩個部署目標，分開計算

| 目標 | 意義 | 誰能滿足 | 是不是 Phase B 的驗收 |
|:--|:--|:--|:--|
| **G1** 使用者不需自行安裝 Python | 終端使用者的機器上不需要事先有 Python／venv | audit §7 形式 D（bundled CPython＋venv）即可，不必搬任何語義 | **否**。G1 是否另做過渡包屬 owner 決定，不在 Phase B |
| **G2** shipping runtime 本身 Python-free | 執行 shipping entrypoint 的 process 裡沒有 Python | 只有 audit §7 形式 A／B | **是**。Phase B 只追 G2 |

bundled Python 滿足 G1，**不**滿足 G2；任何「已經不用裝 Python 了」的敘述都不得被當成 G2 的進度。

G2 的操作定義（Phase B 最後一個 PR 的檢查項，見 §6 PR-12）：

1. shipping 產物的 `readelf -d` NEEDED 閉包裡沒有 `libpython*`、`libtorch_python.so`；
2. 執行時沒有 exec 任何 Python 直譯器，也沒有內嵌直譯器；
3. shipping 產物目錄裡沒有 `.py`／`.pyc`／`site-packages`；
4. 執行時不需要 C 編譯器（也就是沒有 Triton／`torch.compile` JIT）。

LibTorch C++（`libtorch*.so`，不含 `libtorch_python.so`）**不**違反 G2；要不要帶它由 U1 的 head 形式決定（audit §6）。

---

## 1. Scope 詞彙

每個項目只給**一個** primary scope：

| scope | 定義 | Phase B 對它的權限 |
|:--|:--|:--|
| `shipping_runtime` | 在 shipping entrypoint 執行時必須存在的語義或宿主工作 | 以 native 重新實作；Python 版保留為 parity oracle，**不刪、不改語義** |
| `developer_build_debug` | build、資產產生、測試、parity 工具、診斷、profiling、本機驗證 | 可以繼續用 Python；Phase B 可以**新增**這一類 Python 工具（例如 dump／比對腳本） |
| `eval_research` | metrics、benchmark、training、研究旗標、實驗紀錄、eval harness 本身 | 完全不屬於 #465 migration scope；Phase B 不得為了 G2 修改它 |
| `shared_boundary` | 同一份 Python 元件同時服務 shipping 語義與 dev／eval 責任 | 只抽出 shipping 需要的那一窄段（§5）；Python 端的 dev／eval 介面不動 |

---

## 2. 固定的 shipping entrypoint

| 軸 | 固定值 |
|:--|:--|
| 形式 | 新的 native C++ 執行檔（audit §9 暫名 `saccade_track`），audit §7 形式 B |
| **不是** entrypoint | `scripts/eval/mot17.py`（eval harness，保留）；`--cpp-threads` → `saccade_eval_ext`（拒絕 `private_continuation`，head lineage 未綁）；`saccade_node`（demo，沒接 tracker）；`main.py --mode perception`（`TRTYoloDetector`，非 Mamba）。它們的程式碼可以被重用，但都不是 Phase B 的驗收對象 |
| 語義參考（oracle） | 同一台機器上的 `mot17.py --preset mamba_whole_graph --detector SDP --double-buffer`：s backbone、`runs/mamba_gt_v14replica_t3_t1/best.ckpt`、GPU decode、預設 `torch.compile` |
| 支援的設定 | **只有**這一個 preset 解析後的值（U2 產出的 resolved config）。其他 preset、CLI 旋鈕、`SACCADE_*` env hatch 都不進 shipping |
| 輸入 | 一個 sequence 目錄：依檔名順序的 JPEG frames，加上 audit 參考路徑實際讀取的 sequence metadata（`seqinfo.ini` 的 `frameRate`、`imWidth`、`imHeight`） |
| 輸出 | 每個 sequence 一份 MOT txt，列格式與 oracle 相同（`:.2f`／`:.4f`，含 sequence-tail 插值） |
| 排程 | 目標是 oracle 的 double-buffer（detect(N+1) ∥ tracker(N)、event barrier）；serial 只是 PR-5／PR-9 的中間驗收組態 |
| 不在 entrypoint 內 | GT、metrics、MLflow、run manifest、visualize、profiling、跨 sequence 的 run-global ID、video／RTSP 輸入（會改變解碼語義，另開 scope） |

---

## 3. Scope matrix

### 3.1 U1–U6

| 單元 | primary scope | 理由 | shared_boundary 條目 |
|:--|:--|:--|:--|
| **U1** head native 形式＋parity gate | `shared_boundary` | 同一份 head 定義要繼續服務 training／eval（PyTorch），shipping 只消費匯出的 artifact | B1、B5 |
| **U2** resolved runtime config export | `shared_boundary` | 四層預設解析留在 Python；shipping 只讀扁平結果 | B2 |
| **U3a** native post-detector host（replay） | `shipping_runtime` | 新的 native 宿主；replay dump 工具本身屬 `developer_build_debug` | — |
| **U3b** native ingest＋detector | `shipping_runtime` | nvJPEG、normalize、TRT backbone、head、decode 在 shipping process 內 | — |
| **U4** native sequence tail | `shared_boundary` | `post_merge.py`／`GlobalTrackIdMapper` 仍服務 eval／研究修復；只抽 headline 設定下的那一段 | B3、B4 |
| **U5** native graph capture＋double-buffer | `shipping_runtime` | 排程、stream、graph 生命週期只在 shipping 宿主內有意義 | — |
| **U6** link surface 與可攜性 | `shipping_runtime` | 驗收對象是 shipping 產物（無 libpython、`$ORIGIN`、SM＋PTX、glibc baseline）；CMake target 拆分順帶影響 dev build，但驗收不看 dev build | — |
| Phase C／owner 決策列 | 不分類 | 不是工程單元：bundle 政策、engine 分發、平台清單。不在 Phase B | — |

### 3.2 S1–S11（audit §5 的 Python 擁有語義）

| # | 語義 | primary scope | Phase B 單元 |
|:--|:--|:--|:--|
| S1 | Mamba head 前向（eager＋預設 `torch.compile`） | `shared_boundary`（B1） | U1 |
| S2 | whole-detect 後段：640 stretch resize、anchor decode、sigmoid／class max、top-k、座標縮放 | `shipping_runtime` | U3b（可併入 U1 匯出的 graph，見 B1） |
| S3 | JPEG 解碼（nvJPEG via torchvision） | `shipping_runtime` | U3b |
| S4 | ingest normalize（`/255` → f32 CHW） | `shipping_runtime` | U3b |
| S5 | FP hard filter mask | `shipping_runtime` | U3a（`seq_runner.cpp:426` 的 twin 可重用，但要重驗 parity） |
| S6 | frame loop＋double-buffer 排程 | `shipping_runtime` | U3a（serial）→ U5（double-buffer） |
| S7 | GPU 資源所有權：allocator、stream／event、graph capture 與 recapture key | `shipping_runtime` | U3a（eager）→ U5（graph） |
| S8 | config→native 參數映射（`set_params` 等）＋四層預設解析 | `shared_boundary`（B2） | U2 |
| S9 | local→global track ID 映射＋MOT emit 格式 | `shared_boundary`（B3） | U4 |
| S10 | sequence-tail 插值（pandas） | `shared_boundary`（B4） | U4 |
| S11 | 模型載入＋lineage gate（pickle ckpt、`mamba_args`、SHA 比對） | `shared_boundary`（B5） | U1 |

### 3.3 audit §4 其餘 Python 依賴

| 依賴 | primary scope | Phase B 處理 |
|:--|:--|:--|
| CPython、`torch` eager op／allocator／stream | `shipping_runtime` | 由 S2–S7 的 native 實作取代；eval harness 照用 |
| `torch.cuda.graph`／`make_graphed_callables` | `shipping_runtime` | U5 以 native `cudaStreamBeginCapture` 取代 |
| `torch.compile`＋`triton` | `eval_research` | shipping 不帶；compile 輸出只當 U1 的 oracle 之一 |
| `tensorrt` Python bindings（`TRTYoloBackbone`） | `shipping_runtime` | U3b 改用既有 native `TRTEngine` |
| `eval/stages.py` NMS／private continuation 呼叫、`PyGraphedGMC`、`GPUByteTracker`／`GraphedTrackerUpdate` | `shipping_runtime` | U3a 直接呼叫同一批 native 物件；Python 類別保留給 eval |
| `nvidia.dali`、PyTorch NV12 路徑、`SACCADE_NV12_BUFFER` 的 `execve` | `eval_research` | fallback／實驗路徑，不搬 |
| `cv2`（Python）、`post_merge` 其餘功能、Cheb-GR／semantic relink／lifecycle merge／D0 capture | `eval_research` | headline 關閉，不搬 |
| `eval/metrics.py`、TrackEval、motmetrics、MLflow、`run_manifest.py` | `eval_research` | 不搬 |
| `saccade/paths.py`、`saccade_build.pth` | `developer_build_debug` | shipping 用 `$ORIGIN` 與套件內相對路徑，不讀這兩者 |
| `pybind11`、`nvidia-cuda-nvcc`／`nvvm`／`crt`／`cccl`、cmake | `developer_build_debug` | 保留 |
| `scripts/model/*`（engine／ONNX／TorchScript 產生） | `developer_build_debug` | 保留；U1 的匯出落在這裡 |

---

## 4. 明確允許保留的 Python

### 4.1 `developer_build_debug`（允許保留，Phase B 也可新增）

- native build：`native-build` extra（pybind11、nvcc wheels、cmake 驅動）、`saccade_build.pth`、`saccade/paths.py`（ADR 025 的 consumer-run 路徑不變）。
- 資產產生：`scripts/model/*`、U1 新增的 head 匯出與 lineage 紀錄、U2 的 resolved config exporter。
- parity／golden 工具：Phase B 新增的 detector-output dump、tensor／MOT txt 比對、golden fixture 產生器。
- 測試與 CI 檢查：pytest、contract tests、`scripts/pre_push.sh`、docs／scripts／tests 索引生成器。
- 診斷與 profiling：nsys／ncu 輔助腳本、`--profile-*`、`frozen_source_status.py` 等 gate 工具、`tools/resctl.py`。
- pybind extensions（`saccade_tracking_ext`、`saccade_perception_ext`、`saccade_eval_ext`）：eval harness 繼續用；U6 拆 CMake target 時只要求它們行為不變，不要求移除。

### 4.2 `eval_research`（排除於 #465 migration scope）

- eval harness 本身：`scripts/eval/mot17.py`、`evaluator.run_eval`／`run_eval_cpp`、`scripts/eval/config/*`、所有 preset 與 CLI 旋鈕。它是 oracle，Phase B 不得為了 G2 改它。
- training 與 dataset：`runs/*` ckpt 的產生、teacher cache、PP22／MOT 轉換。
- metrics 與紀錄：TrackEval、motmetrics、MLflow、run manifest、benchmark 表、campaign inventory／contract（#421）。
- 研究功能：Cheb-GR、semantic relink、sparse ReID、lifecycle merge、D0 capture、post_merge 的非插值修復、所有 `SACCADE_*` 實驗 hatch。
- 其他入口：`--cpp-threads`、`main.py` 工業串流、workbench。

---

## 5. `shared_boundary` 抽取提案

共通規則：抽出的是一份**資料契約**（artifact、JSON、文字格式），不是一份共用程式碼；Python 端繼續擁有它原本的 dev／eval 介面，native 端只實作 headline 設定下的那一段。

### B1 — Mamba head（S1；U1）

| 欄 | 內容 |
|:--|:--|
| current owner | `mamba_head.py` `MambaDetectionHead`＋`mamba_gated_detector.py` `_whole_graph_fn`；training 與 eval 共用 |
| shipping 責任 | 以 native 形式（TRT engine＋`libsaccade_scan_plugin.so`，或 LibTorch TorchScript）執行**同一份** `v14replica_t3_t1` 權重的前向 |
| dev／eval 責任 | 訓練、PyTorch eager／compile 前向、研究改架構；全部留在 Python |
| extraction boundary | 一份匯出的 head artifact（ONNX→engine 或 TorchScript）＋lineage 紀錄（source ckpt 路徑與 hash、`mamba_args`、匯出工具版本）。S2 的 resize／decode／top-k 可以選擇一起匯出進 artifact，或留給 U3b 以 native kernel 實作；選哪一種在 PR-1 寫明 |
| validation | detection tensor 對 eager **與** compile 兩個 oracle 各比一次（headline claim 是 compile 產生的，所以 compile 為主 oracle）；再比 7-seq MOT 輸出。容差事先宣告；非 bit-exact 時交 owner 決定能否接受。**失敗 ⇒ verdict 升級為 `requires_runtime_redesign`，Phase B 停在這裡** |

### B2 — 設定解析與參數映射（S8；U2）

| 欄 | 內容 |
|:--|:--|
| current owner | `mot17.py`／`scripts/eval/config/*`／`mot17_args.py:167 configure_runtime_env`／`eval/config.py`（四層解析）＋`tracker_gpu.py:587 GPUByteTracker.set_params` 等（映射到 native `set_*`） |
| shipping 責任 | 讀一份扁平 resolved config：native 物件參數直接送進 `set_*`，host／tail 參數由 native 宿主讀取；不做任何預設疊加、不讀 `SACCADE_*`。**shipping 讀到的每一個語義值都必須來自這份 JSON**，U3–U5 不得把 headline 值寫死在程式碼裡 |
| dev／eval 責任 | 四層解析、所有 preset、CLI、env hatch 維持原樣 |
| extraction boundary | `mamba_whole_graph.resolved.json` 是**完整的 shipping-runtime resolved schema**，不是現有 `set_*` 的鏡像。至少分成兩段：<br>• `native_params`：native 物件（`GPUByteTracker`、`PerceptionPipeline`、`GMC`）的 `set_*` 實際收到的值（不是 YAML 鍵），外加 native 端 `getenv` 預設的明確值（例如 `SACCADE_ENABLE_DDA`）；<br>• `host_params`：目前由 Python 宿主消費、沒有對應 `set_*` 的值。這包括 ingest／detector 後段（輸入尺寸、decode／top-k 參數）、stage 順序與 FP hard filter 參數、排程（double-buffer、barrier）、emit，以及 **sequence tail 的每一個條件式步驟**。tail 步驟以 `evaluator.py` 序列尾段為準：deferred alias remap、`filter_low_quality_tracklets`（`min_tracklet_len`／`min_tracklet_score`）、`interpolate_tracklets`（`interpolate_tracklets`／`interpolate_max_gap`／`interpolate_min_track_len`／`interpolate_min_h`）。每一步都要記錄啟用旗標與參數，**停用的步驟也要明列**。<br>另外附 source preset hash。欄位清單由 PR-3 從 oracle 程式碼列舉並寫進 schema，不在本文凍結。exporter 屬 `developer_build_debug` |
| validation | (a) `native_params`：Python 從 preset 解析後送進 native 的參數，與 JSON 逐欄相等；`host_params`：Python 宿主在 oracle 執行時實際使用的 `cfg` 值，與 JSON 逐欄相等；schema 覆蓋檢查要確認 oracle 宿主讀取的每個語義欄位都有對應鍵。(b) native loader 設定後回讀的值與 JSON 相等。若 exporter 要改 `src/saccade/**` 或 `scripts/eval/mot17.py`，會觸發 `decision_relevant` partition 的 runtime-identity attestation，PR 必須照該 gate 處理；優先把 exporter 放在 partition 外 |

### B3 — track ID 與 MOT emit（S9；U4）

| 欄 | 內容 |
|:--|:--|
| current owner | `eval/tracking.py` `GlobalTrackIdMapper`、`evaluator.py` emit 區段 |
| shipping 責任 | 單一 sequence 內：local ID 依首次出現順序映射為從 1 起算的輸出 ID；MOT 列格式 |
| dev／eval 責任 | 跨 sequence 的 run-global 唯一 ID、`dump_lines` 對照表 |
| extraction boundary | MOT 文字格式規格（欄位、`:.2f`／`:.4f`、列順序）＋「per-sequence 首次出現順序」規則。run-global 計數不進 shipping |
| validation | golden test：對 Python 以單一 sequence 執行的輸出 byte-identical；多 sequence 比對時先做首次出現順序重標再比 |

### B4 — sequence tail（S10；U4）

| 欄 | 內容 |
|:--|:--|
| current owner | `evaluator.py` 序列尾段，依序呼叫 `post_merge.py` 的 `apply_deferred_alias`（:310）、`filter_low_quality_tracklets`（:283）、`interpolate_tracklets`（:359，pandas），每一步都由 `cfg` 條件決定是否執行；同一模組也承載研究用的修復。audit S10 只點名插值，這裡把同段其他條件式步驟也納入，讓 shipping 完整覆蓋 oracle 的 tail |
| shipping 責任 | 依 B2 `host_params` 的 tail 段，照 oracle 的順序執行啟用中的 tail 步驟（插值的參數，例如 `interpolate_max_gap`／`interpolate_min_track_len`／`interpolate_min_h`，一律讀 JSON，不寫死），然後寫檔 |
| dev／eval 責任 | 其他插值參數、post_merge 的研究修復、pandas 依賴全部留在 Python |
| extraction boundary | 輸入＝B3 的 MOT 列，輸出＝插值後的 MOT 列；純 host 端函式，不碰 GPU |
| validation | 對 Python tail 的 golden fixture byte-identical：每個啟用步驟單獨比一次，整段再比一次。邊界 case 包括 gap 恰為 max_gap、長度恰為 min_len、單幀 track，以及 tail 旗標停用時的輸入直通 |

### B5 — 模型載入與 lineage gate（S11；U1）

| 欄 | 內容 |
|:--|:--|
| current owner | `mamba_gated_detector.py:726` `torch.load(weights_only=False)`、`mamba_args`、`:729-736` 的 `yolo26s.pt` SHA 比對 |
| shipping 責任 | 載入 B1 artifact 與 backbone engine／ONNX 前，比對 MANIFEST 記錄的 hash；不反序列化 pickle |
| dev／eval 責任 | pickle ckpt 載入、`mamba_args` 解讀、訓練端 lineage 檢查 |
| extraction boundary | lineage 判定移到**匯出時**（`developer_build_debug`）；shipping 只做 hash 驗證 |
| validation | 匯出紀錄與 [training_lineage_inventory.md](../research/training/training_lineage_inventory.md) 的 headline ckpt 對得上；shipping 對被竄改的 artifact fail-closed |

---

## 6. Phase B 拆分（只定邊界與順序，不實作）

每個 PR 只改表內列出的範圍；驗收一律是**同機器、對 §2 oracle** 的對照，不發新的 benchmark claim。native 與 Python 的比對要先在該組態宣告 observed range（重現性是 per-configuration 的），FPS 只能在同 session 附 `--control` 讀「相同或不同」。

| PR | 單元 | 內容 | scope | 驗收 | 依賴 |
|:--|:--|:--|:--|:--|:--|
| **PR-1** | U1a | headline head 匯出＋lineage 紀錄（B1、B5 的 artifact 端） | developer_build_debug 工具 | artifact 可重建、hash 與 lineage 記錄齊全 | — |
| **PR-2** | U1b | head parity：tensor 對 eager／compile，7-seq MOT 對 oracle；owner 判定 | shared_boundary gate | B1 validation。**失敗即停** | PR-1 |
| **PR-3** | U2a | resolved config exporter＋schema＋Python 端等價 contract test | developer_build_debug 工具 | B2 (a) | — |
| **PR-4** | U2b | native resolved-config loader → `set_*` | shipping_runtime | B2 (b) | PR-3 |
| **PR-5** | U3a | detector-output dump 工具（Python）＋native replay 宿主：**main NMS＋private continuation append → FP hard filter → GMC → tracker**；serial、eager。stage 順序必須與 oracle 相同（`evaluator.py`：`_run_nms` → `_run_detection_filters` → `_run_track`），private candidate 也要經過 FP hard filter，不得重排 | shipping_runtime | 同一份 detection 輸入下，tracker 的**結構化輸出** `(frame, local_id, box, score)` 對 Python serial 組態；**不**要求 MOT txt parity（ID 映射、formatter、tail 在 PR-6） | PR-4 |
| **PR-6** | U4 | native ID 映射＋sequence tail＋MOT formatter（B3、B4），先以 golden fixture 單獨驗，再接進 PR-5 宿主 | shipping_runtime（抽 shared_boundary） | B3、B4 validation；接線後，同一份 detection 輸入下 MOT txt 對 Python serial 組態 byte-identical（多 sequence 時用重標後的 parity） | library 部分無依賴；接線依賴 PR-5 |
| **PR-7** | U3b-1 | native ingest：nvJPEG decode＋normalize | shipping_runtime | 解碼像素對 torchvision 單獨量（差異要報告，不併入後段） | PR-5 |
| **PR-8** | U3b-2 | native detector：`TRTEngine` backbone＋B1 head＋S2 後段 | shipping_runtime | detection tensor 對 oracle | PR-2、PR-7 |
| **PR-9** | U3b-3 | 端到端 serial native：ingest→detect→post→tail | shipping_runtime | 7-seq MOT txt 對 Python serial 組態 | PR-6、PR-8 |
| **PR-10** | U5 | native graph capture＋double-buffer／event barrier | shipping_runtime | 7-seq MOT txt 對 §2 oracle（double-buffer）；FPS 只做同 session 對照 | PR-9 |
| **PR-11** | U6a | CMake：`saccade_tracking` 不再連 `saccade_perception`；OpenCV 改可選 | shipping_runtime | 既有 extensions 的 eval 輸出不變；若觸及 frozen／runtime-identity 檔案照該 gate 處理 | — （可提早做） |
| **PR-12** | U6b | `$ORIGIN` RUNPATH、明確 SM 清單＋PTX、glibc baseline、§0 的 G2 四項檢查 | shipping_runtime | G2 定義 1–4 全過；乾淨容器可載入 | PR-10、PR-11 |

關鍵路徑：**PR-1 → PR-2（風險閘門）** 應最先做；PR-3／PR-4、PR-6 library、PR-11 與它平行。PR-2 失敗時，其餘 shipping_runtime PR 都不再開。

Phase B 各 PR 共同不得做的事：修改 eval harness 的語義或預設、刪除任何 Python 路徑、改 preset／threshold／weights、改 benchmark claim、支援 §2 以外的 preset 或輸入形式、決定 CUDA／TRT bundling（Phase C／owner）。

---

## 7. 本文刻意沒有做的事

- 沒有實作 U1–U6、沒有新增 exporter／dump 工具、沒有改 CMake。
- 沒有決定 U1 的 head 形式（TRT 或 LibTorch）、CUDA／TRT 是否 bundle、是否另做 G1 的過渡包、平台清單。
- 沒有量 parity、FPS 或精度；§6 的驗收只是門檻定義。
