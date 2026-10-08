# Native runtime resolved config（#465 Phase B PR-3／U2a）

> 狀態：PR-3 完成 exporter、schema 與 Python 端等價 contract test。PR-4a 加上 native strict loader（§7）。PR-4b 讓 native 物件只從單一參數狀態讀值、native 端不再讀 `SACCADE_*`、shipping 以這份 JSON 建立 GPU 物件並做 set 後回讀（§8）；**U2b 至此完成**。PR-3／PR-4a 不改 native code；PR-4b 改 native code 但 headline 輸出逐位元組不變（§8.6）。PR-5 加上 native post-detector replay 宿主（U3a，§9），並補齊 exporter `steps` 漏列的 3 個 post-detector 分支（§9.2）。PR-6 加上 native MOT 輸出（U4：per-sequence ID、行格式、sequence-tail interpolation，§10），接進 replay 宿主後 7-seq MOT txt 對 Python serial 組態逐位元組相同（多 sequence 以 ID 位移重標），並補上 emit 路徑的 6 個 gate（§10.2）。PR-7 加上 native ingest（U3b-1：nvJPEG 解碼＋normalize，§11），以與 oracle 同一份 nvJPEG binary、只由 resolved config 驅動；parity harness 把解碼、normalize、端到端 ingest 分開對 torchvision ingest 報告。PR-8 加上 native detector（U3b-2：resize、`TRTEngine` backbone、PR-1L LibTorch head、S2，§12），oracle 是 owner 接受的 `A_L`；operator library 以 realization attestation 綁定（PR-1L freeze 不動）；7-seq 5316 幀在 detector 邊界與每個 stage 都逐位元相同（§12.5）。PR-9 把 ingest、detector、post-detector 宿主與 MOT 輸出接成 shipping entrypoint `saccade_track` 的 serial 組態（U3b-3，§13），oracle 是 `A_L` serial；7-seq 5316 幀的 detector rows 與 7/7 sequence 的 MOT txt（重標後）都逐位元組相同（§13.4）。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B2、§6 PR-3。本文沿用該文的編號，不重述它的論證。
> 工具：`scripts/model/export_resolved_shipping_config.py`（`developer_build_debug`，位於 `decision_relevant` partition 之外）。產物：`configs/shipping/mamba_whole_graph.resolved.json`（commit 進 repo）。

---

## 1. 檔案

`saccade.resolved_shipping_config/v1`，五個頂層鍵，缺一不可：

| 鍵 | 內容 |
|:--|:--|
| `source` | preset 路徑與 sha256、oracle argv（boundary §2：`--preset mamba_whole_graph --detector SDP --double-buffer`）、per-sequence 值的來源（`seqinfo.ini` 的 `imWidth`／`imHeight`）、刻意不輸出的 harness 讀取與非 shipping native 物件，各附理由 |
| `native_params` | `GPUByteTracker`、`GMC`、`PerceptionPipelineConfig`、`PerceptionPipeline` **實際收到的值**：constructor 參數、依呼叫順序排列的 `set_*`（13 個 tracker setter）、24 個 config 欄位。參數名稱與未傳入的預設值取自 pybind binding 原始碼 |
| `native_env` | native 原始碼中所有 `SACCADE_*` `getenv`（18 個）在 oracle 環境（皆未設定）下解析出的值；沒有預設值的列出 unset 時的效果 |
| `host_params` | `stage_order`（`_run_frame` 呼叫 stage helper 的順序）、`steps`（每個條件步驟的 gate，**停用的以明確的 `false` 列出**）、detector build、host 端常數、host 模組提到的 29 個 `SACCADE_*` 的值、`cfg`（`evaluator.py`／`pipeline.py`／`stages.py` 讀取的 416 個 `cfg`／`kwargs` 鍵與讀到的值） |

boundary §5 B2 凍結的 tail 在 `host_params.steps` 中逐項對得上：Cheb-GR／evfifo／occ-audit、`post_lifecycle_merge`、deferred alias、tracklet quality filter 都是 `false`，interpolation 是 `true`（`35`／`5`／`0`）。

## 2. 值從哪裡來

exporter **不另抄一份 preset**。它執行 oracle 自己的程式碼，只把 native 類別換成會記錄呼叫的替身：

1. **解析**：執行 `mot17.py` 的 `_load_config_defaults` 與 `__main__` 區塊（parser 建構、argv、`configure_runtime_env`、`eval_kwargs` 過濾、Mamba builder 呼叫），再執行 `run_eval` 開頭的 `parse_eval_config(...)`，得到 oracle 用的同一個 `EvalConfig`。被跳過的 `__main__` 敘述（NV12 relaunch、sequence 列舉、run manifest、multi-process launcher）各有理由寫在 span 定義中。
2. **native 參數**：執行 `run_eval` 從 ReID／FPN 模式選擇到 ONMS 參數的整段，以及 `EvalPipeline.__init__` 從 tracker 建立到 `active_tracker_thresholds` 的兩段（只排除 GMC frame buffer 配置）。tracker 呼叫經過**真正的** Python `GPUByteTracker` wrapper，所以記錄到的是 extension 收到的位置參數與型別轉換後的值。
3. **與 frame rate 無關**：同樣的擷取在 `seqinfo` frameRate＝14、25、30 下重跑，`native_params` 必須相同（`per_seq_adapt` 關閉）。
4. **gate**：`steps` 的每一條是 oracle 自己的 gate 運算式，以 oracle 的 namespace 求值；運算式必須（AST 正規化後）是 oracle 某個分支條件的整體或其 `and` 子句、或某個 assignment／keyword 的值。只有這些位置能保證「`false` ⇒ 該分支不會執行」。

以 import blocker 擋掉 `saccade_*_ext`（模擬 CI pytest job 沒有建置 extension 的環境）重跑 `--check`，輸出與有 extension 時逐位元相同。

## 3. Fail-closed 檢查

exporter 遇到下列情況就停止（`tests/unit/test_resolved_shipping_config.py` 對每一項都有 mutation test）：

- `run_eval`／`EvalPipeline.__init__` 中有一個碰到 native 物件的頂層敘述，既沒有被執行，也不在附理由的 `NOT_EXECUTED` 清單裡（清單項目若反而被執行也算錯）；
- 被執行的敘述呼叫了 binding 中不存在、或無法唯一對上 overload 的 native 方法，或者 `PerceptionPipelineConfig` 有欄位停在 C++ 預設值；
- `cfg` 讀取了 `EvalConfig` 沒有的欄位，或同一鍵在不同位置解析出不同值；
- gate 運算式已不在 oracle 原始碼中、span 錨點移位或不唯一；
- native `getenv` 沒有可解析的預設值，也沒有登記 unset 效果；
- 非 shipping 的 native 物件被啟用（lifecycle merger `enabled` 必須是 false；relinker／ReID extractor／cropper 不得被建立）。

新增的 `cfg` 讀取或 setter 呼叫會改變輸出，所以 `--check` 與 contract test 會失敗，直到檔案重新產生、diff 經過 review。contract test 另外拿 exporter **沒有用到**的來源交叉比對：preset YAML 的每個鍵、`scripts/tools/resolved_bridge_policy_config.py` 的解析結果、boundary 文件凍結的 tail。

## 4. 這次匯出看到的事（交給後續 PR）

- **external FP filter 在 headline 是開的**：`external_fp_filter_mode` 預設是 `rule`，`_run_detection_filters` 會在 FP hard filter **之前**套用它，參數是 `run_eval` 裡的 `RuleBaselineConfig()` 預設值（已輸出在 `host_params.external_fp_rule_config`），`min_score` 是當幀的 score floor。boundary §6 PR-5 的文字只列出「FP hard filter」；在函式層級的順序（`_run_nms` → `_run_detection_filters` → `_run_track`）上兩者一致，但 PR-5 的 native replay 必須包含這一步。boundary 文件的 S5a、B2、§6 PR-4／PR-5 已在本 PR 更正。
- **ReID 關閉時仍有 native 參數**：`set_reid_params`、`set_unified_score_params` 一律會被呼叫；relink bridge 以 `set_relink_params(enabled=False, bidirectional=True, …)` 打開。PR-4 loader 要照原樣套用，不能因為「ReID off」就略過。
- **native 端的 env 預設是語義的一部分**：例如 `SACCADE_ENABLE_DDA=true`、`SACCADE_STABILITY_W=0.1`、`SACCADE_GMC_PCR_THRESH=5.0`。shipping 不讀 env（B2），所以這些值要成為明確常數。
- `GPUByteTracker` 以 `embedding_dim=768` 建立；`GMC` 只傳 `downscale=4`，其他 5 個 constructor 參數取 binding 預設值，已明列。
- `detector.build` 記錄的是 oracle 的 build（PyTorch head＋`torch.compile`，`trt_head_engine=""`）。shipping head 由 PR-2L 的 LibTorch artifact 決定，本檔不編碼 head 形式。

## 5. 限制（本 PR 沒有宣稱的事）

- `host_params.cfg` 是「oracle host 讀了什麼、讀到什麼值」，**不是**「哪些鍵在 headline 上有行為」。是否 active 只在 `steps` 的 gate 層級回答。
- `steps` 只求值 gate 的**設定**部分；`true` 表示設定允許該步驟執行，runtime 子句（例如偵測框非空、`is_cuda`）不在檔案裡。`schedule.double_buffer` 以 `torch.cuda.is_available()=True` 求值（oracle 機器的情況）。
- `cfg` 讀取的機械覆蓋範圍是 `evaluator.py`、`pipeline.py`、`stages.py`；其他模組經由這些讀取的參數拿到值。寫死在 host 程式碼裡、不經過 `cfg` 的常數，只有在被執行的 span 產生（`RuleBaselineConfig`、ONMS 參數、容量）或被 exporter 點名（`_FP_HARD_REJECT_SCORE`）時才會出現；沒有被列舉的部分由 PR-5／PR-6／PR-9 的 parity 兜底。
- `native_env` 的值是從原始碼解析出來的 unset 預設；`SACCADE_KALMAN_ADAPT_MODE` 等沒有預設值的項目只記錄 unset 時的效果。
- native 唯讀屬性 `max_objects`／`max_assoc` 由替身依 `tracker_gpu.cu` 的定義重現（`max_assoc_ = max(1, max_assoc)`），不是執行 native getter 得到的。

## 6. 重現

```bash
.venv/bin/python scripts/model/export_resolved_shipping_config.py            # 重新產生
.venv/bin/python scripts/model/export_resolved_shipping_config.py --check    # 與 commit 的檔案比對
.venv/bin/python -m pytest tests/unit/test_resolved_shipping_config.py
```

不需要 GPU、checkpoint、資料集或已建置的 native extension。

## 7. Native strict loader（PR-4a／U2b-a）

PR-4a 只建立 shipping runtime 的 config ABI：把這份 JSON 嚴格解析成 typed native state。**不**呼叫任何 tracker／GMC／pipeline setter、**不**改 `tracker_gpu.cu` 或 env plumbing，也**不**宣稱 native 已經使用這份 config；這些是 PR-4b（boundary §6 PR-4：PR-4b 通過 readback 與 env-independence 驗收前，U2b 不算完成）。

| 項目 | 位置 |
|:--|:--|
| strict JSON reader／writer | `shipping/include/saccade_shipping/strict_json.hpp`、`shipping/src/strict_json.cpp` |
| typed config＋loader | `shipping/include/saccade_shipping/resolved_config.hpp`、`shipping/src/resolved_config.cpp`（`shipping/CMakeLists.txt` 的 target `saccade_shipping_config`，純 C++17，不連 CUDA／torch） |
| `host_params.cfg` 欄位清單 | `shipping/include/saccade_shipping/host_cfg_fields.inc`，由 `scripts/model/render_shipping_host_cfg_schema.py` 從 commit 的 JSON 產生（只有 key、member 名稱與型別，沒有值）；`--check` 檢查是否過期 |
| 測試 | `tests/native/test_resolved_config.cpp`（ctest `saccade_resolved_config_test`：`cmake -S shipping -B build-shipping && cmake --build build-shipping && ctest --test-dir build-shipping`；CI job `shipping-config-loader` 在無 GPU runner 上執行同一組指令）、`tests/unit/test_shipping_config_loader_sources.py` |

`shipping/` 放在 `src/`、`include/` 之外，因此不屬於 `decision_relevant` partition（在 `h2_path_partition.py` 中是 `unclassified`），PR-4a 不移動 runtime-identity 的 implementation axis；`shipping/CMakeLists.txt` 是獨立 project，root `CMakeLists.txt`（runtime-identity environment recipe 的輸入）不變。PR-4b 把 loader 接進 native 物件時，要和 runtime-identity republication 一起處理這個分類（已處理，見 §8.7）。

**typed 結構**：`ResolvedShippingConfig` → `SourceInfo`、`NativeParams`（`PerceptionPipelineConfigParams`、`PerceptionPipelineParams`、`TrackerRuntimeParams`、`GmcRuntimeParams`）、`NativeEnvParams`、`ShippingHostConfig`（`HostSteps`、`DetectorHostConfig`、`HostCapacities`、`ExternalFpRuleConfig`、`OnmsConfig`、`HostEnv`、`HostCfg`）。每個 struct 的 `visit` 是唯一的 schema 宣告，解析與序列化共用同一份，所以 JSON → typed → snapshot 的欄位名稱、順序與型別不會分歧。`ResolvedShippingConfig` 只能由 loader 產生。

**fail-closed 規則**（每條都有測試）：

- `schema` 必須完全等於 `saccade.resolved_shipping_config/v1`，在其他欄位之前檢查；
- 每一層都要求 schema 宣告的所有欄位、拒絕所有未宣告的鍵；`calls` 的 method 名稱、數量與順序也是 schema；
- 型別嚴格：`bool`／int／float／string／null／list 不互通，float 欄位寫成 int literal（`35` vs `35.0`）也算型別錯誤（與 exporter 的 `json.dumps` 一致）；
- reader 拒絕重複鍵、`NaN`／`Infinity`、溢位成 inf 的數字、超出 int64 的整數；
- enum／range：例如 `kalman_adapt_mode ∈ {0..4}`、`bridge_anchor ∈ {0,1,2}`、`occ_mode ∈ {0,1}`、閾值 ∈ [0,1]、native `int` 參數在 int32 範圍內、`stage_order` 是 10 個已知 stage 的排列、`set_frame_size` 的 `w`／`h` 必須分別是 `imWidth`／`imHeight`、`reid_ptr`／`cropper_ptr` 必須是 0、`set_homography.h` 必須是 null。這些 numeric range 是 PR-4a 的 **shipping-admissibility guard**（shipping ABI 的 fail-closed policy），**不是**舊 native setter 接受域的完整重現：舊 setter 會 clamp／canonicalize 許多值（例如 `confirm_streak=max(1,…)`、`r_scale=max(0.01,…)`、`set_oao_params` 把 `tau`／`score_w` 夾進 [0,1]、`set_occ_params` 的 `ttl=max(1,…)`），guard 有些比它窄、有些比它寬。PR-4b 為了讓 native readback 精確等於 JSON，可以在需要 canonical domain 的地方收緊；
- boundary §5 B2／B4 凍結為停用的 tail 步驟（`tail.cheb_gr_or_occ_audit`、`tail.post_lifecycle_merge`、`tail.deferred_alias`、`tail.tracklet_quality_filter`）被設為 `true` 就失敗；
- `native_env`：有預設值的 getenv 必須給明確值，沒有預設值的必須是 `{"unset_effect": ...}`，兩者不能互換；
- loader 不讀任何環境變數（source guard 測試禁止 `shipping/` 出現 `getenv` 等呼叫），也沒有 fallback 或預設值。

`host_params.cfg` 的 416 個鍵只檢查型別與有限性，不設 range：它記錄的是「oracle 讀了什麼」（§5），native 會用到的值在進入 native 的位置（`native_params`、`steps` 與 host 的 typed 區段）檢查。

**驗收**（`tests/native/test_resolved_config.cpp`，2862 checks、0 failures，g++ 與 clang++ 皆通過）：commit 的 JSON 解析成功，`canonical_snapshot` 與檔案**逐位元組相同**；逐一刪除 732 個欄位、在 69 個 object 中插入未知鍵、1904 個型別替換、53 個 enum／range／語義案例全部被拒；把所有 `native_env`／`host_params.env` 的鍵與其他 `SACCADE_*` 設成各種值（260 次指派）後，snapshot 不變。writer 的 float `repr` 與字串 escape 以 Python 實際輸出為準做了單元測試。

## 8. Native 參數 authority（PR-4b／U2b-b）

PR-4b 讓 native 物件真正使用這份 config：tracker 只有一份參數狀態，legacy env 與 shipping JSON 都寫進它；native 物件不再讀環境變數；第一次 graph capture 之後設定凍結。boundary §6 PR-4 的 readback 與 env-independence 驗收在這裡完成。

| 項目 | 位置 |
|:--|:--|
| tracker 參數狀態 `TrackerParams`（唯一 authority，CUDA-free） | `include/tracking/tracker_params.hpp`、`src/tracking/tracker_params.cpp`（target `saccade_tracker_params`） |
| GMC／PerceptionPipeline 設定與 snapshot | `include/tracking/perception_params.hpp` |
| legacy `SACCADE_*` 解析 | `include/tracking/legacy_env.hpp`、`src/tracking/legacy_env.cpp`（只有 pybind binding 與 `seq_runner.cpp` 呼叫） |
| JSON → native 參數、readback（CUDA-free） | `shipping/include/saccade_shipping/native_config.hpp`、`shipping/src/native_config.cpp` |
| 由 JSON 建立 GPU 物件 | `shipping/include/saccade_shipping/native_build.hpp`、`shipping/src/native_build.cpp`（連 `saccade_tracking`；root build 以 `add_subdirectory(shipping)` 納入） |
| 測試 | `tests/native/test_shipping_native_config.cpp`（CPU，CI `shipping-config-loader`）、`tests/native/test_shipping_native_build.cpp`（GPU，`ENABLE_NATIVE_TESTS`）、`tests/unit/tracking/test_native_params_authority.py`（pybind legacy 路徑）、`tests/unit/test_shipping_config_loader_sources.py`（source guard） |

### 8.1 單一 `params_`

- `GPUByteTracker::Impl` 只有一份 `TrackerParams params_`。原本散在 Impl 的 84 個設定成員（含 constructor 從 env 讀的 9 個）、update path 兩個 process-wide `static`（auction bid 權重），以及原本只存在 device 端的 homography 全部併入；update path 的 kernel 引數都從 `params_` 取。
- 每個 setter 經 `TrackerParams` 的同名成員函式寫入。舊 setter 的 canonicalization（`confirm_streak=max(1,…)`、`r_scale=max(0.01,…)` 等）原樣搬過去，legacy 行為不變。
- `snapshot()` 回傳 `params_` 的拷貝，加上 constructor 尺寸、research hook／診斷是否啟用、`config_frozen`。這是唯一的讀取介面，沒有逐欄 getter，也沒有另存一份 raw config 讓 readback「看起來」相等。pybind 以 `snapshot()` 回傳一個扁平 dict，鍵是 `<setter>.<arg>`／`native_env.<SACCADE_*>`。
- `set_oao_params.score_w`：值 `<= 0` 原樣保存，`> 0` 才夾到 `(0, 1]`。kernel（`oao_score_scale`）一直以 `<= 0` 表示 off，所以行為不變；resolved 的 `-1` 讀回仍是 `-1`（PR-4b 之前會被夾成 `0`）。
- `set_unified_score_params`：PR-4b 之前 native 端直接丟棄；現在存進 `params_` 以便 readback。native kernel 仍然不讀這 5 個值。

### 8.2 `SACCADE_*` 移出 native 物件

tracker、GMC、pipeline 與 `filter_detections_cuda` 都不再呼叫 `getenv`，改收明確參數（`set_hatch_params`、`set_assoc_dump_path`、`GMC::set_pcr_thresh`、`PerceptionPipeline::set_filter_compaction_mode`／`set_private_workload_stats_enabled`、`filter_detections_cuda` 的 `FilterCompactionMode` 參數）。legacy 前端（pybind constructor、`set_params` binding、`filter_detections_cuda` binding、`seq_runner`）以 `legacy_env` 解析，解析規則與預設值和原本相同：

| 變數 | 原本讀取的位置與時機 | PR-4b 之後 |
|:--|:--|:--|
| `ENABLE_DDA`、`DDA_MAX_COST`、`GATE_ADAPT_R_MULT`、`OCC_VEL_DAMP`、`OCC_VEL_OCC_THRESH`、`OUTPUT_MEASUREMENT`、`COAST_*` | tracker constructor | 物件建立時（相同） |
| `FRESHNESS_W`、`STABILITY_W` | 每個 process 第一次 tracker update（`static`） | tracker 建立時 |
| `KALMAN_ADAPT_MODE` | 每次 `set_params` | 每次 `set_params`（在 binding；相同） |
| `ASSOC_DUMP` | 每個 process 第一次 update（`static`） | tracker 建立時；空字串＝off |
| `GMC_PCR_THRESH` | 每個 process 各 launcher 第一次呼叫（`static`） | GMC 建立時 |
| `DETERMINISTIC_FILTER_COMPACTION`、`ATOMIC_FILTER_BASELINE` | 每次 filter 呼叫 | pipeline：建立時；`filter_detections_cuda` binding：每次呼叫（相同） |
| `ASSOC_STATS` | pipeline constructor | pipeline 建立時（相同） |
| `HO_DEBUG_LEVEL` | Cheb-GR handover binding | 不變（binding 本來就是 legacy 前端；shipping 不建立該物件） |

讀取時機改變的幾項只有在 process 內途中改 env 時才有差別；oracle 不會這樣做（`configure_runtime_env` 在建立任何物件之前設定 env）。

**原本的 process-wide `static`**（`FRESHNESS_W`、`STABILITY_W`、`GMC_PCR_THRESH`、`ASSOC_DUMP`）：它們是 function-local `static const`，在 process 第一次 update／estimate 時讀一次 env，之後所有 instance 共用同一個值。初始化之後就不再寫入，所以 instance 之間從未透過它們互相影響；「共用」唯一可觀察的效果是：同一個 process 內、兩次建立物件之間 env 改變時，後建立的 instance 仍沿用第一次讀到的值。PR-4b 改成每個 instance 在建立時讀，這是刻意的語義改變。repo 內寫這些變數的地方：ablation 只把它們傳給子 process；唯一在 process 內寫的是 `tracker_block_divergence_465.py`（`SACCADE_ASSOC_DUMP`），它在 import 與建立任何 tracker 之前就設定好，dump 以 append 模式寫、frame counter 本來就是每個 instance 各自一份，所以行為相同（該 frozen runner 的註解「read once … at its first update」現在不精確，但它的 blob 被凍結，PR-4b 不改它）。`test_native_params_authority.py` 的兩個 two-instance 測試釘住這兩點：env 在兩次建立之間改變時，兩個 instance 各自保有建立時的值，後者不影響前者；env 固定時（harness 的實際情況），兩個 tracker 交錯執行，各自的輸出與單獨執行時逐值相同。exporter 的 `native_env` 掃描現在在 `legacy_env.cpp` 找到同樣的 18 個讀取，輸出逐位元組不變（`--check`）。source guard 測試要求 native 原始碼中只有 `legacy_env.cpp` 與 pybind binding 讀 `SACCADE_*`，`shipping/` 不得 include `legacy_env.hpp`。

### 8.3 Graph capture 之後凍結

`update()`／`update_into()` 第一次在 CUDA stream capture 中執行時（`cudaStreamIsCapturing`），tracker 設 `config_frozen`。之後每一個寫 `params_` 或啟用 research hook 的 setter 都丟 `std::logic_error`（Python 為 `RuntimeError`），因為已 capture 的 graph 會一直 replay 當時寫死的 kernel 引數，新值永遠不會生效。不凍結的是逐幀狀態輸入（`update_reference_features`、`set_clean_embedding_flags*`、`bind_features_buffer`）、drain／clear，以及診斷用的 `set_assoc_dump_path`。

對 eval harness 的影響：headline 在 `EvalPipeline.__init__` 設完所有參數之後才在第一幀 capture tracker graph，所以沒有影響（§8.6）。若某組態在 `use_tracker_graph` 之下於 capture 之後才呼叫 setter（例如 `stages.py` 在逐幀 threshold 改變時呼叫 `set_params`），這個呼叫原本對 replay 的 graph 不起作用；現在會直接報錯。

### 8.4 Shipping 建立物件與 set 後回讀

`build_tracker`／`build_gmc`／`build_perception_pipeline` 以 JSON 的 constructor 值建立物件，先套 `native_env` 的 hatch，再依 oracle 的呼叫順序（schema 順序）呼叫同一批 setter，不依賴任何 native 預設值；接著讀 `snapshot()` 與 JSON 比對，不一致就丟 `ConfigError`：

- snapshot 的每個鍵都要有 JSON 值，JSON 的每個值都要有 snapshot 鍵；
- float 以 setter 實際做的 float32 轉型後**逐位元**比較；`per_sequence` 以該 sequence 的 `seqinfo.ini` 值比較；`{"unset_effect": …}` 對應的診斷必須是 off；
- 沒有 JSON 值的 snapshot 鍵只有下列幾個，各附理由（`tracker_native_only_expectations()`）：`set_reid_min_candidates.min_candidates`（＝2；沒有 pybind binding，oracle 從未設定，exporter 因此沒有值；只在傳入 embeddings 的 association 分支讀取，而 shipping tracker 到不了那個分支，見下）、`embeddings_forbidden`（必須 true）、4 個 research hook（必須 off）、`config_frozen`（必須 false）；
- `native_env` 的 18 個鍵各自歸屬一個物件：tracker 12、GMC 1、pipeline 3；`HO_DEBUG_LEVEL` 與 `KALMAN_ADAPT_MODE` 沒有 shipping consumer（理由寫在 `native_env_consumers()`）。

**不得帶 ReID embeddings**：shipping 沒有 ReID。`build_tracker` 呼叫 `GPUByteTracker::forbid_embeddings()`（單向，不能再打開）；之後 `update()`／`update_into()` 只要收到非 null 的 embeddings 指標就丟 `std::invalid_argument`，在任何 tracker 狀態改變之前。所以 embeddings 分支（以及只有它讀的 `reid_min_candidates`、`set_reid_params` 的值、appearance relink）在 shipping tracker 上不可達；這個 assertion 在 U3 宿主寫出來之前就已經生效，U3 不能漏掉它。legacy 前端從不呼叫它，eval harness 不受影響。ReID 若將來回到 shipping，要另開 scope：補 binding、JSON 欄位與 readback，再拿掉這個 guard。

PR-4a 的範圍 guard 沒有改。會被 setter canonicalize 的值（例如 `confirm_streak=0`、`score_w=1.5`、`occ ttl=0`、`SACCADE_COAST_MAX_AGE=2.5`）可以通過 loader，但會在 readback 失敗，錯誤訊息列出鍵與兩邊的值。

### 8.5 驗收（boundary §6 PR-4）

| 驗收項 | 證據 |
|:--|:--|
| `native_params` set 後回讀 == JSON | CPU：在 `TrackerParams`（tracker 實際使用的 setter 程式碼）與規劃的 GMC／pipeline snapshot 上比對；GPU：在真正的 `GPUByteTracker`／`GMC`／`PerceptionPipeline` 上比對，並確認與 CPU 規劃逐欄相同 |
| 每個 resolved 值都到達恰好一個 native 欄位 | 逐一擾動 123 個值，每次只有對應的那個 snapshot 鍵改變；另有 2 個值被 loader 釘死、無法擾動（`reid_ptr`、`cropper_ptr` 只允許 0） |
| 設定任何 env 都不改變結果 | 把 18 個 `native_env` 變數設成多組值，CPU 規劃與 GPU 物件的 snapshot 都不變 |
| 未知／缺少欄位 fail-closed，刪掉有 native 預設的欄位必失敗 | PR-4a loader（§7，未改） |
| 凍結 | GPU：capture 之後 19 個 setter 全部丟錯，snapshot 不變；未 capture 的 update 不凍結 |
| 不得帶 embeddings | GPU：shipping tracker 的 `update_into`／`update` 帶 embeddings 都丟 `invalid_argument`，snapshot 不變；legacy tracker 不受影響 |
| 原本的 process-wide `static` | pybind：two-instance 測試（§8.2） |

測試計數：CPU 229 checks、GPU 55 checks、pybind 12 tests，全部通過。

### 8.6 Headline 行為

同一台機器、同一個 `build/` 組態，以 boundary §2 的 oracle（`mot17.py --preset mamba_whole_graph --detector SDP --double-buffer`）跑 MOT17 train 7 個 SDP sequence：main `469f159d` 跑兩次彼此逐位元組相同；PR-4b 的 build 跑兩次（第一版與加上 review 修正之後的最終版各一次），每次 7 份 MOT txt 都與 main 逐位元組相同（IDF1 78.3／MOTA 77.9／IDs 429）。這只說明 headline 組態下輸出未變，不是一般性的等價主張。

### 8.7 分類與 runtime identity

`shipping/` 與 `configs/shipping/` 在 `h2_path_partition.py` 中改為 `decision_relevant`：前者在 PR-4b 第一次連到 tracker，後者（這份 resolved JSON）是 shipping builder 唯一的決策輸入，改了它卻不移動 coordinate 會留下 attestation 洞。implementation 軸原本以副檔名排除 `.md`／`.rst`／`.txt`，連帶排除了 `CMakeLists.txt`；`build_runtime_identity.py` 現在保留檔名為 `CMakeLists.txt` 的檔案，所以 `shipping/CMakeLists.txt`（shipping library 的 build recipe）進入 implementation 軸，原本就被漏掉的 `src/tracking/CMakeLists.txt`（fpn_reid extension 的獨立 build）也一併納入。root `CMakeLists.txt` 加入 `saccade_tracker_params` 與 `add_subdirectory(shipping)`，移動 environment recipe；`src/`、`include/`、`shipping/`、`configs/shipping/` 移動 implementation 軸；`h2_path_partition.py` 與 `build_runtime_identity.py` 移動 identity_semantics 軸。runtime coordinate 依 [republication runbook](runbooks/runtime_identity_republication.md) 重新出版，以 stacked PR 和本 PR 一起 land。

### 8.8 限制與留給後續的事

- `reid_min_candidates` 是唯一沒有 JSON 來源、但 kernel 會讀的參數；它只在 embeddings 分支被讀，而 shipping tracker 以 `forbid_embeddings()` 讓該分支不可達（§8.4）。owner 決定不為這個 dormant knob 擴張 ABI。
- Python wrapper 的 `set_reid_min_candidates(1)`（`pipeline.py` 在某些 ReID 組態下呼叫）因為沒有 binding 而是 no-op，這是 PR-4b 之前就存在的獨立 bug，另開 issue 追蹤，PR-4b 不改它。
- readback 證明 native 狀態等於 JSON；kernel 是否用到每個欄位，由逐欄映射測試與 §8.6 的 headline 輸出不變共同支撐，不是由 readback 本身證明。
- U3–U5（native 宿主、ingest、graph／double-buffer）在 PR-4b 時尚未開始；U3a（PR-5）見 §9。

## 9. Native post-detector replay 宿主（PR-5／U3a）

PR-5 建立第一個 native 宿主：吃 detector 輸出，依 oracle 的 serial 順序跑 **main NMS＋private continuation append → external FP rule filter → FP hard filter → GMC → tracker update**（boundary §6 PR-5），eager、不用 graph。驗收是同一份 detection 輸入下，tracker 的結構化輸出（box、score、local id、class）對 Python serial 組態逐位元相同；**不**含 MOT txt（ID 映射、interpolation、formatter 在 PR-6）。

| 項目 | 位置 |
|:--|:--|
| 計畫＋fail-closed gate、CPU detection filter twin（CUDA-free） | `shipping/include/saccade_shipping/post_detector_plan.hpp`、`shipping/src/post_detector_plan.cpp`（併入 `saccade_shipping_native_config`） |
| GPU 宿主 | `shipping/include/saccade_shipping/post_detector_host.hpp`、`shipping/src/post_detector_host.cpp`（併入 `saccade_shipping_native`） |
| dump 工具（`developer_build_debug`） | `scripts/eval/diagnostics/dump_post_detector_replay.py`（格式 `saccade.post_detector_replay/v1`；`--verify`／`--against` 檢查完整性與兩份 dump 是否逐位元相同） |
| replay 工具（`developer_build_debug`） | `shipping/tools/saccade_replay.cpp`（target `saccade_replay`，只在 root build） |
| 測試 | `tests/native/test_shipping_detection_filters.cpp`（CPU，CI `shipping-config-loader`，fixture `tests/native/fixtures/shipping_detection_filters.json` 由 `scripts/model/render_shipping_detection_filters_fixture.py` 產生）、`tests/native/test_shipping_post_detector_host.cpp`（GPU）、`tests/unit/test_post_detector_host_oracle_pins.py`（oracle source pin＋fixture freshness） |

### 9.1 宿主照抄的 oracle 事實

- 空的 detector 輸出：不跑 NMS、GMC、tracker（`_run_frame` 提早 return），也不觸發 pre-roll。
- main NMS 走 graphed 路徑：`copy_pad` 到 `nms_fixed_n`（＝`max_assoc`＝1024）→ `process_detections_main_nms_graph_nocopyback` → `process_detections_split_pipeline_graphed`（private prior＝age ≤ `private_prior_max_age` 的 active track）。
- 兩個 filter 在 host 端以 float32 計算，門檻轉成 float32（torch 對 float32 tensor 與 Python scalar 比較的方式）；external FP 只實作 `rule` 模式、penalty 關閉的分支，其他組態由計畫拒絕。`min_score`（當幀 score floor）只在 penalty 分支被讀，因此不進宿主。
- tracker：`update_into` 一律傳 `num_dets=max_assoc`（尾端補零）、無 embeddings、`light_factor=0`、`mid_thresh_scale=1`、`out_capacity=max_objects`；warp 的初值是 `torch.eye(2, 3)`。
- **pre-roll**：`GraphedTrackerUpdate` 在一個 sequence 第一次真正 update 之前，以全零輸入跑 `update_into` 共 4 次（`_warmup` 1 次＋`make_graphed_callables` 的 `num_warmup_iters=3`；capture 本身只錄 kernel 不執行，host 端只動到沒有輸出讀取的 `processed_frame_count_`）。宿主照做（`kGraphedTrackerUpdatePreRoll=4`），由 `test_post_detector_host_oracle_pins.py` 對 oracle 原始碼釘住。
- GMC／tracker 每個 sequence 一份，`PerceptionPipeline` 每個 run 一份。

### 9.2 Exporter `steps` 的更正

PR-3 的 `steps` 宣稱列出每個條件步驟的 gate，但漏了 post-detector 路徑上的 3 個分支；PR-5 先以直接讀 `cfg`／`env` 的方式 fail-closed，再把它們補進 exporter（新增 4 條，headline 全為 `false`，JSON 只多這 4 個鍵）：

| step | oracle 的 gate（位置） |
|:--|:--|
| `post.scene_adapt` | `cfg.detection.scene_adapt_enabled`（`pipeline.py`，IfExp 條件） |
| `post.narrow_person_bonus` | `0.0 if cfg.detection.scene_adapt_enabled else cfg.detection.narrow_person_score_bonus`（`pipeline.py`，每個 sequence 的 bonus 初值；只有 scene-adapt 會在之後提高它。兩條都是 `false` ⇒ bonus 恆為 0，`apply_narrow_person_score_bonus` 不做事） |
| `filter.stage2_quality_gate` | `cfg.stage2_quality_gate`（`evaluator.py` `_run_frame` 的 `and` 子句） |
| `track.score_jitter` | `os.environ.get("SACCADE_SCORE_JITTER", "")`（`stages.py`） |

`HostSteps` 解析這 4 個欄位，計畫改為只讀 `steps`。

### 9.3 驗收

同一台機器、`build/` 組態，branch commit `22556f7c`（工作樹乾淨），以 `mot17.py --preset mamba_whole_graph --detector SDP`（serial，無 `--double-buffer`）跑 MOT17 train 7 個 SDP sequence 並 dump，再以 `saccade_replay` 重放：

| 驗收項 | 結果 |
|:--|:--|
| 4 個 stage（post-NMS、tracker 輸入、tracker 收到的 warp、tracker 輸出）逐幀比對 | 7/7 sequence、5316/5316 幀全部逐位元相同，數量不符 0、最大絕對差 0；改用 §9.2 重新產生的 JSON 重放，結果相同 |
| oracle 自身的重現性 | 第二次執行（`--frames hash`）的每個 stream 與 frame hash 都和第一次相同；dump 完整性（sha256 對 `meta.json`）通過 |
| 比對器不是空轉的 | 負控制：參考輸出改 1 ulp ⇒ 只有 `tracker_output` 在該幀發散；detector 框 +40 px ⇒ post-NMS／tracker 輸入／輸出從該幀起發散，warp 不受影響 |
| CPU filter twin | Python 函式產生的 golden fixture（2 組門檻 × 337 列，含每個門檻與其 float32 鄰值）逐位元相同；7 個 twin 的變異（`<=`／`<`、double 比較、clamp 下限等）全部被抓到 |
| GPU 宿主生命週期 | 空幀不觸發任何步驟、pre-roll 只在第一次 update 跑一次、首次 update 後拒絕改 pre-roll、計畫拒絕的 config 讓建構子失敗、兩個宿主逐位元相同（50 checks） |

serial 參考的 headline 指標為 IDF1 78.3／MOTA 77.9／IDs 429（與 §8.6 的 double-buffer 數字相同；這裡只記錄觀察值）。

**pre-roll 的證據只來自 source pin**：以 `--pre-roll 0` 與 `--pre-roll 1` 重放同一份 dump，7 個 sequence 也都逐位元相同。所以 replay parity 無法區分 pre-roll 次數；`4` 這個值由 oracle source pin 維持，不是由 replay 證明。

結果目錄：`results/465_pr5_replay/full7_22556f7c/`（`replay_report.json`、`replay_report_steps.json`、`replay_preroll{0,1}.json`、`verify.json`、兩份 dump 的 manifest 與 log；frames 27 GB 不納入版本控制）。

### 9.4 已知的 oracle 缺陷（不修）

external FP rule filter 會刪列，但沒有同步裁切 `geometry_suspect_mask`，之後 mask 與偵測框不再對齊。headline 下游不讀它（`geometry_suspect_support=false`；bank 更新的讀取在 ReID 關閉時不會發生），所以不可觀察。PR-5 不改 Python oracle 的語義，native 宿主也不產生 mask。

### 9.5 限制

- 只驗 serial、eager；graph 與 double-buffer 在 U5。
- 輸入是同一台機器 dump 的 detector 輸出與 GMC 輸入幀；ingest（解碼）與 detector 不在範圍內（PR-7 等）。
- filter 在 host CPU 上執行並來回拷貝；這是 parity 宿主，不是效能主張。
- 驗收是 headline 組態下的觀察，不是一般性的等價主張；計畫拒絕的組態（ONMS、crowd／duplicate／cap filter、birth gate、ReID、stage-2 gate、bonus、jitter 等）沒有 native 實作。

### 9.6 重現

```bash
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --out <dir>/dump
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --out <dir>/repeat_hash --frames hash
.venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --verify <dir>/dump --against <dir>/repeat_hash
cmake --build build --target saccade_replay
build/shipping/saccade_replay --config configs/shipping/mamba_whole_graph.resolved.json --dump <dir>/dump --report <dir>/replay_report.json
```

---

## 10. Native MOT 輸出（PR-6／U4）

PR-6 把 PR-5 宿主的 tracker 輸出變成每個 sequence 的 MOT txt：B3（track ID 與行格式）與 B4（sequence-tail interpolation），先以 golden fixture 對 Python 函式逐位元組驗證，再接進 `saccade_replay`。headline 的 MOT txt 只由 tracker 輸出決定：fast emit 逐列輸出，tail 只有 interpolation（boundary §5 B2 凍結的 tail；其他 tail 步驟 loader 已 fail-closed）。

| 項目 | 位置 |
|:--|:--|
| ID 映射、行格式、interpolation（CUDA-free、不讀 config） | `shipping/include/saccade_shipping/mot_output.hpp`、`shipping/src/mot_output.cpp`（target `saccade_shipping_mot`，`-ffp-contract=off`） |
| 計畫＋fail-closed gate、per-sequence 累積 | `shipping/include/saccade_shipping/sequence_output.hpp`、`shipping/src/sequence_output.cpp`（併入 `saccade_shipping_native_config`） |
| 接線 | `shipping/tools/saccade_replay.cpp`：每次 tracker update 後 `add_frame`，sequence 結束時套 tail、寫 `--mot-out`、與 dump 內 oracle 的 `eval/<seq>.txt` 比對（report `mot_txt`，格式 `saccade.post_detector_replay_report/v2`） |
| 測試 | `tests/native/test_shipping_mot_output.cpp`（CPU，CI `shipping-config-loader`；fixture `tests/native/fixtures/shipping_mot_output.json` 由 `scripts/model/render_shipping_mot_output_fixture.py` 產生）、`tests/unit/test_shipping_mot_output_oracle_pins.py`（oracle source pin＋fixture freshness）、`tests/unit/test_post_detector_replay_dump_verify.py`（dump `--verify`） |

### 10.1 照抄的 oracle 事實

- **emit**：headline 的兩個 emit 點（`stages._run_emit` 的同步路徑、`evaluator._flush_deferred_emit`）都呼叫 `helpers.fast_emit_mot_lines`：每個 tracker 列一行 `frame,id,x1,y1,x2-x1,y2-y1,score,-1,-1,-1`，float32 先轉成 double 再相減，`.2f`／`.4f`。native 以 `std::to_chars`（fixed，等同 "C" locale 的 `printf`，與 Python 一樣正確捨入、不受 locale 影響）格式化；NaN 一律寫 `nan`（Python 不印 NaN 的符號）。
- **ID**：`GlobalTrackIdMapper` 依首次出現順序從 1 編號，計數跨 sequence 延續。shipping 只做 per-sequence 規則（`SequenceIdMapper`），所以單一 sequence 時兩者相同；多個 sequence 時 oracle 的 ID 等於 per-sequence ID 加上前面 sequence 用掉的數量。
- **interpolation**：`post_merge.interpolate_tracklets` 從**文字**重新解析 MOT 行（interpolation 看到的是捨入後的值）。pandas C parser 對這些短小數給出正確捨入的 double，native 用 `std::from_chars`（fixture 產生器逐欄檢查兩者相同）。確認 track（行數 ≥ `min_track_len`）以穩定排序依 `(id, frame)` 排列；gap 是 1..`max_gap` 幀（`min_h > 0` 時兩端 `h ≥ min_h`）；`alpha = k/(gap+1)`，每欄 `r0 + alpha*(r1-r0)`，先乘、再加、各自捨入（numpy 的 ufunc 分開執行，所以 native 關掉 FMA contraction）。輸出是原始行原樣保留加上新行，依 `(frame, id)` 穩定排序；**沒有任何 gap 可補時原樣回傳、不排序**。參數取自 `host_params.cfg`（`interpolate_max_gap`／`_min_track_len`／`_min_h`），是否執行由 `steps.tail.interpolation` 決定。
- **寫檔**：`"\n".join(lines)`，沒有結尾換行；是否寫檔由 `steps.tail.write_output`（`not cfg.latency_only`）決定。

### 10.2 Exporter `steps` 補上的 emit gate

`_run_emit` 只有在下列條件全部成立時才走 fast emit：沒有 semantic relinker、id-stability filter、appearance bank、dynamic ReID controller（`_needs_emit_pipeline`），`reid_mode` 屬於 fast-emit 集合，`id_stability_filter` kwarg 關閉。workbench 路徑則完全繞過 `_run_emit`，用自己的 tracker。PR-3／PR-5 的 `steps` 只列了 relinker 與 `pipeline_relink`，PR-6 補上 6 條（headline 值見括號；JSON 只多這 6 個鍵）：

| step | oracle 的 gate（位置） |
|:--|:--|
| `emit.id_stability_filter`（false） | `cfg.post_lifecycle_appearance_gate`（`pipeline.py`，建立 `IdStabilityFilter` 的 IfExp） |
| `emit.appearance_bank`（false） | `cfg.appearance_bank_enabled`（`pipeline.py`） |
| `emit.dynamic_reid`（false） | `cfg.need_reid_enabled`（`pipeline.py`） |
| `emit.fast_emit_reid_mode`（true） | `cfg.reid_mode in ('off', 'tracker', 'extract')`（`stages.py` `_use_fast_emit` 的 conjunct） |
| `emit.id_stability_kwarg`（false） | `bool(cfg.kwargs.get('id_stability_filter', False))`（同上，oracle 寫成 `not ...`） |
| `track.workbench`（false） | `getattr(cfg, 'workbench', False)`（`pipeline.py`） |

後兩條是寫成 assignment 的 gate。為了讓 exporter 能對 oracle 原始碼驗證它們，exporter 的 gate 收集規則擴充為：assignment 的 `and` 鏈也收 conjunct，`not X` conjunct 也收 `X`（規則只會變寬，既有 step 不受影響）。`plan_sequence_output` 只讀 `steps`；另外要求 `steps.tail.interpolation == cfg.interpolate_tracklets`、`steps.tail.write_output == not cfg.latency_only`，避免手改 JSON 開了 step 卻沒有對應的參數。`track.workbench` 也加進 `plan_post_detector` 的拒絕清單，因為它換掉整條 tracker 路徑，PR-5 原本漏了這條。

### 10.3 驗收

| 驗收項 | 結果 |
|:--|:--|
| B3／B4 golden fixture（Python 函式本身產生） | emit 2 組（兩種精度的十進位捨入中點與 ±1 ulp、±0、極大值、NaN／inf）、interpolation 11 組（gap 恰為 `max_gap` 與多 1、長度恰為 `min_track_len` 與少 1、單幀 track、`min_h` 邊界、重複的 `(id, frame)`、大量相同排序鍵、非有限值、各個提早 return、兩組參數下的隨機壓力測試）逐位元組相同；GCC 13 與 GCC 16 都通過 |
| 比對器不是空轉的 | 對實作做的 14 個變異全部被抓到：FMA、倒數乘法、兩處不穩定排序、`<`／`>` 邊界 3 個、NaN 符號、寬度用 float32 相減、ID 從 0 起算、無 gap 時排序、以 float32 解析、score 精度、解析差 1 ulp |
| 停用的 emit／tail 分支 | 12 個 config 變異（6 個新 step、relinker、`pipeline_relink`、tail step 與 cfg 不一致、`latency_only`）都讓計畫失敗；`track.workbench` 也讓 post-detector 計畫失敗；loader 原本就拒絕其他 tail 步驟（PR-4a） |
| 接線後 7-seq（同機器、`build/`、branch commit `5578c9ee`、工作樹乾淨） | 4 個 stage 5316/5316 幀逐位元相同（與 PR-5 相同）；**7/7 sequence 的 MOT txt 重標後逐位元組相同**，track ID 數相同；MOT17-02（位移 0）不重標就 `cmp` 相同 |
| 單一 sequence | 只 dump MOT17-09：native txt 與 oracle txt 不重標就 `cmp` 相同 |
| 負控制 | oracle txt 改 1 個字元 ⇒ 只有 `mot_txt` 發散，位置指到該行；config 關掉 interpolation ⇒ `mot_txt` 發散（3761 對 4091 行），4 個 stage 不變 |
| oracle 自身的重現性 | 第二次執行（`--frames hash`）的每個 stream、frame hash 與每個 MOT txt 都和第一次相同 |

serial 參考的 headline 指標仍為 IDF1 78.3／MOTA 77.9／IDs 429（只記錄觀察值）。結果目錄：`results/465_pr6_mot/full7_5578c9ee/`（`MANIFEST.md`、`verify.json`、`replay_report.json`、`mot_native/`、`single_seq/`、`negctl/`；frames 不納入版本控制）。

### 10.4 dump 與 replay 的 hardening（PR-5 留下的）

- `saccade_replay`：每個 per-update stream（post_nms、tracker_in、tracker_out，GMC 有跑時還有 gmc）的 record 數都必須等於重放的 tracker update 數。原本只檢查 tracker_out。
- dump 工具：manifest 新增 `mot_reference`，記錄 oracle 的 `eval/<seq>.txt` 與 `_global_id_map.txt` 的 sha256。`--verify` 會逐 record 解析每個 stream，record 數要等於 `meta.json` 的數字，per-update stream 的幀序列要等於非空的 detector 幀，MOT 參考檔要對得上 hash；`--against` 也比對 MOT txt。PR-6 之前的 dump 沒有 `mot_reference`，`--verify` 會判 FAIL，replay 會要求重新 dump。

### 10.5 限制

- 驗的是 serial、eager 的 post-detector 路徑加上 tail；graph 與 double-buffer 在 U5，ingest 與 detector 在 PR-7／PR-8。
- 多 sequence 的 parity 是**重標後**的相同：要求 oracle 對每個 sequence 給出連續的 ID 區塊，位移取自 oracle 自己的 `_global_id_map.txt`。run-global ID 不進 shipping（boundary §5 B3）。
- 結果是 headline 組態下的觀察，不是一般性的等價主張。計畫拒絕的組態（任何非 fast emit 的 emit 路徑、interpolation 以外的 tail 步驟）沒有 native 實作。
- `saccade_replay` 不重算參考檔的 hash，完整性由 dump 工具的 `--verify` 負責（與 stream 檔相同）。

### 10.6 重現

```bash
.venv/bin/python scripts/model/render_shipping_mot_output_fixture.py --check
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --out <dir>/dump
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --out <dir>/repeat_hash --frames hash
.venv/bin/python scripts/eval/diagnostics/dump_post_detector_replay.py --verify <dir>/dump --against <dir>/repeat_hash --report <dir>/verify.json
cmake --build build --target saccade_replay
build/shipping/saccade_replay --config configs/shipping/mamba_whole_graph.resolved.json --dump <dir>/dump --report <dir>/replay_report.json --mot-out <dir>/mot_native
```

---

## 11. Native ingest（PR-7／U3b-1）

PR-7 在 shipping 端加上 ingest：nvJPEG 解碼＋normalize，產出 oracle 的 `pool.frame_buffer`（float32 `[3, imHeight, imWidth]`），也就是 PR-5 宿主 `PostDetectorHost::process` 吃的 `frame_chw`（同一個 layout；本 PR 不把兩者接起來，端到端接線在 PR-9）。行為只由 resolved config 驅動：config 決定 ingest 走哪條路，路徑上 oracle 寫死的常數（glob、RGB、255.0）由 oracle source pin 釘住。驗收依 boundary §6 PR-7：解碼像素對 torchvision **單獨**量，差異分開報告、不併入後段。backbone、head、S2 不在本 PR（PR-8）。

| 項目 | 位置 |
|:--|:--|
| 計畫＋fail-closed gate、normalize 的 host twin、sequence input（`seqinfo.ini`＋`img1/` 列檔）（CUDA-free） | `shipping/include/saccade_shipping/ingest_plan.hpp`、`shipping/src/ingest_plan.cpp`（併入 `saccade_shipping_native_config`） |
| GPU：nvJPEG 解碼器（torchvision 解碼器的 twin）、normalize kernel、per-sequence `IngestHost` | `shipping/include/saccade_shipping/ingest_host.hpp`、`shipping/src/ingest_host.cpp`、`shipping/src/ingest_normalize.cu`（target `saccade_shipping_ingest`，只在 root build） |
| probe 工具（`developer_build_debug`） | `shipping/tools/saccade_ingest_probe.cpp`（target `saccade_ingest_probe`）：獨立 process（不連 Python／torch），把一個 sequence 的解碼 bytes 與 frame buffer 串流到 stdout（`saccade.native_ingest_stream/v1`） |
| parity harness（`developer_build_debug`） | `scripts/eval/diagnostics/native_ingest_parity.py`（`saccade.native_ingest_parity/v1`） |
| 測試 | `tests/native/test_shipping_ingest_plan.cpp`（CPU，CI `shipping-config-loader`）、`tests/native/test_shipping_ingest_host.cpp`（GPU）；fixture `tests/native/fixtures/shipping_ingest.json`＋`shipping_ingest/*.jpg` 由 `scripts/model/render_shipping_ingest_fixture.py` 以 oracle 自己的程式產生；`tests/unit/test_native_ingest_oracle_pins.py`（oracle source pin＋fixture freshness） |

### 11.1 照抄的 oracle 事實

- **列檔**：`TorchvisionGpuStreamer` 列出 `sorted(str(p.absolute()) for p in (seq / "img1").glob("*.jpg"))`。pathlib 的 glob 會列出點開頭的檔名、名稱就是 `.jpg` 的項目、目錄與 symlink（含斷掉的），大小寫敏感；排序是 Python 字串序（ASCII 名稱等於位元組序）。第 `k` 幀是第 `k` 個列出的項目，不看檔名裡的數字。
- **幀數**：`frame_end = min(max_frames or int(1e9), seqLength)`。列出的項目比 `frame_end` 少時，oracle **不會失敗**：兩個 frame loop 遇到 `StopIteration` 就停（serial 的 `_run_frame` 回傳 `False`，double-buffer 的 `_schedule` 回傳 `None`），寫出一份被截短的 sequence。多出的項目不被讀取。
- **解碼**：`decode_jpeg(read_file(f), device="cuda", mode=ImageReadMode.RGB)`，torchvision 0.26.0 的 `CUDAJpegDecoder`：先以 `NVJPEG_BACKEND_HARDWARE` 建 handle（`ARCH_MISMATCH` 時改 `DEFAULT`、不做硬體解碼）；`nvjpegGetImageInfo` 取第 0 個 component 的尺寸，輸出 planar RGB uint8 `[3, H, W]`、pitch `W`；硬體可用且 `nvjpegDecodeBatchedSupported` 說可以（status 不檢查）就走 `nvjpegDecodeBatched`（batch 1、1 個 CPU thread），其他走 decoupled 的 host／transfer／device 三段；解碼器自己的 non-blocking stream，前後各 synchronize 一次。native 解碼器逐呼叫照做。
- **ingest op**：`_run_detect` 的非 NV12 分支是 `torch.div(frame_gpu.permute(2, 0, 1), 255.0, out=pool.frame_buffer)`，buffer 由 `AdaptiveFramePool(h_orig, w_orig)` 以 `seqinfo.ini` 的尺寸、`torch.zeros` 配置；接著的 `apply_frame_preprocess` 在 mode 清單為空時直接 return。torch 的 CUDA kernel 把「除以 Python scalar」算成乘以 float32 倒數，所以值是 `float(x) * (1.0f / 255.0f)`。**這與 `x / 255.0f` 在 256 個輸入裡有 126 個不同（各差 1 ulp）**；CPU 上的 torch 用的是真正的除法，所以 fixture 的 normalize 表只能在 CUDA 上產生。
- **`seqinfo.ini`**：`configparser` 讀 `[Sequence]` 的 `imWidth`／`imHeight`／`seqLength`：key 大小寫不敏感、`=` 或 `:`、`#`／`;` 開頭的行是註解、值前後空白（含 CR）去掉、`getint` 即 `int()`。

### 11.2 由 resolved config 驅動的 gate

`plan_ingest` 只讀 config，下列任一不成立就 fail-closed（`ConfigError`）。沒有新增 exporter step：前三條本來就在 `steps`，`preprocess_modes` 是 `cfg` 的值。

| 條件 | 不成立時 oracle 會做什麼 |
|:--|:--|
| `steps.ingest.gpu_decode == true` | 改用 `DALIStreamerStream`（CPU 解碼，另一個解碼器） |
| `steps.ingest.nv12_buffer == false` | 轉成 NV12；沒有 preprocess mode 時直接跳過 float32 frame buffer |
| `cfg.preprocess_modes == []` | gamma／contrast 改寫 frame buffer，letterbox 改變 detector 的輸入 |
| `steps.track.workbench == false` | workbench 有自己的 ingest（`evaluator.py`，`.float() / 255.0`） |

### 11.3 nvJPEG：與 oracle 同一份 binary

oracle 用的是 torchvision wheel 內附的 `libnvjpeg`（13.0.1）；它與 PyPI `nvidia-nvjpeg==13.0.1.86` wheel 裡的 `libnvjpeg.so.13` 逐位元組相同（sha256 `1a359ba7…`），該 wheel 也附有對應的 `nvjpeg.h`。`shipping/CMakeLists.txt` 以 `FetchContent` 釘住這個 wheel（URL＋sha256，與 TensorRT header 的做法相同）並連結它；configure 時比對它與 `torchvision.libs/libnvjpeg*.so*` 的 sha256，不同就 `FATAL_ERROR`——換一份 nvJPEG 就是換一個解碼器，parity 要重量。沒有放進 `native-build` extra：那個 extra 依 contract 只放 build toolchain（`tests/contract/test_package_native_delivery.py`），而 nvJPEG 是 shipping 的 runtime 函式庫；shipping 是否 bundle 它屬 Phase C／owner 決定，本 PR 不決定。系統 `/opt/cuda` 的 nvJPEG（13.2.3）不使用（#214）。

### 11.4 驗收

同一台機器（RTX 5070 Ti Laptop，nvJPEG 硬體解碼可用）、`build/` 組態、branch commit `2160f957`（工作樹乾淨），MOT17 train 7 個 SDP sequence（5316 幀；6 個 1920×1080、MOT17-05 640×480）：

| 驗收項 | 結果 |
|:--|:--|
| 解碼（native uint8 `[3, H, W]` 對 oracle 解碼結果） | **EXACT**：5316/5316 幀逐位元組相同；每一幀都走 hardware batched 路徑 |
| normalize（native frame buffer 對「oracle op 套在 native 解碼 bytes 上」，與解碼分開） | **EXACT**：5316/5316 幀逐位元相同；kernel 對 0..255 的輸出與 oracle op（連續與 permute 輸入兩種）256/256 相同 |
| 端到端 ingest（native frame buffer 對 oracle frame buffer） | **EXACT**：5316/5316 幀逐位元相同（上兩項的結果，不另作歸因） |
| 列檔與尺寸 | 7/7 sequence 的列檔、consumed frames、`imWidth`／`imHeight` 與 oracle 相同 |
| 重現性 | 第二次執行（`--against`）兩邊每幀解碼 bytes 與 frame buffer 的 sha256 全部相同 |
| 對真實 eval run 的 ingest（`--replay-dump`，commit `62db1356`） | PR-6 的 dump（`results/465_pr6_mot/full7_5578c9ee/dump`）存的是 `mot17.py` serial run 交給 PR-5 宿主的 GMC 輸入幀，也就是該 run 的 `pool.frame_buffer`（以 uint8 無損存放）。native 解碼 bytes 與它、native frame buffer 與「oracle op 套在它上面」：5316/5316 幀**EXACT**。所以 harness 重建的 oracle ingest 與 eval run 實際的 ingest 相同 |
| 解碼器是同一份 | probe 載入的 `libnvjpeg.so.13` 與 torchvision 內附的 sha256 相同；probe 的 NEEDED 沒有 libpython／libtorch |
| 負控制：強制 decoupled 路徑 | decoder **DIFFERS**（5316/5316 幀；28.6 G 個值中 339 M 個不同＝1.18%，|Δ| 最大 3，|Δ|=1／2／3 各 298 M／40.5 M／0.24 M）；normalize 仍 EXACT——差異只出現在 decoder 一節 |
| 負控制：kernel 改成 `x / 255.0f`（MOT17-09） | decoder 仍 EXACT；normalize **DIFFERS**（525/525 幀，最大 1 ulp），normalize 表 126/256 不同——差異只出現在 normalize 一節 |
| 實作變異（fixture 測試） | CPU 測試抓到 21/21（gate、列檔規則、排序、截短、`seqinfo.ini` 讀法、檔案檢查、normalize 用除法）；GPU 測試抓到 8/8（除法、grid-stride、永不走／永遠走硬體路徑、BGR、pitch、尺寸檢查）。過程中有 3 個變異沒被抓到，查出是死規則（CR、`[DEFAULT]`、`%`），已刪除並改寫說明 |
| fixture（oracle 自己產生） | 6 張 JPEG（4:2:0／4:2:2／4:4:4、奇數尺寸、progressive、灰階）：native 解碼與 torchvision 逐位元組相同，兩條路徑都有涵蓋（progressive 走 decoupled）；18 個列檔／`seqinfo.ini` 案例 |

結果目錄：`results/465_pr7_ingest/full7_2160f957/`（`MANIFEST.md`、`run.sh`、四次執行的 `report.json`／`frames.jsonl`／log）與 `results/465_pr7_ingest/replay_dump_62db1356/`；不納入版本控制。

### 11.5 量到的事

- **nvJPEG 的兩條路徑給出不同的像素。** 把同一批 MOT17 幀全部改走 decoupled 路徑（probe 的 `--force-decoupled`，只供量測），每一幀都與 torchvision 不同（1.18% 的值、|Δ| ≤ 3，§11.4）。所以「硬體可用時先走 batched」這條選擇規則本身決定像素，native 必須照抄；也代表 ingest 的輸出依 GPU 是否有硬體 JPEG 解碼器而不同——parity 是同機器的對照，換一類 GPU 時兩邊會一起換路徑，但像素不會與這台機器相同。
- **CPU 與 CUDA 的 torch 對同一個 ingest op 給出不同的 float32**（§11.1）。任何以 CPU 重算 frame buffer 的工具都不等於 oracle。
- **oracle 會把列檔不足的 sequence 截短而不報錯**（§11.1）。shipping 拒絕這種輸入（`InputError`），不重現截短；這是比 oracle 嚴格的地方，fixture 以 `"native": "refuse"` 標出。

### 11.6 限制

- 只驗 ingest：解碼 bytes、frame buffer。backbone、head、S2、detection tensor 在 PR-8；ingest→detect→post→tail 的端到端在 PR-9。
- serial、eager、每幀同步；graph、double-buffer 與 decode prefetch 在 U5。harness 的速度不是效能主張。
- 結果是這台機器（RTX 5070 Ti Laptop、硬體解碼可用）、torchvision 0.26.0、nvJPEG 13.0.1 下的觀察，不是一般性的等價主張；沒有硬體解碼器的 GPU 上沒有量過（兩邊都會走 decoupled）。
- native 對 `seqinfo.ini` 與列檔比 oracle 嚴格：縮排行、`1_920` 這類整數、只在 `[DEFAULT]` 的 key、非 ASCII 檔名、非正的尺寸、列檔不足都拒絕（fixture 的 refuse 案例）。在這些輸入上 native 不重現 oracle，而是拒絕。
- 解碼尺寸與 `seqinfo.ini` 不同時 native 拒絕；oracle 在這種情況下會以 `out=` 重新配置 frame buffer，headline 資料沒有這種幀。

### 11.7 重現

```bash
.venv/bin/python scripts/model/render_shipping_ingest_fixture.py --check
cmake --build build --target saccade_ingest_probe saccade_shipping_ingest_host_test
build/shipping/saccade_shipping_ingest_host_test configs/shipping/mamba_whole_graph.resolved.json tests/native/fixtures/shipping_ingest.json tests/native/fixtures/shipping_ingest
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/native_ingest_parity.py --out <dir>/parity
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/native_ingest_parity.py --out <dir>/repeat --against <dir>/parity/report.json
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/native_ingest_parity.py --out <dir>/negctl_decoupled --force-decoupled
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/eval/diagnostics/native_ingest_parity.py --out <dir>/replay_dump --replay-dump <PR-6 dump>
```

---

## 12. Native detector（PR-8／U3b-2）

PR-8 在 shipping 端加上 detector：從 PR-7 的 `frame_chw`（float32 `[3, H, W]`）經 640×640 bilinear resize、既有的 C++ `TRTEngine` backbone、PR-1L 的 LibTorch TorchScript head、S2 fixed postprocess，產出原圖座標的 detector rows，也就是 PR-5 `detector.bin` 的語義邊界（float32 boxes `[n, 4]`、float32 scores `[n]`、int32 classes `[n]`、`is_tiled = false`、沒有 keypoints）。oracle 是 owner 接受的 `A_L`（PR-2L：headline 組態，head 換成 PR-1L artifact），不是 headline 的 compiled head；serial、eager。main NMS、private continuation、external FP／FP hard filter、GMC、tracker、MOT emit、CUDA graph、double-buffer 與效能都不在本 PR（PR-5／6／9／10）。

| 項目 | 位置 |
|:--|:--|
| 計畫＋fail-closed gate（resolved config、凍結的 PR-1L lineage、realization attestation）（CUDA-free） | `shipping/include/saccade_shipping/detector_plan.hpp`、`shipping/src/detector_plan.cpp`；SHA-256：`sha256.hpp`／`sha256.cpp`（併入 `saccade_shipping_native_config`） |
| GPU：S2 kernel（Inductor lowering 的 twin） | `shipping/include/saccade_shipping/detector_s2.hpp`、`shipping/src/detector_s2.cu` |
| GPU：detector 宿主（head loader、backbone、resize、topk） | `shipping/include/saccade_shipping/detector_host.hpp`、`shipping/src/detector_host.cpp`（target `saccade_shipping_detector`，只在 root build；連 LibTorch 與 `saccade_perception` 的 `TRTEngine`，不連 `libtorch_python`） |
| realization attestation | `configs/shipping/mamba_head_realization.attestation.json`（commit 進 repo，§12.3） |
| probe 工具（`developer_build_debug`） | `shipping/tools/saccade_detector_probe.cpp`（target `saccade_detector_probe`）：獨立 process，跑 PR-7 ingest＋本 PR detector，把每個 stage 的輸出串流到 stdout（`saccade.native_detector_stream/v1`） |
| parity harness（`developer_build_debug`） | `scripts/eval/diagnostics/native_detector_parity.py`（`anchor`／`attest`／`oracle-rows`／`parity`） |

### 12.1 照抄的 oracle 事實

- **resize**：`_whole_graph_fn` 的 `F.interpolate(frame, size=(img_size, img_size), mode="bilinear", align_corners=False)`。native 直接呼叫 ATen 的 `upsample_bilinear2d`（`output_size` 給定、`scale_factors` 為空，即 `F.interpolate` dispatch 到的同一個 kernel、同一份 `libtorch_cuda`），不是另寫一個 bilinear kernel。
- **backbone**：`TRTYoloBackbone.infer_graph`：engine 的第 0 個 tensor 是輸入 `[1, 3, img, img]`，其餘依序為 p3／p4／p5；在呼叫端 stream 上 `execute_async_v3`。native 用既有的 `TRTEngine`（`src/perception/trt_engine.cpp`），連結 oracle 的 Python `tensorrt` 所用的同一份 `tensorrt_libs/libnvinfer`。
- **head**：PR-2L 的 `A_L` 把 PR-1L artifact 放進 `_trt_head` slot（`LibTorchHeadAdapter`：不複製、不改 dtype、不做運算，在目前 stream 上呼叫 module）。native loader：先驗 hash、`dlopen` operator library（註冊 `saccade_native::selective_scan_fwd`）、照 lineage 設定並讀回 runtime requirements（graph executor optimize 關、cuDNN benchmark 關、cuDNN TF32 開、matmul TF32 關）、`torch::jit::load(path)` 不給 device map（PR-2L amendment A1：參數在 cuda:0、tensor constant 留在 CPU），再檢查 inlined graph（無 `prim::PythonOp`、不呼叫 `saccade::selective_scan_fwd`、`saccade_native::selective_scan_fwd` 次數 == lineage）；process 內不得 map 任何 `libpython*`／`libtorch_python*`。
- **S2**：oracle 預設 `torch.compile` `_postprocess_mamba_fixed_eager`（mode `default`）。以 headline 形狀，torch 2.11.0+cu130／triton 3.6.0 的 Inductor 把它降成：(1) 對每個 anchor 的 80 個 class 做一個 reduction kernel，sigmoid 是 `tl.sigmoid`，PTX 為 `sub.f32 t, 0, x` → `mul.f32 t, t, 0f3FB8AA3B` → `ex2.approx.f32` → `add.f32 1.0` → `div.full.f32 1.0, d`；value 用 `triton_helpers.maximum`（NaN 傳遞）、index 用 `maximum_with_index`（NaN 視為最大、同值取較小 index），兩者都與 reduction 順序無關，所以結果不受 autotune 的 block 形狀影響（快取裡的每個 autotune 變體都是同一串指令）；(2) `aten.topk(scores_max, max_det)`（ATen fallback，不是 Triton）；(3) box 的 pointwise kernel：`x1y1 = a − lt`、`x2y2 = a + rb`、`c = (x1y1 + x2y2) × 0.5`、`wh = x2y2 − x1y1`，再 `(c × s) ∓ (wh × s) × 0.5`。Inductor 的 SASS 把 `c × s` 合進 FFMA，但 stride 是 2 的冪，這些乘積都是精確值，所以結果等於逐一捨入的 float32 運算（也等於 eager）；(4) gather 成 `(1, max_det, 6)`：box、top-k score、`float(class index)`。這是**為什麼 eager S2 不是 oracle**：同一組 head 輸出，eager 的 `sigmoid` 與 compiled 的在 score 欄有 1–3 ulp 的差異（合成輸入上每次都看得到）；box 欄相同。native S2：(1) 以 inline PTX 寫出同一串指令與同一組比較規則；(2) 呼叫 ATen `topk`（oracle 自己的 kernel）；(3)(4) 只對選中的 anchor 以明確的 round-to-nearest float32 運算計算（不 contraction）。
- **座標縮放與 rows**：`_whole_graph_fn` 之後 `detections[:, :, [0, 2]] *= sx`、`[:, :, [1, 3]] *= sy`，`sx = float32(w / img_size)`（Python double 除法後存入 float32 tensor，`set_whole_graph_img_dims`）；`detect_single_patch_640` 的 whole-graph 分支（無 letterbox、無 NV12）原樣回傳 `raw[0, :, :4]`、`raw[0, :, 4]`、`raw[0, :, 5]`；`_run_native_tensor_prep`（以及 PR-5 的 dump）把 class 轉成 int32。rows 數 = `max(max_det, _whole_graph_nms_pad)`，pad 恆為 0，所以 = `max_det` = 300；8400 個 anchor 一定填滿 top-300，**沒有 padding 列**（`new_zeros` 的初值全部被覆寫）。`conf_thr` 傳進 fixed S2 但不被讀取。
- **固定常數**（oracle 原始碼字面值，由 `tests/unit/test_native_detector_oracle_pins.py` 釘住）：stride `[8, 16, 32]`、anchor 偏移 `+0.5`、`_whole_graph_nms_pad = 0`、`_whole_graph_fn` 的 stage 順序與縮放、S2 的 op 序列、`_POSTPROCESS_COMPILE_ENABLED` 預設與 `torch.compile(mode="default")`。

### 12.2 由 resolved config 與 lineage 驅動的 gate

`plan_detector` 只讀 resolved config、凍結的 PR-1L lineage 與 realization attestation；任一不成立就 fail-closed（`ConfigError`），沒有替代 preset、沒有預設值。

| 條件 | 不成立時 oracle 會做什麼／為什麼拒絕 |
|:--|:--|
| `build.use_whole_graph == true`、`build.trt_head_engine == ""` | 走非 whole-graph 路徑，或 head slot 放的是 TRT head |
| `build.small_p3_max_threshold == 0.0` | S2 多出 `_fuse_small_p3_scores` |
| `build.postprocess_compile == true` | eager S2：sigmoid 數值不同（§12.1） |
| `head_calls == {set_head_compile: [true], set_block_compile: [true]}` | PR-2L 對 `A_L` 的接受只涵蓋 headline 組態 |
| `detect_fn == detect_native_640`、`cfg.tiling == kwargs.tiling == native_640`、`cfg.tta == false` | 其他 detect 函式（tiling、960、TTA） |
| `cfg.preprocess_modes == []`、`steps.ingest.nv12_buffer == false`、`steps.track.workbench == false` | letterbox／NV12 的 preprocessed 路徑、workbench 自己的 detector 呼叫 |
| `contract.feature_dim == 0`、`fpn_reid_mode == false`、box format `xyxy`（contract 與 `cfg.detector_box_format`） | ReID 特徵、cxcywh 解碼 |
| `img_size` 是 32 的正倍數、`max_det > 0` | — |
| lineage：schema、`tool.git_dirty == false`、preset path／sha256 == config 的 `source`、`mamba_ckpt`／`fpn_backbone_engine`／builder inputs 的路徑 == config 的 `build`、inventory 對上、`mamba_args.use_detail_fusion == false`、`reg_max` 缺省或 1、head 載入無 missing／unexpected、torchscript 輸入形狀 == `[1, c, img/stride, img/stride]`、輸出名稱、float32／static batch 1、`structural_check.bitwise_equal_all`、op library 不連 Python、`graph_executor_optimize == false` | lineage 描述的不是這份 config 的 model，或 artifact 不是 native 能照抄的形式 |

被載入的檔案：backbone engine 的 sha256 取自 lineage `companions.backbone_engine`、artifact 取自 `torchscript.sha256`、operator library 取自 lineage 或 attestation（§12.3）。三者在任何載入之前比對。

### 12.3 operator library 的 realization attestation（PR-1L freeze 不動）

PR-1L lineage 記錄的 operator library build（sha256 `cfea782f…`）已不存在：`build/libsaccade_scan_torchop.so` 之後從同一份凍結的 source blob 重新 build（現在是 `098dd233…`），舊的 binary 沒有保留。lineage 沒有記錄 compiler identity，所以兩次 build 是否用同一個 compiler 無法從凍結紀錄證明；目前這份 build 的 compiler metadata（`.comment`）記錄在 realization attestation 裡。lineage 自己寫明這個 sha256「只識別這次 build」。依 owner 決定（10-03），**PR-1L 的凍結內容與 lineage 檔案都不改**；PR-8 另外為目前的 shipping build 建一層 realization attestation：

- 以 sha256 綁定凍結的 lineage 檔案，並逐項重述它的 torchscript sha256／content sha256 與它所記的 operator library sha256；
- 記錄這次 build 的路徑、sha256、bytes、`DT_NEEDED`（不得含 `libpython*`／`libtorch_python*`）、compiler `.comment`，以及三個 source（`mamba_scan_torchop.cpp`、`mamba_scan.cu`、`mamba_scan.cuh`）的 git blob，且必須等於 lineage tool commit 上的 blob；
- 記錄 `A_L` reproduction：以 PR-2L runner 自己的 `run_arm_child`（只把它的 operator library sha256 換成這次 build）重跑 `A_L`（double-buffer、7 個 sequence），每個 sequence 的 MOT txt 必須與 PR-2L 正式 packet 的 `A_L_1` 逐位元組相同，PR-2L 的 V5 sidecar 檢查與 env override 檢查都要通過。`attest` 只在 anchor 報告 identical、工作樹乾淨時才寫檔。

native loader 只接受：attestation 的 lineage sha256 == 實際 lineage 檔案、它重述的值 == lineage、operator library 檔案的 sha256 == attestation 記錄的 build；沒有給 attestation 時要求等於 lineage 記錄的 build（目前會失敗）。harness 的 oracle 端也綁同一份 attestation。

### 12.4 測量契約（正式 7-seq run 之前寫定）

**oracle**：`A_L` detector path，即 PR-2L runner 的 `A_L` 注入（op library 依 §12.3），headline preset、`--detector SDP`、MOT17 train 7 個 SDP sequence（02／04／05／09／10／11／13，共 5316 幀）。

**stage 分開，各自吃 native 的上一段輸出**（harness `parity`；native 端是 probe 的一次完整串接 run）：

| section | native | oracle（輸入是 native 的上一段） | 比較 |
|:--|:--|:--|:--|
| `decoder` | PR-7 解碼 uint8 `[3, H, W]` | torchvision 解碼 | 逐位元組 |
| `ingest` | PR-7 frame buffer | oracle ingest op 套在 torchvision 解碼上 | 逐位元 |
| `resize` | `[1, 3, 640, 640]` | `F.interpolate(native frame)` | 逐位元 |
| `backbone` | p3／p4／p5 | `TRTYoloBackbone.infer_graph(native resized)` | 逐位元 |
| `head` | 6 個輸出 | `A_L` head（PR-2L adapter＋PR-1L module）吃 native p3／p4／p5 | 逐位元（owner 接受 PR-2L 的條件：native loader 必須對 `A_L` byte／bit identical） |
| `s2_raw`／`s2_scaled` | S2 `[300, 6]`（縮放前／後） | compiled `_postprocess_mamba_fixed` 吃 native 的 6 個 head 輸出，再照 `_whole_graph_fn` 縮放 | 逐位元 |
| `s2_rows` | detector rows | 由上一列 oracle 輸出照 `detect_single_patch_640`／`_run_native_tensor_prep` 切成 rows | 見下 |
| `detector` | detector rows（native 從 JPEG 一路串接） | `oracle-rows`：`A_L` serial run（無 `--double-buffer`）的 `evaluator._run_detect` 輸出，PR-5 `detector.bin` 格式 | 見下 |

**比較 schema**：tensor section 每幀報 `equal`；不相等時報 differing values、`max_abs`（只算有限值）、`max_ulp`（float32 位元序的距離）、非有限的差異數、形狀不符；彙總報 equal frames／frames、合計 differing values、最大 `max_abs`／`max_ulp`、前 20 個發散幀（sequence、frame）。row section 每幀**分開**報：rows 數（native, oracle）、count 相等、membership（每列 6 個值的位元型樣作為 multiset）相等、order（同位置的列）相等、class id 相等、score 位元相等、box 位元相等（含各自的 differing values、`max_abs`、`max_ulp`）、第一個不同的列、全零列數（padding 語義）；彙總為各項不同的幀數與前 20 個發散幀。每幀另記 native 各 stage 與 oracle head／S2／rows 的 sha256（`frames.jsonl`），供 `--against` 做 run-to-run 比對。

**EXACT 驗收規則**：PR-8 的 verdict 是 `EXACT` 若且唯若下列全部成立，否則是 `NOT_EXACT`：

1. 有效性（任一不成立 ⇒ `UNRESOLVED`，不判讀任何 section）：正式 run 在乾淨的 commit 上、gpu0 lease 下執行；attestation 對上 lineage 與 operator library 檔案；anchor（§12.3）identical；`oracle-rows` 報告 ok（V5 sidecar 通過、serial、每個 sequence 錄到 1..frame_end 全部幀、`is_tiled` 全 false）；probe preamble 通過（`python_libraries_mapped == []`、載入的 op library／artifact／engine sha256、runtime readback == lineage、scan 呼叫次數、參數 cuda:0／constant CPU）；每個 sequence 的列檔、幀數與幾何 native == oracle；
2. 9 個 section（`decoder`、`ingest`、`resize`、`backbone`、`head`、`s2_raw`、`s2_scaled`、`s2_rows`、`detector`）在 7 個 sequence、5316 幀上**全部**逐位元相同；
3. 重現性：第二次 `parity --against` 的每幀 sha256（native 與 oracle 兩側）與第一次全部相同。

沒有容差，看到結果之後也不新增容差。任一 section 不是 EXACT：照上面的 schema 分開報告（`max_abs`、`max_ulp`、differing values、第一個發散幀、rows 數／membership／order／class 的變化），**停在 PR-8**，不把差異帶進 PR-9 的 MOT 行為。

**負控制**（MOT17-09，525 幀，對同一份 `oracle-rows`）：probe 的 `--mutation` 每次只改一段，對應的 section 必須是 `DIFFERS`（其他 section 照實記錄）：`backbone_ulp`（p3 第 0 個值 +1 ulp）→ `backbone`；`head_ulp`（cls_p3 第 0 個值 +1 ulp）→ `head`；`s2_threshold`（多加一道 score < 0.05 的門檻）、`s2_topk`（top-299）、`s2_order`（第 0、1 列互換）→ `s2_rows`；`box_ulp`（縮放後第 0 列 x1 +1 ulp）→ `s2_rows`。另以 GPU fixture 測試覆蓋 S2 的邊界值（§12.5）。

### 12.5 驗收

同一台機器（RTX 5070 Ti Laptop）、`build/` 組態、branch commit `a9d657c1`（工作樹乾淨；§12.4 的契約在 `ad234e3d` 就已 commit，早於任何 MOT17 上的 parity 量測），全部步驟在 gpu0 lease 下依序執行（`run.sh`）：

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | anchor：7/7 sequence 的 MOT txt 與 PR-2L `A_L_1` 逐位元組相同，V5 sidecar 與 env override 檢查通過；`oracle-rows`：serial、5316 筆、`is_tiled` 全 false、V5 通過；probe preamble：未 map 任何 Python 函式庫、載入的 op library 是 attestation 的 build（`098dd233…`）、artifact／engine 的 sha256 與 lineage 相同、runtime readback == lineage、scan 呼叫 3 次、參數 cuda:0／constant CPU；7/7 sequence 列檔、幀數、幾何相同；所有解碼都走 hardware batched 路徑 |
| 9 個 section（7 sequence、5316 幀） | **全部 EXACT**：`decoder`、`ingest`、`resize`、`backbone`、`head`、`s2_raw`、`s2_scaled`、`s2_rows`、`detector` 每一幀都逐位元相同（rows：300 列、membership、order、class、score 與 box 位元皆同，無全零列） |
| 重現性 | 第二次 `parity --against`：再次全部 EXACT，且 native 與 oracle 兩側每幀的 sha256 5316/5316 相同 |
| **verdict** | **`EXACT`**（§12.4 第 1–3 條全部成立） |

`head` 是 owner 接受 PR-2L 時要求的 native loader 對 `A_L` 的 bit identity；`detector` 比對的是真正的 `mot17.py` serial run（graph capture 之後的 `A_L` 輸出），與 harness 內 eager 呼叫 oracle 函式得到的 `s2_rows` 也一致。

**負控制**（MOT17-09，525 幀，同一份 `oracle-rows`）全部被抓到，且只在被改的那一段與其下游出現：

| mutation | DIFFERS 的 section（其餘 EXACT） |
|:--|:--|
| `backbone_ulp` | `backbone`（525/525 幀，每幀 1 個值、1 ulp）；oracle head 吃的是 native 特徵，所以 `head` 仍 EXACT；這個 1-ulp 也沒有改變任何一幀的 rows |
| `head_ulp` | `head`（525/525 幀，1 個值、1 ulp）；S2 與 rows 不受影響 |
| `s2_threshold` | `s2_raw`、`s2_scaled`、`s2_rows`、`detector`（rows 數不同，例如前 20 幀只剩 58–78 列，對 300 列） |
| `s2_topk` | 同上（299 對 300 列） |
| `s2_order` | 同上（count、membership、class 相同；order、score 與 box 位元不同） |
| `box_ulp` | `s2_scaled`、`s2_rows`、`detector`（每幀 1 個 box 值、1 ulp）；`s2_raw` 仍 EXACT |

結果目錄：`results/465_pr8_detector/full7_a9d657c1/`（`MANIFEST.md`、`run.sh`、`anchor/`、`oracle_rows/`、`parity/`、`repeat/`、`negctl_*/` 與 log；不納入版本控制）。

### 12.6 實作驗證（fixture 與變異）

| 驗收項 | 結果 |
|:--|:--|
| CPU 計畫測試（CI `shipping-config-loader`，`test_shipping_detector_plan.cpp`） | SHA-256 對 FIPS 180-4 向量（含每個 block 邊界的分段 update）；headline config＋lineage＋attestation 得到預期的每個值；23 個 config gate 翻轉、22 個 lineage 欄位、9 個 attestation 欄位改動全部被拒；lineage 位元組一改，attestation 就拒絕；沒有 attestation 時 operator library 取 lineage 記錄的 build |
| S2 golden fixture（`shipping_detector_s2.json`，oracle 自己的 compiled S2 產生） | 13 個 case（隨機 logits、box distance 用 2⁻¹⁶ 網格與完整 mantissa 兩種、飽和 logits（class argmax 與 top-k 大量同值）、粗整數 logits、極負 logits（subnormal／0 的 score、`div.full` 的縮放分支）、四段單 class logits 掃描、NaN／±inf、三種 sequence 幾何（精確與不精確的縮放））：native S2 縮放前後皆逐位元相同；其中 10 個 case 的 eager S2 與 compiled 不同，所以 fixture 分得出兩者；renderer 重跑 `--check` 結果相同 |
| S2 變異（GPU fixture 測試） | 15 個變異抓到 12 個：`div.rn`、eager sigmoid、class 同值取較大 index、NaN 視為最小、value 不傳遞 NaN、中心用 `a + (rb − lt)/2`、寬度用 `l + r`、x 用 sy 縮放、少了 anchor 偏移、半寬多一次捨入、LTRB 通道錯位、unsorted topk。沒抓到的 3 個是等價變異：`ex2.approx.ftz`（只在 ex2 的輸入或輸出為 subnormal 時不同，前者結果都是 1、後者被 `1 + e` 吸收）、中心改成 `x1·0.5 + x2·0.5`（乘 0.5 與 round-to-nearest 可交換，非 subnormal 時相等）、以 stable sort 取代 topk（在所有 fixture case，包括 8400 個 anchor 同值的飽和 case，都與 `aten.topk` 同序；native 呼叫的就是 oracle 的 `aten.topk`）。第一輪的 fixture 只有 2⁻¹⁶ 網格的 box distance，box 運算都沒有捨入，抓不到中心與寬度的代數變形；加上完整 mantissa 的 case 之後才抓到 |
| loader fail-closed（GPU 測試，model 檔存在時） | operator library／artifact／backbone engine 任一 sha256 錯誤都在載入前拒絕（之後 operator library 沒有被 map）；正確的 plan 載入後：參數 cuda:0、tensor constant CPU、scan 呼叫 3 次、runtime readback 正確、engine I/O 4 個、process 內無 Python 函式庫 |
| oracle source pin（`tests/unit/test_native_detector_oracle_pins.py`） | `_whole_graph_fn` 的每個 statement、stride／anchor 偏移／x、y index、`_whole_graph_nms_pad` 在 `src/`、`scripts/` 只有 `= 0` 一處、`_postprocess_mamba_fixed_eager` 的 AST hash 與 compile 預設、`detect_single_patch_640` 的 whole-graph 分支、torch／triton 版本、harness 的 `oracle_s2`／resize、lineage fixture == 凍結 lineage、attestation 的綁定與 source blob、S2 fixture 新鮮度（CUDA） |
| probe 的連結 | `DT_NEEDED` 與 `ldd` closure 無 `libpython*`／`libtorch_python*`；執行時再由 `/proc/self/maps` 檢查一次 |

### 12.7 限制

- 只驗 serial、eager 的 detector；CUDA graph、double-buffer、decode prefetch 與 event barrier 在 U5（PR-10）。probe 每幀同步並把每個 stage 拷回 host，速度不是效能主張。
- 驗的是這台機器（RTX 5070 Ti Laptop、torch 2.11.0+cu130、triton 3.6.0、TensorRT 10.16、driver 616.92）上、headline 組態下的觀察，不是一般性的等價主張。S2 的 native twin 是照這一版 Inductor 的 lowering 寫的：torch／triton 升級時要重讀 lowering、重產 fixture、重量（pin test 會擋）。
- resize 與 topk 用 ATen（LibTorch C++）的 kernel：與 oracle 同一份 `libtorch_cuda`，所以這兩步的等價是「同一個 kernel」，不是重新實作後量到的。
- operator library 是 per-machine build；這台機器上綁定的方式是 §12.3 的 realization attestation。PR-1L 的 lineage 與 PR-2L 的凍結都沒改。換一台機器或重 build，就要重做 anchor 與 attestation。
- `saccade_shipping_detector` 連 `saccade_perception`（為了既有的 `TRTEngine`），所以 probe 的 `DT_NEEDED` 含 OpenCV；與 detector 無關，拆掉是 PR-11 的範圍。
- harness 的 oracle 端在 harness process 內呼叫 oracle 的函式（`F.interpolate`、`TRTYoloBackbone`、PR-2L adapter、compiled S2），不經過 CUDA graph；`detector` section 則比對真正的 `mot17.py` serial run（graph capture 之後的輸出），兩者合起來才涵蓋 oracle 的實際執行方式。

### 12.8 重現

```bash
cmake --build build --target saccade_detector_probe saccade_shipping_detector_s2_test
.venv/bin/python tools/resctl.py run gpu0 -- .venv/bin/python scripts/model/render_shipping_detector_s2_fixture.py --check
build/shipping/saccade_shipping_detector_s2_test tests/native/fixtures/shipping_detector_s2.json \
    configs/shipping/mamba_whole_graph.resolved.json tests/native/fixtures/shipping_head_lineage.json \
    configs/shipping/mamba_head_realization.attestation.json .
bash results/465_pr8_detector/<label>/run.sh   # anchor, oracle-rows, parity, repeat --against, 6 negctls
```

---

## 13. 端到端 serial native（PR-9／U3b-3）

PR-9 把已分別量過的四段（PR-7 ingest、PR-8 detector、PR-5 post-detector 宿主、PR-6 MOT emit 與 tail）接成 boundary §2 的 shipping entrypoint `saccade_track`，以 serial 組態執行：每個 sequence 目錄（`seqinfo.ini`＋`img1/*.jpg`）產出一份 `<sequence>.txt`。PR-9 不新增任何 stage，新的東西只有接線：哪些物件整個 run 共用、哪些每個 sequence 重建、buffer 由誰擁有、何時可以被覆寫。oracle 是 owner 接受的 `A_L` 的 serial 組態（`mot17.py` 不加 `--double-buffer`，head 換成 PR-1L artifact；#465 決策 1）。graph capture、double-buffer 與效能在 PR-10。

| 項目 | 位置 |
|:--|:--|
| 端到端 runtime | `shipping/include/saccade_shipping/serial_runtime.hpp`、`shipping/src/serial_runtime.cpp`（target `saccade_shipping_runtime`，只在 root build） |
| shipping entrypoint | `shipping/tools/saccade_track.cpp`（target `saccade_track`；`shipping_runtime`）。`--report`／`--trace`／`--max-frames`／`--measurement-mutation` 是 `developer_build_debug` 的量測選項，不屬於 shipping 介面 |
| parity harness（`developer_build_debug`） | `scripts/eval/diagnostics/native_track_parity.py`（`parity`）；oracle 由 `native_detector_parity.py oracle-rows` 產生（§12） |
| 測試 | `tests/native/test_shipping_serial_runtime.cpp`（GPU；需要 model 與 MOT17，缺少時 SKIP）、`tests/unit/test_native_track_parity.py`（harness 比較器與 validity 檢查）、`tests/unit/test_native_track_oracle_pins.py`（oracle 的 per-run／per-sequence 分工） |

### 13.1 照抄的 oracle 事實（接線）

- **per run**（`evaluator.run_eval`，在 `for seq in cfg.seqs` 之前建立、迴圈內不重綁）：detector（head 只載入一次）、`PerceptionPipeline`、`GlobalTrackIdMapper`；torchvision 的 nvJPEG 解碼器每個 process 一份（§11.1）。native：一個 `SerialRuntime` 持有 resolved config 與各計畫、一條 CUDA stream、一個 `JpegDecoder`、一個 `DetectorHost`、一個 `PerceptionPipeline`，以及容量為 `max_det` 的 device detection buffer。
- **per sequence**（`EvalPipeline.__init__`）：`detector.set_whole_graph_img_dims(h_orig, w_orig)`（只設定座標縮放）、`AdaptiveFramePool(h_orig, w_orig)`（frame buffer）、該 sequence `img1` 的 streamer、tracker／GMC／`GraphedTrackerUpdate`（含 pre-roll，§9.1）。native：每個 sequence 一份 `IngestHost`、`PostDetectorHost`、`SequenceOutput`。
- **per frame**，`k = 1..frame_end`：`ingest(k)` → `detect(frame buffer)` → rows 拷進 run 的 device detection buffer → `process(rows, 同一個 frame buffer)`（GMC 讀的就是 detector 讀的那一幀）→ `add_frame(k, tracker rows)`。每個宿主在 return 前同步 stream，所以下一幀的解碼不會在前一幀的任何 stage 讀完之前覆寫 frame buffer。
- **ID**：oracle 的 ID 是 run-global（`GlobalTrackIdMapper`）；shipping 只做 per-sequence ID（boundary §5 B3），所以多 sequence 的比較是重標後的相同（§10.5）。
- 以上 run／sequence 分工由 `test_native_track_oracle_pins.py` 對 oracle 原始碼釘住；`test_shipping_serial_runtime.cpp` 驗證 native 端：同一個 runtime 跑 X、Y、X 時兩次 X 相同；新 runtime 先跑 Y 與先跑 X 再跑 Y 相同；下列三個接線變異各自讓輸出改變。

### 13.2 `DetectorHost` 的介面更正

PR-8 的 `DetectorHost::set_image_dims(h, w)` 同時記下座標縮放與 `detect` 用來解讀 frame buffer 的尺寸。oracle 不是這樣：`set_whole_graph_img_dims` 只設定縮放，frame tensor 自帶它的形狀。兩者綁在一起時，縮放沒有隨 sequence 更新就會以錯誤的尺寸讀 frame buffer（寫 `stale_image_dims` 負控制時，640×480 的 buffer 被當成 1920×1080 讀，越界）。PR-9 把它拆開：`set_image_dims` 只設定縮放，`detect(frame, height, width)` 由呼叫端給 frame 的尺寸（ingest 的 frame buffer 尺寸）。兩者相同時（ingest 拒絕解碼尺寸與 `seqinfo.ini` 不同的幀，§11.6）行為不變；正式 run 會在 PR-9 的 commit 上重跑 PR-8 的 7-seq detector parity 確認（§13.3）。

### 13.3 測量契約（正式 7-seq run 之前寫定）

**oracle**：在 PR-9 的 commit 上重跑 `native_detector_parity.py oracle-rows`：`A_L` serial、MOT17 train 7 個 SDP sequence（02／04／05／09／10／11／13，共 5316 幀），一個 `mot17.py` process 依此順序跑完。它留下每個 sequence 的 MOT txt、`_global_id_map.txt`，以及 `evaluator._run_detect` 每幀的輸出（PR-5 `detector.bin` 格式）。

**native**：一個 `saccade_track` process，依 oracle 的順序跑同樣 7 個 sequence，加 `--trace`（每幀 detector rows，同一格式）與 `--report`。

**section**：

| section | native | oracle | 比較 |
|:--|:--|:--|:--|
| `detector` | 端到端接線上每幀的 detector rows | oracle run 的 `_run_detect` rows | 每幀整筆 record 逐位元組（rows 數、box、score、class） |
| `mot_txt` | `<seq>.txt` | oracle 的 `<seq>.txt`，ID 減去位移（位移取自 oracle 的 `_global_id_map.txt`，該 sequence 的 global ID 必須是連續的一段） | 逐位元組；另要求 track ID 數相同 |

**有效性**（任一不成立 ⇒ `UNRESOLVED`，不判讀 section）：正式 run 在乾淨的 commit 上、gpu0 lease 下執行；attestation 對上 lineage 與 operator library 檔案；`oracle-rows` 報告 ok（V5 sidecar、serial、每個 sequence 錄到 1..frame_end、`is_tiled` 全 false），oracle 的 sequence 順序與 `--max-frames` 和 native 相同，`detector.bin` 的 sha256 與它的 `meta.json` 相同；`saccade_track` exit 0，report 顯示：serial、無 mutation、process 內沒有 Python 函式庫、op library 是 attestation 的 build（經 attestation 綁定）、artifact／engine 的 sha256 與 lineage 相同、runtime readback 與 scan 呼叫次數與 lineage 相同、參數 cuda:0／constant CPU、每個 sequence 的寬、高、幀數與 `seqinfo.ini` 相同。

**EXACT 驗收規則**：PR-9 的 verdict 是 `EXACT` 若且唯若下列全部成立，否則是 `NOT_EXACT`：

1. 有效性成立；
2. `detector` 與 `mot_txt` 在 7 個 sequence 上全部相同（`detector` 5316/5316 幀）；
3. 重現性：第二次 `parity --against` 再次 `EXACT`，且每個 sequence 的 native txt 與 trace 的 sha256 與第一次相同；
4. PR-8 回歸：以 PR-9 commit 的 `saccade_detector_probe` 對同一份 `oracle-rows` 跑 `native_detector_parity.py parity`（7 sequence），9 個 section 全部 `EXACT`（§13.2 的介面更正不改變 detector）。

沒有容差，看到結果之後也不新增容差。任一項不成立：照 section 分開報告（第一個不同的幀／行、rows 數、track ID 數），**停在 PR-9**，不把差異帶進 PR-10。

**負控制**（7 sequence，對同一份 oracle）：`saccade_track --measurement-mutation` 每次只破壞一條接線規則，對應的 section 必須是 `DIFFERS`（其他 section 照實記錄）：

| 負控制 | 破壞的規則 | 必須 `DIFFERS` 的 section |
|:--|:--|:--|
| `shared_post_host` | 第一個 sequence 的 `PostDetectorHost`（tracker、GMC、pre-roll）留給之後的 sequence | `mot_txt` |
| `stale_image_dims` | 座標縮放只在第一個 sequence 設定 | `detector`（只有幾何與第一個 sequence 不同的 MOT17-05 會變） |
| `gmc_previous_frame` | GMC 讀前一幀的 frame buffer | `mot_txt` |
| `--ref-edit`（harness） | 第一個 sequence 的 oracle txt 在記憶體中改 1 個字元 | `mot_txt`（只有 MOT17-02） |

**觀察（不是 gate）**：oracle serial run 的 MOT txt 是否與 PR-2L `A_L_1`（double-buffer）逐位元組相同。PR-8 的 `oracle-rows` 與 anchor 在 `a9d657c1` 上 7/7 相同；這只記錄，PR-10 才以 double-buffer 為驗收組態。

**修訂 A1（第一次正式 run 之後，只改負控制；上面的契約原文不動）**：第一次正式 run（commit `6fcfa788`）中，`shared_post_host` 負控制沒有產生可判讀的結果：MOT17-02（1920×1080）的 `PostDetectorHost` 被留給 MOT17-05（640×480），GMC 以前者的尺寸讀後者的 frame buffer，越界，`saccade_track` 以 CUDA illegal address 結束（harness 判 `UNRESOLVED`、未抓到）。這是負控制的定義錯誤（它破壞的不只是「sequence 狀態」，還破壞了 buffer 尺寸），不是 shipping 路徑的問題；GPU 測試只在同一幾何的兩個 sequence 之間共用，所以沒有暴露。A1 把該變異改成：**前一個 sequence 的 `PostDetectorHost` 只在幾何相同時留給下一個 sequence；幾何不同的 sequence 取得新的宿主**（之後再被留用）。依 7-seq 的順序，預期 MOT17-04、10、11、13 的 `mot_txt` 為 `DIFFERS`，02、05、09 為 `EXACT`；必須 `DIFFERS` 的 section 仍是 `mot_txt`。GPU 測試加上「另一幾何取得新宿主、第二次同幾何改變」兩項。因為變異的程式碼改了，整份正式 run（oracle-rows、parity、repeat、PR-8 回歸、4 個負控制）在 A1 的 commit 上重跑一次；第一次 run 的結果保留並在 §13.4 一併報告。驗收規則與其餘負控制不變。

### 13.4 驗收

同一台機器（RTX 5070 Ti Laptop）、`build/` 組態，全部步驟在 gpu0 lease 下依序執行（`run.sh`）。正式 run 是 A1 的 commit `89658d3b`（工作樹乾淨；§13.3 的契約在 `6fcfa788` 就已 commit，早於任何 MOT17 上的 parity 量測）：

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | `oracle-rows`：`A_L` serial、7 sequence、5316 筆、`is_tiled` 全 false、V5 通過；oracle 與 native 的 sequence 順序相同；`saccade_track`：exit 0、未 map 任何 Python 函式庫、op library 是 attestation 的 build（經 attestation 綁定）、artifact／engine sha256、runtime readback、scan 呼叫 3 次、參數 cuda:0／constant CPU 皆與 lineage 相同；7/7 sequence 的寬、高、幀數與 `seqinfo.ini` 相同；5316 幀全部走 hardware batched 解碼 |
| `detector`（7 sequence、5316 幀） | **EXACT**：端到端接線上每一幀的 detector rows 與 oracle run 的 `_run_detect` 輸出逐位元組相同 |
| `mot_txt`（7 sequence） | **EXACT**：7/7 重標後逐位元組相同，track ID 數相同（113／103／153／37／168／92／149）；MOT17-02（位移 0）不重標就 `cmp` 相同 |
| 重現性 | 第二次 `parity --against`：再次 `EXACT`，每個 sequence 的 native txt 與 trace sha256 與第一次相同 |
| PR-8 回歸 | PR-9 commit 的 `saccade_detector_probe` 對同一份 `oracle-rows`：9 個 section 全部 `EXACT`（5316 幀） |
| **verdict** | **`EXACT`**（§13.3 第 1–4 條全部成立） |

**負控制**（7 sequence，同一份 oracle）全部被抓到：

| 負控制 | DIFFERS 的 section／sequence（其餘 EXACT） |
|:--|:--|
| `shared_post_host`（A1） | `mot_txt`：MOT17-10、11、13；`detector` 5316/5316 EXACT。A1 預期 MOT17-04 也會不同，實際是 `EXACT`（從 MOT17-02 留下的宿主沒有改變 MOT17-04 的輸出）；驗收規則只要求 section，這一項預測沒有成立，照實記錄，未做歸因 |
| `stale_image_dims` | `detector`：只有 MOT17-05（837/837 幀不同，其餘 4479 幀相同）；`mot_txt` 也只有 MOT17-05 不同 |
| `gmc_previous_frame` | `mot_txt`：7/7 sequence；`detector` EXACT |
| `--ref-edit` | `mot_txt`：只有 MOT17-02，第一個不同的行就是改過的那一行（第 0 行） |

**第一次正式 run**（commit `6fcfa788`，A1 之前）：parity `EXACT`、重現性相同、PR-8 回歸 `EXACT`，`stale_image_dims`／`gmc_previous_frame`／`--ref-edit` 抓到（定位與上表相同）；`shared_post_host` 越界結束（`UNRESOLVED`，§13.3 A1）。兩次 run 的 native txt 7/7 逐位元組相同。

**觀察（不是 gate）**：兩次 run 的 oracle serial txt 都與 PR-2L `A_L_1`（double-buffer）7/7 逐位元組相同。

結果目錄：`results/465_pr9_track/full7_89658d3b/`（正式）與 `results/465_pr9_track/full7_6fcfa788/`（第一次）：`MANIFEST.md`、`run.sh`、`oracle_rows/`、`parity/`、`repeat/`、`pr8_regression/`、`negctl_*/` 與 log；不納入版本控制。

### 13.5 限制

- 只驗 serial、eager；CUDA graph、double-buffer、decode prefetch 與 event barrier 在 PR-10，那時的驗收組態是 double-buffer。rows 以 host 往返交給 post-detector 宿主、每個宿主每幀同步一次：這是 serial 的接線，`saccade_track` 的執行時間不是效能主張。
- parity 是同一台機器、headline 組態、`A_L` 的對照，不是一般性的等價主張；它沿用 PR-7／PR-8 的條件（nvJPEG 硬體路徑、torch 2.11.0／triton 3.6.0 的 S2 lowering、per-machine 的 operator library 與 realization attestation）。
- 對 `A_L` 是 EXACT，不代表對 headline：依 owner 的 named limit，native shipping 的 MOT 輸出與 headline 不逐位元組相同，headline 的數字不轉用到 shipping。
- 多 sequence 的 parity 是重標後的相同；shipping 的 ID 是 per-sequence 的（boundary §5 B3）。
- native 對輸入比 oracle 嚴格：oracle 會略過沒有 `seqinfo.ini` 的 sequence、截短列檔不足的 sequence，`saccade_track` 都拒絕（§11.6）。
- 只量了 oracle 的這個 sequence 順序；`test_shipping_serial_runtime.cpp` 另外在 40 幀上檢查 sequence 狀態不跨 sequence 洩漏（X、Y、X 與先跑 Y）。
- `saccade_shipping_runtime` 連 `saccade_perception`（經由 detector 的 `TRTEngine`），所以 `saccade_track` 的 `DT_NEEDED` 含 OpenCV；拆掉是 PR-11。

### 13.6 重現

```bash
cmake --build build --target saccade_track saccade_detector_probe saccade_shipping_serial_runtime_test
.venv/bin/python tools/resctl.py run gpu0 -- build/shipping/saccade_shipping_serial_runtime_test \
    configs/shipping/mamba_whole_graph.resolved.json \
    models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json \
    configs/shipping/mamba_head_realization.attestation.json . datasets/MOT17/train
bash results/465_pr9_track/<label>/run.sh   # oracle-rows, parity, repeat --against, PR-8 regression, 4 negctls

# the shipping entrypoint alone
build/shipping/saccade_track --config configs/shipping/mamba_whole_graph.resolved.json \
    --lineage models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json \
    --attestation configs/shipping/mamba_head_realization.attestation.json \
    --out <dir> datasets/MOT17/train/MOT17-02-SDP datasets/MOT17/train/MOT17-04-SDP ...
```

## 14. Native double buffer＋CUDA graph（PR-10／U5）

PR-10 把 PR-9 的端到端 runtime 換成 oracle 本身的排程：boundary §2 的 oracle 是 `mot17.py --double-buffer`（detect(N+1) ∥ tracker(N)、event barrier），而且 oracle 在這條路徑上 capture 四個 CUDA graph。`saccade_track` 依 resolved config 的排程執行（headline＝double buffer＋四個 graph）；PR-9 的 serial、eager runtime 保留為 developer 選項（`--schedule serial`），是 PR-9 量過的參考。stage 本身不變（PR-5～PR-8），新的東西只有：排程、stream／event、graph 的 capture 時機與 recapture key、以及每個 buffer 何時可以被覆寫。parity oracle 是 owner 接受的 `A_L` 的 double-buffer 組態（#465 決策 1；PR-2L 的 `A_L_1`）。

| 項目 | 位置 |
|:--|:--|
| 排程計畫（CUDA-free） | `shipping/include/saccade_shipping/schedule_plan.hpp`、`shipping/src/schedule_plan.cpp`（併入 `saccade_shipping_native_config`；CPU 測試在 `test_shipping_native_config.cpp`） |
| double-buffer runtime | `shipping/include/saccade_shipping/double_buffer_runtime.hpp`、`shipping/src/double_buffer_runtime.cpp`（併入 `saccade_shipping_runtime`） |
| graph 模式 | `DetectorHost::detect_graphed`（whole-detect graph）、`PostDetectorHost` 的 `GraphMode::Captured`（main NMS、GMC、tracker graph）、`IngestHost` 的兩個 pool（decode 與 normalize 分開） |
| shipping entrypoint | `shipping/tools/saccade_track.cpp`：排程由 config 決定；`--schedule serial` 與新的 `--measurement-mutation` 值屬 `developer_build_debug` |
| parity harness（`developer_build_debug`） | `scripts/eval/diagnostics/native_track_parity.py`（`--schedule double_buffer`，`--oracle-txt` 是 `native_detector_parity.py anchor` 的輸出；`--no-trace`） |
| 測試 | `tests/native/test_shipping_double_buffer_runtime.cpp`（GPU；需要 model 與 MOT17，缺少時 SKIP）、`tests/unit/test_native_double_buffer_oracle_pins.py`（oracle 的排程與 graph 生命週期）、`tests/unit/test_native_track_parity.py`（新增 graph section、oracle log 解析、double-buffer oracle 的有效性）、`tests/unit/test_saccade_track_schedule_cli.py`（torn config 在 entrypoint 上 fail closed，含 `--schedule serial`；§14.7） |

### 14.1 照抄的 oracle 事實（排程與 graph）

由 `test_native_double_buffer_oracle_pins.py` 對 oracle 原始碼釘住：

- **排程的條件**：`_double_buffer_eligible` 要求 `SACCADE_DOUBLE_BUFFER`、`SACCADE_DETECT_BARRIER=event` 與 frame-independent 的 detector（whole graph）。exporter 把結果記成 `steps.schedule.double_buffer`；`plan_schedule` 要求這些輸入與記錄一致，不一致時 fail closed，不自行重新解讀。四個 graph 的開關也由 config 推出（whole graph＋TRT backbone、`SACCADE_MAIN_NMS_GRAPHED` 且無 ONMS、`gmc_mode=gpu` 且無 fg mask、`steps.track.graphed_update`）；native 只實作四個都 capture 的路徑。
- **frame loop**：`_schedule(1)` 先啟動第 1 幀；每次迭代先 `_schedule(k+1)`（`k < frame_end` 時），再 `_run_frame(k, prepared_detection)`。同時只有一個 detection 在飛。
- **parity**：第 k 幀用 `double_buffer_pools[(k-1) % 2]` 與 `double_buffer_events[(k-1) % 2]`；`EvalPipeline` 在 eligible 時建立兩個 pool、一條 side stream、兩組 event。native：每個 sequence 一個 `IngestHost(pools=2)`，每個 run 兩組 device detection buffer。
- **launch**：`input_ready` 記在 main stream，side stream 等它，然後在 side stream 上 normalize（`_run_detect` 的 ingest_preprocess）、whole-detect graph、把輸出 clone 出 graph 的 static buffer，最後記 `ready_event`。`_run_frame` 讓 main stream 等 `ready_event`，並改用該 parity 的 pool（GMC 讀它的 frame buffer）。native 的 clone 是把 static output 拷進該 parity 的 detection buffer，class 的 float→int32 轉換（`_run_native_tensor_prep`）在同一步完成。
- **為什麼覆寫是安全的（event barrier）**：launch(k+1) 重用第 k−1 幀的 frame buffer 與 detection buffer；它們最後的讀者（GMC 的 frame 拷貝、copy_pad）在 main stream 上排在 `input_ready(k+1)` 之前，side stream 等這個 event。oracle 靠 caching allocator 與 `record_stream` 保護 decode 輸出；native 改成每個 parity 一個 `normalized` event，host 在 decode 進該 buffer 前等它。
- **whole-detect graph**：key＝frame 形狀＋image dims＋NMS pad。`set_whole_graph_img_dims` 遇到相同 dims 時保留 graph 與 warm 旗標，不同時清掉兩者。cache miss 時：未 warm 先跑一次 warm-up（`_whole_graph_warmup`），再由 `make_graphed_callables` 在 `frame.clone()` 上 warm-up 3 次後 capture；cache 已有 10 個時先清空。native 用 LibTorch 的 `at::cuda::CUDAGraph`（private memory pool、`thread_local` capture mode），每次 replay 前把 frame 拷進 static input。7-seq 的順序下，oracle 與 native 都只在 MOT17-02、05、09 capture（04、10、11、13 沿用）。
- **main NMS graph**（每個 sequence 一個）：每幀先 copy_pad；第一幀以 eager 的 nocopyback 呼叫 warm-up、同步、capture，然後 replay；之後每幀 replay。private-continuation append 仍是 eager。
- **GMC graph**（每個 sequence 一個）：第一幀把 frame 拷進 `_gmc_frame_buf`、在它上面 eager 估計、同步、capture，**不 replay**；之後每幀拷貝再 replay。
- **tracker graph**（每個 sequence 一個）：4 次 pre-roll（§9.1）之後 capture `update_into`（全零 scratch 輸入、identity warp；`copy_inputs` 在寫入之前 capture），之後每次 update 先拷輸入再 replay。oracle 的 `[TrackerGraph] Captured` 是在建構 `GraphedTrackerUpdate` 時印的，實際的 capture 是 lazy 的，所以這一行只標出 sequence 的開始。
- **沒有照抄的部分（只影響時間，不影響值）**：oracle 的 decode 在 worker thread 上預取（decode 在進 queue 前已完成）；native 在 host thread 上依序 decode。oracle 把 tracker 輸出的 D2H 延後到下一幀（pinned parity buffer），emit 順序不變；native 在每幀結束時同步讀回。torch 在自己的 side stream 上 capture；native 在 runtime 的 stream 上 capture（capture 本身不執行）。

### 14.2 operator library 的重新 attestation

2026-10-04 開發期間，`cmake` 重新 configure 後的完整 build 重新編出 `build/libsaccade_scan_torchop.so`：sha256 從 attestation 綁定的 `098dd233…` 變成 `aa84cccd…`（同一份 source；nvcc 輸出不是逐位元組可重現的）。`098dd233` 的 build 已不存在，所有讀 attestation 的 detector 工具都會 fail closed。這是 §12.3 的機制預期要處理的情況：在乾淨的 commit 上跑 `native_detector_parity.py anchor`（`A_L` double-buffer、7 sequence），**只有**當 7 個 txt 全部與 PR-2L `A_L_1` 逐位元組相同時，才以 `attest` 重寫 `configs/shipping/mamba_head_realization.attestation.json`（PR-1L freeze、lineage 都不動），並以獨立的 commit 提交。不相同 ⇒ 不 attest，PR-10 停在這裡，交 owner。正式 run 在 attestation commit 之上執行。

### 14.3 測量契約（正式 7-seq run 之前寫定）

在本節 commit 之前，PR-10 只在 MOT17 上跑過 native 對 native 的檢查（GPU 測試；`saccade_track` 的 double buffer 與 `--schedule serial` 在 3 個 sequence 的前 40 幀逐位元組相同，三個負控制都改變輸出），沒有跑過任何對 oracle 的 parity。

**oracle**（在 PR-10 的正式 commit 上重跑，同一個 gpu0 lease）：
- `native_detector_parity.py anchor`：`A_L` double-buffer、MOT17 train 7 個 SDP sequence（02／04／05／09／10／11／13，5316 幀），一個 `mot17.py` process 依此順序。它留下 MOT txt、`_global_id_map.txt` 與 log（graph capture 的訊息）；它同時是 §14.2 的檢查（與 `A_L_1` 7/7 相同）。
- `native_detector_parity.py oracle-rows`：`A_L` serial，同樣 7 個 sequence，留下 `_run_detect` 每幀的輸出（§13.3）。oracle 的 detector 是 frame-independent 的：double buffer 只是在 side stream 上跑同一個 `_run_detect`，所以 detector rows 以 serial run 為 oracle。

**native**：一個 `saccade_track` process，依 oracle 的順序跑 7 個 sequence，排程取自 config（double buffer），加 `--trace` 與 `--report`。

**section**：

| section | native | oracle | 比較 |
|:--|:--|:--|:--|
| `detector` | double buffer 下每幀的 detector rows（從該 parity 的 detection buffer 讀回） | `oracle-rows` 的 `_run_detect` rows | 每幀整筆 record 逐位元組 |
| `mot_txt` | `<seq>.txt` | `anchor` 的 `<seq>.txt`，ID 依它的 `_global_id_map.txt` 重標 | 逐位元組；track ID 數相同 |
| `graph_captures` | 每個 sequence 的 whole-detect／main NMS／GMC capture 數，以及 replay 數 | `anchor` log 中該 sequence 區段的 capture 訊息數 | capture 數相等；tracker capture＝1；replay：detector＝幀數，NMS／tracker＝tracker update 數，GMC＝update 數 − 1 |

**有效性**（任一不成立 ⇒ `UNRESOLVED`）：正式 run 在乾淨的 commit 上、gpu0 lease 下執行；attestation 對上 lineage 與 operator library；`anchor` 報告無 problem、與 `A_L_1` 7/7 相同、在同一個 commit 上、是 double buffer、sequence 順序與 native 相同；`oracle-rows` 的條件同 §13.3；`saccade_track` exit 0，report 顯示：schedule `double_buffer`、無 mutation、process 內沒有 Python 函式庫、op library／artifact／engine 經 attestation／lineage 綁定、runtime readback 與 scan 呼叫次數相同、參數 cuda:0／constant CPU、每個 sequence 的寬、高、幀數與 `seqinfo.ini` 相同。

**EXACT 驗收規則**：PR-10 的 verdict 是 `EXACT` 若且唯若下列全部成立，否則是 `NOT_EXACT`：

1. 有效性成立；
2. `detector`（5316/5316 幀）、`mot_txt`（7/7）、`graph_captures`（7/7）全部相同；
3. 重現性：第二次 `parity --against` 再次 `EXACT`，每個 sequence 的 native txt 與 trace sha256 與第一次相同；
4. 不帶 trace（shipping 組態；trace 每幀多一次同步，可能遮住 race）：`parity --no-trace --against` 的 `mot_txt` 與 `graph_captures` `EXACT`，txt sha256 與第一次相同；
5. PR-9 回歸：同一份 `oracle-rows`，`parity --schedule serial` 的 `detector` 與 `mot_txt` 7/7 `EXACT`（`IngestHost`／`DetectorHost` 的重構不改變 serial）；
6. PR-8 回歸：PR-10 commit 的 `saccade_detector_probe` 對同一份 `oracle-rows`，`native_detector_parity.py parity` 的 9 個 section 全部 `EXACT`。

沒有容差，看到結果之後也不新增容差。任一項不成立：照 section 分開報告（第一個不同的幀／行、capture 數），停在 PR-10。

**負控制**（7 sequence，對同一份 oracle）：`saccade_track --measurement-mutation` 每次只破壞一條 graph／parity 規則，對應的 section 必須是 `DIFFERS`（其他 section 照實記錄）：

| 負控制 | 破壞的規則 | 必須 `DIFFERS` 的 section |
|:--|:--|:--|
| `stale_detector_input` | whole-detect graph replay 前不把 frame 拷進 static input | `detector` |
| `stale_gmc_input` | GMC graph replay 前不把 frame 拷進 captured buffer | `mot_txt`（`detector` 應為 `EXACT`） |
| `swapped_detection_parity` | tracker(k) 讀另一個 parity 的 detection buffer（等第 k+1 幀的 ready 後讀它；最後一幀讀到第 k−1 幀的） | `detector` |
| `--ref-edit`（harness） | 第一個 sequence 的 oracle txt 在記憶體中改 1 個字元 | `mot_txt`（只有 MOT17-02） |

拿掉 event wait 這類 race 型的變異不當負控制：結果取決於時序，沒抓到不能證明什麼，抓到也不可重現。

**觀察（不是 gate）**：`anchor`（double buffer）與 `oracle-rows`（serial）的 txt 是否 7/7 相同；native double buffer 與 native serial（第 5 條）的 txt 是否相同；同一個 session 內的 FPS：native double buffer（第 4 條，無 trace）、native serial（第 5 條，有 trace）與 oracle `anchor` 的 `_fps_summary.txt`。三者的定義不同（native 是整個 frame loop 的 wall time、含第 1 幀的 graph capture；oracle 從第 51 幀起算），只並列記錄，不做效能主張（boundary §6 PR-10：FPS 只做同 session 對照）。

### 14.4 驗收

**§14.2 的重新 attestation**：在 `306607e3`（工作樹乾淨）跑 `anchor`，7 個 txt 與 PR-2L `A_L_1` 逐位元組相同、無 problem ⇒ `attest` 把 attestation 改綁新 build（`aa84cccd…`；只有 op library 的 sha256／bytes 與 reproduction 紀錄改變，`A_L` 的 txt hash 不變），以 `44cc91a4` 單獨提交。

同一台機器（RTX 5070 Ti Laptop）、`build/` 組態，全部步驟在 gpu0 lease 下依序執行（`run.sh`）。正式 run 是 `44cc91a4`（工作樹乾淨；§14.3 的契約在 `306607e3` 就已 commit，早於任何對 oracle 的 parity 量測）：

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | `anchor`：與 `A_L_1` 7/7 相同、無 problem、同一個 commit、double buffer；`oracle-rows`：`A_L` serial、5316 筆、V5 通過；sequence 順序三者相同；`saccade_track`：schedule `double_buffer`、exit 0、未 map Python、op library 經 attestation 綁定、artifact／engine／runtime readback／scan 3 次／placement 皆與 lineage 相同、7/7 的寬高幀數與 `seqinfo.ini` 相同 |
| `detector`（5316 幀） | **EXACT**：double buffer 下每幀從 parity detection buffer 讀回的 rows 與 oracle 的 `_run_detect` 逐位元組相同 |
| `mot_txt`（7 sequence） | **EXACT**：7/7 重標後逐位元組相同，track ID 數相同（113／103／153／37／168／92／149） |
| `graph_captures`（7 sequence） | **EXACT**：whole-detect capture 1／0／1／1／0／0／0（MOT17-02／04／05／09／10／11／13），main NMS 與 GMC 每個 sequence 1，與 oracle log 相同；tracker capture 1；replay 數符合 |
| 重現性 | 第二次 `parity --against`：`EXACT`，native txt 與 trace sha256 7/7 相同 |
| 不帶 trace | `parity --no-trace --against`：`mot_txt`、`graph_captures` `EXACT`，txt sha256 7/7 與第一次相同 |
| PR-9 回歸 | `--schedule serial`：`detector` 5316/5316、`mot_txt` 7/7 `EXACT` |
| PR-8 回歸 | `saccade_detector_probe`：9 個 section 全部 `EXACT`（5316 幀） |
| **verdict** | **`EXACT`**（§14.3 第 1–6 條全部成立） |

**負控制**（7 sequence，同一份 oracle）全部被抓到：

| 負控制 | DIFFERS 的 section／sequence |
|:--|:--|
| `stale_detector_input` | `detector`：7/7 sequence，只有 3 幀相同（MOT17-02、05、09 的 capture 幀：static input 就是那一幀）；`mot_txt` 7/7 |
| `stale_gmc_input` | `mot_txt`：7/7；`detector` 5316/5316 `EXACT` |
| `swapped_detection_parity` | `detector`：5316/5316 幀不同；`mot_txt` 7/7 |
| `--ref-edit` | `mot_txt`：只有 MOT17-02，第一個不同的行就是改過的那一行（第 0 行） |

`graph_captures` 在四個負控制下都是 `EXACT`（變異不改變 capture 與 replay 的次數）。

**觀察（不是 gate）**：`anchor`（double buffer）與 `oracle-rows`（serial）的 txt 7/7 相同，serial txt 與 `A_L_1` 7/7 相同；native double buffer 與 native serial 的 txt 7/7 相同。同一個 session 的 FPS（定義不同，只並列）：native double buffer（無 trace，整個 frame loop，含 capture）278.9；native serial（有 trace）105.4；oracle `anchor` 的 `OVERALL` 292.98（第 51 幀起）。

**確認（修訂 R1 之後，§14.7）**：在 `f99bb288`（工作樹乾淨）重跑 `anchor`（與 `A_L_1` 7/7 相同）、`oracle-rows`、主 parity 與 `--schedule serial` 回歸：全部 `EXACT`，主 parity 的 native txt 與 trace sha256 與正式 run 7/7 相同（`--against`）。目錄 `results/465_pr10_track/confirm_f99bb288/`。

結果目錄：`results/465_pr10_track/full7_44cc91a4/`（`MANIFEST.md`、`run.sh`、`anchor/`、`oracle_rows/`、`parity/`、`repeat/`、`no_trace/`、`serial_regression/`、`pr8_regression/`、`negctl_*/` 與 log）與 `results/465_pr10_track/attest_306607e3/`；不納入版本控制。

### 14.5 限制

- parity 是同一台機器、headline 組態、`A_L` 的對照，不是一般性的等價主張；沿用 PR-7／PR-8 的條件（nvJPEG 硬體路徑、torch 2.11.0／triton 3.6.0 的 S2 lowering、per-machine 的 operator library 與 realization attestation）。對 `A_L` 是 EXACT，不代表對 headline（owner 的 named limit）。
- double buffer 的正確性靠 event barrier；race 型的變異沒有當負控制（§14.3）。「不帶 trace」與重現性各量了一次，是同一台機器、同一種負載下的觀察，不是無 race 的證明。
- 只量了 oracle 的 sequence 順序與 headline 的 graph key；GPU 測試另外在 40 幀上檢查 X、Y、X、Z（dims 改變時重新 capture、相同 dims 沿用）。
- FPS 只是同 session 的並列，不是效能主張：三者的時間定義不同，native 的 tracker 輸出仍每幀同步讀回（oracle 延後一幀），decode 在 host thread 上依序執行（oracle 預取）。
- operator library 是 per-build 的：重新 configure／完整 build 可能重新編出不同位元組的 `.so`，之後必須依 §14.2 重新 attest（`anchor` 與 `A_L_1` 相同才可以）。
- `saccade_track` 的 `DT_NEEDED` 仍含 OpenCV（經 `TRTEngine`）；拆掉是 PR-11。

### 14.7 修訂 R1（owner review 之後）

review 發現 `saccade_track` 以 `opt.schedule == "serial" || !plan_schedule(cfg).double_buffer` 選排程：`||` short-circuit，`--schedule serial` 時 `plan_schedule` 完全不執行，所以一份 torn config（`steps.schedule.double_buffer` 與 `SACCADE_DETECT_BARRIER`／`SACCADE_DOUBLE_BUFFER` 不一致）可以用 developer 選項繞過 §14.1 承諾的 fail-closed。修法：`select_schedule(cfg, serial_requested)` 先無條件執行 `plan_schedule`，override 只選 runtime、不跳過驗證；`saccade_track` 改用它。測試：`test_shipping_native_config.cpp` 釘住 `select_schedule`（兩種 torn config × 有無 override 都拒絕）；`test_saccade_track_schedule_cli.py` 直接跑 binary（不需 GPU：在載入任何模型之前就拒絕），torn config 加或不加 `--schedule serial` 都必須 exit 2 並給出 schedule 錯誤。修正前的 binary 剛好在兩個 `--schedule serial` case 失敗。

預設路徑的選擇不變（headline config 仍是 double buffer），§14.4 的正式 run 不重做；修正 commit 上另跑一次確認（主 parity 與 PR-9 serial 回歸），結果記在 §14.4 之後的「確認」一行。

### 14.6 重現

```bash
cmake --build build --target saccade_track saccade_detector_probe saccade_shipping_double_buffer_runtime_test
.venv/bin/python tools/resctl.py run gpu0 -- build/shipping/saccade_shipping_double_buffer_runtime_test \
    configs/shipping/mamba_whole_graph.resolved.json \
    models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json \
    configs/shipping/mamba_head_realization.attestation.json . datasets/MOT17/train
bash results/465_pr10_track/<label>/run.sh   # anchor, oracle-rows, parity, repeat, no-trace, serial + PR-8 regressions, 4 negctls

# the shipping entrypoint alone (schedule from the config: double buffer)
build/shipping/saccade_track --config configs/shipping/mamba_whole_graph.resolved.json \
    --lineage models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json \
    --attestation configs/shipping/mamba_head_realization.attestation.json \
    --out <dir> datasets/MOT17/train/MOT17-02-SDP datasets/MOT17/train/MOT17-04-SDP ...
```

## 15. CMake 拆分：tracking 不連 perception、OpenCV 可選（PR-11／U6a）

PR-11 是 boundary §6 的 U6a：`saccade_tracking` 不再連 `saccade_perception`，OpenCV 改成可選，讓 shipping entrypoint `saccade_track` 兩者都不需要。PR-10 之後 `saccade_track` 的 `DT_NEEDED` 有 5 個 OpenCV 函式庫：`libsaccade_tracking.a` 裡的 GMC 帶進 CPU LK 路徑，`PerceptionPipeline` 的 ReID 路徑帶進 perception 的 `FeatureExtractor`／`Cropper`，後者又帶進 `preprocessor.cpp` 的 `cv::resize`（linker map：`gmc.cpp.o`、`pipeline.cpp.o` → `feature_extractor.cpp.o`、`preprocessor.cpp.o`）。PR-11 不改任何 stage 的計算，只搬程式碼與改 link 結構；boundary 對它的驗收是「既有 extensions 的 eval 輸出不變」，本節把它展開成 §15.3 的規則。

| 項目 | 位置 |
|:--|:--|
| ReID 介面（tracking 端） | `include/tracking/reid_backend.hpp`：`ReidExtractor`、`RoiCropper`；`PerceptionPipeline` 的建構子改收這兩個（擁有；null＝無 ReID） |
| perception 的 adapter | `include/tracking/perception_reid_adapter.hpp`（header-only，只有 tracking extension include） |
| GMC 的 CPU 模式 | `include/tracking/gmc_cpu.hpp`、`src/tracking/gmc_cpu.cpp`（`gmc_estimate_mat`，即 Python 的 `GMC.estimate_mat`） |
| `Preprocessor` 的 CPU 路徑 | `src/perception/preprocessor_cpu.cpp`（只有 `saccade_node` 呼叫） |
| CMake | root `CMakeLists.txt`：`SACCADE_WITH_OPENCV`、`saccade_opencv_host`、`saccade_eval_runner`、`saccade_tracking` 的 link 檢查；`shipping/CMakeLists.txt`：`saccade_track` 以 `--as-needed` link，POST_BUILD 跑 `shipping/cmake/check_link_surface.cmake` |
| parity harness | `native_track_parity.py parity --track-binary`（跑另一個 build 的 entrypoint） |
| 測試 | `tests/unit/test_shipping_link_surface.py`（source 掃描與 link 檢查腳本）；CI `cpp_build.yml` 多一步 OFF build |

### 15.1 搬了什麼

- **`PerceptionPipeline` 的 ReID 後端**：pipeline 原本以 `FeatureExtractor*`／`Cropper*` 直接呼叫 perception。只把 ReID 方法搬到另一個 TU 不夠：`PerceptionPipeline` 實作 `ReidCropStore` 的 virtual 函式（`requery_extract`、`embed_dim`），vtable 會把它們連同 perception 一起拖進任何建構 pipeline 的程式。改成 tracking 自己宣告所需的介面（`extract`、`get_feature_dim`、`get_max_batch`、`get_input_hw`、profiling 三個、`process_gpu`），由 header-only adapter 轉呼叫原本的 `FeatureExtractor`／`Cropper`；perception 類別本身不改。adapter 只在 tracking extension 裡編譯，所以呼叫的仍是該 extension 自己連結的 perception 程式碼（與 PR-11 之前相同），只多一層 virtual 呼叫。`PerceptionPipelineSnapshot` 的 `reid_ptr`／`cropper_ptr` 仍回報建構時傳入的物件位址（adapter 回報它包的物件）。shipping 與 `seq_runner` 傳的都是 null。
- **GMC 的 CPU 模式**：`GMC::estimate_mat` 的本體一字不改搬到 `gmc_cpu.cpp`。它原本讀寫的兩個成員（前一幀灰階 `cv::Mat`、追蹤點）改放在一個由 `gmc_cpu.cpp` 建立的 state 物件裡，GMC 以 type-erased 的 `shared_ptr<void>` 持有，`GMC::reset()` 丟掉它（原本是 `release()`＋`clear()`；下一次呼叫從空 state 開始，與原本相同）。eval 不會走到這條路徑（`stages.py` 對 C++ GMC 先命中 `estimate_into`），只有直接呼叫 Python API 才會。
- **`Preprocessor::process`**（CPU letterbox，`cv::resize`）一字不改搬到 `preprocessor_cpu.cpp`；`preprocessor.hpp` 不再 include OpenCV（header 本身沒用到）。
- **C++ eval runner**（`seq_runner.cpp`、`eval_pool.cpp`，`--cpp-threads` 後面的 `saccade_eval_ext`）移到 `saccade_eval_runner`：它需要 TRT detector（perception）與 `cv::imread`。
- **CMake**：`saccade_tracking` 只連 CUDA 與 `saccade_tracker_params`，configure 時檢查它的 link libraries 沒有 `saccade_perception` 或 OpenCV（fail closed）。OpenCV 的 include 路徑不再是全域的，只給連 OpenCV 的 target。`SACCADE_WITH_OPENCV=OFF` 時不找 OpenCV（`CMAKE_DISABLE_FIND_PACKAGE_OpenCV`），也不定義需要它的 target：`saccade_opencv_host`、`saccade_eval_runner`、`saccade_tracking_ext`、`saccade_eval_ext`、`saccade_node`；其餘（perception、tracking、兩個 scan 函式庫、shipping、`perception_ext`／`media_ext`／`cheb_gr_online_ext`、native tests）照常 build。
- **`saccade_track` 的 link surface**：POST_BUILD 以 `readelf -d` 檢查直接 `DT_NEEDED`，有 `libopencv_*`、`libpython*`、`libtorch_python*` 就讓 build 失敗。另外以 `--as-needed` link：這個 toolchain 預設是 `--no-as-needed`，拆分後 perception 的依賴排到 torch 自己的 `--as-needed` 開關之前，`libnvinfer_plugin.so.10` 在沒有任何 symbol 被用到的情況下仍被記成 NEEDED（PR-10 之前是 link 順序剛好讓它被濾掉）。加上之後 NEEDED 只剩實際用到的函式庫，不再取決於 link 順序。
- **沒有動的**：frozen 的 `tracker_gpu.{hpp,cu}`；operator library 不重新 build（`build/libsaccade_scan_torchop.so` 仍是 attestation 綁定的 `aa84cccd…`，只 build 具名 target，§14.2）；Python 程式碼（`src/saccade/`、eval harness）不改。

### 15.2 開發期間已經看到的（在本節 commit 之前）

只有不涉及 PR-11 對照組的檢查：`build/` 只 build 具名 target 後 op library sha256 不變；`saccade_track` POST_BUILD 通過；`build/` 與 `build-noopencv/` 的 ctest 各 22/22（gpu0）；檢查腳本對 main 的 `saccade_track`（5 個 OpenCV）、tracking extension（OpenCV）與 perception extension（`libtorch_python`＋OpenCV）都報錯；`test_shipping_link_surface.py` 在 main 的樹上 4 個 source 測試失敗、5 個腳本測試通過。對照組的「之前」端已在 main（`7cc076c5`）上量好：headline 跑兩次 7/7 txt 相同（也與 PR-4b 記錄的 `469f159d` 相同）；GMC CPU 檢查跑兩次相同；ReID adapter 檢查以 main 的參考 extensions 跑兩次相同；`mamba_whole_graph_m_extract_ho_live` 以 main 的參考 extensions 跑兩次 7/7 相同。最後這個組態的 log 顯示 offline handover 0 次，沒有任何訊號說明 ReID 抽取影響了它的輸出，所以 ReID adapter 改用直接的 API 檢查（第 6 條），extract 組態只保留為 eval 輸出不變的 gate（第 7 條）。這些 main 端的 run 是在 PR-11 未 commit 的工作樹上、以 main 的 extensions 執行的（Python 程式碼與 main 相同）。沒有跑過任何 PR-11 build 對 main 或對 oracle 的比較。

### 15.3 測量契約（正式 run 之前寫定）

**組態**：同一台機器；`build/`（`SACCADE_WITH_OPENCV=ON`，只 build 具名 target）與 `build-noopencv/`（`-DSACCADE_WITH_OPENCV=OFF`，完整 build）；main 的參考 extensions 由 `7cc076c5` 的 worktree 以同一個 venv build（只 build 五個 extension，兩個 scan 函式庫以 symlink 指向 `build/` 的同一份檔案），以 `SACCADE_BUILD_PATH` 切換；Python 程式碼兩邊相同。全部 GPU 步驟在 gpu0 lease 下依序執行。

**有效性**（任一不成立 ⇒ 對應的 gate 為 `UNRESOLVED`）：正式 run 在乾淨的 commit 上；`build/libsaccade_scan_torchop.so` 的 sha256 在整個 run 前後都等於 attestation 的值；每個 eval run 的 `meta.txt` 顯示載入的 extension 來自指定的 build；main 端的兩次 run 彼此相同（不相同 ⇒ 該 gate `UNRESOLVED`，因為沒有可比較的參考）；ReID adapter 檢查兩邊都抽出非零 embedding、`gather_crops_framed` 取回的數量等於框數。

**PASS 驗收規則**：PR-11 的 verdict 是 `PASS` 若且唯若下列全部成立，否則是 `FAIL`（照 gate 分開報告）：

1. **link 結構**：兩個 build 的 configure 都通過 `saccade_tracking` 的 link 檢查；`test_shipping_link_surface.py` 通過。
2. **ON 的 link surface**：`build/` 的 `saccade_track` POST_BUILD 通過，且 NEEDED 集合＝main 的 `saccade_track` 的 NEEDED 集合減去那 5 個 `libopencv_*`。
3. **OFF build**：`build-noopencv/` configure 與完整 build 成功；`CMakeCache.txt` 沒有 `OpenCV_DIR`；沒有任何 target 的 `flags.make`／`link.txt` 帶 OpenCV 的 include 或 link 路徑；每個產出的 ELF 都沒有 `libopencv_*` NEEDED；ctest 全部通過；`saccade_track` POST_BUILD 通過。
4. **shipping 行為不變**（PR-11 commit 的 `build/`）：
   - `native_detector_parity.py anchor`（`A_L` double buffer，用重新 build 的 extensions）與 PR-2L `A_L_1` 7/7 逐位元組相同、無 problem；
   - `oracle-rows`（`A_L` serial）照 §13.3 有效；
   - `native_track_parity.py parity`（double buffer、trace）`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7，且 `--against` PR-10 正式 run（`results/465_pr10_track/full7_44cc91a4/parity`）：native txt 與 trace 的 sha256 7/7 相同；
   - `parity --schedule serial`：`detector` 與 `mot_txt` `EXACT`；
   - OFF build 的 entrypoint：`parity --track-binary build-noopencv/shipping/saccade_track`（double buffer、trace）`EXACT`，且 `--against` 上面那次 ON 的 parity：txt 與 trace sha256 7/7 相同。
5. **headline 的 eval 輸出不變**：`run_headline.sh`（boundary §2 的 oracle：`mamba_whole_graph`、`--double-buffer`）以 PR-11 的 `build/` 跑 7 sequence，7 份 MOT txt 的 sha256 與 main 的基準相同。
6. **ReID adapter 路徑**：`reid_adapter_check.py` 以 eval 自己的接法（`cpp_ptr`，`workbench.py`）把真的 `FeatureExtractor`（mnv4 ReID engine）與 `Cropper` 接進 `PerceptionPipeline`，在 MOT17-02、05 前 5 幀的 GT 框上呼叫 `extract_reid`、`crop_into_pool`／`extract_from_pool`、`enable_crop_ring`／`stash_crops`／`gather_crops_framed` 與 profiling 開關（經過 adapter 的每個方法），記錄每份 embedding 與取回的 crop 的 sha256、確定性的 profile 計數（`images`、`chunks`），以及 `snapshot()` 是否仍回報傳入物件的位址。以 PR-11 的 extensions 跑，結果與 main 參考 extensions 完全相同，且 snapshot 一項為真。
7. **extract 組態的 eval 輸出**：`run_eval.sh mamba_whole_graph_m_extract_ho_live`（m backbone、crop ring、live handover）以 PR-11 的 `build/` 跑 7 sequence，7 份 txt 與 main 參考 extensions 的 run 相同。它在 eval 裡走 `enable_crop_ring`／`stash_crops`（cropper adapter）；ReID 抽取是否影響它的輸出沒有確立，所以它不算 ReID 抽取的證據。
8. **GMC CPU 模式**：`gmc_cpu_check.py`（`estimate_mat` 對 MOT17-02 前 60 幀、MOT17-05 前 30 幀，兩組參數，中途與 sequence 之間 `reset()`；每個 warp 以 float hex 記錄）以 PR-11 的 tracking extension 跑，結果與 main 的 tracking extension（PR-11 之前 `build/` 的那一份，也就是 headline 基準所用的檔案）完全相同。

沒有容差。第 4–8 條的比較是逐位元組；任一不同就照 gate 報告第一個不同的檔案／行，停在 PR-11。

**負控制**（記錄在結果目錄）：link 檢查腳本對 main 的 `saccade_track`、PR-11 build 的 `saccade_tracking_ext`（OpenCV）與 `saccade_perception_ext`（`libtorch_python`）必須失敗；`test_shipping_link_surface.py` 在 main 的樹上 source 測試必須失敗；pytest 裡以假的 `readelf` 驅動的三種拒絕。

**不做的**：FPS 或任何效能比較；`saccade_node`（demo）只 build 不跑。

### 15.4 驗收

同一台機器（RTX 5070 Ti Laptop），正式 run 在 `6502fbeb`（工作樹乾淨；§15.3 的契約在同一個 commit，早於任何對照），全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library 在 run 前後都是 attestation 綁定的 `aa84cccd…`。

| 驗收項 | 結果 |
|:--|:--|
| 1 link 結構 | 兩個 build 樹重新 configure 都通過 `saccade_tracking` 的 link 檢查；`test_shipping_link_surface.py` 9 passed |
| 2 ON 的 link surface | POST_BUILD 通過；NEEDED 13 項，恰好是 main 的 18 項減去 5 個 `libopencv_*`（video、features、imgproc、geometry、core） |
| 3 OFF build | configure 與完整 build 成功；cache 沒有 `OpenCV_DIR`；0 個 OpenCV include／link 旗標；31 個產出的 ELF 都沒有 `libopencv_*` NEEDED；ctest 22/22；`saccade_track` POST_BUILD 通過 |
| 4 shipping 行為 | `anchor` 與 `A_L_1` 7/7 相同；`oracle-rows` OK；`parity` `EXACT`（`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7），native txt 與 trace sha256 與 PR-10 正式 run 7/7 相同；`--schedule serial` `EXACT`；OFF build 的 `saccade_track` `EXACT`，txt 與 trace 與 ON 7/7 相同 |
| 5 headline eval 輸出 | 7/7 txt sha256 與 main 基準相同（IDF1 78.3／MOTA 77.9／IDs 429，只記錄） |
| 6 ReID adapter | 10 幀的 embedding、取回的 crop、profile 計數與 main 參考 extensions 完全相同；96/96 個 embedding 非零，取回數＝框數；snapshot 仍回報傳入物件的位址 |
| 7 extract 組態的 eval 輸出 | 7/7 txt 與 main 參考 extensions 相同（IDF1 80.2／IDs 353，只記錄） |
| 8 GMC CPU 模式 | 180 次呼叫（174 個 warp）的 float hex 與 main 的 tracking extension 完全相同 |
| **verdict** | **`PASS`**（§15.3 第 1–8 條全部成立） |

**負控制**：link 檢查腳本對 main 的 `saccade_track`（5 個 `libopencv_*`）、PR-11 build 的 `saccade_tracking_ext`（OpenCV）與 `saccade_perception_ext`（`libtorch_python`＋OpenCV）都失敗（exit 1）；`test_shipping_link_surface.py` 在 main 的樹上 4 個 source 測試失敗、5 個腳本測試通過；pytest 的假 `readelf` 三種拒絕都通過。

**main 端的參考**：headline、extract 組態、ReID adapter、GMC CPU 各跑兩次，兩次都相同（有效性條件）。headline 的 main 基準也與 PR-4b 記錄的 `469f159d` 基準 7/7 相同。

結果目錄：`results/465_pr11_cmake/full_6502fbeb/`（`MANIFEST.md`、`run.sh`、各 gate 的 log／diff、`anchor/`、`oracle_rows/`、`parity/`、`serial_regression/`、`parity_off/`、`headline/`、`extract_ho_live/`、`reid_adapter.json`、`gmc_cpu.json`）、`results/465_pr11_cmake/baseline_main_7cc076c5/`（main 端的參考與 main 的 `saccade_track` 副本）、`results/465_pr11_cmake/negctl_source_test_on_main/`；量測腳本 `run_headline.sh`、`run_eval.sh`、`reid_adapter_check.py`、`gmc_cpu_check.py` 在 `results/465_pr11_cmake/`。不納入版本控制。

### 15.5 限制

- 這是 headline、一個額外 eval 組態與兩個 API 檢查下的觀察，不是一般性的等價主張。headline 不開 ReID、GMC 走 GPU；ReID adapter 以 API 直接量過（mnv4、10 幀，不含 `crop_into_pool_async`／`extract_batch_from_pool` 的非同步批次路徑與 relinker 的 `requery_extract`，它們呼叫的是同樣的 adapter 方法），沒有以每幀 ReID 的 eval 組態量過；CPU GMC 只以 Python API 直接量過（eval 不會走到它）。
- `SACCADE_WITH_OPENCV=OFF` 只讓 OpenCV 變成可選。configure 仍需要 GStreamer、pybind11 與 Python（venv 推導 CUDA toolchain，`developer_build_debug`）；它們是否在 shipping build 中可選屬 PR-12／Phase C。
- link surface 的檢查只看 `saccade_track` 的直接 `DT_NEEDED`；閉包、RUNPATH 與其他 G2 項目是 PR-12。

### 15.6 重現

```bash
cmake build   # reconfigure; then build named targets only (never the op library, §14.2)
cmake --build build --target saccade_track saccade_detector_probe saccade_tracking_ext saccade_perception_ext \
    saccade_eval_ext saccade_cheb_gr_online_ext saccade_media_ext saccade_node <native tests>
cmake -S . -B build-noopencv -DSACCADE_WITH_OPENCV=OFF && cmake --build build-noopencv
cmake -DREADELF=readelf -DBINARY=build/shipping/saccade_track -P shipping/cmake/check_link_surface.cmake
bash results/465_pr11_cmake/<label>/run.sh   # §15.3 gates 1-8 and the negative controls
```

## 16. Shipping tree：`$ORIGIN` RUNPATH、SM＋PTX、glibc baseline、G2 四項（PR-12／U6b）

PR-12 是 boundary §6 的 U6b，Phase B 的最後一個 PR：把 shipping entrypoint 做成一個可安裝的目錄（shipping tree），驗收 boundary §0 的 G2 定義 1–4，以及 `$ORIGIN` RUNPATH、明確的 SM 清單＋PTX、glibc baseline，並在乾淨容器裡跑起來。PR-12 不改任何 stage 的計算，也不改 shipping 的 C++ 原始碼：只加安裝規則、檢查工具與 harness 選項。

| 項目 | 位置 |
|:--|:--|
| 安裝規則 | `shipping/CMakeLists.txt`：`cmake --install <build> --component shipping --prefix <tree>`；`SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST`；`SACCADE_ATTESTED_OP_LIBRARY` |
| model root 安裝 | `shipping/cmake/install_model_root.cmake`（安裝時逐檔比對 sha256） |
| G2 檢查工具 | `scripts/native/check_shipping_tree.py`（`loaded`／`deps`／`static`／`runtime`） |
| 乾淨容器 | `scripts/native/run_shipping_container.sh`（`pristine`／`strace`） |
| parity harness | `native_track_parity.py parity --model-root`、`--track-library-path`、`--native-from` |
| 測試 | `tests/unit/test_shipping_g2_checks.py`；CI `cpp_build.yml` 的 OFF build 改用 shipping 的 SM 清單 |

**owner 決策（2026-10-04，契約之前）**：

1. operator library 不重新 build、不重新 attest：shipping tree 帶的是 attestation 綁定的 `aa84cccd…`，位元組不變。
2. **RUNPATH 規則**：PR-12 產出的每個 shipping ELF 都只能有 `$ORIGIN`-relative 的 RUNPATH。凍結且已 attest 的 operator library `aa84cccd` 是唯一列舉的例外：它保留既有的絕對 build RUNPATH，記為 named limitation，並以 hash 驗證。
3. **SM＋PTX**：明確的多 SM＋PTX 要求只適用於新 build 的 `saccade_track` device code。凍結且已 attest 的 operator library 仍只有 sm_120；因此本 PR 不主張超出「所有載入的 CUDA artifact 都支援的硬體」之外的端到端 GPU 可攜性。
4. **glibc baseline＝Ubuntu 24.04**：每個 ELF 需要的 symbol version 不超過 `GLIBC_2.39`、`GLIBCXX_3.4.33`、`CXXABI_1.3.15`；frozen toolchain（#214）不變。
5. **第三方 runtime 函式庫不 bundle**：tree 只放 Saccade 的檔案；CUDA runtime、nvJPEG、TensorRT、LibTorch 由 tree 外的目錄經 `LD_LIBRARY_PATH` 提供，manifest 記錄 SONAME 與 sha256。是否 bundle、怎麼 bundle 留給 Phase C。
6. **乾淨容器驗收＝完整 7-seq parity**。

### 16.1 改了什麼

- **SM 清單**：`find_package(Torch)` 會把 `CMAKE_CUDA_ARCHITECTURES` 設成 OFF，所有 CUDA target 改用 `TORCH_CUDA_ARCH_LIST` 產生的 `-gencode`（未設定時是本機 GPU，所以開發 build 只有 `sm_120`；CI 設 `7.5`）。shipping tree 的 build 用 `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"`，也就是 LibTorch 自己的清單（torch 2.11.0+cu130：`sm_75/80/86/90/100/120`），加上最新一代的 PTX。安裝步驟在其他設定下拒絕執行。開發 build（`build/`）不變。
- **RUNPATH**：`saccade_track` 的 install RPATH 是 `$ORIGIN/../lib`，不帶 link path（`INSTALL_RPATH_USE_LINK_PATH OFF`）。`lib/` 是 Phase C bundle 要用的位置，這裡是空的。build 樹裡的 `saccade_track` 保留原本的絕對 RUNPATH，所以 harness 與開發流程不變。
- **model root**：`share/saccade/` 放 resolved config、frozen head lineage、realization attestation，以及它們綁定的三個檔案（TorchScript head、backbone engine、operator library），每個檔案都放在 **repository 的相對路徑**上：lineage 與 attestation 都以這些路徑命名而且是凍結的，所以 tree 照抄路徑而不改寫它們（例如 operator library 在 `share/saccade/build/libsaccade_scan_torchop.so`）。安裝時從 lineage／attestation 讀出 sha256 逐一比對；operator library 從 `SACCADE_ATTESTED_OP_LIBRARY`（預設 `build/`）複製，不是 shipping build 樹自己編的那一份。
- **沒有動的**：shipping 與 tracking 的 C++ 原始碼、frozen 的 `tracker_gpu.{hpp,cu}`、Python 程式碼（eval harness 只多了上表的選項）、`build/` 的任何產物（PR-12 不在 `build/` build 任何 target）。

### 16.2 開發期間已經看到的（在本節 commit 之前）

都是用工作樹的試做，不是正式 run，也沒有 oracle 端：

- `build-release/`（OpenCV OFF、上述 SM 清單）的 `saccade_track` 6 個 CUDA TU 各帶 6 個 SASS 與 `sm_120` PTX；POST_BUILD link 檢查通過；安裝後 RUNPATH 是 `$ORIGIN/../lib`。
- 在 host 上以 `LD_LIBRARY_PATH`（開發版 `saccade_track` 的 RUNPATH 目錄，同樣順序）跑 tree 的 binary，7 sequence 的 txt 與 trace 和開發版同一 session 的 run 相同，txt 的 sha256 也與 PR-11 正式 run 相同。兩者載入的第三方函式庫（28 個）sha256 集合相同。
- 其中 `libz.so.1` 來自 host OS（Arch 的 `/usr/lib/libz.so.1.3.2`，cuDNN 的 graph 函式庫需要它），不是 venv。乾淨映像本身有 zlib（`zlib1g`，`Priority: required`），所以檢查工具把它和 glibc、GCC runtime 一起歸為 base system，不放進第三方目錄。`libnvcuvid.so.1`（nvJPEG 的硬體解碼）是 driver 的函式庫，所以容器要開 `video` driver capability。
- 系統的 CDI spec（`/etc/cdi/nvidia.yaml`，04-18）還指向 Windows driver 更新前的 WSL driver 目錄，所以 `docker run --gpus all` 會失敗；改為手動掛載 `/dev/dxg` 與 `/usr/lib/wsl` 的 smoke run（MOT17-05 前 30 幀）在 `ubuntu:24.04` 裡跑得起來，strace 版本只有一次 `execve`。正式 run 用重新產生的 CDI spec 與 `--gpus all`。

### 16.3 測量契約（正式 run 之前寫定）

**組態**：同一台機器。`build-release/`：`-DSACCADE_WITH_OPENCV=OFF -DENABLE_NATIVE_TESTS=OFF -DTORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"`，只 build `saccade_track`，安裝到 `$R/tree`。`build/` 不 build 任何 target；oracle（`anchor`、`oracle-rows`）用 `build/` 既有的 extensions。第三方目錄 `$R/deps` 由 `check_shipping_tree.py deps` 從 gate 3 的 host run 產生（hard link）。容器：`ubuntu@sha256:786a8b55…`（`pristine`），以及它加上 `strace` 的映像（`strace`）；無網路，以非 root 使用者執行，`--gpus all`，`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`，tree、第三方目錄與 MOT17 train 以唯讀掛載。全部 GPU 步驟在 gpu0 lease 下依序執行。

**有效性**（任一不成立 ⇒ 受影響的 gate 為 `UNRESOLVED`）：正式 run 在乾淨的 commit 上；`build/libsaccade_scan_torchop.so` 的 sha256 在整個 run 前後都等於 attestation 的值；`anchor` 與 PR-2L `A_L_1` 7/7 相同、`oracle-rows` 有效（§13.3）；`container.txt` 顯示 Ubuntu 24.04、glibc 2.39，且 `python3`、`python`、`cc`、`gcc`、`c++`、`g++`、`clang`、`nvcc`、`ptxas` 都不存在。

**PASS 驗收規則**：PR-12 的 verdict 是 `PASS` 若且唯若下列全部成立，否則是 `FAIL`（照 gate 分開報告）：

1. **build 與安裝**：`build-release/` configure 與 build 成功，`saccade_track` POST_BUILD 通過；`cmake --install --component shipping` 成功（SM 清單檢查，以及 lineage、TorchScript head、backbone engine、operator library 四個檔案的 sha256 比對都通過）；tree 裡只有 `bin/saccade_track` 與 `share/saccade/` 的 6 個檔案（另兩個是 resolved config 與 attestation）。
2. **靜態檢查**（`check_shipping_tree.py static`，五項都 PASS）：
   - **G2-1**：從 `saccade_track` 與 operator library 出發的 NEEDED 閉包，依 `$ORIGIN` RUNPATH、第三方目錄、base system（glibc、GCC runtime、zlib）與 driver 解析，完整，而且沒有 `libpython*`、`libtorch_python*`；
   - **G2-3**：tree 裡沒有 `.py`／`.pyc`／`.pyo`／`.pth`、`__pycache__`、`site-packages`、`dist-packages`；
   - **RUNPATH**：PR-12 產出的每個 ELF（`bin/saccade_track`）只有 `$ORIGIN`-relative 的 RUNPATH、沒有 DT_RPATH；operator library 是唯一列舉的例外，它的 sha256 必須是 attestation 的 `aa84cccd…`（記錄它的絕對 RUNPATH）；
   - **SM＋PTX**：`saccade_track` 的 SASS 恰為 `sm_75/80/86/90/100/120`、PTX 恰為 `sm_120`（記錄 operator library 的 SASS／PTX）；
   - **glibc baseline**：tree 與第三方目錄裡每個 ELF 需要的 `GLIBC`／`GLIBCXX`／`CXXABI` 版本都不超過 2.39／3.4.33／1.3.15。
3. **第三方集合**：host 上以 `LD_DEBUG=files` 分別跑開發版 `build/shipping/saccade_track` 與 tree 的 `saccade_track`（後者的 `LD_LIBRARY_PATH`＝前者 RUNPATH 中存在的目錄，同樣順序），7 sequence。`check_shipping_tree.py deps --reference` 通過：兩者載入的第三方物件（以 sha256 比）相同、載入的 operator library 相同（tree 的那一份＝attestation 的那一份），且沒有載入 Python 函式庫。
4. **host 上的 tree binary**：`native_track_parity.py parity --track-binary $R/tree/bin/saccade_track --model-root $R/tree/share/saccade --track-library-path $R/deps`（double buffer、trace）`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7，且 `--against` PR-11 正式 run（`results/465_pr11_cmake/full_6502fbeb/parity/report.json`）：txt 與 trace 的 sha256 7/7 相同。
5. **乾淨容器**（`run_shipping_container.sh pristine`）：7 sequence 在同一個 process 中跑完、exit 0；`parity --native-from` `EXACT`（三個 section 同第 4 條），且 `--against` 第 4 條的 report：txt 與 trace 7/7 相同。
6. **G2-2／G2-4**（`run_shipping_container.sh strace`，7 sequence，exit 0）：`check_shipping_tree.py runtime` 三項都 PASS：
   - 恰好一次 `execve`／`execveat`，就是 entrypoint 本身且成功（任何其他 exec，包括失敗的嘗試，都算違反）；
   - 沒有任何對 `libpython*`、`libtorch_python*`、`libtriton*`、`.py`／`.pyc`、`site-packages`、`dist-packages`、`__pycache__`、`.triton`、`torchinductor*` 的 open（包括失敗的嘗試）；
   - 成功 open 的共享物件中，第三方目錄與 tree 裡的那些（以 sha256 比）恰好等於第 3 條開發版 run 載入的第三方物件與 operator library，而且沒有其他來源的第三方共享物件。
   
   另外 `parity --native-from` 對這次 run `EXACT`，且 txt 與 trace 與第 5 條 7/7 相同。第 4–6 條的 harness 也檢查 `track_report.json` 的 `python_libraries_mapped` 為空（`/proc/self/maps`）。

沒有容差。第 4–6 條的比較是逐位元組；任一不同就照 gate 報告第一個不同的檔案或項目，停在 PR-12。

**負控制**（記錄在結果目錄；每一個都必須被抓到）：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N1 | 以未設定 SM 清單的 `build-noopencv/` 安裝 | 安裝失敗（SM 清單檢查） |
| N2 | `SACCADE_ATTESTED_OP_LIBRARY` 指向多一個位元組的副本 | 安裝失敗（sha256） |
| N3 | tree 副本的 `bin/saccade_track` 換成開發版 binary | `static`：RUNPATH 與 SM＋PTX 失敗 |
| N4 | tree 副本多一個 `share/saccade/helper.py` | `static`：G2-3 失敗 |
| N5 | tree 副本的 operator library 多一個位元組 | `static`：RUNPATH 規則失敗（例外的 hash 不符） |
| N6 | 第三方目錄副本少 `libnvjpeg.so.13` | `static`：G2-1 失敗；`pristine` 容器 run exit 非 0 |
| N7 | tree 副本多一個需要 `GLIBC_2.44` 的 ELF（host 的 `libzvbi.so.0`） | `static`：glibc baseline 失敗 |
| N8 | `strace` 容器以 `/bin/sh -c` 包住 entrypoint（MOT17-05 前 5 幀） | `runtime`：exec 檢查失敗 |

另有 `tests/unit/test_shipping_g2_checks.py` 以假的 `readelf`／`cuobjdump`／strace 輸入驅動的拒絕案例。

**不做的**：FPS 或任何效能比較；`sm_120` 以外的 SASS／PTX 只檢查有編進去，沒有在其他 GPU 上執行（這台機器只有 `sm_120`）；bundle 第三方函式庫、改用較舊的 glibc／toolchain、shipping CLI 的開發選項（`--trace`／`--report`／`--measurement-mutation`）去留，都屬 Phase C。

### 16.4 驗收

同一台機器（RTX 5070 Ti Laptop，driver 616.92，WSL2），正式 run 在 `c44dd876`（工作樹乾淨；§16.3 的契約在同一個 commit，早於任何正式量測），全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library 在 run 前後都是 attestation 綁定的 `aa84cccd…`。

**r1 無效、r2 是正式結果。** 第一次執行（`full_c44dd876/`）的 gate 3 把 `LD_DEBUG_OUTPUT` 設成 `$R/ld_*/ld`，glibc 因此寫出 `ld_*/ld.<pid>`，解析步驟卻讀 `ld_*/ld/ld.*`；沒有產生第三方目錄，gate 2–6 與 N3–N8 都沒有在真的輸入上執行（容器把不存在的掛載來源當成空目錄，saccade_track 找不到函式庫）。r2（`full_c44dd876_r2/`）是同一份腳本只改這兩個前綴，同一個 commit，從頭重跑（包括重新 build `build-release/`）。r1 留在原處並附 `INVALID.txt`。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 工作樹乾淨；`anchor` 與 `A_L_1` 7/7 相同；`oracle-rows` OK；容器是 Ubuntu 24.04.4、glibc 2.39、libstdc++ 6.0.33，9 個工具都不存在 |
| 1 build 與安裝 | configure、build、install 都 exit 0；POST_BUILD link surface OK（13 NEEDED）；tree 7 個檔案 |
| 2 靜態檢查 | 五項 PASS。閉包 32 項（23 項在第三方目錄、9 項 base system），沒有 Python 函式庫。`saccade_track` 的 RUNPATH＝`$ORIGIN/../lib`；operator library（例外）保留 7 個絕對 RUNPATH 條目。`saccade_track` SASS `sm_75/80/86/90/100/120`＋PTX `sm_120`；operator library 只有 `sm_120` SASS、沒有 PTX。最高版本需求：`saccade_track` GLIBC 2.38／GLIBCXX 3.4.29／CXXABI 1.3.15，operator library 2.32／3.4.21／1.3.15，第三方目錄 2.28／3.4.22／1.3.11 |
| 3 第三方集合 | 兩個 run 各載入 44 個物件；第三方 27 個，sha256 集合相同，operator library 相同；base system 10 個（含 host 的 zlib）、driver 6 個 |
| 4 host 上的 tree binary | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；txt 與 trace 的 sha256 與 PR-11 正式 run 7/7 相同 |
| 5 乾淨容器 | exit 0；`EXACT`（同上）；與第 4 條 7/7 相同 |
| 6 G2-2／G2-4 | 三項 PASS：只有一次 `execve`（`/opt/saccade/bin/saccade_track`，成功）；沒有任何對 Python／Triton／inductor 路徑的 open；開啟的共享物件＝第 3 條開發版 run 的第三方集合與 operator library。`parity --native-from` `EXACT`，與第 5 條 7/7 相同；三次 run 的 `python_libraries_mapped` 都是空的 |
| **verdict** | **`PASS`**（§16.3 第 1–6 條全部成立） |

**負控制**（`negctl/`，全部抓到）：

| # | 結果 |
|:--|:--|
| N1 | 安裝失敗：`TORCH_CUDA_ARCH_LIST=''` 不是 shipping 清單 |
| N2 | 安裝失敗：operator library 的 sha256 `74bf95a5…` ≠ attestation。在那之前 `bin/saccade_track` 與其他 model root 檔案已經複製到目標目錄（安裝不是 atomic；被拒絕的 tree 不完整） |
| N3 | `static`：RUNPATH（9 個絕對條目）與 SM＋PTX（只有 `sm_120`）失敗，其餘通過 |
| N4 | `static`：G2-3 失敗（`share/saccade/helper.py`） |
| N5 | `static`：RUNPATH 規則失敗（例外的 sha256 不是 attestation 的） |
| N6 | `static`：G2-1 失敗（`libnvjpeg.so.13` 無法解析）；容器 run exit 127（loader 找不到 `libnvjpeg.so.13`） |
| N7 | `static`：glibc baseline 失敗（`lib/libzvbi.so.0` 需要 `GLIBC_2.44`） |
| N8 | `runtime`：exec 檢查失敗（`/bin/sh`、`saccade_track` 兩次 `execve`），其餘兩項通過 |

**觀察（不是 gate）**：`libnvrtc.so.13` 有被載入（operator library 的 NEEDED），但 `libnvrtc-builtins` 沒有被開啟，也就是這次 run 沒有 NVRTC JIT。容器的 base system（Ubuntu 的 glibc 2.39、libstdc++ 6.0.33、zlib 1.3）與 host（Arch 的 glibc 2.44、libstdc++ 6.0.36、zlib 1.3.2）不同，輸出仍逐位元組相同；這是這個組態下的觀察。

**run 之後的變更**：`run_shipping_container.sh` 在 tree 或第三方目錄不存在時直接拒絕（r1 的失敗模式：`docker -v` 會建立空的 root 目錄）。只影響這個開發工具，不影響任何量測。

結果目錄：`results/465_pr12_shipping/full_c44dd876_r2/`（`MANIFEST.md`、`run.sh`、`tree/`、`deps/`、`deps.json`、`static.json`、`runtime.json`、`ld_dev/`、`ld_tree/`、`anchor/`、`oracle_rows/`、`parity_host/`、`container_pristine/`、`parity_pristine/`、`container_strace/`、`parity_strace/`、`negctl/`）；r1 在 `results/465_pr12_shipping/full_c44dd876/`。不納入版本控制。

### 16.5 限制

- **GPU 可攜性**：`saccade_track` 帶 6 個 SM 的 SASS 與 `sm_120` PTX，但 operator library 只有 `sm_120` SASS，backbone engine 是為這張 GPU 與這個 TensorRT 版本 build 的。所以 tree 整體只能在 `sm_120` 上執行；本 PR 不主張超出所有載入的 CUDA artifact 都支援的硬體之外的可攜性。其他 SM 的 SASS／PTX 只確認有編進去，沒有在其他 GPU 上執行過。
- **operator library 的 RUNPATH**：它保留 7 個絕對的 build／venv 路徑（列舉的例外，以 hash 綁定）。在乾淨容器裡這些路徑不存在，它的 NEEDED 經 `LD_LIBRARY_PATH` 解析。要去掉它需要重新 build 與重新 attest（owner 決策 1）。
- **第三方函式庫不在 tree 裡**：容器由 `LD_LIBRARY_PATH` 取得 27 個第三方物件。loader 會讀這個環境變數（saccade_track 本身不讀任何環境變數）。是否 bundle、以什麼形式，是 Phase C。base system 依賴映像本身的 glibc、GCC runtime 與 zlib。
- **乾淨容器不是另一台機器**：同一張 GPU、同一個 driver（container toolkit 掛進去），只是 userland 換成沒有 Python、沒有編譯器的 Ubuntu 24.04。
- **glibc baseline** 是 Ubuntu 24.04（2.39），由這個 toolchain（GCC 16、glibc 2.44 的 host）build 出來的 `saccade_track` 需要 2.38 與 CXXABI 1.3.15；更低的 baseline 需要換 toolchain（Phase C）。
- **model root 的路徑**照抄 repository 的相對路徑（operator library 在 `share/saccade/build/`），因為 lineage 與 attestation 是凍結的。configure 仍需要 Python、pybind11、GStreamer（`developer_build_debug`）。shipping CLI 仍有開發選項（`--trace`、`--report`、`--measurement-mutation`）。這些都留給 Phase C。
- **安裝不是 atomic**（N2）：被拒絕時目標目錄可能留下部分檔案。
- G2-2／G2-4 是這一個 7-seq run 的 strace 觀察（`execve`／`execveat`／`open`／`openat`，含失敗的嘗試）。`vfork`／`posix_spawn` 最後也要經過 `execve`，所以有涵蓋；不經 `execve` 也不經 `open` 的程式碼載入（例如 `memfd`）不在觀察範圍內。另外，靜態連結進 binary 的直譯器不會出現在 NEEDED 或 open 裡：tree 的 `saccade_track` 的符號表（`nm`、`nm -D`）沒有任何 `Py*` 符號，字串裡唯一的 `libpython` 是 `detector_host.cpp` 檢查 `/proc/self/maps` 用的字面值（run 後補量，不是 §16.3 的 gate）。

### 16.6 重現

```bash
cmake -S . -B build-release -DSACCADE_WITH_OPENCV=OFF -DENABLE_NATIVE_TESTS=OFF \
    "-DTORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0+PTX"
cmake --build build-release --target saccade_track
cmake --install build-release --component shipping --prefix <tree>
# third-party set from an LD_DEBUG=files run (check_shipping_tree.py loaded / deps), then
.venv/bin/python scripts/native/check_shipping_tree.py static --tree <tree> --deps-manifest deps.json --deps-dir <deps> --report static.json
bash scripts/native/run_shipping_container.sh pristine|strace <tree> <deps> <out>
bash results/465_pr12_shipping/<label>/run.sh   # §16.3 gates 1-6 and N1-N8
```

## 17. Bundle：第三方集合進 tree、launcher、loader provenance check（Phase C PR-C1）

PR-C1 是 Phase C 的第一個 PR（[Phase C scope](native_runtime_phase_c_scope.md) §6）：把 PR-12 驗收過的 27 個第三方物件以原位元組放進 tree（owner 決策 C-D1），執行時不再需要 `LD_LIBRARY_PATH`。PR-C1 不改任何 stage 的計算，也不改 shipping 的 C++ 原始碼；entrypoint 與 operator library 的位元組都與 PR-12 相同。

| 項目 | 位置 |
|:--|:--|
| 第三方集合 pin | `shipping/third_party_set.json`（SONAME、sha256、來源 wheel 與 root、授權檔）；`shipping/THIRD_PARTY.md`；產生工具 `scripts/native/export_third_party_set.py` |
| entrypoint pin | `shipping/entrypoint_pin.json`；`SACCADE_SHIPPING_ENTRYPOINT` |
| launcher | `shipping/launcher/saccade_track.sh` → `bin/saccade_track` |
| loader provenance check | `shipping/src/loader_audit.c` → `lib/saccade_loader_audit.so` |
| 安裝規則 | `shipping/CMakeLists.txt`、`shipping/cmake/install_third_party.cmake` |
| 檢查工具 | `scripts/native/check_shipping_bundle.py`（`static`／`sources`／`runtime`）；`run_shipping_container.sh bundle`／`bundle-strace` |
| 測試 | `tests/unit/test_shipping_bundle_checks.py`（含在真的 loader 上跑 auditor 的案例） |

**owner 指示（2026-10-04，實作之前）**：先 probe glibc loader 的 `--library-path` 做法，再考慮改 `saccade_track`；優先保留 PR-12 executable 與 operator library 的位元組；`/proc/self/maps`／已載入物件的驗證是次要的 fail-closed provenance check；在 nvJitLink／cuFile／nvshmem 的散佈確認之前，bundle 只在本機 build 與測試；C-D5（簽章）＝v1 用 minisign。

### 17.1 Probe（開發觀察，不是正式 run）

`results/465_prc1_probe/p1_474da18b/`：bundle＝PR-12 r2 tree 與 deps 的 hard link，第三方物件放在 `lib/vendor/`；乾淨映像、MOT17-05-SDP 前 30 幀、`LD_DEBUG=files`。

| case | 呼叫方式 | exit | 第三方物件來源 |
|:--|:--|:--|:--|
| P0 | PR-12 方式（tree＋deps，`LD_LIBRARY_PATH`） | 0 | deps 27 |
| P1 | `ld.so --library-path <p>/lib/vendor`，無 `LD_LIBRARY_PATH` | 0 | vendor 27（含 operator library 的 `libnvrtc.so.13`） |
| P2 | 直接 exec ELF，無 `LD_LIBRARY_PATH` | 127 | `libnvinfer.so.10: cannot open` |
| P3 | P1＋`LD_LIBRARY_PATH` 指向一份不同的 libcublas | 0 | vendor 27（環境變數被 `--library-path` 取代） |
| P4 | P1＋在 `<p>/nvidia/cu13/lib` 放一份多一個位元組的 libcublas | 0 | **那一份被載入**，沒有任何訊息 |
| P5 | P1＋`--audit` auditor | 0 | vendor 27；載入集合＝P1＋auditor |
| P6 | P4＋`--audit` | 127 | `foreign copy on the search path: …/nvidia/cu13/lib/libcublas.so.13` |

P0、P1、P3、P4、P5 的 MOT txt 相同（`b3a1d9bb…`）。容器裡的 driver 物件是 7 個（P0 與 P1 相同，含 `libnvidia-ptxjitcompiler.so.1`）；§16.4 的 6 個是 host 上的數字。

讀法：

- `--library-path` 在 loader 的搜尋順序中排在 DT_RUNPATH 之前，所以 operator library（以 `dlopen` 載入、帶凍結的絕對 RUNPATH）的 `libnvrtc.so.13` 由 `lib/vendor` 解析：Phase C scope §5.1 的問題不必改 `saccade_track` 就解決。
- **Phase C scope §5 的更正**：torch 系列（`libtorch*`、`libc10*`、`libgomp`）、`libnvinfer.so.10`、`libnvshmem_host.so.3` 帶的是 **DT_RPATH**，不是 RUNPATH；DT_RPATH 排在 `--library-path` 之前（P4），所以 `--library-path` 單獨擋不住在那些位置被放進去的檔案。把第三方物件放在 `<prefix>/lib/vendor/`（往下兩層）之後，每個相對 RPATH／RUNPATH 都展開在 `<prefix>` 之內；唯一的例外是 `libcusparseLt.so.0` RUNPATH 結尾的空項（目前工作目錄）。
- auditor（rtld-audit，`--audit`）在載入之前就看得到每個搜尋候選路徑，不必改 `saccade_track` 的位元組，所以取代 scope §5.4 原本「在 `saccade_track` 裡讀 `/proc/self/maps`」的提案。

### 17.2 改了什麼

- **tree 的形狀**：`bin/saccade_track` 是 launcher（POSIX sh，只用 builtin）：`exec /lib64/ld-linux-x86-64.so.2 --library-path <prefix>/lib/vendor --audit <prefix>/lib/saccade_loader_audit.so --argv0 "$0" <prefix>/libexec/saccade_track "$@"`；呼叫端的 `LD_PRELOAD`、`LD_AUDIT`、`LD_LIBRARY_PATH` 先被 unset；prefix 含 `:` 時拒絕（loader 的路徑清單以 `:` 分隔）。ELF 移到 `libexec/saccade_track`，RUNPATH 仍是 `$ORIGIN/../lib`（`lib/` 只放 auditor，沒有第三方物件，所以直接 exec ELF 會在 loader 階段失敗，P2）。
- **entrypoint 的位元組**：同一份 entrypoint 原始碼、同樣的 flag 重新 build，得到的 `saccade_track` 與 PR-12 的不同（同大小，10644 個位元組不同，分布在 `.rela.dyn`、`.rela.plt`、`.gnu.version*`、`.strtab` 的 nvcc `tmpxft_*` 名稱與 build-id；只記錄，不歸因）。所以 tree 帶的是 PR-12 正式 run r2 安裝的那一份（`d7c6e0d4…`，`shipping/entrypoint_pin.json`），由 `SACCADE_SHIPPING_ENTRYPOINT` 指定、安裝時比對 sha256；未設定時安裝這次 build 的 `saccade_track`，`static` 檢查會報告它不是 pin。
- **第三方集合**：`install_third_party.cmake` 依 `third_party_set.json` 從 venv site-packages 與 FetchContent 的 nvJPEG wheel 複製 27 個物件到 `lib/vendor/<SONAME>`，逐檔比對 sha256；每個 wheel 的授權檔（也比對 sha256）到 `licenses/<wheel>/`，另放 `licenses/THIRD_PARTY.md` 與 Saccade 自己的 `LICENSE`／`NOTICE`。
- **auditor**（C，只 NEED `libc.so.6`，沒有 RUNPATH）：從自己的路徑推出 `<prefix>`。`la_objsearch`：名字屬於 bundle 集合的，`lib/vendor/` 以外的候選一律跳過，若該候選檔案存在就 exit 127；相對路徑的候選（不論名字）同樣處理。`la_objopen`：bundle 名字的物件（realpath）必須在 `lib/vendor/`，operator library 必須是 model root 的那一份，任何 `libpython*`／`libtorch_python*` 都 exit 127。不要求 symbol binding 事件（`la_objopen` 回傳 0）。它**不**比對 3.5 GiB 的位元組：位元組的完整性由安裝時的 sha256 與 `static` 檢查負責（PR-C3 的 MANIFEST 驗證之後也會）。
- **沒有動的**：shipping 與 tracking 的 C++ 原始碼、operator library、model root、PR-12 的檢查工具（`check_shipping_tree.py`）與它的組態（`run_shipping_container.sh pristine`／`strace` 不變）。
- **不散佈**：PR-C1 產生的 tree 只在本機 build 與測試；在 Phase C scope §4 的授權確認之前，不上傳、不公開任何 tree 或包。

### 17.3 開發期間已經看到的（在本節 commit 之前）

都是工作樹上的試做，不是正式 run：`results/465_prc1_dev/t2`、`t3` 以 pin 的 entrypoint 安裝；`static` 11 項通過；`bundle` 容器跑 7 sequence，exit 0，`LD_LIBRARY_PATH` 未設，MOT txt 7/7 與 PR-12 `container_pristine` 相同；`bundle-strace` 的 `runtime` 3 項通過。host 上經 launcher 跑 MOT17-05 前 30 幀，`sources` 通過（46 個物件，第三方全部來自 `lib/vendor`）。這期間修過兩處：auditor 原本只接受絕對路徑（host 以相對 `--model-root` 執行時誤判 operator library，改成 realpath）；`sources` 原本沿用 PR-12 的名字配對，但 launcher 的 sh 與它 exec 的 loader 共用 pid、寫進同一個 `LD_DEBUG` 檔，改成只讀 auditor 載入之後的 'calling init' 路徑，並以 SONAME 分類。

### 17.4 測量契約（正式 run 之前寫定）

**組態**：同一台機器。`build-release/`：§16.3 的 configure 參數，加 `-DSACCADE_SHIPPING_ENTRYPOINT=<PR-12 r2 tree>/bin/saccade_track`；build `saccade_track`（連帶 build auditor），安裝到 `$R/tree`。oracle（`anchor`、`oracle-rows`）用 `build/` 既有的 extensions，`build/` 不 build 任何 target。容器同 §16.3 的映像與設定，但 bundle 模式：只掛 tree（唯讀）、MOT17 train（唯讀）與輸出目錄，不掛第三方目錄，不設 `LD_LIBRARY_PATH`。全部 GPU 步驟在 gpu0 lease 下依序執行。

**有效性**（任一不成立 ⇒ 受影響的 gate 為 `UNRESOLVED`）：正式 run 在乾淨的 commit 上；`build/libsaccade_scan_torchop.so` 與 PR-12 r2 tree 的 `bin/saccade_track` 的 sha256 在整個 run 前後都分別等於 attestation 與 `entrypoint_pin.json` 的值；`anchor` 與 PR-2L `A_L_1` 7/7 相同、`oracle-rows` 有效（§13.3）；`container.txt` 顯示 Ubuntu 24.04、glibc 2.39、`LD_LIBRARY_PATH` 未設，且 §16.3 列的 9 個工具都不存在。

**PASS 驗收規則**：PR-C1 的 verdict 是 `PASS` 若且唯若下列全部成立，否則是 `FAIL`（照 gate 分開報告）：

1. **build 與安裝**：configure、build 成功，`saccade_track` POST_BUILD 通過；`cmake --install --component shipping` 成功（SM 清單、model root 4 個檔案、entrypoint pin、27 個第三方物件與全部授權檔的 sha256 都通過）。
2. **靜態檢查**（`check_shipping_bundle.py static`，11 項都 PASS）：`layout_exact`（tree 恰好是預期的檔案集合）、`vendor_set_pinned`、`entrypoint_pinned`、`launcher_exact`、`licenses`、`g2_3_no_python_files`、`produced_elves`（entrypoint 只有 RUNPATH `$ORIGIN/../lib`；auditor 只 NEED `libc.so.6`、沒有搜尋路徑；operator library 是 attestation 的）、`search_path_containment`（例外只有 operator library 的絕對 RUNPATH 與 `libcusparseLt.so.0` 的空項，各以 sha256 綁定）、`g2_1_needed_closure`（從 entrypoint 與 operator library 出發，第三方名字全部在 `lib/vendor`，其餘是 base system／driver，沒有 Python）、`sm_ptx_entrypoint`、`glibc_baseline`（tree 裡每個 ELF，含 `lib/vendor`）。
3. **host，經 launcher**：`LD_DEBUG=files` 下 `native_track_parity.py parity --track-binary $R/tree/bin/saccade_track --model-root $R/tree/share/saccade`（不給 `--track-library-path`；double buffer、trace）`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7，且 `--against` PR-12 正式 run 的 `parity_pristine/report.json`：txt 與 trace 的 sha256 7/7 相同；`check_shipping_bundle.py sources` 通過（27 個第三方物件全部來自 `$R/tree/lib/vendor` 且位元組是 pin 的，operator library 與 auditor 來自 tree，沒有其他第三方物件、沒有 Python 函式庫）。
4. **乾淨容器**（`run_shipping_container.sh bundle`）：7 sequence 在同一個 process 中跑完、exit 0；`parity --native-from` `EXACT`（三個 section 同第 3 條），且 `--against` 第 3 條的 report：txt 與 trace 7/7 相同。
5. **G2-2／G2-4**（`run_shipping_container.sh bundle-strace`，7 sequence，exit 0）：`check_shipping_bundle.py runtime` 三項都 PASS：
   - **exec chain**：恰好兩次 `execve`／`execveat`，都成功：先是 `/opt/saccade/bin/saccade_track`（launcher），再是 `/lib64/ld-linux-x86-64.so.2`，argv 以 `--library-path /opt/saccade/lib/vendor --audit /opt/saccade/lib/saccade_loader_audit.so` 開頭且含 `/opt/saccade/libexec/saccade_track`。這是 §16.3 第 6 條「恰好一次 `execve`」在 launcher 形式下的重新定義：兩次 exec 都不是 Python，sh 只執行 builtin；
   - 沒有任何對 Python／Triton／inductor 路徑的 open（同 §16.3，含失敗的嘗試）；
   - 成功 open 的 tree 內共享物件（以 sha256 比）恰好是 27 個 pin 的第三方物件＋operator library＋auditor；沒有任何 bundle 名字的物件從 `lib/vendor` 以外被開啟，也沒有 tree 以外的第三方物件。
   
   另外 `parity --native-from` 對這次 run `EXACT`，且 txt 與 trace 與第 4 條 7/7 相同。第 3–5 條的 harness 也檢查 `track_report.json` 的 `python_libraries_mapped` 為空。

沒有容差。第 3–5 條的比較是逐位元組；任一不同就照 gate 報告第一個不同的檔案或項目，停在 PR-C1。

**負控制**（tree 副本是 hard link；被改的檔案先刪再寫，不寫穿；每一個都必須被抓到）：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N1 | `SACCADE_SHIPPING_ENTRYPOINT` 指向這次 build 的 `saccade_track` | 安裝失敗（entrypoint pin） |
| N2 | tree 副本的 `lib/vendor/libcublas.so.13` 多一個位元組 | `static`：`vendor_set_pinned` 失敗 |
| N3 | tree 副本少 `lib/vendor/libnvrtc.so.13` | `static`：`layout_exact`、`vendor_set_pinned`、`g2_1_needed_closure` 失敗；容器 run（MOT17-05）exit 非 0 |
| N4 | tree 副本在 `nvidia/cu13/lib/` 多一份（多一個位元組的）`libcublas.so.13` | `static`：`layout_exact` 失敗；容器 run（MOT17-05）exit 127，訊息是 auditor 的 `foreign copy on the search path` |
| N5 | N4 的 tree，且 launcher 去掉 `--audit` 那一行 | `static`：`layout_exact`、`launcher_exact` 失敗；`bundle-strace`（MOT17-05）的 `runtime`：exec chain 與 opened set 失敗（那一份被開啟）。這一條顯示 runtime 擋下 N4 的是 auditor |
| N6 | 容器帶 `LD_LIBRARY_PATH` 與 `LD_PRELOAD` 指向一份不同的 libcublas（唯讀掛載），加 `LD_DEBUG=files` | `bundle`（MOT17-05）exit 0、MOT txt 與第 4 條的 MOT17-05 相同、loader log 沒有任何來自那個目錄的物件 |
| N7 | 容器直接 exec `/opt/saccade/libexec/saccade_track`（不經 launcher） | exit 127（loader 找不到 `libnvinfer.so.10`） |
| N8 | tree 副本多一個 `share/saccade/helper.py` | `static`：`layout_exact`、`g2_3_no_python_files` 失敗 |
| N9 | `bundle-strace` 映像以 `/bin/sh -c` 包住 launcher（MOT17-05 前 5 幀） | `runtime`：exec chain 失敗 |

另有 `tests/unit/test_shipping_bundle_checks.py` 的拒絕案例，以及在真的 loader 上：DT_RPATH 的候選優先於 `--library-path`（沒有 auditor 時被放進去的那一份會被載入），有 auditor 時 exit 127。

**修正 A1（2026-10-04，r1 之後、r2 之前）**：第一次正式 run（r1，`results/465_prc1_bundle/full_51840396/`，commit `51840396`）的 gate 1–5 全部成立（`EXACT`，與 PR-12 正式 run 7/7 相同），但有兩個負控制沒有照上表執行，所以 r1 不作為正式結果，整個 run 在 A1 的 commit 上從頭重跑（r2）：

- **N3**：容器那一半照預期失敗（exit 2，`dlopen … libnvrtc.so.13: cannot open`），但 `static` 沒有產出報告：`g2_1_needed_closure` 對 `lib/vendor` 裡不存在的物件跑 `readelf`，工具以 exit 2 結束。修正：閉包遇到不存在的 vendor 物件時回報 `NEEDED … is missing from lib/vendor`（加測試）。這只改檢查工具，不改 tree 或任何量測。
- **N6**：預測不成立。容器的 `LD_PRELOAD` 也作用在 launcher 自己的 sh 上：被 preload 的那份 `libcublas.so.13` 找不到它的 NEEDED `libcublasLt.so.13`，sh 在 loader 階段以 exit 127 結束，`saccade_track` 沒有啟動。這不是 provenance 被繞過（沒有任何東西以那份 libcublas 執行），但這個控制沒有測到它要測的東西：launcher 對 entrypoint process 的保護。N6 改為兩條：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N6a | 容器帶 `LD_LIBRARY_PATH` 指向一個放了多一個位元組的 `libcublas.so.13` 與 `libcudart.so.13` 的目錄（唯讀掛載），加 `LD_DEBUG=files` | `bundle`（MOT17-05）exit 0、MOT txt 與第 4 條的 MOT17-05 相同、auditor 載入之後的 loader log 沒有任何來自那個目錄的物件 |
| N6b | N6a 再加 `LD_PRELOAD` 指向那份 `libcudart.so.13`（它的 NEEDED 都是 base system，sh 能載入） | 同 N6a。另外記錄（不是 gate）：auditor 載入之前，也就是 launcher 的 sh，是否載入了那份 preload |

r1 的原 N6 結果（launcher 的 sh exit 127）記為觀察。其他 gate、負控制與判準不變。

**不做的**：FPS 或任何效能比較；其他 GPU、其他主機（原生 Linux）、其他 glibc；第三方位元組在執行時的雜湊（安裝與 `static` 負責）；CLI 的開發選項（PR-C2）；tarball、MANIFEST、atomic 安裝與 minisign 簽章（PR-C3／C4）；任何散佈。

### 17.5 驗收

同一台機器（RTX 5070 Ti Laptop，driver 616.92，WSL2）。正式結果是 **r2**：commit `16233e92`（含 A1，工作樹乾淨；§17.4 的契約在 `51840396`，A1 在 `16233e92`，都早於 r2 的任何量測），全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library（`aa84cccd…`）與 pin 的 entrypoint（`d7c6e0d4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json` 的值。r1（`full_51840396/`）留在原處並附 `SUPERSEDED.txt`（§17.4 A1）。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 工作樹乾淨；`anchor` 與 `A_L_1` 7/7 相同；`oracle-rows` OK；容器是 Ubuntu 24.04.4、glibc 2.39、`LD_LIBRARY_PATH` 未設，9 個工具都不存在 |
| 1 build 與安裝 | configure、build、install 都 exit 0；entrypoint 以 pin 安裝；tree 3.7 GiB |
| 2 靜態檢查 | 11 項 PASS。containment 的例外只有 `libcusparseLt.so.0` 的空項與 operator library 的 7 個絕對 RUNPATH 條目；閉包 32 項，第三方名字全部解析到 `lib/vendor` |
| 3 host，經 launcher | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；txt 與 trace 與 PR-12 正式 run（`parity_pristine`）7/7 相同；`sources` PASS（27 個第三方物件全部來自 `tree/lib/vendor` 且是 pin 的位元組，operator library 與 auditor 來自 tree） |
| 4 乾淨容器（bundle） | exit 0；`EXACT`（同上）；與第 3 條 7/7 相同 |
| 5 G2-2／G2-4 | 三項 PASS：exec chain＝`/opt/saccade/bin/saccade_track` → `/lib64/ld-linux-x86-64.so.2 --library-path /opt/saccade/lib/vendor --audit /opt/saccade/lib/saccade_loader_audit.so … /opt/saccade/libexec/saccade_track`，兩次都成功；沒有任何對 Python／Triton／inductor 路徑的 open；開啟的 tree 內共享物件＝27 個 pin 的第三方物件＋operator library＋auditor，沒有外來的。`parity --native-from` `EXACT`，與第 4 條 7/7 相同；三次 run 的 `python_libraries_mapped` 都是空的 |
| **verdict** | **`PASS`**（§17.4 第 1–5 條全部成立） |

**負控制**（`negctl/`，全部抓到）：

| # | 結果 |
|:--|:--|
| N1 | 安裝失敗：這次 build 的 `saccade_track` sha256 `25d65611…` ≠ pin `d7c6e0d4…` |
| N2 | `static`：`vendor_set_pinned` 失敗，其餘通過 |
| N3 | `static`：`layout_exact`、`vendor_set_pinned`、`g2_1_needed_closure` 失敗；容器 run exit 2（`dlopen …libsaccade_scan_torchop.so: libnvrtc.so.13: cannot open`） |
| N4 | `static`：`layout_exact` 失敗；容器 run exit 127，`loader provenance check failed: foreign copy on the search path: /opt/saccade/lib/vendor/../../nvidia/cu13/lib/libcublas.so.13` |
| N5 | `static`：`layout_exact`、`launcher_exact` 失敗；容器 run exit 0（那一份被載入），`runtime`：exec chain 失敗（loader 沒有 `--audit`）、opened set 失敗（`/opt/saccade/nvidia/cu13/lib/libcublas.so.13` 被開啟） |
| N6a | exit 0；MOT17-05 txt 與第 4 條相同；auditor 載入之後 47 個物件，沒有來自 `/opt/foreign` 的 |
| N6b | 同 N6a。觀察：launcher 的 sh 載入了 preload 的 `/opt/foreign/libcudart.so.13`（auditor 載入之前） |
| N7 | exit 127：`libnvinfer.so.10: cannot open shared object file` |
| N8 | `static`：`layout_exact`、`g2_3_no_python_files` 失敗 |
| N9 | `runtime`：exec chain 失敗（`/bin/sh`、launcher、loader 三次 `execve`），其餘兩項通過 |

**觀察（不是 gate）**：`saccade_track` 在三次 build 得到三個不同的 sha256（PR-12 r2 `d7c6e0d4…`、§17.3 開發期間 `a14b76fb…`、本 run 的 `build-release` `25d65611…`），原始碼都相同（§17.2）。

結果目錄：`results/465_prc1_bundle/full_16233e92/`（`run.sh`、`tree/`、`static.json`、`ld_host/`、`sources.json`、`anchor/`、`oracle_rows/`、`parity_host/`、`container_bundle/`、`parity_bundle/`、`container_strace/`、`runtime.json`、`parity_strace/`、`negctl/`、`pins_before.txt`／`pins_after.txt`）；probe 在 `results/465_prc1_probe/p1_474da18b/`，開發試做在 `results/465_prc1_dev/`。不納入版本控制。

### 17.6 限制

- **launcher 的 sh 不在保護範圍內**：呼叫端的 `LD_PRELOAD`／`LD_LIBRARY_PATH` 在 launcher 把它們 unset 之前就作用在 sh 自己身上（N6b 的觀察；r1 的原 N6 是一個依賴解析不到的 preload 讓 sh exit 127）。entrypoint process 不受影響（N6a、N6b）。
- **auditor 只管路徑，不管位元組**：執行時不雜湊 3.5 GiB 的第三方物件；被就地改寫的 `lib/vendor` 物件只會被安裝時的 sha256 與 `static` 抓到（N2）。安裝後的完整性驗證在 PR-C3 的 MANIFEST。
- **entrypoint 的位元組來自 PR-12 的結果目錄**：`SACCADE_SHIPPING_ENTRYPOINT` 指向 `results/465_pr12_shipping/full_c44dd876_r2/tree/bin/saccade_track`，那個目錄不在版本控制裡（operator library 在 `build/` 也是同樣的情況）。build 不可重現（§17.5 觀察），所以這份檔案若遺失，就要重新 build 並重做 parity，pin 也要換。
- **`libcusparseLt.so.0` 的空 RUNPATH 項**（工作目錄）：auditor 拒絕任何相對路徑的候選，但這一條只在 cusparseLt 以名字 dlopen 尚未載入的物件時才會用到，本 run 沒有觀察到這種搜尋。
- 仍然只支援 sm_120、只在這一台 WSL2 機器與這個 driver 上驗證過；包約 3.7 GiB；安裝仍不是 atomic（PR-C3）；CLI 的開發選項仍在（PR-C2；parity harness 每次都傳 `--measurement-mutation`）。
- **沒有散佈**：tree 只在本機；nvJitLink／cuFile／nvshmem 的散佈依據確認之前不公開（Phase C scope §4）。

### 17.7 重現

```bash
cmake -S . -B build-release -DSACCADE_WITH_OPENCV=OFF -DENABLE_NATIVE_TESTS=OFF \
    "-DTORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0+PTX" \
    -DSACCADE_SHIPPING_ENTRYPOINT=$PWD/results/465_pr12_shipping/full_c44dd876_r2/tree/bin/saccade_track
cmake --build build-release --target saccade_track
cmake --install build-release --component shipping --prefix <tree>
.venv/bin/python scripts/native/check_shipping_bundle.py static --tree <tree> --report static.json
bash scripts/native/run_shipping_container.sh bundle|bundle-strace <tree> <out>
bash results/465_prc1_bundle/<label>/run.sh   # §17.4 gates 1-5 and the negative controls
```

### 17.8 修正 A2：auditor 必須初始化成功

launcher 與 auditor 的原始碼註解以「§17.8, A2」指到這裡。修正案、契約修正與正式 run 結果在 §18.10–§18.11：A2 改的是 PR-C1 的 launcher 與 auditor，但正式 run 是在 PR-C1＋PR-C2 合在一起的 head 上重跑 §18.4 的整個 run。§17.4 與 §17.5 記錄的是舊 launcher（沒有 probe）。

## 18. Shipping CLI：移除開發選項、measurement 邊界（Phase C PR-C2）

PR-C2 是 Phase C 的第二個 PR（[Phase C scope](native_runtime_phase_c_scope.md) §6）：shipping 的 `saccade_track` 不再有任何開發／量測選項，也不含任何 mutation 程式碼；負控制注入只存在於一個明確、不安裝的 developer build。PR-C2 不改任何 stage 的計算，也不改 engine、operator library、27 個第三方物件的閉包、launcher／auditor、`sm_120` 與 Ubuntu 24.04 的契約。entrypoint 的原始碼改了，所以它的位元組改變，pin 換成新的一份（§18.4）。

| 項目 | 位置 |
|:--|:--|
| shipping entrypoint | `shipping/tools/saccade_track.cpp`（只有 shipping 介面）；共用的 sequence loop／trace／report：`shipping/tools/track_driver.hpp` |
| developer build | `shipping/tools/saccade_track_measurement.cpp` → `build*/shipping/saccade_track_measurement`（不安裝） |
| measurement 邊界 | `SACCADE_SHIPPING_MEASUREMENT_HOOKS`；`shipping/CMakeLists.txt` 的 `saccade_shipping_variants()`：`saccade_shipping_{native,ingest,detector,runtime}` 與各自的 `_measurement` |
| 禁止的位元組字串 | `shipping/measurement_surface.json`；POST_BUILD `shipping/cmake/check_no_measurement_surface.cmake`；`check_shipping_bundle.py static` 的 `entrypoint_no_measurement_surface` |
| 被拒絕的選項 | `check_shipping_bundle.py rejected`（launcher 的 strace log） |
| harness | `native_track_parity.py parity --entrypoint shipping|measurement` |
| 測試 | `tests/unit/test_shipping_measurement_surface.py`、`tests/unit/test_shipping_bundle_checks.py`、`tests/unit/test_native_track_parity.py`；GPU：`tests/native/test_shipping_{serial,double_buffer}_runtime.cpp`、`test_shipping_post_detector_host.cpp`（改連 `_measurement`） |

**owner 指示（2026-10-04，實作之前）**：`--measurement-mutation` 不再被 shipping `saccade_track` 接受，且在載入模型／GPU 之前 fail closed；盤點 CLI，只保留正式 shipping 選項，developer／test-only 功能移到明確的非 shipping 路徑；parity harness 不再依賴 production CLI 的 mutation flag，負控制仍要保留；不改 engine、operator library、27-lib closure、loader／auditor、`sm_120`、Ubuntu 24.04 contract；正式 run 之前凍結新的 CLI／harness 契約，確認 entrypoint／operator library 的 hash 不變；全量驗收仍需 detector 5316/5316 EXACT、MOT 7/7 byte-identical、graph captures 7/7 identical，並與 PR-12 正式 run 完全一致；runtime identity 有變更就照 C1 做 stacked republish。核心原則：不要把 `--measurement-mutation` 換成另一個 hidden flag，shipping path 必須完全沒有 mutation capability。

§13／§14 的負控制寫的是 `saccade_track --measurement-mutation`：PR-C2 之後是 `saccade_track_measurement --measurement-mutation`（harness 的 `--entrypoint measurement`）；那兩節的內容是當時的紀錄，不改寫。

### 18.1 CLI 盤點

| 選項 | PR-C1 | PR-C2 | 理由 |
|:--|:--|:--|:--|
| `--config`、`--lineage`、`--attestation`、`--model-root`、`--out`、`SEQUENCE_DIR...` | shipping | shipping，不變 | entrypoint 的輸入與輸出位置（boundary §2） |
| `--report JSON` | 「developer measurement」 | **shipping** | 只寫檔、不改計算；驗收讀它的 load report、graph 計數與 `python_libraries_mapped`（Phase C scope §2 的預設提案）。格式改為 `saccade.native_track_report/v2`：加 `entrypoint`，移除 `mutation` 與 `max_frames` |
| `--trace DIR` | 「developer measurement」 | **shipping** | 只寫檔、不改計算；detector 5316/5316 的驗收要在 shipping binary 上讀它（scope §2）。每幀多一次 main stream 的 sync（§14 的限制，不變） |
| `--max-frames N` | 開發選項 | **移到 developer build** | 只為了短跑測試：把輸入截成前 N 幀，截斷處的 sequence tail 不是 shipping 的語意；oracle 的 `--max-frames` 也是 eval 腳本的測試選項 |
| `--schedule serial` | 開發選項 | **移到 developer build** | 覆寫 config 的排程。shipping 的排程只由 resolved config 決定（`select_schedule(cfg, false)`）；config 本身若指名 serial，shipping 仍走 `SerialRuntime` |
| `--measurement-mutation M` | 開發選項（負控制） | **移到 developer build** | 負控制注入（owner 指示） |

`--max-frames` 與 `--schedule serial` 的去留是本 PR 的決定（scope §2 只寫了 `--measurement-mutation`），owner review 時可以改。shipping binary 遇到這三個（或任何其他未列出的）選項，在 `parse_args` 就以 exit 2 結束（`saccade_track: unknown argument <option>`），之前不讀任何檔案；用法訊息只列 shipping 介面。

### 18.2 改了什麼

- **measurement 邊界**：四個帶有量測 hook 的 runtime library 各 build 兩次，同一份原始碼：`<name>`（shipping）與 `<name>_measurement`（PUBLIC `SACCADE_SHIPPING_MEASUREMENT_HOOKS=1`，所以使用端看到相同的宣告）。只在 `#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS` 之內宣告與實作的是：`DetectorMutation`（6 種）與 `DetectorHost::set_mutation_for_measurement`；`detect_graphed` 原本的 `refresh_input` 參數改成 `DetectorHost::set_stale_graph_input_for_measurement`（double-buffer runtime 每個 sequence 設一次，語意與原本逐呼叫傳入相同）；`JpegDecoder::force_decoupled_for_measurement`；`PostDetectorHost::set_pre_roll_for_measurement`、`set_stale_gmc_input_for_measurement`；`RuntimeMutation`（3 種）與 `DoubleBufferMutation`（3 種）以及它們的 `parse_*`／`*_name`、setter、成員與分支。shipping 物件裡這些全都不存在：shipping 的呼叫端無法指名它們（沒有宣告），mutation 為 `none` 時 measurement variant 執行的敘述與 shipping 相同。
- **兩個 entrypoint**：`saccade_track.cpp` 只解析 shipping 介面（`track_driver.hpp` 的 `parse_interface_arg`），以 `#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS` `#error` 拒絕以 measurement variant 編譯；`saccade_track_measurement.cpp` 在同一個介面之外加 `--measurement-mutation`、`--schedule serial`、`--max-frames`，以 `#ifndef` `#error` 拒絕以 shipping variant 編譯，report 帶 `entrypoint: "saccade_track_measurement"` 與 `measurement`（`mutation`、`schedule_override`、`max_frames`）。兩者共用 `track_driver.hpp`（不含任何依賴巨集的程式碼）。`saccade_track_measurement` 不在任何 `install()` 裡。
- **使用端**：`saccade_replay`、`saccade_ingest_probe`、`saccade_detector_probe` 與用到 hook 的 GPU 測試（serial／double-buffer runtime、post-detector host）改連 `_measurement`；其餘 GPU 測試連 shipping variant。沒有任何 target 同時連兩種 variant。
- **建置時與安裝後的檢查**：`saccade_track` 的 POST_BUILD 在 link surface 之外再跑 `check_no_measurement_surface.cmake`：binary 含 `measurement_surface.json` 的任一字串就 build 失敗。字串是 mutation 的列舉、名稱與 setter 的片段（`Mutation`、`_mutation`、` mutation`、`mutation_`、`_for_measurement`）、developer build 的名字與三個選項、12 個 mutation 名稱、`stale_graph_input`、`force_decoupled`。`check_shipping_bundle.py static` 多一項 `entrypoint_no_measurement_surface`（共 12 項）。新的 `rejected` 子命令判讀 launcher 收到被移除的選項時的 strace log：exit 2 與 `unknown argument` 訊息、exec chain 仍是 launcher → loader、沒有開啟（含失敗的嘗試）model root 之下任何檔案或 operator library、沒有開啟 GPU 裝置節點（`/dev/dxg`、`/dev/nvidia*`）、沒有任何輸出。
- **harness**：`--entrypoint shipping`（預設）只給 `saccade_track` 7 個介面選項；`--mutation`、`--schedule serial`、`--max-frames` 必須搭配 `--entrypoint measurement`，否則 argparse 拒絕。report 的 validity 檢查 `entrypoint` 是要求的那一個、shipping report 沒有 `measurement` 記錄、measurement report 的記錄等於要求。`--against` 除了 txt 與 trace 的 sha256，也比較每個 sequence 的 native graph capture／replay 計數。
- **沒有動的**：engine、operator library（`aa84cccd…`）、model root、`third_party_set.json`（27 個）、launcher、auditor、安裝規則（除了 entrypoint 的 pin 值）、SM 清單、glibc baseline、容器映像與 `run_shipping_container.sh`。

### 18.3 開發期間已經看到的（在本節 commit 之前）

都是工作樹上的試做，不是正式 run（`results/465_prc2_dev/t1/`，oracle 借用 PR-C1 r2 的 `anchor`／`oracle_rows`，所以 harness 的 verdict 是 `UNRESOLVED`：oracle 不在這個 commit）：

- **位元組字串**（`measurement_surface.json` 的 23 個）：`build-release` 與 `build/`（Debug）的新 shipping `saccade_track` 都是 0 個；PR-12 pin 的 `d7c6e0d4…` 有 20 個（例如 `_mutation` 10 次、`_for_measurement` 4 次、`--measurement-mutation`、`--max-frames` 與全部 12 個 mutation 名稱；`nm` 看得到 `DetectorHost::set_mutation_for_measurement`、`parse_detector_mutation` 等符號）；`saccade_track_measurement` 在 `build-release` 有 22 個、Debug 有 23 個。第一版清單用了 `measurement` 與 `mutation` 兩個泛用字，Debug 的 shipping `saccade_track` 因此被 POST_BUILD 誤判：tracker 的 Kalman filter 有 `measurement`、`output_measurement`，config 檢查有 `PermutationOf`（含 `mutation`）。改成上列的精確片段。
- **parity**（gpu0 lease）：新 shipping binary `EXACT`（`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7），`--against` PR-12 正式 run 的 `parity_pristine`：txt、trace、graph 計數 7/7 相同；`--entrypoint measurement`（mutation `none`）與 shipping 7/7 相同；`stale_gmc_input` 的 `mot_txt` 是 `DIFFERS`。
- **被拒絕的選項**（乾淨容器＋strace，dev tree）：`--measurement-mutation none`、`--schedule serial`、`--max-frames 5` 三者的 `rejected` 5 項都通過；對 PR-C1 r2 一次正常 run 的 strace log 跑 `rejected`，exit／model root／GPU 裝置／輸出 4 項失敗、exec chain 通過。`static` 在 dev tree 上除了 `entrypoint_pinned`（pin 仍是 PR-12 的）之外 11 項通過。
- **GPU 測試**：`build/` 的 `ctest -R 'saccade_shipping|saccade_resolved'` 12/12 通過（含 serial／double-buffer runtime 的 mutation 測試，改連 `_measurement`）。`build/libsaccade_scan_torchop.so` 在 build 前後都是 `aa84cccd…`（只 build 具名 target）。
- **「GPU 之前」的定義**：`LD_DEBUG=files` 顯示即使參數錯誤，`libcuda.so.1` 也在 `main` 之前被 `libcublasLt.so.13` 的初始化以 `dlopen` 載入，所以「在 GPU 之前 fail closed」不能以 `libcuda` 沒有載入來判定；改以「沒有開啟 GPU 裝置節點、沒有開啟 model root 與 operator library」判定（正常 run 會開啟 `/dev/dxg`、`/dev/nvidia-uvm`）。

### 18.4 測量契約（正式 run 之前寫定）

**entrypoint pin**：`shipping/entrypoint_pin.json` 換成 `92f74ef4724ff5e4cb5e7d4c7ce7e799563cec59b7b1ad806e682a9a95ceff34`（8736640 位元組）：commit `04f6f135`（工作樹乾淨）以全新的 `build-prc2-pin/`（§17.4 的 configure 參數，不設 `SACCADE_SHIPPING_ENTRYPOINT`）build `saccade_track`、`cmake --install` 之後的 `libexec/saccade_track`（RUNPATH `$ORIGIN/../lib`），放在 `results/465_prc2_cli/entrypoint_04f6f135/saccade_track`（唯讀）。它的 POST_BUILD 兩項都通過，裝出來的 tree 上 `static` 除 `entrypoint_pinned`（當時 pin 仍是 PR-12 的）之外 11 項通過。build 不可重現（§17.2），所以正式 run 安裝的是這一份，不是 run 中重新 build 的那一份。

**組態**：同一台機器。`build-release/`：§17.4 的 configure 參數，`-DSACCADE_SHIPPING_ENTRYPOINT=<上面那份>`；build `saccade_track` 與 `saccade_track_measurement`，安裝到 `$R/tree`。developer build 是這次 run 在 `build-release/` 編出的 `saccade_track_measurement`（不安裝，以 build tree 的 RUNPATH 在 host 上執行）。oracle（`anchor`、`oracle-rows`）用 `build/` 既有的 extensions，`build/` 不 build 任何 target。容器同 §17.4（bundle 模式）。全部 GPU 步驟在 gpu0 lease 下依序執行。

**有效性**（任一不成立 ⇒ 受影響的 gate 為 `UNRESOLVED`）：正式 run 在乾淨的 commit 上；`build/libsaccade_scan_torchop.so` 與 pin 檔案的 sha256 在整個 run 前後都分別等於 attestation 與 `entrypoint_pin.json` 的值；編出 pin 的原始碼不變：`git diff 04f6f135 HEAD -- shipping/src shipping/include shipping/tools src include` 為空，`shipping/CMakeLists.txt` 與 `shipping/cmake/` 的差異只有註解行；`anchor` 與 PR-2L `A_L_1` 7/7 相同、`oracle-rows` 有效（§13.3）；`container.txt` 同 §17.4。

**PASS 驗收規則**：PR-C2 的 verdict 是 `PASS` 若且唯若下列全部成立，否則是 `FAIL`（照 gate 分開報告）：

1. **build 與安裝**：同 §17.4 第 1 條；另外 `saccade_track` 的 POST_BUILD 兩項（link surface、measurement surface）都通過，`saccade_track_measurement` build 成功。
2. **靜態檢查**：`check_shipping_bundle.py static` 12 項都 PASS（§17.4 的 11 項＋`entrypoint_no_measurement_surface`）。
3. **host，經 launcher**：同 §17.4 第 3 條（`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；`sources` 通過），且 `--against` PR-12 正式 run 的 `parity_pristine/report.json`：txt、trace 與 native graph 計數 7/7 相同。harness 只給 7 個介面選項；report 的 `entrypoint` 是 `saccade_track`，沒有 `measurement` 記錄。
4. **乾淨容器**：同 §17.4 第 4 條，`--against` 第 3 條：txt、trace、graph 計數 7/7 相同。
5. **G2-2／G2-4**：同 §17.4 第 5 條，`--against` 第 4 條 7/7 相同。
6. **CLI surface**：
   - **a. 被移除的選項**：乾淨容器＋strace（`bundle-strace` 的映像與掛載），launcher 以第 4 條的完整參數加上 `--measurement-mutation none`、`--schedule serial`、`--max-frames 5` 之一（MOT17-05），三次的 `check_shipping_bundle.py rejected` 5 項都 PASS：exit 2 且訊息是 `saccade_track: unknown argument <option>`；exec chain 是 launcher → loader；沒有開啟（含失敗的嘗試）`/opt/saccade/share/saccade/` 之下任何路徑或 `libsaccade_scan_torchop.so`；沒有開啟 `/dev/dxg`、`/dev/nvidia*`；沒有 `native/` 或 `track_report.json`。
   - **b. developer build 的基準**：`parity --entrypoint measurement`（mutation `none`、double buffer、trace）`EXACT`，且 `--against` 第 3 條：txt、trace、graph 計數 7/7 相同。負控制的基準因此就是 shipping 的計算。
   - **c. serial 參考**：`parity --entrypoint measurement --schedule serial` 對 `oracle-rows`（serial oracle）`EXACT`（`detector` 5316/5316、`mot_txt` 7/7）。
   - **d. 負控制保留**：每一個都 `CAUGHT`（指定的 section 是 `DIFFERS`、沒有 validity 問題），7 sequence：double buffer `stale_detector_input` → `detector`、`stale_gmc_input` → `mot_txt`、`swapped_detection_parity` → `detector`；serial `shared_post_host` → `mot_txt`、`stale_image_dims` → `detector`、`gmc_previous_frame` → `mot_txt`（`--entrypoint measurement`）；`--ref-edit` → `mot_txt`（shipping，經 launcher）。

沒有容差。第 3–6 條的比較是逐位元組（graph 計數逐項相等）；任一不同就照 gate 報告第一個不同的檔案或項目，停在 PR-C2。

**負控制**（新的檢查本身；tree 副本是 hard link，被改的檔案先刪再寫；每一個都必須被抓到）：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| M1 | 對 PR-C1 r2 的 tree（PR-12 的 entrypoint）跑 `static` | `entrypoint_no_measurement_surface` 與 `entrypoint_pinned` 失敗（舊的 surface 被抓到；pin 已換） |
| M2 | tree 副本的 `libexec/saccade_track` 換成這次 run 的 `saccade_track_measurement` | `static`：`entrypoint_no_measurement_surface`、`entrypoint_pinned` 失敗（其餘照實記錄） |
| M3 | `check_no_measurement_surface.cmake` 對 `saccade_track_measurement` | 失敗（POST_BUILD 會擋下連到 `_measurement` 的 shipping build） |
| M4 | `rejected` 對第 5 條（正常 run）的 strace log | `exit_2_unknown_argument`、`no_model_root_open`、`no_gpu_device_open`、`no_output` 失敗 |
| M5 | `parity --mutation stale_gmc_input`（沒有 `--entrypoint measurement`） | harness 在執行任何東西之前拒絕（exit 2） |
| M6 | `parity --entrypoint shipping --track-binary build-release/shipping/saccade_track_measurement` | `UNRESOLVED`（report 的 `entrypoint` 不是 `saccade_track`） |

另外以新的 pin 重跑 §17.4 的 N1–N5、N6a、N6b、N7–N9，必須的結果同 §17.4（N1 的「這次 build 的 `saccade_track`」是 `build-release` 的那一份）；N7、N9 原本帶 `--max-frames 5`，這個選項已經不在 shipping 介面裡，改跑完整的 MOT17-05。

**不做的**：FPS 或任何效能比較；其他 GPU、其他主機、其他 glibc；`saccade_track_measurement` 的容器驗收（它不安裝）；tarball、MANIFEST、atomic 安裝與 minisign 簽章（PR-C3／C4）；任何散佈。

### 18.5 驗收

同一台機器（RTX 5070 Ti Laptop，driver 616.92，WSL2）。正式 run：commit `e63fde3f`（工作樹乾淨；§18.4 的契約與新的 pin 在這個 commit，早於任何量測），全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library（`aa84cccd…`）與 pin 的 entrypoint（`92f74ef4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json` 的值；`git diff 04f6f135 HEAD -- shipping/src shipping/include shipping/tools src include` 為空，`shipping/CMakeLists.txt`／`shipping/cmake/` 的差異只有註解行（`validity_sources.txt`）。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 工作樹乾淨；pin 的原始碼不變；`anchor` 與 `A_L_1` 7/7 相同；`oracle-rows` OK；容器是 Ubuntu 24.04.4、glibc 2.39、`LD_LIBRARY_PATH` 未設，9 個工具都不存在 |
| 1 build 與安裝 | configure、build、install 都 exit 0；`saccade_track` 的 POST_BUILD 兩項通過（13 NEEDED；measurement surface 23 個字串都不在）；`saccade_track_measurement` build 成功；entrypoint 以 pin 安裝；tree 3.7 GiB |
| 2 靜態檢查 | 12 項 PASS（含 `entrypoint_pinned` 與 `entrypoint_no_measurement_surface`） |
| 3 host，經 launcher | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；`--against` PR-12 `parity_pristine`：txt、trace、graph 計數 7/7 相同；`sources` PASS；report 的 `entrypoint` 是 `saccade_track`，沒有 `measurement` 記錄 |
| 4 乾淨容器（bundle） | exit 0；`EXACT`（同上）；與第 3 條 7/7 相同 |
| 5 G2-2／G2-4 | `runtime` 三項 PASS；`parity --native-from` `EXACT`，與第 4 條 7/7 相同；三次 run 的 `python_libraries_mapped` 都是空的 |
| 6a 被移除的選項 | `--measurement-mutation none`、`--schedule serial`、`--max-frames 5`：三次 `rejected` 各 5 項 PASS（exit 2、`unknown argument`；exec chain launcher → loader；沒有開啟 model root、operator library、`/dev/dxg`／`/dev/nvidia*`；沒有輸出） |
| 6b developer build 的基準 | `--entrypoint measurement`（mutation `none`）`EXACT`，與第 3 條：txt、trace、graph 計數 7/7 相同 |
| 6c serial 參考 | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7 |
| 6d 負控制 | 7 個都 `CAUGHT`：`stale_detector_input`（`detector`、`mot_txt` DIFFERS）、`stale_gmc_input`（`mot_txt`）、`swapped_detection_parity`（`detector`、`mot_txt`）、`shared_post_host`（`mot_txt`）、`stale_image_dims`（`detector`、`mot_txt`）、`gmc_previous_frame`（`mot_txt`）、`ref_edit`（`mot_txt`）；double buffer 的三個 `graph_captures` 都是 `EXACT` |
| **verdict** | **`PASS`**（§18.4 第 1–6 條全部成立） |

**負控制**（`negctl/`，全部抓到）：

| # | 結果 |
|:--|:--|
| M1 | PR-C1 r2 的 tree：`static` 的 `entrypoint_no_measurement_surface`、`entrypoint_pinned` 失敗，其餘通過 |
| M2 | entrypoint 換成 `saccade_track_measurement`：`entrypoint_no_measurement_surface`、`entrypoint_pinned` 失敗；另外 `produced_elves`、`search_path_containment` 失敗（build tree 的 RUNPATH 指向 venv，不是 `$ORIGIN/../lib`） |
| M3 | POST_BUILD 的 surface 檢查對 `saccade_track_measurement` exit 1 |
| M4 | `rejected` 對第 5 條的正常 run：`exit_2_unknown_argument`、`no_model_root_open`、`no_gpu_device_open`、`no_output` 失敗（exec chain 通過） |
| M5 | harness 以 exit 2 拒絕（`use --entrypoint measurement`），沒有執行任何東西 |
| M6 | `UNRESOLVED`：`report entrypoint 'saccade_track_measurement' != 'saccade_track'`、`a shipping saccade_track report with a measurement record`（三個 section 本身 `EXACT`，validity 擋下） |
| N1 | 安裝失敗：這次 build 的 `saccade_track`（`83005519…`）≠ pin |
| N2 | `static`：`vendor_set_pinned` 失敗 |
| N3 | `static`：`layout_exact`、`vendor_set_pinned`、`g2_1_needed_closure` 失敗；容器 run exit 2（`dlopen …libsaccade_scan_torchop.so: libnvrtc.so.13: cannot open`） |
| N4 | `static`：`layout_exact` 失敗；容器 run exit 127，`foreign copy on the search path: /opt/saccade/lib/vendor/../../nvidia/cu13/lib/libcublas.so.13` |
| N5 | `static`：`layout_exact`、`launcher_exact` 失敗；容器 run exit 0，`runtime`：exec chain 與 opened set 失敗 |
| N6a | exit 0；MOT17-05 txt 與第 4 條相同；auditor 載入之後 47 個物件，沒有來自 `/opt/foreign` 的 |
| N6b | 同 N6a。觀察：launcher 的 sh 載入了 preload 的 `/opt/foreign/libcudart.so.13`（auditor 載入之前，同 §17.5） |
| N7 | exit 127：`libnvinfer.so.10: cannot open shared object file`（完整 MOT17-05） |
| N8 | `static`：`layout_exact`、`g2_3_no_python_files` 失敗 |
| N9 | `runtime`：exec chain 失敗，其餘兩項通過（完整 MOT17-05） |

**觀察（不是 gate）**：本 run 在 `build-release` 編出的 `saccade_track` 是 `83005519…`、`saccade_track_measurement` 是 `e6d98c84…`；pin（`build-prc2-pin`，同一份原始碼）是 `92f74ef4…`。

結果目錄：`results/465_prc2_cli/full_e63fde3f/`（`run.sh`、`validity_sources.txt`、`tree/`、`static.json`、`anchor/`、`oracle_rows/`、`parity_host/`、`ld_host/`、`sources.json`、`container_bundle/`、`parity_bundle/`、`container_strace/`、`runtime.json`、`parity_strace/`、`cli/`、`negctl/`、`pins_before.txt`／`pins_after.txt`）；pin 在 `results/465_prc2_cli/entrypoint_04f6f135/`，開發試做在 `results/465_prc2_dev/`。不納入版本控制。

### 18.6 限制

- **「沒有 mutation capability」的證據是三層，不是證明**：(1) 原始碼：每一個 hook 都在 `#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS` 之內，shipping 的 entrypoint 以 `#error` 拒絕 measurement variant（`test_shipping_measurement_surface.py` 逐行檢查 hook 名稱）；(2) binary：pin 不含 `measurement_surface.json` 的 23 個字串（POST_BUILD、`static`）；(3) 行為：被移除的選項在讀任何檔案之前被拒絕（6a）。字串檢查只能抓到清單上的名字；一個不帶這些名字的新 hook 要靠第 (1) 層與 review。
- **resolved config 仍然是輸入**：shipping binary 依 config 計算；config 由 strict loader 驗證與排程檢查把關，但不在 PR-C2 的範圍內以 hash 綁定（PR-C3 的 MANIFEST 會記錄它的 sha256）。這是輸入，不是注入 hook。
- **`--trace` 留在 shipping**：它只寫檔，但每幀多一次 main stream 的 sync（§14），不是 shipping 的預設組態；不帶 `--trace` 的 shipping run 在本 PR 沒有另外量（同 §14 以來的狀態）。
- **launcher 的註解**仍寫「the PR-12 executable, bytes unchanged」：launcher 的位元組被 `launcher_exact` 綁定，PR-C2 不改 launcher，所以這句註解留到下一次必須改 launcher 的時候。
- **pin 來自結果目錄**：`results/465_prc2_cli/entrypoint_04f6f135/saccade_track` 不在版本控制裡（同 §17.6 對 PR-12 pin 的限制）；build 不可重現，遺失就要重新 build、重做 parity 並換 pin。
- `--max-frames` 與 `--schedule serial` 移出 shipping 是本 PR 的決定（§18.1），owner review 時可以改；改回 shipping 需要重新寫 surface 清單與這一節的驗收。
- 其餘同 §17.6：只支援 sm_120、只在這一台 WSL2 機器與 driver 上驗證；launcher 的 sh 不在保護範圍；安裝仍不是 atomic（PR-C3）；沒有散佈。

### 18.7 重現

```bash
# the pin (once; build at the implementation commit on a clean tree)
cmake -S . -B build-prc2-pin -DSACCADE_WITH_OPENCV=OFF -DENABLE_NATIVE_TESTS=OFF \
    "-DTORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0+PTX"
cmake --build build-prc2-pin --target saccade_track
cmake --install build-prc2-pin --component shipping --prefix <stage>   # keep <stage>/libexec/saccade_track
# the formal run
bash results/465_prc2_cli/<label>/run.sh   # §18.4 gates 1-6, M1-M6, N1-N9
# by hand
.venv/bin/python scripts/native/check_shipping_bundle.py static --tree <tree> --report static.json
.venv/bin/python scripts/native/check_shipping_bundle.py rejected --strace-prefix <out>/strace/s \
    --log <out>/saccade_track.log --out-dir <out> --option=--measurement-mutation \
    --tree-mount /opt/saccade --report rejected.json
.venv/bin/python scripts/eval/diagnostics/native_track_parity.py parity --entrypoint measurement \
    --out <out> --oracle-rows <oracle_rows> --oracle-txt <anchor> --mutation stale_gmc_input
```

### 18.8 Review 修正：拒絕檢查的路徑與 trace 輸出

Review 在 `60d66da7`／republish `2e0ae90c` 重現兩個 checker 漏檢：只有 `trace/` 輸出時 `no_output` 仍通過；model-root 本身與 directory-relative `openat` 也可能被漏掉。原始三次拒絕紀錄沒有這些存取或 trace 輸出；這是驗收工具的覆蓋缺口，沒有發現 shipping binary 違反拒絕契約。

修正後 `rejected` 的五個 gate 名稱不變，判定收緊：

- `no_model_root_open` 包含 model-root 本身。相對 `openat` 以 `strace -yy` 的 dirfd／`AT_FDCWD` cwd 註記解析；不推測未追蹤的 fd／cwd 狀態。相對路徑無法解析、或 open 紀錄不完整時，model／GPU／output 三項都 fail closed，報告列出 `unresolved`。
- `no_output` 包含 `native/`、`track_report.json`、`trace/`，並從 exec argv 讀取實際的 `--out`／`--report`／`--trace` 路徑。即使檔案沒有留下，對這些路徑的寫入 open 嘗試（含失敗）也必須被抓到。
- 回歸測試涵蓋存留 trace、directory-relative model open、cwd-relative model open、GPU 裝置、無法解析／不完整的 open、失敗的 trace 寫入、自訂 trace 路徑，以及能解析的無關唯讀 open。

後續驗證保留 §18.5 的原始紀錄，另建新結果目錄：以更新的 checker 重驗三次歷史拒絕 log，再在同一乾淨容器以 `strace -yy` 新跑三次被移除的選項。這次只驗證拒絕 gate 與 checker，沒有重跑完整 detector／MOT parity；shipping 原始碼與 entrypoint／operator library pin 不變，§18.5 的正式 run 仍是原來的證據。

**驗證結果**：乾淨 commit `2a3cce80`，`results/465_prc2_cli/review_fix_2a3cce80/`（`run.py`、`summary.json`、`historical_*.json`、`rejected_*/`）。歷史重驗 3/3 PASS；新容器拒絕 3/3 PASS，每次五項 gate 全通過。checker 回歸測試 47/47 PASS；shipping sources 與 `04f6f135` 相同，pin／operator hash 在新 run 前後均相同。`summary.json` sha256：`8c2b8d824774e09a5ae20169c5cd27e188257a372302c13e5a5968fe59a0c9e0`。這是 §18.8 的後續驗證，不取代 §18.5 的正式 parity。

### 18.9 Review 修正：alias 目標與相對輸出參數

Review 在 `2e74f3ee`（#530 合入後的 #529 head）用合成 trace 重現兩個 checker 漏檢，都是 `rejected` 的覆蓋缺口，沒有觀察到 shipping binary 違反拒絕契約：

- 相對的 `--out`／`--trace`／`--report` 保持相對路徑，但 open 紀錄已解析成絕對路徑，兩者無法比對。例：`--trace custom_trace` 配 `openat(AT_FDCWD</out>, "custom_trace/detector.bin", O_WRONLY|O_CREAT, 0666) = -1 EACCES`，五項 gate 全過。
- 只分類請求的路徑，忽略 `strace -yy` 對回傳 fd 的目標註記。經 symlink／裝置別名的成功 open（`/tmp/config-alias` → model-root 內的 config、`/dev/char/195:0` → `/dev/nvidia0`）能通過 `no_model_root_open`／`no_gpu_device_open`。

修正後五個 gate 名稱不變，判定收緊：

- model-root、GPU 裝置與輸出寫入三項同時檢查請求路徑與回傳 fd 的目標（含裝置的巢狀 `<char M:N>` 註記）。帶 dirfd 註記（即 `-yy` 格式）的成功 open 若缺少目標註記，視為不完整紀錄，三項都 fail closed。
- 相對輸出參數以所有 `AT_FDCWD` 註記出現過的 cwd 解析；一個 cwd 都沒有時（例如非 `-yy` 的 log），`no_output` fail closed，報告的 `unresolved_outputs` 列出該參數。存留檔案檢查也套用解析後落在 `/out/` 下的路徑。
- 非 `-yy` 的 log 沒有目標註記，無法排除經別名的存取；這是舊格式證據的 named limit，不是 PASS 的依據擴大。§18.5 的三次拒絕 run 只用預設的絕對輸出路徑。

回歸測試新增 10 項：三種相對輸出參數的失敗寫入、相對 `--report` 的存留檔、無 cwd 時 fail closed、四種別名目標（config、operator library、GPU 裝置、trace 寫入）、缺目標註記的 `-yy` 紀錄。這 10 項在 `2e74f3ee` 的 checker 上全部失敗，修正後全部通過；checker 回歸測試 57/57 PASS。

**驗證結果**：乾淨 commit `2c6fceae`，`results/465_prc2_cli/review_fix_2c6fceae/`（`run.py`、`summary.json`、`historical_*.json`、`yy_2a3cce80_*.json`、`m4_replay.json`、`runtime_replay.json`）。只重播既有 log，不新跑容器：shipping binary 未變，`2a3cce80` 的三次 `-yy` run 已帶目標與 cwd 註記。§18.5 歷史拒絕 log 3/3 PASS；`-yy` 拒絕 log 3/3 PASS；M4 負控制失敗的 gate 與原本逐項相同；正式 run 的 runtime 重播 PASS。shipping sources 與 `04f6f135` 相同，pin／operator hash 前後相同；runtime identity `--mode attested` exit 0，checker 不在 identity 輸入內，不需 republish。`summary.json` sha256：`38abaa0dc3d8dff979383870bd4b8ad60937a3b7dd85578894d804d1746d4f41`。不取代 §18.5 的正式 parity。

### 18.10 修正 A2：auditor 必須初始化成功（PR-C1 launcher）

Review 在 `b73d0a65`（#527 合入 C1＋C2 的 head）重現：auditor 不存在或無法載入時，loader 印出 `cannot be loaded as audit interface … ignored` 後照樣執行。以真的 loader 與原本的 launcher，刪除或截斷 auditor 後，被放在 DT_RPATH 位置的那一份會被載入，process exit 0，不是 127。§17 的 provenance 保護因此不是 fail-closed：只有 auditor 確實載入時才成立，§17.5 的 r2 與 §18.5 的 run 都是在 auditor 完整時量的。同一次 review 另有兩個 checker 漏檢（`runtime` 不看 alias 目標；`rejected` 的 model root 寫死），在 `0f26d08a` 修正。在乾淨的 `0f26d08a` 上，用修正後的 checker 重播 C1 與 C2 的 13 份既有報告（runtime、N5、N9 各兩份；6 份拒絕；M4），每份的逐項結果都與原本相同（`results/465_prc1_bundle/review_fix_0f26d08a/`，`summary.json` sha256 `8c767f2e…`）。A2 改了 exec chain 之後，舊 launcher 的 log 不會再通過 exec chain；那些是舊 launcher 的紀錄，不重播。

**改了什麼**（`71acc415`）：

- **auditor**：`la_version` 推出 `<prefix>` 之後，若 `SACCADE_AUDIT_PROBE=1`，就在 stdout 寫 `saccade-loader-audit-ready`，然後在程式執行前 exit 0。
- **launcher**：先 unset 呼叫端的 `SACCADE_AUDIT_PROBE`，再以同一個 loader、同樣的 `--library-path` 與 `--audit` 對 `/bin/sh -c :` 跑一次 probe（指令替換，子 process）。讀不到那一行就印 `saccade_track: the loader provenance auditor did not initialize (…)` 並 exit 127；讀到之後，`exec` 一行與之前相同。在同一個 exec 內沒辦法做到 fail-closed：`--audit` 與 `--preload` 載入失敗都只是被忽略，而 entrypoint 是 pin 的，不能加 NEEDED。
- **檢查**：G2-2 的 exec chain 改成恰好三次、都成功。launcher process 內依序是 launcher、loader（argv 同 §17.4）；另一個 process（`strace -ff` 的另一個 log）是 probe，argv 恰好是 `/lib64/ld-linux-x86-64.so.2 --library-path /opt/saccade/lib/vendor --audit /opt/saccade/lib/saccade_loader_audit.so /bin/sh -c :`。
- **測試**：真的 loader 上，auditor 不存在、截斷一半、空檔、換成沒有 `la_version` 的 ELF 四種情況，都 exit 127、什麼都沒執行（舊 launcher 在這四種情況會執行被放進去的那一份）；呼叫端設 `SACCADE_AUDIT_PROBE=1` 時照常執行。
- **沒有動的**：entrypoint pin（`92f74ef4…`）、operator library（`aa84cccd…`）、engine、model root、`third_party_set.json`（27 個）、安裝規則、SM 清單、glibc baseline、容器映像與 `run_shipping_container.sh`。auditor 是獨立的 library target（`saccade_loader_audit`），不進 entrypoint。

**開發期間已經看到的**（`results/465_prc1_dev/a2_71acc415/`，不是正式 run）：

- `static` 12 項通過。
- `bundle-strace`（MOT17-05）exit 0。三次 exec，probe 在獨立的 log；`runtime` 三項通過。
- 刪掉 auditor 的 tree：exit 127，訊息如上。只有 launcher 與 probe 兩次 exec，沒有輸出。
- host 經 launcher 跑 MOT17-05（`LD_DEBUG=files`，probe 與主程式各一份 log）：`sources` 通過。
- 對 PR-C1 r2 的 tree 跑 `static`：`entrypoint_pinned`、`launcher_exact`、`entrypoint_no_measurement_surface` 失敗。

**契約修正**（正式 run 之前寫定）：正式 run＝§18.4 的整個 run（§17.4 第 1–5 條、§18.4 第 6 條、M1–M6、N1–N9），在 A2 的乾淨 commit 上重跑，以下不同：

- **有效性**：編出 pin 的原始碼不變的檢查改成 `git diff 04f6f135 HEAD -- shipping/src shipping/include shipping/tools src include ':!shipping/src/loader_audit.c'` 為空；`loader_audit.c` 與 `04f6f135` 的差異只有 A2 的 probe（照實記錄）。
- **第 5 條與第 6a 條的 exec chain**：照上面的三次 exec。
- **M1**：`entrypoint_no_measurement_surface`、`entrypoint_pinned`、`launcher_exact` 失敗（launcher 已換）。
- **N5** 改用 PR-C1 r2 tree 的 launcher（沒有 probe）去掉 `--audit` 那一行，放進 N4 的 tree。必須的結果同 §17.4：`static` 的 `layout_exact`、`launcher_exact` 失敗；`bundle-strace`（MOT17-05）exit 0，那一份被載入；`runtime` 的 exec chain 與 opened set 失敗。直接對新 launcher 刪 `--audit` 會連 probe 一起失效，變成 exit 127，測不到 N5 要測的東西。
- **新的負控制**：N10–N12 都用 N4 的 tree（`nvidia/cu13/lib/` 有被放進去的那一份），只改 auditor，跑 `bundle-strace`（MOT17-05）：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N10 | 刪掉 `lib/saccade_loader_audit.so` | exit 127，log 有 `the loader provenance auditor did not initialize`；沒有 `native/`、`track_report.json`；strace 恰好兩次 exec（launcher、probe），沒有執行 entrypoint 的 loader |
| N11 | auditor 換成它的前一半位元組 | 同 N10 |
| N12 | auditor 換成 `lib/vendor/libcudart.so.13` 的副本（ELF，沒有 `la_version`） | 同 N10 |
| N13 | 正常 tree，容器加 `-e SACCADE_AUDIT_PROBE=1`，跑 `bundle`（MOT17-05） | exit 0，MOT17-05 txt 與第 4 條相同 |

其他 gate、負控制、判準與「不做的」不變。結果放在 `results/465_prc2_cli/full_<A2 commit>/`；§17.5 與 §18.5 的結果保留，作為舊 launcher 的紀錄。

### 18.11 A2 正式 run 驗收

同一台機器（RTX 5070 Ti Laptop，WSL2）。commit `036d6b95`：含 A2 的實作 `71acc415` 與契約 §18.10，工作樹乾淨，契約早於任何量測。全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library（`aa84cccd…`）與 pin 的 entrypoint（`92f74ef4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json` 的值。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 工作樹乾淨。`git diff 04f6f135 HEAD -- shipping/src shipping/include shipping/tools src include ':!shipping/src/loader_audit.c'` 為空。`loader_audit.c` 的差異只有 A2 的 probe。CMake 沒有非註解的變更。`anchor` 與 `A_L_1` 7/7 相同，`oracle-rows` OK |
| 1 build 與安裝 | configure、build（`saccade_track`、`saccade_track_measurement`）、install 都 exit 0 |
| 2 靜態檢查 | 12 項 PASS（含 `launcher_exact`：tree 的 launcher 是 A2 的版本） |
| 3 host，經 launcher | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7。`--against` PR-12 `parity_pristine` 相同。`sources` PASS（probe 與主程式各一份 loader log） |
| 4 乾淨容器（bundle） | exit 0，`EXACT`，與第 3 條 7/7 相同 |
| 5 G2-2／G2-4 | 三項 PASS。exec chain 恰好三次、都成功：launcher process 內依序是 launcher → loader（`--library-path /opt/saccade/lib/vendor --audit /opt/saccade/lib/saccade_loader_audit.so … /opt/saccade/libexec/saccade_track`）；另一個 process 是 probe（`… /bin/sh -c :`）。`parity --native-from` `EXACT`，與第 4 條 7/7 相同。三次 run 的 `python_libraries_mapped` 都是空的 |
| 6a 被移除的選項 | 三次 `rejected` 各 5 項 PASS（exit 2、`unknown argument`；三次 exec 的 chain；沒有 model root／operator library／GPU 裝置的 open；沒有輸出） |
| 6b developer build 基準 | `EXACT`，與第 3 條 7/7 相同 |
| 6c serial 參考 | `EXACT`（`detector` 5316/5316、`mot_txt` 7/7） |
| 6d 負控制保留 | 7 個都 `CAUGHT`：`stale_detector_input`、`swapped_detection_parity`、`stale_image_dims` 的 `detector` `DIFFERS`；`stale_gmc_input`、`shared_post_host`、`gmc_previous_frame`、`ref_edit` 的 `mot_txt` `DIFFERS`；沒有 validity 問題 |
| **verdict** | **`PASS`**（§18.4 第 1–6 條，依 §18.10 修正，全部成立） |

**負控制**（`negctl/`，全部抓到）：

| # | 結果 |
|:--|:--|
| M1 | PR-C1 r2 的 tree：`entrypoint_pinned`、`launcher_exact`、`entrypoint_no_measurement_surface` 失敗，其餘通過 |
| M2 | `entrypoint_pinned`、`entrypoint_no_measurement_surface` 失敗；另外 `produced_elves`、`search_path_containment` 失敗（同 §18.5：build tree 的 RUNPATH） |
| M3 | exit 1 |
| M4 | `exit_2_unknown_argument`、`no_model_root_open`、`no_gpu_device_open`、`no_output` 失敗；exec chain 通過（三次 exec） |
| M5 | harness 在執行前 exit 2 |
| M6 | `UNRESOLVED`（report 的 `entrypoint` 是 `saccade_track_measurement`） |
| N1 | 安裝失敗（`build-release` 的 `saccade_track` sha256 ≠ pin） |
| N2 | `static`：`vendor_set_pinned` 失敗 |
| N3 | `static`：`layout_exact`、`vendor_set_pinned`、`g2_1_needed_closure` 失敗；容器 exit 2（`libnvrtc.so.13: cannot open`） |
| N4 | `static`：`layout_exact` 失敗；容器 exit 127，`foreign copy on the search path: …/nvidia/cu13/lib/libcublas.so.13` |
| N5（A2） | PR-C1 r2 的 launcher 去掉 `--audit`：`static` 的 `layout_exact`、`launcher_exact` 失敗。容器 exit 0（被放進去的那一份被載入）。`runtime`：exec chain 失敗、opened set 失敗（`/opt/saccade/nvidia/cu13/lib/libcublas.so.13`） |
| N6a | exit 0；MOT17-05 txt 與第 4 條相同；auditor 載入之後 49 個物件，沒有來自 `/opt/foreign` 的 |
| N6b | 同 N6a。觀察：launcher 的 sh 載入了 preload 的 `/opt/foreign/libcudart.so.13`（同 §17.6 的限制） |
| N7 | exit 127：`libnvinfer.so.10: cannot open shared object file` |
| N8 | `static`：`layout_exact`、`g2_3_no_python_files` 失敗 |
| N9 | `runtime`：exec chain 失敗（`/bin/sh`、launcher、probe、loader 四次 exec），其餘兩項通過 |
| N10（A2） | auditor 刪除：exit 127，`the loader provenance auditor did not initialize`；只有 launcher 與 probe 兩次 exec，entrypoint 沒有執行；沒有輸出 |
| N11（A2） | auditor 截成一半：同 N10 |
| N12（A2） | auditor 換成 `libcudart.so.13`：同 N10 |
| N13（A2） | 呼叫端 `SACCADE_AUDIT_PROBE=1`：exit 0，MOT17-05 txt 與第 4 條相同 |

N10–N12 用的都是 N4 的 tree，被放進去的 `libcublas.so.13` 還在。用舊的 launcher（§17.5 N5）時，那一份會被載入，exit 0。

**限制**（在 §17.6 之外）：

- probe 與正式 exec 是兩次 loader 啟動，中間 auditor 檔案仍可能被換掉。能在這之間改寫 tree 的人，也能直接改寫 launcher，所以這不比 §17.6 的威脅模型更弱；這裡不宣稱防得住 race。
- probe 只證明 auditor 能載入、`la_version` 推得出 `<prefix>`。auditor 的位元組沒有在執行時驗證，同 §17.6「auditor 只管路徑，不管位元組」。安裝後的完整性驗證留給 PR-C3 的 MANIFEST。

結果目錄：`results/465_prc2_cli/full_036d6b95/`（`run.sh`、`tree/`、`static.json`、`ld_host/`、`sources.json`、`anchor/`、`oracle_rows/`、`parity_*`、`container_*`、`runtime.json`、`cli/`、`negctl/`、`pins_before.txt`／`pins_after.txt`、`validity_sources.txt`）；開發試做在 `results/465_prc1_dev/a2_71acc415/`。不納入版本控制。§17.5 與 §18.5 的結果保留原處，是舊 launcher 的紀錄。

### 18.12 修正 A3：auditor 以真實路徑分類、trace 帶 fd 目標

Review 在 `93f4de43`（#531 合入後的 #527 head）提出三點。都不是在正式 shipping run 裡觀察到的 parity 違反，是保護與證據的覆蓋缺口：

- **auditor 可被 symlink 別名繞過**：`la_objopen` 先用請求路徑的 basename 分類，名字不屬於 bundle 集合也不是 operator library 就直接放行，不呼叫 `realpath`。真實 loader 的 CPU 控制：直接載入外部的 `libfoo.so` 會 exit 127；經 `payload → libfoo.so` 載入則 exit 0，執行了外部的函式。
- **`runtime` 對不完整的 trace 仍回報 PASS**：parser 已經把 `openat(7, "payload", O_RDONLY) = 9` 這類紀錄標成 unresolved，但 `runtime` 的三項檢查都沒有看 unresolved。
- **trace 收集沒有 `-yy`**：A2 正式 run 的 6,533 筆 open 都沒有 fd 目標，所以 alias 檢查在那份證據上沒有作用。

**改了什麼**（`f1d98b93`、`06002566`）：

- **auditor**：`la_objopen` 先 `realpath`，再用請求名與真實名兩個 basename 分類（bundle 名字、operator library、Python）。解析不了的物件（vDSO）只有在兩項檢查都不適用於它的請求名時才放行。
- **`runtime`**：unresolved 紀錄讓 Python open 與 opened set 兩項失敗。
- **`runtime`／`rejected` 預設要求 fd 目標**：成功的 open 沒有回傳 fd 的目標註記，就算 unresolved。`--legacy-plain-trace` 用來評估沒有 `-yy` 的歷史 log，報告會記下 alias 目標沒有被檢查。fd 目標若不是路徑（例如 `pipe:[N]`），也算有名字。
- **fd 目標按 SONAME 家族分類**：`-yy` 把 base system 的 SONAME symlink 標成帶完整版本號的實體檔（`libstdc++.so.6` → `libstdc++.so.6.0.33`、`libz.so.1` → `libz.so.1.3`），所以分類時逐段去掉尾端的版本號，直到某個名字能分類。這跟 SONAME 本身一樣是按名字判斷，不是按位元組。名字像 base system、實際指向第三方物件的 alias 仍然算外來的。
- **`run_shipping_container.sh bundle-strace`** 改成 `strace -ff -qq -yy`。PR-12 的 `strace` 模式不變。
- **測試**：
  - 真實 loader：經 `payload` alias 的外部 `libfoo.so`，exit 127，沒有輸出（舊 auditor 會印出外部的結果）。
  - `runtime` 遇到 unresolved（dirfd 沒有註記、cwd 相對路徑、不完整、`-yy` 紀錄缺目標）時失敗。
  - plain log 必須加 `--legacy-plain-trace` 才能評估。
  - SONAME 家族分類。
- **沒有動的**：launcher、entrypoint pin、operator library、engine、model root、`third_party_set.json`、安裝規則。

**開發期間已經看到的**（`results/465_prc1_dev/a3_f1d98b93/`，不是正式 run）：

- `static` 12 項通過。
- `bundle-strace`（MOT17-05，`-yy`）exit 0。
- 第一次 `runtime` 的 opened set 失敗：兩筆 base system 的版本號檔名被當成外來的（上面 SONAME 家族那一點的由來）。修正後三項 PASS，0 筆 unresolved。

**契約修正**（正式 run 之前寫定）：正式 run＝§18.10 修正後的整個 run，在 A3 的乾淨 commit 上重跑，以下不同：

- 第 5 條、N5、N9、N10–N12 的 `bundle-strace`，以及第 6a 條與 N9 直接呼叫的 strace，都帶 `-yy`。
- `runtime` 與 `rejected` 不帶 `--legacy-plain-trace`。第 5 條要求另外成立：Python open 與 opened set 兩項的 `unresolved` 為空。
- **新的負控制**：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N14 | 對 A2 正式 run（`full_036d6b95`）的 plain `container_strace` 跑 `runtime` | 不帶 `--legacy-plain-trace`：Python open、opened set 兩項失敗，`unresolved` 列出缺目標的 open；帶 `--legacy-plain-trace`：三項 PASS，報告 `legacy_plain_trace: true` |

symlink alias 的控制只在 CPU 上的真實 loader 測試裡做（上面的測試）。在 shipping tree 上要做出「非 bundle 名字的 NEEDED 經 alias 指到外來 bundle 物件」，必須改 pin 的 ELF，所以不做。

其他 gate、負控制、判準與「不做的」不變。結果放在 `results/465_prc2_cli/full_<A3 commit>/`；§18.11 的 A2 結果保留，作為沒有 `-yy` 的證據紀錄。

### 18.13 A3 正式 run 驗收

同一台機器。commit `276659a8`：含 A3 的實作 `f1d98b93`、`06002566` 與契約 §18.12，工作樹乾淨，契約早於任何量測。全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library（`aa84cccd…`）與 pin 的 entrypoint（`92f74ef4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json` 的值。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 同 §18.11：工作樹乾淨。原始碼檢查（排除 `loader_audit.c`）為空。CMake 沒有非註解的變更。`anchor` 與 `A_L_1` 7/7 相同，`oracle-rows` OK |
| 1 build 與安裝 | configure、build、install 都 exit 0 |
| 2 靜態檢查 | 12 項 PASS |
| 3 host，經 launcher | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7。`--against` PR-12 `parity_pristine` 相同。`sources` PASS |
| 4 乾淨容器（bundle） | exit 0，`EXACT`，與第 3 條 7/7 相同 |
| 5 G2-2／G2-4 | `-yy` trace：6,533 筆 open，6,028 筆成功的全部帶 fd 目標，0 筆 unresolved。`runtime`（不帶 `--legacy-plain-trace`）三項 PASS：三次 exec 的 chain、沒有 Python 路徑、opened set 等於 bundle。`parity --native-from` `EXACT`，與第 4 條 7/7 相同。三次 run 的 `python_libraries_mapped` 都是空的 |
| 6a 被移除的選項 | 三次 `rejected` 各 5 項 PASS（`-yy`；`rejected_schedule` 50 筆成功的 open 全部帶目標） |
| 6b–6d | 同 §18.11：developer build 基準 `EXACT` 且與第 3 條相同；serial 參考 `EXACT`；7 個 mutation 負控制都 `CAUGHT`，指定的 section `DIFFERS`，沒有 validity 問題 |
| **verdict** | **`PASS`**（§18.4 第 1–6 條，依 §18.10 與 §18.12 修正，全部成立） |

**負控制**（`negctl/`，全部抓到）：M1–M6、N1–N13 的結果與 §18.11 相同。

- M1：`entrypoint_pinned`、`launcher_exact`、`entrypoint_no_measurement_surface` 失敗。
- M4：exit／model root／GPU／輸出四項失敗，exec chain 通過。
- N4：exit 127，`foreign copy on the search path`。
- N5：容器 exit 0，`runtime` 的 exec chain 與 opened set 失敗（`/opt/saccade/nvidia/cu13/lib/libcublas.so.13`）。
- N7：exit 127（`libnvinfer.so.10`）。
- N9：四次 exec，exec chain 失敗。
- N10–N12：exit 127，entrypoint 沒有執行，沒有輸出。
- N13：exit 0，MOT17-05 相同。
- 新的 N14：A2 的 plain `container_strace` 不帶 `--legacy-plain-trace` 時，Python open 與 opened set 兩項失敗，`unresolved` 各 6,028 筆；帶這個旗標時三項 PASS，報告 `legacy_plain_trace: true`。

**限制**（在 §17.6、§18.11 之外）：

- auditor 與 checker 都按名字分類：真實路徑的 basename、SONAME 家族。一份改了名字的外來實體檔（不是 symlink），若以非 bundle 名字被 NEEDED 或 `dlopen`，兩者都不會辨認出來；位元組的完整性仍靠安裝時的 sha256、`static` 與 PR-C3 的 MANIFEST。
- symlink alias 的控制只在 CPU 上的真實 loader 測試裡做（§18.12）。

結果目錄：`results/465_prc2_cli/full_276659a8/`；開發試做在 `results/465_prc1_dev/a3_f1d98b93/`。不納入版本控制。§18.11 的 A2 結果保留，作為沒有 `-yy` 的紀錄。

### 18.14 修正 A4：版本號 alias、完整的 loader argv、不完整的 exec、預設 model root

Review 在 `ff7359d4`（#532 合入後的 #527 head）提出五點，四個 P2、一個 P3。同樣都不是在正式 shipping run 裡觀察到的 parity 違反，是保護與 checker 的覆蓋缺口：

- **auditor 可被指向帶版本號檔名的 alias 繞過**：`payload → libcudart.so.13.1.0`，請求名與真實名都不等於 bundle 的 SONAME `libcudart.so.13`，所以被當成非 bundle 物件放行。真實 loader 重現：exit 0，執行了外來的函式。
- **exec chain 沒有檢查 loader 實際執行的程式**：只比對 loader argv 的前四個參數，以及 entrypoint 有沒有出現在 argv 裡。把 loader 的參數換成 `--argv0 /opt/saccade/libexec/saccade_track /bin/true`，`runtime` 的每一項仍然 PASS，但程式位置上是 `/bin/true`。
- **不完整的 exec 紀錄被丟掉**：parser 的 fallback 只認得 open。在一份 PASS 的 trace 後面接一筆 unfinished 的 Python `execve`，結果仍然 PASS，checker 照樣認證「恰好三次 exec」。
- **`rejected` 沒有保護預設的 model root**：沒給 `--model-root` 時，`track::Options` 預設 `.`。這項檢查只保護 argv 明確給的輸入與安裝的 model root；cwd 為 `/work`、open `/work/models/yolo/yolo26s_backbone_640_best.engine` 的 trace，五項都 PASS。
- **launcher 只拒絕 `:`**（P3）：glibc 切 `--library-path` 時 `;` 也算分隔符，跟 shell 的引號無關。tree 放在含 `;` 的目錄下時，會通過 launcher 的檢查與 readiness probe，然後 loader 找不到 bundle 的函式庫，exit 127。

**改了什麼**（`e6b57ac8`）：

- **auditor**：`la_objopen` 用 SONAME 家族比對請求名與真實名。家族指去掉尾端數字版本段之後、以 `.so` 結尾的名字，跟 checker 的 `_object_class` 同一個規則，所以 `libcudart.so.13.1.0` 屬於 bundle 家族 `libcudart.so`，必須來自 `lib/vendor`。operator library 也按家族比對。`la_objsearch` 不變，因為搜尋候選的 basename 就是請求的 NEEDED 名。
- **launcher**：前綴含 `:` 或 `;` 時 exit 2。
- **exec chain**：loader 的 argv 必須**完整等於** `[ld.so, --library-path, <mount>/lib/vendor, --audit, <mount>/lib/saccade_loader_audit.so, --argv0, <launcher 的 argv[0]>, <mount>/libexec/saccade_track, <launcher 的 argv[1:]>]`。
- **不完整的 exec**：`execve`／`execveat` 開頭、但 strace 沒有記錄完整（unfinished、截斷）的行，會留下成一筆失敗的 exec（`incomplete: true`），所以 exec chain 失敗。
- **預設 model root**：照 `track_driver.hpp` 的 `parse_interface_arg` 模擬 entrypoint 的 argv parse。只要有一次 entrypoint exec 的 parse 在停下來（遇到未知選項或缺值）之前沒有取到 `--model-root`，就把每一個有證據的 cwd 都當成 model root 保護。沒有 cwd 證據時算 unresolved，檢查失敗。另外修正 `_within` 在 root 為 `/` 時不涵蓋任何路徑的問題。
- **測試**：
  - 真實 loader：經 `payload` 指到 `libfoo.so.1.0.0`、`libfoo.so.13`（bundle 家族 `libfoo.so`）都 exit 127；指到其他家族（`libfoobar.so.1.0`）照常執行。
  - exec chain：`/bin/true` 放在程式位置、entrypoint 排在程式之後、轉送的參數多了或少了、`--argv0` 不同、多一個 loader 選項，都會失敗。
  - 三種不完整的 exec 都會失敗。
  - 預設 model root：cwd `/work`、cwd `/`、parse 沒走到 `--model-root`、沒有 cwd 證據。
  - launcher 前綴含 `:` 或 `;`。
  - 新加的測試在 `ff7359d4` 的原始碼上全部失敗，修正後全部通過，共 104 項。
  - fixture 的 launcher／loader argv 改成跟正式 run 一樣帶 `--model-root`。
- **沒有動的**：entrypoint pin、operator library、engine、model root、`third_party_set.json`、安裝規則、`run_shipping_container.sh`。

**開發期間已經看到的**（`results/465_prc1_dev/a4_e6b57ac8/`，不是正式 run）：

- `static` 12 項通過。
- `bundle-strace`（MOT17-05）exit 0，`runtime` 三項 PASS，MOT17-05 txt 與 A3 的第 4 條相同。
- `rejected`（`--measurement-mutation`）5 項 PASS。
- 新 checker 重新評估 A3 正式 run 的 `runtime` 與三次 `rejected`：全部 PASS。預設 model root 沒有被加入，因為 argv 有給 `--model-root`。
- 下面 N15–N17 的合成控制，在開發 trace 上：新 checker 都在對應的那一項失敗；A3 的 checker（`ff7359d4`）全部 PASS。
- `;` 前綴：A4 launcher exit 2；A3 launcher 通過 probe 之後，loader 找不到 `libnvinfer.so.10`。

**契約修正**（正式 run 之前寫定）：正式 run＝§18.12 修正後的整個 run，在 A4 的乾淨 commit 上重跑，以下不同：

- `runtime`、`rejected` 用 A4 的 checker。第 5 條與第 6a 條的 exec chain 以完整的 loader argv 判定。
- N9 仍然必須失敗（四次 exec）。N14 帶 `--legacy-plain-trace` 時三項仍然必須 PASS（A2 plain trace 的 loader argv 也要通過完整比對）。
- **新的負控制**：N15–N17 是合成的，在這次 run 的真實 `-yy` trace 複本上做一處修改（修改腳本寫在 `run.sh` 裡）。每一個也用 A3 的 checker（`ff7359d4` 的版本）評估一次，作為重現紀錄，不列入判準。

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N15 | 第 5 條 `container_strace` 的複本：loader exec 改成 `--argv0 /opt/saccade/libexec/saccade_track /bin/true` | `runtime`：exec chain 失敗，其他兩項 PASS |
| N16 | 第 5 條 `container_strace` 的複本：launcher 的 log 後面加一筆 `execve("/usr/bin/python3", …) <unfinished ...>` | `runtime`：exec chain 失敗，報告的 `incomplete` 紀錄就是那一筆；其他兩項 PASS |
| N17a | 第 6a 條 `rejected_measurement_mutation` 的複本：launcher 與 loader 的 argv 去掉 `--model-root <值>` | `rejected`：`no_model_root_open` 失敗，`model_inputs` 含 cwd `/`；其他四項 PASS |
| N17b | N17a，另外把每一個 `AT_FDCWD</>` 改成 `AT_FDCWD</work>`，再接一筆 open `/work/models/yolo/yolo26s_backbone_640_best.engine` | `rejected`：`no_model_root_open` 失敗，`attempted` 正好是那個 engine 路徑；其他四項 PASS |
| N18 | 乾淨 image（不給 GPU），tree 掛在 `/opt/sac;cade`，執行 `/opt/sac;cade/bin/saccade_track` | A4 tree：exit 2，訊息是 `must not contain ':' or ';'`。A3 tree（`full_276659a8`）exit 127，只作重現紀錄 |

版本號 alias 的控制只在 CPU 上的真實 loader 測試裡做（上面的測試），理由與 §18.12 相同：要在 shipping tree 上做，必須改 pin 的 ELF。

其他 gate、負控制、判準與「不做的」不變。結果放在 `results/465_prc2_cli/full_<A4 commit>/`；§18.13 的 A3 結果保留。

### 18.15 A4 正式 run 驗收

同一台機器。commit `1ae402c2`：含 A4 的實作 `e6b57ac8` 與契約 §18.14，工作樹乾淨，契約早於任何量測。全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。operator library（`aa84cccd…`）與 pin 的 entrypoint（`92f74ef4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json` 的值。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 同 §18.13：工作樹乾淨。原始碼檢查（排除 `loader_audit.c`）為空。CMake 沒有非註解的變更。`anchor` 與 `A_L_1` 7/7 相同，`oracle-rows` OK |
| 1 build 與安裝 | configure、build、install 都 exit 0 |
| 2 靜態檢查 | 12 項 PASS（`launcher_exact`：tree 的 launcher 等於 A4 的原始碼） |
| 3 host，經 launcher | `EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7。`--against` PR-12 `parity_pristine` 相同。`sources` PASS |
| 4 乾淨容器（bundle） | exit 0，`EXACT`，與第 3 條 7/7 相同 |
| 5 G2-2／G2-4 | `-yy` trace：6,485 筆 open，5,992 筆成功的全部帶 fd 目標，0 筆 unresolved。`runtime` 三項 PASS，其中 exec chain 以完整的 loader argv 判定。`parity --native-from` `EXACT`，與第 4 條 7/7 相同。三次 run 的 `python_libraries_mapped` 都是空的 |
| 6a 被移除的選項 | 三次 `rejected` 各 5 項 PASS（`-yy`；每次 50 筆成功的 open 全部帶目標；argv 有給 `--model-root`，所以沒有加入預設 model root） |
| 6b–6d | 同 §18.13：developer build 基準 `EXACT` 且與第 3 條相同；serial 參考 `EXACT`；7 個 mutation 負控制都 `CAUGHT`，指定的 section 都不同，沒有 validity 問題 |
| **verdict** | **`PASS`**（§18.4 第 1–6 條，依 §18.10、§18.12 與 §18.14 修正，全部成立） |

**負控制**（`negctl/`，全部抓到）：M1–M6、N1–N14 的結果與 §18.13 相同。N9 四次 exec，exec chain 失敗。N14 帶 `--legacy-plain-trace` 時三項 PASS，表示 A2 plain trace 的 loader argv 也通過完整比對。新的控制：

| # | A4 checker／launcher（判準） | A3（`ff7359d4`，只作重現紀錄） |
|:--|:--|:--|
| N15 程式位置換成 `/bin/true` | exec chain 失敗，其他兩項 PASS | 三項 PASS |
| N16 接一筆 unfinished 的 Python `execve` | exec chain 失敗，`incomplete` 紀錄正好是那一筆；其他兩項 PASS | 三項 PASS |
| N17a 去掉 `--model-root` | `no_model_root_open` 失敗，`model_inputs` 含 cwd `/`（96 筆 attempted）；其他四項 PASS | 五項 PASS |
| N17b N17a＋cwd `/work`＋engine open | `no_model_root_open` 失敗，`attempted` 只有 `/work/models/yolo/yolo26s_backbone_640_best.engine`；其他四項 PASS | 五項 PASS |
| N18 tree 掛在 `/opt/sac;cade` | exit 2，`the install prefix must not contain ':' or ';'` | exit 127，loader 找不到 `libnvinfer.so.10` |

**限制**（在 §17.6、§18.11、§18.13 之外）：

- SONAME 家族仍然是名字：一份改了名字、名字不屬於任何 bundle 家族的外來實體檔，auditor 與 checker 都不會辨認出來（§18.13 的限制不變）。
- 版本號 alias 的控制只在 CPU 上的真實 loader 測試裡做（§18.14）。
- N15–N17 是在真實 trace 上做一處修改的合成控制，證明的是 checker 的判定，不是 runtime 的行為。
- 預設 model root 的判定靠模擬 entrypoint 的 argv parse；parse 的規則改了，checker 要跟著改（`_ENTRYPOINT_VALUE_OPTIONS` 指向 `track_driver.hpp`）。

結果目錄：`results/465_prc2_cli/full_1ae402c2/`；開發試做在 `results/465_prc1_dev/a4_e6b57ac8/`。不納入版本控制。§18.13 的 A3 結果保留。

## 19. Package：tarball、MANIFEST、package digest、atomic 安裝器（Phase C PR-C3）

PR-C3 是 Phase C 的第三個 PR（[Phase C scope](native_runtime_phase_c_scope.md) §6）：把 PR-C1／C2 的 shipping tree 包成一個可以從乾淨系統安裝的 package，安裝失敗時目標目錄不動（PR-12 §16.5 N2 的「安裝不是 atomic」）。PR-C3 不改任何 stage 的計算、entrypoint（pin `92f74ef4…`）、operator library（`aa84cccd…`）、27 個第三方物件、launcher、auditor、安裝規則、SM 清單與 glibc baseline；package 裡的 tree 就是 `cmake --install --component shipping` 寫出的那一份，加一個 `MANIFEST.json`。

| 項目 | 位置 |
|:--|:--|
| 安裝器 | `shipping/package/install.sh` → 隨 package 發佈為 `<name>.install.sh` |
| 產生 package | `scripts/native/build_shipping_package.py` |
| MANIFEST 格式與 package 檢查 | `scripts/native/check_shipping_package.py`（`tarball`、`install-trace`）；`check_shipping_bundle.py static --manifest`（安裝後的 tree） |
| 容器內安裝 | `scripts/native/run_package_container.sh install`／`install-strace` |
| 測試 | `tests/unit/test_shipping_package.py` |

實作之前沒有額外的 owner 指示；範圍是 Phase C scope §6 的 PR-C3 那一列。§19.1 的設計是本 PR 的決定，owner review 時可以改。

### 19.1 設計決定

- **發佈的是三個檔案**：`<name>.tar.gz`、`<name>.install.sh`、`<name>.sha256`。`<name>`＝`saccade-<version>-linux-x86_64-cu<x.y>-trt<x.y>-sm120-glibc<x.y>`，目前是 `saccade-0.1.0-linux-x86_64-cu13.0-trt10.16-sm120-glibc2.39`：version 讀 `pyproject.toml`，CUDA 與 TensorRT 讀 `third_party_set.json` 的 `nvidia_cuda_runtime`、`tensorrt_cu12_libs` wheel，SM 是 C-D2，glibc 是 §16 的 baseline。
- **安裝器在 tarball 外面**：要先驗證 tarball 才解開它，所以安裝器不能在 tarball 裡。`<name>.sha256`（package digest）是兩行 `sha256sum` 格式，涵蓋 tarball 與安裝器；PR-C4 的 minisign 簽的就是這個檔案。安裝器只能驗 tarball，驗不了自己：安裝器本身的完整性由使用者 `sha256sum -c`（PR-C4 之後是簽章）負責。
- **digest 不是簽章**：能換掉 tarball 的人也能換掉 `<name>.sha256`。安裝器擋得住損壞與不完整的下載，擋不住一份重新包過、digest 也重算過的 package（§19.4 P3 照實記錄這一點）；那是 PR-C4 的範圍。
- **MANIFEST.json**：tarball 裡唯一的頂層目錄 `<name>/` 下，安裝後在 `<prefix>/MANIFEST.json`。除了 `files` 之外的每個欄位都由 source commit 的 repository 推出（`check_shipping_package.manifest_head`），不是複製：版本 pin（每個 wheel、CUDA runtime、TensorRT、torch、cuDNN）、`gpu`（支援 `sm_120`；entrypoint 的 SASS／PTX 清單）、`platform`（loader、glibc baseline）、`source`（commit、commit 時間、工作樹是否乾淨、runtime identity 是否 current）、`pins`（`third_party_set.json` 的 sha256、entrypoint pin、launcher、auditor、operator library 與它的 attestation、安裝器）、`model_root`（resolved config、attestation、lineage、TorchScript head、backbone engine 各自的 sha256；resolved config 與 attestation 必須等於 source commit 的版本）、`runtime_identity`（`docs/reference/runtime_identity.generated.json` 的 sha256 與五軸座標）。推導時若 tree 的檔案不是 repository 的 pin，就拒絕產生。`files` 是 tree 裡每個檔案的 path、sha256、大小、mode（0755 或 0644），一行一個物件、照 path 排序：安裝器沒有 JSON parser（base system 沒有），用 `sed` 讀這個固定格式，任何不符合格式的行或 `file_count` 對不上都會失敗；`check_shipping_package.py` 以「重新序列化後逐位元組相同」確認 MANIFEST 是 canonical 的。
- **runtime identity 必須 current**：MANIFEST 記錄的座標要描述 package 的原始碼。`shipping/package/install.sh` 是 identity 的輸入（`shipping/**`），所以 PR-C3 的 republish 要在正式 run **之前**（C1／C2 是在之後）；builder 以 `check_runtime_identity_staleness.py --mode attested` 判定，不 current 或工作樹不乾淨就拒絕，`--trial` 照樣產生但記錄下來，`tarball` 檢查會讓這種 package 失敗。
- **tarball 是決定性的**：USTAR、member 依 path 排序、owner 0 且沒有名字、每個 mtime 都是 source commit 的時間、目錄 0755、檔案 0755（有任何 execute bit）或 0644、gzip header 沒有檔名與時間（level 6）。同一台機器、同一個 tree、同一個 commit 產生相同的位元組（Python 的 zlib；換一台機器不保證）。
- **安裝器的步驟**（POSIX sh；Ubuntu 24.04 base system 的工具：dash、coreutils、tar、gzip、sed、grep、findutils）：
  1. 檢查參數：TARGET 不存在（任何種類：目錄、空目錄、檔案、symlink、dangling symlink），它的上層目錄存在，路徑不含 `:`／`;`（launcher 會拒絕這種 prefix，§18.14）。
  2. `<name>.sha256` 恰好一行指名 `<name>.tar.gz`，sha256 相符。
  3. 在 TARGET 的上層目錄 `mktemp -d .saccade-install.XXXXXX`（同一個檔案系統，mode 0700），以 `--no-same-owner --no-same-permissions --keep-old-files` 解開。
  4. 解開的結果恰好是一個目錄 `<name>/`；其中恰好是 MANIFEST 列的檔案加 `MANIFEST.json`，沒有其他任何項目（symlink、裝置、空目錄）；每個檔案是 regular file、大小與 sha256 相符；依 MANIFEST 設定 mode（目錄 0755）後再讀回確認。
  5. `mv -n -T <staging>/x/<name> TARGET`：GNU coreutils 9.4 的這個呼叫是一次 `renameat2(…, RENAME_NOREPLACE)`（開發期間以 strace 確認），TARGET 在這之間出現就不取代它。之後以「來源已經不在」確認 rename 確實發生（某些 mv 版本略過時 exit 0）。
  6. 任何失敗或 HUP／INT／TERM：刪除 staging 目錄，TARGET 不會被建立。只有 SIGKILL（或當機）會留下 staging；下一次安裝會指出它，但不刪除（可能屬於另一個正在執行的安裝）。
  
  `install.sh --verify PREFIX` 以同一套規則（第 4 步，只檢查不設定 mode）驗證已安裝的 tree。安裝器沒有任何會改變檢查內容的選項或環境變數（測試檢查它只讀 `TMPDIR` 與自己設定的 `LC_ALL`），與 PR-C2 的原則相同：沒有 hidden hook。
- **`cmake --install` 不變**：開發用的安裝路徑仍然不是 atomic；atomic 的是 package 的安裝路徑。

### 19.2 改了什麼

- 新檔案：`shipping/package/install.sh`、`scripts/native/build_shipping_package.py`、`scripts/native/check_shipping_package.py`、`scripts/native/run_package_container.sh`、`tests/unit/test_shipping_package.py`。
- `check_shipping_bundle.py static --manifest`：layout 多一個 `MANIFEST.json`，多一項 `manifest_tree`（MANIFEST canonical、恰好列出 tree 的其他檔案及其 sha256／大小／mode、目錄 0755）。不帶 `--manifest` 時與之前相同（12 項）。它的 `# status:` 註解移到 docstring 前面：原本在第 68 行，超出 scripts index 的 60 行掃描範圍，生成的 index 因此漏掉了它的標籤。
- **沒有動的**：shipping 與 tracking 的 C++ 原始碼、entrypoint pin、operator library、engine、model root、`third_party_set.json`、launcher、auditor、安裝規則、`run_shipping_container.sh`、PR-12／C1／C2 的檢查。

### 19.3 開發期間已經看到的（在本節 commit 之前）

都是工作樹上的試做，不是正式 run（`results/465_prc3_dev/t1/`）：

- 以 PR-C2 A4 正式 run 的 tree（`results/465_prc2_cli/full_1ae402c2/tree`）、`--trial`（identity 尚未 republish）產生 package：94 秒；tarball 2.22 GiB（tree 3.7 GiB），56 個檔案。`tarball` 檢查除了 `metadata`（identity 不 current，照預期）之外 6 項通過。
- 乾淨容器（dash，沒有 Python／編譯器）安裝：exit 0。時間：digest 約 1.5 秒、解開約 15 秒、逐檔驗證約 2 秒（page cache 是熱的）。`install-trace` 5 項通過：TARGET 只被一次 `renameat2(…, RENAME_NOREPLACE) = 0` 碰到，其他 274 筆有寫入性質的呼叫全部在 staging 之內，staging 被刪除。安裝後的 tree：`static --manifest` 13 項通過、`install.sh --verify` 通過。第一次 trace 用了 `%desc`，記下了每一筆 read／write 的資料（11 GB），改成 `%file` 加 fd 類的寫入呼叫（1.3 MB）。
- 安裝後的 tree 在乾淨容器跑 MOT17-05（bundle 模式）：exit 0，MOT txt 與 A4 正式 run 的 `container_bundle` 相同（`6fe80348…`）。
- 安裝器負控制（容器）：1 GiB 的 tmpfs（磁碟滿）exit 1，tmpfs 是空的；15 秒時 SIGTERM（`timeout`，整個 process group）：TARGET 不存在、staging 已刪；15 秒時 SIGKILL：TARGET 不存在，staging（3.3 GB）留下。
- `tar` 的 `--no-overwrite-dir` 與 `--keep-old-files` 不能同時使用，去掉前者（staging 是空的，沒有可以覆寫的目錄）。

### 19.4 測量契約（正式 run 之前寫定）

**順序**：本節 commit 之後先 republish runtime identity（`docs/reference/runbooks/runtime_identity_republication.md`；`shipping/package/install.sh` 是新的 identity 輸入），正式 run 在 republish 之後的乾淨 commit 上執行，所以 package 記錄的座標描述它自己的 source commit。

**組態**：同一台機器。`build-release/`：§18.4 的 configure 參數（`-DSACCADE_SHIPPING_ENTRYPOINT=results/465_prc2_cli/entrypoint_04f6f135/saccade_track`），只 build `saccade_track`，安裝到 `$R/tree`。package 產生在 `$R/dist`。容器：安裝用 §17.4 的 pinned `ubuntu:24.04`（`install`）與加 strace 的映像（`install-strace`），不給 GPU、不給網路；執行用 §17.4 的 bundle 模式，tree 是從 package 安裝出來的那一份（`$R/installed/saccade`），唯讀掛載。oracle（`anchor`、`oracle-rows`）用 `build/` 既有的 extensions。全部 GPU 步驟在 gpu0 lease 下依序執行。

**有效性**（任一不成立 ⇒ 受影響的 gate 為 `UNRESOLVED`）：正式 run 在乾淨的 commit 上；`check_runtime_identity_staleness.py --mode attested` 在 run 開始時 exit 0；`build/libsaccade_scan_torchop.so` 與 entrypoint pin 檔案的 sha256 在 run 前後都分別等於 attestation 與 `entrypoint_pin.json` 的值；`git diff 1ae402c2 HEAD -- shipping/src shipping/include shipping/tools shipping/launcher shipping/cmake shipping/CMakeLists.txt shipping/third_party_set.json shipping/entrypoint_pin.json src include` 為空（tree 的來源與 PR-C2 A4 正式 run 相同）；`anchor` 與 PR-2L `A_L_1` 7/7 相同、`oracle-rows` 有效（§13.3）；兩種容器的 `container.txt` 顯示 Ubuntu 24.04、glibc 2.39，安裝容器的 `/bin/sh` 是 dash、沒有 Python 與編譯器。

**PASS 驗收規則**：PR-C3 的 verdict 是 `PASS` 若且唯若下列全部成立，否則是 `FAIL`（照 gate 分開報告）：

1. **build 與安裝**：同 §17.4 第 1 條（只 build `saccade_track`）。
2. **靜態檢查**：`check_shipping_bundle.py static` 對 `$R/tree` 12 項都 PASS。
3. **package**：
   - `build_shipping_package.py`（不帶 `--trial`）exit 0，記錄 `tree_clean: true`、`identity_current: true`。
   - `check_shipping_package.py tarball` 7 項都 PASS：`release_set`、`package_digest`、`installer_exact`、`tar_members`、`manifest_exact`、`pinned_tree`、`metadata`。
   - **決定性**：以同一個 tree 在同一個 commit 再產生一次到 `$R/dist_again`，三個檔案逐位元組相同。
   - MANIFEST 的 `files` 恰好是 `$R/tree` 的檔案，sha256 與大小逐項相同。
4. **從 tarball 安裝到乾淨容器**：
   - `run_package_container.sh install-strace $R/dist $R/installed/saccade` exit 0。
   - `check_shipping_package.py install-trace` 5 項都 PASS：`one_staging_directory`（在 TARGET 的上層目錄）、`target_only_by_one_noreplace_rename`（TARGET 只被一次成功的 `renameat2(<staging>/x/<name>, TARGET, RENAME_NOREPLACE)` 碰到，沒有其他寫入性質的呼叫，失敗的嘗試也算）、`other_mutations_in_staging`、`staging_removed`、`trace_complete`。這一條就是「PR-12 的 N2 不再成立」的直接證據。
   - 安裝後的 tree：`check_shipping_bundle.py static --manifest` 13 項都 PASS；安裝容器內 `install.sh --verify /install/saccade` exit 0；除 `MANIFEST.json` 之外，每個檔案與 `$R/tree` 的對應檔案逐位元組相同，沒有多也沒有少。
5. **乾淨容器執行（從 package 安裝的 tree）**：`run_shipping_container.sh bundle $R/installed/saccade`：7 sequence 在同一個 process 中跑完、exit 0；`parity --native-from` `EXACT`（`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7），且 `--against` PR-C2 A4 正式 run 的 `results/465_prc2_cli/full_1ae402c2/parity_bundle/report.json`：txt、trace 與 native graph 計數 7/7 相同。
6. **G2-2／G2-4（從 package 安裝的 tree）**：`bundle-strace`，`check_shipping_bundle.py runtime` 三項都 PASS；`parity --native-from` `EXACT`，且 `--against` 第 5 條 7/7 相同。

沒有容差。第 3–6 條的比較是逐位元組；任一不同就照 gate 報告第一個不同的檔案或項目，停在 PR-C3。

**安裝器負控制**（容器內以 dash 執行；被改的 package 放在各自的 dist 副本，未改的檔案是 hard link；「被重新包過」的 tarball 以 Python 從原 tarball 串流改寫一個 member，`<name>.sha256` 重算；每一條都要記錄 `install.log`、上層目錄安裝前後的列表與 `after.txt`）。除 P3、P10b 之外，必須的結果都包含：**TARGET 不存在、上層目錄沒有留下 `.saccade-install.*`、上層目錄的列表與安裝前相同**。

| # | 操作 | 必須的結果（在上面的共同結果之外） |
|:--|:--|:--|
| P1 | tarball 中間一個位元組反轉（digest 不重算） | exit 1，`sha256 is not the one in`；log 沒有 `extracting`（沒有解開任何東西） |
| P2 | 重新包：`lib/vendor/libcublas.so.13` 最後一個位元組反轉（大小不變），digest 重算 | exit 1，`lib/vendor/libcublas.so.13: sha256 is not`；`tarball`：`manifest_exact`、`pinned_tree` 失敗 |
| P3 | P2 的改動，再把 MANIFEST 裡那個檔案的 sha256 改成改過之後的值（一份「別人重新包過」、內部一致的 package），digest 重算 | **安裝器 exit 0**（digest 不是簽章，§19.1）；`tarball`：`pinned_tree`、`metadata` 失敗；安裝後的 tree：`static --manifest` 的 `vendor_set_pinned` 失敗。這一條記錄的是 PR-C3 擋不住什麼、由誰抓到 |
| P4 | 重新包：多一個 `share/saccade/helper.py`，digest 重算 | exit 1，訊息列出 `share/saccade/helper.py` |
| P5 | 重新包：少 `lib/vendor/libnvrtc.so.13`，digest 重算 | exit 1，訊息列出 `lib/vendor/libnvrtc.so.13` |
| P6a–d | TARGET 已存在：(a) 有一個 sentinel 檔案的目錄，(b) 空目錄，(c) 一般檔案，(d) 指向另一個目錄的 symlink | exit 2，`exists; nothing was changed`；TARGET 與 sentinel 的內容、種類不變（此列的「TARGET 不存在」改為「TARGET 與安裝前相同」） |
| P7 | `/install` 是 1 GiB 的 tmpfs（磁碟滿） | exit 1，`extraction failed`；`after.txt` 顯示 `/install` 是空的 |
| P8 | `timeout -s TERM 5`（解開期間） | log 的最後一行是 `extracting`；（`timeout` 的 exit 124） |
| P9 | `timeout -s TERM <d>`（逐檔驗證期間），`<d>` 依序試 17、16.5、17.5、16、18 秒，直到 log 的最後一行是 `checking` | 每一次嘗試都要滿足共同結果；`<d>` 與每次的最後一行照實記錄。五個值都沒落在驗證期間 ⇒ P9 記為 `UNRESOLVED`（不是 PASS） |
| P10a | `timeout -s KILL 8`（解開期間） | TARGET 不存在；**staging 留下**（照實記錄大小） |
| P10b | P10a 之後，同一個上層目錄正常安裝 | exit 0，log 有 `note: … is left from another installation`；P10a 的 staging 內容不變；安裝後 `install.sh --verify` exit 0 |
| P11 | 原 package 的 tarball 不經安裝器、直接 `tar -xzf` 解到上層目錄，TARGET＝`/install/<name>`（strace 映像，同第 4 條的 trace 設定） | `install-trace`：`target_only_by_one_noreplace_rename` 失敗。這一條顯示第 4 條的檢查分辨得出非 atomic 的安裝 |
| P12 | 安裝後 tree 的副本（hard link），`lib/vendor/libcudart.so.13` 先刪再寫入多一個位元組的版本 | `install.sh --verify` exit 1（`size is not`）；`static --manifest`：`manifest_tree`、`vendor_set_pinned` 失敗 |
| P13a | 刪掉 `<name>.sha256` | exit 2，`no package digest` |
| P13b | `<name>.sha256` 的 tarball 那一行重複一次 | exit 1，`does not name … exactly once` |
| P14 | tarball 與安裝器改名成 `saccade-0.1.1-…`（內容不變），digest 重算 | exit 1，`the tarball does not hold exactly one directory` |
| P15 | TARGET 的上層目錄名稱含 `:` | exit 2，`must not contain`；上層目錄沒有任何新項目 |

另外 `tests/unit/test_shipping_package.py` 涵蓋 symlink member、`..` member、MANIFEST 改名、MANIFEST 格式錯誤、TARGET 為 dangling symlink 與 `install-trace` 的拒絕案例；這些在合成 package 上做，不在 3.7 GiB 的 package 上重做。

**不做的**：FPS 或任何效能比較；host 經 launcher 的 parity 與 `sources`（tree 的位元組與 PR-C2 A4 正式 run 相同，第 4 條逐位元組確認）；PR-C2 的 CLI gate（6a–6d，entrypoint 沒有變）；PR-C1／C2 的 N1–N18、M1–M6（它們檢查的 tree、launcher、auditor、checker 都沒有變）；其他 GPU、其他主機、其他 glibc；minisign 簽章（PR-C4）；任何散佈（Phase C scope §4）。

### 19.5 驗收

同一台機器（RTX 5070 Ti Laptop，WSL2）。正式 run：commit `9217ed92`＝§19.4 的契約（`1f58aefe`）加上 runtime identity 的 republish（`chore/465-prc3-republish`；implementation 軸 286 → 287 個檔案，新增的只有 `shipping/package/install.sh`；probe 重跑，behavior `2dabed0b` 與 A4 出版相同），工作樹乾淨，契約早於任何量測。全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）；判定由 `evaluate.py` 從 run 的產物逐條讀出（`evaluation.json`）。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 工作樹乾淨；`check_runtime_identity_staleness.py --mode attested` exit 0；`git diff 1ae402c2 HEAD` 在 tree 的來源路徑上為空；operator library（`aa84cccd…`）與 pin 的 entrypoint（`92f74ef4…`）在 run 前後都等於 attestation 與 `entrypoint_pin.json`；`anchor` 與 `A_L_1` 7/7 相同，`oracle-rows` OK；安裝容器是 Ubuntu 24.04.4、glibc 2.39、`/bin/sh`＝dash、沒有 Python 與編譯器 |
| 1 build 與安裝 | configure、build、install 都 exit 0 |
| 2 靜態檢查 | 12 項 PASS |
| 3 package | builder exit 0，`tree_clean: true`、`identity_current: true`；`tarball` 7 項 PASS（87 個 member、56 個檔案）；再產生一次，三個檔案逐位元組相同（tarball `21c9ac08…`，2,384,765,322 位元組；安裝器 `afa5d06a…`；digest 檔 `8876e3d3…`）；MANIFEST 的 `files` 與 `$R/tree` 逐項相同 |
| 4 從 tarball 安裝到乾淨容器 | exit 0。`install-trace` 5 項 PASS：TARGET 只被一次 `renameat2("/install/.saccade-install.V3UXg2/x/<name>", "/install/saccade", RENAME_NOREPLACE) = 0` 碰到；其他 274 筆寫入性質的呼叫全部在 staging 之內；staging 已刪；0 筆 unresolved。安裝後的 tree：`static --manifest` 13 項 PASS，容器內 `install.sh --verify` exit 0，除 `MANIFEST.json` 外與 `$R/tree` 逐位元組相同 |
| 5 乾淨容器執行（從 package 安裝的 tree） | exit 0；`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；`--against` PR-C2 A4 正式 run 的 `parity_bundle`：txt、trace、graph 計數 7/7 相同 |
| 6 G2-2／G2-4 | `runtime` 三項 PASS；`EXACT`，與第 5 條 7/7 相同；兩次 run 的 `python_libraries_mapped` 都是空的 |
| **verdict** | **`PASS`**（§19.4 第 1–6 條全部成立） |

**安裝器負控制**（`negctl/`，全部照必須的結果）：

| # | 結果 |
|:--|:--|
| P1 | exit 1，`sha256 is not the one in`；log 沒有 `extracting`；上層目錄不變，沒有 staging |
| P2 | exit 1，`lib/vendor/libcublas.so.13: sha256 is not`；上層目錄不變；`tarball`：`manifest_exact`、`pinned_tree`、`metadata` 失敗 |
| P3 | **安裝器 exit 0**（內部一致、digest 重算過的 package；digest 不是簽章）；`tarball`：`pinned_tree`、`metadata` 失敗；安裝後的 `static --manifest`：`vendor_set_pinned` 失敗 |
| P4 | exit 1，訊息列出 `share/saccade/helper.py`；上層目錄不變 |
| P5 | exit 1，訊息列出 `lib/vendor/libnvrtc.so.13`；上層目錄不變 |
| P6a–d | 四種都 exit 2，`exists; nothing was changed`；上層目錄（含 TARGET 與 sentinel）不變，沒有 staging |
| P7 | exit 1，tar 的 `No space left on device` 之後 `extraction failed`；`/install` 是空的 |
| P8 | 5 秒 SIGTERM：最後一行是 `extracting`；上層目錄不變，沒有 staging（`timeout` exit 124） |
| P9 | 第一個延遲（17 秒）就落在驗證期間（最後一行是 `checking 56 files against MANIFEST.json`）；上層目錄不變，沒有 staging |
| P10a | 8 秒 SIGKILL：TARGET 不存在；staging `.saccade-install.Htt5Lc` 留下（1,769,617,041 位元組） |
| P10b | exit 0，log 先有 `note: /install/.saccade-install.Htt5Lc is left from another installation`；那個 staging 的列表不變；`--verify` exit 0 |
| P11 | 直接 `tar -xzf`：`install-trace` 的 `target_only_by_one_noreplace_rename`、`one_staging_directory`、`staging_removed` 失敗 |
| P12 | `--verify` exit 1（`size is not`）；`static --manifest`：`manifest_tree`、`vendor_set_pinned` 失敗 |
| P13a | exit 2，`no package digest` |
| P13b | exit 1，`does not name … exactly once` |
| P14 | exit 1，`the tarball does not hold exactly one directory` |
| P15 | exit 2，`TARGET must not contain ':' or ';' (/install/a:b/saccade)`；上層目錄不變 |

**評估器的修正（不是契約的修改）**：第一次評估（`evaluate.r1.py`）把 P10a 判成失敗：它用遞迴的列表找留下的 staging，把 staging 底下的子目錄也算成「留下的 staging」（6 筆）。§19.4 要的是 TARGET 不存在、staging 留下，這兩點在產物上都成立；`evaluate.py` 改成只看上層目錄那一層，其他判定沒有變。`evaluation.json` sha256：`6f8e1210…`。

**PR-12 N2 的狀態**：package 的安裝路徑上，被拒絕的安裝不會在 TARGET 留下任何東西（P1、P2、P4、P5、P7、P8、P9、P13–P15），而成功的安裝在 TARGET 上只有一次不會取代既有項目的 rename（第 4 條）；直接解開的對照（P11）被同一個檢查抓到。`cmake --install` 仍然不是 atomic（開發路徑，§19.1）。

結果目錄：`results/465_prc3_package/full_9217ed92/`（`run.sh`、`evaluate.py`、`evaluation.json`、`tree/`、`dist/`、`dist_again/`、`static*.json`、`package.json`、`install/`、`install_trace.json`、`installed/`、`anchor/`、`oracle_rows/`、`container_*`、`parity_*`、`runtime.json`、`negctl/`、`pins_before.txt`／`pins_after.txt`、`validity_*.txt`）；開發試做在 `results/465_prc3_dev/t1/`。不納入版本控制。

### 19.6 限制

- **digest 不是簽章**（P3）：安裝器擋得住損壞、被截斷、被改了一個檔案而 MANIFEST 沒跟著改的 package，擋不住一份重新包過而內部一致、digest 也重算的 package。抓到 P3 的是 `tarball` 檢查與 `static --manifest`（對照 repository 的 pin），使用者手上沒有它們；簽章是 PR-C4。
- **安裝器驗不了自己**：`<name>.install.sh` 在 digest 裡，但驗證要由使用者做（`sha256sum -c`，PR-C4 之後是簽章）。
- **安裝後的 `--verify` 信任 tree 裡的 MANIFEST**：同時改了檔案與 MANIFEST 的 tree 會通過 `--verify`（同上一點，要 PR-C4 簽 MANIFEST 或由 digest 對照）。
- **digest 與解開讀 tarball 兩次**：在兩次讀取之間被換掉的 tarball，會以解開時的內容與它自己的 MANIFEST 比對。能在安裝期間改寫 tarball 的人通常也能改寫安裝器，這裡不宣稱防得住。
- **SIGKILL 會留下 staging**（P10a）：TARGET 不受影響；staging（最多約 tree 的大小）要使用者自己刪，下一次安裝會指出它。
- **`RENAME_NOREPLACE` 依賴檔案系統**：GNU coreutils 在檔案系統不支援時會退回「先檢查再 rename」，那時「TARGET 在安裝期間出現」的保護不是 atomic。本 run 只在 WSL2 的 ext4（bind mount）上確認是 `renameat2(…, RENAME_NOREPLACE) = 0`；其他檔案系統沒有驗證。
- **安裝器只在 dash（容器）上正式驗證**：單元測試在 host 的 `sh`（bash 的 POSIX 模式）上跑；其他 sh 實作沒有驗證。
- **決定性只在同一台機器上確認**：tarball 的 gzip 位元組依賴 Python 的 zlib；換一台機器重新產生不保證相同。發佈的是 digest 綁定的那一份。
- **P9 的時機**：逐檔驗證在 page cache 熱的時候只有約 2 秒，P9 靠事先宣告的延遲清單命中；它證明的是「這次落在驗證期間的 SIGTERM」，不是任意時間點。
- 其餘同 §17.6、§18.6：只支援 sm_120、只在這一台 WSL2 機器與 driver 上驗證；launcher 的 sh 不在保護範圍；package 約 2.2 GiB（tarball）／3.7 GiB（安裝後）；**沒有散佈**（Phase C scope §4 的授權確認之前不公開 package）。

### 19.7 重現

```bash
cmake --install build-release --component shipping --prefix <tree>      # §18.7 configure + build
.venv/bin/python scripts/native/build_shipping_package.py --tree <tree> --out <dist> --static-report static.json
.venv/bin/python scripts/native/check_shipping_package.py tarball --dist <dist> --report package.json
bash scripts/native/run_package_container.sh install-strace <dist> <parent>/saccade <out>
.venv/bin/python scripts/native/check_shipping_package.py install-trace --strace-prefix <out>/strace/i \
    --target /install/saccade --package <name> --report install_trace.json
.venv/bin/python scripts/native/check_shipping_bundle.py static --tree <parent>/saccade --manifest --report static.json
sh <dist>/<name>.install.sh <dist>/<name>.tar.gz <target>          # an end user's install
sh <dist>/<name>.install.sh --verify <target>
bash results/465_prc3_package/<label>/run.sh && .venv/bin/python results/465_prc3_package/<label>/evaluate.py
```

## 20. Release readiness：minisign 簽章、授權稽核、README（Phase C PR-C4）

PR-C4 是 Phase C 的最後一個 PR（[Phase C scope](native_runtime_phase_c_scope.md) §6）：C-D5 的 minisign 簽章落地，§4 的授權讀法逐物件重新稽核並寫進 `THIRD_PARTY.md`，tree 多一份 `README.txt`，並以一次正式 run 從乾淨容器驗證「簽章可被驗證、竄改後驗證失敗、從最終 tarball 安裝後跑完 7-seq EXACT」。PR-C4 不改任何 stage 的計算、entrypoint（pin `92f74ef4…`）、operator library（`aa84cccd…`）、27 個第三方物件與 `third_party_set.json`、launcher、auditor、安裝器（`shipping/package/install.sh` 與 PR-C3 逐位元組相同）、SM 清單與 glibc baseline。

| 項目 | 位置 |
|:--|:--|
| 授權稽核（逐物件的證據、條件、狀態、風險） | `shipping/license_audit.json` |
| 稽核檢查與 `THIRD_PARTY.md` 的 render | `scripts/native/license_audit.py`（`check`、`render`） |
| README（裝在 `<prefix>/README.txt`） | `shipping/package/README.txt` |
| 簽章、trusted comment、minisign 格式 reader | `scripts/native/sign_shipping_package.py` |
| package 檢查 | `check_shipping_package.py tarball --pubkey`（第四個檔案與 `signature` 檢查） |
| 使用者端的驗證（容器） | `run_package_container.sh verify` |
| 測試 | `tests/unit/test_license_audit.py`、`tests/unit/test_package_signature.py` |

**owner 指示（2026-10-08）**：

- §4：不把 nvJitLink、cuFile、nvshmem 一律判為不可再散布，也不直接放行公開發行。要重新稽核：逐一核對每個實際打包的 `.so`、它的來源 wheel 與版本、wheel 附帶的授權檔，以及版本對應的官方條款，判定是否適用。
  - **不得以授權檔 sha256 相同推斷授權相同**。
  - 證據未閉合前維持 local-only／no-public-distribution，PR-C4 的工程驗證可以繼續。
  - `THIRD_PARTY.md` 逐項列出證據、適用條件與未解風險。
- 金鑰：release key 由 owner 產生並保管，只 commit 公鑰；驗收與負控制用一次性的 test key，secret key 不經過 agent。
- 驗證位置：使用者在安裝**之前**驗證（`minisign -V` 再 `sha256sum -c`）。安裝器維持只用 base system 工具，不改。

### 20.1 設計決定

**授權稽核**

- `shipping/license_audit.json` 對 `third_party_set.json` 的 27 個物件各一筆，每筆記錄四件事。
  - 來源 wheel，以及 release：wheel 版本與官方版本的對應，例如 CUDA wheel 對 CUDA 13.0 Update 2 release notes 的 component 表。
  - wheel 附帶、隨 package 出貨的授權檔（`licenses/<wheel>/`）裡對這個物件的宣稱，三種之一：
    - Attachment A 列出／沒列出它的名字；
    - 一段逐字引用的條款；
    - 某個字串不存在。
  - 版本對應的官方條款裡對它的宣稱：URL、抓取時間、snapshot 的 sha256，以及 Attachment A、引用條款、或「與出貨的檔案相同」三種之一。
  - 條件、狀態、風險。
- 狀態四種：
  - `grant_in_bundled_and_official`
  - `grant_in_both_texts_differ`
  - `grant_in_official_only`（出貨的文本沒提到它，版本對應的官方條款准許）
  - `no_licence_text_shipped`
- **每個物件以它自己 wheel 的檔案逐一檢查。** 10 個 CUDA 系列 wheel 與 nvshmem wheel 的 `License.txt` 位元組相同（`ad6f5853…`），但稽核不從這一點推論。檢查器讀每個物件自己 wheel 的檔案，找那個物件的名字；測試以兩個位元組相同的合成 wheel 確認：對 A 成立的宣稱不會延用到 B。
- 「官方條款」是 NVIDIA／上游以版本標示的頁面（§20.8 review 修正後的規則）：
  - docs 的 version switcher 值等於 wheel 版本；
  - 或 release notes 的 component 表中**該物件那一列**的版本等於 wheel 版本（不是頁面上任何地方出現這個版本）；
  - 或 URL 的 tag 等於該版本。
  - **每一個准許散佈的官方宣稱**，它的來源要對應到這個物件已驗證的 release：同一個來源、或同版本的 archive 頁面（CUDA EULA `archive/13.0.2/` 對 release notes `archive/13.0.2/`）、或 switcher／tag 對得上。只供參考的頁面（目前版的 CUDA EULA、NVSHMEM 2.8.0）不能當依據。
  - 狀態宣稱官方條款准許（三種 `grant_*`），就至少要有一個准許散佈的官方宣稱；出貨文本准許的物件不能標成 `grant_in_official_only`。
  - 沒有版本的來源只在「它就是這個物件自己的 release 來源、而且風險已記錄」時接受，記成 `unmatched_releases`（不是通過的證據）；其他情況一律失敗。目前有兩筆：cuSPARSELt 的授權頁沒有版本；libgomp 的 GCC 版本不在物件裡。
- snapshot 是抓取當時的 HTML／文字，放在 `results/465_prc4_license/sources_<date>/`，不納入版本控制。稽核記錄每個 snapshot 的 sha256，條款原文則以逐字引用記在 JSON 裡（只引用判定所需的句子）。
- **稽核結果**（2026-10-08；讀法，不是法律結論）：

  | 物件 | 狀態 | 依據 |
  |:--|:--|:--|
  | cudart、cufft、cublas、cublasLt、curand、cusparse、nvrtc、nvjpeg、cupti（9 個） | `grant_in_bundled_and_official` | wheel 的 CUDA EULA（2018 文本）與 CUDA 13.0 Update 2 EULA（docs archive，last updated 2025-01-07）的 Attachment A 都列出 |
  | **libnvJitLink、libcufile** | `grant_in_official_only` | wheel 的 2018 文本**沒有**列出；CUDA 13.0 Update 2 EULA 的 Attachment A 列出 `libnvJitLink.so`、`libcufile.so`，而 13.0 Update 2 的 component 表正是這兩個 wheel 的版本（13.0.88、1.15.1.6） |
  | **libnvshmem_host** | `grant_in_official_only` | wheel 附的是 CUDA EULA（不提 NVSHMEM）；NVSHMEM 3.4.5 文件（version switcher＝3.4.5）的 SLA supplement：「distributable under the Agreement: any portion of the SDK」 |
  | cuDNN（5 個） | `grant_in_both_texts_differ` | wheel：「runtime files .so and .h, cudnn64_7.dll, and cudnn.lib」；9.19.0 文件：「runtime files .so and .dll」 |
  | libnvinfer | `grant_in_both_texts_differ` | wheel 的 `LICENSE.txt` 是 NVIDIA Software License Agreement＋TensorRT Supplement §12.1（libnvinfer／plugin，一年期自動續約，下游須同等限制）；10.16.1 文件是 SDK agreement（v. May 24, 2021）＋supplement「runtime files .so and .dll」：兩份不同的協議 |
  | cusparseLt | `grant_in_bundled_and_official` | wheel 與官方頁的 supplement 相同（v. October 12, 2020）；官方頁沒有版本 |
  | NCCL | `grant_in_bundled_and_official` | BSD-3-Clause；wheel 的 `License.txt` 與 tag v2.28.9-1 的 `LICENSE.txt` 逐位元組相同 |
  | torch 系列（6 個） | `grant_in_bundled_and_official` | wheel 的 `LICENSE` 開頭等於 tag v2.11.0 的 `LICENSE`（後面接 66 段 bundled 第三方），`NOTICE` 逐位元組相同 |
  | **libgomp** | `no_licence_text_shipped` | GNU libgomp（字串帶 `../../../libgomp/`），上游為 GPL-3.0-or-later WITH GCC-exception-3.1；torch 的 `LICENSE`／`NOTICE` 都沒有提到它（`LICENSE` 裡的 GPL-3.0 全文屬於 `cpr/test`）。package 沒有附 GPL／exception 文本與 source offer，物件也沒有記錄 GCC 版本（**更正見 §22.1**：版本有記錄；#547 已附文本與 source 指引） |

  分佈：17 筆 `grant_in_bundled_and_official`，6 筆 `grant_in_both_texts_differ`，3 筆 `grant_in_official_only`，1 筆 `no_licence_text_shipped`。`distribution.status`＝`local-only`，`owner_confirmation`＝null。
- **仍待 owner 的項目**（`THIRD_PARTY.md` 的「Open items」逐物件列出）：
  1. 出貨文本沒提到、官方版本對應條款准許的三個物件，哪一份文本適用於「以 wheel 取得」的物件。若要以官方條款為據，是否把那份條款文本也放進 `licenses/`。
  2. libnvinfer、cuDNN 兩份文本不同時以哪一份為準；TensorRT wheel 文本的一年期與「下游同等限制」條件如何滿足。
  3. libgomp：附上 GPL-3.0 與 GCC Runtime Library Exception 文本並決定 source 的提供方式，或改用其他處理。
  4. NVIDIA 各條款共通的條件：「material additional functionality」、「only accessed by your application」、「不得使之受 open source license 約束」。package 已把它們標成第三方元件、不在 Apache-2.0 之下，但「是否滿足」是 owner 的判斷。

  上述項目關閉之前不公開任何 package。這是 owner 的決定，不是 PR-C4 run 的 gate。
- `THIRD_PARTY.md` 由 `license_audit.py render` 從 `third_party_set.json`＋`license_audit.json` 產生，不再由 `export_third_party_set.py` 產生（它的 `--notice` 移除；換第三方集合就要重做稽核）。MANIFEST 多一個 `licenses` 鍵，記錄稽核檔的 sha256 與 `distribution`（`local-only`），讓 package 自己帶著散佈狀態；source commit 早於 PR-C4（沒有稽核檔）時不寫這個鍵，所以舊 package 的 MANIFEST 仍然推得出來。

**簽章**

- **簽的是 `<name>.sha256`**（PR-C3 的 package digest，已經涵蓋 tarball 與安裝器），簽章檔是 `<name>.sha256.minisig`，release set 變成四個檔案。只簽一個檔案，使用者只要一次 `minisign -V`，安裝器的完整性也一起涵蓋（§19.6「安裝器驗不了自己」）。
- **trusted comment**＝`package=<name> commit=<source commit> manifest_sha256=<MANIFEST.json 的 sha256>`，由 tarball 裡的 MANIFEST 推出（`sign_shipping_package.py trusted-comment`）。minisign 對 trusted comment 另有一個簽章，所以它也是被簽的內容。這一項處理 §19.6 的「安裝後的 `--verify` 信任 tree 裡的 MANIFEST」：使用者以 `sha256sum <prefix>/MANIFEST.json` 對照簽過的 `manifest_sha256`。
- **使用者在安裝之前驗證**（owner 指示）：`minisign -Vm <name>.sha256 -p minisign.pub` 再 `sha256sum -c <name>.sha256`，之後才執行安裝器。安裝器不讀簽章、不需要 minisign，與 PR-C3 逐位元組相同，所以 runtime identity 的 installer pin 也不變。
- **兩個獨立的驗證**：`tarball --pubkey` 的 `signature` 檢查同時要求兩者成立：
  - `minisign -V`（外部程式）；
  - `sign_shipping_package.py` 自己的 reader（`cryptography` 的 Ed25519；`ED`＝BLAKE2b-512 prehash，`Ed`＝legacy）；
  - 另外，trusted comment 等於 tarball 的 MANIFEST 推出的那一個。

  容器裡的使用者流程（`run_package_container.sh verify`）用 Ubuntu 24.04 apt 的 minisign 0.11，與 host 的 minisign 是不同的 build。
- **金鑰管理**：
  - 產生：release key 由 owner 以 `minisign -G`（有密碼）在自己的機器產生；secret key 不進 repository、CI 或 agent 的環境。
  - 公開：只 commit 公鑰 `shipping/package/minisign.pub`（以 owner review 的 PR），key id 寫進本節。README 要求使用者從 repository 取得公鑰，不從下載 package 的地方取得。
  - 簽署：`sign_shipping_package.py sign --dist <dist> --secret-key <key>`（呼叫 minisign，密碼由 minisign 詢問）。
  - 輪替：新公鑰檔先以舊 key 簽（`minisign -S -m minisign.pub`），新舊公鑰與這個簽章一起 commit；之後的 release 只用新 key。
  - 撤銷：key 外洩時，從 repository 移除公鑰並記錄撤銷，以新 key 重簽仍要提供的 release。
  - **本 PR 沒有 commit release 公鑰**（owner 產生之後另以 PR 加入）；正式 run 只用 test key。
- **README.txt** 裝在 `<prefix>/README.txt`（`install_third_party.cmake`），內容是：
  - 需求：GPU＝sm_120；driver：CUDA 13.0 Update 2 的 ≥ 580.95.05（NVIDIA release notes），驗證過的是 WSL2 的 616.92（Linux UMD 615.71.09）；glibc ≥ 2.39；約 3.7 GiB；
  - 驗證、安裝、`--verify` 的步驟；
  - 執行的用法；
  - named limits；
  - 授權的位置與散佈狀態。

  README 不寫版本字串（以 `<name>` 指 MANIFEST 的 `package`），所以換版本不必改它。`static` 的 `layout_exact` 與 `licenses`、`tarball` 的 `pinned_tree` 都把它算進去（與 repository 的檔案逐位元組相同）。
- **runtime identity**：`shipping/**` 是 identity 的輸入（README、稽核、`THIRD_PARTY.md`、install 規則都在裡面），所以與 PR-C3 相同，**republish 在正式 run 之前**。

### 20.2 改了什麼

- 新檔案：
  - `shipping/license_audit.json`、`shipping/package/README.txt`；
  - `scripts/native/license_audit.py`、`scripts/native/sign_shipping_package.py`；
  - `tests/unit/test_license_audit.py`、`tests/unit/test_package_signature.py`。
- `shipping/THIRD_PARTY.md`：改由稽核 render，加上散佈狀態、每個物件的證據與狀態、Open items、條件原文、來源清單。
- `shipping/cmake/install_third_party.cmake`：多裝 `README.txt`。`shipping/CMakeLists.txt` 只改註解。
- `check_shipping_bundle.py`：layout 多 `README.txt`，`licenses` 也比對 README，項目數不變（12／13）。
- `check_shipping_package.py`：
  - `tarball --pubkey`（release set 四個檔案，多一項 `signature`）；
  - `pinned_tree` 比對 README；
  - MANIFEST 的 `licenses` 鍵；
  - `reading` 改寫。
- `run_package_container.sh verify`；`export_third_party_set.py` 去掉 `--notice`。
- **沒有動的**：shipping 與 tracking 的 C++ 原始碼、`install.sh`、launcher、auditor、`third_party_set.json`、`entrypoint_pin.json`、model root、operator library。

### 20.3 開發期間已經看到的（在本節 commit 之前）

都是工作樹上的試做，不是正式 run（`results/465_prc4_dev/t1/`）：

- `build-release/`（§18.4 的 configure，沒有重新 build）`cmake --install` 寫出含 `README.txt` 的 tree：`static` 12 項 PASS。
- `license_audit.py check --licenses <tree>/licenses --sources results/465_prc4_license/sources_20261008`：四項 PASS。
  - 第一次 `official_terms` 失敗：HTML 版 EULA 的 Attachment A 開頭是「The following CUDA Toolkit files may be distributed…」，不是 wheel 文本的「distributable under the Agreement」，取 section 的條件太窄。修正為「heading 到 Attachment B 之間最長的一段」。
- `--trial` package：92 秒，tarball 2,384,777,133 位元組，安裝器 `afa5d06a…`（與 PR-C3 相同）。
- host 沒有 minisign（Arch），所以試做的 test key 與簽章在 `saccade-minisign:ubuntu24.04` 容器裡以 minisign 0.11 產生（`-G -W`，演算法 `ED`）。之後：
  - reader 驗證通過，trusted comment 等於 MANIFEST 推出的值；
  - `run_package_container.sh verify` exit 0（`Signature and comment signature verified`、兩行 `OK`）。
  - digest 第一個 hex 字元改掉（簽章不變）：容器 `Signature verification failed` exit 1，reader `the signature does not verify`。

### 20.4 測量契約（正式 run 之前寫定）

**順序**：本節 commit 之後先 republish runtime identity。正式 run 在 republish 之後的乾淨 commit 上執行；host 要有 `minisign`（`signature` 檢查與 `sign` 用它）。

**組態**：同一台機器，與 §19.4 相同。差別：
- `$R/tree` 由同一個 `build-release/` 安裝，多 `README.txt`；
- test key 在 run 開始時以容器的 minisign 產生到 `$R/testkey/`（`-G -W`，不設密碼，標記為 TEST；secret key 留在結果目錄，不納入版本控制）；
- 簽章以 host 的 `sign_shipping_package.py sign`；
- 授權 snapshot 從 `results/465_prc4_license/sources_20261008/` 複製到 `$R/license_sources/`；
- oracle 與容器同 §19.4。

**有效性**（任一不成立 ⇒ 受影響的 gate 為 `UNRESOLVED`）：
- 乾淨 commit；run 開始時 `check_runtime_identity_staleness.py --mode attested` exit 0。
- operator library 與 entrypoint pin 檔案的 sha256 在 run 前後都等於 attestation 與 `entrypoint_pin.json`。
- `git diff 1ae402c2 HEAD -- shipping/src shipping/include shipping/tools shipping/launcher shipping/third_party_set.json shipping/entrypoint_pin.json src include` 為空；`git diff 9217ed92 HEAD -- shipping/package/install.sh` 為空。
- `anchor` 與 `A_L_1` 7/7 相同，`oracle-rows` 有效。
- 容器顯示 Ubuntu 24.04、glibc 2.39；安裝容器 `/bin/sh`＝dash、沒有 Python 與編譯器；verify 容器只多 minisign。

**PASS 驗收規則**：verdict 是 `PASS` 若且唯若下列全部成立，否則 `FAIL`（照 gate 分開報告）：

1. **build 與安裝**：同 §19.4 第 1 條。
2. **靜態檢查**：`check_shipping_bundle.py static` 對 `$R/tree` 12 項 PASS。
3. **授權稽核**：`license_audit.py check --licenses $R/tree/licenses --sources $R/license_sources` exit 0，`coverage`、`bundled_texts`、`official_terms`、`notice` 四項 PASS，`complete: true`；`$R/tree/licenses/THIRD_PARTY.md` 與 render 相同。
4. **package 與簽章**：
   - builder（不帶 `--trial`）exit 0，`tree_clean`、`identity_current` 為 true；MANIFEST 的 `licenses` 等於稽核檔的 sha256 與 `local-only`。
   - `sign` exit 0。
   - `tarball --pubkey $R/testkey/test.pub` 8 項 PASS：`release_set`（四個檔案）、`package_digest`、`installer_exact`、`tar_members`、`manifest_exact`、`pinned_tree`、`metadata`、`signature`（minisign exit 0、reader 無問題、trusted comment 相同）。
   - **決定性**（§20.4a 修訂）：同一個 tree、同一個 commit 再產生到 `$R/dist_again`，三個 package 檔案（tarball、安裝器、digest）逐位元組相同；再以同一把 test key 簽，`dist_again` 的簽章在同一把公鑰下以 `minisign -V` 與 reader 都驗證通過，trusted comment 與 `dist` 的相同。簽章檔本身不要求逐位元組相同。
5. **使用者端驗證**：`run_package_container.sh verify $R/dist $R/testkey/test.pub` exit 0；log 有 `Signature and comment signature verified`、等於第 4 條的 trusted comment、兩行 `: OK`。
6. **從 tarball 安裝到乾淨容器**：同 §19.4 第 4 條（`install-strace` exit 0、`install-trace` 5 項、`static --manifest` 13 項、容器內 `--verify` exit 0、除 MANIFEST 外與 `$R/tree` 逐位元組相同），加上：安裝後 `MANIFEST.json` 的 sha256 等於 trusted comment 的 `manifest_sha256`。
7. **乾淨容器執行**：`run_shipping_container.sh bundle $R/installed/saccade`：
   - 7 sequence、exit 0；
   - `parity --native-from` `EXACT`（`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7）；
   - `--against` PR-C3 正式 run 的 `results/465_prc3_package/full_9217ed92/parity_bundle/report.json` 7/7 相同。
8. **G2-2／G2-4**：`bundle-strace`，`runtime` 三項 PASS；`EXACT` 且 `--against` 第 7 條 7/7 相同。

沒有容差。

**簽章負控制**：每一條都在自己的 dist 副本做（未改的檔案是 hard link），記錄兩項：
- 容器 `verify`（minisign 0.11）的 `verify.log`；
- host `tarball --pubkey` 的 `signature`（minisign＋reader）。

「verify 失敗」＝`verify.log` 的 exit 不是 0。

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| S1 | `<name>.sha256` 第一個 hex 字元改掉（簽章不變） | verify 失敗（`Signature verification failed`）；`signature` FAIL（minisign 非 0、reader `the signature does not verify`） |
| S2 | §19.4 P3 的重新包（`libcublas.so.13` 最後一個位元組反轉、MANIFEST 跟著改），digest 重算，簽章不變 | verify 失敗；`signature` FAIL。這一條是 §19.6「digest 不是簽章」被簽章擋下 |
| S3 | 安裝器多一行註解，digest 重算，簽章不變 | verify 失敗；`signature` FAIL |
| S4 | `.minisig` 的 trusted comment 的 `manifest_sha256` 改成 64 個 0 | verify 失敗（comment signature）；`signature` FAIL（reader `the trusted comment's signature does not verify`） |
| S5 | 以另一把 test key（`$R/testkey2`）簽，trusted comment 正確 | verify 失敗；`signature` FAIL（key id） |
| S6 | tarball 中間一個位元組反轉，digest **不**重算 | `verify.log` 有 `Signature and comment signature verified`（簽章本身有效），之後 `sha256sum -c` 的 tarball 那一行 `FAILED` ⇒ verify 失敗。這一條說明兩步都要做。host 的 `tarball` 讀不了被改的 gzip（exit 2）或 `package_digest` FAIL，照實記錄 |
| S7 | 刪掉 `.minisig` | verify 失敗；`release_set`、`signature` FAIL |
| S8 | 以 test key 對正確的 digest 簽，但 trusted comment 的 `manifest_sha256` 是 64 個 0 | **minisign 與 verify 通過**（簽章本身有效）；`signature` FAIL（trusted comment 不同）。這一條記錄 trusted comment 的比對要由使用者或檢查器做 |
| S9 | 安裝後 tree 的副本（hard link）：`lib/vendor/libcudart.so.13` 換成多一個位元組的版本，MANIFEST 的那一行跟著改 | `install.sh --verify` **exit 0**（§19.6 的限制）；`sha256sum MANIFEST.json` 不等於 trusted comment 的 `manifest_sha256`；`static --manifest` 的 `vendor_set_pinned` FAIL |

**授權稽核負控制**：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| L1 | `$R/tree/licenses` 的副本：`nvidia_cufile-1.15.1.6/License.txt` 換成 CUDA 13.0.2 EULA 的文字 snapshot（裡面列出 `libcufile.so`） | `bundled_texts` FAIL（sha256 不是集合的；`libcufile.so.0` 的「Attachment A 沒列出」不成立） |
| L2 | `$R/license_sources` 的副本：`nvshmem_sla_3.4.5.html` 最後一個位元組改掉 | `official_terms` FAIL（snapshot sha256） |
| L3 | 稽核檔的副本：`libnvJitLink.so.13` 的 `bundled.attachment_a.listed` 改成 true、狀態改成 `grant_in_bundled_and_official` | `bundled_texts` FAIL（`does not list 'libnvJitLink.so'`） |

**不做的**：
- FPS；
- host 經 launcher 的 parity；
- PR-C3 的安裝器負控制 P1–P15（安裝器沒有變）；
- PR-C1／C2 的 N1–N18、M1–M6；
- 其他 GPU、主機、glibc；
- release key 的產生與 commit（owner）；
- 任何散佈（§20.1，`local-only`）。

### 20.4a 契約修訂（r1 之後，2026-10-08）

**r1 的結果**：正式 run r1（`results/465_prc4_release/full_bfd086c7/`，commit `bfd086c7`）依 §20.4 原文是 `FAIL`。其他部分都成立：
- 有效性、gate 1–3、5–8 都成立；
- gate 4 的 `tarball --pubkey` 8/8 PASS；
- S1–S9、L1–L3 全部照必須的結果。

唯一不成立的是 gate 4 的「四個檔案逐位元組相同」：三個 package 檔案相同，但兩個 `.minisig` 不同。

**原因**：原文的前提「Ed25519 簽章是決定性的」對這台 host 的簽署工具不成立，是契約寫錯，不是 package 的問題。
- 小檔案的對照：host 的 minisign 0.12（Arch，libsodium 1.0.22）同一把 key、同一個檔案簽兩次，簽章不同；容器裡的 minisign 0.11（Ubuntu 24.04）兩次相同。
- `dist_again` 的簽章在同一把公鑰下以 `minisign -V` 與 reader 都驗證通過，trusted comment 與 `dist` 的相同。

**修訂**：gate 4 的決定性改成上面那一條：三個 package 檔案逐位元組相同，第二次的簽章可驗證且 trusted comment 相同。其他條文不變。

**順序**：本修訂 commit 之後，在新的乾淨 commit 上**從頭**重跑整個正式 run（r2），不沿用 r1 的任何產物。r1 的產物與它的 `FAIL` 保留，在 §20.5 照實記錄。


### 20.5 驗收

同一台機器（RTX 5070 Ti Laptop，WSL2，driver 616.92）。

**r1**（`results/465_prc4_release/full_bfd086c7/`，commit `bfd086c7`）＝`FAIL`。

- 唯一不成立的是 gate 4 原文的「四個檔案逐位元組相同」（§20.4a）。
- 其他 gate、S1–S9、L1–L3 都照必須的結果。
- r1 的產物不被 r2 使用。

**r2** 是正式結果。
- commit `bc75dcab`＝§20.4 的契約（`992213ee`）＋§20.4a 修訂（`f2ff5700`）＋runtime identity 的 republish（`chore/465-prc4-republish`）。republish 內容：
  - implementation 軸 287 → 288 個檔案：新增 `shipping/license_audit.json`，改 `shipping/cmake/install_third_party.cmake` 與 `shipping/CMakeLists.txt`；
  - probe 重跑，behavior `2dabed0b` 與 PR-C3 出版相同。
- 工作樹乾淨，契約與修訂都早於 r2 的任何量測。
- 全部 GPU 步驟在 gpu0 lease 下依序執行（`run.sh`）。
- 判定由 `evaluate.py` 從產物讀出（`evaluation.json` sha256 `daa17f1c…`）。
- host minisign 0.12（Arch），容器 minisign 0.11（Ubuntu 24.04 apt）。
- test key `52320762B09F4D86`；S5 用的第二把 `3D223A859A587302`。

| 驗收項 | 結果 |
|:--|:--|
| 有效性 | 下列全部成立 |
| 有效性：工作樹與 identity | 工作樹乾淨；`--mode attested` exit 0 |
| 有效性：來源未變 | tree 的來源路徑對 `1ae402c2` 沒有 diff；`install.sh` 對 `9217ed92` 沒有 diff |
| 有效性：pin | operator library 與 entrypoint pin 在 run 前後都等於 attestation 與 pin |
| 有效性：oracle | `anchor` 與 `A_L_1` 7/7 相同；`oracle-rows` OK |
| 有效性：容器 | 安裝與 verify 容器都是 Ubuntu 24.04.4、dash；安裝容器沒有 Python 與編譯器 |
| 1 build 與安裝 | configure、build、install 都 exit 0 |
| 2 靜態檢查 | 12 項 PASS（含 `README.txt` 的 layout 與位元組） |
| 3 授權稽核 | `coverage`、`bundled_texts`、`official_terms`、`notice` 四項 PASS，`complete: true`；安裝的 `THIRD_PARTY.md` 等於 repository 的；狀態 17／6／3／1（§20.1），`distribution`＝`local-only` |
| 4 package 與簽章 | builder exit 0，`tree_clean`、`identity_current` 為 true；`sign` exit 0 |
| 4：`tarball --pubkey` | 8 項 PASS（88 個 member、57 個檔案）；`signature` 的 minisign exit 0、reader 無問題、trusted comment 相同 |
| 4：決定性（§20.4a） | 三個 package 檔案逐位元組相同：tarball `c753496a…`（2,384,778,504 位元組）、安裝器 `afa5d06a…`（與 PR-C3 相同）、digest `9f3012d6…` |
| 4：第二次簽章 | `dist_again` 的簽章以 minisign 與 reader 都驗證通過，trusted comment 相同；簽章檔本身不同（`bcf5ab8f…` 對 `6e1e2674…`，§20.4a） |
| 4：MANIFEST | `files` 等於 `$R/tree`；`licenses` 等於稽核檔的 sha256 與 `local-only` |
| 5 使用者端驗證 | 容器 exit 0：`Signature and comment signature verified`、trusted comment＝`package=<name> commit=bc75dcab… manifest_sha256=fcd200aa…`、tarball 與安裝器兩行 `OK` |
| 6 從 tarball 安裝 | exit 0；`install-trace` 5 項 PASS：TARGET 只被一次 `renameat2(…, RENAME_NOREPLACE)` 碰到，其他 277 筆寫入性質的呼叫都在 staging 之內，staging 已刪，0 筆 incomplete |
| 6：安裝後的 tree | `static --manifest` 13 項 PASS；`--verify` exit 0；除 MANIFEST 外與 `$R/tree` 逐位元組相同；**安裝後 `MANIFEST.json` 的 sha256（`fcd200aa…`）等於簽過的 `manifest_sha256`** |
| 7 乾淨容器執行 | exit 0；`EXACT`：`detector` 5316/5316、`mot_txt` 7/7、`graph_captures` 7/7；`--against` PR-C3 正式 run 的 `parity_bundle` 7/7 相同 |
| 8 G2-2／G2-4 | `runtime` 三項 PASS；`EXACT`，與第 7 條 7/7 相同；兩次的 `python_libraries_mapped` 都是空的 |
| **verdict** | **`PASS`**（§20.4 經 §20.4a 修訂後的第 1–8 條全部成立） |

**簽章負控制**（`sigctl/`）：

| # | 結果 |
|:--|:--|
| S1 | verify exit 1，`Signature verification failed`；`signature` FAIL（minisign exit 1、reader `the signature does not verify`） |
| S2 | verify exit 1；`signature` FAIL（簽章與 trusted comment 都不對）。§19.6 的 P3（內部一致、digest 重算的重新包）被簽章擋下 |
| S3 | verify exit 1；`signature`、`installer_exact`、`metadata` FAIL |
| S4 | verify exit 1，`Comment signature verification failed`；`signature` FAIL（reader `the trusted comment's signature does not verify`） |
| S5 | verify exit 1（key id 不同）；`signature` FAIL（reader：key id 不同、簽章與 comment 簽章都不成立） |
| S6 | verify exit 1：先 `Signature and comment signature verified`，之後 tarball 那一行 `FAILED`。host 的 `tarball` exit 1，`package_digest`、`manifest_exact`、`pinned_tree`、`metadata` FAIL（被改的 gzip 讀得完，沒有到 exit 2） |
| S7 | verify exit 2（沒有簽章檔）；`release_set`、`signature` FAIL |
| S8 | **verify exit 0、minisign exit 0**（簽章本身有效，trusted comment 是 64 個 0 的 `manifest_sha256`）；`signature` FAIL（trusted comment 不同） |
| S9 | `install.sh --verify` **exit 0**（§19.6 的限制照舊）；`sha256sum MANIFEST.json` 不等於簽過的 `manifest_sha256`；`static --manifest` 的 `vendor_set_pinned` FAIL |

**授權稽核負控制**（`auditctl/`）：
- **L1**：exit 1。`bundled_texts` FAIL，原因兩個：cuFile 的檔案 sha256 不是集合的；這份文本 Attachment A 列出 `libcufile.so`，與稽核的「沒列出」不同。
- **L2**：exit 1。`official_terms` FAIL：`nvshmem_sla_3.4.5` 的 snapshot sha256 不同。
- **L3**：exit 1。`bundled_texts` FAIL（`does not list 'libnvJitLink.so'`）；`notice` 也 FAIL，因為改過的稽核 render 出不同的 `THIRD_PARTY.md`。

**這個 PR 對 §19.6 限制的處理**：
- 「digest 不是簽章」：S2 被簽章擋下。
- 「安裝器驗不了自己」：S3 被簽章擋下。
- 「`--verify` 信任 tree 裡的 MANIFEST」：沒有消失（S9），但安裝後的 MANIFEST 現在可以對照簽過的 sha256（第 6 條、S9）。要做這個對照，使用者要有 release 公鑰，而它還沒有 commit。

結果目錄：`results/465_prc4_release/full_bc75dcab/`（r2）與 `full_bfd086c7/`（r1），內容包括：
- `run.sh`、`evaluate.py`、`evaluation.json`；
- `tree/`、`dist/`、`dist_again/`、`testkey*/`、`license_sources/`、`license_audit.json`；
- `verify/`、`install/`、`installed/`；
- `container_*`、`parity_*`；
- `sigctl/`、`auditctl/`。

開發試做在 `results/465_prc4_dev/t1/`，授權 snapshot 在 `results/465_prc4_license/sources_20261008/`。都不納入版本控制。

### 20.6 限制

- **不是可以散佈的 release。**
  - `distribution`＝`local-only`；§20.1 的四類 Open items 要 owner 決定。
  - 稽核是對授權文本的讀法，不是法律意見。
  - snapshot 是 2026-10-08 的網頁，NVIDIA 之後的修改不在裡面。
- **沒有 release key。** 正式 run 與負控制只用一次性的 test key（無密碼，secret key 在結果目錄）。release 公鑰 `shipping/package/minisign.pub` 由 owner 產生並以另一支 PR 加入。README 已經指向這個路徑，在那之前它不存在。
- **簽章不是決定性的**（minisign 0.12／libsodium 1.0.22，§20.4a）。同一份 digest 重簽會得到不同的 `.minisig`，發佈的是簽過的那一份。
- **trusted comment 的比對要由使用者做**（S8）。minisign 只證明簽章有效，不比對內容。檢查器會比對，使用者流程則要看 minisign 印出的 trusted comment。
- **`--verify` 仍信任 tree 裡的 MANIFEST**（S9）。要以簽過的 `manifest_sha256` 對照，README 有寫，但不是自動的。
- **使用者要有 minisign。** Ubuntu 24.04 的 universe 有它（0.11），base system 沒有。安裝器本身仍只用 base system 工具。
- **README 只描述這一台機器驗證過的事。** driver 下限引用 NVIDIA 對 CUDA 13.0 Update 2 的表，沒有在其他 driver 上執行過。
- 其餘同 §19.6（只支援 sm_120、一台 WSL2 機器、`RENAME_NOREPLACE` 只在 ext4 上確認、SIGKILL 留下 staging 等）。

### 20.7 重現

```bash
.venv/bin/python scripts/native/license_audit.py render --check             # THIRD_PARTY.md == rendering
.venv/bin/python scripts/native/license_audit.py check --licenses <tree>/licenses --sources <snapshots> --report audit.json
.venv/bin/python scripts/native/build_shipping_package.py --tree <tree> --out <dist> --static-report static.json
.venv/bin/python scripts/native/sign_shipping_package.py sign --dist <dist> --secret-key <key>   # the owner's release key
.venv/bin/python scripts/native/check_shipping_package.py tarball --dist <dist> --pubkey <pub> --report package.json
bash scripts/native/run_package_container.sh verify <dist> <pub> <out>      # the user's check, Ubuntu 24.04
minisign -Vm <dist>/<name>.sha256 -p <pub> && (cd <dist> && sha256sum -c <name>.sha256)   # an end user, before installing
bash results/465_prc4_release/<label>/run.sh && .venv/bin/python results/465_prc4_release/<label>/evaluate.py
```

### 20.8 Review 修正（`59de3f28` 的 review，2026-10-08）

review 在 `59de3f28` 重現了三個稽核的假 PASS，以及一個舊 MANIFEST 推導的退化，四項都是 P2。四項都是檢查器的覆蓋缺口，不是 package 或 r2 結果的錯誤。修正在 `96bb41c0`，只改檢查器與測試（`scripts/native/*` 不是 runtime identity 的輸入）。
- 修正後合併 head 的 `--mode attested` 仍然 exit 0；
- `shipping/license_audit.json`、`THIRD_PARTY.md`、README、安裝規則都沒有改；
- 所以 r2 的 package 與正式 run 仍然描述這個 source。

| # | 缺口 | 修正 |
|:--|:--|:--|
| 1 | release notes 的版本比對是整頁子字串：`nvidia_cuda_runtime-13.0.88` 會借 NVRTC／nvJitLink 的 13.0.88 而通過（cudart 那一列是 13.0.96） | 只看該物件在 component 表的那一列（`RELNOTES_COMPONENT`，CUPTI 的列名是 `CUPTI`）；那一列的版本要恰好等於 wheel 版本，不認得的 wheel 失敗 |
| 2 | 官方宣稱只檢查文字，不檢查它的來源是不是這個 release 的：cuFile 改用只供參考的目前版 CUDA EULA（13.4）仍然通過 | 每一個准許散佈的官方宣稱都要綁到已驗證的 release：同一個來源、同版本的 archive，或 switcher／tag 對得上。沒有版本的來源只在「它就是物件自己的 release 來源而且風險已記錄」時接受 |
| 3 | `official` 清單被清空時，`grant_in_official_only` 照樣四項 PASS | 宣稱官方准許的狀態至少要有一個准許散佈的官方宣稱；出貨文本准許的物件不能標成 `grant_in_official_only` |
| 4 | source commit 早於 PR-C4 時，`manifest_head` 不寫 `licenses`，卻用新的 `reading`，所以 PR-C3 的 MANIFEST（`9217ed92`）推導不出來，`metadata` 失敗 | 沒有稽核檔的 commit 用 PR-C3 的 `reading`（`READING_PRE_C4`） |

**重播**（`results/465_prc4_release/review_fix_96bb41c0/replay.sh`，在 r2 的產物上執行，不產生新的 package、不用 GPU）。

- gate 3：r2 的 tree 與 snapshot 在修正後的檢查器下仍是四項 PASS，`unmatched_releases` 兩筆：cuSPARSELt、libgomp，都有記錄風險。
- L1–L3 照舊各自失敗：L1 `bundled_texts`，L2 `official_terms`，L3 `bundled_texts`＋`notice`。
- review 的三個情境在真實稽核上重做，各自重新 render notice 與集合，所以只有宣稱不同：
  - **R1**（cudart 宣稱為 13.0.88，授權目錄也跟著改名）：`official_terms` FAIL。
    - release 列：`row 'CUDA Runtime (cudart)' gives ['13.0.96'], the wheel is 13.0.88`；
    - 它的 EULA 也失去綁定。
  - **R2**（cuFile 改用 `cuda_eula_current`）：`official_terms` FAIL：`terms cuda_eula_current is not tied to version 1.15.1.6`。
  - **R3**（nvJitLink 的 `official` 清空）：`coverage` FAIL：`status grant_in_official_only without an official claim that grants distribution`。
- **R4**：
  - 以修正後的 `manifest_head` 推導 PR-C3 的 MANIFEST（`9217ed92`）與 r2 的（`bc75dcab`），兩者都逐欄相同。
  - r2 的 `tarball --pubkey` 仍是 8 項 PASS。
  - PR-C3 package 的 `tarball` 現在 `metadata` PASS。它的 `pinned_tree` 仍然 FAIL：目前 repository 的 layout 多了 `README.txt`，`THIRD_PARTY.md` 也不同了，這是預期的，`tarball` 一向以目前的 pin 檢查。

測試：`tests/unit/test_license_audit.py` 加了 component 列（含 13.0.88 的借用）、terms 與 release 的綁定（同 archive／不同 archive／沒有版本），以及狀態與證據的一致性；`tests/unit/test_package_signature.py` 釘住 PR-C3 的 `reading`。原本一個合成測試改為較嚴的行為：沒有版本、也沒有記錄風險的 release，現在是失敗，不再只是 note。

## 21. Release policy：local package 的簽章改為可選（#546，#465 closeout）

#546 是 #465 的收尾：把 native runtime 的工程結果整理成可稽核的驗收紀錄（[closeout 文件](native_runtime_closeout.md)），並簡化 release policy。本節只處理 release policy：**SHA-256 digest 與 MANIFEST 的完整性檢查維持強制；minisign 的發行者認證對 local-only package 改為可選。** 本節不改任何 stage 的計算、entrypoint（pin `92f74ef4…`）、operator library（`aa84cccd…`）、27 個第三方物件、launcher、auditor、安裝器（`shipping/package/install.sh` 與 PR-C3 逐位元組相同）、SM 清單、glibc baseline、授權稽核與散佈狀態（`local-only`）。

| 項目 | 位置 |
|:--|:--|
| package 檢查的 `authentication` 欄位、未驗證簽章的拒絕 | `scripts/native/check_shipping_package.py`（`tarball`） |
| 使用者說明 | `shipping/package/README.txt`（「Verify, then install」） |
| 測試 | `tests/unit/test_package_signature.py`、`tests/unit/test_shipping_package.py` |

### 21.1 設計決定

- **兩種檢查，兩種意義。**
  - 完整性（強制）：`<name>.sha256` 的 sha256、MANIFEST 的檔案集合／sha256／大小／mode、staging 後一次 `RENAME_NOREPLACE` 的 atomic 安裝、任何失敗都不建立 TARGET、`tarball` 檢查的 pin 與 runtime identity（`metadata`）。它們證明檔案是 digest 指名的那一份、完整且沒有被改，**不證明是誰做的**：能換掉 tarball 的人也能換掉 digest（§19.6）。
  - 發行者認證（可選）：`<name>.sha256.minisig`，以 `minisign -V` 對 repository 的公鑰驗證（§20.1）。
- **安裝器不變。** 它本來就只做完整性檢查、不讀簽章、不需要 minisign（§20.1 的 owner 指示）；`install.sh` 是 runtime identity 的輸入，不改它也就不需要 republish。所以「簽章可選」在安裝路徑上不是新行為，而是把既有行為寫成 policy。
- **release set**：三個檔案（tarball、安裝器、digest）；簽過的 release 多一個 `.minisig`。local-only package 不需要簽，engineering closeout 不需要 production release key。
- **簽章的路徑仍是明確的 opt-in**，而且 fail-closed：
  - `tarball --pubkey`：要求第四個檔案，`signature` 檢查（`minisign -V`＋reader＋trusted comment）任何一項不成立就 FAIL（§20.5 S1–S8 不變）。
  - `tarball` 不帶 `--pubkey`、dist 裡卻有 `.minisig`：`release_set` FAIL，訊息說明要以 `--pubkey` 驗證。簽章要嘛被驗證、要嘛檢查失敗，**不會被略過**。
  - 使用者端：README 的第 1 步（簽過的 release）失敗就停止，不安裝。
- **報告不把未簽的 package 說成已認證。** `tarball` 報告多一個 `authentication`：
  - 不帶 `--pubkey`：`method: none`、`publisher_authenticated: false`、`reading`＝「integrity only; the publisher is not authenticated」；
  - 帶 `--pubkey`：`method: minisign`，`publisher_authenticated` 只在 `signature` PASS 時為 true；`reading` 說明 key 是否屬於發行者取決於公鑰的來源。
  
  `signed` 欄位保留（＝是否要求簽章），舊的評估器照常讀得到。
- **README**：release 是三個檔案，簽過的多一個；列出 digest／MANIFEST 與簽章各自證明什麼；明寫「digest 不是簽章」「未簽的 package 不是 authenticated，不要說它是 signed／verified」；minisign 那一步標成「Signed release only」。
- **MANIFEST 的 `reading` 不改。** PR-C4 的 `reading` 提到 `.minisig`；`manifest_head` 以 source commit 推出 `reading`，改它就要再一個依 commit 選擇的版本。它描述的是簽章檔「若存在」時涵蓋什麼，不是認證宣稱；未簽的 package 有沒有被認證，以 `tarball` 報告的 `authentication` 與 README 為準（§21.5 記為限制）。

### 21.2 改了什麼

- `check_shipping_package.py`：`release_set` 對「有 `.minisig` 但沒有 `--pubkey`」給出明確的問題；報告多 `authentication`（`authentication()`）；最後一行印出 `authentication: <method> (<reading>)`；docstring 與 `--pubkey` 的 help。
- `sign_shipping_package.py`：只改 docstring。
- `shipping/package/README.txt`：「Verify, then install」改寫（上述）。README 不是 runtime identity 的輸入（prose，`build_runtime_identity._is_prose`）；`check_runtime_identity_staleness.py --mode attested` 在本節的變更後仍 exit 0。
- 測試：
  - `test_package_signature.py`：未簽的 release 跑完全部七項完整性檢查並報告 `none`；未簽時 digest 被改仍然 FAIL；有 `.minisig` 但沒有 `--pubkey` ⇒ `release_set` FAIL、沒有 `signature` 項；錯的 key 與被改的 trusted comment ⇒ `signature` FAIL 且 `publisher_authenticated: false`；`authentication()` 的真值表。
  - `test_shipping_package.py`：安裝器不提 `minisig`；未簽的 package 與旁邊放一個壞 `.minisig` 的 package 安裝輸出相同（除 staging 名稱），都不提 sign／authentic。
- **沒有動的**：`install.sh`、launcher、auditor、shipping 與 tracking 的 C++ 原始碼、`third_party_set.json`、`entrypoint_pin.json`、`license_audit.json`、`THIRD_PARTY.md`、model root、operator library、runtime identity 出版。

### 21.3 驗證契約（量測之前寫定）

不跑 GPU：runtime 的位元組不變（下面 V3、E1 檢查），7-seq parity 沿用 PR-C4 r2（`results/465_prc4_release/full_bc75dcab/`，gate 7–8 `EXACT`）。結果目錄 `results/546_closeout/full_<commit>/`。

**有效性**（任一不成立 ⇒ 受影響的項目 `UNRESOLVED`）：
- V1：乾淨 commit；`check_runtime_identity_staleness.py --mode attested` exit 0。
- V2：operator library 與 pin 的 entrypoint 的 sha256 等於 attestation 與 `entrypoint_pin.json`。
- V3：`git diff 24dab817 HEAD -- shipping` 只改 `shipping/package/README.txt`；`git diff 9217ed92 HEAD -- shipping/package/install.sh` 為空。

**PASS 規則**（全部成立 ⇒ `PASS`）：

1. **E1 tree**：`cmake --install build-release --component shipping`（不 configure、不 build）到 `$R/tree`；`static` 12 項 PASS；每個檔案的 sha256 與 r2 的 `gate1_tree.sha256` 相同，唯一的差別是 `README.txt`。
2. **E2 未簽的 package**：builder（不帶 `--trial`）exit 0；`tarball`（不帶 `--pubkey`）exit 0，七項 PASS，`signed: false`，`authentication.method`＝`none`、`publisher_authenticated`＝false；安裝器 sha256＝`afa5d06a…`（PR-C3）；MANIFEST 的 `files` 等於 `$R/tree`。
3. **E3 未簽的 package 安裝到乾淨容器**：`run_package_container.sh install-strace` exit 0；`install-trace` 5 項 PASS；`static --manifest` 13 項 PASS；容器內 `--verify` exit 0；除 MANIFEST 外與 `$R/tree` 逐位元組相同。
4. **E4 簽章 opt-in**：容器 minisign 產生一次性 test key（`-G -W`），`sign` exit 0；`tarball --pubkey` exit 0，八項 PASS，`authentication.method`＝`minisign`、`publisher_authenticated`＝true；`run_package_container.sh verify` exit 0。

**負控制**（各自的 dist 副本，未改的檔案是 hard link）：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| U1 | 未簽的 dist：digest 的 tarball 那一行第一個 hex 字元改掉 | `tarball`（不帶 `--pubkey`）`package_digest` FAIL；容器安裝器 exit 1（`sha256 is not the one in`），TARGET 沒有建立、沒有 staging |
| U2 | 簽過的 dist，`tarball` 不帶 `--pubkey` | `release_set` FAIL（`present but no --pubkey`）；沒有 `signature` 項；`authentication.method`＝`none` |
| U3 | 簽過的 dist：digest 第一個 hex 字元改掉，簽章不變，帶 `--pubkey` | `signature` FAIL，`publisher_authenticated`＝false；容器 `verify` exit 非 0 |
| U4 | 以第二把 test key 簽，以第一把驗 | `signature` FAIL（key id），`publisher_authenticated`＝false；容器 `verify` exit 非 0 |
| U5 | 未簽的 package 安裝到已存在的 TARGET（空目錄） | 容器安裝器 exit 2（`exists; nothing was changed`），上層目錄不變 |

**沿用、不重跑的證據**：安裝器的 P1–P15（PR-C3 §19.5）與簽章的 S1–S9（PR-C4 §20.5）：安裝器與簽章檢查的邏輯沒有變（安裝器逐位元組相同；`signature` 的檢查函式沒有改）。stale identity 的拒絕：builder 不帶 `--trial` 時拒絕、`tarball` 的 `metadata` 對 `identity_current: false` 失敗（§19.1，單元測試）。

**不做的**：GPU、parity、FPS；release key；任何散佈。

### 21.4 驗收

同一台機器。commit `c7581100`＝§21.1–§21.3（契約早於任何量測），工作樹乾淨，沒有 GPU 步驟。判定由 `evaluate.py` 從產物讀出（`evaluation.json` sha256 `2125347e…`）。host minisign 0.12，容器 minisign 0.11；test key `A26198D9D83803A0`，U4 的第二把 `4152F05B395B1E83`。

| 項目 | 結果 |
|:--|:--|
| V1 | 工作樹乾淨；`--mode attested` exit 0 |
| V2 | operator library `aa84cccd…`＝attestation；entrypoint `92f74ef4…`＝pin |
| V3 | `git diff 24dab817 HEAD -- shipping` 只有 `shipping/package/README.txt`；`install.sh` 對 `9217ed92` 沒有 diff |
| E1 | `cmake --install` exit 0；`static` 12 項 PASS；57 個檔案中只有 `README.txt` 的 sha256 與 r2 的 tree 不同 |
| E2 | builder exit 0；`tarball` 七項 PASS，`signed: false`，`authentication`＝`none`／`publisher_authenticated: false`；安裝器 `afa5d06a…`（＝PR-C3）；MANIFEST 的 `files` 等於 tree（57 個），`licenses.distribution`＝`local-only` |
| E3 | 乾淨容器（Ubuntu 24.04.4、glibc 2.39、dash，沒有 Python 與編譯器）安裝 exit 0；`install-trace` 5 項 PASS；`static --manifest` 13 項 PASS；容器內 `--verify` exit 0；除 MANIFEST 外與 tree 逐位元組相同 |
| E4 | `sign` exit 0；`tarball --pubkey` 八項 PASS，`authentication`＝`minisign`／`publisher_authenticated: true`；容器 `verify` exit 0（`Signature and comment signature verified`、兩行 `OK`） |
| **verdict** | **`PASS`** |

**負控制**（`negctl/`）：

| # | 結果 |
|:--|:--|
| U1 | `tarball` 只有 `package_digest` FAIL，`authentication`＝`none`；容器安裝器 exit 1（`sha256 is not the one in`），沒有 `extracting`；上層目錄是空的（沒有 TARGET、沒有 staging） |
| U2 | `release_set` FAIL：`… .minisig is present but no --pubkey was given: a signature is verified or the check fails, never ignored`；沒有 `signature` 項；`authentication`＝`none` |
| U3 | `signature` FAIL（minisign exit 1 `Signature verification failed`、reader `the signature does not verify`），`package_digest` 也 FAIL；`publisher_authenticated: false`；容器 `verify` exit 1 |
| U4 | `signature` FAIL（minisign 與 reader 都是 key id `4152F05B395B1E83` ≠ `A26198D9D83803A0`）；`publisher_authenticated: false`；容器 `verify` exit 1 |
| U5 | 容器安裝器 exit 2（`/install/saccade exists; nothing was changed`）；上層目錄（TARGET 與 sentinel）不變 |

測試與 CI：`pre_push.sh` 在本 PR 的 head 上通過（lint、format、mypy、pytest）。

結果目錄：`results/546_closeout/full_c7581100/`（`run.sh`、`evaluate.py`、`evaluation.json`、`tree/`、`dist/`、`dist_signed/`、`testkey*/`、`install/`、`installed/`、`e4_verify/`、`negctl/`、各 log），不納入版本控制。

### 21.5 限制

- **未簽的 package 不認證發行者**：這是 policy 本身，不是缺陷。integrity 檢查擋得住損壞與不一致；擋不住一份重新包過、digest 也重算過的 package（§19.6 P3）。local-only package 只應在信任來源的情況下使用。
- **MANIFEST 的 `reading` 沒有跟著改**（§21.1）：它仍以 PR-C4 的句子提到 `.minisig`。是否認證以 `tarball` 報告與 README 為準；要改句子，需要一個依 source commit 選擇的新版本（closeout §7）。
- **`install.sh` 標頭的註解「signing is PR-C4」過時**：改它會動 runtime identity 的輸入（`shipping/package/install.sh`），需要 republish；本節不改。
- **只有 test key**：release 公鑰不存在（§20.6）；公開散佈時簽章是強制的（closeout §1 D2）。
- 其餘同 §19.6、§20.6。

### 21.6 重現

```bash
.venv/bin/python scripts/native/check_shipping_package.py tarball --dist <dist> --report package.json            # unsigned: authentication none
.venv/bin/python scripts/native/check_shipping_package.py tarball --dist <dist> --pubkey <pub> --report package.json  # signed, opt in
bash results/546_closeout/<label>/run.sh && .venv/bin/python results/546_closeout/<label>/evaluate.py
```

---

## 22. 公開散佈的授權證據：L-1～L-3 補正、L-4 稽核（#547）

#547 追蹤 `ENGINEERING_COMPLETE` 與 `PUBLIC_DISTRIBUTION_READY` 之間的 blocker（[closeout](native_runtime_closeout.md) §6）。本節照 owner 的兩次決定（2026-10-08）補齊 L-1～L-3 的文本與來源證據，並記錄 L-4 的稽核。**本節不宣告合規，不填 `owner_confirmation`，`distribution.status` 維持 `local-only`，所有 open item 維持 OPEN。** 不改模型、tracker、runtime 行為與任何 benchmark 數字：entrypoint pin（`92f74ef4…`）、operator library（`aa84cccd…`）、27 個第三方物件、launcher、auditor、安裝器、SM 清單與 glibc baseline 都不變。

| 項目 | 位置 |
|:--|:--|
| 稽核（schema v2：`supplied_texts`、`open_items`、`corresponding_source`、`downstream_terms`） | `shipping/license_audit.json` |
| package 自己附的授權文本 | `shipping/licenses/terms/`（4 份 NVIDIA 官方條款）、`shipping/licenses/libgomp/`（`COPYING3`、`COPYING.RUNTIME`、`SOURCE.txt`） |
| 下游條款草稿（不在 package 裡、不生效） | `shipping/DOWNSTREAM_TERMS.draft.md`（由稽核 render） |
| 檢查、render、HTML 條款抽取、RPM／ELF 比對 | `scripts/native/license_audit.py` |
| 安裝 | `shipping/cmake/install_third_party.cmake`（依 `supplied_texts` 安裝並比對 sha256） |
| tree／package 檢查 | `scripts/native/check_shipping_bundle.py`（`expected_files`、`licenses`）、`check_shipping_package.py`（`pinned_tree`） |
| L-4 稽核紀錄 | `results/547_l4_audit/20261008T125352Z/audit.md`（不納入版本控制）；摘要在 `open_items` 的 L-4 |
| 測試 | `tests/unit/test_license_audit.py`、`tests/unit/test_shipping_bundle_checks.py` |

### 22.1 設計決定

- **三層分開。** 每個 open item（L-1、L-2、L-3、L-4，以及新的 M-1）都有三個各自非空的清單：`technical_evidence`（量到或讀到的）、`licence_interpretation`（條款文字的讀法）、`legal_uncertainty`（只有法律結論能定的）。`check_coverage` 要求三層都在；只要有一項是 OPEN，`distribution.status` 就必須是 `local-only`、`owner_confirmation` 必須是 null。工具不會關閉任何一項；只有 owner 能關。
- **L-1（nvJitLink、cuFile、nvshmem）**：保留 wheel 的 `License.txt`，另附版本對應的官方條款全文（`licenses/terms/cuda_eula_13.0.2.txt`、`nvshmem_sla_3.4.5.txt`），每份記錄來源 snapshot 與 sha256。**附上全文不代表那份文本優先適用**；兩份文本的差異與適用性的疑問寫在 L-1，狀態仍是 `grant_in_official_only`。
- **L-2（cuDNN×5、TensorRT）**：兩套條款都附（wheel 與 `licenses/terms/` 的官方版）。TensorRT wheel §12.2 的期間（一年、自動續約、NVIDIA 可在續約年開始前 90 天書面通知終止）加進 `tensorrt_wheel` conditions 並逐字檢查。新發現的疑問：wheel SLA §2 (iii)(d) 禁止「by means of the internet」提供，除非 AGREEMENT 明示授權，記入 `libnvinfer` 的 risk 與 L-2。
  - **下游條款**整理成 `downstream_terms`（D-1～D-6），與 L-4 的共通條件合成同一份草稿。
  - 每一條都標出適用的物件（以物件或 conditions key 表示）。每一句引文另有自己的適用範圍，並在**每一個被引用物件自己 wheel 的文本**裡逐字檢查。例如 D-2 原本引用的「stand-alone product」在 cuDNN wheel 的文本裡找不到，於是改用每個 NVIDIA wheel 都有的「as incorporated in object code format into a software application」。
  - 條款不得涵蓋 Saccade（Apache-2.0）、torch／NCCL（BSD-3）與 libgomp（GPL-3.0）：GPL 禁止附加限制，checker 會擋。
  - 草稿不在 package 裡、不生效；README 明寫 package 裡的檔案是 notice、不是協議：**README 的聲明不能視為已完成「與 Customers 的協議」義務**（L-2 的 legal uncertainty）。
- **L-3（libgomp）**：
  - 附上 `COPYING3` 與 `COPYING.RUNTIME`，從物件的 source package 取出，逐位元組相同。`COPYING3` 與 gnu.org 現行的 `gpl-3.0.txt` 只差 4 個 http→https 的網址。
  - `SOURCE.txt` 寫明 source package 的名稱、sha256、大小與取得位置。
  - **對應關係以比對建立，不以 debuglink 或字串宣告**：
    1. shipped `libgomp.so.1`（`e28fb289…`）與 AlmaLinux `libgomp-8.5.0-28.el8_10.alma.1.x86_64.rpm` 的 `/usr/lib64/libgomp.so.1.0.0`（`e985bcbb…`）有相同的 build ID，24 個 code／data section 逐位元組相同且位址不變。
    2. 只有 `.dynamic`、`.dynstr`、`.dynsym` 不同：多一筆 DT_RPATH（torch wheel 的 build 加的），`.dynstr` 搬移並加長；`.dynsym` 的名稱、值、大小、binding 都相同，只有 section index 重編。
    3. 這個 binary package 的 RPM header `SOURCERPM` 是 `gcc-8.5.0-28.el8_10.alma.1.src.rpm`（sha256 `c94dbbd2…`，65715094 bytes）。
    4. 兩個 package 的 sha256 都在 AlmaLinux 的 repository metadata 裡（primary.xml，其 checksum 在 repomd.xml），repomd.xml 的 detached signature 以 key `BC5EDDCA…CED7258B` 驗證為 Good。key 本身取自 repo.almalinux.org，沒有經獨立管道核對指紋。
    5. **沒有從 source 重建出相同的位元組。** 這是比對證據，不是可重現 build 的證明。
  - `license_audit.py check --gomp-rpm --srpm` 會重做比對：任何 section 不同、RPATH 以外的 `.dynamic` 差異、或 COPYING 文本不是 source package 裡的那一份，都會 FAIL。
  - **source 提供機制**：
    - 公開 release 時，把 source package 以同名、同 sha256 放在 package 旁邊，同一個地方發布；AlmaLinux 的網址是第二來源，不是 offer（`corresponding_source.mirror`）。
    - 目前沒有公開 release，所以狀態是 `not published`；位置取決於 owner 的發布決定。
  - 未確認的部分維持 OPEN：被第三方工具改寫過的物件，它的 Corresponding Source 是否需要包含那個改寫步驟；RLE 的 Eligible Compilation Process；source 要提供多久。
- **L-4**：依 owner 的指示逐項稽核三個條件，`--library-path` 不算證明。結論：
  - material additional functionality：事實支持，OPEN；
  - only accessed by your application：inward 充分，outward 沒有技術上的排他，OPEN；
  - 不受 open source 授權約束：Apache-2.0 與 libgomp 不構成，YOLO26 的 AGPL 血統可能構成，OPEN。
- **M-1（新）**：model root 的授權（YOLO26s engine 的 Ultralytics AGPL-3.0 血統，ADR 023；訓練資料的條款未取得）不在 `lib/vendor/` 的稽核範圍內，原本完全沒有紀錄；現在以 open item 記錄。模型與 runtime 的分離由 #549 處理，**不改變任何授權狀態**。
- **官方條款文本的抽取是確定性的**（`terms_text`）：
  - 只取文件頁的 article body（`itemprop="articleBody"` 或 `<article class="bd-article">`）；
  - 去掉導覽、script、heading 的 permalink；
  - 每個 block 一行，`td`／`th` 也算 block。
  
  checker 從 snapshot 重新抽取並逐位元組比對 shipped 文本。
- **libgomp 的狀態**由 `no_licence_text_shipped` 改為新的 `grant_in_supplied_text`：wheel 的文本沒有涵蓋它，由 package 自己附上授權文本。它的 conditions（`gpl3_rle`）在 package 附的文本裡逐字檢查，不在 torch wheel 的文本裡檢查。
- **runtime identity**：`license_audit.json`、`install_third_party.cmake` 與新的 `COPYING3`／`COPYING.RUNTIME`（不是 `.txt`，不算 prose）屬於 implementation 軸，所以要依 [runbook](runbooks/runtime_identity_republication.md) republish（stacked PR）。`*.txt`、`*.md` 是 prose，不在軸上。

> 更正（#547）：§20.1 表中 libgomp 那一列的「物件也沒有記錄 GCC 版本」不成立。物件有 `.gnu_debuglink`（`libgomp.so.1.0.0-8.5.0-28.el8_10.alma.1.x86_64.debug`）與 annobin 註記（`running gcc 8.5.0 20210514`）；PR-C4 只查了 `.comment` section。版本與對應關係的證據見上。

### 22.2 改了什麼

- `shipping/license_audit.json` 改為 schema v2：
  - L-1／L-2 的物件多了 `supplied` claim；libgomp 改為 `grant_in_supplied_text` 與 `gpl3_rle`；
  - TensorRT 多了 conditions 與 risk；
  - 新增 `supplied_texts`（7 份）、`corresponding_source`、`open_items`（5 項）、`downstream_terms`（6 條）。
- `scripts/native/license_audit.py`：
  - `terms_text`（HTML 條款抽取）；
  - RPM header／cpio、tar.xz、ELF section／dynamic 的讀取；
  - `check_supplied_records`、`check_open_items`、`check_downstream_records`、`check_corresponding_records`（都在 `coverage` 裡）；
  - 新的檢查項 `supplied_texts`、`downstream_basis`、`corresponding_source`；`official_terms` 多了重新抽取的比對；
  - `render` 也產生 `DOWNSTREAM_TERMS.draft.md`。
- `shipping/cmake/install_third_party.cmake`：依 `supplied_texts` 安裝到 `licenses/terms/`、`licenses/libgomp/`，sha256 不符就中止。
- `check_shipping_bundle.py`：`expected_files` 與 `licenses` 涵蓋 supplied texts。`check_shipping_package.py`：`pinned_tree` 比對它們的 sha256。
- `shipping/package/README.txt`：Licenses 一節列出各目錄，並寫明 package 裡的檔案是 notice、不是協議。`THIRD_PARTY.md`：open items 以三層呈現，另有 supplied texts、corresponding source 與 downstream terms 的說明。
- **沒有動的**：`install.sh`、launcher、auditor、shipping 與 tracking 的 C++ 原始碼、`third_party_set.json`、`entrypoint_pin.json`、model root、operator library、任何 benchmark 數字。

### 22.3 驗證契約（量測之前寫定）

不跑 GPU 的 parity：runtime 的位元組不變（V3、E1 檢查），7-seq parity 沿用 PR-C4 r2（gate 7–8 `EXACT`）。結果目錄 `results/547_licence/full_<commit>/`，在 republish 之後的乾淨 head 上跑。

**有效性**（任一不成立 ⇒ 受影響的項目 `UNRESOLVED`）：
- V1：乾淨 commit；`check_runtime_identity_staleness.py --mode attested` exit 0（republish 之後）。
- V2：operator library 與 pin 的 entrypoint 的 sha256 等於 attestation 與 `entrypoint_pin.json`。
- V3：`git diff 54be85ad HEAD -- shipping` 只動 `license_audit.json`、`THIRD_PARTY.md`、`DOWNSTREAM_TERMS.draft.md`、`licenses/**`、`cmake/install_third_party.cmake`、`package/README.txt`；`install.sh` 對 `9217ed92` 沒有 diff。

**PASS 規則**（全部成立 ⇒ `PASS`）：

1. **E1 tree**：
   - `cmake --install build-release --component shipping`（不 configure、不 build）到 `$R/tree`；`static` 全部 PASS。
   - 與 #546 的 tree（`results/546_closeout/full_c7581100/e1_tree.sha256`，57 個檔案）相比：只有 `README.txt` 與 `licenses/THIRD_PARTY.md` 的 sha256 不同，多出的恰好是 7 份 supplied texts（共 64 個檔案）。
2. **E2 稽核**：`license_audit.py check --licenses $R/tree/licenses --sources results/465_prc4_license/sources_20261008 --gomp-rpm … --srpm …` exit 0。`coverage`、`bundled_texts`、`supplied_texts`、`downstream_basis`、`official_terms`、`corresponding_source`、`notice` 七項 PASS，`complete: true`，`distribution`＝`local-only`。
3. **E3 未簽的 package**：
   - builder（不帶 `--trial`）exit 0；`tarball` 七項 PASS；`authentication`＝`none`。
   - MANIFEST 的 `files` 等於 `$R/tree`；`licenses.distribution`＝`local-only`；`licenses.sha256` 等於 `shipping/license_audit.json` 的 sha256。
4. **E4 安裝到乾淨容器**：`run_package_container.sh install-strace` exit 0；`install-trace` PASS；`static --manifest` PASS；容器內 `--verify` exit 0；除 MANIFEST 外與 `$R/tree` 逐位元組相同。

**負控制**：

| # | 操作 | 必須的結果 |
|:--|:--|:--|
| N1 | tree 的副本：`licenses/terms/cuda_eula_13.0.2.txt` 加一行 | `static` 的 `licenses` FAIL；`license_audit.py check --licenses <副本>` 的 `supplied_texts` FAIL |
| N2 | tree 的副本：刪掉 `licenses/libgomp/SOURCE.txt` | `static` 的 `layout_exact` FAIL |
| N3 | repo root 的最小副本（只有安裝需要的檔），其中 `shipping/licenses/libgomp/COPYING3` 被改，以 `cmake -P install_third_party.cmake` 安裝 | exit 非 0，訊息含 `sha256`，`licenses/libgomp/COPYING3` 沒有寫出 |
| N4 | `check --srpm` 改給 binary package | `corresponding_source` FAIL（source package 的 sha256／大小不符） |
| N5 | 稽核副本：`distribution.status` 改為 `public`（open items 仍為 OPEN） | `coverage` FAIL（`while items are OPEN`） |

**不做的**：GPU、parity、FPS；release key；source package 的公開 mirror；任何散佈；任何 open item 的關閉。

### 22.4 驗收

同一台機器，沒有 GPU 步驟。head `664208d1`＝§22.1–§22.3 的契約 commit `048034ec` 加上 republish（stacked，契約早於任何量測），工作樹乾淨。第一次執行在 E1 安裝途中中斷（主機磁碟滿，session 結束）；清掉該次的部分產物後，同一個 head 從頭重跑，以下是重跑的結果。判定由 `evaluate.py` 從產物讀出（`evaluation.json` sha256 `645940a1…`）。

| 項目 | 結果 |
|:--|:--|
| V1 | 工作樹乾淨；`--mode attested` exit 0（implementation `8e758b23…`） |
| V2 | operator library `aa84cccd…`＝attestation；entrypoint `92f74ef4…`＝pin |
| V3 | `git diff 54be85ad HEAD -- shipping` 只有契約列出的檔案（12 個）；`install.sh` 對 `9217ed92` 沒有 diff |
| E1 | `cmake --install` exit 0；`static` 12 項 PASS；64 個檔案。與 #546 的 tree 相比，只有 `README.txt`、`licenses/THIRD_PARTY.md` 的 sha256 不同，多出的恰好是 7 份 supplied texts |
| E2 | `license_audit.py check` exit 0：`coverage`、`bundled_texts`、`supplied_texts`、`downstream_basis`、`official_terms`、`corresponding_source`、`notice` 七項 PASS，`complete: true`，`distribution`＝`local-only` |
| E3 | builder exit 0；`tarball` 七項 PASS，`authentication`＝`none`／`publisher_authenticated: false`；安裝器 `afa5d06a…`（＝PR-C3）；tarball `3e7f0c5c…`；MANIFEST 的 `files` 等於 tree（64 個），`licenses.sha256`＝`1e884f44…`＝`shipping/license_audit.json`，`licenses.distribution`＝`local-only` |
| E4 | 乾淨容器安裝 exit 0；`install-trace` 5 項 PASS；`static --manifest` 13 項 PASS；容器內 `--verify` exit 0（`64 files match`）；除 MANIFEST 外與 tree 逐位元組相同 |
| **verdict** | **`PASS`** |

**負控制**（`negctl/`）：

| # | 結果 |
|:--|:--|
| N1 | `static` 的 `licenses` FAIL（`licenses/terms/cuda_eula_13.0.2.txt`）；稽核的 `supplied_texts` FAIL（`tree: … is not sha256 1762c8a0…`） |
| N2 | `layout_exact` FAIL（missing `licenses/libgomp/SOURCE.txt`） |
| N3 | `cmake -P install_third_party.cmake` exit 1（`CMake Error … COPYING3 sha256 … != …`）；`licenses/libgomp/COPYING3` 沒有寫出 |
| N4 | `corresponding_source` FAIL（`source package is not sha256 c94dbbd2… / 65715094 bytes`） |
| N5 | `coverage` FAIL（`distribution 'public' while items are OPEN`） |

**這個 PASS 只表示**：補上的文本與證據都在 package 裡，而且可以重新驗證。**它不表示任何授權項目已經滿足。** L-1～L-4 與 M-1 全部仍是 OPEN。

結果目錄：`results/547_licence/full_664208d1/`（`run.sh`、`evaluate.py`、`evaluation.json`、`tree/`、`dist/`、`install/`、`installed/`、`negctl/`、各 log）；republish 的捕捉在 `results/547_licence/republish/`；libgomp 的下載、repository metadata 與比對在 `results/547_l3_gomp/20261008/`；L-4 稽核在 `results/547_l4_audit/20261008T125352Z/`。以上都不納入版本控制。

### 22.5 限制

- **不是法律意見。** 三層紀錄中，`licence_interpretation` 只是條款文字的讀法；`legal_uncertainty` 是未解的問題，不是風險評估。
- **libgomp 的對應關係是比對出來的，不是重建出來的。** source package 沒有被重新 build。AlmaLinux 簽章 key 的指紋沒有經獨立管道核對。
- **source package 還沒有公開 mirror。** 規則已寫定（`corresponding_source.mirror`），位置取決於 owner 的發布決定。
- **官方條款的文本是 2026-10-08 的 snapshot。** `terms_text` 抽取的是 article body 的文字，不保留原頁面的版面（表格一格一行）。cuSPARSELt 的官方頁沒有版本（PR-C4 的 risk 沿用）。
- **下游條款只是草稿**，不在 package 裡、不生效；條文的措辭與是否「at least as restrictive」需要 owner 與法律意見。
- **M-1（模型）只做了紀錄**，沒有稽核：權重血統與訓練資料的條款都沒有結論。
- 其餘同 §20.6、§21.5。

### 22.6 重現

```bash
.venv/bin/python scripts/native/license_audit.py render --check
.venv/bin/python scripts/native/license_audit.py check --licenses <tree>/licenses \
    --sources results/465_prc4_license/sources_20261008 \
    --gomp-rpm <libgomp-8.5.0-28.el8_10.alma.1.x86_64.rpm> --srpm <gcc-8.5.0-28.el8_10.alma.1.src.rpm> --report audit.json
bash results/547_licence/<label>/run.sh && .venv/bin/python results/547_licence/<label>/evaluate.py
```

兩個 package 的取得位置與 sha256 見 `shipping/license_audit.json` 的 `corresponding_source`。
