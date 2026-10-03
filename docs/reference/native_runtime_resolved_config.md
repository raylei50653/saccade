# Native runtime resolved config（#465 Phase B PR-3／U2a）

> 狀態：PR-3 完成 exporter、schema 與 Python 端等價 contract test。PR-4a 加上 native strict loader（§7）。PR-4b 讓 native 物件只從單一參數狀態讀值、native 端不再讀 `SACCADE_*`、shipping 以這份 JSON 建立 GPU 物件並做 set 後回讀（§8）；**U2b 至此完成**。PR-3／PR-4a 不改 native code；PR-4b 改 native code 但 headline 輸出逐位元組不變（§8.6）。PR-5 加上 native post-detector replay 宿主（U3a，§9），並補齊 exporter `steps` 漏列的 3 個 post-detector 分支（§9.2）。PR-6 加上 native MOT 輸出（U4：per-sequence ID、行格式、sequence-tail interpolation，§10），接進 replay 宿主後 7-seq MOT txt 對 Python serial 組態逐位元組相同（多 sequence 以 ID 位移重標），並補上 emit 路徑的 6 個 gate（§10.2）。PR-7 加上 native ingest（U3b-1：nvJPEG 解碼＋normalize，§11），以與 oracle 同一份 nvJPEG binary、只由 resolved config 驅動；parity harness 把解碼、normalize、端到端 ingest 分開對 torchvision ingest 報告。
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
