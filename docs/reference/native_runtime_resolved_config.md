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
| 測試 | `tests/native/test_shipping_double_buffer_runtime.cpp`（GPU；需要 model 與 MOT17，缺少時 SKIP）、`tests/unit/test_native_double_buffer_oracle_pins.py`（oracle 的排程與 graph 生命週期）、`tests/unit/test_native_track_parity.py`（新增 graph section、oracle log 解析、double-buffer oracle 的有效性） |

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

結果目錄：`results/465_pr10_track/full7_44cc91a4/`（`MANIFEST.md`、`run.sh`、`anchor/`、`oracle_rows/`、`parity/`、`repeat/`、`no_trace/`、`serial_regression/`、`pr8_regression/`、`negctl_*/` 與 log）與 `results/465_pr10_track/attest_306607e3/`；不納入版本控制。

### 14.5 限制

- parity 是同一台機器、headline 組態、`A_L` 的對照，不是一般性的等價主張；沿用 PR-7／PR-8 的條件（nvJPEG 硬體路徑、torch 2.11.0／triton 3.6.0 的 S2 lowering、per-machine 的 operator library 與 realization attestation）。對 `A_L` 是 EXACT，不代表對 headline（owner 的 named limit）。
- double buffer 的正確性靠 event barrier；race 型的變異沒有當負控制（§14.3）。「不帶 trace」與重現性各量了一次，是同一台機器、同一種負載下的觀察，不是無 race 的證明。
- 只量了 oracle 的 sequence 順序與 headline 的 graph key；GPU 測試另外在 40 幀上檢查 X、Y、X、Z（dims 改變時重新 capture、相同 dims 沿用）。
- FPS 只是同 session 的並列，不是效能主張：三者的時間定義不同，native 的 tracker 輸出仍每幀同步讀回（oracle 延後一幀），decode 在 host thread 上依序執行（oracle 預取）。
- operator library 是 per-build 的：重新 configure／完整 build 可能重新編出不同位元組的 `.so`，之後必須依 §14.2 重新 attest（`anchor` 與 `A_L_1` 相同才可以）。
- `saccade_track` 的 `DT_NEEDED` 仍含 OpenCV（經 `TRTEngine`）；拆掉是 PR-11。

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
