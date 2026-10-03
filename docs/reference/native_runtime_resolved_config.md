# Native runtime resolved config（#465 Phase B PR-3／U2a）

> 狀態：PR-3 完成 exporter、schema 與 Python 端等價 contract test。PR-4a 加上 native strict loader（§7）。PR-4b 讓 native 物件只從單一參數狀態讀值、native 端不再讀 `SACCADE_*`、shipping 以這份 JSON 建立 GPU 物件並做 set 後回讀（§8）；**U2b 至此完成**。PR-3／PR-4a 不改 native code；PR-4b 改 native code 但 headline 輸出逐位元組不變（§8.6）。
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

讀取時機改變的幾項只有在 process 內途中改 env 時才有差別；oracle 不會這樣做（`configure_runtime_env` 在建立任何物件之前設定 env）。exporter 的 `native_env` 掃描現在在 `legacy_env.cpp` 找到同樣的 18 個讀取，輸出逐位元組不變（`--check`）。source guard 測試要求 native 原始碼中只有 `legacy_env.cpp` 與 pybind binding 讀 `SACCADE_*`，`shipping/` 不得 include `legacy_env.hpp`。

### 8.3 Graph capture 之後凍結

`update()`／`update_into()` 第一次在 CUDA stream capture 中執行時（`cudaStreamIsCapturing`），tracker 設 `config_frozen`。之後每一個寫 `params_` 或啟用 research hook 的 setter 都丟 `std::logic_error`（Python 為 `RuntimeError`），因為已 capture 的 graph 會一直 replay 當時寫死的 kernel 引數，新值永遠不會生效。不凍結的是逐幀狀態輸入（`update_reference_features`、`set_clean_embedding_flags*`、`bind_features_buffer`）、drain／clear，以及診斷用的 `set_assoc_dump_path`。

對 eval harness 的影響：headline 在 `EvalPipeline.__init__` 設完所有參數之後才在第一幀 capture tracker graph，所以沒有影響（§8.6）。若某組態在 `use_tracker_graph` 之下於 capture 之後才呼叫 setter（例如 `stages.py` 在逐幀 threshold 改變時呼叫 `set_params`），這個呼叫原本對 replay 的 graph 不起作用；現在會直接報錯。

### 8.4 Shipping 建立物件與 set 後回讀

`build_tracker`／`build_gmc`／`build_perception_pipeline` 以 JSON 的 constructor 值建立物件，先套 `native_env` 的 hatch，再依 oracle 的呼叫順序（schema 順序）呼叫同一批 setter，不依賴任何 native 預設值；接著讀 `snapshot()` 與 JSON 比對，不一致就丟 `ConfigError`：

- snapshot 的每個鍵都要有 JSON 值，JSON 的每個值都要有 snapshot 鍵；
- float 以 setter 實際做的 float32 轉型後**逐位元**比較；`per_sequence` 以該 sequence 的 `seqinfo.ini` 值比較；`{"unset_effect": …}` 對應的診斷必須是 off；
- 沒有 JSON 值的 snapshot 鍵只有下列幾個，各附理由（`tracker_native_only_expectations()`）：`set_reid_min_candidates.min_candidates`（＝2；沒有 pybind binding，oracle 從未設定，exporter 因此沒有值；只在傳入 embeddings 的 association 分支讀取，shipping 關閉 ReID，不會走這個分支）、4 個 research hook（必須 off）、`config_frozen`（必須 false）；
- `native_env` 的 18 個鍵各自歸屬一個物件：tracker 12、GMC 1、pipeline 3；`HO_DEBUG_LEVEL` 與 `KALMAN_ADAPT_MODE` 沒有 shipping consumer（理由寫在 `native_env_consumers()`）。

PR-4a 的範圍 guard 沒有改。會被 setter canonicalize 的值（例如 `confirm_streak=0`、`score_w=1.5`、`occ ttl=0`、`SACCADE_COAST_MAX_AGE=2.5`）可以通過 loader，但會在 readback 失敗，錯誤訊息列出鍵與兩邊的值。

### 8.5 驗收（boundary §6 PR-4）

| 驗收項 | 證據 |
|:--|:--|
| `native_params` set 後回讀 == JSON | CPU：在 `TrackerParams`（tracker 實際使用的 setter 程式碼）與規劃的 GMC／pipeline snapshot 上比對；GPU：在真正的 `GPUByteTracker`／`GMC`／`PerceptionPipeline` 上比對，並確認與 CPU 規劃逐欄相同 |
| 每個 resolved 值都到達恰好一個 native 欄位 | 逐一擾動 123 個值，每次只有對應的那個 snapshot 鍵改變；另有 2 個值被 loader 釘死、無法擾動（`reid_ptr`、`cropper_ptr` 只允許 0） |
| 設定任何 env 都不改變結果 | 把 18 個 `native_env` 變數設成多組值，CPU 規劃與 GPU 物件的 snapshot 都不變 |
| 未知／缺少欄位 fail-closed，刪掉有 native 預設的欄位必失敗 | PR-4a loader（§7，未改） |
| 凍結 | GPU：capture 之後 19 個 setter 全部丟錯，snapshot 不變；未 capture 的 update 不凍結 |

測試計數：CPU 228 checks、GPU 50 checks、pybind 10 tests，全部通過。

### 8.6 Headline 行為

同一台機器、同一個 `build/` 組態，以 boundary §2 的 oracle（`mot17.py --preset mamba_whole_graph --detector SDP --double-buffer`）跑 MOT17 train 7 個 SDP sequence：main `469f159d` 跑兩次彼此逐位元組相同；PR-4b 的 build 跑一次，7 份 MOT txt 與 main 逐位元組相同（IDF1 78.3／MOTA 77.9／IDs 429）。這只說明 headline 組態下輸出未變，不是一般性的等價主張。

### 8.7 分類與 runtime identity

`shipping/` 在 `h2_path_partition.py` 中改為 `decision_relevant`（PR-4b 第一次把它連到 tracker）。root `CMakeLists.txt` 加入 `saccade_tracker_params` 與 `add_subdirectory(shipping)`，移動 environment recipe；`src/`、`include/` 的改動移動 implementation 軸；`h2_path_partition.py` 移動 identity_semantics 軸。runtime coordinate 依 [republication runbook](runbooks/runtime_identity_republication.md) 重新出版，以 stacked PR 和本 PR 一起 land。

### 8.8 限制與留給後續的事

- `reid_min_candidates` 是唯一沒有 JSON 來源、但 kernel 會讀的參數（只在 embeddings 分支）。U3 的 shipping 宿主必須以 null embeddings 呼叫 update；另一個做法是讓 exporter 也輸出它。
- Python wrapper 的 `set_reid_min_candidates(1)`（`pipeline.py` 在某些 ReID 組態下呼叫）因為沒有 binding 而是 no-op，這是 PR-4b 之前就存在的情況，PR-4b 沒有改它。
- readback 證明 native 狀態等於 JSON；kernel 是否用到每個欄位，由逐欄映射測試與 §8.6 的 headline 輸出不變共同支撐，不是由 readback 本身證明。
- U3–U5（native 宿主、ingest、graph／double-buffer）尚未開始；PR-4b land 之後 PR-5 才能開始。
