# Native runtime resolved config（#465 Phase B PR-3／U2a）

> 狀態：PR-3 完成 exporter、schema 與 Python 端等價 contract test。**不改** eval harness、preset、threshold、native code，也**不**跑 MOT17、不量 GPU、不碰 parity 門檻。native loader（解析、fail-closed、set 後回讀）是 PR-4／U2b。
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
