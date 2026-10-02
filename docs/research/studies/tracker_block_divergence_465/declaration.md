# Tracker block divergence：R_T 在 tracker 內部第一次結構分歧落在哪個 block（#465 follow-up，探索級）

<!-- evidence-tier: exploratory -->

> 狀態：**宣告（未執行）**。tier = `exploratory`，由 `study.yaml` 的 §20.2 欄位機械推導（所有 terminal → §20.7 `none`，output class 只有 `diagnostic`）。
> 規則 owner：[experiment contract §20.11](../../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)。
> **不可引用**：任何 formal chain（contract、evidence ledger、NO-GO registry、formal study）都不得引用本研究。結果可以引導下一步；若要據此做設計決定，必須另寫 formal 宣告並重新執行（§20.5），不能改標籤。
> **不改 production behavior**：本研究不改任何 `src/`、C++ 或 preset。replay 只在 runner 的 worker process 內替換兩個 Python 屬性（§2.1），並使用 tracker 既有的 `SACCADE_ASSOC_DUMP` debug 輸出。

---

## 0. 承接

[`tf32_off_head_drift_stage_465_r2`](../tf32_off_head_drift_stage_465_r2/results.md)（#500）的 terminal 是 `TRACKER_INPUT_SAME_AT_FSTAR`（7/7）：在凍結的結構定義下（同 class、inversion signature 相同、canonical IoU ≥ 0.9 的 eligible-pair 完美配對），`R_T` 的 tracker output 第一次與 `R_C` 結構分歧的 frame（f\* = 283/3/15/319/6/49/14，序列 02/04/05/09/10/11/13），進入 tracker 的 `tracker_input` 與 `R_C` 結構相同。04、05、10 到 f\* 為止 `tracker_input` 沒有任何結構分歧；`R_E`（容差內基準）沒有 f\*。

因此分歧首次出現在 tracker 的觀測面上；它可能涉及低於結構門檻的輸入差異（座標、score）與既有的 tracker 狀態，#500 沒有區分這兩者。本研究把 tracker 的一步切成三個 block，找出第一個結構分歧的 (frame, block)。

## 1. 問題

**在 R_T 對 R_C 的 tracker replay 中，第一個結構分歧的 (frame, block) 是什麼？block 在 7 個序列中是否一致？**

block（§2.2）依 tracker 一步內的執行順序：

- **A association**：predict → cost／candidate → Sinkhorn → auction（S0–S2）之後，每個 active track 的 `trk_to_det`。
- **P state transition**：state update、Kalman update、spawn、bridge 之後，step 結束時的離散狀態。
- **E emission**：compact 之後的 tracker output（r2 的 f\* 就是 E 的第一個分歧）。

第一個分歧可能早於 f\*（A 或 P 的分歧沒有立刻出現在 output，例如 tentative track），也可能就在 f\*。

**不在範圍內**：A 內部的 auction stage（S0–S2 個別結果沒有輸出）；P 內部 state update／spawn／bridge 的拆分（只做 report-only 的描述）；IDs +21 的逐事件歸因；分歧的數值成因；任何 tracker 修改、參數選擇或 runtime 路線；TF32-on（沒有 stage capture）；H_M／H_V arm。

## 2. 方法

### 2.1 Replay

輸入只有 r2 localization packet（`study.yaml` `inputs.r2`，external，manifest 與 r2 stage study 的 `inputs_r2.json` 逐位元相同）。使用其中 `R_C_1`、`R_T_1`、`R_E_1` 三個 arm 的 `.evidence/<seq>.npz`。

每個 replay 是一個 worker process，執行一次未修改的 `scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer`（7 個序列，與 r2 localization 相同 argv），只替換兩個屬性：

1. `evaluator._run_track`：把該 frame 的 `fused_boxes`／`fused_scores`／`fused_classes` 換成該 arm 的 r2 `tracker_input` capture（probe 沒有記錄的 frame 視為 0 列），其他參數不變。即時偵測器照常執行，但它的偵測結果不會到達 tracker 讀取的任何輸入。GMC 不使用前景遮罩（headline `gmc_fg_mask=False`），只由畫面計算；`geometry_mid_scale=False`；headline 沒有 ReID embedding。
2. `GraphedTrackerUpdate._capture`：先做同樣的 warm-up，再裝入一個 eager callable，它發出與 CUDA graph 相同的 `update_into` 呼叫（同一組固定 buffer，`num_dets = max_assoc` 零填充，無 embedding，`light_factor 0`，`mid_thresh_scale 1`）。這樣 tracker 既有的 `SACCADE_ASSOC_DUMP` 才能執行（它做 host I/O，graph capture 不允許）。

這兩個替換是否忠實，不靠推論：`V_REPLAY` 要求每個 replay 在每個 frame 都逐位元重現該 arm 在 r2 中捕捉到的 tracker output（ids、det_idx、boxes）。

每次 `_run_track` 返回後，worker 立即呼叫 tracker 既有的兩個唯讀介面 `get_state_snapshots()` 與 `get_tentative_candidates()`（step 結束狀態，§2.2 P）。兩者都只在 update 所在的 stream 上把 device 陣列拷到 host 再 synchronize，不寫任何 tracker 狀態；它們清掉的 `h_dirty_` 只管 ReID 路徑用的 host slot-map 快取，而它們剛拷回的 `active`／`track_ids` 正是該快取要讀的值。這是否改變 tracker 行為同樣由 `V_REPLAY` 驗證。

replay 順序：`R_C#1`、`R_T#1`、`R_E`、`R_C#2`、`R_T#2`。terminal 由 R_C 對 R_T 的 A／P／E 觀測決定，所以 R_C 與 R_T 各跑兩次，用於 `V_REPEAT`；`R_E` 只報告，跑一次。比較用 `R_C#1` 與 `R_T#1`。

### 2.2 量

`SACCADE_ASSOC_DUMP` 在每次 tracker update、所有 auction stage 之後、state update 之前，對每個 active track 寫一列 `TRK`（id、lifecycle state、age、predict 後的框、candidate 數、最小 cost、`trk_to_det`），以及它的每個 candidate 一列 `CND`（det index、cost），並在開頭寫一列 `GMC`。worker 記錄每次 `_run_track` 前後的 dump 檔大小，把 dump 切成每個 frame 一段；序列的第一段也含 graph warm-up 的那一塊，只取最後一個 `GMC` 之後的部分。

框與偵測的比較沿用 r2 的定義：*eligible pair* = 同 class、inversion signature 相同、canonical IoU ≥ **0.9**。

每個序列、X ∈ {`R_T#1`, `R_E`} 對 `R_C#1`：

- **P(f)**，f = 1..N：step f 結束時 active track 的 (id, lifecycle state, age) multiset 是否相同。直接取自上述 post-update snapshot：`get_state_snapshots()` 給出每個 active slot 的 (id, age, uid, generation)，`get_tentative_candidates()` 給出其中 lifecycle 為 tentative 的那些；以 (id, age, uid, generation) 一對一配上後，tentative 記為 state 1，其餘 active 記為 state 2。active slot 只會被寫成 1 或 2（spawn 寫 1 或 2；post-update kernel 只在同時設 `active=false` 時寫 0），所以這個對應是精確的。不從下一 frame 的 dump 反推：下一 frame 的 predict 會先把 age ≥ max_age 的 track 停用，dump 只列 active track，兩邊在 step 結束時的差異可能因此消失。
- **A(f)**，f = 1..N：若 frame f 的 dump 中 (id, state, age) multiset 不同，A(f) 未定義（predict 與 pre-state kernel 是這三個欄位的函數，所以這只會發生在 P(f−1) 已分歧之後）。否則依 id 分組，要求存在完美配對，配對的兩個 entry 必須 state、age 相同，且 `trk_to_det` 指向結構相同的偵測：兩邊都是 −1；或兩邊都指向該 frame 的真實 `tracker_input` 列，且兩列是 eligible pair；或兩邊都指向同一個零填充 index。
- **E(f)**，f = 1..N：tracker output 的結構相等，與 r2 完全相同（id multiset 相同，且每個 id 的框存在 eligible 完美配對）。

**第一個分歧**：對 f = 1..N 依序檢查 A(f)、P(f)、E(f)；第一個不相等的 (frame, block)。

凍結的自由度：IoU 門檻 0.9 與 r2 結構定義、三個 block 的切法與順序、P 只看 (id, state, age)、A 以 eligible pair 判 `trk_to_det`、`R_C#1` 為參考、5-of-7 support threshold。這些都是事前的選擇，沒有調過。

## 3. Validity

依序檢查，第一個不成立的就是該 attempt 的 invalid criterion（`study.yaml` `validity_criteria` 為準）：

1. `V_COMPLETE`：要注入的 21 個 npz 都在 manifest 內，讀回的 digest 相符。
2. `V_FORMAT`：這些 npz 的 `tracker_input` 與 tracker 陣列 ragged count 恰好覆蓋 body，frame 不重複且在 1..N，`tracker_input` 列為 6 欄且 finite。
3. `V_REPLAY`：每個 worker exit 0；每個序列的 tracker 恰好依序在 frame 1..N 各被呼叫一次，注入列數等於該 frame 的 capture 列數；emit 的 tracker output 在每個 frame 都與該 arm 的 r2 run-1 capture 逐位元相同。
4. `V_RECORD`：dump 的每個 byte 都落在某次 tracker 呼叫內；每段都有 `GMC` 列，只含格式正確的 `GMC`／`TRK`／`CND` 列（`CND` 緊接在同 id 的 `TRK` 之後，`t2d ≥ −1`）；每個序列 frame 1..N 各有一筆 step 結束狀態記錄，其中每個 tentative entry 都恰好配上一個 active entry。
5. `V_REPEAT`：R_C 的兩次 replay、R_T 的兩次 replay，每個序列每個 frame 的 dump 段逐位元相同，step 結束狀態記錄完全相同。
6. `V_RUNNER`：freeze 通過後，計算或報告程式碼丟出上面五條都不涵蓋的例外（runner bug）；只由例外決定。

前兩條在任何 replay 啟動前對全部輸入檢查完。這六條都與結果是否有利無關。attempt 政策：`first_valid`、最多 1 次 valid、最多 3 次 attempt；有效執行的結果不重跑。

## 4. Terminal

只由 `R_T` 7 個序列的第一個分歧 block 決定，門檻 5-of-7：

| terminal | 條件 | §20.7 |
|:--|:--|:--|
| `FIRST_DIVERGENCE_ASSOCIATION` | ≥ 5 個序列的第一個分歧在 A | none |
| `FIRST_DIVERGENCE_STATE_TRANSITION` | ≥ 5 個序列在 P | none |
| `FIRST_DIVERGENCE_EMISSION` | ≥ 5 個序列在 E | none |
| `SPLIT` | 其他 | none |

terminal 只陳述「在凍結的結構定義下，第一個結構分歧出現在哪個 block」，不陳述成因，也不陳述這個分歧導致了 f\* 或 IDs +21。

## 5. 只報告

- `R_E` 的同一組量（容差內基準）：`R_E` 沒有 f\*，若它的 A／P 仍有分歧，表示這類 latent 分歧不是 T 特有的。
- 每個序列：第一個分歧的 (frame, block)；每個 block 的第一個分歧 frame 與分歧 frame 數（A 另列未定義的 frame 數）；第一個分歧與 f\* 的距離。
- 第一個分歧當下：`tracker_input` 結構／位元是否相同、列數、之前的 `tracker_input` 結構分歧 frame 數。
- A：每個分歧 track 的 id、state、age、兩邊的 `trk_to_det` 列（含 score）、candidate 列表與 cost、predict 後的框，以及 candidate 集合是否結構相同（區分「candidate 集合不同」與「同樣的 candidate、不同的指派」）。
- P：只在一邊出現的 (id, state, age)、只在一邊出現的 id，以及該 frame 的 A 是否相同。
- E：只在一邊 emit 的 id、同 id 但框不 eligible 的 id。
- 每個序列兩邊 `GMC` 列（dump 中 3 位小數）不同的 frame 數。
- E 的第一個分歧必須等於 r2 stage study 的 f\*（`V_REPLAY` 與同一個比較器的推論）；結果文件並列。

## 6. 解讀限制

- 結論只涵蓋 TF32-off 的 r2 replay，且只在 eager replay 與 r2 capture 逐位元相同（`V_REPLAY`）的前提下代表 graph 執行。
- P 只觀測 (id, lifecycle state, age)（uid、generation 只用於 tentative 配對與 `V_REPEAT`）。hit_streak、class、uid、occlusion partner、foot ring、bridge 記錄等其他離散欄位沒有觀測；它們的分歧要等到影響這三個欄位、`trk_to_det` 或 output 時才會被看到，所以「第一個分歧」是在這三個觀測面上的第一個。
- A 只看所有 auction stage 之後的結果，不能分辨分歧出現在哪個 stage；candidate 集合與 cost 只報告。
- dump 中的框與 cost 是格式化後的文字（框 1 位小數、cost 4 位小數）；它們只用於報告。A 的判定只用 index 與 r2 capture 的原始列。
- 第一個分歧是「最早可觀測到的結構差異」，不是 f\* 的成因；之前的 `tracker_input` 結構分歧（02、09、11、13）也可能影響它，本研究不做因果歸屬。
- 若 `R_E` 在同一個 block 也有早期分歧，該 block 的分歧本身就不是 T 特有的；結果文件必須並列 `R_E`。
- 0.9 門檻是事前固定的，不做敏感度分析。

## 7. 事前看過的資料

- r2 localization 與 r2 stage study（#500）的結果文件與 report（f\*、各 stage 的分歧 frame、census）。
- tracker 原始碼（`src/tracking/tracker_gpu.cu` 的 `run_update_device` 與相關 kernel、`get_state_snapshots`／`get_tentative_candidates`／`ensure_slot_map`、`GraphedTrackerUpdate`、evaluator 的 `_run_track` 呼叫點）以及 headline 的 golden config。
- 宣告 review round 1（owner，2026-10-02）：P 原本從下一 frame 的 dump 反推，會漏掉被 predict 停用的 track（改為直接 snapshot）；`V_REPEAT` 原本只重播 R_C（補上 R_T#2）。可行性 probe 在這次修改之前執行，沒有呼叫 snapshot 介面。
- **修改後的 worker smoke**（同樣在 frozen handle 之外，2026-10-02）：用本 runner 的 `run_worker`（含 snapshot 呼叫）只跑 `R_C_1`／MOT17-05，只與 `R_C_1` 自己的 r2 tracker capture 比較：837/837 frame 逐位元相同，snapshot 配對在 837 個 frame 都成立（tentative 列 778 筆，最多 23 個 active track）。沒有跑 R_T、沒有任何跨 arm 比較，輸出已刪除。
- **可行性 probe**（在 frozen handle 之外直接讀 r2 npz，2026-10-02，與本 runner 同樣的兩個替換）：只跑 MOT17-05，`R_C_1` 與 `R_T_1` 各一次，並各自只與**自己**的 r2 tracker capture 比較。兩者都是 837/837 frame 逐位元相同，837 次 tracker 呼叫各有非空的 dump 段；`R_C_1` 另外確認 837 段恰好覆蓋整個 dump 檔（第一段含 warm-up 的 `GMC` 列）。probe **沒有**做任何 R_C↔R_T 或跨 arm 比較，也沒有解析 `TRK`／`CND` 的內容（只看了 `R_C_1` dump 的前幾列以確認格式）。probe 的輸出（dump、emit、log）已刪除，未保留。

## 8. 流程

同一個 draft PR：commit `study.yaml`、本宣告、manifest、runner 與其 unit test → review → `git tag -a freeze/tracker_block_divergence_465/1` 並 push → 在該 commit 的乾淨 tree 上，持有 `gpu0`（或 `machine-bench`）lease 執行：

```
.venv/bin/python tools/resctl.py run gpu0 -- \
    .venv/bin/python scripts/eval/diagnostics/tracker_block_divergence_465.py \
    --raw-out results/tracker_block_divergence_465/<UTC stamp>
```

`--raw-out` 是一個新的目錄（以 `run_manifest.open_run` claim），保存每個 replay 的 dump、呼叫記錄、emit、worker log 與 mot17 輸出（mot17 的 txt 混合了注入的 tracker 與即時偵測器的其他輸出，不是證據），並寫出 `MANIFEST.json`；attempt 的 `result.json` 記錄該 manifest 的 SHA-256。→ commit attempt 與 `results.md` → 以 merge commit 合併。
