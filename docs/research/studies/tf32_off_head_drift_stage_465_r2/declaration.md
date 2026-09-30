# TF32-off head drift：軌跡第一次結構分歧時，tracker input 是否已經不同（#465 follow-up r2，探索級）

<!-- evidence-tier: exploratory -->

> 狀態：**宣告（未執行）**。tier = `exploratory`，由 `study.yaml` 的 §20.2 欄位機械推導（所有 terminal → §20.7 `none`，output class 只有 `diagnostic`）。
> 規則 owner：[experiment contract §20.11](../../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)。本研究是 #493 PR-3b 的新流程試跑。
> **不可引用**：任何 formal chain（contract、evidence ledger、NO-GO registry、formal study）都不得引用本研究。結果可以引導下一步；若要據此做設計決定，必須另寫 formal 宣告並重新執行（§20.5），不能改標籤。

---

## 0. 與 r1 的關係

本研究取代 [`tf32_off_head_drift_stage_465`](../tf32_off_head_drift_stage_465/declaration.md)（r1）。r1 的唯一 attempt（001，freeze `763db72d`，tag `freeze/tf32_off_head_drift_stage_465/1`）是 `invalid`，criterion `V_FORMAT`，`terminal: null`，沒有執行任何 arm 間比較（[r1 結案](../tf32_off_head_drift_stage_465/results.md)）。原因是 r1 的 `V_FORMAT` 要求 x2 ≥ x1、y2 ≥ y1，但 head 會輸出少數左右或上下顛倒的框。r1 的宣告、tag 與 attempt 保持原樣，r1 不再執行。

**研究問題、f\*、IoU 門檻 0.9、5-of-7 support threshold、terminal 全部與 r1 相同。** 唯一的方法修正有兩處：

1. `V_FORMAT` 不再要求座標排序：顛倒的框是合法輸入，只列為 report-only 異常（§3、§5）。
2. 結構比較的 eligible pair 多一個條件：**inversion signature 相同**，IoU 以 canonical 矩形計算（§2）。

第 2 點的理由：production tracker（`include/tracking/box_ops.hpp`）的 `width()`／`height()` 是 `max(0, x2−x1)`，`iou()` 直接使用原始座標，runtime 不會把顛倒的框正規化。所以「正常框」與「座標排序後相同、但左右顛倒的框」對 tracker 不是同一個輸入，不應判成結構相同。反過來，兩邊是同一個顛倒框時，若直接用 runtime IoU（面積 0 ⇒ IoU 0），bit-identical 的 row 會被判成結構不同，這也不對。

## 1. 問題

[r2 localization](../../../reference/native_runtime_head_failure_localization_r2_result.md) 在 TF32-off 下給出 `EAGER_NUMERICS_WITHIN` 與機制標籤 `COMMON_ANCHOR_VALUES_SUFFICIENT`：在 membership 邊界之外的共同 anchor 上，取 T 的數值就足以重現 `R_T` 全部的偏差（IDF1 −0.933、IDs +21）。r2 localization 的 row 層級歸屬大多是 `undetermined`，沒有說明軌跡分叉時 tracker 看到的輸入長什麼樣子。

本研究只問一件事：**`R_T` 的軌跡第一次與 `R_C` 結構分歧的那個 frame（f\*），進入 tracker 的偵測集合（`tracker_input`）是否已經結構不同？** 只用 r2 localization 已經存在的逐 stage capture，不跑任何模型。

這**不是**「分歧從哪一層進來」的歸因：f\* 之前可能有過 `tracker_input` 的結構分歧、之後又收斂，這種情況 f\* 當下仍判 `same`（見 §6）。

**不在範圍內**：TF32-on 的逐 stage 分解（PR-2 packet 沒有 stage capture，需要新的 GPU capture run，另開研究）；IDs +21 的逐事件歸因；任何 head 形式或 runtime 路線的選擇；顛倒框的成因。

## 2. 輸入與量

輸入（`study.yaml` `inputs`，皆為 external，以已 commit 的 SHA-256 manifest 綁定；三份 manifest 與 r1 的 manifest 逐位元相同）：

| 名稱 | packet | 用途 |
|:--|:--|:--|
| `r2` | `results/native_head_failure_localization_465_r2/20260928T081021Z` | 主量：5 arm × 2 run × 7 序列的 `.evidence/*.npz` |
| `pr2` | `results/native_head_parity_465/20260927T133422Z`（TF32 allowed） | 只報告：`A_C_1`／`A_T_1` 的 txt |
| `pr2r` | `results/native_head_parity_465_tf32_off/20260927T151100Z`（TF32 disabled） | 只報告：`A_C_1`／`A_T_1` 的 txt |

對照：arm X ∈ {`R_T`, `R_E`, `H_M`, `H_V`}，每個都對 `R_C`；run 1 對 run 1、run 2 對 run 2。主 arm = `R_T`。

**框的比較（所有 stage、tracker output 與 txt 共用）：**

- *inversion signature*：`(x2 < x1, y2 < y1)`。
- *canonical 矩形*：`[min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]`。
- *canonical IoU*：兩個 canonical 矩形的 IoU。兩個 canonical 矩形完全相同時 IoU = 1（包括面積為 0 的情況）；其他 union 為 0 的 pair，IoU = 0。
- *eligible box pair*：inversion signature 相同，且 canonical IoU ≥ **0.9**。

例：C `[0,0,10,10]` 對 T `[10,0,0,10]` signature 不同 ⇒ 不 eligible；C `[10,0,0,10]` 對 T `[10,0,0,10]` signature 相同、canonical IoU = 1 ⇒ eligible。

每個序列、每個 frame f ∈ 1..N：

- **probe stage**（`detector_output`、`post_nms`、`tracker_input`；row = x1, y1, x2, y2, score, class）：probe 沒記錄的 frame 視為 0 列。
  - *bit 分歧*：兩邊 row 的 shape 或 bytes 不同。
  - *結構分歧*：列數不同，或**不存在**只用 eligible pair 的一對一完美配對。row 的 eligible pair = 同 class 且 box eligible。判定方式是對「非 eligible」成本做 assignment，檢查結果是否全部落在 eligible pair 上；這等於問「是否存在 threshold-feasible perfect matching」，**不是**先取總 IoU 最大的配對再檢查門檻。**score 不比較。**
- **tracker output**（每 frame 的 track id 與 xyxy box；沒有 emit 的 frame 視為 0 個 track）：
  - *bit 分歧*：id 或 box 的 bytes 不同。
  - *結構分歧*：id multiset 不同，或某個 id 的 box 之間**不存在**只用 eligible pair 的完美配對。同一 frame 內同一個 id 出現 k 次時，以 k × k 的 eligible 矩陣判定（同一 id 內 box 的順序不是結構）；id 唯一時就是檢查那一對 box 是否 eligible。不要求每個 frame 的 id 唯一。
- **f\***：tracker output 第一個結構分歧的 frame。
- **f\* label**（f\* 存在時）：`tracker_input` 在 f\* 結構分歧 ⇒ `different`；否則 ⇒ `same`。只描述 f\* 當下的 tracker 邊界。

凍結的自由度：IoU 門檻 0.9、「未記錄 = 0 列」、eligible-pair 完美配對（tracker output 為逐 id 的完美配對）、inversion signature 必須相同、canonical IoU（相同矩形 = 1）、以 `tracker_input` 判 f\* label、**5-of-7 support threshold**（比 7 序列的 strict majority 4/7 更強）、run 1 為報告用的 run（run 2 必須相同，見 V_RUN_REPRO）。這些都是事前的選擇，沒有調過；所有結果都寫成「在凍結的 0.9 定義下」。

## 3. Validity

依序檢查，第一個不成立的就是該 attempt 的 invalid criterion（`study.yaml` `validity_criteria` 為準）。每一條都先對**全部**輸入（r2 evidence 與 txt）檢查完，才進入下一條，所以 criterion 不會因輸入讀取順序而改變：

1. `V_COMPLETE`：所有宣告的 member 都在 manifest 裡且讀回 digest 相符；npz 帶齊所需陣列、frame 數等於序列長度（02 600、04 1050、05 837、09 525、10 654、11 900、13 750）。
2. `V_FORMAT`：三個 probe stage（`detector_output`、`post_nms`、`tracker_input`）與 tracker output，每個 arm／run／序列都在比較之前檢查：row 為 6 欄、box 為 4 欄，座標皆為 finite；ragged count 恰好覆蓋 body；frame 不重複且在 1..N。txt 每列可解析、left／top／w／h 皆為 finite、frame 在 1..N。**顛倒的框（x2 < x1 或 y2 < y1；txt 為 w < 0 或 h < 0）是合法輸入**，不構成 invalid。
3. `V_REF_SELF`：`R_C` run 1 與 run 2 在每個 stage、每個 frame 都逐位元相同。
4. `V_RUN_REPRO`：每個 arm、每個序列，run 1 與 run 2 的整份比較結果（含 inverted-signature-count mismatch frame 數）相同。
5. `V_RUNNER`：freeze 通過後，計算或報告程式碼丟出上面四條都不涵蓋的例外（runner bug）；只由例外決定，不看任何計算出的量。

這五條都與結果是否有利無關。attempt 政策：`first_valid`、最多 1 次 valid、最多 3 次 attempt。有效執行的結果不重跑；invalid attempt 修正後以新 attempt（新 tag `freeze/tf32_off_head_drift_stage_465_r2/<n>`）追加，舊 attempt 保留。

## 4. Terminal

只由 `R_T` 的 7 個 f\* label 決定，門檻是 5-of-7 support：

| terminal | 條件 | §20.7 |
|:--|:--|:--|
| `TRACKER_INPUT_SAME_AT_FSTAR` | ≥ 5 個序列為 `same` | none |
| `TRACKER_INPUT_DIFFERENT_AT_FSTAR` | ≥ 5 個序列為 `different` | none |
| `SPLIT` | 其他（包括 f\* 不存在的序列太多） | none |

terminal 只陳述「在凍結的 0.9 定義下，f\* 當下 `tracker_input` 結構相同／不同」，不陳述分歧的來源層或成因。

## 5. 只報告

- 其餘 arm（`R_E`、`H_M`、`H_V`）的同一組量，特別是 `R_E` 當作「容差內的數值雜訊」基準。
- 每個 stage 的第一個 bit／結構分歧 frame 與整個序列的結構分歧 frame 數；f\* 當下各 stage 是否結構分歧、`tracker_input` 列數、f\* 之前 `tracker_input` 的結構分歧 frame 數。
- **顛倒框 census**：每個 arm／run／序列，三個 probe stage 與 tracker output 各自的 `x_inverted_rows`、`y_inverted_rows`、`either_inverted_rows`。
- **inverted-signature-count mismatch frame 數**：每個 arm 對 `R_C`、每個 probe stage，兩邊「只 x 顛倒、只 y 顛倒、兩者皆顛倒」三個個數不同的 frame 數（未顛倒的個數不列入）。不另設 terminal。
- 最終 txt 層（含後處理）的第一個 bit／結構分歧 frame，四組：TF32-on T vs C、TF32-off T vs C、T 的 TF32-on vs off、C 的 PR-2 vs PR-2R。

## 6. 解讀限制

- terminal 只描述 f\* 當下 `tracker_input` 的結構狀態，不是分歧的來源層，也不是 IDs +21 的成因；f\* 之後 tracker 狀態已經不同，之後的分歧不歸因。
- f\* 之前若曾有 `tracker_input` 結構分歧而後收斂，f\* label 仍可能是 `same`；`tracker_input_structural_frames_before` 只報告這件事，不改 label。
- 若 `R_E`（容差內）的 f\* label pattern 與 `R_T` 相同，terminal 就沒有說出任何 T 特有的東西；結果文件必須並列 `R_E`。
- `detector_output`／`post_nms`／`tracker_input` 是否完全不讀 tracker 狀態，本研究沒有驗證；整序列的 stage 結構分歧數只在這個假設下才是 tracker-independent 的比較。
- IoU 0.9 是事前固定的值；不同門檻可能改變 label，本研究不做門檻敏感度。
- inversion signature 規則是對 runtime box semantics 的近似：它保留「顛倒與否」這個 tracker 會看到的差異，但同 signature 的兩個顛倒框仍以 canonical 矩形比較，不模擬 tracker 對顛倒框實際算出的 IoU（通常為 0）。
- TF32-on 只有最終 txt 層，不能與 TF32-off 做 stage 層級的比較。

## 7. 事前看過的資料

- r2 localization、PR-2、PR-2R 的結果文件（含 r2 localization §4 報告的第一個 `tracker_input`／txt 分歧 frame 範圍與 census 加總）及 PR-2 packet `l2` 區塊的已公開彙總數字。
- r2 evidence npz 的**欄位名稱與形狀**（一個 arm、一個序列），以及 r2／PR-2 packet 的檔案清單；另外對三份 packet 計算了 manifest digest（只雜湊 bytes）。
- **r1 attempt 001**：`invalid`（`V_FORMAT`）、`terminal: null`，detail 為 `R_C_1/MOT17-02-SDP detector_output rows are not finite xyxy`。r1 的 freeze tag 與 attempt 已存在。
- **r1 失敗後的格式 probe**（在 frozen handle 之外直接讀檔，只看一個 arm／run／序列：`R_C_1`／MOT17-02）。只計算了非有限值與顛倒列的數量：

  | stage | 列數 | 非有限值 | x2<x1 | y2<y1 |
  |:--|--:|--:|--:|--:|
  | `detector_output` | 180000 | 0 | 36 | 1 |
  | `post_nms` | 15144 | 0 | 6 | 0 |
  | `tracker_input` | 14789 | 0 | 4 | 0 |

  probe **沒有**做任何 C↔T 或跨 arm 比較，沒有讀其他 arm、run 或序列，也沒有讀 tracker output 或 txt。
- **沒有看過任何跨 arm 比較的結果**，也沒有讀過任何 txt 內容。

## 8. 流程

同一個 draft PR（#500，與 r1 同一個）：commit `study.yaml`、本宣告、三份 manifest、runner 與其 unit test → review → `git tag -a freeze/tf32_off_head_drift_stage_465_r2/1` 並 push → 在該 commit 的乾淨 tree 上執行 runner（`open_frozen_study` 先驗 freeze 才給資料）→ commit attempt 與 `results.md` → 以 merge commit 合併。
