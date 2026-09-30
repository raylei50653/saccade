# TF32-off head drift r2：結果（探索級）

<!-- evidence-tier: exploratory -->

> **不可引用**：exploratory，任何 formal chain 都不得引用（§20.11.2）。要據此做設計決定，必須另寫 formal 宣告並重新執行。
> 採用的 attempt：[attempts/001](attempts/001/attempt.json)（`valid`，`first_valid`）。freeze commit `f8a4716f`，tag `freeze/tf32_off_head_drift_stage_465_r2/1`。完整數字見 [report.md](attempts/001/report.md) 與 `result.json`。

## Terminal

**`TRACKER_INPUT_SAME_AT_FSTAR`**：`R_T` 的 7 個序列全部是 `same`（7/7，門檻 5/7）。

在凍結的定義下（inversion signature 相同、canonical IoU ≥ 0.9、eligible-pair 完美配對），`R_T` 的 tracker output 第一次與 `R_C` 結構分歧時，`tracker_input` 與 `R_C` 結構相同。7 個序列在 f\* 當下的 `detector_output`、`post_nms`、`tracker_input` 也都結構相同，`tracker_input` 列數兩邊一致。

## R_T 逐序列

| 序列 | f\* | f\* 當下 `tracker_input` 列數 | f\* 之前 `tracker_input` 結構分歧 frame 數 | `tracker_input` 第一個結構分歧 frame |
|:--|--:|--:|--:|--:|
| 02 | 283 | 37 | 18 | 6 |
| 04 | 3 | 40 | 0 | 15 |
| 05 | 15 | 13 | 0 | 35 |
| 09 | 319 | 9 | 5 | 16 |
| 10 | 6 | 34 | 0 | 13 |
| 11 | 49 | 18 | 8 | 3 |
| 13 | 14 | 35 | 3 | 5 |

- 04、05、10 這 3 個序列，到 f\* 為止 `tracker_input` 沒有任何結構分歧。也就是說，tracker output 的結構分歧出現時，送進 tracker 的偵測集合只有低於結構門檻的數值差異。
- 另外 4 個序列（02、09、11、13）在 f\* 之前有過 `tracker_input` 結構分歧（3–18 個 frame），但到 f\* 當下已經結構相同。依宣告 §6，這不改 label，本研究也不判斷這些較早的分歧是否導致了 f\*。

## 只報告

- **`R_E`（容差內基準）**：7 個序列的 tracker output 都沒有結構分歧（沒有 f\*），`tracker_input` 結構分歧只有 0–1 個 frame。所以 `R_T` 的 f\* pattern 不是 `R_E` 那種程度的數值雜訊就會產生的（宣告 §6 第三點的檢查）。
- **`H_V`**：7 個序列的 f\* 與 `R_T` 完全相同，label 也全是 `same`。
- **`H_M`**：只有 MOT17-04 有 f\*（557），label 為 `different`；其他 6 個序列沒有 f\*。
- **tracker output 的第一個 bit 分歧**：`R_T`、`R_E`、`H_V` 在 7 個序列都是 frame 3。`R_E` 也是，所以 bit 分歧本身不區分這些 arm；區分它們的是結構分歧。
- **顛倒框**：`R_C` 在 04 與 09 沒有顛倒框，其餘序列的 `detector_output` 有 16–183 列、`tracker_input` 有 1–41 列；tracker output 在所有 arm 都沒有顛倒框。所有 arm／stage 的 inverted-signature-count mismatch frame 數都是 0，因此 inversion signature 規則沒有影響任何 label。
- **最終 txt 層**：TF32-off T vs C 的第一個結構分歧，02 在 frame 283，其餘 6 個序列都在 frame 3。TF32-on T vs C 的第一個結構分歧，02 在 frame 129，其餘 6 個序列都在 frame 3。C 在 PR-2 與 PR-2R 之間逐位元相同。txt 含後處理，與 tracker output 不是同一層，不做逐 frame 對應。

## 解讀限制

- terminal 只描述 f\* 當下 `tracker_input` 的結構狀態，不是分歧的來源層，也不是 IDs +21 的成因。
- 「結構相同」是在 IoU 0.9 的凍結定義下；低於門檻的座標差異與 score 差異都不算。本研究沒有做門檻敏感度，更嚴的門檻可能改變 label。
- 本研究沒有驗證 probe stage 完全不讀 tracker 狀態。
- 結果只涵蓋 TF32-off 的 r2 replay；TF32-on 只有 txt 層。
