# Native runtime head failure localization r2 — 預宣告（#465 Phase B redesign，localization study 重新宣告）

> 狀態：**預宣告，於 r2 的第一個量測 run 之前凍結。** **declaration 凍結點** = 本文所在 PR merge；review 期間的修訂列在 §11，凍結之後只能以 §11 的 append-only amendment 修訂，不得 inline 編輯。**執行凍結點**見 §4 V1。
> 這是**新的**宣告，不是 [r1 宣告](native_runtime_head_failure_localization_declaration.md)（blob `38f15007`）的 amendment。r1 的正式 run `results/native_head_failure_localization_465/20260928T070925Z/` 的 terminal **`UNRESOLVED`** 永久保留，不重跑、不覆蓋，也不作為本文任何門檻或定義的輸入（§1）。
> 權威 seal bar：[experiment contract §20.8](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（引用，不複述）。
> 前置結果：[PR-2](native_runtime_head_parity_result.md)（TF32 allowed）與 [PR-2R](native_runtime_head_parity_tf32_off_result.md)（TF32 disabled）皆 `HEAD_PARITY_OUT_OF_TOLERANCE`。

---

## 0. 這個 study 回答什麼

研究問題、arm 組、容差、decision terminal 與它們對應的主線轉移都與 r1 相同。唯一的實質修改是 hybrid arm 的**替換集合**（§2 的 `S(f)`，r1 是 `Δ(f)`）與 V3 的 invariant（§4）；原因與不讀資料的結構驗證見 §2.1。

> TRT head 與 oracle head 之間的小幅 tensor 差異，是在哪一個離散邊界被放大成 MOT association divergence？

| | 狀態 |
|:--|:--|
| 替換集合 S(f)（Δ ∪ top-300 cutoff 帶）取 T 的完整 logits，是否就足以造成 divergence | **回答**（§5 factorial，機制標籤 `S_suff`） |
| S(f) 固定取 C、其餘 anchor 取 T 的數值（membership／class 經 V3 確認等於 C），是否就足以造成 divergence | **回答**（§5 factorial，機制標籤 `V_suff`） |
| membership 邊界本身（與替換 anchor 上的 score／box 數值分開）是否為原因 | **不回答**：`H_M` 對 S 替換的是完整 84 維 logits，membership 與數值一起改變 |
| 一個與 oracle 同為 PyTorch kernel、但不 bit-exact 的 head（E：eager head）在 replay 系統中是否在容差內 | **回答**（§5 decision terminal；決定 LibTorch 是否優先） |
| 每個 crossing anchor、top-k、tracker input、association 的第一個分歧 | **報告，不判定**（§6） |
| TF32 與 tactic selection 的效應分離 | **不回答** |
| 任何 head 形式是否可接受 | **不回答**；本 study 不是 parity trial，不產生 parity terminal。`R_E` 在容差內**不等於** LibTorch 形式被接受：LibTorch artifact 仍須自己的 PR-1 與 parity 宣告 |
| 與 r1 的通道比較 | **不回答**：r1 沒有產生任何 hybrid 結果；本文的 `S_suff` 與 r1 的 `D_suff` 通道定義不同（S ⊇ Δ），不得並列 |
| FPS／latency、benchmark claim | 不回答、不改 |

target decision layer = `none (cross-layer substrate work)`；κ 見 §5。

## 1. 宣告前已看過的資料

全部是 archived reference，**不參與**任何門檻或定義的設定（§5 的門檻沿用 PR-2 宣告在看到任何結果之前凍結的 floor）：

- r1 宣告 §1 列出的全部內容（PR-1 冒煙、#473 smoke、PR-2 與 PR-2R 正式 run 的全部結果、PR-1R 零輸入結構檢查）。
- r1 正式 run `20260928T070925Z` 中**只**讀過：`packet.json` 的 terminal（`UNRESOLVED`）、`unresolved_reasons`、`aborted_after`（`H_M#1`）、V1 21/21、每個 run 的 exit 狀態（`R_C#1`／`R_T#1` exit 0），以及 `H_M#1` stdout 的例外訊息 `V3 failed at replay call 977 (H_M): ['recomputed membership differs from the in-Δ source']`（call 977 = MOT17-04 frame 378）。`R_C#1`／`R_T#1` 的 txt、evidence、metric 與 r1 的 Δ census **都沒有讀、沒有評分**，本文也不引用。
- §2.1 的結構驗證只用合成 logits，沒有讀任何 MOT17 frame。

本文撰寫時**沒有**執行任何新的 frame、replay 或 instrumentation 量測。

## 2. 被分解的系統

whole-graph detect 在 headline 設定下是**當前 frame 的純函數**（與 r1 §2 相同的依據：`mamba_gated_detector.py` `_whole_graph_fn`、`set_gmc_warp`；`pipeline._double_buffer_eligible`），所以 C 與 T 的 detector 輸出可以在同一 frame 上逐 anchor 對齊比較。detector 出口之後第一個讀 tracker 狀態的步驟是 private continuation 的 active-track priors（`stages.py` `_run_native_tensor_prep`／`_run_nms`）。

anchor a 對 head X 的 **membership**（與 r1 相同）：
`m_X(a) = [ s_X(a) ≥ 0.05 ] ∧ [ rank_X(a) ≤ 300 ]`，`s_X(a)` = 80 類 sigmoid 的最大值，`rank_X` = 同 frame 8400 個 anchor 依 `s_X` 的 top-k 名次（`max_det = 300`，`conf_thr = 0.001`，small-P3 fusion 關）；member 的 class = argmax。
**Δ(f)**（與 r1 相同）= `{a : m_C(a) ≠ m_T(a)} ∪ {a : m_C(a) = m_T(a) = 1 ∧ class_C(a) ≠ class_T(a)}`。
0.05 = headline 的 `base_score_floor`；0.05 之上的其他門檻（private `min_score` 0.10、`new_track_thresh` 0.28、`confirm_score_thresh` 0.50、NMS IoU、match）屬於 **values** 通道。

**cutoff 門檻與 cutoff 帶（新）**：
- `θ_X(f) = max(0.05, s_X^(300)(f))`，`s_X^(300)` = 該 frame 依 `s_X` 排序的第 300 個 score。frame 內 `s_X ≥ 0.05` 的 anchor 不超過 300 個時 top-300 不在 0.05 之上截斷，`θ_X = 0.05`（除非恰好 300 個）；超過 300 個時 `θ_X` 是第 300 名的 score。
- `L(f) = min(θ_C, θ_T)`，`U(f) = max(θ_C, θ_T)`。
- **cutoff 帶** `B(f) = {a : s_C(a) ∈ [L, U] ∨ s_T(a) ∈ [L, U]}`：在任一 head 的 score 落在兩個 cutoff 之間（含端點）的 anchor，也就是 top-300 名次會因「其他 anchor 的數值由哪個 head 提供」而改變的 cutoff-equivalence candidates。
- **替換集合** `S(f) = Δ(f) ∪ B(f)`。兩個 head 都不在 0.05 之上截斷時，`L = U = 0.05`，B 只含 score 恰為 0.05 的 anchor，S 實質上等於 Δ。

### 2.1 為什麼 r1 的 Δ 不封閉，以及 S 為什麼封閉（不讀資料）

**r1 的缺陷**：r1 的 hybrid 以 Δ 組合後重算 top-300。不在 Δ 內的 anchor 保留另一個 head 的 score，卻和 Δ 內換成 T 的 score 競爭同一個 cutoff。top-300 在 0.05 之上截斷時，這個競爭可以把組合後的 membership 推離來源 head。反例（k = 2 代替 300）：C 的 score a 0.9／b 0.8／d 0.75／c 0.1，T 的 score a 0.9／c 0.6／d 0.5／b 0.4 ⇒ C 的 member {a,b}、T 的 member {a,c}、Δ = {b,c}；`H_M` 組合 = a 0.9／b 0.4／c 0.6／d 0.75（d 不在 Δ，保留 C），top-2 = {a,d} ≠ T 的 {a,c}。所以 r1 V3(ii) 在截斷 frame 上不是可成立的 invariant。

**合成驗證（r1 runner 自己的 `membership`／`delta_set`／`compose`／`v3_problems`，blob `3b88ff3c`）**：8400 anchor、T = C + 高斯擾動；0.05 之上 150／290／300 個 anchor（不截斷）時 600 個 frame 的 r1 V3 失敗 = 0；310／400／1200 個 anchor（截斷）時每組 200 frame 失敗 25–112 次（H_M 與 H_V 都有）。這證明失敗是結構性的，只發生在截斷 frame；**它不歸因 MOT17-04 frame 378 的具體原因**（本文沒有讀那個 frame 的資料）。

**S 的封閉性**（以 `H_M` 為例；`H_V` 將 C、T 對調）。S 之外的 anchor 只有兩類：(high) `s_C, s_T > U`，兩個 head 都是 member 且 class 相同（否則在 Δ）；(low) `s_C, s_T < L`，兩個 head 都不是 member。組合後 high anchor 的值 > U，S 內是 T 的值，low anchor 的值 < L。T 的 member = 全部 high ∪ S 內依 T 排序進入前 300 且 ≥ 0.05 的 anchor；組合後值 > U 的 anchor 數與 T 中相同（≤ 300），S 內的相對順序就是 T 的順序，而 low anchor 的值 < L ≤ θ_T 且 L ≥ 0.05，排不進 T 的 member 之前。所以組合後的 membership 與 class 等於 T，**唯一例外是 cutoff 上的完全平手**：`topk` 不保證在同分的 index 中回傳哪一個，平手類內的個別 anchor 可以不同，但 member 個數相同。以同一個合成產生器（加上強制平手與 class flip）測 1440 個 frame，S 與下面的 V3 invariant 的失敗 = 0。

## 3. 實作與 arms

replay detector 與 r1 §3 相同（unmodified `scripts/eval/mot17.py`，`--preset mamba_whole_graph --detector SDP --double-buffer`，7 個 SDP 序列；只把 detector 的 `detect_raw` 換成組合函數；同一份 TRT backbone 特徵同時餵 C（compiled head＋block，oracle）、T、E（由 C compile 之前 deepcopy 的獨立 instance，head／block compile 皆關）；以 anchor index 為鍵整體替換 84 維向量；以 oracle 的 compiled `_postprocess_mamba_fixed` 解碼、乘 `sx/sy`，回傳 `(1, 300, 6)`）。唯一改變：hybrid 的替換集合由 Δ(f) 改為 S(f)。

| arm | anchor a ∈ S(f) | a ∉ S(f) | 意義 |
|:--|:--|:--|:--|
| `R_C` | C | C | 純 C（replay 系統內的 oracle corner） |
| `R_T` | T | T | 純 T |
| `H_M` | **T** | C | 替換集合取 T 的完整 logits（membership／class 等於 T），其餘取 C |
| `H_V` | **C** | T | 替換集合固定取 C，其餘取 T 的數值（membership／class 等於 C） |
| `R_E` | E | E | 純 E（eager head 數值＋compiled S2；S 不參與） |

每個 arm 跑兩次；順序 `R_C#1, R_T#1, H_M#1, H_V#1, R_E#1, R_C#2, R_T#2, H_M#2, H_V#2, R_E#2`，同一 session、同一 `machine-bench` 租約。

**受測 T**：只用 PR-1R engine（engine `c77148a8…`，manifest 檔案 sha `a015bce0…`），與 r1 相同。不 build 任何新 engine。

## 4. Validity

| gate | 條件 | 不成立 ⇒ |
|:--|:--|:--|
| V1 凍結輸入 | 與 r1 V1 相同的凍結輸入（engine／manifest 檔案 sha256／ONNX／ckpt／backbone／preset／環境／資料 5316 frames／clean tree／caller 無 `SACCADE_*`／租約直接 child；runner hard-bound，識別 study 的輸入都不是 CLI 選項）；**declaration blob** = runner 內釘的本文凍結 blob；**執行凍結點** = r2 runner PR 的 merge commit，由 annotated tag **`freeze/465-head-localization-r2`** 標示（不得重用 r1 的 `freeze/465-head-localization`）。runner 檢查：HEAD 是 2-parent commit、在 `origin/main` 的 first-parent 鏈上；`git cat-file -t refs/tags/freeze/465-head-localization-r2` = `tag`；本地 `…^{commit}` = HEAD；`git ls-remote origin refs/tags/freeze/465-head-localization-r2^{}` = HEAD；比較一律用 commit SHA。packet 根目錄 = `results/native_head_failure_localization_465_r2/`；runner 不寫入 r1 的 packet 根目錄 | `UNRESOLVED` |
| V2 確定性 | 每個 arm 兩次 run 的 7 個 txt 逐位元相同 | `UNRESOLVED` |
| V3 組合正確 | 每個 frame、每個 hybrid arm，in-S 來源記為 X（`H_M`：X = T；`H_V`：X = C）：(i) 組合後的 84 維 logits 逐 anchor 逐位元等於指定來源（S 內等於 X、S 外等於另一方），且 per-level split 可逐位元還原；(ii) 由組合後 logits 重算 membership／class：令 `Q_X` = X 的 cutoff 平手類 `{a : s_X(a) 與 s_X^(300) 逐位元相同 ∧ s_X(a) ≥ 0.05}`，則 `Q_X` 之外的每個 anchor 的 membership 等於 `m_X`，`Q_X` 內的 member 個數等於 X 的，組合後或 X 的任一 member 的 class 等於 X 的 class。依 §2.1，(ii) 在正確實作下恆成立；任一 frame 不成立 = 實作錯誤，runner 立即 fail-closed | `UNRESOLVED` |
| V4 現象重現 | `R_T` 對 `R_C` 依 §5 κ 為「出界」 | `UNRESOLVED` |

只報告：`R_C` 的 txt 是否與 PR-2／PR-2R 的 `A_C` byte-identical；`R_T` 是否與 PR-2R 的 `A_T` byte-identical（不列為 gate）。V4 不涉及 `R_E`。

## 5. 判定

**出界（沿用 PR-2 的 κ_L2，容差固定為 floor，不設 reference arm；與 r1 相同）**：arm X 對 `R_C` 出界 ⇔ 7-seq combined、未四捨五入，`|Δ IDF1| > 0.20` 或 `|Δ HOTA| > 0.20` 或 `|Δ MOTA| > 0.20` 或 `|Δ IDs| > 5`（雙向）。

- `E_out` = `R_E` 出界
- `S_suff` = `H_M` 出界：替換集合 S 取 T 的完整 logits 就足以造成 parity failure。**它把 divergence 定位到 S 通道（Δ 加上 cutoff 帶），不把 membership 當成獨立的因果變數。** 截斷 frame 上 S 可能包含兩個 head membership 相同的 cutoff 帶 anchor，它們的數值也一起換成 T。
- `V_suff` = `H_V` 出界：S 固定取 C、其餘 anchor 取 T，且 V3 確認 membership／class 與 C 相同時，仍足以造成 parity failure

**Decision terminal（窮盡，依序；只由 validity 與 `E_out` 決定；與 r1 相同）**：

| # | terminal | 條件 | 主線轉移 |
|:--|:--|:--|:--|
| 1 | `UNRESOLVED` | V1–V4 任一不成立，或任何執行錯誤、缺 packet | 無；不重跑，重新宣告由 owner 決定 |
| 2 | `EAGER_NUMERICS_WITHIN` | 非 `E_out` | 下一個候選 head 形式**優先 LibTorch**（與 oracle 同為 PyTorch kernel）；需自己的 PR-1 與 parity 宣告，本 terminal 不接受任何形式 |
| 3 | `EAGER_NUMERICS_OUT` | `E_out` | 連 eager 級的非 bit-exact head 都出界 ⇒ LibTorch 不因數值接近而優先；下一步由 owner 決定接受 named limit（parity 改以 `A_T` 類組態為 oracle）或要求 bit-exact 形式 |

**機制標籤（與 decision terminal 一起報告，不改變 decision terminal，也不直接選 S2 artifact 或 LibTorch）**：只在 V1–V4 全部成立時給出。

| 標籤 | 條件 |
|:--|:--|
| `SWAP_SET_SUFFICIENT` | `S_suff` 且非 `V_suff` |
| `COMMON_ANCHOR_VALUES_SUFFICIENT` | `V_suff` 且非 `S_suff` |
| `MIXED` | `S_suff` 且 `V_suff`；或兩者皆否（交互作用） |

兩個 sufficient 標籤都指向「head 數值要更接近 oracle」，差別只在需要多接近；它們是機制 evidence，不是路線。`SWAP_SET_SUFFICIENT` 不得寫成「membership 邊界是原因」或「S2 邊界是原因」。

## 6. 只報告、不判定（7 個序列全部報）

與 r1 §6 相同的四類 evidence（crossing anchor 的去向、top-k 第一次不同、tracker input 第一次不同、輸出第一次不同能否回溯），另加：

- 每個 frame 的 `θ_C`、`θ_T`、兩個 head 是否在 0.05 之上截斷、|Δ(f)|、|B(f) \ Δ(f)|、|S(f)|；同樣的 census 對 (E, C) 報告（只報告，不參與組合）。
- detection row 的通道歸屬同時以 Δ 與 S 兩種旗標記錄；平手歧義（同 score bits、同 class、旗標相反，於 top-k 截斷前以全部 8400 個 anchor 判定）對 Δ、S、Δ_E 分別記錄，歧義列不做確定歸因。

row→anchor 的對應由 replay detector 在組合時記錄（eager top-k 索引，與輸出 row 做一致性比對；不一致率一併報告）。

## 7. Packet

`results/native_head_failure_localization_465_r2/<UTC>/`，以 `open_run` claim，附 manifest：本文與 runner blob、freeze commit／tag、V1 各項實測、10 個 run 的完整輸出與 stdout、每 frame 的 V3 檢查結果、每 frame 的 Δ／B／S census、stage probe 摘要、metric counts 與未四捨五入值、判定過程與 terminal。任何一個 run 失敗即停止其餘 run（與 r1 相同），packet 仍寫出。

## 8. 本文刻意沒有做的事

- 沒有跑任何量測；沒有讀 r1 正式 run 的 txt、evidence 或 metric；沒有 build 任何 engine。
- 沒有依 PR-2／PR-2R 的偏差大小設定任何門檻；沒有修改 arm、容差、terminal 規則或主線轉移。
- 沒有修改 r1 宣告、r1 runner 或 r1 的 packet。

## 9. 與 r1 的關係與後續

- r1 宣告（blob `38f15007`）、r1 runner（blob `3b88ff3c`）、tag `freeze/465-head-localization` 與 r1 packet 維持原狀；r1 的 terminal 是 `UNRESOLVED`，永久有效。
- r2 runner 另開 PR，在本文凍結之後才 merge：hard-bind 本文 blob、輸入、輸出與 terminal；共用的程式與 r1 runner 逐字相同的部分以測試釘住（只允許本文列出的差異）。runner PR 只實作並驗證 runner（unit test、不讀任何 MOT17 frame 的結構檢查）。
- 正式量測只在 runner merge、tag `freeze/465-head-localization-r2` 建立、#465 freeze record 貼出之後，從該 SHA 執行一次。
- #465 Phase B 的 PR-3 以後維持暫停，直到本 study 得出 `EAGER_NUMERICS_WITHIN` 或 `EAGER_NUMERICS_OUT`。

## 10. Owner 決定

- 2026-09-28：r1 `UNRESOLVED` 之後重新宣告，不重跑 r1；先不讀正式資料驗證 cutoff 結構問題，確認後修正替換集合與 V3，保持容差、研究問題與「機制標籤不直接選 LibTorch／S2」的限制。

## 11. Review 修訂與 amendments

**凍結前的 review 修訂**：（無）

**凍結後的 amendments**（append-only）：（無）
