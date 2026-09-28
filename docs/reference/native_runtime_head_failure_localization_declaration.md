# Native runtime head failure localization — 預宣告（#465 Phase B redesign，localization study）

> 狀態：**草稿，未凍結。** §10 的兩個 owner 決定已寫入（2026-09-28：O1 採 (a)，O2 只用 PR-1R engine），待 review 後 merge。**declaration 凍結點** = 本文所在 PR merge；之後只能以 §11 append-only amendment 修訂。**執行凍結點**另見 §4 V1。
> 權威 seal bar：[experiment contract §20.8](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（引用，不複述）。
> 前置結果：[PR-2](native_runtime_head_parity_result.md)（TF32 allowed）與 [PR-2R](native_runtime_head_parity_tf32_off_result.md)（TF32 disabled）皆 `HEAD_PARITY_OUT_OF_TOLERANCE`；#465 記錄的 owner 路線：停止 TRT head-only 的 builder 調整，先做本 study。

---

## 0. 這個 study 回答什麼

> TRT head 與 oracle head 之間的小幅 tensor 差異，是在哪一個離散邊界被放大成 MOT association divergence？

| | 狀態 |
|:--|:--|
| divergence 是否由 **anchor membership**（0.05 floor、top-300、argmax class）的差異造成 | **回答**（§5 factorial，機制標籤） |
| divergence 是否由**同一組 member 的數值**（box、score）差異造成 | **回答**（§5 factorial，機制標籤） |
| 一個與 oracle 同為 PyTorch kernel、但不 bit-exact 的 head（E：eager head）在 replay 系統中是否在容差內 | **回答**（§5 decision terminal；決定 LibTorch 是否優先） |
| 每個 crossing anchor、top-k、tracker input、association 的第一個分歧 | **報告，不判定**（§6） |
| TF32 與 tactic selection 的效應分離 | **不回答**（兩個 engine 各一次 build，現有設計無法分離） |
| 任何 head 形式是否可接受 | **不回答**；本 study 不是第三次 parity trial，不產生 parity terminal。`R_E` 在容差內**不等於** LibTorch 形式被接受：LibTorch artifact 仍須自己的 PR-1 與 parity 宣告 |
| FPS／latency、benchmark claim | 不回答、不改 |

target decision layer = `none (cross-layer substrate work)`；κ 見 §5。

## 1. 宣告前已看過的資料

全部是 archived reference，**不參與**任何門檻設定（§5 的門檻沿用 PR-2 宣告在看到任何結果之前凍結的 floor）：PR-1 冒煙、#473 smoke、PR-2 與 PR-2R 正式 run 的全部結果（含 per-sequence、crossing 數、A_C／A_N 在兩個 study 間 byte-identical）、PR-1R 零輸入結構檢查。
本文撰寫時**沒有**執行任何新的 frame、replay 或 instrumentation 量測。

## 2. 被分解的系統

whole-graph detect 在 headline 設定下是**當前 frame 的純函數**：`_whole_graph_fn` 只讀 frame 與每序列固定的 `_whole_graph_sx/sy`；GMC warp 只進入 eager temporal 路徑，tracker 狀態不進入 detector（`mamba_gated_detector.py` `_whole_graph_fn`、`set_gmc_warp`；`pipeline._double_buffer_eligible`）。因此 C 與 T 的 detector 輸出可以在同一 frame 上逐 anchor 對齊比較，差異在 detector 出口之前沒有閉迴路。

detector 出口之後、association 之前第一個讀 tracker 狀態的步驟是 private continuation 的 active-track priors（`stages.py` `_run_native_tensor_prep`／`_run_nms`）；main NMS、score floor、filter 都是 frame-only。

anchor a 的 **membership**（對 head X）：
`m_X(a) = [ s_X(a) ≥ 0.05 ] ∧ [ rank_X(a) ≤ 300 ]`，其中 `s_X(a)` = 80 類 sigmoid 的最大值，`rank_X` = 同 frame 依 `s_X` 的 top-k 名次（`max_det = 300`，`conf_thr = 0.001`，small-P3 fusion 關）；member 的 class = argmax。
**Δ(f)**（membership 差異集合）= `{a : m_C(a) ≠ m_T(a)} ∪ {a : m_C(a) = m_T(a) = 1 ∧ class_C(a) ≠ class_T(a)}`。
0.05 = headline 的 `base_score_floor`；0.05 以下的 detection 不會到達 tracker。0.05 之上的其他門檻（private `min_score` 0.10、`new_track_thresh` 0.28、`confirm_score_thresh` 0.50、NMS IoU、match）屬於 **values** 通道：數值差異造成的這些翻轉，本文計入 values，不計入 membership。

## 3. 實作與 arms

**Replay detector（runner 端，不改 harness）**：runner 以 unmodified `scripts/eval/mot17.py`（`--preset mamba_whole_graph --detector SDP --double-buffer`，§2 序列）執行，只把 harness 建出的 detector 的 `detect_raw` 換成組合函數。detector 物件本身（tracker 擁有權、`use_whole_graph`、`_trt_backbone`、`set_whole_graph_img_dims`）不變，所以 frame 解碼、ingest、NMS 與 tracker 路徑全部是 harness 的。每個 frame：

1. `F.interpolate(frame, 640, bilinear, align_corners=False)` → 同一個 TRT backbone → p3/p4/p5；
2. 同一份特徵同時餵 C（compiled head＋block，oracle）、T 與 E（`_forward_eager`，由 C compile 之前 deepcopy 的獨立 instance，head／block compile 皆關；與 PR-2 runner 的 E 同一建法），得到三組 cls／reg logits；三個 head 每個 frame 都執行，與 arm 無關；
3. 依 arm 在 anchor 層組合 logits（下表）。head 輸出是 `cls_p3..p5`（80 通道）與 `reg_p3..p5`（4 通道，reg_max = 1，無 DFL），anchor index = P3／P4／P5 依序 flatten 後的 8400 個位置（與 `_postprocess_mamba_fixed_eager` 的 concat 順序相同）。組合以 anchor index 為鍵，把整個 84 維向量（80 cls＋4 reg）整體替換；不混用同一 anchor 的 cls 與 reg 來源；
4. 以 oracle 的 `_postprocess_mamba_fixed`（compiled，與 `A_C` 相同）解碼，乘 `sx/sy`，回傳 `(1, 300, 6)`。

| arm | anchor a ∈ Δ(f) | a ∉ Δ(f) | 意義 |
|:--|:--|:--|:--|
| `R_C` | C | C | 純 C（replay 系統內的 oracle corner） |
| `R_T` | T | T | 純 T |
| `H_M` | **T** | C | membership 取 T、其餘數值取 C |
| `H_V` | **C** | T | membership 取 C、其餘數值取 T |
| `R_E` | E | E | 純 E（eager head 數值＋compiled S2；Δ 不參與） |

每個 arm 跑兩次；順序 `R_C#1, R_T#1, H_M#1, H_V#1, R_E#1, R_C#2, R_T#2, H_M#2, H_V#2, R_E#2`，同一 session、同一 `machine-bench` 租約。

**受測 T**：只用 PR-1R engine（engine `c77148a8…`，manifest 檔案 sha `a015bce0…`；§10 O2）。PR-1 engine 不參與。不 build 任何新 engine。

## 4. Validity

| gate | 條件 | 不成立 ⇒ |
|:--|:--|:--|
| V1 凍結輸入 | engine／manifest（檔案 sha256）／ONNX／ckpt／backbone／preset／環境／資料 5316 frames／clean tree／caller 無 `SACCADE_*`／租約直接 child；runner hard-bound（識別 study 的輸入都不是 CLI 選項）；**declaration blob**：HEAD 上本文的 blob 等於 runner 內釘的凍結 blob；**執行凍結點**：runner 另開 PR（§9），所以正式 HEAD 是 **runner PR 的 merge commit**，不是本文的 merge commit。runner merge 後建立 annotated tag `freeze/465-head-localization` 指向該 merge commit、push，並在 #465 貼 freeze record（SHA＋tag）；runner 檢查 HEAD 是 2-parent commit、在 `origin/main` 的 first-parent 鏈上、等於本地 tag 的 target，且 `git ls-remote origin` 回報的 tag target 相同（採納 #475 review 的 hardening note：tag 指向 exact SHA，不只靠 merge subject） | `UNRESOLVED` |
| V2 確定性 | 每個 arm 兩次 run 的 7 個 txt 逐位元相同 | `UNRESOLVED` |
| V3 組合正確 | 每個 frame、每個 hybrid arm：(i) 組合後的 84 維 logits 逐 anchor 逐位元等於表中指定的來源（Δ 內等於一方、Δ 外等於另一方）；(ii) 由組合後 logits 重新計算 membership 與 class：`H_M` 的 m 與 class 必須逐 anchor 等於 T，`H_V` 必須等於 C（top-300 名次會因替換而移動，所以要在組合後重算，不是假設）。任一 frame 不成立，runner 立即 fail-closed | `UNRESOLVED` |
| V4 現象重現 | `R_T` 對 `R_C` 依 §5 κ 為「出界」。replay 系統若不重現 parity failure，就沒有可分解的對象 | `UNRESOLVED` |

只報告：`R_C` 的 txt 是否與 PR-2／PR-2R 的 `A_C` byte-identical；`R_T` 是否與 PR-2R 的 `A_T` byte-identical（replay 在 CUDA graph 之外執行，bitwise 相同不是先驗保證，所以不列為 gate；V4 才是 gate）。V4 不涉及 `R_E`：`R_E` 的結果不影響 validity。

## 5. 判定

**出界（沿用 PR-2 的 κ_L2，容差固定為 floor，不設 reference arm）**：arm X 對 `R_C` 出界 ⇔ 7-seq combined、未四捨五入，`|Δ IDF1| > 0.20` 或 `|Δ HOTA| > 0.20` 或 `|Δ MOTA| > 0.20` 或 `|Δ IDs| > 5`（雙向）。floor 在 PR-2 宣告凍結時就已固定，不依任何已看過的結果調整。

- `E_out` = `R_E` 出界（eager 級的非 bit-exact head 數值就足以造成 parity failure）
- `M_suff` = `H_M` 出界（只換 membership 就足以造成 parity failure）
- `V_suff` = `H_V` 出界（membership 不變、只換數值就足以造成 parity failure）

**Decision terminal（窮盡，依序；只由 validity 與 `E_out` 決定）**：

| # | terminal | 條件 | 主線轉移 |
|:--|:--|:--|:--|
| 1 | `UNRESOLVED` | V1–V4 任一不成立，或任何執行錯誤、缺 packet | 無；不重跑，重新宣告由 owner 決定 |
| 2 | `EAGER_NUMERICS_WITHIN` | 非 `E_out` | 下一個候選 head 形式**優先 LibTorch**（與 oracle 同為 PyTorch kernel）；需自己的 PR-1 與 parity 宣告，本 terminal 不接受任何形式 |
| 3 | `EAGER_NUMERICS_OUT` | `E_out` | 在此 replay 系統中，連 eager 級的非 bit-exact head 都出界 ⇒ LibTorch 不因數值接近而優先；下一步由 owner 決定接受 named limit（parity 改以 `A_T` 類組態為 oracle）或要求 bit-exact 形式 |

**機制標籤（與 decision terminal 一起報告，不改變 decision terminal，也不直接選 S2 artifact 或 LibTorch）**：只在 V1–V4 全部成立時給出。

| 標籤 | 條件 |
|:--|:--|
| `S2_BOUNDARY_LOCALIZED` | `M_suff` 且非 `V_suff` |
| `HEAD_NUMERICS_LOCALIZED` | `V_suff` 且非 `M_suff` |
| `MIXED` | `M_suff` 且 `V_suff`；或兩者皆否（`R_T` 出界但任一單一通道都不足以出界 = 交互作用） |

依 §10 O1 的分析，兩個 localized 標籤都指向「head 數值要更接近 oracle」，差別只在需要多接近；它們是機制 evidence，不是路線。

## 6. 只報告、不判定（owner 要求的四類 evidence，7 個序列全部報）

1. **crossing anchor 的去向**：每個 frame 的 |Δ(f)|、floor crossing 數、class flip 數、top-300 是否在 0.05 之上截斷；同樣的 census 也對 (E, C) 報告（Δ_E，只報告，不參與組合）；`R_T` run 中，Δ anchor 產生的 detection 是否出現在 `tracker_input`、是否被某個輸出 track 的 `det_idx` 引用。
2. **top-k 第一次不同**：每序列 C 與 T 的 member 集合、class、排序第一次不同的 frame。
3. **tracker input 第一次不同**：`R_C` 對 `R_T`（以及對 `H_M`、`H_V`、`R_E`）在 harness 的 `stage_probe_callback` `tracker_input` 階段第一次不同的 frame，及造成差異的 detection 屬於 Δ anchor 或數值差異。
4. **輸出第一次不同能否回溯**：每序列 txt 第一次分歧的 frame，是否在其當下或之前有 (3) 的 tracker-input 差異，以及對應的 `det_idx`。

row→anchor 的對應由 replay detector 在組合時記錄（eager top-k 索引，與輸出 row 做一致性比對；不一致率一併報告）。MOT17-04／10／13 可以重點敘述，但所有表格都包含 7 個序列。

## 7. Packet

非 scratch 的 timestamped 目錄，以 `open_run` claim，附 manifest：本文與 runner blob、freeze commit／tag、V1 各項實測、10 個 run 的完整輸出與 stdout、每 frame 的 V3 檢查結果、每 frame 的 Δ census、stage probe 摘要、metric counts 與未四捨五入值、判定過程與 terminal。

## 8. 本文刻意沒有做的事

- 沒有跑任何量測；沒有 build 任何 engine；沒有重跑任何 TF32 builder variant。
- 沒有依 PR-2／PR-2R 的偏差大小設定任何門檻。
- 不回答 TF32 與 tactic 的分離，不產生任何 parity terminal。

## 9. 相關但不在本文範圍

- runner 另開 PR（hard-bind 本文 blob、輸入、輸出、terminal），在本文凍結之後才寫。runner PR 只實作並驗證 runner（unit test、不讀任何 MOT17 frame 的結構檢查）；正式 frame 量測只在 runner merge、tag 建立、#465 freeze record 貼出之後，從該 SHA 執行一次。

## 10. Owner 決定（凍結前）

**已決定（2026-09-28）**：O1 採 **(a)**，已寫入 §0／§3／§5；O2 = **只用 PR-1R engine**（§3）。以下保留決定前的分析作為紀錄。

**O1 — terminal → 路線的對應。** 草稿依 owner 提議的名稱寫了 terminal，但「`S2_BOUNDARY_LOCALIZED` ⇒ 下一個候選是含 S2 的 artifact」這個對應在因果上不成立，需要 owner 在凍結前重新決定：

- `A_T` 的 S2 已經是 oracle 的同一份程式（compiled `_postprocess_mamba_fixed`）；membership 翻轉來自 T 的 raw head 數值落在 0.05／top-k／argmax 邊界的另一側。把 S2 移進 TRT artifact，head 數值不變，翻轉仍在。
- 所以 `S2_BOUNDARY_LOCALIZED` 與 `HEAD_NUMERICS_LOCALIZED` 都指向「head 數值要更接近 oracle」，差別只在「需要多接近」（只在邊界附近 vs 全面）。兩者都不區分 S2 artifact 與 LibTorch。
- 與 S2 vs LibTorch 真正相關的問題是：**一個與 oracle 同為 PyTorch kernel、但不是 bit-exact 的 head（例如 E：eager head）是否在容差內？** LibTorch 形式的數值會接近 E 而不是 T。已看過的資料裡，E 對 C 的 crossing 只有 5 個（T 為 93／160），`A_N`（eager head＋eager postprocess）在兩個 study 的 IDF1／MOTA／IDs 都與 `A_C` 相同；但這是事後觀察，`A_N` 同時關掉 postprocess compile，不能直接當 LibTorch 的證據。
- 建議的修改（擇一，需 owner 決定）：
  - (a) 在 §3 加一個 corner `R_E`（E head＋compiled S2，其餘同 `R_C`），並把決策 terminal 改成「E 級數值誤差是否在容差內」：在容差內 ⇒ 優先 LibTorch；出界 ⇒ 任何非 bit-exact 的 head 都會出界，下一步是 owner 決定是否接受 named limit（parity 改以 `A_T` 類組態為 oracle）或要求 bit-exact 形式。本文的 factorial 保留為機制證據。
  - (b) 維持 factorial 為唯一判定，terminal 只記錄機制，不直接選 S2／LibTorch；選擇另開 study。

**O2 — 受測 T。** 草稿預設 PR-1R engine（最近一次、偏差較大、訊號較強）。選項：只用 PR-1 engine；或兩者都用、要求分類一致（不一致 ⇒ `MIXED`）。兩者都是既有 build，不涉及新的 builder variant；但 owner 已指示本輪不重跑 TF32 variant，所以預設只用一個。

## 11. Review 修訂與 amendments

**凍結前的 review 修訂**：

- R1（2026-09-28，owner 決定 O1(a)／O2）：加 `R_E` arm；decision terminal 改為 `EAGER_NUMERICS_WITHIN`／`EAGER_NUMERICS_OUT`，原三個 localized terminal 降為機制標籤；受測 T 定為 PR-1R engine。門檻（PR-2 floor）與 `R_C`／`R_T`／`H_M`／`H_V` 四個 arm 的定義不變。
- R2（2026-09-28）：V3 明確寫出以 anchor index 整體替換 84 維 logits 並逐位元驗證來源，組合後重算 membership／class，fail-closed。
- R3（2026-09-28）：執行凍結點改為 runner PR 的 merge commit＋annotated tag（runner 依 §9 另開 PR，本文的 merge commit 不會包含 runner）。

**凍結後的 amendments**（append-only）：（無）
