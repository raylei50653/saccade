# Native runtime head failure localization r2 — 結果（#465 Phase B redesign，localization study）

> 狀態：**decision terminal = `EAGER_NUMERICS_WITHIN`；機制標籤 = `COMMON_ANCHOR_VALUES_SUFFICIENT`**（正式 run，一次，valid）。
> 依據：[r2 預宣告](native_runtime_head_failure_localization_r2_declaration.md)（blob `71b5bb0702aee22e938500be67e682e2bac36c58`，§11 無凍結前修訂、無凍結後 amendment）。r1（[宣告](native_runtime_head_failure_localization_declaration.md)，packet `results/native_head_failure_localization_465/20260928T070925Z/`）的 terminal `UNRESOLVED` 永久保留，本文不引用 r1 的任何量測。
> 本文只寫 terminal、機制標籤與支撐它們的數字（宣告 §5–§7）；§6 的內容是報告，不是判定；不讀任何時間量、不改 benchmark claim。

---

## 1. 這次 run

| 項目 | 值 |
|:--|:--|
| declaration freeze | #479 merge `a429944a23cbc2dcd88bab7816717d1a71f13086` |
| execution freeze | #480 merge `77ba233a118c259d4c177fcde2785842d34b8ab2`（parents `a429944a` + `d49ccbac`，在 `origin/main` first-parent 鏈上）；annotated tag `freeze/465-head-localization-r2`，tag object `2376525d74dc30d71877aa6365579e054ec84f8f`，peeled（本地與 `origin`）= `77ba233a`；freeze record 已貼在 #465 |
| runner | `scripts/eval/diagnostics/native_head_failure_localization_r2.py`，blob `47e737669cefafb1b51fb16215163b716e7e6523` |
| 執行 | detached checkout `77ba233a`，clean tree；正式 run 前先在 `machine-bench` 租約下呼叫 `check_v1()` 做不寫 packet 的 V1 dry check（21/21）；正式 run 為 `tools/resctl.py run machine-bench` 的直接 child，2026-09-28 08:10:21–08:38:26Z，開始與結束為同一份租約 |
| packet | `results/native_head_failure_localization_465_r2/20260928T081021Z/`（gitignored，本機保留）；`packet.json` sha256 `207fd737…`，`MANIFEST.json` sha256 `b063eee8…` |
| 事前看過的資料 | 宣告 §1 所列；r2 之前沒有任何 r2 frame 輸出 |

## 2. Validity

| gate | 結果 |
|:--|:--|
| V1 | 21/21：engine `c77148a8…`／lineage manifest `a015bce0…`／ONNX `6e919dad…`／ckpt／backbone／preset／環境／7 序列 5316 frames／clean tree／caller 無 `SACCADE_*`／宣告 blob／runner committed／HEAD = tag peeled commit／租約直接 child |
| V2 | 5 個 arm 各自兩次 run 的 7 個 txt 逐位元相同；每 frame 的 C/T/E census 在全部 10 個 run 間相同 |
| V3 | `H_M`、`H_V` 的每一個 frame（兩次 run、7 序列）通過：S 內／外逐位元等於指定來源、per-level split 還原、tie 類之外 membership 等於來源、tie 類內 member 數相同、member class 相同。MOT17-04 的 1050 個 frame 在 C 與 T 都是 top-300 在 0.05 之上截斷（r1 在此序列 fail-closed 的 frame 類別） |
| V4 | `R_T` 對 `R_C` 出界（下表） |

只報告：`R_C` 的 7 個 txt 與 PR-2、PR-2R 的 `A_C` 逐位元相同；`R_T` 與 PR-2R 的 `A_T` 逐位元相同。replay 系統重現了 PR-2R 的 oracle 與 parity failure 本身，不只是 metric 相近。

## 3. §5 判定

7-seq combined、未四捨五入，對 `R_C`（IDF1 78.2899、HOTA 69.9589、MOTA 77.8801、IDs 429）；floor = 0.20／0.20／0.20／5：

| arm | Δ IDF1 | Δ HOTA | Δ MOTA | Δ IDs | 判定 |
|:--|--:|--:|--:|--:|:--|
| `R_T` | **−0.933** | **−0.593** | −0.050 | **+21** | 出界 ⇒ V4 成立 |
| `H_M`（S 取 T，其餘 C） | 0.000 | +0.00004 | 0.000 | 0 | 容差內 ⇒ `S_suff` = false |
| `H_V`（S 取 C，其餘 T） | **−0.933** | **−0.593** | −0.050 | **+21** | 出界 ⇒ `V_suff` = true |
| `R_E`（eager head） | 0.000 | +0.0005 | 0.000 | 0 | 容差內 ⇒ `E_out` = false |

- decision terminal（§5，只由 validity 與 `E_out` 決定）：非 `E_out` ⇒ **`EAGER_NUMERICS_WITHIN`**。
- 機制標籤：`V_suff` 且非 `S_suff` ⇒ **`COMMON_ANCHOR_VALUES_SUFFICIENT`**。

只報告、不判定：

| | R_C | R_T | H_M | H_V | R_E |
|:--|--:|--:|--:|--:|--:|
| DetA | 70.221 | 70.263 | 70.221 | 70.263 | 70.221 |
| AssA | 69.910 | 68.696 | 69.910 | 68.696 | 69.911 |
| FP | 3471 | 3522 | 3471 | 3522 | 3471 |
| FN | 20940 | 20924 | 20940 | 20924 | 20940 |

per-sequence Δ（IDF1／IDs，對 `R_C`）：`R_T` 與 `H_V` 相同 —— 02 +0.18／−2、04 −1.16／+3、05 −0.06／+1、09 −1.00／+3、10 −1.95／+9、11 −0.03／+1、13 −1.49／+6；`H_M` 與 `R_E` 在 7 個序列都是 0.00／0。

txt 層級：`H_M` 與 `R_C` 在 6／7 個序列逐位元相同（只有 MOT17-04 不同，第一個分歧在 frame 157，metric 差異僅 HOTA +4e-5）；`H_V` 與 `R_T` 在 6／7 個序列逐位元相同（只有 MOT17-04 不同）；`R_E` 的 7 個 txt 都與 `R_C` 不同（第一個分歧在 frame 3–5），但 IDF1／MOTA／IDs 與 `R_C` 相同。

## 4. §6 報告（不判定）

**census（C vs T，全部 frame 加總）**：

| 序列 | \|Δ\| | floor crossing | class flip | \|B \ Δ\| | \|S\| | 截斷 frame（C／T） |
|:--|--:|--:|--:|--:|--:|--:|
| 02 | 19 | 19 | 0 | 0 | 19 | 0／0 |
| 04 | 196 | 52 | 0 | 1210 | 1406 | 1050／1050 |
| 05 | 10 | 10 | 0 | 0 | 10 | 0／0 |
| 09 | 11 | 11 | 0 | 0 | 11 | 0／0 |
| 10 | 22 | 22 | 0 | 0 | 22 | 0／0 |
| 11 | 18 | 18 | 0 | 0 | 18 | 0／0 |
| 13 | 28 | 28 | 0 | 0 | 28 | 0／0 |

截斷只發生在 MOT17-04，而且是每一個 frame；其他序列的 S 等於 Δ（全部是 floor crossing，沒有 class flip）。(E, C) 的 |Δ_E| 全部 7 序列合計 5，MOT17-04 為 0。

**第一個分歧**：C 與 T 的 member 順序在每個序列的前 1–4 frame 內就不同，member 集合第一次不同在 frame 4–71。`R_T`、`H_V`、`R_E` 的 `tracker_input` 在每個序列的 frame 1 就與 `R_C` 不同，txt 在 frame 3–5 第一次分歧，且都在 tracker-input 分歧之後。`H_M` 的 `tracker_input` 在 02／04／09／11／13 晚很多才分歧（frame 6–511），05 與 10 完全不分歧；只有 04 的 txt 分歧。

**通道歸屬的限制**：各序列 `R_C` 中 0.05 以上的 row 有 20–27% 無法通過 row→anchor 一致性檢查（eager 重算的 top-k 與 compiled S2 輸出的 score bits 不同，`inconsistent`），`R_T` 的 `tracker_input` row 有約 9% 無法以 full-row 對回 replay row（`unmapped`）。因此第一個 tracker-input 分歧的解釋幾乎全部是 `undetermined`；以 S 為鍵，`R_T` 的輸出 track 所引用的 detection 中，可歸屬的 69,985 個裡有 14 個來自 S anchor、69,971 個來自 S 之外（另有 17,047 `inconsistent`、2,055 `unmapped`）。這些數字與機制標籤方向一致，但 row 層級的歸屬本身不支持任何更強的結論。

## 5. Terminal 與主線轉移

**`EAGER_NUMERICS_WITHIN`**（宣告 §5 terminal 2）：

- 下一個候選 head 形式**優先 LibTorch**（與 oracle 同為 PyTorch kernel）。它需要自己的 PR-1（artifact）與 parity 宣告；本 terminal **不接受任何形式**，`R_E` 在容差內不等於 LibTorch artifact 在容差內。
- 機制標籤 `COMMON_ANCHOR_VALUES_SUFFICIENT` 是機制證據，不是路線：在這個 replay 系統中，T 在 membership 邊界（Δ 與 cutoff 帶）上的替換單獨不足以出界，而 S 之外的共同 anchor 取 T 的數值就足以重現 `R_T` 的全部偏差。它不指定 S2 artifact，也不說明 membership 邊界是原因。
- #465 維持 `requires_runtime_redesign`；PR-3 以後依 LibTorch PR-1 → parity 宣告的路線恢復。headline 數字不變。
