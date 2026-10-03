# Native runtime head parity (LibTorch) — 結果（#465 Phase B PR-2L／U1 redesign）

> 狀態：**terminal = `HEAD_PARITY_WITHIN_TOLERANCE`**（正式 run，一次，valid）；**owner 決定 `ACCEPT`**（#465 issuecomment-5967756337，2026-10-03）。
> 依據：[PR-2L 預宣告](native_runtime_head_parity_libtorch_declaration.md)（blob `361639bd0e565e8f40031720c3b8ccf93189913d`，含 §11 A1）；受測 artifact：[PR-1L](native_runtime_head_artifact_libtorch.md)。
> 本文是 archival 紀錄：只寫 terminal、支撐它的數字與 owner 決定；不做歸因、不讀任何時間量、不改 benchmark claim，也不改變該決定。

---

## 1. 這次 run

| 項目 | 值 |
|:--|:--|
| declaration freeze | #485 merge `883c16a8`；§11 A1 amendment 由 #486 merge `9715e3b1` 加入，blob `361639bd` |
| execution freeze | #487 merge commit `504ab0e256e44dc3606db1b4cca57b0484601234`；annotated tag `freeze/465-pr2l-libtorch-parity`（tag object `21591172`，peeled == `504ab0e2`）；freeze record = #465 issuecomment-5873025204 |
| runner | `scripts/eval/diagnostics/native_head_parity_libtorch.py`，blob `437bb97e116cf9af6597f7431ecd850d3053a89e` |
| CI | `504ab0e2` 的 push CI 6/6 success；`runtime_identity.yml`（宣告 §2 要求）run `36441427709` success |
| 租約 | `tools/resctl.py run machine-bench`，runner 為租約 owner 的直接 child，開始與結束為同一份租約 |
| packet | `results/native_head_parity_465_libtorch/20260928T152911Z/`（gitignored，本機保留）；`packet.json` sha256 `e614f517…addc`，`MANIFEST.json` sha256 `9352178b…187f`；2026-09-28 15:29:11Z → 15:35:21Z |
| 事前看過的資料 | 宣告所列；structural check 只用合成 frame；正式 run 之前沒有任何 MOT17 frame 經過 L |

## 2. Validity

| gate | 結果 |
|:--|:--|
| V1（§2 凍結輸入，39 項） | 全部通過：artifact 檔案 sha256 `1663ec97…`、content sha256 `f6a540ed…`、operator library sha256 `cfea782f…`、ckpt／backbone／preset sha256、driver 616.92／`cuDriverGetVersion` 13040／`cudaRuntimeGetVersion` 13000、單一 libcudart、HEAD 為 freeze commit、tag peeled、clean tree、caller 無 `SACCADE_*` |
| V2（L1 重跑） | 140 frames，全部相同；輸入沒有被改動；JIT scan fallback 呼叫 0 次 |
| V3（L2 兩次 run） | `A_C`、`A_L`、`A_N` 各自兩次的 7 個 txt 逐位元相同；6 個 run exit 0 |
| V4（oracle anchor） | `A_C#1` 的 7 個 txt == PR-2R packet `A_C_1`，7/7 |
| V5（注入 sidecar） | `A_L` 兩次都通過：slot 已安裝、adapter 被呼叫、guard 呼叫 0 次、runtime readback（optimize 關、cuDNN benchmark 關、cuDNN TF32 開、matmul TF32 關）、artifact 參數 cuda:0／tensor constant CPU |

## 3. L1 — head tensor

5316 frames，三個 head 共用同一份 TRT backbone 特徵：

| pair | `score_maxabs` | `box_maxabs_px` | 門檻內 anchors | 跨 0.05 的 anchors |
|:--|--:|--:|--:|--:|
| **(L,C)** | **0.00137** | **0.466** | 897,492 | 5 |
| (L,E) | 0 | 0 | 897,489 | 0 |
| (E,C) | 0.00137 | 0.466 | 897,492 | 5 |

κ_L1：0.00137 ≤ 0.05 且 0.466 ≤ 4.0 ⇒ **`L1_PASS`**（無 non-finite）。只報告、不判定：L 與 E 在 5316/5316 frames 上逐位元相同。

## 4. L2 — 7-seq MOT

`A_L` 與 `A_C` 不是 byte-identical（第一個分歧幀：02／04／09／10／11 在 frame 3，05 在 5，13 在 4）。7-seq combined，未四捨五入：

| metric | A_C | A_L | Δ_L | Δ_N | b_m | \|Δ_L\| ≤ b_m |
|:--|--:|--:|--:|--:|--:|:--:|
| IDF1 | 78.2899 | 78.2899 | 0.000 | 0.000 | 0.20（floor） | 是 |
| HOTA | 69.9589 | 69.9594 | +0.0005 | +0.0005 | 0.20（floor） | 是 |
| MOTA | 77.8801 | 77.8801 | 0.000 | 0.000 | 0.20（floor） | 是 |
| IDs | 429 | 429 | 0 | 0 | 5（floor） | 是 |

κ_L2：四項皆在容差內 ⇒ **`L2_WITHIN`**。

**只報告、不判定**：

| | A_C | A_L | A_N |
|:--|--:|--:|--:|
| DetA | 70.2209 | 70.2213 | 70.2213 |
| AssA | 69.9101 | 69.9107 | 69.9107 |
| FP | 3471 | 3471 | 3471 |
| FN | 20940 | 20940 | 20940 |

byte identity（report-only）：`A_L` 的 7 個 txt 與 failure-localization r2 的 `R_E_1` 7/7 相同；`A_N` 與 PR-2R 的 `A_N_1` 7/7 相同；`A_L` 與 `A_N` 只在 MOT17-09、MOT17-10 相同。

## 5. Terminal 與 owner 決定

L1 PASS 且 L2 WITHIN（非 exact）⇒ **`HEAD_PARITY_WITHIN_TOLERANCE`**（宣告 §6 row 4，交 owner 判定）。

owner 決定 **`ACCEPT`**（#465 issuecomment-5967756337）：

- U1 關閉；PR-1L 的 LibTorch TorchScript＋native selective-scan 形式是 Phase B 的 shipping head。
- 帶著宣告的 named limit：**native shipping 的 MOT 輸出與 headline oracle 不是 byte-identical**；這不是「LibTorch／native shipping 組態重現 headline 組態」的主張，也不改變既有的 headline 結果。
- native loader 取代 `A_L` 注入路徑之前，必須先證明對 `A_L` byte-identical（PR-8 做到了：`native_runtime_resolved_config.md` §12）。
- PR-9、PR-10 以 `A_L` 為 parity oracle。
- shipping 產品引用的任何 accuracy、MOT、latency、FPS 數字都必須來自 native／`A_L` 組態；headline 的量測不得轉用。
- 被否決的 TRT 與 TF32-off 形式維持否決並封存；compiled-head 與 block-scope 研究是診斷分支，不 gate 這個決定。

## 6. 之後的事（只記錄，不屬於本 terminal）

正式 run 綁定的 operator library build（`cfea782f…`）之後被從同一份凍結 source blob 重新 build（`098dd233…`），舊的 binary 沒有保留。PR-8 沒有改 PR-1L／PR-2L 的任何凍結內容，而是另建 realization attestation：以 PR-2L runner 自己的函式重跑 `A_L`（double-buffer、7 sequences），7 個 txt 與本 packet 的 `A_L_1` 逐位元組相同（`native_runtime_resolved_config.md` §12.3；`configs/shipping/mamba_head_realization.attestation.json`）。本 terminal 描述的仍是 `cfea782f` 那一次 run。

## 7. 本文刻意沒有做的事

- 沒有重跑、沒有調整容差或 reference arm。
- 沒有對 `A_L` 與 `A_C` 的差異或 per-sequence 分佈做歸因。
- 沒有修改宣告（它的開頭狀態行維持凍結時的內容）。
