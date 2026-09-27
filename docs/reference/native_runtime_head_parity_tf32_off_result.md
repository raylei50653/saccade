# Native runtime head parity (TF32 off) — 結果（#465 Phase B PR-2R／U1b redesign）

> 狀態：**terminal = `HEAD_PARITY_OUT_OF_TOLERANCE`**（正式 run，一次，valid）。
> 依據：[PR-2R 預宣告](native_runtime_head_parity_tf32_off_declaration.md)（blob `2f48ddfd37bb1352aa28a945551f16c30ec9e029`，§9 只有凍結前的 R1，無凍結後 amendment）；受測 artifact：[PR-1R](native_runtime_head_artifact_tf32_off.md)。
> 本文只寫 terminal 與支撐它的數字（宣告 §7）；不做歸因、不讀任何時間量、不改 benchmark claim。

---

## 1. 這次 run

| 項目 | 值 |
|:--|:--|
| freeze commit | `00611af12f783de2d79b0bc53ceb0af19a58633d`（#475 merge；freeze record 已在 #475／#465 公布）；正式 run 以 detached checkout 在此 SHA 執行，main CI 綠、`origin/main` 未前進 |
| runner | `scripts/eval/diagnostics/native_head_parity_tf32_off.py`，blob `bf7f798c598249d0aca5090153dd68c3b489b216`（freeze commit 的那一份） |
| 租約 | `tools/resctl.py run machine-bench`，runner 為租約 owner 的直接 child，開始與結束為同一份租約 |
| packet | `results/native_head_parity_465_tf32_off/20260927T151100Z/`（gitignored，本機保留）；`packet.json` sha256 `bb5b040e…1bcb`，`MANIFEST.json` sha256 `8c82f0a5…742d` |
| 事前看過的資料 | 宣告 §1 所列；沒有 smoke，沒有任何 PR-1R 真實 frame 的輸出 |

## 2. Validity

| gate | 結果 |
|:--|:--|
| V1（§2 凍結輸入） | 全部通過，包括：lineage manifest 檔案 sha256 `a015bce0…`、`engine.precision == fp32-no-tf32`、build 前後 flag 讀回皆 `{fp16:false, tf32:false}`、engine `c77148a8…` == manifest、`--precision fp32-no-tf32 --check` OK、HEAD 為 freeze commit、宣告 blob == frozen、clean tree、caller 無 `SACCADE_*`、7 序列 5316 frames |
| V2（L1 重跑） | 140 frames × C／E／T，全部 `torch.equal` |
| V3（L2 兩次 run） | `A_C`、`A_T`、`A_N` 各自兩次的 7 個 txt 逐位元相同；六個 run 的 `run_manifest.json` cmdline／commit／`dirty=false` 皆相符 |

## 3. L1 — head tensor

判定 pair (T,C)，5316 frames：

| pair | `score_maxabs` | `box_maxabs_px`（C∪T ≥ 0.05） | 門檻內 anchors | 跨 0.05 的 anchors |
|:--|--:|--:|--:|--:|
| **(T,C)** | **0.00395** | **0.775** | 897,574 | 160 |
| (T,E) | 0.00395 | 0.775 | 897,573 | 159 |
| (E,C) | 0.00137 | 0.466 | 897,492 | 5 |

κ_L1：0.00395 ≤ 0.05 且 0.775 ≤ 4.0 ⇒ **`L1_PASS`**（無 non-finite）。

## 4. L2 — 7-seq MOT

`A_T` 與 `A_C` 不是 byte-identical（7 個序列的第一個分歧都在 frame 3）。7-seq combined，未四捨五入：

| metric | A_C | A_T | Δ_T | Δ_N | b_m | \|Δ_T\| ≤ b_m |
|:--|--:|--:|--:|--:|--:|:--:|
| IDF1 | 78.2899 | 77.3572 | **−0.933** | 0.000 | 0.20（floor） | **否** |
| HOTA | 69.9589 | 69.3661 | **−0.593** | +0.0005 | 0.20（floor） | **否** |
| MOTA | 77.8801 | 77.8302 | −0.050 | 0.000 | 0.20（floor） | 是 |
| IDs | 429 | 450 | **+21** | 0 | 5（floor） | **否** |

κ_L2：IDF1、HOTA、IDs 超過容差 ⇒ **`L2_OUT`**。`A_N` 與 `A_C` 的關係與 PR-2 相同（txt 不 byte-identical，IDF1／MOTA／IDs 相同，HOTA +0.0005），四個容差都落在 floor。

**只報告、不判定**：

| | A_C | A_T | A_N |
|:--|--:|--:|--:|
| DetA | 70.221 | 70.263 | 70.221 |
| AssA | 69.910 | 68.696 | 69.911 |
| FP | 3471 | 3522 | 3471 |
| FN | 20940 | 20924 | 20940 |

per-sequence Δ_T（IDF1／IDs）：02 +0.18／−2、04 −1.16／+3、05 −0.06／+1、09 −1.00／+3、10 −1.95／+9、11 −0.03／+1、13 −1.49／+6。

## 5. 與 PR-2 並列（觀察，不歸因）

| | PR-2（TF32 allowed） | PR-2R（TF32 disabled） |
|:--|--:|--:|
| L1 (T,C) score／box | 0.00315／0.585 px | 0.00395／0.775 px |
| L1 跨 0.05 的 anchors | 93 | 160 |
| L2 Δ IDF1／HOTA／MOTA／IDs | −0.257／−0.084／−0.046／+7 | −0.933／−0.593／−0.050／+21 |

兩個 study 除了受測 engine 之外條件相同（同 oracle、reference、資料、門檻），`A_C` 與 `A_N` 的 7 個 txt 在兩個 study 之間逐位元相同（兩個 packet 的 `A_C_1`、`A_N_1` 比對）。TF32 disabled 的 engine 在 L1 與 L2 都離 oracle 更遠，不是更近。依宣告 §0，本文不對這個差異做歸因；兩次 run 各只有一個 build，也不足以把差異歸給 TF32 flag 本身而非 build 之間的 tactic 選擇。

## 6. Terminal 與主線轉移

L1 PASS 且 L2 OUT ⇒ **`HEAD_PARITY_OUT_OF_TOLERANCE`**（宣告 §6 terminal 5 = terminal 2 的轉移）：

- PR-1R 形式（FP32 with TensorRT TF32 disabled，head-only）被否決；#465 維持 `requires_runtime_redesign`，Phase B 仍停止。
- **停止調整 TRT head-only 的 builder 設定**（不再試 workspace、tactic 或其他 flag）。
- 下一步：預宣告的 failure-localization study，之後再決定 S2 artifact 或 LibTorch。
- PR-2 的 terminal 不受影響；兩個 terminal 各自描述自己的 artifact。headline 數字不變。

## 7. 本文刻意沒有做的事

- 沒有重跑、沒有調整容差或 reference arm。
- 沒有對 PR-2 與 PR-2R 的差異、per-sequence 分佈或 floor crossing 做歸因；那屬於 failure-localization study，需要另一份宣告。
- 沒有評估任何其他 head 形式或 builder 設定。
