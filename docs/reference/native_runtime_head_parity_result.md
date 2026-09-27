# Native runtime head parity — 結果（#465 Phase B PR-2／U1b）

> 狀態：**terminal = `HEAD_PARITY_OUT_OF_TOLERANCE`**（正式 run，一次，valid）。
> 依據：[預宣告](native_runtime_head_parity_declaration.md)（blob `0941a010bdc4ca30344e580108403d6074ebdc67`，§9 無 amendment）；受測 artifact：[native_runtime_head_artifact.md](native_runtime_head_artifact.md)。
> 本文只寫 terminal 與支撐它的數字（宣告 §7）；不做歸因、不讀任何時間量、不改 benchmark claim。

---

## 1. 這次 run

| 項目 | 值 |
|:--|:--|
| runner | `scripts/eval/diagnostics/native_head_parity.py`，blob `a1a15d1a10860837c5c26b06957aab175b2a7b7e`（HEAD `80d0730e`，#473 CI 全綠後執行） |
| 租約 | `tools/resctl.py run machine-bench`，runner 為租約 owner 的直接 child；開始與結束時為同一份租約（start `2026-09-27T13:34:21Z`） |
| packet | `results/native_head_parity_465/20260927T133422Z/`（gitignored，本機保留）；`packet.json` sha256 `c04b5397…0323`，`MANIFEST.json` sha256 `e33ecd84…34fa` |
| 事前看過的資料 | 宣告 §1 之外，只有 #473 揭露的 smoke（2 序列 × 30 frames，非證據） |

## 2. Validity

| gate | 結果 |
|:--|:--|
| V1（§2 凍結輸入） | 全部通過：ONNX `6e919dad…`、engine == manifest、`--check` OK、ckpt／backbone／preset hash、clean tree、宣告 blob == frozen、runner 已 commit、TF32／TensorRT 10.16.1.11／host 相符、caller 無 `SACCADE_*`、7 序列 5316 frames |
| V2（L1 重跑） | 140 frames × C／E／T，全部 `torch.equal` |
| V3（L2 兩次 run） | `A_C`、`A_T`、`A_N` 各自兩次的 7 個 txt 逐位元相同；六個 run 的 `run_manifest.json` cmdline／commit／`dirty=false` 皆相符 |

## 3. L1 — head tensor

判定 pair (T,C)，5316 frames：

| pair | `score_maxabs` | `box_maxabs_px`（C∪T ≥ 0.05） | 門檻內 anchors | 跨 0.05 的 anchors |
|:--|--:|--:|--:|--:|
| **(T,C)** | **0.00315** | **0.585** | 897,543 | 93 |
| (T,E) | 0.00315 | 0.585 | 897,542 | 92 |
| (E,C) | 0.00137 | 0.466 | 897,492 | 5 |

κ_L1：0.00315 ≤ 0.05 且 0.585 ≤ 4.0 ⇒ **`L1_PASS`**（無 non-finite）。per-sequence 值與 p99.9 在 packet 的 `l1/l1_pairs.csv`。

## 4. L2 — 7-seq MOT

`A_T` 與 `A_C` 不是 byte-identical（7 個序列的第一個分歧都在 frame 3）。7-seq combined，未四捨五入：

| metric | A_C | A_T | Δ_T | Δ_N | b_m | |Δ_T| ≤ b_m |
|:--|--:|--:|--:|--:|--:|:--:|
| IDF1 | 78.2899 | 78.0331 | **−0.257** | 0.000 | 0.20（floor） | **否** |
| HOTA | 69.9589 | 69.8752 | −0.084 | +0.0005 | 0.20（floor） | 是 |
| MOTA | 77.8801 | 77.8338 | −0.046 | 0.000 | 0.20（floor） | 是 |
| IDs | 429 | 436 | **+7** | 0 | 5（floor） | **否** |

κ_L2：IDF1 與 IDs 超過容差 ⇒ **`L2_OUT`**。

`A_N` 的 txt 與 `A_C` 也不 byte-identical（第一個分歧在 frame 3–5），但 IDF1／MOTA／IDs／FP／FN 與 `A_C` 相同，HOTA 差 +0.0005；因此四個容差都落在 floor。

**只報告、不判定**：

| | A_C | A_T | A_N |
|:--|--:|--:|--:|
| DetA | 70.221 | 70.226 | 70.221 |
| AssA | 69.910 | 69.743 | 69.911 |
| FP | 3471 | 3505 | 3471 |
| FN | 20940 | 20951 | 20940 |

per-sequence Δ_T（IDF1／IDs）：02 −0.03／+1、04 +0.28／−3、05 0.00／0、09 +0.01／0、10 **−2.80／+6**、11 +0.06／0、13 −0.51／+3。Δ_N 在每個序列的這兩項都是 0。本文不對這個分佈做歸因。

## 5. Terminal 與主線轉移

L1 PASS 且 L2 OUT ⇒ **`HEAD_PARITY_OUT_OF_TOLERANCE`**（宣告 §6 terminal 5）。依宣告：

- **被否決的只有本文凍結的形式**：TRT FP32、head-only、batch 1、PR-1 的這份 ONNX。LibTorch、其他 TRT build 設定、含 S2 的 artifact 都**沒有**被回答（§0、§20.8 item 4）。
- 依 boundary §5 B1，#465 verdict 升級為 **`requires_runtime_redesign`**，**Phase B 停止**（PR-3 以後不開工）。
- owner 可另開不同形式的新 PR-1＋新宣告；那是新的 study，不是這次的重跑。
- headline 數字不變；`A_T` 組態的任何數字都不可引用為 headline。

## 6. 本文刻意沒有做的事

- 沒有重跑、沒有調整容差，也沒有換 reference arm。
- 沒有對 IDF1／IDs 的偏差做歸因（例如集中在哪個序列、是否來自 floor crossing）；那需要另一份宣告。
- 沒有評估任何替代的 head 形式。
