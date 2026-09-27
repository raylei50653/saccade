# Native runtime head parity (TF32 off) — 預宣告（#465 Phase B PR-2R／U1b redesign）

> 狀態：**預宣告，於 PR-2R 的第一個量測 run 之前凍結。** 凍結點 = 本文所在 PR merge；review 期間的修訂列在 §9。凍結之後只能以 §9 的 append-only amendment 修訂，不得 inline 編輯。
> 這是**新的**宣告，不是 [PR-2 宣告](native_runtime_head_parity_declaration.md) 的 amendment。PR-2 的 terminal `HEAD_PARITY_OUT_OF_TOLERANCE`（[結果](native_runtime_head_parity_result.md)）永久保留，描述的是 PR-1 的 TF32-allowed artifact。
> 權威 seal bar：[experiment contract §20.8](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（本文引用，不複述）。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B1；受測 artifact：[native_runtime_head_artifact_tf32_off.md](native_runtime_head_artifact_tf32_off.md)（PR-1R，#474）。

---

## 0. 這個 gate 回答什麼

> PR-1R 匯出的 TRT head（FP32 with TensorRT TF32 disabled，`saccade::SelectiveScan` plugin，head-only）在 headline 設定下，能不能取代 oracle 的 PyTorch head？

**與 PR-2 唯一的實質差異：受測 head T 換成 PR-1R 的 engine。** oracle、reference、資料、L1／L2 的量、門檻、容差政策、terminal 集合與判定順序全部沿用 PR-2 宣告的文字。這個設計讓 PR-2 與 PR-2R 構成「TF32 allowed vs TF32 disabled、其他條件不變」的對照；但本文只判定 PR-1R 形式，不對兩者的差異做歸因。

| | 狀態 |
|:--|:--|
| TRT head（TF32 off）與 oracle head 的 tensor 差異 | **回答**（L1） |
| 替換後 7-seq MOT 輸出與 metric 的差異 | **回答**（L2） |
| 替換是否可接受 | 由 §6 的 terminal 決定；非 bit-exact 時**交 owner 判定**（boundary §5 B1） |
| FPS／latency | **不回答** |
| 其他 head 形式（LibTorch、FP16、其他 TRT builder 設定、含 S2 的 artifact） | **不回答**（§20.8 item 4） |
| PR-2 結果為何出界、TF32 是否為原因 | **不回答** |
| benchmark claim | **不改** |

target decision layer = `none (cross-layer substrate work)`；κ 見 §4、§5。

## 1. 宣告前已看過的資料

以下全部是 **archived reference**，**不參與**任何門檻或容差的設定（門檻與容差政策原樣沿用 PR-2 宣告，在看到 PR-2 結果之前就已凍結）：

- PR-1 的冒煙比對（[PR-1 artifact doc §4](native_runtime_head_artifact.md)）。
- #473 的 smoke（2 序列 × 30 frames，非證據）。
- **PR-2 正式 run 的全部結果**（PR-1 TF32-allowed engine）：L1 (T,C) score 0.00315／box 0.585 px；L2 IDF1 −0.257、IDs +7、HOTA −0.084、MOTA −0.046；per-sequence 數字；`A_N` 與 `A_C` 在 IDF1／MOTA／IDs 上相同。
- PR-1R 的結構檢查：全零輸入，六個輸出 shape 正確且 finite。**沒有**在任何真實 frame 上看過 PR-1R engine 的輸出，也沒有任何 PR-1R 的 MOT run。

PR-1 的 TF32-allowed engine **不是**本 study 的 arm：它已有完整的正式 packet，重跑只會增加變因。

## 2. 凍結輸入（validity gate V1）

量測開始前，runner 逐項比對，任何一項不符 ⇒ `UNRESOLVED`：

| 項目 | 凍結值 |
|:--|:--|
| head ONNX | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.onnx`，sha256 `6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b`（== PR-1） |
| head lineage manifest | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.lineage.json`，**檔案 sha256 `a015bce04884c68df2b5c303dc9197dfd9c4856301e9bea6d2acf7cf251358fd`**（凍結 manifest 本身，避免 formal run 前把 manifest 與 engine 一起換成另一個合法 build） |
| manifest 內容 | `engine.precision == "fp32-no-tf32"`；`engine.builder_flag_readback.before_build` 與 `.after_build` 皆為 `{fp16: false, tf32: false}`；`onnx.sha256` == 上列 ONNX |
| head engine | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.engine`；sha256 == 上列 manifest 記錄的 `engine.sha256`（engine bytes 不可重現，凍結的是 manifest 綁定的那一份 build，不把 engine hash 當可重建的 contract），且 `export_headline_mamba_head.py --precision fp32-no-tf32 --check` 回 `OK` |
| checkpoint | `runs/mamba_gt_v14replica_t3_t1/best.ckpt`，sha256 `c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876` |
| backbone engine | `models/yolo/yolo26s_backbone_640_best.engine`，sha256 == manifest `companions.backbone_engine.sha256` |
| preset | `configs/presets/mamba_whole_graph.yaml`，sha256 == manifest `preset.sha256` |
| 程式碼 | clean tree；PR-2R runner（`scripts/eval/diagnostics/native_head_parity_tf32_off.py`）已 commit；本文與 runner 的 git blob sha 寫進 packet |
| runner 形式 | engine、manifest、宣告、容差、arms **不是** runner 的 CLI 選項，全部 hard-code；正式 invocation 不帶任何 flag（`--smoke-frames` 只產生 `evidence: false` 的 packet） |
| 環境 | `torch.backends.cudnn.allow_tf32` 與 `torch.backends.cuda.matmul.allow_tf32` == manifest `environment` 的值；TensorRT 版本 == manifest `engine.tensorrt_version`；同一台機器（manifest `environment.host`） |
| env hatch | caller 不得設定任何 `SACCADE_*`；每個 arm 記錄 `resolved_env_overrides()`，三個 arm 必須相同 |
| 資料 | `datasets/MOT17/train`，序列固定為 `MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP`（共 5316 frames），依此順序 |
| 資源 | 全部量測在同一個 `tools/resctl.py run machine-bench` 租約內執行，runner 是租約 owner 的直接 child，開始與結束時為同一份租約 |

## 3. 三種 head 實作

| 代號 | 實作 | 角色 |
|:--|:--|:--|
| **C** | PyTorch head，`set_head_compile(True)`＋`set_block_compile(True)`（oracle 預設） | **主 oracle**（headline claim 由它產生） |
| **E** | PyTorch head，eager（不 compile） | 次 oracle／reference |
| **T** | **PR-1R TRT engine（TF32 off）**，經 `TRTMambaHead.infer_graph` | 受測 |

## 4. L1 — head tensor（gross-error screen）

**用途**：抓「算的不是同一個函數」（plugin 讀錯 shape、權重錯位、layout 錯誤）。它**不是**接受準則，接受與否在 L2。

**輸入**：§2 的 5316 frames。每張 frame：GPU `decode_jpeg` → `float()/255` → `F.interpolate(640×640, bilinear, align_corners=False)` → 同一個 TRT backbone `infer` → 同一組 `p3/p4/p5` 同時餵 C、E、T。C 與 E 是兩個獨立的 head instance（同一份 state dict），避免 compile 切換。C 在 CUDA graph 之外執行；graph 內的組態由 L2 涵蓋。

**量**（每個 pair ∈ {(T,C), (T,E), (E,C)}，所有 frame 與 anchor 取最大值）：

- `score_maxabs`：`sigmoid(cls)` 逐元素 |Δ| 的最大值（全部 80 類、全部 anchor）；
- `box_maxabs_px`：以 `_postprocess_mamba_fixed_eager` 相同的 anchor／stride 解碼成 xyxy，再乘 `(W_orig/640, H_orig/640)` 換到原圖座標後，逐座標 |Δ| 的最大值。計入的 anchor 是該 pair **兩邊的聯集**：對 pair (X,Y)，mask = `(max_class sigmoid_X ≥ 0.05) | (max_class sigmoid_Y ≥ 0.05)`；判定用的 (T,C) 即 `(score_C ≥ 0.05) | (score_T ≥ 0.05)`。這樣同時涵蓋「T 新跨過門檻」（C < 0.05 ≤ T）與「T 掉出門檻」（T < 0.05 ≤ C），兩者都是 tracker 可能消費的 candidate。0.05 = headline 的 `base_score_floor` = `min(conf_threshold, track_thresh)`（`evaluator.py` 的 score floor）；headline 的 `crowd_low_score_mode` 為 false，所以 crowd 的較低門檻不會啟用。
- **跨越 0.05 本身不是 gross error**：threshold crossing 的實際後果交給 L2 判斷；L1 只檢查跨過門檻的那些 anchor，box 有沒有錯到 gross 的程度。

**κ_L1** =（空間：5316 frames × 全部 anchor；關係：(T,C) 的上述兩個最大值；規則：`score_maxabs ≤ 0.05` **且** `box_maxabs_px ≤ 4.0` ⇒ `L1_PASS`，否則 `L1_GROSS_ERROR`）。
(T,E)、(E,C) 只報告，不參與判定；另報各 pair 的 per-sequence max 與 99.9 percentile。

**L1 validity（V2）**：每個序列的前 20 frames，C、E、T 各重跑一次，輸出必須與第一次逐位元相同；否則 `UNRESOLVED`。

## 5. L2 — 7-seq MOT

**Arms**（皆為 `scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer --sequences <§2> --output <新目錄>`，不改 harness）：

| arm | 額外參數 | 意義 |
|:--|:--|:--|
| `A_C` | （無） | oracle |
| `A_T` | `--mamba-head-engine models/yolo/mamba_head_s_v14replica_t3_t1_fp32_notf32.engine` | 只把 head 換成 T；S2 與其餘設定同 oracle（postprocess compile 仍開） |
| `A_N` | `--no-compile` | reference：repository 已經當成「同一個模型」的另一個組態；它的偏差定義容差尺度。**這是 composite nuisance reference，不是 head-only reference**：`--no-compile` 同時關掉 head／block compile **與** `set_postprocess_compile`，而 `A_T` 保留 compiled postprocess。所以 `|Δ_N|` **不是**對 head compile effect 的估計，只是同 session、同模型、較廣的數值實作差異的參考尺度，可能比純 eager-head 的漂移大；cap 限制了它能把容差放寬到多少 |

**執行順序**：`A_C#1, A_T#1, A_N#1, A_C#2, A_T#2, A_N#2`，同一 session、同一租約。

**L2 validity（V3）**：每個 arm 的兩次 run，7 個 `MOT17-*-SDP.txt` 必須逐位元相同；任一 run 失敗、缺檔或兩次不同 ⇒ `UNRESOLVED`。（兩次相同不證明確定性；它只是讓「arm 之間的差異可歸因於 head」成立的必要條件。）

**Metric**：以 `saccade.perception.eval.metrics` 對每個 arm 的 #1 輸出計算 7-seq combined，**不經四捨五入**：IDF1、MOTA 由 `_evaluate_single_sequence` 的 counts 直接算（`_format_overall_metrics_from_counts` 的公式），HOTA 用 `_calculate_hota` 的原始 float，以百分點表示；IDs = `num_switches` 總和。`run_motmetrics_evaluation` 回傳的一位小數字串只作顯示。

**判定**：

1. 若 `A_T` 與 `A_C` 的 7 個 txt 全部逐位元相同 ⇒ `L2_EXACT`。
2. 否則，對 m ∈ {IDF1, HOTA, MOTA, IDs}：
   - Δ_T,m = m(A_T) − m(A_C)，Δ_N,m = m(A_N) − m(A_C)；
   - 容差 b_m = min(max(|Δ_N,m|, floor_m), cap_m)；
   - floor = 0.20 pt（IDF1／HOTA／MOTA）、5（IDs）；cap = 1.00 pt、30（IDs）。
   - **κ_L2** =（空間：7-seq combined 的四個 metric；關係：|Δ_T,m| 對 b_m；規則：四個都 `|Δ_T,m| ≤ b_m` ⇒ `L2_WITHIN`，任一超過 ⇒ `L2_OUT`）。判定是雙向的：變好與變差同樣算偏差。

floor 的理由：reference arm 可能恰好與 oracle 相同（b 會變成 0，任何 flip 都會失敗）。cap 的理由：reference arm 的偏差若很大，不能因此把容差放寬到會影響 headline 判讀的量級（1 pt 約是本專案單一 lever 的決策量級）。兩者都在看到任何 MOT 資料之前固定。**cap 不是「允許退化 1 pt／30 IDs」**：它是非 exact 結果最多還能進入 owner 判定（terminal 4）的範圍；`WITHIN_TOLERANCE` 本身不自動接受。

**只報告、不判定**：DetA、AssA、FP、FN、per-sequence 的四個 metric、每個序列第一個分歧的 frame、`A_N` 與 `A_C` 是否 byte-identical。


## 6. Terminal（窮盡，依序判定）

terminal 集合、條件與判定順序與 PR-2 宣告 §6 相同；主線轉移依 #465 記錄的 owner 路線：

| # | terminal | 條件 | 主線轉移 |
|:--|:--|:--|:--|
| 1 | `UNRESOLVED` | V1、V2、V3 任一不成立；或任何 runner／harness／engine 載入錯誤、例外、缺 packet（execution-invalid 一律歸此，fail-closed） | 無。只關閉這一次量測；重跑要先寫 amendment 說明原因與修正 |
| 2 | `HEAD_PARITY_GROSS_ERROR` | L1 = `L1_GROSS_ERROR` | PR-1R 形式被否決；#465 維持 `requires_runtime_redesign`。**停止調整 TRT head-only 的 builder 設定**（不再試 workspace、tactic 或其他 flag），下一步是預宣告的 failure-localization study，之後再決定 S2 artifact 或 LibTorch |
| 3 | `HEAD_PARITY_EXACT` | L1 PASS 且 `L2_EXACT` | U1 關閉，無 named limit；PR-1R 成為 Phase B 的 head 形式，後續 PR 恢復；後續 MOT parity 仍以 §2 oracle 為準 |
| 4 | `HEAD_PARITY_WITHIN_TOLERANCE` | L1 PASS 且 `L2_WITHIN` | **待 owner 判定**：<br>• `ACCEPT` ⇒ U1 關閉，named limit：native shipping 的 MOT 輸出不與 headline byte-identical。PR-1R 成為 Phase B 的 head 形式，後續 PR 恢復；PR-9／PR-10 的 parity oracle 改為 `A_T` 組態（Python harness＋`--mamba-head-engine` PR-1R engine），使其餘單元仍能以 byte-identity 驗收；shipping 產物的任何數字只能引用 native／`A_T` 的量測，不得沿用 headline 數字<br>• `REJECT` ⇒ 同 terminal 5 |
| 5 | `HEAD_PARITY_OUT_OF_TOLERANCE` | L1 PASS 且 `L2_OUT` | 同 terminal 2 |

任何 terminal 都不撤銷 PR-2 的結果；PR-2 的 `HEAD_PARITY_OUT_OF_TOLERANCE` 描述 TF32-allowed artifact，永久保留。

## 7. Packet

runner 把以下內容寫到非 scratch 的 timestamped 目錄（`results/native_head_parity_465_tf32_off/<UTC>/`，以 `open_run` claim）並附 manifest：本文與 runner 的 git blob sha、V1 每一項的實測值、L1 的全部 pair 統計（CSV）、6 個 run 的完整輸出目錄與 stdout、metric 計算的 counts 與未四捨五入值、判定過程與 terminal。PR-2R 的結果文件引用這個 packet，只寫 terminal 與支撐它的數字。

## 8. 本文刻意沒有做的事

- 沒有執行任何 PR-1R 的量測。
- 沒有修改 PR-2 的門檻、容差政策或 reference arm；沒有加入 PR-1 engine 的 arm。
- 沒有預先決定 owner 在 terminal 4 的判定。
- 沒有宣告 FP16、LibTorch、其他 TRT builder 設定或含 S2 的 artifact 的任何事，也不對 PR-2 的出界原因做歸因。

## 9. Review 修訂與 amendments

**凍結前的 review 修訂**：

（無）

**凍結後的 amendments**（append-only）：

（無）
