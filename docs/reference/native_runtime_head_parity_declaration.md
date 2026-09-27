# Native runtime head parity — 預宣告（#465 Phase B PR-2／U1b）

> 狀態：**預宣告，於 PR-2 的第一個量測 run 之前 commit。** commit 之後只能以文末 §9 的 append-only amendment 修訂，不得 inline 編輯。
> 權威 seal bar：[experiment contract §20.8](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（本文引用，不複述）。
> 邊界依據：[native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) §5 B1 validation、§6 PR-2；受測 artifact：[native_runtime_head_artifact.md](native_runtime_head_artifact.md)（PR-1，#471）。

---

## 0. 這個 gate 回答什麼

> PR-1 匯出的 TRT FP32 head（`saccade::SelectiveScan` plugin，head-only）在 headline 設定下，能不能取代 oracle 的 PyTorch head？

| | 狀態 |
|:--|:--|
| TRT head 與 oracle head 的 tensor 差異 | **回答**（L1） |
| 替換後 7-seq MOT 輸出與 metric 的差異 | **回答**（L2） |
| 替換是否可接受 | 由 §6 的 terminal 決定；非 bit-exact 時**交 owner 判定**（boundary §5 B1） |
| FPS／latency | **不回答**。本 gate 不讀任何時間量 |
| 其他 head 形式（LibTorch、FP16、含 S2 的 artifact） | **不回答**。負面 terminal 只對本文凍結的形式成立（§20.8 item 4） |
| benchmark claim | **不改**。任何 terminal 都不改寫 headline 數字 |

§20.2 的 layer／intent 軸是給 signal-family study 用的；本文是 fidelity gate：target decision layer = `none (cross-layer substrate work)`，每個可判定單元以 κ =（量化空間, 比較關係, 判定規則）宣告（§4、§5）。

## 1. 宣告前已看過的資料

- PR-1 的冒煙比對（[artifact doc §4](native_runtime_head_artifact.md)）：3 張 frame 的 head tensor，TRT vs eager logits max|Δ| ~1.4e-2、reg ~7.8e-3、score ~1e-3；compile vs eager cls ~4.9e-3。
- **沒有**看過任何 arm 的 MOT 輸出或 metric。
- 下文 L1 門檻刻意設在與冒煙量級無關的 gross-error 等級（§4），不是依冒煙數字擬合；L2 容差的尺度來自同一 session 量到的 reference arm（§5），不來自任何已看過的資料。

## 2. 凍結輸入（validity gate V1）

量測開始前，runner 逐項比對，任何一項不符 ⇒ `UNRESOLVED`：

| 項目 | 凍結值 |
|:--|:--|
| head ONNX | sha256 `6e919dad14af81083a25679225930a3473a8cdd6ebf9828ea07b414a9316b58b` |
| head engine | `models/yolo/mamba_head_s_v14replica_t3_t1_fp32.engine`；sha256 == 當下 lineage manifest 記錄的值，且 `export_headline_mamba_head.py --check` 回 `OK`（engine bytes 不可重現，所以凍結的是「manifest 綁定的那一份 build」，不是固定 hash） |
| checkpoint | `runs/mamba_gt_v14replica_t3_t1/best.ckpt`，sha256 `c161c88e50b894d8b51cc614c46c3700370373decf05a15825bdf00ccf0e0876` |
| backbone engine | `models/yolo/yolo26s_backbone_640_best.engine`，sha256 == lineage manifest `companions.backbone_engine.sha256` |
| preset | `configs/presets/mamba_whole_graph.yaml`，sha256 == lineage manifest `preset.sha256` |
| 程式碼 | clean tree（`git status --porcelain` 為空）；PR-2 runner 與本文的 git blob sha 寫進 packet |
| 環境 | `torch.backends.cudnn.allow_tf32` 與 `torch.backends.cuda.matmul.allow_tf32` == lineage manifest `environment` 的值；TensorRT 版本 == manifest `engine.tensorrt_version`；同一台機器（manifest `environment.host`） |
| env hatch | runner 不設定任何 `SACCADE_*`；每個 arm 記錄 `resolved_env_overrides()`，三個 arm 必須相同 |
| 資料 | `datasets/MOT17/train`，序列固定為 `MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP`（共 5316 frames），依此順序 |
| 資源 | 全部量測在同一個 `tools/resctl.py run machine-bench` 租約內執行 |

## 3. 三種 head 實作

| 代號 | 實作 | 角色 |
|:--|:--|:--|
| **C** | PyTorch head，`set_head_compile(True)`＋`set_block_compile(True)`（oracle 預設） | **主 oracle**（headline claim 由它產生） |
| **E** | PyTorch head，eager（不 compile） | 次 oracle／reference |
| **T** | PR-1 TRT engine，經 `TRTMambaHead.infer_graph` | 受測 |

## 4. L1 — head tensor（gross-error screen）

**用途**：抓「算的不是同一個函數」（plugin 讀錯 shape、權重錯位、layout 錯誤）。它**不是**接受準則，接受與否在 L2。

**輸入**：§2 的 5316 frames。每張 frame：GPU `decode_jpeg` → `float()/255` → `F.interpolate(640×640, bilinear, align_corners=False)` → 同一個 TRT backbone `infer` → 同一組 `p3/p4/p5` 同時餵 C、E、T。C 與 E 是兩個獨立的 head instance（同一份 state dict），避免 compile 切換。C 在 CUDA graph 之外執行；graph 內的組態由 L2 涵蓋。

**量**（每個 pair ∈ {(T,C), (T,E), (E,C)}，所有 frame 與 anchor 取最大值）：

- `score_maxabs`：`sigmoid(cls)` 逐元素 |Δ| 的最大值（全部 80 類、全部 anchor）；
- `box_maxabs_px`：以 `_postprocess_mamba_fixed_eager` 相同的 anchor／stride 解碼成 xyxy，再乘 `(W_orig/640, H_orig/640)` 換到原圖座標後，逐座標 |Δ| 的最大值；只計入在 **C** 中 `max_class sigmoid ≥ 0.05` 的 anchor（0.05 = headline 的 `base_score_floor` = `min(conf_threshold, track_thresh)`，`evaluator.py` 的 score floor；headline 的 `crowd_low_score_mode` 為 false，所以 crowd 的較低門檻不會啟用）。

**κ_L1** =（空間：5316 frames × 全部 anchor；關係：(T,C) 的上述兩個最大值；規則：`score_maxabs ≤ 0.05` **且** `box_maxabs_px ≤ 4.0` ⇒ `L1_PASS`，否則 `L1_GROSS_ERROR`）。
(T,E)、(E,C) 只報告，不參與判定；另報各 pair 的 per-sequence max 與 99.9 percentile。

**L1 validity（V2）**：每個序列的前 20 frames，C、E、T 各重跑一次，輸出必須與第一次逐位元相同；否則 `UNRESOLVED`。

## 5. L2 — 7-seq MOT

**Arms**（皆為 `scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer --sequences <§2> --output <新目錄>`，不改 harness）：

| arm | 額外參數 | 意義 |
|:--|:--|:--|
| `A_C` | （無） | oracle |
| `A_T` | `--mamba-head-engine models/yolo/mamba_head_s_v14replica_t3_t1_fp32.engine` | 只把 head 換成 T；S2 與其餘設定同 oracle（postprocess compile 仍開） |
| `A_N` | `--no-compile` | reference：repository 已經當成「同一個模型」的另一個組態；它的偏差定義容差尺度 |

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

floor 的理由：reference arm 可能恰好與 oracle 相同（b 會變成 0，任何 flip 都會失敗）。cap 的理由：reference arm 的偏差若很大，不能因此把容差放寬到會影響 headline 判讀的量級（1 pt 約是本專案單一 lever 的決策量級）。兩者都在看到任何 MOT 資料之前固定。

**只報告、不判定**：DetA、AssA、FP、FN、per-sequence 的四個 metric、每個序列第一個分歧的 frame、`A_N` 與 `A_C` 是否 byte-identical。

## 6. Terminal（窮盡，依序判定）

| # | terminal | 條件 | 主線轉移 |
|:--|:--|:--|:--|
| 1 | `UNRESOLVED` | V1、V2、V3 任一不成立；或任何 runner／harness／engine 載入錯誤、例外、缺 packet（execution-invalid 一律歸此，fail-closed） | 無。只關閉這一次量測；重跑要先寫 amendment 說明原因與修正 |
| 2 | `HEAD_PARITY_GROSS_ERROR` | L1 = `L1_GROSS_ERROR` | TRT FP32 head-only 形式被否決（僅此形式）。依 boundary §5 B1，verdict 升級為 `requires_runtime_redesign`，Phase B 停止；owner 可另開新的 PR-1（不同形式）＋新宣告，那是新的 study，不是重跑 |
| 3 | `HEAD_PARITY_EXACT` | L1 PASS 且 `L2_EXACT` | U1 關閉，無 named limit；PR-8 使用此 artifact，後續 MOT parity 仍以 §2 oracle 為準 |
| 4 | `HEAD_PARITY_WITHIN_TOLERANCE` | L1 PASS 且 `L2_WITHIN` | **待 owner 判定**：<br>• `ACCEPT` ⇒ U1 關閉，named limit：native shipping 的 MOT 輸出不與 headline byte-identical。PR-9／PR-10 的 parity oracle 改為 `A_T` 組態（Python harness＋`--mamba-head-engine`），使其餘單元仍能以 byte-identity 驗收；shipping 產物的任何數字只能引用 native／`A_T` 的量測，不得沿用 headline 數字<br>• `REJECT` ⇒ 同 terminal 5 |
| 5 | `HEAD_PARITY_OUT_OF_TOLERANCE` | L1 PASS 且 `L2_OUT` | 同 terminal 2：此形式被否決、verdict 升級、Phase B 停止；owner 可另開不同形式的新宣告 |

## 7. Packet

runner 把以下內容寫到非 scratch 的 timestamped 目錄並附 manifest（raw 保留規則）：本文與 runner 的 git blob sha、V1 每一項的實測值、L1 的全部 pair 統計（CSV）、6 個 run 的完整輸出目錄與 stdout、metric 計算的 counts 與未四捨五入值、判定過程與 terminal。PR-2 的結果文件引用這個 packet，只寫 terminal 與支撐它的數字。

## 8. 本文刻意沒有做的事

- 沒有寫 runner、沒有執行任何量測。
- 沒有預先決定 owner 在 terminal 4 的判定。
- 沒有宣告 FP16、LibTorch 或含 S2 的 artifact 的任何事。

## 9. Amendments（append-only）

（無）
