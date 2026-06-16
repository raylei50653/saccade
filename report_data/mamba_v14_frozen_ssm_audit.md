# v14 frozen-SSM gradient audit

Date: 2026-06-12

## Finding

整條 legacy v14 lineage（distill 2026-05-27 → GT1 → v14 GT 60 epochs）中，
Mamba SSM 內部參數 **從未接受任何梯度更新**：`A_log`、`D`、`conv1d`、
`x_proj`、`dt_proj`（以及 `in_proj` 的 x 半邊）全程停留在初始值。
生產模型 v14 的「Mamba」實際上是一個凍結的結構化隨機 SSM mixer；
所有學習都發生在 gate（`in_proj` 的 z 半邊）、`out_proj`、
`input_proj`/`downsample`/PixelShuffle 路徑與 cls/reg heads。

## Evidence

### 1. Git timeline（程式碼層）

| Commit | 日期 | `_selective_scan` 行為 |
|---|---|---|
| `37d51062` | 05-22 | 初版（pure PyTorch JIT，可微分） |
| `abc92c15` | 05-27 01:17 | 切換為無條件呼叫 raw CUDA forward ext（fp16 Mamba scan）。輸出 `torch.empty_like` + 指標填值，**無 grad_fn** |
| `9b5e9d94` | 05-29 | 同上；kernel 以 `A.shape[0]` 當 dstate → shared A (1,16) 時 **forward N=1**（且 B/C 讀取錯位） |
| `77fcc262` | 05-31 23:22 | 修正 N：`A.numel()` → forward N=16。**仍無 backward** |
| 06-07 / 06-09 | — | 才加入可微分 JIT 訓練路徑與 CUDA fwd+bwd autograd Function |

v14 各訓練階段（05-27 distill、05-27 GT1、05-31 GT2）全部落在
「scan 無梯度」窗口內。

### 2. Checkpoint 逐 tensor 比對（artifact 層）

v14 `best.ckpt`（60 epochs GT 訓練後）vs 其 warm-start parent
`mamba_gt_pixelshuffle_crossscan/best.ckpt`：

- **逐 bit 相同（21 tensors）**：`A_log`×3、`D`×3、`conv1d`×6、`x_proj`×3、
  `dt_proj`×6 —— 恰為 scan 上游的全部 SSM 內部參數。
- **有更新（48 tensors）**：`in_proj`、`out_proj`、`input_proj`、
  `downsample`、cls/reg heads —— 恰為 no-grad-scan 梯度拓撲允許更新的集合。

### 3. 初始值驗證（追溯到 distill 起點）

parent checkpoint 的 `A_log == log(arange(1..16))`（逐 bit）、`D == ones`，
即模型建構時的確定性初始值 → SSM 內部從 distill 第一天起就沒被訓練過，
不只是 GT 階段。

### 4. 對照組

現行 `mamba_gt_v14r_strict_holdout02`（具 scan backward）的 `A_log`
已偏離 init（max Δ 0.6280）→ 新訓練 regime 與 v14 是**不同 model class**。

### 5. 梯度拓撲驗證（新 flag）

`--scan-stop-grad`（`MambaBlock` 在 scan 後 `y.detach()`）單元驗證：
開啟時 SSM 內部 21 tensors 零梯度、in/out_proj 正常；關閉時全部可訓。
與歷史 artifact 的更新集合完全一致。

## 連帶修正

### 「N=1 → N=16 curriculum」是誤判

scan 參數從未學習，「N=1 訓練階段」不存在訓練機制上的意義；N=1 只是
`A.shape[0]` bug 造成的 forward/eval artifact（且 B/C 讀取錯位，並非
「只用第一個 state」）。「修 kernel 後不重訓即 72.2 IDF1」的乾淨解釋：
scan 內部本來就是 N=16-shaped 的未訓練 init，修正只是解鎖完整 init
dynamics。多段 warm-start 的真實貢獻是 gate/readout/heads 的累積訓練量。

### teacher「7.18% relative L2 change」被 BN running stats 灌水

| 成分 | gated_det_v1 vs 原始 yolo26s.pt |
|---|---|
| learned weights（conv/affine） | **1.84%** |
| BN running stats | 15–17% |

teacher prior 的主要成分是 BN 統計量的 MOT17 重校準 + 1.8% 權重微調。
另經 epoch 軌跡（e1=0.48% 平滑升至 e12=1.84%）與 epoch_0001 optimizer
step count（665 ≈ 單 epoch）確認：現存 `runs/gated_det_v1` 為單段、
從原始 yolo26s.pt 起訓（05-23）；更早（05-19/20）的 gated run 已被覆蓋，
未進入下游 lineage。

### v2 audit 異常已解（2026-06-12 補）

`mamba_v14_training_audit.md` / mamba-head-training.md §5.2 記載的
「GT 後 temporal in_proj/out_proj 有變、A_log/conv/SSM projection 不變」
由兩個因素共同解釋：

1. **未 commit 的 T1/T3 雙 loss 工作樹**：Claude session
   `34cabc95`（標題 `temporal-alternating-t1-t3-training`，2026-05-29 →
   05-31，branch `feat/option-f-mamba`）記錄當時 GT 訓練實際跑的是
   每 batch 同時計算 T=3 與 T=1 loss 的修改版（log 格式
   `loss=… T3=… T1=…`），temporal blocks 確實執行——不是 committed
   HEAD 的逐幀 bypass。audit 對「未保存工作樹版本」的懷疑證實。
2. **no-grad scan 梯度拓撲**：temporal blocks 執行時也只有
   in_proj/out_proj 能收到梯度，A_log/conv/SSM projection 不變。

該 T1/T3 實驗於 05-31 02:58 全部回滾；v14（05-31 04:31 起「修 N=16 +
重新訓練」）建立於回滾後的樹、為純空間模型，**T1/T3 未進入 v14
lineage**，復刻協議無需納入。

## 對 v14 重現的意涵

strict v14-R recall 崩壞（MOT17-02 all 0.94→0.48）與 legacy v14 之間有
**兩個 confound**：

1. teacher prior（MOT17-adapted vs 原始 YOLO）——主嫌（全 bins 含大目標崩）；
2. gradient regime（frozen SSM vs 全可訓 SSM）——「直接訓 N=16 反而退化」
   的唯一一次全梯度訓練即 strict run 本身。

復刻協議見
[`docs/modules/detection/mamba-v14-replication-protocol.md`](../docs/modules/detection/mamba-v14-replication-protocol.md)。
