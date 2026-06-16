# Paper Algorithms: Temporally Shaped Mamba Detection

本文件只描述目前論文主線。完整 Kalman、GMC、association、birth、
ReID 與 relink 公式移至
[pipeline_algorithms_reference.md](pipeline_algorithms_reference.md)；
bridge 閘門細節另見
[relink_gates_and_formulas.md](relink_gates_and_formulas.md)。

## 1. Problem Definition

給定影片影格 \(I_t\)，凍結的 YOLO backbone 產生三尺度特徵：

\[
\mathcal{F}_t =
\left\{
F_t^3,\ F_t^4,\ F_t^5
\right\},\qquad
F_t^l \in \mathbb{R}^{C_l \times H_l \times W_l}.
\]

目標是學習 detection head \(h_\theta\)：

\[
D_t = h_\theta(\mathcal{F}_t),
\]

使單幀輸出 \(D_t\) 不只具有檢測能力，也具有較穩定的跨幀 box 與 score，
從而改善後端 IoU association。推論時不保留 temporal state。

## 2. Spatial Mamba Detection Head

### 2.1 Multi-scale projection

對每個 FPN level \(l\)，先投影到固定寬度 \(d=128\)，再以倍率
\(r=4\) 降採樣：

\[
X_t^l = \operatorname{Down}_r
\left(
\operatorname{Conv}_{1\times1}^l(F_t^l)
\right)
\in \mathbb{R}^{d\times H_l'\times W_l'}.
\]

降採樣後序列長度為 \(L_l=H_l'W_l'\)，避免直接在完整 FPN resolution
執行 selective scan。

### 2.2 Four-direction spatial scan

定義四個掃描方向：

\[
\mathcal{S}(X)=
\left[
X,\ \operatorname{flip}_{hw}(X),\
\operatorname{flip}_{w}(X),\
\operatorname{flip}_{h}(X)
\right].
\]

各方向展平成序列後共用 Mamba block：

\[
Z_k = \operatorname{Mamba}_s
\left(
\operatorname{flatten}(\mathcal{S}_k(X))
\right)
+ \operatorname{flatten}(\mathcal{S}_k(X)).
\]

輸出還原方向並平均：

\[
\bar{Z} =
\frac{1}{4}
\sum_{k=1}^{4}
\mathcal{S}_k^{-1}(Z_k).
\]

最後使用 learned PixelShuffle 回復原 FPN 尺度：

\[
U_t^l =
\operatorname{PixelShuffle}_r
\left(
\operatorname{Conv}_{3\times3}^l(\bar{Z}_t^l)
\right).
\]

分類與回歸分支同時接收 backbone local feature 與 Mamba context：

\[
\begin{aligned}
P_{\mathrm{cls}}^l &=
g_{\mathrm{cls}}^l
\left(
\left[F_t^{l,\mathrm{proj}};\ U_t^l\right]
\right),\\
P_{\mathrm{box}}^l &=
g_{\mathrm{box}}^l
\left(
\left[F_t^{l,\mathrm{proj}};\ U_t^l\right]
\right).
\end{aligned}
\]

### 2.3 Frozen-SSM gradient topology

Mamba block 的 selective scan 為：

\[
Y = \operatorname{Scan}(X,\Delta,A,B,C,D).
\]

v14 regime 在 scan 輸出執行 stop-gradient：

\[
\tilde{Y} = \operatorname{sg}(Y),
\qquad
O = W_o\left(\tilde{Y}\odot\operatorname{SiLU}(Z)\right).
\]

因此 scan dynamics 與輸入支路不接收 detection loss 梯度；可學習訊號主要
經 gate/readout、FPN projection、upsampling 與 prediction heads 傳遞。
這是小資料 regime 的容量約束，不應描述成所有 Mamba 參數完全凍結。

基礎單幀 head 為 10,126,636 parameters。加入訓練期 temporal blocks
後 checkpoint 為 11,368,540 parameters。

## 3. Temporal Shaping

### 3.1 Temporal training operator

對長度 \(T\) 的 clip，在每個空間位置 \((x,y)\) 組成時間序列：

\[
Q_{b,l,x,y} =
\left[
U_{b,1,x,y}^l,\ldots,U_{b,T,x,y}^l
\right]
\in\mathbb{R}^{T\times d}.
\]

Temporal Mamba block 施加 residual mixing：

\[
\hat{Q}_{b,l,x,y}
=
\operatorname{Mamba}_t(Q_{b,l,x,y})+Q_{b,l,x,y}.
\]

所有影格皆使用各自 GT 監督：

\[
\mathcal{L}_{T}
=
\frac{1}{T}
\sum_{\tau=1}^{T}
\mathcal{L}_{\mathrm{det}}
\left(
h_{\theta,\phi}(\mathcal{F}_{t+\tau}),
Y_{t+\tau}
\right),
\]

其中 \(\theta\) 是部署時保留的 spatial path，\(\phi\) 是訓練期 temporal
blocks；\(\mathcal{L}_{\mathrm{det}}\) 為 YOLO v8 detection loss
（classification、box regression 與 DFL）。

### 3.2 T3-to-T1 curriculum

Plain baseline 與 curriculum 使用相同 GT1 initialization 及總計
30 epochs：

```text
Plain GT2:
    T=1, 30 epochs

T3-to-T1:
    Phase A: T=3, temporal blocks enabled, 15 epochs
    Phase B: T=1, temporal blocks bypassed, 15 epochs
```

演算法可寫成：

```text
Input:
    GT1 checkpoint theta_0
    cached backbone features and labels

Phase A:
    initialize temporal parameters phi
    for epoch = 1 ... 15:
        sample clips of length 3, stride 6
        update theta and phi using all-frame detection loss

Phase B:
    discard temporal execution but retain learned spatial parameters theta
    for epoch = 1 ... 15:
        sample single frames, stride 2
        update theta using single-frame detection loss

Output:
    single-frame spatial detector theta_star
```

Phase A 的作用是假設性的 temporal consistency pressure；Phase B 用來消除
train/inference temporal mismatch，並將可用解收斂到 T1 deployment
manifold。

## 4. Single-Frame Deployment

部署時固定 \(T=1\)，程式中的 temporal 條件 \(T>1\) 不成立：

\[
D_t = h_{\theta^\star}(\mathcal{F}_t),\qquad
\phi\ \text{bypassed}.
\]

因此：

- 不需要前一幀 feature buffer；
- 不執行 temporal Mamba blocks；
- 不引入 temporal recurrent state；
- T3-to-T1 與 plain head 使用相同單幀 execution path。

Whole-detect CUDA graph 封裝：

\[
I_t
\rightarrow
\operatorname{Resize}
\rightarrow
\operatorname{TRTBackbone}
\rightarrow
h_{\theta^\star}
\rightarrow
\operatorname{Decode/NMS}.
\]

其後使用固定的 GMC 與 GPU tracker：

\[
D_t
\rightarrow
\operatorname{GMC}
\rightarrow
\operatorname{IoUAssociation}
\rightarrow
\operatorname{BridgeRelink}
\rightarrow
\mathcal{T}_t.
\]

GMC、tracker 與 bridge 是評估後端，不是 T3-to-T1 的組成。它們必須在
所有 detection-head ablations 中保持固定。

## 5. Mechanism Tests

論文主張不是只由最終分數推測，而由以下對照界定。

### 5.1 Shaping versus temporal inference

\[
\Delta_{\mathrm{stream}}
=
\operatorname{IDF1}(\text{Phase-A},T=3)
-
\operatorname{IDF1}(\text{Phase-A},T=1)
\approx 0.74.
\]

Phase-A T3 streaming 仍比 final T3-to-T1 低約 5.6 IDF1，且吞吐由
103.6 FPS 降至 46.6 FPS。主要增益因此來自訓練 curriculum，而不是
temporal inference。

### 5.2 Consistency versus identity discrimination

若增益來自 identity embedding，Mamba feature cosine similarity 應能區分
same-ID 與 different-ID candidate。然而 hard-pool AUC 僅 0.438，且和
plain head 相差約 0.001。

因此目前支持的傳導路徑是：

\[
\text{temporal shaping}
\rightarrow
\text{box/score consistency}
\rightarrow
\text{IoU association stability}
\rightarrow
\text{higher IDF1/AssA}.
\]

### 5.3 Curriculum-order boundary

T3-to-T1 後再執行 full-gradient SSM fine-tuning：

\[
\mathrm{DetA}\uparrow,\qquad
\mathrm{AssA}\downarrow.
\]

反向順序只能部分恢復 AssA，權重插值亦無協同峰。這表示 consistency
solution 位於局部且脆弱的權重區域，最後一段 training objective
決定部署特徵構型。

## 6. What Is and Is Not the Proposed Algorithm

**主算法：**

1. frozen-SSM spatial Mamba detection head；
2. T3 temporal shaping；
3. T1 spatial readaptation；
4. stateless single-frame deployment。

**固定評估後端：**

- YOLO backbone；
- GMC；
- GPU ByteTrack association；
- bridge relink。

**目前不納入主算法：**

- ReID embedding；
- dynamic trigger；
- lifecycle variants；
- detail branch；
- full-gradient SSM fine-tuning；
- model soup。

這個邊界可避免將多個獨立模組的增益合併成無法歸因的方法。

## 7. Implementation Map

| Method component | Source |
|---|---|
| Spatial/temporal Mamba head | `src/saccade/perception/temporal_yolo/mamba_head.py` |
| Detector wrapper and T1 bypass | `src/saccade/perception/temporal_yolo/mamba_gated_detector.py` |
| GT training and all-frame loss | `scripts/train/temporal_yolo/train_mamba_gt.py` |
| T3-to-T1 recipe | `scripts/train/temporal_yolo/run_v14replica_t3t1_seed.sh` |
| Whole-graph evaluation | `src/saccade/perception/eval/evaluator.py` |
| Training protocol | `docs/modules/detection/mamba-v14-replication-protocol.md` |
| Mechanism study | `docs/modules/detection/research/mamba-t3t1-curriculum-20260613.md` |
| Recomputed results | `report_data/tables/` |

## 8. Evidence Boundary

目前所有結果仍屬 replication/development evidence：

- lineage 使用全部七個 MOT17-SDP train sequences；
- 僅兩組 checkpoint metadata 可確認為嚴格同 seed 配對；
- 尚缺 equal-budget CNN/MLP、temporal convolution 與 attention controls；
- 尚缺第二資料集及正式 held-out benchmark。

詳細限制見 [evidence_and_limitations.md](evidence_and_limitations.md)。
