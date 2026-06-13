# Saccade Pipeline Algorithms Reference

> 對應實作：`src/saccade/perception/eval/`、`src/tracking/`、`include/tracking/`
> 每節標註對應原始碼檔案與行號，方便交叉參照。

---

## 1. Detection Quality Scoring

**來源**：`quality.py:5–44`、`tracker_gpu.cu:173–208`

對每一個檢測框 \( b = (x_1, y_1, x_2, y_2) \)，定義幾何品質分數 \( Q(b) \in [0, 1] \)：

\[
\begin{aligned}
w &= x_2 - x_1, \quad h = y_2 - y_1 \\[4pt]
\text{aspect ratio} &\quad a = h / w \\
Q_{\text{asp}} &= \exp\left(-\frac{1}{2}\left(\frac{a - 2.5}{1.2}\right)^2\right) \\[6pt]
\text{center normalized} &\quad c_x = \frac{x_1 + x_2}{2W},\quad c_y = \frac{y_1 + y_2}{2H} \\
Q_{\text{ctr}} &= \operatorname{clip}\!\left(
4 \cdot \min(c_x, 1 - c_x, c_y, 1 - c_y),\ 0,\ 1
\right) \\[6pt]
\text{area ratio} &\quad \rho = \frac{w \cdot h}{W \cdot H} \\
Q_{\text{area}} &= \exp\left(-\frac{1}{2}\left(\frac{\rho - 0.01}{0.01}\right)^2\right) \\[6pt]
Q(b) &= 0.50 \cdot Q_{\text{asp}} + 0.30 \cdot Q_{\text{ctr}} + 0.20 \cdot Q_{\text{area}}
\end{aligned}
\]

預設將原始檢測分數乘以 \( Q(b) \) 作為 quality boost。w_aspect、w_center、w_area 可透過 env var 調整。

---

## 2. ByteTrack GPU — Kalman Filter

**來源**：`include/tracking/kalman_gpu.cuh:1–297`

### 2.1 狀態向量

8 維狀態空間（constant velocity model）：

\[
\mathbf{x} = [c_x,\; c_y,\; a,\; h,\; \dot{c}_x,\; \dot{c}_y,\; \dot{a},\; \dot{h}]^T
\]

其中 \( c_x, c_y \) 為框中心，\( a = w/h \) 為寬高比，\( h \) 為高度。

### 2.2 預測步 (Predict)

\[
\begin{aligned}
\mathbf{x}_{k|k-1} &= \mathbf{F} \mathbf{x}_{k-1|k-1}, \quad
\mathbf{F} =
\begin{bmatrix} \mathbf{I}_4 & \mathbf{I}_4 \\ \mathbf{0} & \mathbf{I}_4 \end{bmatrix} \\[6pt]
\mathbf{P}_{k|k-1} &= \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}
\end{aligned}
\]

過程噪聲 \( \mathbf{Q} \) 依預測後的高度 \( h = x_3 \) 縮放：

\[
\begin{aligned}
\sigma_{\text{pos}} &= h / 20, \quad \sigma_{\text{vel}} = h / 160 \\
\mathbf{Q} &= \operatorname{diag}(\sigma_{\text{pos}}^2,\; \sigma_{\text{pos}}^2,\; 10^{-4},\; \sigma_{\text{pos}}^2,\;
\sigma_{\text{vel}}^2,\; \sigma_{\text{vel}}^2,\; 10^{-10},\; \sigma_{\text{vel}}^2)
\end{aligned}
\]

### 2.3 測量模型

測量向量 \( \mathbf{z} = [c_x^{\text{det}}, c_y^{\text{det}}, a^{\text{det}}, h^{\text{det}}]^T \) 由檢測框轉換：

\[
c_x^{\text{det}} = \frac{x_1 + x_2}{2},\quad c_y^{\text{det}} = \frac{y_1 + y_2}{2},\quad
a^{\text{det}} = \frac{w}{h},\quad h^{\text{det}} = y_2 - y_1
\]

觀測矩陣 \( \mathbf{H} = [\mathbf{I}_4 \; \mathbf{0}] \)。

測量噪聲 \( \mathbf{R} \)（支援 NSA-Kalman 與 lightness factor）：

\[
\mathbf{R} = r_{\text{scale}} \cdot \eta_{\text{nsa}} \cdot (1 + 2\lambda_{\text{light}}) \cdot
\operatorname{diag}(\sigma_{\text{pos}}^2,\; \sigma_{\text{pos}}^2,\; 10^{-2},\; \sigma_{\text{pos}}^2)
\]

其中 \( \eta_{\text{nsa}} = \max(0.05, (1 - s_{\text{det}})^2) \) 為噪聲自適應乘數（Noise Scale Adaptive）。

### 2.4 更新步 (Update)

\[
\begin{aligned}
\mathbf{S} &= \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R} \quad\text{(innovation covariance)} \\
\mathbf{K} &= \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}^{-1} \quad\text{(Kalman gain)} \\
\mathbf{y} &= \mathbf{z} - \mathbf{H} \mathbf{x}_{k|k-1} \quad\text{(innovation)} \\
\mathbf{x}_{k|k} &= \mathbf{x}_{k|k-1} + \mathbf{K} \mathbf{y} \\
\mathbf{P}_{k|k} &= (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}_{k|k-1}
\end{aligned}
\]

4×4 矩陣求逆以 Cramer's rule 展開優化（`kalman_gpu.cuh:9–142`）。

---

## 3. Global Motion Compensation (GMC)

**來源**：`gmc_kernel.cu:1–354`、`gmc.py:10–77`

### 3.1 GPU Phase Correlation 路徑

對兩個連續幀計算平移位移：

**Step 1 — 灰階降採樣 + Hanning 窗**：
\[
G_k(x, y) = \big(0.299R + 0.587G + 0.114B\big) \cdot w_x \cdot w_y
\]
\[
w_x = \tfrac{1}{2}\!\left(1 - \cos\frac{2\pi x}{W-1}\right),\quad
w_y = \tfrac{1}{2}\!\left(1 - \cos\frac{2\pi y}{H-1}\right)
\]

**Step 2 — Cross-Power Spectrum**：
\[
C(u, v) = \frac{\mathcal{F}\{G_{k-1}\}^* \cdot \mathcal{F}\{G_k\}}{|\mathcal{F}\{G_{k-1}\}^* \cdot \mathcal{F}\{G_k\}| + \varepsilon}
\]

**Step 3 — Inverse FFT + Peak Detection**：
\[
r(x, y) = \mathcal{F}^{-1}\{C(u, v)\}
\]
次像素峰值定位（3×3 質量中心）：
\[
x_{\text{peak}} = \frac{\sum_{dx,dy \in \{-1,0,1\}} v(dx,dy) \cdot (x_0 + dx)}{\sum v(dx,dy) + \varepsilon}
\]

**Step 4 — PCR Quality Check**：
\[
\text{PCR} = \frac{\max(r)}{\text{RMS}(r)} > 5 \quad\text{(否則標記為 uncertain)}
\]

**Step 5 — 2×3 Affine Translation Warp**：
\[
\mathbf{W}_{\text{gmc}} =
\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix},
\quad t_x = x_{\text{peak}} \cdot s_{\text{downscale}},\; t_y = y_{\text{peak}} \cdot s_{\text{downscale}}
\]

### 3.2 Python Sparse Optical Flow 路徑 (fallback)

使用 OpenCV Lucas-Kanade 光流法：

1. `cv2.goodFeaturesToTrack()` 提取特徵點
2. `cv2.calcOpticalFlowPyrLK()` 追蹤
3. `cv2.estimateAffinePartial2D()` 擬合 2×3 仿射矩陣

### 3.3 GMC Warp 應用於 Track State

對每個活躍軌跡的 Kalman 狀態施加 GMC 補償（`tracker_gpu.cu:73–96`）：

\[
\begin{bmatrix} x_0' \\ x_1' \end{bmatrix} =
\mathbf{H}_{\text{gmc}} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} + \mathbf{t}_{\text{gmc}},\quad
\mathbf{H}_{\text{gmc}} = \begin{bmatrix} w_{00} & w_{01} \\ w_{10} & w_{11} \end{bmatrix}
\]
\[
\begin{bmatrix} x_4' \\ x_5' \end{bmatrix} =
\mathbf{H}_{\text{gmc}} \begin{bmatrix} x_4 \\ x_5 \end{bmatrix}
\]
協方差矩陣對應區塊同樣以 \(\mathbf{H}_{\text{gmc}}\) 旋轉：\(\mathbf{P}' = \mathbf{H}_{\text{gmc}} \mathbf{P} \mathbf{H}_{\text{gmc}}^T\)。

---

## 4. Association — Sinkhorn-Auction Hybrid

**來源**：`include/tracking/sinkhorn.hpp:1–103`、`tracker_gpu.cu:556–699`

### 4.1 Two-Stage Gating

**Stage 1 Gate**（`tracker_gpu.cu:272–301`）：檢測-軌跡對 \((t, d)\) 通過若：

\[
\operatorname{IoU}(\text{pred}_t, \text{det}_d) > \tau_{\text{iou}}
\quad\text{OR}\quad
\operatorname{Mahalanobis}^2(\text{pred}_t, \text{det}_d) < \tau_{\text{maha}}
\]

Mahalanobis 距離定義為：
\[
D^2_M = (\mathbf{z}_d - \mathbf{H}\mathbf{x}_t)^T \mathbf{S}_t^{-1} (\mathbf{z}_d - \mathbf{H}\mathbf{x}_t)
\]
其中 \(\mathbf{S}_t\) 為 innovation covariance (預先計算於 `compute_S_inv`)。

若提供 homography 矩陣，改用 2D ground-plane MMD：
\[
D^2_{\text{MMD}} = \| \Pi(\mathbf{p}_t^{\text{bottom}}) - \Pi(\mathbf{p}_d^{\text{bottom}}) \|^2 \cdot 0.01
\]

### 4.2 Cost Matrix（無 ReID）

\[
C_{t,d} = 1 - \operatorname{IoU}(t, d) \cdot (1 - w_{\text{score}} \cdot s_d)
\]

附加項 (OC-SORT)：
- **Velocity direction penalty**：\( C_{t,d} \mathrel{+}= w_{\text{vel}} \cdot \max(0, -\cos\theta) \)
  其中 \( \cos\theta = \frac{\mathbf{v}_t \cdot (\mathbf{p}_d - \mathbf{p}_t)}{\|\mathbf{v}_t\| \cdot \|\mathbf{p}_d - \mathbf{p}_t\|} \)
- **OAO occlusion penalty**：\( C_{t,d} \mathrel{+}= \tau_{\text{oao}} \cdot \max_{j \neq t} \operatorname{IoU}(\text{pred}_t, \text{pred}_j) \)

### 4.3 Cost Matrix（有 ReID）

當軌跡擁有乾淨 embedding 且 candidate count ≥ 門檻時：

\[
C_{t,d} = 1 - \big( w_{\text{cos}} \cdot \cos(\mathbf{e}_t, \mathbf{e}_d) + w_{\text{iou}} \cdot \operatorname{IoU}(t, d) + w_{\text{score}} \cdot s_d \big)
\]

附加 decay：\( \cos' = \cos \cdot \exp\left(-2 \cdot \frac{\|\mathbf{c}_t - \mathbf{c}_d\|^2}{\tau_{\text{gate}}^2}\right) \)

### 4.4 Sinkhorn 演算法

給定 cost matrix \(\mathbf{C} \in \mathbb{R}^{N \times M}\)（擴展為方陣 \(n = \max(N, M)\)，padding 以 cost=1），
以熵正則化參數 \(\lambda = 30\) 求解最優傳輸。

**親和矩陣**：
\[
\mathbf{K}_{ij} = \exp(-\lambda \mathbf{C}_{ij})
\]

**Sinkhorn-Knopp 迭代**（最多 50 次）：
\[
\mathbf{u}^{(k+1)} = \frac{1}{\mathbf{K} \mathbf{v}^{(k)} + \varepsilon},\quad
\mathbf{v}^{(k+1)} = \frac{1}{\mathbf{K}^T \mathbf{u}^{(k+1)} + \varepsilon}
\]
初始 \( \mathbf{u}^{(0)} = \mathbf{v}^{(0)} = 1/n \)。

**軟分配矩陣**：
\[
\mathbf{P}_{ij} = \mathbf{u}_i \cdot \mathbf{K}_{ij} \cdot \mathbf{v}_j
\]

**貪婪解碼**：將 \(\mathbf{P}_{ij}\) 由大至小排序，依序分配未使用的 bidder-item 對。

### 4.5 Parallel Auction 演算法

GPU 向量化拍賣演算法（`tracker_gpu.cu:645–699`）：

**Top-K 預選** (fused Sinkhorn top-k kernel)：每條軌跡 \(t\) 保留親和力最高的 \(K=3\) 個檢測。

**兩層次投標**：
- Level 1：block 內 shared memory `atomicMax` 解決衝突
- Level 2：block 勝出者以 global `atomicMax` 提交最終投標

設初始價格 \( p_j^{(0)} = 0 \)，對軌跡 \(t\) 的最佳檢測 \(d^*\)：
\[
\text{bid}_t = p_{d^*} + (v_t^{(1)} - v_t^{(2)} + \varepsilon)
\]
其中 \( v_t^{(k)} = \mathbf{P}_{t,d_k} - p_{d_k} \) 為淨價值（扣除當前價格）。

價格更新（取較高投標）：
\[
p_{d^*} \leftarrow \max\big(p_{d^*},\; \text{bid}_t\big)
\]

Tie-breaking：`bid` 的下 32 bits 編碼 \( (n_{\text{trk}} - t) \)，確保分數相同時優先分配較晚的軌跡。

---

## 5. Multi-Signal Birth

**來源**：`multi_birth.py:48–179`

### 5.1 聯合證據模型

對於分數低於 `new_track_thresh` 但高於 `min_score` 的次閾值檢測，
跨幀追蹤候選人，當累積證據超過門檻時將其分數提升為正常出生分數。

**證據公式**（各項均歸一化至 \([0, 1]\)）：

\[
E = w_{\text{score}} \cdot \bar{s} + w_{\text{motion}} \cdot \bar{m} + w_{\text{quality}} \cdot g + w_{\text{streak}} \cdot \sigma
\]

**Score 項**：
\[
\bar{s} = \min\!\left(1, \max\!\left(0, \frac{s_{\text{best}} - s_{\text{min}}}{s_{\text{thresh}} - s_{\text{min}}}\right)\right)
\]
其中 \( s_{\text{best}} \) 為該候選人的歷史最高分數。

**Motion 項**（平均幀間中心位移）：
\[
\bar{m} = \min\!\left(1, \frac{1}{n-1}\sum_{i=1}^{n-1} \frac{\|\mathbf{c}_i - \mathbf{c}_{i-1}\|}{d_{\text{target}}}\right)
\]

**Geometry 項** \(g\)：
\[
g(a) = \begin{cases}
0 & a < 1.0 \\
a - 1 & 1.0 \le a < 2.0 \\
1.0 & 2.0 \le a \le 4.0 \\
\max(0, 1 - \frac{a-4.0}{3.0}) & a > 4.0
\end{cases}
\]
其中 \( a = h/w \) 為高寬比。另有 min_aspect / max_area 硬拒絕閘。

**Streak 項**：
\[
\sigma = \min\!\left(1, \frac{n-1}{\max(n_{\text{min}}-1, 1)}\right)
\]

### 5.2 出生觸發條件

若 \( E \ge E_{\text{threshold}} \)（預設 0.60）：將該檢測分數提升至 `new_track_thresh + ε`，允許 ByteTrack 將之作為新軌跡出生。

若 \( E \ge E_{\text{replace}} \)（預設 0.85）：於 replace mode 下，同時抑制同一幀中競爭的高分檢測（以 IoU 匹配判定競爭）。

---

## 6. Semantic Relink — Identity Resolution

**來源**：`relink.py:1–1595`、`tracker_gpu_python.cpp:438–1645`（C++ `SemanticRelinkerCpp`）

> **C++ 移植 （2026-06）**：`SemanticRelinker` 自 `relink.py` 遷移至 `tracker_gpu_python.cpp`（`SemanticRelinkerCpp`，行 438–1645），
> 包含雙向重鏈接（bidirectional）、物理卡爾曼閘、碰撞拆分等全部功能。
> 遷移後 C++ 路徑於 MOT17-SDP 全量結果與 Python 位級对齐（MOTA 77.7%，ID 差 1）。

### 6.1 Feature EMA

對每個 canonical identity 維護外觀嵌入的指數移動平均：

\[
\mathbf{e}_c^{(t)} = \beta \cdot \mathbf{e}_c^{(t-1)} + (1 - \beta) \cdot \mathbf{e}_{\text{query}},\quad \beta = 0.83
\]

可選 buffer mode：維護最多 `buffer_size` 個歷史嵌入，以 mean/max/top2-mean/weighted 計算相似度（`rerank_mode`）。

### 6.2 Unified Joint Score

對候選人 \(c\)（lost identity）與當前檢測 \(q\)（raw track）計算聯合分數：

\[
S(c, q) = w_{\text{sim}} \cdot \cos(\mathbf{e}_c, \mathbf{e}_q) + w_{\text{iou}} \cdot \operatorname{IoU}(b_c, b_q) + w_{\text{maha}} \cdot M_{\text{score}} + B_{\text{motion}}
\]

其中 \( M_{\text{score}} = \max(0, 1 - \frac{D^2_M}{\tau_{\text{maha}}}) \) 為 Mahalanobis 歸一化分數。

**動態權重調整**（含 ambiguity 與 lost age 偏移）：
\[
\begin{aligned}
\alpha &= \min\!\left(1, \frac{n_{\text{gate\_pass}} - 1}{8}\right) \quad\text{(ambiguity factor)} \\
\lambda &= \min\!\left(1, \frac{\text{lost\_frames}}{\text{ttl}}\right) \quad\text{(age factor)} \\
w_{\text{sim}}' &= w_{\text{sim}} + \delta_{\text{amb}} \cdot \alpha + \delta_{\text{age}} \cdot \lambda \\
w_{\text{iou}}' &= w_{\text{iou}} - \delta_{\text{amb}} \cdot \alpha - \delta_{\text{age}} \cdot \lambda
\end{aligned}
\]

正規化：\( w_i \leftarrow w_i / \sum w_j \)。

### 6.3 Motion Bonus

基於 EMA velocity/acceleration 模型預測軌跡位置，計算預測框與檢測框的 IoU：

\[
B_{\text{motion}} = w_{\text{motion\_iou}} \cdot \operatorname{IoU}(\text{pred}_c, b_q)
\]

僅當 motion consistency 檢查通過時計入。

### 6.4 Reject Gate Chain

侯選人必須通過多層閘門才能被考慮為匹配：

1. **Age gate**：`lost_frames ∈ [min_lost, ttl]`
2. **Physical speed gate**：估計速度不超過人類極限 \(v_{\text{max}}\) m/s
3. **Spatial gate** (static fallback)：center_norm ≤ spatial_gate AND IoU ≥ min_iou
4. **Kalman probabilistic gate**：\(D^2_M(\text{extrapolated}, b_q) \le \chi^2_{\text{crit}}\)
5. **Velocity direction gate**：\(\cos(\mathbf{v}, \mathbf{p}_q - \mathbf{p}_{\text{last}}) \ge \cos_{\text{min}}\)
6. **Similarity gate**：\(\cos(\mathbf{e}_c, \mathbf{e}_q) \ge \tau_{\text{sim}}\)
7. **Speed-weighted foot-bridge**：對稱完整外推殘差與空間鄰近的速度加權混合（見 6.6）
8. **Reciprocal margin**：\(S_{\text{best}} - S_{\text{second}} \ge \tau_{\text{margin}} + \Delta_{\text{crowd}} + \Delta_{\text{age}}\)

### 6.5 Appearance-First 模式

當 `experimental_mode == "appearance_first"` 時，高相似度候選人 (\( \cos \ge \tau_{\text{af}} \)) 可繞過 spatial gate（appearance-first bypass）。

### 6.6 Speed-Weighted Foot-Bridge Relink（Python + C++）

雙向重鏈接：對每一對 (lost_id, cand_id)，以 4 幀 regressed velocity 計算「對稱完整外推殘差」與「空間鄰近」，並依 lost 退出速度做凸組合。速度向量在 MOT17 慢速行人下是噪聲（per-frame 中位 ~0.01 h/f，低於 box 抖動），故只在快速軌道才信任外推；慢速退化為純空間鄰近。

GPU 實作見 `relink_bidir_propose_kernel`（`tracker_gpu.cu`），C++/CPU 見 `SemanticRelinkerCpp::midpoint_bridge_dist()` 與 `regress_velocity_4()`（`tracker_gpu_python.cpp`）。

\[
\begin{aligned}
\mathbf{v}_{\text{lost}} &= \operatorname{Regress4}(\mathbf{p}_{-4}, \dots, \mathbf{p}_{-1}), \quad
\mathbf{v}_{\text{cand}} = \operatorname{Regress4}(\mathbf{p}_{0}, \dots, \mathbf{p}_{3}) \\
r_{\text{fwd}} &= \tfrac{\|(\mathbf{p}_{\text{lost}} + \mathbf{v}_{\text{lost}}\,g) - \mathbf{p}_{\text{cand}}\|}{h_{\text{ref}}}, \quad
r_{\text{bwd}} = \tfrac{\|(\mathbf{p}_{\text{cand}} - \mathbf{v}_{\text{cand}}\,g) - \mathbf{p}_{\text{lost}}\|}{h_{\text{ref}}} \\
d_{\text{spatial}} &= \frac{\|\mathbf{p}_{\text{lost}} - \mathbf{p}_{\text{cand}}\|}{h_{\text{ref}}}, \quad
s_{\text{lost}} = \frac{\|\mathbf{v}_{\text{lost}}\|}{h_{\text{ref}}}, \quad
w = \operatorname{clip}\!\Big(\sqrt{s_{\text{lost}}/0.12},\, 0,\, 1\Big) \\
d_{\text{bridge}} &= w \cdot \tfrac{1}{2}(r_{\text{fwd}} + r_{\text{bwd}}) + (1 - w)\, d_{\text{spatial}}
\end{aligned}
\]

接受條件 \(d_{\text{bridge}} \le \texttt{bridge\_px}\)（預設 0.25，最佳 0.25–0.30 平台）。\(g\) 為 gap（幀）。4 幀速度回歸（closed form）：\(v_x = (3x_3 + x_2 - x_1 - 3x_0)/10\)。

線上驗證（MOT17 train SDP, `mamba_whole_graph`, `--relink-bridge-enabled`，含 `margin=0.05` + scale gate `[0.75, 1.33]`）：**IDF1 75.1 / HOTA 68.2 / AssA 66.6 / IDs 482**（vs. no-bridge baseline 73.3 / 66.7 / 64.7）；詳見 `docs/modules/semantic/research/offline_relink_candidate_analysis.md`。

---

## 7. Detection Postprocessing Pipeline

**來源**：`detection.py:1–1400`、`tracking/pipeline.cpp`

### 7.1 Letterbox Preprocessing

將原始幀 resize 至 \(960 \times 960\)，保持長寬比，灰邊填充：
\[
r = 960 / \max(H_{\text{orig}}, W_{\text{orig}})
\]

### 7.2 NMS (Non-Maximum Suppression)

CUDA 加速的 IoU-NMS：
\[
\forall b_i, b_j \in \text{detections}, \quad
\operatorname{IoU}(b_i, b_j) > \tau_{\text{nms}} \implies \text{保留 } \arg\max(s_i, s_j)
\]

### 7.3 Cross-Tile Merge (Tiled Detection)

當使用 \(960_p\_2\!\times\!2\) 或 \(960_p\_3\!\times\!2\) tiling 時：
- Seam-aware 判定：對 seam 附近的 box pair 放寬 duplicate 閾值
- 融合座標偏低非 seam 候選（seam boxes 降低權重）
- Cross-tile score penalty：合併後的框依位置不確定性降低分數

### 7.4 Quality Gates

| Gate | 條件 | 動作 |
|------|------|------|
| FP hard filter | 低分 + 大面積 | 移除此類可疑 FP |
| Detection cap | per-frame count > limit | 保留高分前 N 個 |
| Stage2 quality gate | mid-score 區間 + 幾何不良 | 移除 |
| Consecutive birth gate | 跨幀出現的 sub-threshold 框 | 提升分數 |
| Birth quality gate | 高品質 sub-threshold 框 | 提升分數 |

---

## 8. 符號索引

| 符號 | 意義 | 預設值 |
|------|------|--------|
| \(\mathbf{x}_t\) | Kalman 狀態向量 (8D) | — |
| \(\mathbf{P}_t\) | 卡爾曼協方差 (8×8) | — |
| \(\mathbf{z}_d\) | 檢測測量向量 (4D) | — |
| \(\mathbf{Q}\) | 過程噪聲 | \(h\)-adaptive |
| \(\mathbf{R}\) | 測量噪聲 | \(h\)-adaptive, NSA |
| \(\mathbf{W}_{\text{gmc}}\) | GMC 仿射 warp (2×3) | — |
| \(\tau_{\text{iou}}\) | Stage-1 IoU 閘值 | 0.01 |
| \(\tau_{\text{maha}}\) | Stage-1 Mahalanobis 閘值 | 9.4877 |
| \(\lambda\) | Sinkhorn 正則化參數 | 30 |
| \(\varepsilon\) | Auction 最小增量 | — |
| \(\beta\) | ReID 嵌入 EMA 係數 | 0.83 |
| \(E_{\text{threshold}}\) | Multi-birth 證據門檻 | 0.60 |

---

*最後更新：2026-06-06，摘自 saccade 主線程式碼。本文是完整管線參考，
不是目前論文的主算法描述。*
