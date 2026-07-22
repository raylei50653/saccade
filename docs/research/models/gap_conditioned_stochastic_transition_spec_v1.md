<!-- doc-status: draft -->
<!-- doc-promotion: none; D1 canonical model specification (seed); §2–§3 frozen at WP-A1, §4 frozen at WP-A2 -->
<!-- doc-date: 2026-07-22 -->
<!-- doc-module: semantic -->

# Gap-conditioned stochastic transition model — canonical specification (D1, v1)

**用途：** GCTM 的 canonical model specification（charter 所定義的 D1）。本檔由
active charter 的 work packets 逐段填入；**已凍結段落只能經 append-only
correction 修改**。

**Authority 邊界：**

- Task/lifecycle owner：[GCTM task charter](../threads/gap_conditioned_stochastic_transition_model_task.md)
  （Issue [#175](https://github.com/raylei50653/saccade/issues/175)）。
- Production-object 描述來源：[existing-online object analysis](../../modules/semantic/research/existing_online_object_analysis_for_gctm_alignment_20260718.md)
  （supplemental；runtime authority 仍是 `src/tracking/tracker_gpu.cu`）。
- 本檔**不**做 bridge-runtime claim：無 H0 faithful capture、無 accepted
  runtime-fidelity edge；所有指向 production 的欄位都只是 *declared
  correspondence target*，其 runtime 忠實性留待未來 H0/B1 路徑另行建立。
- 不授權 data、fitting、runtime、online、production 工作。

## §1 Document state

| Section | State | Frozen by |
|:--|:--|:--|
| §2 Canonical observation/time interface (nine fields) | **frozen** | WP-A1 (owner merge) |
| §3 Observation modes and causal availability | **frozen** | WP-A1 (owner merge) |
| §4 Canonical state, affine M2 transition \(K_\Delta\), \(Q_\Delta\), parameter domains | **frozen** | WP-A2 (owner merge) |
| §5 Identifiability and leakage matrix | reserved — not yet specified | WP-A3+ |
| §6 Schema-only interface for a future B1 input | reserved — not yet specified | after WP-A5 gates |

Reserved sections carry no obligations-resolved claim. WP-A2 resolves charter
obligation 4 (canonical-state affine M2 transition); obligations 2 and 3 remain
unresolved until their owning packets (WP-A4 / WP-A5) land, and §4 makes no
claim about them.

## §2 Canonical observation/time interface（obligation 1 — nine fields, frozen）

Binding-class vocabulary（描述性使用 charter obligation 3 的 typed relations）：

- **normative** — A 層 canonical convention，由本檔定義；
- **documented-production** — 照抄 existing-online analysis 對 production
  operator 的描述；照抄即權威轉引，**不**構成 runtime-fidelity claim；
- **declared-target** — 指名未來 correspondence 的 production 對象；在 accepted
  fidelity edge 之前地位是 proxy/hypothesis。

| # | Field | Frozen definition | Binding class · source |
|:--|:--|:--|:--|
| 1 | `coordinate_substrate_id` | A 層 latent state 定義在 substrate-agnostic 的 \(\mathbb R^k\)。宣告的 production observation projection 目標為 **`S_A`**：active CUDA bridge 的 native geometry substrate（image-plane position，經 \(h_{\mathrm{ref}}=\max\big(\tfrac{h_L^{\mathrm{ema}}+h_C^{\mathrm{ema}}}{2},1\big)\) 高度正規化；Step-0 substrate audit rev.3 的 A 支）。 | declared-target · substrate audit + existing-online §4.1 |
| 2 | `frame_time_unit` | Canonical 時間單位 = **1 frame interval**（discrete frame index）。所有 gap/horizon 量以 frames 計；A 層不綁 wall-clock/fps；秒級換算須另行宣告映射。 | normative |
| 3 | `physical_gap_definition` | \(g_{\mathrm{phys}}\) := **physical inter-endpoint gap** = lost track 最後被觀測 frame（exit endpoint）到 candidate 第一次被觀測 frame（entry endpoint）之間的 frame-interval 數。以 production 量表示：\(g_{\mathrm{phys}} = \mathrm{gap\_len} = \mathrm{la} - \mathrm{bridge\_at} + 1\)。`gap_len` 在 production 只用於 occupancy/fidelity 路徑，**不是** score-extrapolation factor。 | documented-production · existing-online §3 |
| 4 | `online_horizon_definition` | \(\Delta_{\mathrm{on}}\) := \(\mathrm{la} = \mathrm{age[lost]} \in \mathbb N\)，唯一 production score-extrapolation horizon（進 lost forward / candidate backward extrapolation、speed-gate time、directional blend gap scale）。 | documented-production · existing-online §3（boxed authority）+ §9.5 |
| 5 | `g_phys_to_delta_on_mapping` | Exact identity：\(\boxed{\Delta_{\mathrm{on}} = g_{\mathrm{phys}} + (\mathrm{bridge\_at} - 1)}\)。GCTM 擁有此映射；**兩者永不得默認相等**。Canonical A 層 transition index：\(K_\Delta\) 的 \(\Delta\) = 被建模 transition 兩個 anchor time 之間的 physical elapsed frames；bridge pair 的 canonical anchors 為 **exit endpoint → entry endpoint**，故 canonical \(\Delta = g_{\mathrm{phys}}\)（entry observation \(y_1\)、其不確定度 \(R_1\) 都活在 entry endpoint）。Production 的 **derived construction** 是把 horizon \(\Delta_{\mathrm{on}}=\mathrm{la}\) 套在 exit/entry anchors 上（lost anchor = exit endpoint；candidate anchor \(c_{x0}\) = **entry endpoint**，head-4 的 endpoint 0）——**不是**把 candidate 移到 fire frame；因此 exact-CV null 下誘導出 deterministic **anchor/horizon mismatch offset** \(\pm(\mathrm{bridge\_at}-1)v\)（row 8）。任何 production-corresponding instantiation 必須宣告其 anchor pair、套用上式並重現該 offset。 | normative（mapping 本身）；anchor 事實為 documented-production · `tracker_gpu.cu`（`bridge_anchor4` endpoint 0） |
| 6 | `bridge_at_convention` | `bridge_at` = candidate hit-streak 觸發門檻：bridge event 在 `hit_streak == bridge_at` 時 fire；fire 時 candidate 已連續被觀測 `bridge_at` frames，故 entry endpoint = fire frame − (`bridge_at` − 1)。`bridge_at` 是 runtime-configured 值（當前 production default 4）；所有 identity 使用 event 當下綁定的值，**不**凍結常數。 | documented-production · runtime binding `tracker_gpu_python.cpp` / `assoc_basis.py`（trigger timing） |
| 7 | `continuous_dt_conversion` | Canonical \(dt \equiv 1\) frame interval。連續時間 SDE（M1/M2）參數以 per-frame-unit 表示；離散 transition = 在 \([0,\Delta]\)（\(\Delta\in\mathbb N\) frames）上的 exact integration。物理秒需另行宣告 fps 映射，不在 A 層範圍。 | normative |
| 8 | `production_cv_null_offset` | **Velocity-dependent signed offset pair（一般非零）。** Active operator 以 \(\Delta_{\mathrm{on}}=\mathrm{la}\) 外推，但 candidate anchor \(c_{x0}\) 是 **entry endpoint**（早於 fire frame \(\mathrm{bridge\_at}-1\) frames）。Null model（exact CV、matched identity、\(v_L=v_C=v\)，故 \(p_C=p_L+v\,g_{\mathrm{phys}}\)）下：\(\delta_{\mathrm{fwd}}=(p_L+v\,\mathrm{la})-p_C=+(\mathrm{bridge\_at}-1)v\)，\(\delta_{\mathrm{bwd}}=(p_C-v\,\mathrm{la})-p_L=-(\mathrm{bridge\_at}-1)v\)。僅當 \(\mathrm{bridge\_at}=1\) 或 \(v=0\) 時為零。又 \(dist_h=g_{\mathrm{phys}}\lVert v\rVert/h_{\mathrm{ref}}\)，故 directional branch inactive 時完整 base score 的 null 值 \(b_0=\frac{\lVert v\rVert}{h_{\mathrm{ref}}}\big[w(\mathrm{bridge\_at}-1)+(1-w)\,g_{\mathrm{phys}}\big]\neq 0\)。Legacy「中點外推」\(\mathrm{gap}/2\) 只存在於 Path C mirrors，**不是** production baseline object。 | documented-production · `tracker_gpu.cu`（\(c_{x0}\)=head-4 endpoint 0；residual forms）+ existing-online §4.1 + #251 review（bounded correction） |
| 9 | `null_offset_treatment` | **兩層必須分開，不得互相冒充**：（i）**canonical physical layer** — CV transition 在 \(\Delta=g_{\mathrm{phys}}\)（exit→entry）上是 **zero innovation**；canonical family 的 production-corresponding member 於 \(\gamma\to 0\)（nesting）精確回到此 CV mean。（ii）**production operator layer** — operator 把 \(\Delta_{\mathrm{on}}=\mathrm{la}\) 套在 exit/entry anchors 上，任何 production-corresponding derived construction 必須**重現** row 8 的 deterministic anchor/horizon mismatch offset \(\pm(\mathrm{bridge\_at}-1)v\)，並記為 **operator-layer deterministic offset**——不得誤列為 M2 drift、不得當成 unaccounted bias。M2 在 \(\gamma>0\) 的 mean 偏差 \((a-\Delta)u_0\) 屬 canonical layer 的宣告 model drift，是另一回事。任何其他非零 additive offset（如 \(\bar v(c)\neq 0\) 的 context drift \(d_\Delta(c)\)）必須是顯式宣告的 model component，**不得**被靜默繼承。 | normative |

**Freeze 邊界：** 本表凍結的是 *定義與 convention*。它不宣稱任何 runtime
擷取值、不建立 fidelity edge、不選 model 參數、不解除 charter obligations
2–4。修改須 append-only correction（原行保留、註記 superseded）。

## §3 Observation modes and causal availability（frozen）

沿用 charter 的兩種 observation mode，凍結其地位：

\[
H_x=\begin{bmatrix}I&0\end{bmatrix}
\qquad\text{(position-only)},
\qquad
H_{xv}=I
\qquad\text{(joint position–velocity)}.
\]

1. 兩種 mode 是**不同的 claim object**，永不可互換使用。
2. **Production-corresponding instantiation 綁 \(H_x\)**（position-only entry
   observation，於 entry endpoint 取 \(y_1\)，不確定度 \(R_1\)）。
3. \(H_{xv}\) 只在**顯式宣告 causal availability** 後可用：entry velocity 若需
   lagged/future frames，宣告必須寫明其可得時點與來源；未宣告即不可引用。

## §4 Canonical state, affine M2 transition, and process covariance（obligation 4 — frozen）

本節把 charter 的 M2 residual-velocity narrative 落成 canonical latent state 上
**單一完整的 affine stochastic transition interface**，即 primary object A 的
transition kernel

\[
\boxed{\;
K_\Delta(z_0,c)=\mathcal N\!\big(A_\Delta z_0+d_\Delta(c),\;Q_\Delta\big)
\;}
\]

於 canonical transition index \(\Delta=g_{\mathrm{phys}}\)（§2 field 5：exit
endpoint → entry endpoint；`dt`\(\equiv 1\) frame，§2 field 7）。本節凍結 state、
SDE、\(A_\Delta\)、\(d_\Delta(c)\)、\(Q_\Delta\) 的積分構造與 closed form、
\(\gamma=0\) continuous extension、parameter domains、causal assumptions 與 units。

### §4.0 本節做什麼／不做什麼（typed boundary）

**做（frozen）：** 定義 forward latent-state process transition kernel
\(K_\Delta(z_0,c)\) 的完整 affine 形狀與其所有係數的 closed form 與 domain。

**不做（留給後續 WP，見 §4.8）：** 不做 \(P_0\)/\(R_1\)/\(S_\Delta\) 的 innovation
composition（WP-A4，obligation 3），不宣稱 PSD／nesting／semigroup／asymptotics
的**正式證明**（WP-A3，D2），不做 calibration／ranking claim（WP-A5，obligation
2），不做 reverse-time atom、data、fitting、runtime、online、production。本節出現
的 \(\gamma\to0\) 極限只作為**介面在 \(\gamma=0\) 的定義閉合**（§4.6），不構成
nesting lemma 的完成宣稱。

四條 canonical boundary（與 §2 一致，永不得逾越）：

1. **Canonical index = \(g_{\mathrm{phys}}\)。** \(K_\Delta\) 的 \(\Delta\) 就是
   exit→entry physical elapsed frames \(=g_{\mathrm{phys}}\)（§2 field 5）。
2. **Production horizon 是 separate derived construction。**
   \(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}+(\mathrm{bridge\_at}-1)\)（§2 field 5）
   與其誘導的 operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\)（§2 rows 8–9）
   **不得**進入 canonical \(d_\Delta(c)\)，**不得**被列為 M2 model drift。它們是
   operator layer 的 deterministic offset，另見 §4.4。
3. **Candidate backward atom ≠ canonical reverse-time OU transition。**
   本節只定義 forward kernel \(K_\Delta\)；production 的 candidate-backward
   extrapolation 是 derived operator construction，**不**宣稱為 \(K_\Delta\) 的
   time reversal。
4. **Context 在單一 interval 內固定且 causally available at exit。** \(c\) 與
   \(\bar v(c)\) 在 \([0,\Delta]\) 內為常數，且只能用 exit time 已可得的資訊
   （不得使用 entry／future frames）。

### §4.1 Canonical state and dimension

令 \(d\) = position／velocity coordinate dimension：

\[
x_t,\,v_t,\,\bar v(c)\in\mathbb R^{d},
\qquad
z_t=\begin{bmatrix}x_t\\ v_t\end{bmatrix}\in\mathbb R^{2d}.
\]

這是 frozen D1 §2 field 1 的 substrate-agnostic latent state \(\mathbb R^k\)
（latent state 即 \(z\)）的**具體化**，故

\[
k=2d .
\]

Production-corresponding image-plane instantiation 取 \(d=2\)，故 frozen substrate
dimension \(k=4\)。Position-only observation（§3）的 \(H_x=\begin{bmatrix}I_d&0\end{bmatrix}
:\mathbb R^{2d}\to\mathbb R^{d}\)（§3 的 \(H_x=[\,I\ 0\,]\) 中 \(I=I_d\)）。
Block-matrix 慣例：\(2\times2\) block form，每個 block 為 \(d\times d\)，\(I\) 表
\(I_d\)。**本節不把 \(k\) 重新定義為 spatial dimension**（\(k\) 仍為 §2 的 full
latent-state substrate dimension \(=2d\)）。

### §4.2 M2 SDE（canonical model）

Velocity 分解 \(v_t=\bar v(c)+u_t\)，residual velocity \(u_t\) 為 OU：

\[
\mathrm dv_t=-\gamma\big(v_t-\bar v(c)\big)\,\mathrm dt+L\,\mathrm dW_t,
\qquad
\mathrm dx_t=v_t\,\mathrm dt,
\]

等價地 \(\mathrm du_t=-\gamma u_t\,\mathrm dt+L\,\mathrm dW_t\)。其中 \(\gamma\ge0\)、
\(D=LL^\top\in\mathbb R^{d\times d}\succeq0\)、\(W_t\) 為 standard \(d\)-dim
Brownian motion，\(\{W_t\}_{t\in(0,\Delta]}\) 與 \(z_0\)、\(\bar v(c)\) 獨立。堆成
\(z\)：

\[
\mathrm dz_t=\big(Fz_t+G(c)\big)\,\mathrm dt+B\,\mathrm dW_t,
\quad
F=\begin{bmatrix}0&I\\0&-\gamma I\end{bmatrix},\;
G(c)=\begin{bmatrix}0\\ \gamma\,\bar v(c)\end{bmatrix},\;
B=\begin{bmatrix}0\\ L\end{bmatrix}.
\]

\(\gamma=0\)（\(F\) nilpotent 上三角、drift \(=0\)）即 M1 constant-velocity /
white-acceleration family（§4.6）。

### §4.3 Discrete affine transition — \(A_\Delta\) 與 \(d_\Delta(c)\)

在 \([0,\Delta]\)（\(\Delta\in\mathbb N\) frames）上的 exact integration 給出
affine transition

\[
z_\Delta=A_\Delta z_0+d_\Delta(c)+\eta_\Delta,
\qquad
\eta_\Delta\sim\mathcal N(0,Q_\Delta),
\]

其中 \(A_\Delta=e^{F\Delta}\)，\(d_\Delta(c)=\Big(\int_0^\Delta e^{F u}\,\mathrm du\Big)G(c)\)。
以 \(b=e^{-\gamma\Delta}\)、\(a=\dfrac{1-b}{\gamma}\)（\(a\gamma=1-b\)）：

\[
A_\Delta=\begin{bmatrix}I&aI\\0&bI\end{bmatrix},
\qquad
d_\Delta(c)=\begin{bmatrix}(\Delta-a)\,\bar v(c)\\ (1-b)\,\bar v(c)\end{bmatrix}.
\]

\(d_\Delta(c)\) 是**顯式宣告的 context mean-reversion drift**（§2 row 9 所指的
「\(\bar v(c)\neq0\) 的 context drift」）；\(\bar v(c)=0\) 時 \(d_\Delta(c)=0\)，
\(\gamma\to0\) 時 \(d_\Delta(c)\to0\)（§4.6）。它**不**含、也**不得**吸收任何
operator-layer horizon offset（boundary 2）。

### §4.4 M2 mean evolution 與三個必須分離的量

Transition mean \(m_\Delta:=A_\Delta z_0+d_\Delta(c)\)，位置分量
\(m_\Delta^x=x_0+a\,v_0+(\Delta-a)\bar v(c)\)，速度分量
\(m_\Delta^v=b\,v_0+(1-b)\bar v(c)\)。下列三者是**不同 layer 的不同量，不得互相冒充**：

| 量 | 定義 | Layer / 地位 |
|:--|:--|:--|
| **Canonical M2 model drift** | 位置：\((a-\Delta)\,u_0\)，\(u_0=v_0-\bar v(c)\)（M2 位置 mean 減 CV 位置 mean \(x_0+\Delta v_0\)）。\(\gamma>0\) 時 \(a<\Delta\Rightarrow\) 沿 \(u_0\) 方向的 mean-reversion 收縮；\(\gamma\to0\) 時 \(\to0\)。 | **canonical layer**：OU 的宣告 model drift（§2 row 9）。 |
| **Context drift** | \(d_\Delta(c)\)（§4.3），由 \(\bar v(c)\) 驅動的顯式 model component。 | **canonical layer**：顯式宣告，不得靜默繼承。 |
| **Operator-layer offset** | \(\pm(\mathrm{bridge\_at}-1)v\)（§2 rows 8–9），源自 production 把 \(\Delta_{\mathrm{on}}=\mathrm{la}\) 套在 exit/entry anchors 的 anchor/horizon mismatch。 | **operator layer**：deterministic offset，**非** \(K_\Delta\)、**非** M2 drift。任何 production-corresponding instantiation 必須另行重現它（§2 field 5），但**不得**塞進 \(d_\Delta(c)\)。 |

Canonical layer 的 CV null（exact CV、matched identity）於 \(\Delta=g_{\mathrm{phys}}\)
是 zero innovation（§2 row 9(i)）：\(\gamma\to0\)、\(\bar v(c)\) 任意時 M2 mean 精確
回到 \(x_0+\Delta v_0\)、\(v_0\)。

### §4.5 Process covariance \(Q_\Delta\) — noise-integral 構造與 closed form

**Noise-integral 構造（controllability-gramian / Lyapunov 形）：**

\[
\eta_\Delta=\int_0^\Delta e^{F(\Delta-s)}B\,\mathrm dW_s,
\qquad
Q_\Delta=\operatorname{Cov}(\eta_\Delta)
=\int_0^\Delta e^{F\tau}\,\Sigma\,e^{F^\top\tau}\,\mathrm d\tau,
\quad
\Sigma=\begin{bmatrix}0&0\\0&D\end{bmatrix}.
\]

以 response kernels \(g(\tau)=\dfrac{1-e^{-\gamma\tau}}{\gamma}\)（position）、
\(h(\tau)=e^{-\gamma\tau}\)（velocity），\(e^{F\tau}\) 第二 block-column 為
\([\,g(\tau)I;\,h(\tau)I\,]\)，故

\[
Q_\Delta=
\begin{bmatrix}
\big(\int_0^\Delta g^2\big)D & \big(\int_0^\Delta gh\big)D\\[2pt]
\big(\int_0^\Delta gh\big)D & \big(\int_0^\Delta h^2\big)D
\end{bmatrix}
=
\begin{bmatrix}q_{xx}\,D & q_{xv}\,D\\ q_{xv}\,D & q_{vv}\,D\end{bmatrix}.
\]

**Closed form**（\(b=e^{-\gamma\Delta}\)，\(a=(1-b)/\gamma\)，\(\gamma>0\)）：

\[
q_{vv}=\frac{1-b^2}{2\gamma},
\qquad
q_{xv}=\frac{(1-b)^2}{2\gamma^2},
\qquad
q_{xx}=\frac{1}{\gamma^2}\!\left(\Delta-\frac{2(1-b)}{\gamma}+\frac{1-b^2}{2\gamma}\right)
=\frac{2\gamma\Delta-3+4b-b^2}{2\gamma^3}.
\]

各 block 為 scalar \(\times\,D\)，故 \(Q_\Delta\) 由單一 \(d\times d\) 的
\(D=LL^\top\) 與三個 scalar 完全決定。（\(Q_\Delta\succeq0\) 的**正式 PSD 論證**
屬 WP-A3／D2，本節不宣稱其完成，見 §4.8。）

### §4.6 \(\gamma=0\) continuous extension（M1 boundary）

\(q_{xx},q_{xv},q_{vv},a\) 在 \(\gamma\to0^+\) 皆為 removable singularity
（分子零階抵消 \(1/\gamma^{\,\cdot}\) 極點）。**介面於 \(\gamma=0\) 以下列極限值定義**，
使 \(K_\Delta\) 對所有 \(\gamma\ge0\) 為 total、無未定義式：

\[
a\big|_{0}=\Delta,\quad
A_\Delta\big|_{0}=\begin{bmatrix}I&\Delta I\\0&I\end{bmatrix},\quad
d_\Delta(c)\big|_{0}=0,
\]
\[
q_{vv}\big|_{0}=\Delta,\quad
q_{xv}\big|_{0}=\frac{\Delta^2}{2},\quad
q_{xx}\big|_{0}=\frac{\Delta^3}{3}.
\]

此即 charter §M1 的 \(\Phi_{M1}(\Delta)\)、\(Q_{M1}(\Delta)\) 與 constant-velocity /
white-acceleration family。**注意界線：** 上述極限值是**介面的定義閉合**；把它們
升格為「M2\(\to\)M1 nesting lemma（mean+covariance）已證」的 claim 屬 WP-A3／D2，
本節**不**作此宣稱（§4.8、驗收重點）。

### §4.7 Parameter domains, causal assumptions, units

| Symbol | Domain | Units | Note |
|:--|:--|:--|:--|
| \(d\) | \(\mathbb N_{\ge1}\) | — | coordinate dim；production-corresponding \(=2\)（\(\Rightarrow\) §2 substrate \(k=2d=4\)） |
| \(z_0=[x_0;v_0]\) | \(\mathbb R^{2d}\) | \(x\):\(\ell\)；\(v\):\(\ell/\mathrm{frame}\) | exit-endpoint canonical state；\(\ell=S_A\) 高度正規化位置單位（§2 field 1） |
| \(\gamma\) | \([0,\infty)\) | \(\mathrm{frame}^{-1}\) | mean-reversion rate（scalar，作用為 \(\gamma I_d\)）；\(\gamma=0\)=M1 boundary（§4.6） |
| \(D=LL^\top\) | \(\{M\in\mathbb R^{d\times d}:M\succeq0\}\) | \(\ell^2\,\mathrm{frame}^{-3}\) | white-acceleration diffusion；\(L\in\mathbb R^{d\times m}\)，units \(\ell\,\mathrm{frame}^{-3/2}\) |
| \(\bar v(c)\) | \(\mathbb R^{d}\) | \(\ell/\mathrm{frame}\) | context mean velocity；interval-fixed、exit-time causally available（boundary 4） |
| \(\Delta\) | \(\mathbb N\)（積分對 \([0,\infty)\) 有效） | frame | canonical transition index \(=g_{\mathrm{phys}}\)（boundary 1） |
| \(b,a\) | \(b\in(0,1]\)；\(a\in(0,\Delta]\) | \(b\):—；\(a\):frame | \(b=e^{-\gamma\Delta}\)，\(a=(1-b)/\gamma\)（\(\gamma=0\):\(b=1,a=\Delta\)） |

Dimensional consistency（sanity，非 D2 lemma）：\(A_\Delta\) 的 \(aI\) block 把
\(\ell/\mathrm{frame}\) 映到 \(\ell\)；\(Q_\Delta\) 的 \(q_{xx}D\sim\ell^2\)、
\(q_{xv}D\sim\ell\cdot(\ell/\mathrm{frame})\)、\(q_{vv}D\sim(\ell/\mathrm{frame})^2\)。

**Causal assumptions：** \(\{W_t\}_{(0,\Delta]}\perp z_0\)、\(\perp\bar v(c)\)；
\(c,\bar v(c)\) 在 \([0,\Delta]\) 常數且僅依 exit-time 資訊（boundary 4）。本節**不**
宣告 prediction error 與 entry-observation error 的 independence／cross-covariance
——該決定屬 WP-A4（§4.8）。

### §4.8 本節顯式不解決（typed deferrals）

| 項目 | 擁有 WP | 狀態 |
|:--|:--|:--|
| \(P_0\)/\(Q_\Delta\)/\(R_1\)/\(S_\Delta\) innovation composition；independence 或顯式 \(C\)（obligation 3） | WP-A4 | unresolved |
| \(Q_\Delta\succeq0\) 的 PSD 論證、\(M2\to M1\) nesting、semigroup、short/long-gap asymptotics 的**正式證明**（D2） | WP-A3 | unresolved |
| calibration-only gain vs candidate-local ranking gain 為不同 claim（obligation 2） | WP-A5 | unresolved |
| reverse-time / candidate-backward atom 的 canonical 地位 | 後續 | typed boundary only（boundary 3） |
| B1/O1、H0、runtime、online、production、data、fitting | — | 不授權（charter Non-scope） |

## §5–§6 Reserved

`GCTM_MODEL_SPEC_SEALABLE` 之前必須完成（見 charter frozen terminal partition
與 obligation-status table）；在各自 work packet 落地前，本檔不預先陳述。

## History

- 2026-07-22 — D1 seed created by WP-A1: §2 nine-field canonical
  observation/time interface + §3 observation modes frozen (charter
  obligation 1); §4–§6 reserved.
- 2026-07-22 — bounded correction per #251 owner review (BLOCKED → fixed
  pre-merge; nothing was frozen yet): rows 5/8/9 corrected — candidate anchor
  \(c_{x0}\) is the entry endpoint, so the production exact-CV null offset is
  the signed pair \(\pm(\mathrm{bridge\_at}-1)v\) (zero only at
  \(\mathrm{bridge\_at}=1\) or \(v=0\)), recorded as an operator-layer
  deterministic offset distinct from canonical zero-innovation CV at
  \(\Delta=g_{\mathrm{phys}}\) and from M2 model drift. Verified against
  `tracker_gpu.cu` (`bridge_anchor4` endpoint 0; residual forms).
- 2026-07-22 — **§4 frozen by WP-A2** (charter obligation 4): canonical state
  \(z=[x;v]\in\mathbb R^{2d}\) (coordinate dim \(d\); concretizes §2 substrate
  \(\mathbb R^k\) with \(k=2d\), production \(d=2\Rightarrow k=4\)); M2 SDE
  \(\mathrm dv=-\gamma(v-\bar v(c))\mathrm dt+L\,\mathrm dW\), \(\mathrm dx=v\,\mathrm dt\);
  affine transition \(K_\Delta(z_0,c)=\mathcal N(A_\Delta z_0+d_\Delta(c),Q_\Delta)\)
  at \(\Delta=g_{\mathrm{phys}}\) with \(A_\Delta=[[I,aI],[0,bI]]\),
  \(d_\Delta(c)=[(\Delta-a)\bar v;(1-b)\bar v]\); \(Q_\Delta\) noise-integral
  (\(\int_0^\Delta e^{F\tau}\Sigma e^{F^\top\tau}\mathrm d\tau\)) + closed form
  \(q_{vv}=\frac{1-b^2}{2\gamma}\), \(q_{xv}=\frac{(1-b)^2}{2\gamma^2}\),
  \(q_{xx}=\frac{2\gamma\Delta-3+4b-b^2}{2\gamma^3}\); \(\gamma=0\) continuous
  extension to M1 (\(a\to\Delta\), blocks \(\to\Delta,\Delta^2/2,\Delta^3/3\));
  parameter domains/causal assumptions/units. Canonical drift, M2 mean
  evolution, and the production operator-layer offset
  \(\pm(\mathrm{bridge\_at}-1)v\) kept strictly separated (§4.4); the offset is
  **not** in \(d_\Delta(c)\) and **not** M2 drift. §4.8 defers innovation
  composition/independence (WP-A4), PSD/nesting/asymptotics proofs (WP-A3/D2),
  and calibration-vs-ranking (WP-A5); no such obligation is claimed resolved.
  §2–§3 unchanged. Closed forms verified: \(A_\Delta\) = matrix exponential,
  both \(q_{xx}\) forms = numerical gramian, blocks nest to \(Q_{M1}\) as
  \(\gamma\to0\).
