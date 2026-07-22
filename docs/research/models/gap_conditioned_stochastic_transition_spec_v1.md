<!-- doc-status: draft -->
<!-- doc-promotion: none; D1 canonical model specification (seed); §2–§3 frozen at WP-A1, §4 frozen at WP-A2, §5 frozen at WP-A4 -->
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
| §5 Innovation composition (\(P_0\)/\(P^-_\Delta\)/\(R_1\)/\(S_\Delta\); independence vs explicit \(C\)) | **frozen** | WP-A4 (owner merge) |
| §6 Identifiability and leakage matrix | reserved — not yet specified | WP-A5+ |
| §7 Schema-only interface for a future B1 input | reserved — not yet specified | after WP-A5 gates |

Reserved sections carry no obligations-resolved claim. WP-A2 resolves charter
obligation 4 (canonical-state affine M2 transition); WP-A4 resolves charter
obligation 3 (independence vs explicit cross-covariance \(C\); §5). Obligation 2
(calibration vs ranking claim-space) remains unresolved until its owning packet
WP-A5 lands; §4 makes no claim about obligations 2–3, and §5 makes no claim
about obligation 2.

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

等價地 \(\mathrm du_t=-\gamma u_t\,\mathrm dt+L\,\mathrm dW_t\)。其中 \(\gamma\ge0\)；
noise dimension \(m\ge1\)、\(W_t\in\mathbb R^{m}\) 為 standard \(m\)-dim Brownian
motion、\(L\in\mathbb R^{d\times m}\)、\(D=LL^\top\in\mathbb R^{d\times d}\succeq0\)
（\(m\) 為自由參數，只有 \(D=LL^\top\) 進入 transition covariance；\(m<d\) 時 \(D\)
rank-deficient）；\(\{W_t\}_{t\in(0,\Delta]}\) 與 \(z_0\)、\(\bar v(c)\) 獨立。堆成
\(z\)：

\[
\mathrm dz_t=\big(Fz_t+G(c)\big)\,\mathrm dt+B\,\mathrm dW_t,
\quad
F=\begin{bmatrix}0&I\\0&-\gamma I\end{bmatrix},\;
G(c)=\begin{bmatrix}0\\ \gamma\,\bar v(c)\end{bmatrix},\;
B=\begin{bmatrix}0\\ L\end{bmatrix}\in\mathbb R^{2d\times m},
\]

其中 \(BB^\top=\Sigma=\begin{bmatrix}0&0\\0&D\end{bmatrix}\)（§4.5）。

\(\gamma=0\)（\(F\) nilpotent 上三角、drift \(=0\)）即 M1 constant-velocity /
white-acceleration family（§4.6）。

### §4.3 Discrete affine transition — \(A_\Delta\) 與 \(d_\Delta(c)\)

在 \([0,\Delta]\) 上的 exact integration 給出 affine transition

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

**\(\Delta\) domain 與 \(\Delta=0\) boundary。** 兩個 domain 明確分開：**bridge
evaluation** 取 \(\Delta\in\mathbb N_{\ge1}\)（canonical index \(=g_{\mathrm{phys}}\ge1\)，
§2 field 3）；**analytic family** 取 \(\Delta\in\mathbb R_{\ge0}\)（closed form 與
noise integral 對此連續，且對 §4.6 的 \(\gamma\ge0\) 皆定義）。\(\Delta=0\)
（empty interval）邊界由直接代入 \(b=e^{0}=1\)、\(a=(1-b)/\gamma=0\) 閉合，全部
well-defined：

\[
a_0=0,\quad b_0=1,\quad A_0=I,\quad d_0=0,\quad Q_0=0.
\]

（此邊界對 \(\gamma\ge0\) 一致：\(\gamma=0\) 的極限式在 \(\Delta=0\) 同樣給
\(a=\Delta=0\)、\(Q=0\)。）

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

**Degenerate Gaussian。** 因 \(D=LL^\top\succeq0\)（未必 \(\succ0\)），\(Q_\Delta\)
一般可能 singular，故 kernel \(K_\Delta(z_0,c)=\mathcal N(A_\Delta z_0+d_\Delta(c),Q_\Delta)\)
是**(可能退化的) Gaussian measure**：rank-deficient 時支撐於一個 affine subspace，
對 Lebesgue 無密度。其 well-posed 定義取 Gaussian measure／characteristic
function \(\hat K(\xi)=\exp\!\big(i\xi^\top m_\Delta-\tfrac12\xi^\top Q_\Delta\xi\big)\)，
**不**要求 \(Q_\Delta\) 可逆；任何需要 \(Q_\Delta^{-1}\) 的 innovation／NLL 量另屬
WP-A4（§4.8）。

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
| \(m\) | \(\mathbb N_{\ge1}\) | — | noise dimension；只有 \(D=LL^\top\) 進 covariance |
| \(z_0=[x_0;v_0]\) | \(\mathbb R^{2d}\) | \(x\):\(\ell\)；\(v\):\(\ell/\mathrm{frame}\) | exit-endpoint canonical state；\(\ell=S_A\) 高度正規化位置單位（§2 field 1） |
| \(c\) | \(\mathcal C_{\mathrm{exit}}\) | — | exit-time context set；kernel argument；僅 exit-causally-available 資訊（boundary 4） |
| \(\bar v\) | \(\bar v:\mathcal C_{\mathrm{exit}}\to\mathbb R^{d}\)（measurable） | \(\ell/\mathrm{frame}\) | context mean velocity map；\(\bar v(c)\) interval-fixed（boundary 4） |
| \(\gamma\) | \([0,\infty)\) | \(\mathrm{frame}^{-1}\) | mean-reversion rate（scalar，作用為 \(\gamma I_d\)）；\(\gamma=0\)=M1 boundary（§4.6） |
| \(L\) | \(\mathbb R^{d\times m}\) | \(\ell\,\mathrm{frame}^{-3/2}\) | diffusion factor |
| \(D=LL^\top\) | \(\{M\in\mathbb R^{d\times d}:M\succeq0\}\) | \(\ell^2\,\mathrm{frame}^{-3}\) | white-acceleration diffusion（\(\succeq0\)，未必 \(\succ0\)） |
| \(\Delta\) | bridge eval \(\mathbb N_{\ge1}\)；analytic family \(\mathbb R_{\ge0}\) | frame | canonical transition index \(=g_{\mathrm{phys}}\)（boundary 1）；\(\Delta=0\) boundary §4.3 |
| \(b,a\) | \(b\in(0,1]\)；\(a\in[0,\Delta]\) | \(b\):—；\(a\):frame | \(b=e^{-\gamma\Delta}\)，\(a=(1-b)/\gamma\)（\(\Delta=0\):\(a=0,b=1\)；\(\gamma=0\):\(b=1,a=\Delta\)） |

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

## §5 Innovation composition and total innovation covariance（obligation 3 — frozen）

本節在 frozen D1 §4 forward transition kernel \(K_\Delta\) 之上，凍結 prediction /
entry-observation / innovation 的 uncertainty composition：四個必須分開的
uncertainty object \(P_0,Q_\Delta,R_1,S_\Delta\)、prediction-error 符號約定、
innovation residual，以及 charter obligation 3 要求的「**declare independence 或
define explicit cross-covariance \(C\)**」二選一決定。這是 charter obligation 3 的
落點。

> **狀態來源註：** frozen §4.8 的 typed-deferral 表（列 WP-A3／WP-A4 `unresolved`）是
> **WP-A2 freeze-time 的快照**，依 byte-frozen 規則**不修改**；GCTM 的現行全域狀態以
> §1 document-state 表、本 §5 與 charter updates 為準（WP-A3/D2 已 landed、obligation
> 3 由本節 resolved）。同一文件不因此並存兩個「現行」狀態來源。

### §5.0 本節做什麼／不做什麼（typed boundary）

**做（frozen）：** 定義 \(P_0\)（exit-state estimation uncertainty）、\(P^-_\Delta\)
（prediction-error covariance）、\(R_1\)（entry-observation uncertainty）、
\(S_\Delta\)（total innovation covariance）四個 object 與其 composition；固定
prediction-error 符號 \(e^-=z_\Delta-m^-_\Delta\)；**決定 canonical A 層採
independence（\(C=0\)）**，並凍結 dependent-error 情形所必須改用的 expanded
\(S_\Delta\) 形狀與其符號。

**不做（留給後續 WP）：** 不計算 standardized innovation \(q=r^\top S_\Delta^{-1}r\)、
\(\log\det S_\Delta\)、NLL 或任何 ranking／calibration claim（那是 obligation 2 =
WP-A5；其 shared-\(S_\Delta\) 下 \(q\)/NLL 同序證明屬後續 D2 增量，見 D2 §7）；不做
identifiability／leakage matrix（§6，reserved）；不選 model 參數（\(P_0,R_1\) 的
數值）；不做 reverse-time atom、data、fitting、runtime、online、production；不建立
任何 fidelity edge。

沿用 §4.0 的四條 canonical boundary（永不逾越），本節額外強調兩點：

- **Innovation 形成於 canonical index \(\Delta=g_{\mathrm{phys}}\)**（exit→entry，
  §2 field 5）：entry observation \(y_1\)、其不確定度 \(R_1\) 都活在 entry endpoint
  （§2 field 5、§3.2）；prediction \(m^-_\Delta\) 由 exit-time 資訊經 \(K_\Delta\)
  推到 entry endpoint。
- **Operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\)（§2 rows 8–9、§4.4）不進
  \(m^-_\Delta\)、不進 \(e^-\)、不進 \(S_\Delta\)。** 它是 operator layer 的
  deterministic anchor/horizon mismatch，任何 production-corresponding instantiation
  另行重現（§2 field 5），**不得**被 innovation composition 靜默吸收（§4.0
  boundary 2）。

### §5.1 四個必須分開的 uncertainty object（沿用 charter，凍結地位）

| Object | 定義 | Layer / 來源 |
|:--|:--|:--|
| \(P_0\) | **exit-state estimation uncertainty** = exit-endpoint canonical state \(z_0\) 的估計 covariance，\(P_0=\operatorname{Cov}(\delta z_0)\)、\(\delta z_0=z_0-\hat z_0\)、\(P_0\in\mathbb R^{2d\times2d}\)、\(P_0\succeq0\)；僅依 exit-time 資訊（§4.0 boundary 4）。 | canonical · initial condition |
| \(Q_\Delta\) | **process uncertainty accumulated over the gap** = §4.5 的 process covariance（noise-integral／closed form）。**不重定義**，直接引 §4.5。 | canonical · frozen §4.5 |
| \(R_1\) | **entry-observation uncertainty** = entry endpoint 觀測 \(y_1\) 的 noise covariance，\(\epsilon_1\sim\mathcal N(0,R_1)\)、\(R_1\succeq0\)。維度隨 observation mode：\(H_x\) 下 \(R_1\in\mathbb R^{d\times d}\)（position-space），\(H_{xv}\) 下 \(\in\mathbb R^{2d\times2d}\)。 | canonical · entry-time measurement |
| \(S_\Delta\) | **total innovation covariance** = innovation residual \(r\) 的 covariance（§5.3–§5.4）。 | canonical · derived from 上三者 |

**分離規則（frozen）：** 上四者是**不同 layer 的不同量**，永不得互相冒充或合併記為
單一 scalar（呼應 §2.4 混層禁令與 charter「uncertainty objects must remain
separate」）。\(P_0\) 是 initial-condition covariance、\(Q_\Delta\) 是 transition
noise、\(R_1\) 是 measurement noise、\(S_\Delta\) 是三者經 propagation 後的
innovation covariance；四者不得壓成單一量。

### §5.2 Prediction 與 prediction error（符號約定 frozen）

給定 exit-state 估計 \(\hat z_0\)（\(\operatorname{Cov}(\delta z_0)=P_0\)），canonical
prediction mean 於 entry endpoint 為

\[
m^-_\Delta:=A_\Delta\hat z_0+d_\Delta(c),
\]

即把 §4.3 的 affine transition 套在估計 \(\hat z_0\) 上。**Prediction-error 符號約定
（frozen）：**

\[
\boxed{\,e^-:=z_\Delta-m^-_\Delta\,}
\qquad(\text{state minus prediction；charter obligation 3 / plan Step 3 convention}).
\]

由 \(z_\Delta=A_\Delta z_0+d_\Delta(c)+\eta_\Delta\)（§4.3，\(\eta_\Delta\sim
\mathcal N(0,Q_\Delta)\)）與上式，

\[
e^-=A_\Delta\,\delta z_0+\eta_\Delta .
\]

**Initial-state／process-noise 假設（obligation 3 明列項；declared assumption，非
定理）：**

\[
\boxed{\,\eta_\Delta\perp\delta z_0\,}
\qquad(\text{estimator }\hat z_0\text{ 不攜帶 gap 內 process noise 的資訊}).
\]

**注意這不是 §4.7 的推論。** §4.7 只宣告 \(\{W_t\}_{(0,\Delta]}\perp z_0\)（與
\(\perp\bar v(c)\)）；而 \(W\perp z_0\) **並不**蘊含 \(W\perp\delta z_0\)——
\(\hat z_0\) 是另一個 estimator，其與 process noise 的相關結構須**另行宣告**，不能
從 \(W\perp z_0\) 免費得到。本節因此把 \(\eta_\Delta\perp\delta z_0\) 作為 charter
obligation 3 明列的「required initial-state/process-noise assumption」**顯式凍結**。
（可**推出**本假設的較強**充分條件**（**非**等價）：宣告 exit-time sigma-field
\(\mathcal F_0\) 使 \(z_0,\hat z_0,c\) 皆 \(\mathcal F_0\)-measurable 且
\(\{W_t-W_0:0<t\le\Delta\}\perp\mathcal F_0\)——此條件蘊含 \(\eta_\Delta\perp
\delta z_0\)，但反向不成立（\(\eta_\Delta\) 可只與 \(\delta z_0\) 獨立而仍與其他
\(\mathcal F_0\)-measurable 量相關）。本節取較小的直接宣告即足以滿足 obligation 3。）
於是
\(\operatorname{Cov}(\delta z_0,\eta_\Delta)=0\)，prediction-error covariance 無
cross term：

\[
\boxed{\,P^-_\Delta:=\operatorname{Cov}(e^-)=A_\Delta\,P_0\,A_\Delta^\top+Q_\Delta\,}
\qquad(P^-_\Delta\succeq0).
\]

（此即 charter「observation and uncertainty boundary」的 \(P^-_\Delta=\Phi P_0\Phi^\top
+Q_\Delta\)，以 canonical \(A_\Delta=e^{F\Delta}\)（§4.3）具體化。）\(P^-_\Delta\) 一般
可能 singular（\(P_0\) 或 \(Q_\Delta\) 退化時；\(Q_\Delta\) 的退化刻畫見 D2 L1），故
\(P^-_\Delta\) 亦為 possibly-degenerate covariance。

### §5.3 Entry observation 與 innovation residual

Entry endpoint 觀測（§3；production-corresponding 綁 \(H_x\)，§3.2）：

\[
y_1=Hz_\Delta+\epsilon_1,\qquad \epsilon_1\sim\mathcal N(0,R_1),
\qquad H\in\{H_x,H_{xv}\}\ (\text{§3}).
\]

Innovation residual \(r:=y_1-Hm^-_\Delta\)：

\[
r=H(z_\Delta-m^-_\Delta)+\epsilon_1=He^-+\epsilon_1 .
\]

**Null 下的均值。** Canonical layer 的 exact-CV matched-identity null（§2 row 9(i)、
§4.4）下，估計無偏（\(\mathbb E[\delta z_0]=0\)）、\(\mathbb E[\eta_\Delta]=0\)、
\(\mathbb E[\epsilon_1]=0\)，故 \(\mathbb E[e^-]=0\)、\(\mathbb E[r]=0\)：canonical
innovation 於 \(\Delta=g_{\mathrm{phys}}\) 是 **zero-mean**。這與 operator-layer 的
deterministic offset \(\pm(\mathrm{bridge\_at}-1)v\) 是**兩回事**（後者非 innovation
mean；§4.0 boundary 2 / §4.4）。

### §5.4 Obligation 3 的決定：canonical independence（\(C=0\)），與 dependent-error 的 expanded 形狀

Innovation covariance 的一般式（不預設 independence）：以 \(C:=\operatorname{Cov}
(e^-,\epsilon_1)\)，

\[
S_\Delta=\operatorname{Cov}(r)=\operatorname{Cov}(He^-+\epsilon_1)
=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top .
\]

（符號綁定於 frozen 約定 \(e^-=z_\Delta-m^-_\Delta\)、\(r=He^-+\epsilon_1\)；若改用
相反的 prediction-error 定義 \(m^-_\Delta-z_\Delta\)，\(r\) 與 \(HC+C^\top H^\top\)
的符號**必須一起改**，不得只留半邊——本節已凍結前一約定。）

**決定（frozen，二選一取 independence）：**

\[
\boxed{\;\text{Canonical A 層宣告 }e^-\perp\epsilon_1\ (\Rightarrow C=0)\;}
\qquad\Longrightarrow\qquad
\boxed{\;S_\Delta=HP^-_\Delta H^\top+R_1\;}
\]

**理由（declared assumption，非定理）：** \(\epsilon_1\) 是 entry endpoint 的
detection／measurement noise（entry frame 的一次獨立觀測），與 (i) exit-side 估計誤差
\(\delta z_0\)（不同 frame 的 measurement／filtering，屬 \(\mathcal F_0\)）、(ii)
process noise \(\eta_\Delta\)（trajectory 的物理隨機加速，§4.2）皆為不同物理來源，故取
\(e^-=A_\Delta\delta z_0+\eta_\Delta\perp\epsilon_1\)。此為標準 Kalman
measurement-noise independence，於此**顯式宣告**而非默認。合此與 §5.2 的
\(\eta_\Delta\perp\delta z_0\)，canonical innovation composition 完整為

\[
P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta,\qquad
S_\Delta=HP^-_\Delta H^\top+R_1 .
\]

**Dependent-error path（frozen deviation，非 canonical default）：** 若某
instantiation 的 \(\epsilon_1\) 與 \(e^-\) **不**獨立（例如 entry／exit 共用
detector／preprocessing 狀態而耦合，或 \(H_{xv}\) 下 entry velocity 由與 transition
window 重疊的 frames 導出），則**不得**沿用 \(C=0\)；必須**顯式宣告**
\(C=\operatorname{Cov}(e^-,\epsilon_1)\) 並改用

\[
S_\Delta=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top
\]

（符號如上）。此 deviation 必須是**顯式宣告的 model component**，與 §3.3 對
\(H_{xv}\) 的 causal-availability 宣告一致，**不得**被靜默繼承（呼應 §2 row 9「任何
其他非零 additive offset／偏移結構必須顯式宣告」的精神）。

**合法 \(C\) 的約束（非任意矩陣）：** 作為真實 cross-covariance，\((e^-,\epsilon_1)\)
的 joint covariance 必須 PSD：

\[
\boxed{\;
\begin{bmatrix}P^-_\Delta & C\\ C^\top & R_1\end{bmatrix}\succeq0
\;}
\qquad(\text{§5.6 domain}).
\]

此約束保證 expanded \(S_\Delta\) 仍 PSD（§5.5，congruence）。canonical default 仍是
\(C=0\)（block-diagonal，自動滿足此約束）。

### §5.5 \(S_\Delta\) 的結構性質（interface-level sanity，正式 PSD／inv 論證屬 D2）

**Canonical \(C=0\) 下：**

- **PSD：** \(P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta\succeq0\)（\(P_0\succeq0\)、
  \(Q_\Delta\succeq0\)，後者見 §4.5／D2 L1），故 \(HP^-_\Delta H^\top\succeq0\)，加
  \(R_1\succeq0\) 得 \(S_\Delta\succeq0\)。
- **\(R_1\) 下界與可逆性：** \(S_\Delta=HP^-_\Delta H^\top+R_1\succeq R_1\)，故
  \(\boxed{R_1\succ0\Rightarrow S_\Delta\succ0}\)（可逆），**即使** \(Q_\Delta\)／
  \(P^-_\Delta\) degenerate（D2 L1：\(Q_\Delta\) singular \(\iff D\) singular）。即
  entry-observation noise \(R_1\succ0\) 對 innovation covariance 具**正則化**作用，與
  transition-noise 是否退化無關。

**Dependent-error path（\(C\neq0\)）下的 PSD：** expanded \(S_\Delta\) 是 joint
covariance 的 congruence，

\[
S_\Delta=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top
=\begin{bmatrix}H&I\end{bmatrix}
\begin{bmatrix}P^-_\Delta&C\\ C^\top&R_1\end{bmatrix}
\begin{bmatrix}H^\top\\ I\end{bmatrix},
\]

故只要 §5.4／§5.6 的 joint-PSD 約束
\(\big[\begin{smallmatrix}P^-_\Delta&C\\ C^\top&R_1\end{smallmatrix}\big]\succeq0\)
成立，即 \(S_\Delta\succeq0\)（對任意 \(H\)）；canonical \(C=0\) 是其 block-diagonal
特例。這也說明為何 §5.6 的 \(C\) domain 不能放寬為任意矩陣：joint covariance 若非
PSD，expanded \(S_\Delta\) 可能不是合法 covariance。**但此 constraint 只保
\(S_\Delta\succeq0\)，不保可逆：** 即使 \(R_1\succ0\)，反相關的 \(C\) 仍可令
\(S_\Delta\) singular（最小反例 \(H=1,\;P^-_\Delta=R_1=1,\;C=-1\)：joint
\(\big[\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\big]\succeq0\)、\(R_1\succ0\)，
卻 \(S_\Delta=1+1-1-1=0\)——measurement error 與 prediction error 完全反相關而抵消
innovation）。故 dependent path 的可逆性需**額外 nondegeneracy 假設或直接要求
\(S_\Delta\succ0\)**，不能由 \(R_1\succ0\) 推得。

上述為 interface-level sanity（如 §4.7 的 dimensional-consistency note，非 D2
lemma）：本節**不**計算 \(q=r^\top S_\Delta^{-1}r\)、\(\log\det S_\Delta\) 或 NLL
（obligation 2 = WP-A5；shared-\(S_\Delta\) 下的 \(q\)/NLL 同序證明於後續 D2 增量，
D2 §7）；此處僅宣告「**canonical \(C=0\) 下** \(R_1\succ0\Rightarrow S_\Delta\succ0\)，故
\(S_\Delta^{-1}\) 存在、那些量 well-defined」的**充分條件**（dependent-error path 的
joint-PSD 只保 \(S_\Delta\succeq0\)，可逆性需上段的額外假設），不作任何 ranking claim。

### §5.6 Parameter domains, causal availability, units

| Symbol | Domain | Causal availability | Units |
|:--|:--|:--|:--|
| \(P_0\) | \(\{M\in\mathbb R^{2d\times2d}:M\succeq0\}\) | exit-time（§4.0 boundary 4） | block: \(xx\):\(\ell^2\)；\(xv\):\(\ell^2\,\mathrm{frame}^{-1}\)；\(vv\):\(\ell^2\,\mathrm{frame}^{-2}\) |
| \(P^-_\Delta\) | \(\succeq0\)（\(2d\times2d\)） | derived @ entry endpoint | 同 \(P_0\) |
| \(R_1\) | \(\succeq0\)；\(H_x\):\(d\times d\)、\(H_{xv}\):\(2d\times2d\) | entry-time（\(y_1\) 到達時） | \(H_x\): \(\ell^2\)（position-space）；\(H_{xv}\): 同 state |
| \(S_\Delta\) | \(\succeq0\)（維度 \(=R_1\)）；**canonical \(C=0\) only:** \(R_1\succ0\Rightarrow\succ0\)（dependent path 只保 \(\succeq0\)，可逆需額外假設，§5.5） | derived @ entry endpoint | 同 \(R_1\) |
| \(C\) | \(\Big\{C\in\mathbb R^{2d\times p}:\big[\begin{smallmatrix}P^-_\Delta&C\\ C^\top&R_1\end{smallmatrix}\big]\succeq0\Big\}\)，\(p=\dim\epsilon_1\)（canonical \(=0\)，自動滿足） | 僅 dependent-error path 顯式宣告 | \(\operatorname{Cov}(e^-,\epsilon_1)\) 對應 units |

\(\ell\) = §2 field 1 的 \(S_A\) 高度正規化位置單位；\(H_x=[\,I_d\ 0\,]\)（§3、§4.1）。
Dimensional consistency（sanity）：\(HP^-_\Delta H^\top\) 於 \(H_x\) 取 \(P^-_\Delta\)
的位置 block \(\sim\ell^2\)，與 \(R_1\sim\ell^2\)、\(S_\Delta\sim\ell^2\) 一致。

### §5.7 本節顯式不解決（typed deferrals）

| 項目 | 擁有 WP | 狀態 |
|:--|:--|:--|
| \(q=r^\top S_\Delta^{-1}r\)、\(\log\det S_\Delta\)、NLL 定義與 shared-\(S_\Delta\) 下 \(q\)/NLL 同序（obligation 2 的 ranking 面） | WP-A5（claim）＋後續 D2 增量（proof，D2 §7） | unresolved |
| calibration-only gain vs candidate-local ranking gain 為不同 claim（obligation 2） | WP-A5 | unresolved |
| identifiability／leakage matrix（terminal 3 predicate 對象） | §6（reserved） | unresolved |
| reverse-time／candidate-backward atom | 後續 | typed boundary only（§4.0 boundary 3） |
| B1/O1、H0、runtime、online、production、data、fitting | — | 不授權（charter Non-scope） |

**Freeze 邊界：** 本節凍結的是 uncertainty composition 的**定義、prediction-error
符號約定與 independence 決定**。它不宣稱任何 runtime 擷取值、不建立 fidelity edge、
不選 \(P_0/R_1\) 數值、不解除 obligation 2。修改須 append-only correction（原文保留、
註記 superseded）。

## §6–§7 Reserved

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
- 2026-07-22 — bounded interface corrections per #252 owner review (BLOCKED →
  fixed pre-merge; §4 not yet frozen; §2–§3 untouched, no append-only
  correction; WP-A3–A5 boundaries unchanged): (1) **noise dimension** made
  consistent — \(m\ge1\), \(W_t\in\mathbb R^m\), \(L\in\mathbb R^{d\times m}\),
  \(B\in\mathbb R^{2d\times m}\), \(BB^\top=\Sigma\), \(D=LL^\top\); (2)
  **\(\Delta=0\) domain closed** — split bridge evaluation
  \(\Delta\in\mathbb N_{\ge1}\) vs analytic family \(\Delta\in\mathbb R_{\ge0}\),
  with \(A_0=I,\,d_0=0,\,Q_0=0,\,a_0=0,\,b_0=1\) and \(a\in[0,\Delta]\); (3)
  **context argument given a formal domain** — \(c\in\mathcal C_{\mathrm{exit}}\),
  \(\bar v:\mathcal C_{\mathrm{exit}}\to\mathbb R^d\) measurable, and
  \(\mathcal N(m_\Delta,Q_\Delta)\) noted as a possibly-degenerate Gaussian
  measure since \(D\succeq0\) (no \(Q_\Delta^{-1}\) required at this layer).
- 2026-07-22 — **§5 frozen by WP-A4** (charter obligation 3): innovation
  composition over the frozen §4 kernel. Four uncertainty objects kept
  separate — \(P_0\) (exit-state estimation, \(\succeq0\)), \(Q_\Delta\)
  (§4.5, unchanged), \(R_1\) (entry-observation, \(\succeq0\)), \(S_\Delta\)
  (total innovation). Prediction-error sign convention frozen as
  \(e^-=z_\Delta-m^-_\Delta\) with \(m^-_\Delta=A_\Delta\hat z_0+d_\Delta(c)\),
  giving \(e^-=A_\Delta\delta z_0+\eta_\Delta\) and (via \(\eta_\Delta\perp
  \delta z_0\) from §4.7) \(P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta\);
  innovation residual \(r=y_1-Hm^-_\Delta=He^-+\epsilon_1\), canonically
  zero-mean at \(\Delta=g_{\mathrm{phys}}\) (distinct from the operator-layer
  offset \(\pm(\mathrm{bridge\_at}-1)v\), which stays out of
  \(m^-_\Delta/e^-/S_\Delta\)). **Obligation-3 decision (exactly one, chosen):**
  canonical A-layer declares independence \(e^-\perp\epsilon_1\Rightarrow C=0\),
  so \(S_\Delta=HP^-_\Delta H^\top+R_1\); the dependent-error path is a frozen,
  explicitly-declared deviation using
  \(S_\Delta=HP^-_\Delta H^\top+R_1+HC+C^\top H^\top\),
  \(C=\operatorname{Cov}(e^-,\epsilon_1)\), with signs tied to the frozen
  \(e^-\) convention. Interface-level sanity: \(S_\Delta\succeq R_1\), so
  \(R_1\succ0\Rightarrow S_\Delta\succ0\) even when \(Q_\Delta\) is degenerate
  (D2 L1) — no \(q\)/\(\log\det S\)/NLL computed here (obligation 2 = WP-A5;
  \(q\)/NLL ordering proof = later D2 increment, D2 §7). Reserved sections
  renumbered (§5→§6 identifiability/leakage, §6→§7 B1 schema). §2–§4 untouched
  (byte-frozen; no append-only correction); no fidelity edge, no \(P_0/R_1\)
  value selected, no runtime/data/production change.
- 2026-07-22 — bounded corrections per #254 owner review (REQUEST CHANGES →
  fixed pre-merge; §5 not yet frozen; §2–§4 untouched, byte-frozen): (1)
  **§5.2 initial-state/process-noise assumption** — \(\eta_\Delta\perp\delta z_0\)
  is now **declared and frozen** as the required assumption, **not** derived
  from §4.7 (which gives only \(\{W_t\}_{(0,\Delta]}\perp z_0\); \(W\perp z_0\)
  does not imply \(W\perp\delta z_0\), since \(\hat z_0\) is a separate
  estimator), with the fuller \(\mathcal F_0\)-measurable formulation noted as
  an equivalent stronger option. (2) **§5.4/§5.6 dependent-path \(C\) domain**
  — narrowed from all of \(\mathbb R^{2d\times p}\) to the genuine-cross-covariance
  set \(\{C:[P^-_\Delta,C;C^\top,R_1]\succeq0\}\), which keeps the expanded
  \(S_\Delta\) PSD via the congruence
  \(S_\Delta=[H\ I]\,[P^-_\Delta,C;C^\top,R_1]\,[H^\top;I]\) (§5.5). (3) **§5
  intro status note** — §4.8's frozen deferral table marked as the WP-A2
  freeze-time snapshot; current status source = §1 table + §5 + charter. The
  canonical \(C=0\) independence decision is unchanged.
- 2026-07-22 — second bounded correction per #254 owner review (still
  pre-merge; §5 not yet frozen; §2–§4 byte-frozen): (1) **§5.2** — the
  \(\mathcal F_0\) formulation relabelled a **stronger sufficient condition
  (not equivalent)**: it implies \(\eta_\Delta\perp\delta z_0\) but not
  conversely. (2) **§5.5/§5.6** — the invertibility claim
  \(R_1\succ0\Rightarrow S_\Delta\succ0\) restricted to **canonical
  \(C=0\)**; the dependent-error joint-PSD constraint guarantees only
  \(S_\Delta\succeq0\) (counterexample \(H=1,P^-_\Delta=R_1=1,C=-1\Rightarrow
  S_\Delta=0\)), so its invertibility needs an extra nondegeneracy assumption.
  Independence decision and the PSD (\(\succeq0\)) results unchanged.
