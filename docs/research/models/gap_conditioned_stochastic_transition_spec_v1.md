<!-- doc-status: draft -->
<!-- doc-promotion: none; D1 canonical model specification (seed); §2–§3 frozen at WP-A1, §4 frozen at WP-A2, §5 frozen at WP-A4, §6 frozen at WP-A5, §7 frozen at WP-A6 -->
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
| §6 Calibration vs candidate-local ranking claim space (obligation 2) | **frozen** | WP-A5 (owner merge) |
| §7 Identifiability and leakage matrix (terminal-3 predicate object) | **frozen** | WP-A6 (owner merge) |
| §8 Schema-only interface for a future B1 input | reserved — not yet specified | WP-A7 (planned) |

Reserved sections carry no obligations-resolved claim. WP-A2 resolves charter
obligation 4 (canonical-state affine M2 transition; §4); WP-A4 resolves charter
obligation 3 (independence vs explicit cross-covariance \(C\); §5); WP-A5
resolves charter obligation 2 (calibration-only gain vs candidate-local ranking
gain as distinct claims; §6). **All four numbered activation-contract
obligations are now resolved.** WP-A6 additionally freezes the D1
identifiability/leakage matrix (§7; the terminal-3 predicate object). A sealable
terminal (`GCTM_MODEL_SPEC_SEALABLE`) additionally requires the remaining D1
deliverable — the schema-only B1 input interface (§8) — plus terminal review;
that section remains reserved. §4 makes no claim about obligations 2–3, §5 makes
no claim about obligation 2, §6 makes no claim about identifiability/leakage, and
§7 measures no data and selects no terminal — it specifies the identifiability
boundary only, and makes no B1/O1/runtime/production quantity claim.

> **Append-only renumber correction (WP-A5).** WP-A5 inserts §6 (calibration vs
> ranking) directly after frozen §5, so the former reserved §6/§7 shift to §7/§8.
> Frozen §5 is kept **byte-frozen**; its two in-body references to the
> identifiability/leakage section as "§6" (in §5.0 and the §5.7 deferral table)
> are therefore **superseded — read them as §7**. This note is the append-only
> correction of record for that renumber; the frozen §5 text is not edited in
> place.

> **Append-only status correction (WP-A6).** WP-A6 freezes §7
> (identifiability/leakage matrix). The frozen §5.7 and §6.6 typed-deferral tables
> list the identifiability/leakage matrix as `reserved`/`unresolved` — a
> **freeze-time snapshot** that is now **superseded**: §7 is **frozen by WP-A6**
> (the terminal-3 predicate object). Frozen §5 and §6 are kept **byte-frozen** and
> are not edited in place; the current section status is the §1 document-state
> table above. This note is the append-only correction of record.

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

## §6 Calibration vs candidate-local ranking claim space（obligation 2 — frozen）

本節落地 charter obligation 2：把 **calibration-only gain** 與 **candidate-local
ranking gain** 定義為**兩個不同的 claim**，各有**不同的 null、metric family、
evaluation unit 與 consequence**，並凍結兩者之間的 **invariance／separation 結構**
（為何一者的 gain 不蘊含另一者）。它建立在 frozen §5 的 innovation \(r\) 與 total
innovation covariance \(S_\Delta\) 之上，不重定義任何 §5 物件。這是 obligation 2 的
落點；charter 的「Score and probability semantics」段落是其上位敘述。

### §6.0 本節做什麼／不做什麼（typed boundary）

**做（frozen）：** (i) 定義 score 量 \(q\)、\(\log\det S_\Delta\)、Gaussian NLL、
candidate-region probability，並固定其 well-defined 的 regime；(ii) 定義 claim
**CAL**（cross-event calibration）與 claim **RANK**（candidate-local ranking），各附
null／metric family／evaluation unit／consequence；(iii) 凍結 separation 結構——
shared-\(S_\Delta\) 下 \(q\) 與 NLL 同序、isotropic／shared gap-scaling 對
candidate-local order **不變**（僅改 calibration）、ranking 對 event 內**統一**套用的
共同嚴格遞增重參數化不變（per-candidate 不同 \(\varphi_i\) 可改序；統一變換破壞
calibration ⇒ ranking gain 不蘊含 calibration gain），以及 candidate-specific
covariance 何時**可**改變 order。分佈層（Gaussian／\(\chi^2\)）敘述需 §6.1 的 CAL
Gaussian working null。

**不做（留給後續 WP 或本 packet 的 D2 增量）：** 不做 identifiability／leakage
matrix（reserved，見 §1 表 §7）；不寫 schema-only B1 interface（§8，reserved）；
不選 terminal（terminal review 為後續 packet，依 charter frozen decision procedure）；
**不執行任何 data／fitting／calibration 量測、不選 threshold／metric 參數、不宣稱任何
gain 數值**（本節只給 claim-space 定義與 invariance 結構，皆為 §5 之上的純數學）。
shared-\(S_\Delta\) 下 \(q\)/NLL 同序的**正式證明**是本 packet 的 D2 增量（D2 §7，
Lemma L5），本節只**陳述** claim 並指向該證明。不建立 fidelity edge、不做 bridge-runtime
claim。

沿用 §4.0／§5.0 的 canonical boundary（永不逾越），本節額外固定一條：

- **CAL 與 RANK 是不同 capability，永不得互相冒充或彼此替代。** 一個評估必須先宣告它
  測的是 CAL 還是 RANK；CAL 的結論**不**轉移為 RANK，反之亦然（這正是 obligation 2 的
  要求）。這與 §5.1 的「uncertainty objects 必須分開」同一治理精神。

### §6.1 Score 量（定義，over frozen §5 innovation；well-defined regime）

沿用 §5.3 的 innovation \(r=He^-+\epsilon_1\)、其 covariance \(S_\Delta\)（§5.4）、
prediction mean \(Hm^-_\Delta\)（§5.2），令 \(k=\dim r\)（\(H_x\):\(k=d\)；\(H_{xv}\):
\(k=2d\)）。**Regime（frozen）：** 需要 \(S_\Delta^{-1}\) 的量只在 \(S_\Delta\succ0\)
定義；由 §5.5，**canonical \(C=0\) 且 \(R_1\succ0\)** 即保 \(S_\Delta\succ0\)。退化
\(S_\Delta\)（dependent path、或 \(R_1\) singular）下這些量須改用 support-subspace 上的
pseudo-inverse／degenerate-Gaussian 形式——**本節不處理，亦不作任何 claim**。

**CAL Gaussian working null（declared，additional to frozen §5；frozen 於本節）.**
Frozen §5 已有的是 innovation 的**一二階矩**與**兩條 independence**：
\(\mathbb E[r]=0\)（§5.3 zero-mean）、\(\operatorname{Cov}(r)=S_\Delta\)（§5.4）、
\(\mathbb E[\delta z_0]=0\)、\(P_0=\operatorname{Cov}(\delta z_0)\)（§5.1/§5.3），以及
\(\eta_\Delta\perp\delta z_0\)（§5.2）、\(e^-\perp\epsilon_1\)（§5.4）。它**未**宣告
\(\delta z_0\) 的**分佈**。因此 predictive law 為 \(\mathcal N(Hm^-_\Delta,S_\Delta)\)、
\(q\sim\chi^2_k\)、Gaussian NLL 為**絕對 probability semantics** 等**分佈層**敘述，
**不是** frozen §5 的推論。CAL claim 因此在下列**額外宣告的 Gaussian working null** 下才
well-defined：

\[
\boxed{\;\delta z_0\sim\mathcal N(0,P_0),\qquad
\eta_\Delta\perp\delta z_0\ (\text{§5.2}),\qquad
e^-\perp\epsilon_1\ (\text{§5.4})\;}
\]

其中後兩條 independence 就是 frozen §5 既有的兩條（非三 primitive 的 chain
independence），此處只重述；**唯一新增**的是 \(\delta z_0\) 的 Gaussianity（其
unconditional zero-mean 與 covariance \(P_0\) 已在 §5.1/§5.3）。推導：

1. \(\delta z_0\sim\mathcal N(0,P_0)\) 與 \(\eta_\Delta\sim\mathcal N(0,Q_\Delta)\)
   （§4.3 已 Gaussian）independent（§5.2）\(\Rightarrow e^-=A_\Delta\delta z_0+\eta_\Delta\)
   為 Gaussian；
2. frozen §5.4 已宣告 \(e^-\perp\epsilon_1\)，且 \(\epsilon_1\sim\mathcal N(0,R_1)\)
   （§5.1 已 Gaussian）；
3. \(\Rightarrow r=He^-+\epsilon_1\sim\mathcal N(0,S_\Delta)\Rightarrow q\sim\chi^2_k\)。

**若 \(\delta z_0\) 非 Gaussian**，\(S_\Delta\) 仍是正確 covariance，但 \(r\) 一般**非**
Gaussian、\(q\not\sim\chi^2_k\)，故此 working null 是 CAL 分佈層 claim 的**必要**宣告，
非默認。下表四量作為 \(r,S_\Delta\) 的**函數**始終 well-defined（不需 working null）；
只有其**機率解釋**（density、\(\chi^2\)、coverage）需此 working null。

| 量 | 定義 | 說明 |
|:--|:--|:--|
| **standardized innovation（Mahalanobis）** | \(q:=r^\top S_\Delta^{-1}r\ (\ge0)\) | residual 相對於宣告不確定度的大小；**在上述 CAL Gaussian working null 下** \(q\sim\chi^2_k\)（frozen §5 只保 \(\mathbb E[r]=0\)、\(\operatorname{Cov}(r)=S_\Delta\)，不含分佈）。 |
| **predictive log-volume** | \(\log\det S_\Delta\) | 預測不確定度的體積尺度；與 \(r\) **無關**（不含 alignment 資訊）。 |
| **Gaussian NLL（per candidate）** | \(E:=\tfrac12 q+\tfrac12\log\det S_\Delta+\tfrac{k}{2}\log(2\pi)\)（\(=-\log\mathcal N(y_1;Hm^-_\Delta,S_\Delta)\)，其密度解釋需 working null） | 合併 residual fit（\(q\)）與 uncertainty volume（\(\log\det S_\Delta\)）。 |
| **candidate-region probability** | \(\Pi(\Omega):=\int_\Omega\mathcal N\!\big(y;Hm^-_\Delta,S_\Delta\big)\,\mathrm dy\)，\(\Omega\subseteq\) obs space（Gaussian model 下） | 在宣告 region \(\Omega\) 上積分密度；**依賴 \(\Omega\) 的 geometry／volume**，不只 alignment。 |

以上四量作為函數皆為**定義**：不選 \(P_0/R_1\) 數值、不選 \(\Omega\)、不選任何 threshold。
四者不得壓成單一 scalar（承 §5.1）。特別地 \(\log\det S_\Delta\) 與 \(q\) 度量**不同**的
東西（volume vs alignment），\(\Pi(\Omega)\) 又混入 region volume——這三者的**不可
互換**正是 CAL/RANK 分離的來源。

### §6.2 兩個不同的 claim（凍結定義）

**Candidate event \(\mathcal E\)（frozen 定義）：** 一條 lost track（固定 exit
anchor 與其 state 估計 \(\hat z_0,P_0\)）與一組有限的 entry candidates
\(\{i\}_{i\in\mathcal E}\)；candidate \(i\) 有自身的 entry endpoint，故自身的 gap
\(\Delta_i=g_{\mathrm{phys},i}\)、context \(c_i\)、observation mode \(H_i\)、
covariance \(S_{\Delta,i}\)、innovation \(r_i\) 與 score \((q_i,E_i,\Pi_i)\)。**注意
generically candidates 不共用 \(\Delta_i\)**（不同 entry frame），故 \(S_{\Delta,i}\)
generically candidate-specific（§6.4）。

| | **CAL — cross-event calibration** | **RANK — candidate-local ranking** |
|:--|:--|:--|
| **性質（property）** | 預測律 \(\mathcal N(Hm^-_\Delta,S_\Delta)\) 在 events／gaps 的**母體**上校準（在 §6.1 CAL Gaussian working null 下 true match 的 \(q\) 服從名目 \(\chi^2_k\)；credible-region coverage／PIT uniformity／conditional calibration error 的正確性） | 在**單一** event \(\mathcal E\) **內**，score 把 true match 排在 distractors **之上**（event-local ordering） |
| **null** | \(H_0^{\mathrm{CAL}}\)：reference model 已校準（無 calibration gain） | \(H_0^{\mathrm{RANK}}\)：candidate-local ordering 不優於 baseline（如 raw distance／M0） |
| **metric family** | **coverage／PIT uniformity／conditional calibration error**，跨 events + gaps **聚合**（**不含** generic log／proper score——見下註） | event-local ordering metric（true-match rank、top-1、event-conditional AUC），**event 內**計算後再跨 event 平均 |
| **evaluation unit** | 母體（cross-event, cross-gap） | event 內 ordering（再平均） |
| **一個 "gain" 的意義** | 更好的**絕對機率語義** | 更好的 **event-local 判別** |
| **consequence** | 關於 probability semantics；**不**蘊含 ordering 改善 | 關於 event-local discrimination；**不**蘊含 calibration 改善 |

（此表為 charter「Calibration and candidate-local ranking must be evaluated as
different capabilities」的凍結落點。metric family 只**命名族類**，不選具體 metric／
threshold／data。）

**log／proper score 不屬 CAL metric family（凍結界定）：** generic log score／NLL
（及其他 proper scoring rules）評估的是**整個 predictive distribution**，同時受
**calibration 與 sharpness** 影響；改善 NLL **不等於** calibration-only gain（可由更
sharp 但同樣校準、甚至校準更差但更 sharp 而得）。故 log／proper score 另列為
**distribution-quality metric**，**不得單獨支持** CAL claim；CAL 的判定用 coverage／
PIT／conditional calibration error 這類**純 calibration** 診斷。

### §6.3 Separation（invariance 結構）—— 兩 claim 為何邏輯獨立

**(I) Ranking 不變而 calibration 可被改變（shared-covariance rescaling）。** 設 event
\(\mathcal E\) 內所有 candidate 共用同一 covariance \(S_{\Delta,i}=S\)（這是 L5 採用的
**唯一顯式前提與充分條件，不宣稱為必要條件**；universal \(q\)/NLL 同序真正需要的只是
additive 項 \(\tfrac12\log\det S_{\Delta,i}+\tfrac{k_i}2\log2\pi\) 在 candidates 間相同，
完整矩陣 \(S_{\Delta,i}=S\) 相同是易檢查的充分條件，且 \(S_i\neq S_j\) 時 \(q\) 與 NLL
仍可能碰巧同序，見 §6.4／D2 L5.2。**不**要求同 \(\Delta,c,H\)——\(R_1\) 是 entry-time
quantity、\(c\) 只影響 drift mean 不影響 covariance）。則
\(E_i=\tfrac12 q_i+\kappa(S)\)，其中 \(\kappa(S)=\tfrac12\log\det S+\tfrac k2\log2\pi\)
**與 \(i\) 無關**，故 \(q\) 與 \(E\) 在 event 內誘導**相同** ordering（rank-equivalent；
正式證明 D2 §7 L5）。又 gap-conditioned rescaling \(S\mapsto\alpha_\Delta S\)
（\(\alpha_\Delta>0\) 只依 event）給 \(q_i\mapsto q_i/\alpha_\Delta\)（正的、candidate-
independent 縮放，order-preserving）、\(E_i\mapsto\tfrac12 q_i/\alpha_\Delta+
\kappa(\alpha_\Delta S)\)（candidate-independent 的仿射重參數化），故 **event-local
order 不變**；**非平凡的** rescaling（\(\alpha_\Delta\neq1\)）**可以**改變 \(q\) 的絕對
分佈與 NLL 的 level ⇒ **可影響／改善 calibration**（\(\alpha_\Delta=1\) 為 identity，
什麼都不改）。
特例 \(S=\alpha_\Delta I\)（isotropic）：\(q_i=\lVert r_i\rVert^2/\alpha_\Delta\)，
order \(=\lVert r_i\rVert^2\) order，與 \(\alpha_\Delta\) 無關 ⇒ **isotropic
gap-conditioned scaling 對 candidate-local order 是 calibration-only**（charter 的
\(S_\Delta=\alpha_\Delta I\) 陳述）。

**(II) Ranking 保持不變而 calibration 可被改變（uniform monotone reparametrization）。**
在同一 event \(\mathcal E\) 內對**所有** candidates **統一**套用**同一個**嚴格遞增函數
\(\varphi_{\mathcal E}\)（\(q_i\mapsto\varphi_{\mathcal E}(q_i)\) for all \(i\in\mathcal E\)）
保持 event-local ordering，卻一般**破壞** calibration（\(\varphi_{\mathcal E}(q)\) 不再
\(\chi^2_k\)）。**量詞要緊：** 若每個 candidate 用**不同**的 \(\varphi_i\)（即使各自嚴格
遞增），ordering **可被改變**；保持 order 的是 event 內**統一**的 \(\varphi_{\mathcal E}\)，
非 per-candidate 變換。**完整非蘊含論證：** 取任一已具 **ranking gain** 的 score，對其在
每個 event 內統一套用一個破壞 calibration 的嚴格遞增 \(\varphi_{\mathcal E}\)——ordering
不變故 ranking gain **仍存在**，但 calibration 被破壞；故 **ranking gain 不保證
calibration gain**（兩個 event-local ordering 完全相同的 model 可有任意不同的
calibration）。

**Region-probability caveat（凍結）：** \(\Pi_i(\Omega_i)\) 依賴 region 的 volume／
geometry；若 candidates 用**非全等**的 region \(\Omega_i\)，\(\Pi\)-order 可與
\(q\)-order **不一致**（較大 region 可因 probability mass 勝出，儘管 alignment 較差）。
故 candidate-region probability **不是**純 alignment／ranking score，除非 regions 全等
——這正是 charter 把它與 \(q\)/NLL **分列**的原因。（數值 sanity 見 D2 §7 註。）

**小結（frozen）：** (I) 給「calibration gain ⇏ ranking gain」；(II) 給「ranking gain
⇏ calibration gain」。兩方向合起來即 obligation 2 要求的「兩個不同 claim，各有不同
null／metric／consequence，互不蘊含」。

### §6.4 何時 candidate-specific covariance **可**改變 ordering（宣告門檻）

Generically（§6.2）candidates **不**共用 \(\Delta_i\)（各自 entry endpoint）或
\(c_i\)，故 \(S_{\Delta,i}\) candidate-specific。此時
\[
E_i=\tfrac12 q_i+\tfrac12\log\det S_{\Delta,i}+\tfrac k2\log2\pi,
\]
其中 \(\tfrac12\log\det S_{\Delta,i}\) 為 **candidate-dependent** 項，故 \(q\) 與 \(E\)
可誘導**不同** ordering，且 \(S_{\Delta,i}\) 真實影響 ranking（predictive covariance
較大的 candidate 在 NLL 被 \(\log\det\) 罰）。正式的 tightness／counterexample 見 D2 §7
（L5.2）。

**宣告門檻（frozen）：** candidate-specific covariance 只有在其**來源與 causal
availability 顯式宣告**時，才可在 ranking claim 中改變 order（承 §3.3 的 causal-
availability 紀律：如 candidate-specific \(\Delta_i\) 於 entry 可得、\(c_i\) 於 exit
可得）。**未宣告來源的 candidate-specific covariance 不得靜默驅動 ranking**（承 §2 row 9
「任何偏移／結構須顯式宣告」的精神）。這是 charter「Candidate-specific covariance can
alter ordering only when its source and causal availability are explicitly
declared」的凍結落點。

### §6.5 Domains, causal availability, units

| 量 | Domain | Causal availability | Units |
|:--|:--|:--|:--|
| \(q\) | \([0,\infty)\)（需 \(S_\Delta\succ0\)，§6.1 regime） | derived @ entry endpoint | dimensionless |
| \(\log\det S_\Delta\) | \(\mathbb R\)（\(S_\Delta\succ0\)） | derived @ entry endpoint | \(=\log\) of \(\det\) units（volume scale） |
| \(E\) | \(\mathbb R\) | derived @ entry endpoint | nats |
| \(\Pi(\Omega)\) | \([0,1]\) | derived @ entry endpoint；\(\Omega\) 須顯式宣告 | dimensionless |
| \(\alpha_\Delta\)（§6.3） | \((0,\infty)\)，只依 event | event-level | 依 \(S_\Delta\) 的 scaling |

\(S_\Delta,r,H,P^-_\Delta,R_1\) 的 domains／units 沿用 §5.6，不重定義。所有量在
\(S_\Delta\) 退化時的定義另屬 pseudo-inverse／degenerate 形式（§6.1，不處理）。

### §6.6 本節顯式不解決（typed deferrals）

| 項目 | 擁有 WP | 狀態 |
|:--|:--|:--|
| shared-\(S_\Delta\) 下 \(q\)/NLL 同序、isotropic-scaling ranking invariance、candidate-specific tightness 的**正式證明** | 本 packet 的 D2 增量（D2 §7，L5／L5.1／L5.2） | **本 packet 提供** |
| identifiability／leakage matrix（terminal 3 predicate 對象） | §7（reserved；WP-A6 planned） | unresolved |
| schema-only B1 input interface | §8（reserved；WP-A7 planned） | unresolved |
| terminal review（checklist artifact + terminal selection） | 後續 packet（WP-A8 planned） | unresolved |
| CAL／RANK 的實際量測、gain 數值、metric／threshold 選擇 | — | 不授權（charter Non-scope；需 data/B1/O1 授權） |
| reverse-time／candidate-backward atom | 後續 | typed boundary only（§4.0 boundary 3） |
| B1/O1、H0、runtime、online、production、data、fitting | — | 不授權（charter Non-scope） |

**Freeze 邊界：** 本節凍結的是 **calibration/ranking claim-space 的定義與 invariance
結構**。它不宣稱任何 runtime 擷取值、不建立 fidelity edge、不選 \(P_0/R_1/\Omega\)／
metric／threshold、不量測任何 gain、不選 terminal。修改須 append-only correction
（原文保留、註記 superseded）。

## §7 Identifiability and leakage matrix（terminal-3 predicate object — frozen）

本節落地 charter D1 deliverable「identifiability and leakage matrix」，並作為
terminal partition 第 3 順位 predicate `GCTM_IDENTIFIABILITY_UNRESOLVED`（"the
intended claim cannot be identified under the declared observations or leakage
boundary"）的**predicate object**：它精確界定 primary object A 及其 §6 claim
（CAL／RANK）的**identifiability target**、可識別所需的 **observation／data-design
regime**、以及各量之間的 **leakage（confounding）結構**，使 terminal review 能對
identifiability row 作機械判定。它建立在 frozen §4 kernel、§5 innovation composition
與 §6 claim space 之上，不重定義任何既有物件。

### §7.0 本節做什麼／不做什麼（typed boundary）

**做（frozen）：** (i) 定義 identifiability target set（要識別哪些量／claim，從哪些
observable）；(ii) 定義 observation／data-design regime 軸（\(H_x\) vs \(H_{xv}\)、
single-event vs multi-gap population、context 是否宣告、label 是否可得）；(iii) 陳述
core confounding（為何 single position-only event 不可識別）與 multi-gap 下的
separation 條件；(iv) 凍結 **leakage matrix**（各 latent contributor 混入哪個
apparent observable、以及**阻斷**該 leakage 的宣告／observation／data-design 條件）；
(v) 陳述 identifiability boundary／verdict 作為 terminal-3 predicate object。

**不做（明確不授權／留給後續）：** **不執行任何 data、fitting、estimation、
identification 量測**（本檔不授權 data，charter Non-scope）；不宣稱任何 empirical
identifiability 已被**建立**（只**specify** boundary，不 demonstrate）；不選
\(P_0/R_1/\gamma/D/\bar v\) 的數值；不寫 schema-only B1 interface（§8，reserved）；
**不選 terminal**（terminal selection 屬 WP-A8 terminal review，依 charter frozen
decision procedure；本節只提供 predicate object）；不建立 fidelity edge、不做
bridge-runtime claim、不做 reverse-time atom。

沿用 §4.0／§5.0／§6.0 的 canonical boundary（永不逾越），本節額外固定一條：

- **Identifiability 是 conditional statement，永遠綁定其宣告的 observation／
  data-design／declaration 前提。** 「可識別」不得脫離其 regime 被引用；某量在
  multi-gap population 可識別**不**蘊含它在 single event 可識別。任何 instantiation
  引用某 identifiability 結論時，必須同時滿足該結論所列的前提。

### §7.1 Identifiability target set（要識別的量／claim）

| 類 | Target | 來源／定義 | 性質 |
|:--|:--|:--|:--|
| 轉移參數 | \(\gamma\)（mean-reversion rate）、\(D=LL^\top\)（diffusion）、\(\bar v(\cdot)\)（context mean-velocity map） | §4.2／§4.7 | \(Q_\Delta\) 由 \((\gamma,D,\Delta)\) 決定（§4.5，derived，非獨立 target）；\(\bar v(c)\) 驅動 context drift \(d_\Delta(c)\)（§4.3） |
| 不確定度物件 | \(P_0\)（exit-state est cov）、\(R_1\)（entry-obs cov） | §5.1 | \(\succeq0\)；\(S_\Delta=H(A_\Delta P_0A_\Delta^\top+Q_\Delta)H^\top+R_1\)（canonical \(C=0\)，§5.4）為 derived |
| deterministic mean | operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\)（§2 rows 8–9／§4.4） | declared operator quantity | **非** fitted target：由 known `bridge_at`+\(v\) 決定；此處只問「未宣告時它是否混入 apparent mean」 |
| claim | **CAL**（cross-event calibration）、**RANK**（candidate-local ranking）（§6.2） | §6 | 各自的 identifiability 見 §7.5 |

**Observable channel（可據以識別的量）：** 於 entry endpoint 由 §5 得到 innovation
\(r\)（其實現）與（在宣告 population／null 下）其一二階矩 \(\mathbb E[r]=Hm^-_\Delta\)、
\(\operatorname{Cov}(r)=S_\Delta\)；跨 events 得到 family \(\{(\mathbb E[r_\Delta],
S_\Delta)\}_\Delta\)；在有 true-match label 時得到 true match 的 \(q\) 分佈（CAL）與
event-local ordering（RANK）。label 之取得屬 data／B1 路徑，本節列為**前提**，不執行。

### §7.2 Observation / data-design regime（identifiability 的條件軸）

| 軸 | 值 | 對 identifiability 的作用 |
|:--|:--|:--|
| observation mode | \(H_x\)（position-only）／\(H_{xv}\)（joint，需 §3.3 宣告 causal availability） | \(H_x\) 只觀測 position block；velocity-相關參數只能靠 \(\Delta\)-shape 間接識別。\(H_{xv}\) 增方程式但 entry velocity 若由重疊 frames 得出⇒dependent-error path（\(C\neq0\)，§5.4），\(C\) 成為額外 unknown |
| data design | single event（單一 \(\Delta\)、單一 lost track）／multi-gap population（跨 events、涵蓋 \(\ge\) 若干**相異** \(\Delta\)、共享參數） | single event 的 observable 遠少於 unknown（§7.3）；multi-gap 用 \(\Delta\)-shape 分離加項（§7.4） |
| context 可觀測性 | \(c\)（及 \(\bar v(c)\)）是否宣告／觀測且**變化** | 未宣告或不變的 \(\bar v\) 與常數 mean bias／operator offset 混淆（§7.6 mean-level） |
| label | true-match label 是否可得 | CAL／RANK 皆需 label 條件化「true match」（前提，屬 data／B1） |

### §7.3 Core confounding：single position-only event 不可識別

在**單一** \(\Delta\)、\(H_x\) 下，covariance 通道只給**一個** \(d\times d\) 矩陣

\[
S_\Delta=H_x\big(A_\Delta P_0 A_\Delta^\top+Q_\Delta(\gamma,D)\big)H_x^\top+R_1 ,
\]

而未知量為 \(P_0\)（\(2d\times2d\)，\(\succeq0\)）、\((\gamma,D)\)、\(R_1\)（\(d\times d\)，
\(\succeq0\)）——自由度遠多於方程式。三個加項 \(H_xA_\Delta P_0A_\Delta^\top H_x^\top\)、
\(H_xQ_\Delta H_x^\top\)、\(R_1\) 在**同一** \(\Delta\) 只以**和**出現，彼此**加性
混淆**：任一 \(S_\Delta\) 可由結構迥異的 \((P_0,\gamma,D,R_1)\) 實現（數值 sanity C1：
不同 \((\gamma,D,P_0,R_1)\) 給同一 \(S_\Delta\)）。mean 通道 \(\mathbb E[r]=y_1\) 的
期望同理把 exit-state 估計、\(\bar v(c)\) 與（若未宣告）operator offset 混在一起
（§7.6）。故 **single position-only event 對 \(\{P_0,\gamma,D,R_1,\bar v\}\) 不可
識別**——這是 leakage boundary 的最強收縮點。

### §7.4 Multi-gap population 下的 separation（identifiability 條件）

分離的槓桿是**加項對 \(\Delta\) 的不同 shape**（皆由 frozen §4／已證 D2 lemma 提供，
非本節新證）：

- \(R_1\) 對 \(\Delta\) **常數**；
- \(Q_\Delta\) 的 position block \(q_{xx}(\gamma,\Delta)\,D\) 隨 \(\Delta\) 成長（\(\gamma\to0\)
  時 \(\sim\tfrac{\Delta^3}{3}D\)，D2 §4.6／L2；OU 飽和見 D2 L4）；
- \(A_\Delta P_0A_\Delta^\top\) 的 position block 經 \(a(\gamma,\Delta)\)（\(a\in[0,\Delta]\)）
  以另一 shape 進入。

**結論（interface-level）：** 在**共享參數的 population** 假設下（同一 \(P_0,\gamma,D,
R_1\) 跨 events）且涵蓋足夠多**相異** \(\Delta\)，family \(\{S_\Delta\}_\Delta\) 一般
（generically）可分離這些加項，使 \(\{P_0,\gamma,D,R_1\}\) 可識別（數值 sanity C2：
§7.3 中在單一 \(\Delta_0\) 不可分的兩組參數，跨 gaps 的 \(S_\Delta\) 相對差達 \(O(1)\)）。
所需相異 \(\Delta\) 個數隨欲識別的自由度增加；gap spread 過小則接近退化、識別條件
**病態**（weakly identified）。

**position-only leakage（\(H_x\)）：** \(H_x\) 從不直接觀測 velocity，故 \(P_0\) 的
velocity block 與「position innovation 中屬 \(P_0\)-propagation vs 屬 \(Q_\Delta\)」的
拆分**全靠** \(\Delta\)-shape；某些方向（尤其 gap spread 小時）保持 weakly identified。
這是 position-only 的 identifiability 代價。

**joint mode（\(H_{xv}\)）：** 於 entry 觀測 velocity 增方程式、改善 conditioning；但
須 §3.3 宣告 causal availability，且若 entry velocity 由與 transition window 重疊的
frames 導出，則進入 §5.4 的 dependent-error path（\(C\neq0\)），\(C\) 成為**額外**待
識別／宣告量，其 joint-PSD domain（§5.6）與可逆性 caveat（§5.5）一併適用。

**context \(\bar v(c)\)：** 只有在 \(c\) 被**宣告／觀測**且在 population 中**變化**、且為
exit-causally available（§4.0 boundary 4）時，\(\bar v(c)\) 才可與常數 mean 分離而
識別；否則常數 \(\bar v\) 與常數 mean bias／operator offset 混淆（§7.6）。

### §7.5 Claim-level identifiability（CAL vs RANK，承 §6 separation）

§6.3 的 separation 直接給出 claim-level 的**互不識別**：

- **RANK identifiable** 自 event-local ordering + label（true-match rank／top-1／
  event-conditional AUC）。但 ordering 對 shared-scale \(\alpha_\Delta\)（§6.3(I)）與
  event 內**統一**的嚴格遞增重參數化（§6.3(II)）**不變**⇒ **ranking 資料不能識別
  calibration scale \(\alpha_\Delta\)**（數值 sanity C4a：\(\alpha\) 變動時 order 不變）。
- **CAL identifiable** 自 population 的 coverage／PIT／conditional calibration error +
  label + §6.1 CAL Gaussian working null；calibration 能**pin 住**絕對 level（含
  \(\alpha_\Delta\)）。但 calibration **不**識別 event-local ordering 改善：一個完美
  校準的 model 可以 rank 得並不更好（數值 sanity C4a/C4b：uniform monotone reparam 令
  order 不變卻破壞 \(\chi^2\) 校準）。
- ⇒ **CAL 與 RANK 互不識別（mutually non-identifying）：** 各自對一個會移動另一者的
  變換保持不變。這是 §6 separation 的 identifiability 面重述，也正是 obligation 2 要求
  兩者為**不同 claim** 的根據。兩者皆以 true-match label 為前提（data／B1，本節不執行）。

### §7.6 Leakage matrix（凍結）

「Leakage」:= 某 latent contributor 的變動被**吸收／混淆**進某個 apparent observable，
使後者無法唯一歸因。下表凍結 leakage 結構與其**阻斷條件**（哪個 declaration／
observation／data-design 移除該 leakage）。分 covariance-level、mean-level、
claim-level 三塊。

**Covariance-level（皆加性進入 \(S_\Delta\)，§7.3）：**

| Contributor 對 | 在哪混淆 | Leak? | 阻斷條件 |
|:--|:--|:--:|:--|
| \(P_0\)-propagation ↔ \(Q_\Delta\) | \(S_\Delta\) position block @ 單一 \(\Delta\) | **L** | multi-gap 相異 \(\Delta\) 的 \(a(\gamma,\Delta)\) vs \(q_{xx}(\gamma,\Delta)\) shape（§7.4）；\(H_{xv}\) 增條件 |
| \(Q_\Delta\) ↔ \(R_1\) | \(S_\Delta\) @ 單一 \(\Delta\) | **L** | multi-gap：\(Q_\Delta\) 隨 \(\Delta\) 成長 vs \(R_1\) 常數（§7.4；D2 L2/L4） |
| \(P_0\) ↔ \(R_1\) | \(S_\Delta\) @ 單一 \(\Delta\) | **L** | multi-gap \(\Delta\)-shape；\(P_0\) velocity block 於 \(H_x\) 僅 weakly identified |
| \(\gamma\) ↔ \(D\) | \(Q_\Delta=\text{scalar}(\gamma,\Delta)\,D\) | **L**（單一 \(\Delta\) 只見乘積） | multi-gap：OU scalar 的 \(\Delta\)-shape 定 \(\gamma\)，整體尺度定 \(D\)（§7.4） |

**Mean-level（皆進入 apparent \(\mathbb E[r]\)／residual bias）：**

| Contributor 對 | Leak? | 阻斷條件 |
|:--|:--:|:--|
| operator-offset \(\pm(\mathrm{bridge\_at}-1)v\) ↔ context drift \(\bar v(c)\)／常數 bias | **L**（未宣告時） | **顯式宣告** operator offset（§2 row 9，known `bridge_at`+\(v\)）即從 residual 減除（數值 sanity C3）；否則與常數 mean 混淆 |
| context drift \(\bar v(c)\) ↔ 常數 mean bias | **L**（\(c\) 未觀測／不變時） | 宣告且**變化**的 exit-causal context（§7.4） |
| exit-state 估計 bias ↔ 上二者 | **L**（單一 event） | population + null（\(\mathbb E[\delta z_0]=0\)，§5.3）；multi-gap |

**Claim-level（§7.5）：**

| From → To | Leak? | 說明 |
|:--|:--:|:--|
| CAL-scale \(\alpha_\Delta\) → RANK order | **—**（不 leak） | ranking 對 \(\alpha_\Delta\) 不變 ⇒ 反過來 ranking **不識別** \(\alpha_\Delta\)（單向 blindness） |
| RANK order（uniform monotone reparam）→ CAL | **—**（order 不動） | order 不變卻破壞 calibration ⇒ order **不識別** CAL |

（"—" 在 claim-level 指「不互相污染 order／calibration 之**不變量**」，其後果正是
**互不識別**：一者對移動另一者的變換保持不變，故觀測一者無法定另一者。）

### §7.7 Identifiability boundary / verdict（terminal-3 predicate object）

**Verdict（frozen，conditional）：** 在宣告的 observation interface（§2–§3）與 §6
separation 結構下，primary object A 的 intended claims 的 identifiability 為
**conditionally specified**：

1. \(\{P_0,\gamma,D,R_1\}\) 於 **multi-gap population（共享參數、足夠 gap spread）**
   generically 可識別；於 **single position-only event 不可識別**（§7.3–§7.4）。
2. \(\bar v(c)\) 於 **宣告且變化的 exit-causal context** 可識別，否則與常數 mean／
   operator offset 混淆（§7.4／§7.6）。
3. \(H_{xv}\) velocity-相關識別需 §3.3 causal-availability 宣告；overlap 導出的 entry
   velocity 引入 dependent-\(C\)（§5.4）作為額外待宣告量。
4. **CAL 與 RANK 互不識別**，各需 true-match label（data／B1 前提）；shared-scale
   \(\alpha_\Delta\) 由 CAL 而非 RANK 決定（§7.5）。

**性質界定（誠實邊界）：** 上述皆為 **specification of the identifiability boundary**，
非**已建立**的 empirical identification——本檔不授權 data，故不 demonstrate 任何識別。
沒有任何 intended claim 被證為「作為 specification 不可識別」；non-identifiability 只
發生在**違反宣告條件**的 instantiation（single position-only event、未宣告 offset／
context、gap spread 過小的病態設計）。因此本節**specify** 了 terminal-3 predicate 所讀的
identifiability／leakage boundary：intended claim 在宣告 observation 下**於所列條件成立
時可識別**；當這些條件不可滿足時，對應 claim 在該 instantiation **被宣告 non-identifiable**，
落入 terminal review 的 terminal-3 rejection region。

**本節不選 terminal。** 是否對某具體 instantiation 觸發 `GCTM_IDENTIFIABILITY_UNRESOLVED`
（或標 identifiability row 為 `complete`／`rejection-established`）是 WP-A8 terminal
review 依 charter frozen decision procedure 的機械判定；本節只提供其 predicate object。

### §7.8 Domains / regime summary

| 量 | 可識別所需最小 regime | 未達 regime 時的地位 |
|:--|:--|:--|
| \(R_1,Q_\Delta(\gamma,D),P_0\) | multi-gap population，\(\ge\) 足夠相異 \(\Delta\)，共享參數 | single \(\Delta\)：加性混淆，不可識別（§7.3） |
| \(\gamma\) vs \(D\) | 同上（OU scalar 的 \(\Delta\)-shape） | 單一 \(\Delta\)：只見乘積 |
| \(P_0\) velocity block | 更佳於 \(H_{xv}\)；\(H_x\) 僅靠 \(\Delta\)-shape | \(H_x\)+小 gap spread：weakly identified |
| \(\bar v(c)\) | 宣告、變化、exit-causal context | 否則與常數 bias／offset 混淆 |
| CAL scale \(\alpha_\Delta\) | CAL 路徑（coverage/PIT + label + working null） | RANK 路徑不識別它 |
| RANK order | event-local ordering + label | CAL 路徑不識別它 |

### §7.9 本節顯式不解決（typed deferrals）

| 項目 | 擁有 WP | 狀態 |
|:--|:--|:--|
| schema-only B1 input interface | §8（reserved；WP-A7 planned） | unresolved |
| terminal review（checklist artifact + terminal selection；含對 identifiability row 的機械判定） | 後續 packet（WP-A8 planned） | unresolved |
| 實際 identification／estimation／fitting／data、label 取得、任何 gain／參數數值 | — | 不授權（charter Non-scope；需 data/B1/O1 授權） |
| reverse-time／candidate-backward atom | 後續 | typed boundary only（§4.0 boundary 3） |
| B1/O1、H0、runtime、online、production | — | 不授權（charter Non-scope） |

**Freeze 邊界：** 本節凍結的是 **identifiability target、regime、leakage 結構與
boundary verdict** 之**定義**。它不宣稱任何 runtime 擷取值、不建立 fidelity edge、不
執行 identification、不選參數／metric／threshold、不選 terminal。修改須 append-only
correction（原文保留、註記 superseded）。

## §8 Reserved

`GCTM_MODEL_SPEC_SEALABLE` 之前必須完成（見 charter frozen terminal partition
與 obligation-status table）；在其 work packet 落地前，本檔不預先陳述。**§8**
schema-only B1 input interface（WP-A7 planned）。

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
- 2026-07-22 — **§6 frozen by WP-A5** (charter obligation 2): calibration vs
  candidate-local ranking claim space over the frozen §5 innovation. Defines the
  score quantities \(q=r^\top S_\Delta^{-1}r\) (\(\sim\chi^2_k\) under the
  canonical null), \(\log\det S_\Delta\), Gaussian NLL
  \(E=\tfrac12 q+\tfrac12\log\det S_\Delta+\tfrac k2\log2\pi\), and
  candidate-region probability \(\Pi(\Omega)\), all in the invertible regime
  (canonical \(C=0\), \(R_1\succ0\Rightarrow S_\Delta\succ0\); degenerate
  \(S_\Delta\) out of scope). Freezes **two distinct claims** — CAL (cross-event
  calibration) and RANK (candidate-local ranking) — each with its own null,
  metric family, evaluation unit, and consequence, and the **separation
  structure**: (I) shared-\(S_\Delta\) rescaling / isotropic \(S=\alpha_\Delta I\)
  changes calibration but **not** candidate-local order (calibration gain ⇏
  ranking gain); (II) ranking is invariant to a **common** strictly-increasing
  reparametrization applied **uniformly** across candidates within an event
  (per-candidate distinct \(\varphi_i\) can reorder), which generally breaks
  calibration (ranking gain ⇏ calibration gain); plus the region-probability
  caveat (non-congruent regions make \(\Pi\)-order disagree with \(q\)-order) and
  the rule that candidate-specific covariance may alter ordering **only** when its
  source and causal availability are explicitly declared (§3.3). The
  \(\chi^2_k\)/Gaussian distributional statements hold only under an explicitly
  declared **CAL Gaussian working null** (\(\delta z_0\sim\mathcal N(0,P_0)\), with
  the two independences \(\eta_\Delta\perp\delta z_0\) (§5.2) and
  \(e^-\perp\epsilon_1\) (§5.4) already frozen in §5, so only \(\delta z_0\)'s
  Gaussianity is new; §6.1) — an addition beyond frozen §5 (which gives only
  zero-mean + covariance); generic
  log/proper score is a distribution-quality metric (calibration+sharpness),
  excluded from the CAL metric family. The shared-\(S_\Delta\)
  \(q\)/NLL ordering-equivalence **proof** is the D2 increment of this packet (D2
  §7, Lemma L5). **All four numbered obligations are now resolved;** a sealable
  terminal still needs §7 (identifiability/leakage), §8 (B1 schema), and terminal
  review. No data/fitting/calibration measurement, no metric/threshold selection,
  no gain value, no terminal selected. Reserved sections renumbered (former §6
  identifiability/leakage → §7, former §7 B1 schema → §8); frozen §5's two
  in-body "§6" references to identifiability are superseded (read as §7) via the
  append-only renumber note after the §1 table — §5 kept byte-frozen. §2–§5
  untouched (byte-frozen; no in-place edit); no fidelity edge, no \(P_0/R_1\)
  value, no runtime/data/production change.
- 2026-07-22 — bounded corrections per #255 owner review (BLOCKED → fixed
  pre-merge; §6 not yet frozen; §2–§5 byte-frozen): (1) **CAL Gaussian working
  null** — §6.1 declares \(\delta z_0\sim\mathcal N(0,P_0)\) with the two
  independences \(\eta_\Delta\perp\delta z_0\) (§5.2) and \(e^-\perp\epsilon_1\)
  (§5.4) — the ones **already frozen** in §5, restated (not a 3-primitive chain);
  the **only** new assumption is \(\delta z_0\)'s Gaussianity (its unconditional
  zero-mean + covariance \(P_0\) are already in §5.1/§5.3). Derivation: \(e^-=
  A_\Delta\delta z_0+\eta_\Delta\) Gaussian (indep. Gaussians), \(e^-\perp\epsilon_1\)
  + \(\epsilon_1\) Gaussian ⇒ \(r\sim\mathcal N(0,S_\Delta)\) ⇒ \(q\sim\chi^2_k\).
  The \(\chi^2_k\)/Gaussian-predictive-law/absolute-probability statements are
  scoped to this working null (the four score quantities remain defined as
  functions of \(r,S_\Delta\) without it, only their probabilistic interpretation
  needs it). (2) **§6.3(II) title + quantifier** — retitled
  "Ranking 保持不變而 calibration 可被改變"; the order-preserving transform is a
  **common** strictly-increasing \(\varphi_{\mathcal E}\) applied **uniformly**
  across candidates in an event (distinct per-candidate \(\varphi_i\) can reorder),
  with the full non-implication argument (start from a ranking-gain score, apply a
  uniform calibration-breaking monotone map, ranking gain persists ⇒ ranking gain
  ⇏ calibration gain). (3) **CAL metric family** — generic log/proper score
  removed from CAL (it mixes calibration + sharpness) and listed as a separate
  distribution-quality metric; CAL keeps coverage/PIT/conditional calibration
  error. (4) **§6.3(I) shared-\(S\) condition** — dropped the incorrect
  "同 \(\Delta,c,k,H\)" equivalence; the lemma's condition is exactly
  \(S_{\Delta,i}=S\) (\(R_1\) is entry-time, \(c\) enters drift mean not
  covariance). CAL/RANK separation and the invertible regime unchanged.
- 2026-07-22 — second bounded correction per #255 owner re-review (still
  pre-merge; §6 not yet frozen; §2–§5 byte-frozen): (1) **working-null
  self-consistency** — the boxed null rewritten as \(\delta z_0\sim\mathcal N(0,P_0)\)
  plus the two **already-frozen** independences \(\eta_\Delta\perp\delta z_0\)
  (§5.2), \(e^-\perp\epsilon_1\) (§5.4), replacing the earlier 3-primitive chain
  \(\delta z_0\perp\eta_\Delta\perp\epsilon_1\) and the conditional
  \(\delta z_0\mid\hat z_0\) form (which had silently added conditional
  centering/covariance); this keeps "only \(\delta z_0\) Gaussianity is new" honest,
  and the derivation now routes through the frozen \(e^-\perp\epsilon_1\). Same
  notation fixed in the charter note/History and PR body. (2) **§6.3(I)
  sufficient-not-necessary** — "充要條件" corrected: \(S_{\Delta,i}=S\) is L5's sole
  explicit premise and a **sufficient** condition, not necessary (the true
  requirement is a candidate-independent \(\tfrac12\log\det S_{\Delta,i}+
  \tfrac{k_i}2\log2\pi\); unequal \(S_i\) may still coincide in order — consistent
  with L5.2). (3) **\(\alpha_\Delta=1\) nit** — rescaling changes calibration only
  for **non-trivial** \(\alpha_\Delta\neq1\) (identity changes nothing); D1 §6.3(I)
  and D2 L5.1 wording softened to "may affect calibration, not ranking." Core L5
  algebra, counterexample, separation, and lease unchanged.
- 2026-07-22 — **§7 frozen by WP-A6** (D1 identifiability/leakage matrix; the
  terminal-3 predicate object). Specifies the identifiability target set
  (\(\{\gamma,D,\bar v,P_0,R_1\}\), the deterministic operator offset, and the two
  §6 claims), the observation/data-design regime axes (\(H_x\)/\(H_{xv}\),
  single-event vs multi-gap population, context observability, labels), the core
  confounding (single position-only event is non-identifiable — \(P_0\)-propagation,
  \(Q_\Delta\), \(R_1\) are additively conflated at one \(\Delta\)), the multi-gap
  separation conditions (distinct-\(\Delta\) shapes of \(a(\gamma,\Delta)\) vs
  \(q_{xx}(\gamma,\Delta)\) vs constant \(R_1\), reusing frozen §4.6/D2 L2/L4), the
  claim-level result (CAL and RANK are **mutually non-identifying** — ranking does
  not identify the calibration scale \(\alpha_\Delta\); a uniform monotone reparam
  leaves order fixed but breaks calibration), and a frozen **leakage matrix**
  (covariance-/mean-/claim-level, each with the declaration/observation/data-design
  condition that blocks the leak). The verdict is conditional: identifiability is
  **specified**, not empirically established (no data authorized); no intended claim
  is unidentifiable **as a specification** — non-identifiability arises only for
  instantiations violating the declared conditions. §7 selects **no terminal** — it
  is the predicate object for WP-A8 terminal review. Frozen §2–§6 kept byte-frozen;
  the stale `reserved`/`unresolved` identifiability references in §5.7/§6.6 are
  superseded via an append-only status note after the §1 table (no in-place edit).
  No new file ⇒ no master_map regeneration. Claims numerically sanity-checked
  (single-gap non-identifiability, multi-gap separation, offset/drift mean
  confounding, CAL/RANK mutual non-identification).
