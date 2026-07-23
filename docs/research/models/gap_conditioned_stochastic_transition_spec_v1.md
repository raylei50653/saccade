<!-- doc-status: draft -->
<!-- doc-promotion: none; D1 canonical model specification (seed); §2–§3 frozen at WP-A1, §4 frozen at WP-A2, §5 frozen at WP-A4, §6 frozen at WP-A5, §7 frozen at WP-A6, §8 frozen at WP-A7 -->
<!-- doc-date: 2026-07-23 -->
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
| §8 Schema-only interface for a future B1 input | **frozen** | WP-A7 (owner merge) |

Reserved sections carry no obligations-resolved claim. WP-A2 resolves charter
obligation 4 (canonical-state affine M2 transition; §4); WP-A4 resolves charter
obligation 3 (independence vs explicit cross-covariance \(C\); §5); WP-A5
resolves charter obligation 2 (calibration-only gain vs candidate-local ranking
gain as distinct claims; §6). **All four numbered activation-contract
obligations are now resolved.** WP-A6 additionally freezes the D1
identifiability/leakage matrix (§7; the terminal-3 predicate object), and WP-A7
freezes the last D1 deliverable — the schema-only B1 input interface (§8). **All
D1 deliverable items are therefore complete**; a sealable terminal
(`GCTM_MODEL_SPEC_SEALABLE`) still requires terminal review (WP-A8), which is
where any terminal is selected. §4 makes no claim about obligations 2–3, §5 makes
no claim about obligation 2, §6 makes no claim about identifiability/leakage, §7
measures no data and selects no terminal — it specifies the identifiability
boundary only — and §8 is schema-only: it defines what a future B1 input would
have to supply, activates no B1, asserts no runtime availability, and makes no
B1/O1/runtime/production quantity claim.

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

> **Append-only status correction (WP-A7).** WP-A7 freezes §8 (schema-only B1
> input interface). The frozen §6.6, §7.9 typed-deferral tables and the §6.0/§7.0
> boundary paragraphs describe §8 as `reserved`/`unresolved` — **freeze-time
> snapshots** that are now **superseded**: §8 is **frozen by WP-A7**. Frozen
> §5–§7 are kept **byte-frozen** and are not edited in place; the current section
> status is the §1 document-state table above. This note is the append-only
> correction of record. §8 resolves no numbered obligation (all four remain
> resolved by WP-A1/A2/A4/A5) and selects no terminal.

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

### §7.4 Multi-gap population 下的 separation（identifiability 條件；含結構性不可識別方向）

寫 \(P_0=\big[\begin{smallmatrix}P_{xx}&P_{xv}\\ P_{xv}^\top&P_{vv}\end{smallmatrix}\big]\)
（\(P_{xv}\) 為一般 cross-block，**未必對稱**）。由 \(A_\Delta=\big[\begin{smallmatrix}
I&aI\\0&bI\end{smallmatrix}\big]\)（§4.3），position-only（\(H_x\)）observable 展開為
**四個 \(\Delta\)-shape 的線性組合**：

\[
\boxed{\;
S_\Delta=\underbrace{(P_{xx}+R_1)}_{\text{shape }1}
+\underbrace{\operatorname{sym}(P_{xv})}_{\text{shape }a}\cdot a
+\underbrace{P_{vv}}_{\text{shape }a^2}\cdot a^2
+\underbrace{D}_{\text{shape }q_{xx}}\cdot q_{xx}(\gamma,\Delta)
\;}
\]

其中 \(\operatorname{sym}(P_{xv})=P_{xv}+P_{xv}^\top\)（\(H_xA_\Delta P_0A_\Delta^\top H_x^\top
=P_{xx}+a\,\operatorname{sym}(P_{xv})+a^2P_{vv}\)），shape scalars \(\{1,a(\gamma,\Delta),
a^2(\gamma,\Delta),q_{xx}(\gamma,\Delta)\}\) 由 frozen §4／已證 D2 lemma 提供（\(q_{xx}\)：
D2 §4.6／L2，OU 飽和 D2 L4），非本節新證。

**可分離的是 coefficient matrices，不是完整 \(P_0,R_1\)——而且要 \(\gamma\) 已知。** 這四個
shape scalar 對相異 \(\Delta\) **線性獨立**（數值 sanity：over \(\ge4\) 相異 gap 的 shape
矩陣 rank \(=4\)）；因此 **在 \(\gamma\) 已知／固定的前提下**，共享參數 population
（同一 \(P_0,\gamma,D,R_1\) 跨 events）且 \(\ge4\) 相異 \(\Delta\) 使 family
\(\{S_\Delta\}_\Delta\) generically 識別出**四個係數矩陣**
\[
\{\,P_{xx}+R_1,\ \operatorname{sym}(P_{xv}),\ P_{vv},\ D\,\}.
\]
**但這不是完整的 \(\{P_0,R_1\}\)（見下兩個 gauge），且 \(\gamma\) 本身另有一層
non-identifiability（見下段）。**

**\(\gamma\) unknown 時，4 個 gap 不足以識別 \(\gamma\)（自由度計數）。** rank-4 論證預設
shapes \(a(\gamma,\cdot),q_{xx}(\gamma,\cdot)\) 的 \(\gamma\) 已知；當 \(\gamma\) 也是 unknown，
它**非線性**地進入 shapes。以每個 covariance 有 \(r=d(d+1)/2\) 自由度計，**恰 4 個 gap**：
\[
\text{observations}=4r,\qquad \text{unknowns}=4r+1\ (\text{多出的 }1=\gamma).
\]
故對任一候選 \(\gamma'\)，shape 矩陣 \(M(\gamma')\) 仍可逆，重解得四個係數矩陣
\(C'_j=M(\gamma')^{-1}O\)，在這 4 個 gap 上**精確**重現同一組 \(S_\Delta\)；且對 \(\gamma\)
鄰域內的 \(\gamma'\)，重解矩陣仍 PSD-admissible（interior），PSD 約束**不**消除此自由度
（數值 sanity G3：真 \(\gamma=0.6\) 下，\(\gamma'\in[0.40,0.68]\) 皆 exact-refit 且
admissible，構成 continuum ⇒ \(\gamma\) 由 4 gap **不可識別**）。**因此不得把「\(\ge5\)
gaps」寫成充分條件**——它只是自由度計數上的**必要**修正。\(\gamma\) 的識別需要
（i）\(>4\) 相異 gap，且（ii）joint nonlinear map \((\gamma,C_1,\dots,C_4)\mapsto
\{S_\Delta\}\) 的識別性條件——**global injectivity**（\(\Rightarrow\) **global**
identification）或**至少** **full-Jacobian-rank**（只 \(\Rightarrow\) **local**
identification）——加 non-degenerate coefficients。**注意這兩者不等價**：full-Jacobian-rank
是較弱的 local 條件，不蘊含 global injectivity（可有分離的 \(\gamma\) 解共享同一
observable），故不得以「\(/\)」當同義詞混用。本規格**不證明**任一條件，故**保守地不宣稱
已識別 \(\gamma\)**（數值上 \(\ge7\) gap 時 wrong-\(\gamma'\) best-fit 殘差非零、真 \(\gamma\)
殘差為零，僅為 generic identifiability 的 illustrative，非 sufficiency 證明）。

**兩個結構性不可識別方向（\(H_x\) 下，任意多 gap 皆不可識別；與 \(\gamma\) 是否已知無關）：**

1. **\(P_{xx}\leftrightarrow R_1\) gauge。** \(P_{xx}\) 與 \(R_1\) **同乘 constant shape \(1\)**，
   故 \(P_{xx}\mapsto P_{xx}+E,\ R_1\mapsto R_1-E\) 對**所有** \(\Delta\) 保持相同
   \(S_\Delta\)（數值 sanity G1：max \(\lVert\Delta S\rVert\approx10^{-15}\)）。只有**和**
   \(P_{xx}+R_1\) 可識別，split 不可。
2. **\(\operatorname{asym}(P_{xv})\) 不可觀測。** cross-block 的反對稱部分
   \(\tfrac12(P_{xv}-P_{xv}^\top)\) 完全**不進入** \(H_x\) map（只有 \(\operatorname{sym}\)
   出現），故對任意多 gap invisible（數值 sanity G2：對加反對稱擾動的 \(P_0\)，\(H_x\)
   observable 零變動）。

因此 **\(H_x\) multi-gap（\(\gamma\) 已知時）最多識別 quotient**
\(\{D,P_{vv},\operatorname{sym}(P_{xv}),P_{xx}+R_1\}\)；完整 \(P_0\)（其 \(P_{xx}\) vs
\(R_1\) split 與 \(\operatorname{asym}P_{xv}\)）**不可識別**，而 \(\gamma\) 本身在 unknown
時另需上段的 \(>4\)-gap + joint-map global-injectivity（或至少 full-Jacobian-rank）條件
（本規格不宣稱）。原始 C2 sanity 只證「某一組選定反例會被其他 gap 分開」，**不**證 mapping
injective——此處以 gauge 論證、G1/G2 與 G3（joint-\(\gamma\) DOF）更正。

**joint mode（\(H_{xv}\)）能救什麼、不能救什麼。** \(H_{xv}=I\) 觀測全 state，cross-block
\[
S_{xv}(\Delta)=bP_{xv}+ab\,P_{vv}+q_{xv}(\gamma,\Delta)D+R_1^{xv}
\]
含 \(P_{xv}\)（連同其反對稱部分），故 \(\operatorname{asym}(P_{xv})\) **不再是 \(H_x\) 下的
結構性 invisible、變得可觀測**（數值 sanity G2-Hxv：cross-block 隨反對稱擾動變動）。
**但「可觀測」\(\neq\)「單一 event 即可識別」：** 上式 cross-block 同時含 process
covariance \(q_{xv}(\gamma,\Delta)D\) 與 measurement covariance \(R_1^{xv}\)，單一
\(H_{xv}\) event 仍**無法唯一分離** \(P_{xv}\)——\(\operatorname{asym}(P_{xv})\) 於單一
event 與 \(R_1^{xv}\) 互換（數值 sanity G4a：\(\operatorname{asym}(P_{xv})+N\)、
\(R_1^{xv}-bN\) 給同一 \(S_{xv}\)）。其**真正識別**仍需 multi-gap、共享參數、\(\gamma\) 已知
（或未來另建的 joint nonlinear identifiability），multi-gap+已知 \(\gamma\) 下 \(bN\)（隨
\(\Delta\) 變）與常數 \(R_1^{xv}\)-shift 的 shape 相異方可分離（數值 sanity G4b）。
**但 \(P_{xx}\leftrightarrow R_1\) 的 position initial-state vs
measurement-noise confound 在 \(H_{xv}\) 下仍持續**：\(E\) 僅置於 \(xx\) 角、以 \(R_1\)
補償，對全 state \(A_\Delta E A_\Delta^\top\) 亦 \(\Delta\)-constant（數值 sanity G1-Hxv：
max \(\lVert\Delta S_{\text{full}}\rVert\approx10^{-15}\)）。此 gauge 只能由**額外宣告的
獨立資訊**固定——例如 detector 的已知 measurement-noise model 給定 \(R_1\)，或獨立宣告
position initial-state covariance \(P_{xx}\)——**不能**由 innovation family 單獨識別。此外
\(H_{xv}\) 若 entry velocity 由與 transition window 重疊的 frames 導出，進入 §5.4 的
dependent-error path（\(C\neq0\)），\(C\) 成為**額外**待宣告量（joint-PSD domain §5.6、
可逆性 caveat §5.5）。

**gap spread 病態：** 相異 \(\Delta\) 過少或過密則 shape 矩陣接近退化，quotient 亦僅
weakly identified。

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

**Covariance-level（皆加性進入 \(S_\Delta\)；§7.4 shape 分解）。** 依 §7.4，\(H_x\) 下四個
\(\Delta\)-shape 只給四個係數矩陣 \(\{P_{xx}+R_1,\operatorname{sym}(P_{xv}),P_{vv},D\}\)+\(\gamma\)。
下表區分 **multi-gap 可阻斷** 的 leak 與 **結構性（任意多 gap 仍在）** 的 leak：

| Contributor 對 | 在哪混淆 | Leak? | 阻斷條件 |
|:--|:--|:--:|:--|
| \(\operatorname{sym}(P_{xv})\)／\(P_{vv}\)／\(D\) 三者 ↔ 彼此 | \(S_\Delta\) position block @ 單一 \(\Delta\) | **L**（單一 \(\Delta\)） | **multi-gap 可阻斷**：shapes \(a,a^2,q_{xx}\) 線性獨立（§7.4） |
| \(\gamma\) ↔ \(D\)（及全體 coeff matrices） | \(Q_\Delta=\text{scalar}(\gamma,\Delta)\,D\)；\(\gamma\) 非線性入 shapes | **L** | **\(\gamma\) 已知**：\(D\)=\(q_{xx}\)-shape 係數，可識別。**\(\gamma\) unknown**：4 gap 有 1 個 DOF（\(=\gamma\)），任意 \(\gamma'\) 可 exact-refit（§7.4 G3）⇒ 需 \(>4\) gap + joint-map global-injectivity（global id）或至少 full-Jacobian-rank（local id；二者不等價）（本規格不證）；「\(\ge5\) gaps」只是**必要**非充分 |
| \(P_{xx}\) ↔ \(R_1\)（position initial-state vs measurement noise） | 兩者同乘 constant shape \(1\) | **L（結構性）** | **multi-gap 不可阻斷、\(H_{xv}\) 亦不可**（只識別和 \(P_{xx}+R_1\)，§7.4 G1）；僅由**額外宣告獨立資訊**固定（已知 \(R_1\) 或 \(P_{xx}\) gauge） |
| \(\operatorname{asym}(P_{xv})\) ↔ observable | 反對稱部分不進 \(H_x\) map | **L（結構性，\(H_x\)）** | **\(H_x\) 下任意多 gap invisible**；\(H_{xv}\) 使其**可觀測**（脫離 null space，非單一 event 即識別——cross-block 尚含 \(q_{xv}D+R_1^{xv}\)），識別仍需 multi-gap+共享參數+\(\gamma\) 已知（§7.4 G2/G4） |

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

1. **covariance 參數只在 quotient 意義下可識別，且要 \(\gamma\) 已知。**
   在 **\(\gamma\) 已知／固定** 的前提下，\(H_x\) multi-gap population（共享參數、
   \(\ge4\) 相異 gap）generically 識別 \(\{D,P_{vv},\operatorname{sym}(P_{xv}),
   P_{xx}+R_1\}\)；**完整 \(\{P_0,R_1\}\) 仍不可識別**——存在兩個結構性 gauge：
   \(P_{xx}\leftrightarrow R_1\)（\(H_{xv}\) 亦不可破，僅由額外宣告的已知 \(R_1\)／
   \(P_{xx}\) 固定）與 \(\operatorname{asym}(P_{xv})\)（\(H_x\) 結構性 invisible；\(H_{xv}\)
   使其**可觀測**，但識別仍需 multi-gap+共享參數+\(\gamma\) 已知，非單一 \(H_{xv}\) event
   即得，§7.4/G4）（§7.4／§7.6）。**\(\gamma\) unknown** 時 4 gap 有一個 DOF 使 \(\gamma\)
   **不可識別**（任意 \(\gamma'\) exact-refit，§7.4 G3）；其識別需 \(>4\) gap +
   joint-map **global injectivity**（global id）或至少 **full-Jacobian-rank**（local id；
   二者不等價）+ non-degenerate coefficients，**本規格不證明、不宣稱已識別 \(\gamma\)**
   （「\(\ge5\) gaps」只是必要非充分）。single position-only event 更強：連 quotient 都
   不可識別（§7.3）。
2. \(\bar v(c)\) 於 **宣告且變化的 exit-causal context** 可識別，否則與常數 mean／
   operator offset 混淆（§7.4／§7.6）。
3. \(H_{xv}\) velocity-相關識別需 §3.3 causal-availability 宣告；overlap 導出的 entry
   velocity 引入 dependent-\(C\)（§5.4）作為額外待宣告量。
4. **CAL 與 RANK 互不識別**，各需 true-match label（data／B1 前提）；shared-scale
   \(\alpha_\Delta\) 由 CAL 而非 RANK 決定（§7.5）。

**性質界定（誠實邊界）：** 上述皆為 **specification of the identifiability boundary**，
非**已建立**的 empirical identification——本檔不授權 data，故不 demonstrate 任何識別。
本節**不**宣稱完整 \(\{P_0,R_1\}\) generically identifiable；相反，它**明列**了不可
識別成分，各自 scope 不同：**(i)** \(P_{xx}\leftrightarrow R_1\) gauge 是**結構性
不可識別**，即使在 \(H_{xv}\) 下仍在（僅由額外宣告的已知 \(R_1\)／\(P_{xx}\) 固定）；
**(ii)** \(\operatorname{asym}(P_{xv})\) 只在 \(H_x\) 下 structural invisible——進入
\(H_{xv}\) 後**變為 observable**（不再 structural invisible），但其識別仍需 multi-gap、
共享參數與相應的 \(\gamma\) regime；**(iii)** \(\gamma\) unknown 是**未滿足的
identifiability regime**（需 \(>4\)-gap + joint-map 條件）。此外明列 identifiable 的
quotient。因此 identifiability boundary 是：某 intended claim 若**只**
依賴 identifiable quotient（例如 innovation 的 \(S_\Delta\) 本身、或 CAL/RANK 這類不依賴
\(P_{xx}\) vs \(R_1\) split 的量），且其所需 regime（\(\gamma\) 已知或滿足 \(>4\)-gap
joint-map 條件、multi-gap、共享參數、宣告 context）**成立**，則可識別；若某 instantiation
的 claim **必須**依賴一個**不可識別的分量或未滿足的 identifiability regime**——包括
（a）未宣告 gauge-fixing 而要求 \(P_{xx}\) 與 \(R_1\) 分離、（b）\(H_x\) 下要求
\(\operatorname{asym}P_{xv}\)、或（c）**\(\gamma\) unknown 且缺 \(>4\)-gap 與 joint-map
（global-injectivity／local full-Jacobian-rank）條件**——則該 claim 在該 instantiation
**non-identifiable**，落入 terminal review 的 terminal-3 rejection region。上述 (a)(b) 的
結構性 non-identifiability 是**宣告事實**（gauge 論證 + G1/G2）；(c) 的 \(\gamma\) 情形是
**未經證明的 identifiability regime**（DOF 必要條件已知、充分性未證，G3），本規格保守地
不宣稱其成立——兩類皆非有待本檔 data 建立。

**本節不選 terminal。** 是否對某具體 instantiation 觸發 `GCTM_IDENTIFIABILITY_UNRESOLVED`
（或標 identifiability row 為 `complete`／`rejection-established`）是 WP-A8 terminal
review 依 charter frozen decision procedure 的機械判定；本節只提供其 predicate object。

### §7.8 Domains / regime summary

| 量 | 可識別所需最小 regime | 未達 regime 時的地位 |
|:--|:--|:--|
| \(\{D,P_{vv},\operatorname{sym}(P_{xv}),P_{xx}+R_1\}\)（identifiable quotient，**\(\gamma\) 已知時**） | \(H_x\) multi-gap population，\(\ge4\) 相異 \(\Delta\)，共享參數，**\(\gamma\) 已知** | single \(\Delta\)：加性混淆，連 quotient 都不可識別（§7.3） |
| \(\gamma\)（unknown） | \(>4\) 相異 gap **且** joint-map global-injectivity（global id）或至少 full-Jacobian-rank（local id；二者不等價）+ non-degenerate coeffs（**本規格不證**） | 恰 4 gap：1 個 DOF，任意 \(\gamma'\) exact-refit ⇒ 不可識別（§7.4 G3）；「\(\ge5\)」僅必要非充分 |
| \(P_{xx}\) vs \(R_1\) split | **額外宣告獨立資訊**（已知 \(R_1\) 或 \(P_{xx}\)） | multi-gap／\(H_{xv}\) 皆**不可**識別（結構性 gauge，§7.4 G1） |
| \(\operatorname{asym}(P_{xv})\) | \(H_{xv}\)（使其可觀測）**＋** multi-gap＋共享參數＋\(\gamma\) 已知 | \(H_x\)：任意多 gap invisible（§7.4 G2）；單一 \(H_{xv}\) event：與 \(R_1^{xv}\) 互換不可分（§7.4 G4） |
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

## §8 Schema-only interface for a future B1 input（D1 deliverable — frozen）

本節落地 charter D1 的最後一項 deliverable「**schema-only interface for a
separately declared future B1 input**」。它規定：一個**未來**的 B1 input 若宣稱
instantiate 本 D1，必須提供哪些**欄位**、各欄位的**型別／domain／單位／causal
availability**、哪些量是 **input** 而哪些是 **derived（不得由 input 冒充）**、以及
哪些 **well-formedness predicate** 必須 fail-closed 通過。它建立在 frozen §2–§7 之上，
不重定義任何既有物件。

**Schema-only 的意思（凍結）：** 只定義 *interface shape*——欄位存在性、語意綁定與
可機械判定的合法性條件。它**不**啟用 B1、不取 data、不 fit、不選任何數值／metric／
threshold／fold／file format，也**不**建立 fidelity edge。

### §8.0 本節做什麼／不做什麼（typed boundary）

**做（frozen）：** (i) 定義 schema 的 consumption rule（何謂「本 D1 的合法
instantiation」）與 authority 分工；(ii) 定義四個 block 的欄位集——declaration-level
（Block D）、event-level（Block E）、pair-level（Block P）、derived（Block X）——各附
required/optional 標記、domain、單位、causal availability 與其 frozen 來源 §；(iii)
凍結 **well-formedness predicates**（W1–W9，fail-closed、機械可判定）；(iv) 凍結
**claim-restriction map**：某欄位／宣告缺席時，哪些 claim 因 §7 leakage 而**不可主張**。

**不做（明確不授權／留給後續）：** **不執行任何 data、capture、fitting、evaluation**；
不宣稱任何 runtime 擷取值可得（本 schema 的欄位是 *requirement*，不是 *availability
claim*）；不選 storage format／serialization／schema-version 機制；不定義 evaluation
design（fold、trial unit、minimum exposure、metric、threshold、blind/reveal 協定——皆
B1 charter 所有）；不決定 **B1-slot identity**、不寫 **score-layer contract**（兩者皆為
B1 charter 的 owner 前置，本節只 link，不複寫）；**不選 terminal**（WP-A8）；不建立
fidelity edge、不做 bridge-runtime claim。

沿用 §4.0／§5.0／§6.0／§7.0 的 canonical boundary（永不逾越），本節額外固定兩條：

- **Schema ≠ activation。** 一個滿足本 schema 的 input **不因此**成為 accepted B1
  evidence：B1 的 activation gate、frozen inputs、slot identity 與 score-layer 前置
  由 [B1 charter](../threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
  獨占；滿足本 schema 只表示「A 層物件可被無歧義 instantiate」。
- **Schema ≠ fidelity。** 欄位指名 production 對象時，其地位仍是 §2 的
  *declared-target*；runtime 忠實性須 H0／fidelity-edge 路徑另行建立。本 schema 的
  任何欄位都不得被引用為 runtime-fidelity 證據。

### §8.1 Consumption rule 與 authority 分工（frozen）

**Consumption rule（frozen，fail-closed）：** 一個未來 B1 input 宣稱 instantiate 本
D1，當且僅當

1. 提供 Block D／E／P 的**全部 `required` 欄位**，並對每個 `required-if` 欄位滿足其
   觸發條件；且
2. 通過 §8.6 的**全部** well-formedness predicates（W1–W9）；且
3. 其宣稱的 claim 集合落在 §8.7 claim-restriction map 依**實際宣告到的欄位**所允許的
   範圍內。

三者任一不成立 ⇒ 該 input **不是**本 D1 的 instantiation，**不得**引用 D1 的任何
identifiability／separation 結論（§6／§7 的結論永遠綁其 regime 前提，§7.0）。
不合法 record 一律 **inadmissible**，**不得**以預設值補齊（W9）。

**Authority 分工（link, don't restate）：**

| 對象 | 擁有者 | 本節的關係 |
|:--|:--|:--|
| A 層 canonical 物件（obs/time interface、transition、innovation、claim space、identifiability） | 本檔 §2–§7（frozen） | 本節**綁定**，不重定義 |
| B1 activation gate、frozen inputs 清單、**B1-slot identity**、**score-layer contract**、evaluation design（fold／trial unit／metric／threshold／blind-reveal） | [B1 charter](../threads/gctm_b1_runtime_grounded_offline_attribution_task.md) | 本節只**指向**，不複寫、不預決 |
| 跨 task 共享語意（hook scope、reserved-symbol 規則，如 base score 寫 \(s_{\mathrm{base}}\) 而 `s0` 為保留名） | [B1/O1 synthesis core](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md) §0／§2.1 | 本節只**指向** |
| runtime observability／evidence fidelity | H0 declaration | 本節只**指向** |

**欄位命名（frozen）：** 與 B1 charter frozen-inputs 同名的欄位（如
`coordinate_substrate_id`、`observation_mode`、`causal_availability`、
`context_definition`、`P0_exit_cov_id`、`R1_obs_cov_id`）即**同一物件**，沿用其名，
**不另造名**；本節新增的欄位只補 A 層 instantiation 所必需者。

### §8.2 Block D — declaration-level 欄位（每個 study 宣告一次）

| 欄位 | 必要性 | 值域／型別 | 綁定 |
|:--|:--|:--|:--|
| `gctm_spec_id` | required | 本檔的 exact identity（path + commit/hash） | 指名被 instantiate 的 frozen §2–§8 |
| `coordinate_substrate_id` | required | substrate 識別；A 層 latent state 於 \(\mathbb R^{2d}\) | §2 field 1（declared-target 地位不變） |
| `coordinate_dim_d` | required | \(d\in\mathbb N_{\ge1}\)（production-corresponding \(d=2\)） | §4.1／§4.7 |
| `frame_time_unit` | required | canonical = 1 frame interval；秒級須另附映射 | §2 field 2／field 7 |
| `observation_mode` | required | 恰一個 \(\in\{H_x,H_{xv}\}\) | §3；production-corresponding 綁 \(H_x\) |
| `causal_availability` | required-if `observation_mode=`\(H_{xv}\) | entry velocity 的**可得時點與來源**宣告 | §3.3 |
| `error_dependence` | required | \(\in\{\texttt{independent},\ \texttt{dependent}\}\)；canonical = `independent`（\(C=0\)） | §5.4 |
| `C_cross_cov_id` | required-if `error_dependence=dependent` | \(C\) 的來源；須落在 joint-PSD domain | §5.4／§5.6 |
| `gamma_status` | required | \(\in\{\texttt{known},\ \texttt{unknown}\}\) | §7.4 |
| `gamma_value` | required-if `gamma_status=known` | \(\gamma\in[0,\infty)\)，單位 \(\mathrm{frame}^{-1}\)（\(\gamma=0\)=M1 boundary） | §4.6／§4.7 |
| `joint_map_condition_status` | required-if `gamma_status=unknown` | \(\in\{\texttt{global\_injectivity\_established},\ \texttt{local\_full\_rank\_established},\ \texttt{not\_established}\}\)＋證據來源；**本規格不證明任一者** | §7.4／§7.7 |
| `gauge_fixing` | required | \(\in\{\texttt{R1\_declared},\ \texttt{Pxx\_declared},\ \texttt{none}\}\)（固定 \(P_{xx}\!\leftrightarrow\!R_1\) gauge 的**額外獨立資訊**） | §7.4 G1／§7.6 |
| `parameter_sharing_scope` | required | 哪些 events 共享同一 \((P_0,\gamma,D,R_1)\)（identifiability 的 population 定義） | §7.4 |
| `context_definition` | required | \(\mathcal C_{\mathrm{exit}}\) 的定義；須僅含 exit-causal 資訊 | §4.0 boundary 4／§4.7 |
| `context_varies` | required | 布林：\(c\) 在 population 中是否**變化**（決定 \(\bar v(c)\) 可否與常數 mean 分離） | §7.4／§7.6 mean-level |
| `claim_target` | required | 恰一個 \(\in\{\texttt{CAL},\ \texttt{RANK}\}\)；兩者須**各自**一份宣告 | §6.2（CAL/RANK 永不互相冒充） |
| `cal_working_null` | required-if `claim_target=CAL` | §6.1 的 Gaussian working null（\(\delta z_0\sim\mathcal N(0,P_0)\)）之顯式宣告 | §6.1 |
| `region_definition` | required-if 使用 \(\Pi(\Omega)\) | \(\Omega\) 的定義；candidates 間是否全等須寫明 | §6.1／§6.3 region caveat |
| `operator_offset_declared` | required | 布林＋其構成（`bridge_at` 與所用 \(v\) 的來源） | §2 rows 8–9／§4.4 |
| `missing_value_rule` | required | 必須為 fail-closed（缺 required 欄位 ⇒ record inadmissible），**不得**為 imputation | W9 |

### §8.3 Block E — event-level 欄位（每個 candidate event 一列）

Candidate event \(\mathcal E\) 的定義沿用 §6.2（一條 lost track 的固定 exit anchor
＋一組有限 entry candidates），不重定義。

| 欄位 | 必要性 | 值域／型別 | Causal availability | 綁定 |
|:--|:--|:--|:--|:--|
| `event_key` | required | 全域唯一 | — | §6.2；RANK 的 evaluation unit |
| `exit_anchor_frame` | required | frame index（lost track 最後被觀測 frame） | exit-time | §2 field 3（exit endpoint） |
| `z0_hat` | required | \(\hat z_0=[\hat x_0;\hat v_0]\in\mathbb R^{2d}\)；\(x\):\(\ell\)、\(v\):\(\ell/\mathrm{frame}\) | exit-time | §5.2 |
| `P0_exit_cov_id` | required | \(P_0\succeq0\)（\(2d\times2d\)）之來源／識別 | exit-time | §5.1／§5.6 |
| `context_c` | required | \(c\in\mathcal C_{\mathrm{exit}}\) | exit-time | §4.7 |
| `candidate_count` | required | \(\lvert\mathcal E\rvert\in\mathbb N_{\ge1}\) | — | §6.2 |

### §8.4 Block P — pair-level 欄位（每個 (event, candidate) 一列）

| 欄位 | 必要性 | 值域／型別 | Causal availability | 綁定 |
|:--|:--|:--|:--|:--|
| `event_key` | required | 指向 Block E | — | conservation（W8） |
| `candidate_key` | required | event 內唯一 | — | §6.2 |
| `entry_anchor_frame` | required | frame index（candidate 第一次被觀測 frame＝entry endpoint） | entry-time | §2 field 5／field 6 |
| `g_phys` | required | \(g_{\mathrm{phys}}=\Delta\in\mathbb N_{\ge1}\)，單位 frame | 兩 endpoint 皆定後 | §2 field 3；canonical transition index |
| `bridge_at` | required | event 當下綁定的 runtime 值（**不**凍結常數） | 宣告時 | §2 field 6 |
| `delta_on` | required | \(\Delta_{\mathrm{on}}=\mathrm{la}\in\mathbb N\)，單位 frame | entry-time | §2 field 4 |
| `anchor_pair` | required | 明寫本列 transition 的兩個 anchor（canonical＝exit endpoint → entry endpoint） | — | §2 field 5（W2） |
| `y1_obs` | required | entry observation；\(H_x\):\(\mathbb R^{d}\)（\(\ell\)）、\(H_{xv}\):\(\mathbb R^{2d}\) | entry-time | §5.3 |
| `R1_obs_cov_id` | required | \(R_1\succeq0\)，維度隨 \(H\) | entry-time | §5.1／§5.6 |
| `H_i` | required | 本列使用的 observation mode，須等於 Block D 的 `observation_mode` | — | §3（W3） |
| `operator_offset` | required-if `operator_offset_declared=true` | \(\pm(\mathrm{bridge\_at}-1)v\)，**獨立欄位** | derived @ entry endpoint | §2 rows 8–9（W2） |
| `label_true_match` | required-if 進行 CAL 或 RANK 評估 | 布林／候選標記；**evaluation-only** | post-hoc（**不得**進入任何 score 欄位） | §6.2／§7.5（W8 label isolation） |
| `admissibility` | required | \(\in\{\texttt{admitted},\ \texttt{dropped}\}\)；`dropped` 須附**恰一個**列舉理由 | — | W9 |

### §8.5 Block X — derived 量（由 D1 計算；**不得**由 input 冒充）

下列量**不是** input：它們由 Block D/E/P 的欄位依 frozen §4–§6 唯一決定。schema 允許
input 附帶它們**僅作為 cross-check**；若附帶，其值必須與依 frozen 公式重算者一致
（W5），否則該 record inadmissible。**任何以外部值取代重算的做法，即脫離本 D1 的
instantiation。**

| Derived 量 | 由誰決定 | 來源 § |
|:--|:--|:--|
| \(a,b\)、\(A_\Delta\) | \((\gamma,\Delta)\) | §4.3 |
| \(d_\Delta(c)\) | \((\gamma,\Delta,\bar v(c))\) | §4.3；**不含** operator offset（§4.4） |
| \(Q_\Delta\) | \((\gamma,D,\Delta)\) | §4.5（\(\gamma=0\)：§4.6） |
| \(m^-_\Delta=A_\Delta\hat z_0+d_\Delta(c)\)、\(e^-=z_\Delta-m^-_\Delta\) | Block E＋上列 | §5.2（符號約定 frozen） |
| \(P^-_\Delta=A_\Delta P_0A_\Delta^\top+Q_\Delta\) | \(P_0,A_\Delta,Q_\Delta\) | §5.2 |
| \(r=y_1-Hm^-_\Delta\) | Block P＋上列 | §5.3 |
| \(S_\Delta=HP^-_\Delta H^\top+R_1\)（canonical \(C=0\)）／＋\(HC+C^\top H\)（dependent） | `error_dependence` | §5.4 |
| \(q,\ \log\det S_\Delta,\ E,\ \Pi(\Omega)\) | \(r,S_\Delta,\Omega\) | §6.1（需 \(S_\Delta\succ0\) regime） |

### §8.6 Well-formedness predicates（frozen；fail-closed、機械可判定）

| # | Predicate | 違反後果 | 來源 |
|:--|:--|:--|:--|
| **W1** | **Anchor identity**：\(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}+(\mathrm{bridge\_at}-1)\) 逐列成立；且 \(g_{\mathrm{phys}}=\mathrm{entry\_anchor\_frame}-\mathrm{exit\_anchor\_frame}\ge1\)、`bridge_at`\(\ge1\) | record inadmissible | §2 field 5／field 3 |
| **W2** | **Anchor 宣告與 offset 分離**：每列 `anchor_pair` 明寫；production-corresponding instantiation 必須為 exit→entry 並**重現** \(\pm(\mathrm{bridge\_at}-1)v\)，且該 offset 只能出現在 `operator_offset` 欄位——**不得**折進 \(d_\Delta(c)\)、\(m^-_\Delta\) 或 \(e^-\) | record inadmissible；折入者為 layer 混淆 | §2 rows 8–9／§4.4 |
| **W3** | **Observation mode 一致性**：全 study 恰一個 \(H\)，且每列 `H_i` 與之相同；\(H_{xv}\) 須有 `causal_availability`；若 entry velocity 由與 \((0,\Delta]\) **重疊**的 frames 導出 ⇒ `error_dependence` 必須為 `dependent` 且 `C_cross_cov_id` 已宣告 | declaration inadmissible | §3.3／§5.4／§7.2 |
| **W4** | **PSD／可逆性**：\(P_0\succeq0\)、\(R_1\succeq0\)、\(D\succeq0\)；dependent path 另需 \(\big[\begin{smallmatrix}P^-_\Delta&C\\ C^\top&R_1\end{smallmatrix}\big]\succeq0\)。使用 \(q/E/\Pi\) 的列另需 \(S_\Delta\succ0\)（canonical \(C=0\)＋\(R_1\succ0\) 即足；dependent path 需額外 nondegeneracy） | 違反 PSD ⇒ inadmissible；\(S_\Delta\) 退化 ⇒ 該列不得產生 \(q/E/\Pi\) | §5.5／§5.6／§6.1 |
| **W5** | **維度／單位／derived 一致性**：所有矩陣維度與 `coordinate_dim_d` 及 \(H\) 相容；單位依 §4.7／§5.6／§6.5；附帶的 Block X 值必須與依 frozen 公式重算者一致 | record inadmissible | §4.7／§5.6／§8.5 |
| **W6** | **Identifiability regime**（承 §7，逐項機械判定）：(a) `gamma_status=known` 時，quotient claim 需 `parameter_sharing_scope` 內 **\(\ge4\) 相異 \(\Delta\)**；single-\(\Delta\) population 連 quotient 都不可識別。(b) `gamma_status=unknown` 時，任何依賴 \(\gamma\)（或 \(D\) 與 \(\gamma\) 分離）的 claim 需 **\(>4\) 相異 \(\Delta\)** **且** `joint_map_condition_status`\(\neq\)`not_established`；「\(\ge5\) gaps」只是**必要**非充分。(c) 需要 \(P_{xx}\) 與 \(R_1\) **分離**的 claim 需 `gauge_fixing`\(\neq\)`none`。(d) 需要 \(\operatorname{asym}(P_{xv})\) 的 claim 需 \(H_{xv}\) ＋ multi-gap ＋ 共享參數 ＋ \(\gamma\) regime 滿足。(e) 需要 \(\bar v(c)\) 的 claim 需 `context_varies=true` 且 context 為 exit-causal | 對應 claim **non-identifiable**（§8.7；落入 terminal-3 rejection region 的判定屬 WP-A8） | §7.3／§7.4／§7.6／§7.7 |
| **W7** | **Claim／evaluation-unit 一致性**：`claim_target=CAL` 的評估只在 \(\{(e,i^\star)\}\)（true match）上做且需 `cal_working_null`；`claim_target=RANK` 只做 event-local ordering（event 內計算後再跨 event 平均）。**同一份宣告不得同時主張兩者**；CAL 結論不得轉述為 RANK，反之亦然 | claim inadmissible | §6.2／§6.0 |
| **W8** | **Conservation 與 label isolation**（A 層最小集）：每個 admitted pair 屬**恰一個** event；每個 event 恰一個 exit anchor 與一組不重複 candidates；`label_true_match` 與任何 GT 導出量為 **evaluation-only**，**不得**作為 Block D/E/P 任一 score-relevant 欄位的輸入。（fold／partition／trial-unit／blind-reveal 的完整 conservation 由 B1 charter 所有，本節不複寫） | record 或 study inadmissible | §6.2；B1 charter conservation identities |
| **W9** | **Fail-closed missing-value**：缺任一 `required`（或已觸發的 `required-if`）欄位 ⇒ 該 record／declaration **inadmissible**，**不得**以預設值、鄰值或全域常數補齊；`dropped` 列須附**恰一個**列舉理由；optional 欄位缺席須**明確記錄**，其後果見 §8.7 | inadmissible；靜默補值即脫離本 D1 | §2 row 9 精神（不得靜默繼承） |

### §8.7 Claim-restriction map（缺宣告 ⇒ 哪些 claim 不可主張；承 §7 leakage matrix）

本表把 §7.6 的**阻斷條件**翻譯成 schema-level 的機械判定：左欄的宣告缺席時，右欄的
claim 因 leakage／未滿足 regime 而**不可主張**（其餘 claim 不受影響）。

| 缺席的宣告／欄位 | 因此**不可主張**的 claim | 仍可主張者 | 依據 |
|:--|:--|:--|:--|
| `gauge_fixing=none` | 任何需要 \(P_{xx}\) 與 \(R_1\) **分離**的 claim（結構性 gauge，\(H_{xv}\) 亦不可破） | 只依 \(P_{xx}+R_1\) 之和的 quotient claim | §7.4 G1／§7.6 |
| `gamma_status=unknown` 且（\(\le4\) 相異 \(\Delta\) 或 `joint_map_condition_status=not_established`） | 任何識別 \(\gamma\)、或需 \(\gamma\) 與 \(D\) 分離的 claim | 給定 \(\gamma\) 條件下的 quotient claim（須明寫其 conditional 前提） | §7.4 G3／§7.6／§7.7 |
| `observation_mode=`\(H_x\) | 任何需要 \(\operatorname{asym}(P_{xv})\) 的 claim（\(H_x\) 下 structural invisible） | \(\operatorname{sym}(P_{xv})\)／\(P_{vv}\)／\(D\)（在 W6(a) regime 下） | §7.4 G2／§7.6 |
| `context_varies=false` 或 `context_definition` 未宣告 | 任何 \(\bar v(c)\)（context drift）claim | 常數-mean 層級的敘述（須標明混淆） | §7.4／§7.6 mean-level |
| `operator_offset_declared=false` | 任何 mean／residual-bias claim（operator offset 與 \(\bar v(c)\)／常數 bias 混淆） | 純 covariance-level claim | §7.6 mean-level／§2 row 9 |
| `label_true_match` 不可得 | **CAL 與 RANK 皆不可主張**（兩者皆以 true-match label 為前提） | 僅 model-side 的定義性敘述（無 claim） | §7.5 |
| `cal_working_null` 未宣告 | CAL 的**分佈層** claim（\(q\sim\chi^2_k\)、coverage／PIT） | \(q,\log\det S_\Delta,E,\Pi\) 作為 \(r,S_\Delta\) 的**函數**仍 well-defined | §6.1 |
| `region_definition` 未宣告（卻使用 \(\Pi\)） | 任何 \(\Pi\)-based claim | \(q\)／NLL-based claim | §6.1／§6.3 |
| single-event population（`parameter_sharing_scope` 只含一個 \(\Delta\)） | \(\{P_0,\gamma,D,R_1,\bar v\}\) 的**任何**識別 claim（連 quotient 都不可） | 無（此 regime 下 identifiability 最強收縮） | §7.3 |
| `error_dependence=dependent` 但 \(S_\Delta\succ0\) 未另行確立 | 任何需 \(S_\Delta^{-1}\) 的 claim（\(q/E/\Pi\)） | \(S_\Delta\succeq0\) 層級的敘述 | §5.5／§6.1 |

**跨欄位的合成規則（frozen）：** 多個缺席同時發生時，不可主張的 claim 集合為各列的
**聯集**；一個 claim 只要落入任一列的禁區即不可主張。CAL 與 RANK 之間**互不救援**
（一者可主張不使另一者可主張，§7.5）。

### §8.8 本節顯式不解決（typed deferrals）

| 項目 | 擁有 WP／文件 | 狀態 |
|:--|:--|:--|
| terminal review（checklist artifact + 機械 terminal selection） | 後續 packet（WP-A8 planned） | unresolved |
| **B1-slot identity**、**score-layer contract**、B1 activation gate、frozen-inputs 清單 | [B1 charter](../threads/gctm_b1_runtime_grounded_offline_attribution_task.md)（owner 前置） | 不由本節決定 |
| evaluation design：fold／trial unit／dependence treatment／minimum exposure／minimum effect／short-gap retention bar／metric／threshold／blind-reveal 協定 | B1 charter | 不在 A 層 schema |
| reserved-symbol 規則（base score 寫 \(s_{\mathrm{base}}\)、`s0` 為保留名）與 hook scope | [synthesis core](../../modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md) §0／§2.1 | 只指向，不複寫 |
| storage format／serialization／schema versioning 機制 | — | 本節只定 field semantics，不選格式 |
| runtime 可觀測性、H0 guarantee／fidelity edge、欄位是否**真的**可從 production 取得 | H0 路徑 | 不授權（本 schema 只列 requirement，非 availability claim） |
| 實際 data／capture／fitting／identification／任何 gain 數值 | — | 不授權（charter Non-scope） |
| reverse-time／candidate-backward atom | 後續 | typed boundary only（§4.0 boundary 3） |
| B1/O1、H0、runtime、online、production | — | 不授權（charter Non-scope） |

**Freeze 邊界：** 本節凍結的是 **schema 的 interface shape**——欄位集、型別／單位／
causal-availability 綁定、input-vs-derived 分界、well-formedness predicates 與
claim-restriction map。它不宣稱任何 runtime 擷取值、不建立 fidelity edge、不啟用 B1、
不取 data、不選任何數值／metric／format、不選 terminal。修改須 append-only correction
（原文保留、註記 superseded）。

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
  separation **as a 4-shape decomposition**
  (\(S_\Delta=(P_{xx}+R_1)+a\,\mathrm{sym}(P_{xv})+a^2P_{vv}+q_{xx}D\), reusing frozen
  §4.6/D2 L2/L4) — so, **given \(\gamma\) known**, \(H_x\) multi-gap identifies only
  the **quotient** \(\{D,P_{vv},\mathrm{sym}(P_{xv}),P_{xx}+R_1\}\) (with \(\gamma\)
  **unknown**, 4 gaps leave a degree of freedom so \(\gamma\) is unidentified —
  identifying it needs \(>4\) gaps + joint-map global injectivity (global id) or at
  least full-Jacobian-rank (local id; not equivalent), not claimed proven), leaving
  **two structural non-identifiable directions** (the \(P_{xx}\leftrightarrow R_1\)
  gauge, not broken
  even by \(H_{xv}\), only by declared known \(R_1\)/\(P_{xx}\); and
  \(\mathrm{asym}(P_{xv})\), \(H_x\)-invisible — under \(H_{xv}\) it becomes
  **observable** but its identification still needs multi-gap + shared params +
  known \(\gamma\), not a single \(H_{xv}\) event) — the
  claim-level result (CAL and RANK are **mutually non-identifying** — ranking does
  not identify the calibration scale \(\alpha_\Delta\); a uniform monotone reparam
  leaves order fixed but breaks calibration), and a frozen **leakage matrix**
  (covariance-/mean-/claim-level, separating multi-gap-blockable leaks from the
  structural gauges, each with its blocking condition). The verdict is conditional:
  identifiability is **specified**, not empirically established (no data authorized);
  a claim is non-identifiable exactly when it must rely on an unidentifiable
  component **or an unmet identifiability regime** (the two structural gauges, or
  \(\gamma\) unknown without the \(>4\)-gap joint-map condition). §7 selects **no
  terminal** — it is the predicate object for WP-A8
  terminal review. Frozen §2–§6 kept byte-frozen;
  the stale `reserved`/`unresolved` identifiability references in §5.7/§6.6 are
  superseded via an append-only status note after the §1 table (no in-place edit).
  No new file ⇒ no master_map regeneration. Claims numerically sanity-checked
  (single-gap non-identifiability; the multi-gap quotient + the two structural
  gauges G1/G2 including \(H_{xv}\)-persistence; the joint-\(\gamma\) 4-gap DOF
  ambiguity G3; offset/drift mean confounding; CAL/RANK mutual non-identification).
- 2026-07-22 — bounded correction per #256 owner review (BLOCKED → fixed
  pre-merge; §7 not yet frozen; §2–§6 byte-frozen): the §7.4 multi-gap claim that
  the **full** \(\{P_0,\gamma,D,R_1\}\) is generically identifiable under \(H_x\) was
  **wrong**. Corrected via the shape decomposition
  \(S_\Delta=(P_{xx}+R_1)+a\,\mathrm{sym}(P_{xv})+a^2P_{vv}+q_{xx}D\): \(H_x\)
  multi-gap identifies only the quotient \(\{\gamma,D,P_{vv},\mathrm{sym}(P_{xv}),
  P_{xx}+R_1\}\); the \(P_{xx}\leftrightarrow R_1\) split and \(\mathrm{asym}(P_{xv})\)
  are structurally non-identifiable (the former even under \(H_{xv}\), fixable only
  by declared independent \(R_1\)/\(P_{xx}\); the latter \(H_x\)-invisible,
  \(H_{xv}\)-identifiable). Updated §7.4, the §7.6 covariance leakage table (split
  the old \(P_0\!\leftrightarrow\! R_1\) row into multi-gap-blockable vs structural),
  the §7.7 verdict item 1 + honesty boundary, and the §7.8 regime summary. The old
  C2 sanity (one chosen pair separates) does not prove injectivity; replaced by the
  gauge argument and the G1/G2 numeric checks.
- 2026-07-22 — second bounded correction per #256 owner re-review (still
  pre-merge; §7 not yet frozen; §2–§6 byte-frozen): the §7.4 rank-4 argument
  silently assumed \(\gamma\) **known** (the shapes \(a(\gamma,\cdot),q_{xx}(\gamma,\cdot)\)
  depend on \(\gamma\)). With \(\gamma\) **unknown** and exactly 4 gaps the DOF count
  is \(4r\) observations vs \(4r+1\) unknowns (\(r=d(d+1)/2\), the \(+1=\gamma\)), so
  \(\gamma\) is **not identifiable**: any \(\gamma'\) re-solves the invertible
  \(4\times4\) shape system and refits the same four \(S_\Delta\) exactly, staying
  PSD-admissible for \(\gamma'\) near \(\gamma\) (numeric G3: a continuum of exact
  admissible refits). Corrected to make coefficient-matrix identifiability
  **conditional on \(\gamma\) known** (quotient \(\{D,P_{vv},\mathrm{sym}(P_{xv}),
  P_{xx}+R_1\}\)); \(\gamma\) identification now requires \(>4\) gaps **and** the
  joint-map identifiability condition + non-degenerate coefficients (see round 3 for
  the injectivity-vs-Jacobian precision), **not proven here** (so the spec does not
  claim \(\gamma\) identified; "\(\ge5\) gaps" is only the DOF-necessary correction,
  not sufficient). Updated §7.4 (γ-DOF paragraph), the §7.6 \(\gamma\!\leftrightarrow\! D\)
  row, §7.7 verdict item 1, §7.8 regime summary (dedicated \(\gamma\)-unknown row),
  plus charter status/History. New numeric check G3 (verify_wp_a6_gamma.py). The
  \(P_{xx}\leftrightarrow R_1\) and \(\mathrm{asym}(P_{xv})\) structural gauges from
  round 1 are unchanged.
- 2026-07-22 — third bounded correction per #256 owner re-review (still pre-merge;
  §7 not yet frozen; §2–§6 byte-frozen): three precision fixes. (1) **\(H_{xv}\) makes
  \(\operatorname{asym}(P_{xv})\) observable, not single-event identifiable** — the
  canonical cross-block \(S_{xv}(\Delta)=bP_{xv}+ab\,P_{vv}+q_{xv}(\gamma,\Delta)D+R_1^{xv}\)
  also carries process + measurement covariance, so a single \(H_{xv}\) event cannot
  separate \(P_{xv}\) (\(\operatorname{asym}(P_{xv})\) trades with \(R_1^{xv}\), numeric
  G4a); identification still needs multi-gap + shared params + known \(\gamma\) (G4b).
  Reworded §7.4 joint-mode paragraph, the §7.6 \(\operatorname{asym}\) row, §7.7 item 1,
  and the §7.8 \(\operatorname{asym}\) regime (no longer "\(H_{xv}\) cross-block" alone).
  (2) **honesty boundary widened** — "non-identifiable exactly when it must rely on an
  unidentifiable **gauge component**" → "an unidentifiable component **or an unmet
  identifiability regime**", explicitly listing \(\gamma\) unknown (without the
  \(>4\)-gap joint-map condition) as a terminal-3 case, alongside the two gauges (§7.7;
  same fix in charter). (3) **injectivity ≠ full-Jacobian-rank** — the two are not
  synonyms: joint-map **global injectivity** ⇒ global identification, **full-Jacobian-rank**
  ⇒ only local identification; the "\(/\)" was replaced by the explicit
  global/local distinction in §7.4, §7.6, §7.7, §7.8. New numeric check
  verify_wp_a6_hxv.py (G4a/G4b). Round-1/round-2 results otherwise unchanged.
- 2026-07-22 — fourth (sentence-level) bounded correction per #256 owner re-review
  (still pre-merge; §7 not yet frozen; §2–§6 byte-frozen): the §7.7 honesty summary
  still said the **two** structural directions persist "even under \(H_{xv}\)", which
  is true only for \(P_{xx}\leftrightarrow R_1\); it contradicted the round-3 body for
  \(\operatorname{asym}(P_{xv})\) (which becomes **observable** under \(H_{xv}\)). Rewrote
  the sentence to scope each direction separately: (i) \(P_{xx}\leftrightarrow R_1\)
  structural, persists under \(H_{xv}\); (ii) \(\operatorname{asym}(P_{xv})\) invisible
  only under \(H_x\), observable under \(H_{xv}\) (identification still needs multi-gap +
  shared params + \(\gamma\) regime); (iii) \(\gamma\) unknown = unmet regime. No other
  change.
- 2026-07-23 — **§8 frozen by WP-A7** (last D1 deliverable: schema-only interface
  for a separately declared future B1 input). Defines the consumption rule
  (a future input instantiates this D1 **iff** it supplies every `required` /
  triggered `required-if` field, passes all well-formedness predicates, and keeps
  its claims inside the claim-restriction map — otherwise it is **not** an
  instantiation and may not cite any §6/§7 conclusion); the authority split
  (B1-slot identity, score-layer contract, activation gate, evaluation design and
  reserved-symbol rules stay owned by the B1 charter / synthesis core — linked,
  not restated); four field blocks — **D** declaration-level (spec identity,
  \(d\), observation mode + causal availability, `error_dependence`,
  `gamma_status`/`joint_map_condition_status`, `gauge_fixing`,
  `parameter_sharing_scope`, context + `context_varies`, `claim_target`,
  `cal_working_null`, `region_definition`, `operator_offset_declared`,
  fail-closed `missing_value_rule`), **E** event-level (`event_key`, exit anchor,
  \(\hat z_0\), `P0_exit_cov_id`, context, candidate count), **P** pair-level
  (entry anchor, \(g_{\mathrm{phys}}\), `bridge_at`, \(\Delta_{\mathrm{on}}\),
  `anchor_pair`, \(y_1\), `R1_obs_cov_id`, \(H_i\), `operator_offset`,
  evaluation-only `label_true_match`, `admissibility`), and **X** derived
  (\(a,b,A_\Delta,d_\Delta(c),Q_\Delta,m^-_\Delta,e^-,P^-_\Delta,r,S_\Delta,
  q,\log\det S_\Delta,E,\Pi\)) which are **inputs to nothing** — supplying them
  in place of recomputation leaves the instantiation. Nine fail-closed
  predicates W1–W9 (anchor identity \(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}+
  (\mathrm{bridge\_at}-1)\); anchor declaration with the operator offset kept in
  its own field, never folded into \(d_\Delta/m^-_\Delta/e^-\); observation-mode
  consistency with the overlap ⇒ dependent-\(C\) rule; PSD/invertibility;
  dimension/unit/derived consistency; the §7 identifiability regime; CAL/RANK
  evaluation-unit separation; conservation + **label isolation**; fail-closed
  missing values) and a claim-restriction map translating the §7.6 blocking
  conditions into per-field mechanical verdicts (union rule across fields; CAL
  and RANK never rescue each other). §8 is schema-only: no B1 activation, no
  runtime-availability claim, no data/fitting, no format choice, no terminal.
  §2–§7 byte-frozen; the §6.6/§7.9 "§8 reserved" rows are superseded by the §1
  append-only correction. Predicate decidability checked on synthetic records in
  session scratchpad: a conforming declaration passes all nine predicates, 17
  injected violations (one or more per predicate, incl. \(\Delta_{\mathrm{on}}=
  g_{\mathrm{phys}}\), offset folded into the mean, \(H_{xv}\) overlap without a
  declared \(C\), non-PSD \(R_1\), supplied \(S_\Delta\ne\) recomputed, each W6
  regime case, CAL claim on a RANK declaration, label used in a score, duplicate
  pair key, unfilled `required-if`) are each caught by the intended predicate,
  and the §8.7 union rule leaves the quotient claim admissible while forbidding
  the three blocked ones. No code committed.
