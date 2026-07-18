# Existing Online Object Analysis for GCTM Alignment

**用途：** 作為 GCTM 對齊時的 existing-online-object authority。
**範圍：** production native bridge 的數學／runtime object；不定義 GCTM 本身，不做 H0、exporter、實驗或模型選型。
**Runtime authority：** `src/tracking/tracker_gpu.cu`，已對照 blob `6b2e731a806eddc3510cf68ca2f15cb5ac5e430e`。
**既有裁決：**

```text
MATCH_WITH_BOUNDED_CORRECTIONS
GAIN_ONLY
NO_CANONICAL_GLOBAL_CONTRACTION
```

---

# 0. Executive answer

既有 online object 不是單一 track 的 motion transition，也不是內生 stochastic kernel：

$$
P_\Delta(x,\cdot).
$$

它是一個在單次 bridge fire event 上執行的、per-lost-gap-indexed、
context-conditioned、deterministic hybrid decision operator。其 candidate-local
proposal stage 與 full event operator 必須分開：

$$
P_j:
\left(
x_{C_j},q_j,
\{(x_L^{(i)},\Delta_i)\}_{i\in\mathcal I_{\mathrm{struct},j}},
m_{\mathrm{gate}},\theta
\right)
\longrightarrow
\left(z_{c,j},\operatorname{proposal}_j\right),
\qquad
\Delta_i=\operatorname{age}[L_i],
$$

$$
\boxed{
F_{\mathrm{event}}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j
:
\mathcal X_{\mathrm{event}}
\longrightarrow
\mathcal Z_c\times\mathcal Z_d
}
$$

其中：

- $\Delta_i$ 是 lost $L_i$ 的 production `age[lost]`，亦即該 pair score extrapolation 使用的 `la`；
- $\mathcal X_{\mathrm{event}}$ 是完整 candidate universe、每個 candidate 的 structural lost competitors、pair-specific horizons，以及 gate／claim 所需 runtime state；
- $\mathcal C$ 是固定 policy 與不作為幾何擾動軸的外生條件；actual claim set 是上式的中間輸出，不是預先固定的 context；
- $\mathcal Z_c$ 是連續或 piecewise-smooth 的 score quantities；
- $\mathcal Z_d$ 是 gate、rank、claim、commit 等離散結果；
- 固定 runtime input 時，輸出是確定的；
- commit 只進行 identity adoption，不把 lost motion state merge 到 candidate motion state。

candidate-local score/proposal family 可等價記為

$$
P_j
\equiv
F^{\mathrm{proposal}}_{j\mid\mathcal C_j}.
$$

它接收該 candidate 的 per-lost
\(\{(x_L^{(i)},\Delta_i)\}_i\)，且不輸出 claim/commit；單一
\(F_{\Delta\mid\mathcal C}\) 不得再同時指稱 full claim/commit event。

最精確的 production decomposition 是：

$$
\boxed{
\left\{
R_j
\rightarrow
M_{\mathrm{pre},j}
\rightarrow
G_{\Delta_i}
\rightarrow
M_{\mathrm{post},j}
\rightarrow
S_{\mathrm{rank},j}
\rightarrow
\operatorname{proposal}_j
\right\}_{j\in\mathcal U_{\mathrm{event}}}
\rightarrow
A_{\mathrm{claim}}
\rightarrow
\mathrm{commit}
}
$$

因此 GCTM 對齊的起點不是「建立一個取代 online 的 transition」，而是：

> 明確說明 GCTM 的 latent／probabilistic object 如何對應、包覆或條件化這個既有 deterministic hybrid operator。

---

# 1. Object boundary

## 1.1 Event boundary

既有 object 發生在單一 bridge fire frame：

```text
same-frame association completed
→ matched state / foot history updated
→ candidate satisfies bridge fire condition
→ native bridge propose / rank / claim
→ optional commit
```

事件輸入位於 commit 前；事件輸出包含 commit decision。

自然的完整事件表示是：

$$
\left(
\left\{
\left(
x_{C_j}^{\mathrm{entry}},
q_j,
\{(x_L^{(i),\mathrm{exit}},\Delta_i)\}_{i\in\mathcal I_{\mathrm{struct},j}}
\right)
\right\}_{j\in\mathcal U_{\mathrm{event}}},
\mathcal C
\right)
\longmapsto
(z_c,z_d).
$$

其中 candidate-local $P_j$ 先產生 $\operatorname{proposal}_j$；只有完整
universe 的 proposal 再經 $A_{\mathrm{claim}}$ 才能產生 claim/commit。

它不是多幀 recurrent state evolution：

$$
x_t\to x_{t+1}\to x_{t+2}.
$$

gap 的歷史效果已被壓入：

- lost frozen native history；
- candidate live entry history；
- 每個 lost 的 pair horizon $\Delta_i$；
- same-frame competitor／claim context。

## 1.2 What the object is not

既有 object **不是**：

1. 單一 lost track 的 Markov transition；
2. lost motion state 向 candidate motion state的收斂；
3. stochastic transition kernel；
4. offline trajectory reconstruction；
5. GT-correct／wrong label；
6. 整個 tracker 的全時域 dynamical system；
7. GCTM 的 latent physical transition 本身。

其中 GT correct／wrong relink 是事後 attribution label，不屬於 pure online operator output。

---

# 2. Input object

## 2.1 Native pair state

對每個 lost–candidate pair，native reduction 前的輸入包括：

### Lost side

- frozen foot／center history；
- last four points，短窗不足時使用 last one；
- native EMA height；
- lost age；
- structural state／eligibility fields；
- optional gate inputs。

### Candidate side

- live head-four history；
- native EMA height；
- candidate fire condition；
- current track／detection score；
- structural state／eligibility fields；
- optional gate inputs。

pair-level native reduction：

$$
R(x_L,x_C)
=
(u_L,u_C)
$$

其中：

$$
u_L=(p_L,v_L,h_L^{\mathrm{ema}}),
\qquad
u_C=(p_C,v_C,h_C^{\mathrm{ema}}).
$$

`bridge_vel4` 的 production velocity 為：

$$
v
=
\frac{3y_3+y_2-y_1-3y_0}{10}.
$$

短 lost window 不足四點時：

$$
v_L=0.
$$

## 2.2 Event-level geometry state

單一 pair 不足以決定 rank outcome。event-level geometry 必須包含：

$$
x_{\mathrm{geom}}
=
\left(
x_C,
\{x_L^{(i)}\}_{i\in\mathcal I_{\mathrm{struct}}}
\right).
$$

其中 $\mathcal I_{\mathrm{struct}}$ 是本次 candidate 可掃描的 structural lost set。

重要區分：

$$
\mathcal I_{\mathrm{struct}}
\neq
\mathcal I_{\mathrm{pre}}
\neq
\mathcal I_{\mathrm{rank}}.
$$

它們依序由 runtime 產生：

$$
\mathcal I_{\mathrm{pre}}
=
M_{\mathrm{pre}}
(\mathcal I_{\mathrm{struct}}),
$$

$$
\mathcal I_{\mathrm{rank}}
=
M_{\mathrm{post}}
\left(
G_\Delta(\mathcal I_{\mathrm{pre}})
\right).
$$

因此：

> $\mathcal I_{\mathrm{rank}}$ 不應被預先放進 context 定義；它是 operator 中間產物。

這點對 GCTM 很重要，否則會把 gate／cutoff outcome 當成先驗條件，形成循環定義。

## 2.3 Claim state

candidate-local ranking 後，還存在跨 candidate 的 claim competition：

$$
\mathcal J
=
\{
j:
\text{candidate }j\text{ proposes the same lost}
\}.
$$

claim 使用：

$$
q_j
=
\text{candidate detection／track score}
$$

以及 candidate index 建 packed key。它不重新比較 bdist。

因此 claim state 與 geometry rank state 必須分開：

$$
\mathcal I:
\text{lost competitors for one candidate},
$$

$$
\mathcal J:
\text{candidate competitors for one lost}.
$$

## 2.4 External context $\mathcal C$

建議 GCTM 對齊時把 context 寫成：

$$
\mathcal C
=
\left(
\theta,
\mathcal T_{\mathrm{rank}},
\mathcal T_{\mathrm{claim}},
m_{\mathrm{gate}}
\right),
$$

其中：

- $\theta$：固定 policy／preset；
- $\mathcal T_{\mathrm{rank}}$：competitor topology，例如 structural lost labels 與 scan ordering；
- $\mathcal T_{\mathrm{claim}}$：claim participant topology；
- $m_{\mathrm{gate}}$：不納入當前純 geometry metric 的 occ、appearance、Kalman state gate fields 等 meta。

這不是說 competitor geometry 不在 input；competitor geometry仍是 event state。
context 表示的是：

- 哪些 entity／branch 存在；
- 哪些 policy 固定；
- 哪些變數在當前 theorem 中被條件化而不是擾動。

---

# 3. Gap semantics

唯一 score authority：

$$
\boxed{
\Delta
:=
\mathrm{la}
=
\mathrm{age[lost]}
\in\mathbb N
}
$$

它進入：

- lost forward extrapolation；
- candidate backward extrapolation；
- speed gate time；
- directional blend gap scale。

它不等於：

$$
\mathrm{gap\_len}
=
\mathrm{la}
-
\mathrm{bridge\_at}
+
1.
$$

`gap_len` 只用於 occupancy／fidelity 路徑，不是 $G_\Delta$ 的 extrapolation factor。

GCTM 應將每個 pair 的 $\Delta_i$ 視為 pair-geometry family index：

$$
\{G_{\Delta_i}\}_{\Delta_i}.
$$

不建議在第一層把 $\Delta$ 當成普通 geometry coordinate 塞入 $x$，因為這會混淆：

- state perturbation；
- operator change；
- lost age progression。

若 GCTM 要建 gap evolution path，可以另定義：

$$
\Delta_i
\mapsto
G_{\Delta_i}(x_{C_j},x_L^{(i)}),
$$

而不是抹平 operator family 與 state path 的區別。

---

# 4. Continuous score layer

## 4.1 Native geometry map

對每個 pair：

$$
G_\Delta
:
(u_L,u_C)
\longmapsto
a,b.
$$

權威 height scale：

$$
h_{\mathrm{ref}}
=
\max
\left(
\frac{h_L^{\mathrm{ema}}+h_C^{\mathrm{ema}}}{2},
1
\right).
$$

extrapolation：

$$
p_L^\rightarrow
=
p_L+v_L\Delta,
$$

$$
p_C^\leftarrow
=
p_C-v_C\Delta.
$$

residuals：

$$
fwd_r
=
\frac{
\|p_L^\rightarrow-p_C\|_2
}{
h_{\mathrm{ref}}
},
$$

$$
bwd_r
=
\frac{
\|p_C^\leftarrow-p_L\|_2
}{
h_{\mathrm{ref}}
},
$$

$$
dist_h
=
\frac{
\|p_L-p_C\|_2
}{
h_{\mathrm{ref}}
}.
$$

speed weight：

$$
s_L
=
\frac{\|v_L\|_2}{h_{\mathrm{ref}}},
$$

$$
w
=
\sqrt{
\operatorname{clip}
\left(
\frac{s_L}{0.12},
0,
1
\right)
}.
$$

base score：

$$
\boxed{
b_0
=
w\cdot
\frac{fwd_r+bwd_r}{2}
+
(1-w)dist_h
}
$$

directional branch：

$$
b
=
(1-\alpha)b_0+\alpha b_{\mathrm{dir}}.
$$

若 directional branch inactive：

$$
b=b_0,
\qquad
\alpha=0.
$$

## 4.2 Continuous outcome

建議定義：

$$
\mathcal Z_c
\ni
\left(
\{a^{(i)}\},
\{b^{(i)}\},
b_{\mathrm{best}},
b_{\mathrm{second}},
m
\right),
$$

其中：

$$
m
=
b_{\mathrm{second}}
-
b_{\mathrm{best}}.
$$

這些 quantities 是 continuous 或 piecewise-smooth；best／second value 在 tie surface 上不可微，但仍是實值 quantity。

identity labels：

$$
i_{\mathrm{best}},
i_{\mathrm{second}}
$$

不屬於 $\mathcal Z_c$，而屬於離散 outcome。

## 4.3 Known continuous properties

既有 object 已支持：

1. 單一 residual atom 的 velocity Jacobian 帶有 $\Delta$ scaling；
2. position／static channel 不一定隨 $\Delta$ 放大；
3. directional blend、weight derivative、residual direction rotation 使完整 score gain不保證隨 gap 單調；
4. production range 內的 gap comparison 與 $\Delta\to\infty$ formal statement 必須分開；
5. natural bdist map 不支持 canonical global contraction branding。

連續層的正確名稱是：

```text
local / regional Lipschitz gain
incremental gain
```

不是：

```text
global contraction
Ricci curvature of production transition
```

---

# 5. Mask and ranking layer

## 5.1 Production order

權威次序：

```text
structural lost filter
→ M_pre:
   height
   speed(states, Δ)
   spatial(states)
→ G_Δ:
   compute pair bdist
→ M_post:
   cutoff
   occupancy
   appearance
   portable tail
→ rank:
   best / second over final-eligible set
→ margin
→ proposal
→ claim
→ commit
```

因此：

$$
\mathcal I_{\mathrm{rank}}
=
M_{\mathrm{post}}
\circ
G_\Delta
\circ
M_{\mathrm{pre}}
\left(
\mathcal I_{\mathrm{struct}}
\right).
$$

ranking 只發生在 final-eligible pairs 上。

## 5.2 Set-valued dependence

pair score：

$$
b^{(i)}
=
G_\Delta(u_L^{(i)},u_C)
$$

不直接依賴其他 lost。

但是：

$$
b_{\mathrm{best}},
b_{\mathrm{second}},
m,
i_{\mathrm{best}},
i_{\mathrm{second}}
$$

依賴整個 $\mathcal I_{\mathrm{rank}}$。

因此 existing object 有兩種不同的 continuous dependence：

1. pair-local geometry dependence；
2. event-level set dependence。

GCTM 若只預測單一 pair transition likelihood，不能自動推出 rank probability；它還需要 competitor set semantics。

---

# 6. Discrete decision layer

## 6.1 Discrete outcome

$$
\mathcal Z_d
\ni
\left(
\mathrm{eligibleMask},
i_{\mathrm{best}},
i_{\mathrm{second}},
\mathrm{cutoffPass},
\mathrm{marginPass},
\mathrm{proposalEmit},
\mathrm{claimWin},
\mathrm{commit},
\mathrm{packedClaimKey}
\right).
$$

full-event 離散 map 可概括為：

$$
F_{\mathrm{event}}^{(d)}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j
$$

但更 production-faithful 的寫法應顯式保留 masks：

$$
F_{\mathrm{event}}^{(d)}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}
\left(
D_{\mathrm{rank/margin},j}
\circ
M_{\mathrm{post},j}
\circ
G_{\Delta_i}
\circ
M_{\mathrm{pre},j}
\circ
R_j
\right).
$$

## 6.2 Claim semantics

claim key 為：

$$
\mathrm{sq}
=
\left\lfloor
\operatorname{clip}(q,0,1)
\cdot32767
\right\rfloor,
$$

$$
\mathrm{key}
=
(\mathrm{sq}\ll16)
\mid
(\mathrm{cand\ index}\ \&\ 0xffff).
$$

claim winner由 `atomicMax` 決定：

- higher quantized detection score wins；
- score tie 時 higher candidate index wins；
- claim 不使用 bdist；
- claim loser不 fallback 到 second-best lost。

因此 geometry ranking 與 claim arbitration 是兩個不同的 order：

$$
\arg\min bdist
$$

對比：

$$
\arg\max packed(q,\mathrm{index}).
$$

## 6.3 Commit semantics

commit 只做：

```cpp
track_ids[cand] = track_ids[lost];
active[lost]    = false;
```

它不更新：

- candidate motion；
- candidate foot history；
- candidate EMA；
- candidate Kalman state；
- lost motion到candidate motion的融合。

因此 existing object 的 terminal 是 identity transfer，不是 state convergence。

---

# 7. Hybrid structure

既有 operator 的核心特性是：

$$
\boxed{
\text{continuous score map}
+
\text{thresholded/set-dependent discrete decision map}
}
$$

定義 decision boundary：

$$
\mathcal B_{\mathrm{event},\mathcal C}
=
\left\{
x:
F_{\mathrm{event}}^{(d)}
\text{ 在 }x\text{ 不局部常數}
\right\}.
$$

可能出現：

$$
d_c
\left(
F_{\mathrm{event}}^{(c)}(x),
F_{\mathrm{event}}^{(c)}(y)
\right)
\ll1
$$

但：

$$
F_{\mathrm{event}}^{(d)}(x)
\neq
F_{\mathrm{event}}^{(d)}(y).
$$

這就是：

```text
score-stable / decision-unstable
```

可用兩個分離量描述：

### Continuous gain

$$
L_{\mathrm{event}}(x\mid\mathcal C)
=
\limsup_{y\to x}
\frac{
d_c(F_{\mathrm{event}}^{(c)}(x),F_{\mathrm{event}}^{(c)}(y))
}{
d_X(x,y)
}.
$$

### Geometry-conditioned decision robustness radius

$$
\rho_{\mathrm{event}}^{\mathrm{geom}}(x\mid\mathcal C)
=
\inf
\left\{
d_X(x,y):
F_{\mathrm{event}}^{(d)}(x)\neq F_{\mathrm{event}}^{(d)}(y)
\right\}.
$$

claim-score perturbation不應偷偷放進 geometry radius；它應是：

- cross-context comparative statics；或
- 獨立的 claim-key robustness quantity。

---

# 8. Determinism and stochasticity boundary

## 8.1 Production fact

對固定：

$$
(x_{\mathrm{event}},\mathcal C,\theta)
$$

production 輸出是確定的：

$$
F_{\mathrm{event}}(x_{\mathrm{event}})
=
z.
$$

因此沒有內生：

$$
P_{\mathrm{event}}(x,\cdot)
$$

可直接被解釋為 production transition law。

若硬寫：

$$
P_{\mathrm{event}}(x,\cdot)
=
\delta_{F_{\mathrm{event}}(x)},
$$

則：

$$
W_p
\left(
\delta_{F(x)},
\delta_{F(y)}
\right)
=
d_Z(F(x),F(y)),
$$

measure geometry退化為普通 output distance。

## 8.2 Legitimate stochastic lifts

GCTM 可以引入 stochasticity，但必須明確標來源。合法來源包括：

1. latent physical transition uncertainty；
2. observation uncertainty；
3. uncertainty over native reduced state；
4. context uncertainty；
5. event population law；
6. stochastic model residual。

例如：

$$
X_{\mathrm{event}}
\sim
K_{\boldsymbol{\Delta}}(\cdot\mid s_0),
$$

再由 existing operator產生：

$$
Z
=
F_{\mathrm{event}}(X_{\mathrm{event}}).
$$

此時 distribution：

$$
F_{\mathrm{event}\#}
K_{\boldsymbol{\Delta}}(\cdot\mid s_0)
$$

是 GCTM／population object，不是 production bridge 自帶的 transition kernel。

---

# 9. GCTM alignment contract

## 9.1 GCTM must declare its primary object

GCTM 必須明確選擇自己研究的是哪一層：

### A. Latent state transition

$$
K_\Delta
:
s_0
\mapsto
\mathcal P(\mathcal S_\Delta).
$$

回答 physical／latent state 如何隨 gap 演化。

### B. Native-state uncertainty

$$
Q_{\boldsymbol{\Delta}}
:
s_0
\mapsto
\mathcal P(\mathcal X_{\mathrm{event}}).
$$

回答 GCTM 如何產生 online-native state 或其不確定性。

### C. Score distribution

$$
(F_{\mathrm{event}}^{(c)})_\#Q_{\boldsymbol{\Delta}}.
$$

回答 GCTM uncertainty 經 online score map 後形成什麼 score distribution。

### D. Decision probability

$$
\Pr_{X\sim Q_{\boldsymbol{\Delta}}}
\left[
F_{\mathrm{event}}^{(d)}(X)=z_d
\right].
$$

回答相同 latent／native uncertainty 下，離散 online decision 的機率。

這四個 object 不應混成單一「transition likelihood」。

## 9.2 Required state correspondence

GCTM 必須提供 mapping：

$$
\phi:
\mathcal S_{\mathrm{GCTM}}
\longrightarrow
\mathcal X_{\mathrm{native}}
$$

或：

$$
\Phi:
\mathcal S_{\mathrm{GCTM}}
\longrightarrow
\mathcal P(\mathcal X_{\mathrm{native}}).
$$

至少對應：

- lost exit anchor；
- candidate entry anchor；
- native velocity semantics；
- EMA height semantics；
- gap $\Delta$；
- structural competitor set；
- final eligibility生成條件；
- claim participants與detection-score key；
- optional gate meta。

若無法對應，GCTM conclusion 只能停在 abstract／offline level，不能宣稱對應 existing online decision。

## 9.3 Required output correspondence

GCTM 應分開對應：

### Continuous

- pair bdist；
- best／second values；
- margin；
- score atoms或其 sufficient summary。

### Discrete

- eligibility；
- rank winner；
- margin pass；
- proposal；
- claim win；
- commit。

不能只對應 final commit probability，卻不說明中間 score／rank／claim semantics。

## 9.4 Context alignment

GCTM 必須區分：

$$
\mathcal I:
\text{lost competition}
$$

與：

$$
\mathcal J:
\text{claim competition}.
$$

若 GCTM 只建 pairwise transition probability：

$$
p(i\leftrightarrow C),
$$

仍不足以決定：

- best／second；
- margin；
- claim winner；
- commit。

它還需要 event-level aggregation／competition operator。

## 9.5 Gap alignment

GCTM 的 gap 必須說明是否等於：

$$
\Delta=\mathrm{age[lost]}.
$$

若使用其他時間量，例如：

- missing-frame count；
- elapsed physical time；
- fire-window displacement；
- `gap_len`；
- normalized time；

必須提供明確映射，不能直接與 online $\Delta$ 混用。

---

# 10. Compatibility constraints

## C1 — Preserve the production-operator boundary

GCTM 不應重新描述、或悄悄取代 existing
$F_{\mathrm{event}}$／production baseline operator。

正確關係應是：

$$
\text{GCTM state／law}
\longrightarrow
\text{native event state／law}
\longrightarrow
F_{\mathrm{event}}
\longrightarrow
\text{online score／decision}.
$$

未來若有 frozen L2 contract，可**顯式提出** named pair-score 或 ranking
subcomponent replacement intervention；這不等於取代整個
$F_{\mathrm{event}}$。其餘 gate、claim arbitration、fallback、commit
stage 必須保留或另行 re-charter。

## C2 — Do not invent intrinsic runtime randomness

不得因為 GCTM 是 probabilistic model，就宣稱 production bridge 本身是 stochastic kernel。

隨機性來源必須獨立聲明。

## C3 — Preserve continuous／discrete separation

任何 GCTM likelihood 都必須說明它對應：

- continuous score uncertainty；
- discrete decision uncertainty；
- 或兩者的 composition。

不得用單一 scalar likelihood 同時取代 bdist gain 與 decision boundary。

## C4 — Preserve pair／event distinction

pair-level transition quality不等於 event-level winner probability。

event-level decision還依賴：

- competitor set；
- final eligibility；
- best／second margin；
- claim topology；
- detection-score keys。

## C5 — Preserve geometry／claim separation

geometry score order：

$$
\arg\min bdist
$$

不等於 claim order：

$$
\arg\max packed(q,\mathrm{index}).
$$

GCTM 不應把 claim loss歸因為 motion transition mismatch，除非模型顯式包含 claim context。

## C6 — No canonical global contraction claim

existing online object 的 terminal 是：

```text
NO_CANONICAL_GLOBAL_CONTRACTION
```

GCTM 可以研究自己定義的 stochastic process 是否 contraction，但不得把該結論倒灌成 native bridge deterministic map 的既有性質。

## C7 — Population analysis changes the object

若研究：

$$
(F_{\mathrm{event}})_\#\mu,
$$

必須明確標示研究物件已變成：

$$
(F_{\mathrm{event}},\mu).
$$

這是 population-level property，不是 $F_{\mathrm{event}}$ 單獨的 operator property。

## C8 — Runtime-native reduction is authoritative

不得以 offline window mean、trajectory endpoint或 Kalman velocity 冒充 native reduction $R$。

既有 D0 結果已表明：

```text
same score function + different reduction
≠
runtime-faithful score object
```

GCTM 若無 native correspondence，必須降級其 online claim。

---

# 11. Prohibited interpretations

GCTM 文件中應禁止以下敘述，除非另有明確新 object：

1. 「online bridge 是 Markov kernel $P_\Delta$」；
2. 「commit 表示 lost motion converges to candidate motion」；
3. 「Wasserstein／Ricci 描述 production bridge 的內生 contraction」；
4. 「pair transition probability直接等於 final relink probability」；
5. 「增加 competitor一定降低 robustness radius」；
6. 「gap 越大，完整 score gain 必然單調增加」；
7. 「bdist 最佳者一定是 claim／commit winner」；
8. 「offline reconstructed state 等同 runtime native state」；
9. 「final commit 是唯一需要建模的 outcome」；
10. 「GCTM 可忽略 gate、rank、margin與claim仍宣稱對應 online」。

---

# 12. What GCTM may add

existing object 已固定 online operational semantics，但仍留下 GCTM 的合理空間。

## 12.1 Latent transition semantics

GCTM 可以回答：

> lost exit 與 candidate entry 是否由同一 latent physical trajectory生成？

這是 existing $F$ 不回答的問題。

## 12.2 Gap-conditioned uncertainty

GCTM 可以建立：

$$
K_\Delta
$$

描述 gap 增加時：

- position uncertainty；
- velocity uncertainty；
- shape／height uncertainty；
- observation uncertainty；

如何演化。

## 12.3 Likelihood calibration

GCTM 可以把 native residual：

$$
r_\Delta
$$

轉成具有明確概率語義的：

$$
p(r_\Delta\mid\Delta,s,\mathcal C).
$$

但它必須說明：

- 這是替代 bdist；
- 校準 bdist；
- 還是只作額外 feature。

## 12.4 Boundary-aware decision semantics

GCTM 可以研究：

$$
\Pr[
F_{\mathrm{event}}^{(d)}(X)=z_d
]
$$

或：

$$
\Pr[
\rho_{\mathrm{event}}^{\mathrm{geom}}(X)<\varepsilon
].
$$

這能把 continuous uncertainty 與 discrete boundary連起來，而不抹平兩層。

## 12.5 Attribution separation

GCTM 可幫助區分：

- pair geometry model failure；
- competitor-induced rank loss；
- gate rejection；
- margin ambiguity；
- claim loss；
- commit consequence。

這比單一 relink success probability更符合 existing operator。

---

# 13. Recommended GCTM document insertion

建議在 GCTM 主文加入一節：

```text
Existing online operator and compatibility boundary
```

內容至少包含：

## 13.1 Online authority

$$
P_j:
\left(
x_{C_j},q_j,
\{(x_L^{(i)},\Delta_i)\}_{i\in\mathcal I_{\mathrm{struct},j}},
m_{\mathrm{gate}},\theta
\right)
\to
\left(z_{c,j},\operatorname{proposal}_j\right),
$$

$$
F_{\mathrm{event}}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j
:
\mathcal X_{\mathrm{event}}
\to
\mathcal Z_c\times\mathcal Z_d.
$$

## 13.2 Runtime composition

$$
\left\{
R_j
\to
M_{\mathrm{pre},j}
\to
G_{\Delta_i}
\to
M_{\mathrm{post},j}
\to
S_{\mathrm{rank},j}
\to
\operatorname{proposal}_j
\right\}_{j\in\mathcal U_{\mathrm{event}}}
\to
A_{\mathrm{claim}}
\to
\mathrm{commit}.
$$

## 13.3 GCTM correspondence obligations

- latent-to-native mapping；
- gap mapping；
- geometry／claim context mapping；
- continuous／discrete output mapping；
- stochasticity source；
- runtime-fidelity boundary。

## 13.4 Negative constraints

- no intrinsic runtime kernel claim；
- no motion convergence interpretation；
- no global contraction branding；
- no pair-to-final-decision shortcut；
- no offline proxy substitution without fidelity proof。

---

# 14. Alignment checklist

GCTM 對齊時逐項回答：

| ID | Question | Required answer |
|---|---|---|
| O1 | GCTM primary state 是什麼？ | explicit space |
| O2 | 它如何映射到 native $R$ inputs？ | $\phi$ or $\Phi$ |
| O3 | GCTM gap 與 `age[lost]` 的關係？ | exact mapping |
| O4 | stochasticity 來自哪裡？ | explicit source |
| O5 | pair probability 如何進 competitor ranking？ | aggregation rule |
| O6 | 如何表示 $\mathcal I_{\mathrm{struct/pre/rank}}$？ | mask semantics |
| O7 | 如何表示 claim set $\mathcal J$ 與 $q_j$？ | separate claim model |
| O8 | output 對應 score 還是 decision？ | layer declaration |
| O9 | 是否保留 best／second／margin？ | yes or justified omission |
| O10 | 是否建模 gate／cutoff？ | explicit boundary |
| O11 | 是否區分 rank winner 與 claim winner？ | required |
| O12 | commit probability如何由前層得到？ | explicit composition |
| O13 | 使用 offline state 時如何證明 runtime fidelity？ | evidence requirement |
| O14 | population law $\mu$ 是否成為 object 一部分？ | explicit declaration |
| O15 | GCTM claim 是否超出 online guarantee envelope？ | boundary statement |

若其中 O2、O3、O5、O7、O8、O11 無明確答案，GCTM 尚未真正對齊 existing online object。

---

# 15. Minimal formal interface proposal

GCTM 與 online 之間可用以下最小介面：

## Candidate-local proposal interface

令 $\mathcal U_{\mathrm{event}}$ 為 bridge event 的完整 candidate universe。
對每個 candidate $j$，先以其 own state 與 structural lost competitors
形成 proposal（或 no proposal）：

$$
P_j:
\left(
\Delta_j,
x_{C_j},
\{x_L^{(i)}\}_{i\in\mathcal I_{\mathrm{struct},j}},
m_{\mathrm{gate}},
\theta
\right)
\longrightarrow
\operatorname{proposal}_j.
$$

## Claim-arbitration interface

對一個 lost $L$，actual claim set 是 candidate-local proposal 之後的中間產物：

$$
\mathcal J_L
=
\{j\in\mathcal U_{\mathrm{event}}:
\operatorname{proposal}_j=L\}.
$$

它不是 predeclared NativeEvent input。claim arbitration 再消費完整 universe 的
proposal、detection/track score、candidate identity：

$$
A_{\mathrm{claim}}:
\left\{
\left(\operatorname{proposal}_j,q_j,j\right)
\right\}_{j\in\mathcal U_{\mathrm{event}}}
\longrightarrow
\text{claim/commit},
$$

$$
F_{\mathrm{event}}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j.
$$

因此 singular $x_C$ 與 $\{q_j\}_{j\in\mathcal J_L}$ 都不足以單獨推導
cross-candidate claim／commit；GCTM 若不建模此 composition，必須明示
claim/commit unchanged 或 audit-only。

## GCTM output interface

至少選一種：

### State law

$$
Q_{\boldsymbol{\Delta}}
\in
\mathcal P(\mathcal X_{\mathrm{event}}).
$$

### Pair likelihoods

$$
\ell_i
=
\log p
\left(
x_{C_j},x_L^{(i)}
\mid
\Delta_i
\right).
$$

### Calibrated native score

$$
\tilde b_i
=
T_{\mathrm{GCTM}}
\left(
b_i,
\ell_i,
\Delta_i
\right).
$$

### Decision distribution

$$
\pi(z_d)
=
\Pr_{X\sim Q_{\boldsymbol{\Delta}}}
\left[
F_{\mathrm{event}}^{(d)}(X)=z_d
\right].
$$

## Required composition declaration

GCTM 必須選擇並寫明：

```text
augment
calibrate
replace pair score
replace ranking
shadow-only diagnostic
```

`replace` 只可表示 frozen L2 contract 下的 named subcomponent intervention，
不可表示重寫或默默取代 $F_{\mathrm{event}}$。不能只說「GCTM 與 online
對接」，卻不指出插入位置。

---

# 16. Final position

既有 online object 可以壓成：

$$
\boxed{
F_{\mathrm{event}}
=
A_{\mathrm{claim}}
\circ
\prod_{j\in\mathcal U_{\mathrm{event}}}P_j
=
\text{deterministic full-event hybrid operator}
}
$$

其內部：

$$
\boxed{
\text{native reduction}
\to
\text{gap-conditioned pair geometry}
\to
\text{eligibility masks}
\to
\text{set-dependent ranking}
\to
\text{claim arbitration}
\to
\text{identity commit}
}
$$

它已經具有自己的 operational math，並且已被 runtime fidelity audit 支持。

GCTM 的正確角色不是重新定義 existing online object，而是補充它沒有提供的：

- latent transition semantics；
- uncertainty law；
- calibrated likelihood；
- population／boundary probability；
- attribution structure。

最重要的對齊原則是：

$$
\boxed{
\text{GCTM 可增加 probabilistic semantics，
但必須落回 existing continuous-score／discrete-decision operator。}
}
$$

若無法落回：

$$
\left\{
R_j
\to
M_{\mathrm{pre},j}
\to
G_{\Delta_i}
\to
M_{\mathrm{post},j}
\to
S_{\mathrm{rank},j}
\to
\operatorname{proposal}_j
\right\}_{j\in\mathcal U_{\mathrm{event}}}
\to
A_{\mathrm{claim}}
\to
\mathrm{commit},
$$

則 GCTM 目前只是一個可能適用 online 的上層模型，尚未形成對 production online object 的實際對應。
