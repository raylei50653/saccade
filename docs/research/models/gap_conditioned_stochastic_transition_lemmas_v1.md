<!-- doc-status: draft -->
<!-- doc-promotion: none; D2 proof appendix for the D1 canonical model spec; WP-A3 proves internal properties of the frozen D1 §4 kernel only -->
<!-- doc-date: 2026-07-22 -->
<!-- doc-module: semantic -->

# Gap-conditioned stochastic transition model — lemma & proof appendix (D2, v1)

**用途：** GCTM 的 lemma and proof appendix（charter 所定義的 D2）。本檔**只**證明
frozen D1 §4 transition kernel 的**內部數學性質**，不引入任何 D1 §4 以外的物件、
不修改 frozen interface、不選 terminal。

**由 WP-A3 填入的 lemma：** `Q_Δ ⪰ 0`（PSD，L1）、`M2 →_{γ→0} M1` mean + covariance
nesting（L2）、transition semigroup 與 covariance composition（L3）、short-/long-gap
asymptotics（L4）。

**由 WP-A5 填入（本次增量）：** §7 的 `q`/NLL ranking-equivalence lemma（L5）及其
corollary／tightness boundary（L5.1／L5.2），建立在 frozen D1 §5 的 `S_Δ` 與 D1 §6 的
claim 定義之上（obligation 2 的 proof 面）。L1–L4（§2–§5）除一處 owner-flagged 文字
更正外**不變**；本檔仍**不選** terminal。

**Authority 邊界：**

- Model authority：frozen
  [D1 §4](gap_conditioned_stochastic_transition_spec_v1.md)（canonical state、M2
  SDE、\(A_\Delta\)、\(d_\Delta(c)\)、\(Q_\Delta\)、domains、units）。本檔的所有符號
  **沿用** D1 §4，**不重新定義**。
- Task/lifecycle owner：[GCTM task charter](../threads/gap_conditioned_stochastic_transition_model_task.md)
  （Issue [#175](https://github.com/raylei50653/saccade/issues/175)）；D2 是 charter
  「Future deliverables」列的 lemma/proof appendix，由 WP-A3 owning packet 建立。
- 本檔**不**做 bridge-runtime claim、不建立 fidelity edge、不選 model 參數、不解除
  charter obligations 2/3、不觸及 WP-A4/A5 物件（\(P_0,R_1,S_\Delta,C\)），也**不**
  授權 data、fitting、runtime、online、production。

## §1 Scope, boundary, and imported notation

### §1.1 本檔做什麼／不做什麼（typed boundary）

**做（WP-A3 acceptance）：** 證明 frozen D1 §4 kernel
\(K_\Delta(z_0,c)=\mathcal N\!\big(A_\Delta z_0+d_\Delta(c),\,Q_\Delta\big)\) 的四個
內部性質——L1（PSD）、L2（\(\gamma\to0\) nesting）、L3（semigroup / Chapman–Kolmogorov
covariance composition）、L4（short-/long-gap asymptotics）。

**不做：**

- **不**修改 frozen D1 §2–§4 任何定義或介面（本檔為 append-only 的獨立 appendix；
  D1 保持 byte-frozen）。
- **不**引入 \(P_0\)/\(R_1\)/\(S_\Delta\)/cross-covariance \(C\)（WP-A4，obligation
  3）；**不**證明 \(q\)/NLL ranking equivalence（依賴 \(S_\Delta\)，見 §7）。
- **不**做 calibration-vs-ranking claim（WP-A5，obligation 2）。
- **不**選 terminal。L1/L3 的 well-posedness 內容**支持**terminal 1/2 不因這些性質
  被觸發，但 terminal 選擇只在 terminal review 依 charter frozen decision procedure
  進行，本檔不代行。
- **不**碰 reverse-time / candidate-backward atom（D1 §4 boundary 3）、data、
  fitting、runtime、online、production。

### §1.2 Imported notation（沿用 frozen D1 §4，不重定義）

以下全部引自 frozen D1 §4，僅為本檔可讀性重述，**不**構成再定義或修改：

- State \(z=[x;v]\in\mathbb R^{2d}\)，coordinate dim \(d\in\mathbb N_{\ge1}\)
  （D1 §4.1）。
- Generator 與 diffusion（D1 §4.2）：
  \[
  F=\begin{bmatrix}0&I\\0&-\gamma I\end{bmatrix},\quad
  \Sigma=BB^\top=\begin{bmatrix}0&0\\0&D\end{bmatrix},\quad
  D=LL^\top\in\mathbb R^{d\times d},\ D\succeq0,\quad \gamma\ge0.
  \]
- Affine transition（D1 §4.3），\(b=e^{-\gamma\Delta}\)、\(a=(1-b)/\gamma\)：
  \[
  A_\Delta=e^{F\Delta}=\begin{bmatrix}I&aI\\0&bI\end{bmatrix},\qquad
  d_\Delta(c)=\begin{bmatrix}(\Delta-a)\bar v(c)\\(1-b)\bar v(c)\end{bmatrix}.
  \]
- Process covariance（D1 §4.5），noise-integral 與 block-scalar 形：
  \[
  Q_\Delta=\int_0^\Delta e^{F\tau}\,\Sigma\,e^{F^\top\tau}\,\mathrm d\tau
  =\begin{bmatrix}q_{xx}D&q_{xv}D\\ q_{xv}D&q_{vv}D\end{bmatrix},
  \]
  其中（closed form，\(\gamma>0\)）
  \[
  q_{vv}=\frac{1-b^2}{2\gamma},\quad
  q_{xv}=\frac{(1-b)^2}{2\gamma^2},\quad
  q_{xx}=\frac{2\gamma\Delta-3+4b-b^2}{2\gamma^3}.
  \]
- Response kernels（D1 §4.5）\(g(\tau)=\dfrac{1-e^{-\gamma\tau}}{\gamma}\)（position）、
  \(h(\tau)=e^{-\gamma\tau}\)（velocity）；\(e^{F\tau}\) 第二 block-column 為
  \([\,g(\tau)I;\,h(\tau)I\,]\)。故 scalar coefficients 為 kernel Gram 積：
  \[
  q_{xx}=\int_0^\Delta g^2,\quad q_{xv}=\int_0^\Delta gh,\quad
  q_{vv}=\int_0^\Delta h^2 .
  \]
- **Scalar 2×2 Gram matrix**（本檔工作量，源自上式）：
  \[
  M_\Delta:=\begin{bmatrix}q_{xx}&q_{xv}\\ q_{xv}&q_{vv}\end{bmatrix}
  =\int_0^\Delta\begin{bmatrix}g(\tau)\\ h(\tau)\end{bmatrix}
  \begin{bmatrix}g(\tau)&h(\tau)\end{bmatrix}\mathrm d\tau .
  \]
- \(\Delta\) domain（D1 §4.3/§4.7）：bridge evaluation \(\Delta\in\mathbb N_{\ge1}\)；
  analytic family \(\Delta\in\mathbb R_{\ge0}\)。\(\gamma=0\) 為 M1 boundary
  （D1 §4.6），closed-form 以 removable-singularity 極限值定義。

**型別約定（本檔全程遵守）：** \(q_{xx},q_{xv},q_{vv}\) 是 **scalar**；covariance
**block** 是 \(q_{\bullet\bullet}D\)（\(d\times d\)）。極限與 asymptotics 對 scalar 與
block 分開陳述，兩者不混寫。

### §1.3 WP-A5 增量與 scope 更新（本次）

§1.1 的「做／不做」是 **WP-A3 acceptance** 的 scope 快照。本次 **WP-A5** 增量在
**§7** 填入 `q`/NLL ranking-equivalence lemma（L5），其依賴的 \(S_\Delta\) 已由 frozen
D1 §5（WP-A4）定義、claim-space 已由 frozen D1 §6（WP-A5）定義，故 §1.1「不證明
`q`/NLL ranking equivalence（見 §7）」一項對 WP-A3 仍為真、但**現已由 WP-A5 於 §7 完成**。
本增量：

- **只**引用 frozen D1 §5 的 \(S_\Delta\)（canonical \(C=0\)、\(S_\Delta\succ0\) regime）
  與 D1 §6 的 score 定義（\(q,\,\log\det S_\Delta,\,E\)），**不重定義**，也**不**引入
  D1 §4 以外的新 transition 物件。
- L1–L4（§2–§5）數學內容**byte 不變**，唯一例外是一處 owner-flagged 文字更正
  （§5.2「有限矩陣」→「有限向量」，`d_Δ` 為 vector \(\in\mathbb R^{2d}\)；#253 review
  non-blocking nit，owner 指示於下次 touch D2 時受 review 修正）。
- **不**選 terminal（terminal review 為後續 packet）；**不**做 calibration/ranking 的
  實際量測、gain 數值、metric／threshold 選擇（那需 data/B1/O1 授權，非本檔）。

## §2 Lemma L1 — \(Q_\Delta\succeq0\)（PSD），含退化 rank 刻畫

> **Lemma L1（PSD 與 rank）.** 對所有 \(\gamma\ge0\) 與所有 \(\Delta\ge0\)，
> \[
> Q_\Delta\succeq0 .
> \]
> 進一步，對 **\(\Delta>0\)** 有 \(M_\Delta\succ0\)，且
> \[
> \operatorname{rank}Q_\Delta=2\operatorname{rank}D,\qquad
> Q_\Delta\ \text{singular}\iff D\ \text{singular}.
> \]
> **\(\Delta=0\)** 是分開的退化邊界：\(M_0=0\)，故 \(Q_0=0\)（與 \(D\) 無關，即使
> \(D\succ0\) 亦然）。

**Proof (a) — structural integral argument（只需 \(\Sigma\succeq0\)）.**
對任意 \(\tau\)，\(e^{F\tau}\Sigma e^{F^\top\tau}=(e^{F\tau}B)(e^{F\tau}B)^\top\succeq0\)
（\(\Sigma=BB^\top\)）。PSD 錐對非負線性組合與（Riemann/Bochner）積分封閉，故
\(Q_\Delta=\int_0^\Delta e^{F\tau}\Sigma e^{F^\top\tau}\,\mathrm d\tau\succeq0\)。此論證
對任意 \(\Delta\ge0\)、\(\gamma\ge0\) 成立，且只用到 \(\Sigma\succeq0\)（等價
\(D\succeq0\)）。∎(a)

**Proof (b) — Kronecker / Gram 結構（給出 rank 刻畫）.**
由 block-scalar 形，
\[
Q_\Delta=\begin{bmatrix}q_{xx}D&q_{xv}D\\ q_{xv}D&q_{vv}D\end{bmatrix}
= M_\Delta\otimes D ,
\]
即 \(Q_\Delta\) 是 \(2\times2\) 的 \(M_\Delta\) 與 \(d\times d\) 的 \(D\) 的 Kronecker
積（block \((i,j)\) 為 \((M_\Delta)_{ij}D\)）。Kronecker 積的 eigenvalue 為兩因子
eigenvalue 的兩兩乘積：若 \(M_\Delta\) 有 eigenvalues \(\{\mu_1,\mu_2\}\)、\(D\) 有
\(\{\lambda_1,\dots,\lambda_d\}\)，則 \(Q_\Delta\) 的 eigenvalues 為
\(\{\mu_i\lambda_j\}\)。\(M_\Delta\succeq0\)（下述）且 \(D\succeq0\)\(\Rightarrow\)
所有 \(\mu_i\lambda_j\ge0\)\(\Rightarrow Q_\Delta\succeq0\)，與 (a) 一致。

\(M_\Delta\succeq0\)：由 §1.2，\(M_\Delta=\int_0^\Delta \phi(\tau)\phi(\tau)^\top\,
\mathrm d\tau\)（\(\phi=[g;h]\)）是 outer-product 的積分，故 \(\succeq0\)；等價地，
\(M_\Delta\) 是函數對 \(\{g,h\}\) 在 \(L^2[0,\Delta]\) 的 Gram matrix。∎(b, PSD)

**\(\Delta>0\)：\(M_\Delta\succ0\)（\(\{g,h\}\) linearly independent on \([0,\Delta]\)）.**
Gram matrix PD \(\iff\) 生成函數線性獨立，故只需證 \(\{g,h\}\) 在任意 \([0,\Delta]\)
（\(\Delta>0\)）線性獨立。分兩情形：

- **\(\gamma>0\)：** 由 \(g(\tau)=\dfrac{1-e^{-\gamma\tau}}{\gamma}=\dfrac1\gamma-\dfrac1\gamma
  h(\tau)\)。設 \(\alpha g+\beta h\equiv0\) on \([0,\Delta]\)，代入得
  \(\dfrac\alpha\gamma+\big(\beta-\dfrac\alpha\gamma\big)h(\tau)\equiv0\)。因
  \(h(\tau)=e^{-\gamma\tau}\) 在 \([0,\Delta]\) 非常數（\(\gamma>0,\Delta>0\)），常數項與
  \(h\)-係數須各自為零：\(\alpha/\gamma=0\Rightarrow\alpha=0\)，再由 \(\beta h\equiv0\)、
  \(h\not\equiv0\) 得 \(\beta=0\)。故線性獨立。
- **\(\gamma=0\)（M1 boundary，D1 §4.6 極限）：** \(g(\tau)=\tau\)、\(h(\tau)=1\)；
  \(\alpha\tau+\beta\equiv0\) on \([0,\Delta]\)（\(\Delta>0\)）迫使 \(\alpha=\beta=0\)。

兩情形皆線性獨立，故 \(\Delta>0\Rightarrow M_\Delta\succ0\)。此時 \(M_\Delta\) 的
eigenvalues \(\mu_1,\mu_2>0\)，故 \(Q_\Delta=M_\Delta\otimes D\) 的非零 eigenvalues
恰對應 \(D\) 的非零 eigenvalues（每個重複 2 次）：
\(\operatorname{rank}Q_\Delta=2\operatorname{rank}D\)，且
\(Q_\Delta\ \text{singular}\iff D\ \text{singular}\)。∎(rank)

**\(\Delta=0\) 退化邊界.** \(M_0=\int_0^0(\cdot)=0\)，故 \(Q_0=M_0\otimes D=0\)，與 D1
§4.3 的 \(Q_0=0\) 一致；此處 \(Q_0\) singular 與 \(D\) 無關（\(M_0\) 已退化），故
上段 rank 等式僅對 \(\Delta>0\) 成立。∎

**Remarks.**

1. L1 對 D1 §4.5 的「\(Q_\Delta\) 一般可能 singular（degenerate Gaussian）」給出**精確**
   來源：對 \(\Delta>0\)，退化**唯一**來自 \(D\)（noise diffusion）的 rank 不足，
   而非 gap-index 結構；gap 結構（\(M_\Delta\)）在 \(\Delta>0\) 恆 nondegenerate。
2. 全程只用 \(D\succeq0\)（frozen domain），不需 \(D\succ0\)，與 D1 §4.5 的 degenerate
   Gaussian well-posed 定義一致，不涉任何 \(Q_\Delta^{-1}\)。

## §3 Lemma L2 — \(M2\xrightarrow{\gamma\to0}M1\) nesting（mean + covariance）與 \(\gamma\) 連續性

> **Lemma L2.** 固定 \(\Delta\ge0,\ D,\ \bar v(c)\)。當 \(\gamma\to0^+\)：
> \[
> A_\Delta\to\Phi_{M1}(\Delta)=\begin{bmatrix}I&\Delta I\\0&I\end{bmatrix},\qquad
> d_\Delta(c)\to0,
> \]
> \[
> (q_{xx},q_{xv},q_{vv})\to\Big(\tfrac{\Delta^3}{3},\ \tfrac{\Delta^2}{2},\ \Delta\Big)
> \ \Longrightarrow\
> Q_\Delta\to Q_{M1}(\Delta)=\begin{bmatrix}\tfrac{\Delta^3}{3}D&\tfrac{\Delta^2}{2}D\\
> \tfrac{\Delta^2}{2}D&\Delta D\end{bmatrix}.
> \]
> 因此 M2 transition 於 \(\gamma\to0\) 在 **mean 與 covariance 兩者**上 nest 回 M1
> constant-velocity / white-acceleration family。又 \(\gamma\mapsto(A_\Delta,d_\Delta,Q_\Delta)\)
> 在 \([0,\infty)\) **連續**（\(\gamma=0\) 為 removable singularity），故 D1 §4.6 的
> \(\gamma=0\) 定義閉合是此連續延拓的**唯一**取值。

**Proof（mean）.** \(b=e^{-\gamma\Delta}=1-\gamma\Delta+\tfrac{(\gamma\Delta)^2}{2}
-\tfrac{(\gamma\Delta)^3}{6}+O(\gamma^4)\)。故
\[
a=\frac{1-b}{\gamma}=\Delta-\frac{\gamma\Delta^2}{2}+O(\gamma^2)\xrightarrow{\gamma\to0}\Delta,
\qquad b\to1,
\]
給出 \(A_\Delta\to\Phi_{M1}\)。而 \(d_\Delta(c)=[(\Delta-a)\bar v;(1-b)\bar v]\)，其中
\(\Delta-a=\tfrac{\gamma\Delta^2}{2}+O(\gamma^2)\to0\)、\(1-b=\gamma\Delta+O(\gamma^2)\to0\)，
故 \(d_\Delta(c)\to0\)（與 D1 §4.3/§4.6 一致）。∎(mean)

**Proof（covariance；removable singularity）.** 分項 Taylor：
\[
q_{vv}=\frac{1-b^2}{2\gamma}=\frac{2\gamma\Delta-2\gamma^2\Delta^2+O(\gamma^3)}{2\gamma}
=\Delta-\gamma\Delta^2+O(\gamma^2)\to\Delta,
\]
\[
q_{xv}=\frac{(1-b)^2}{2\gamma^2}
=\frac{(\gamma\Delta-\tfrac{\gamma^2\Delta^2}{2}+O(\gamma^3))^2}{2\gamma^2}
=\frac{\Delta^2}{2}-\frac{\gamma\Delta^3}{2}+O(\gamma^2)\to\frac{\Delta^2}{2}.
\]
\(q_{xx}\) 有 \(1/\gamma^3\) 極點，須把分子展到 \(O(\gamma^3)\) 才見抵消。以
\(b=1-\gamma\Delta+\tfrac{\gamma^2\Delta^2}{2}-\tfrac{\gamma^3\Delta^3}{6}+O(\gamma^4)\)、
\(b^2=e^{-2\gamma\Delta}=1-2\gamma\Delta+2\gamma^2\Delta^2-\tfrac{4\gamma^3\Delta^3}{3}+O(\gamma^4)\)：
\[
\begin{aligned}
2\gamma\Delta-3+4b-b^2
&=2\gamma\Delta-3+\big(4-4\gamma\Delta+2\gamma^2\Delta^2-\tfrac{2\gamma^3\Delta^3}{3}\big)\\
&\quad-\big(1-2\gamma\Delta+2\gamma^2\Delta^2-\tfrac{4\gamma^3\Delta^3}{3}\big)+O(\gamma^4)\\
&=\underbrace{(2-4+2)}_{0}\gamma\Delta+\underbrace{(-3+4-1)}_{0}
+\underbrace{(2-2)}_{0}\gamma^2\Delta^2
+\Big(-\tfrac23+\tfrac43\Big)\gamma^3\Delta^3+O(\gamma^4)\\
&=\frac{2}{3}\gamma^3\Delta^3+O(\gamma^4).
\end{aligned}
\]
\(0\) 階、\(\gamma\)、\(\gamma^2\) 項全數抵消，pole 為 removable：
\[
q_{xx}=\frac{\tfrac23\gamma^3\Delta^3+O(\gamma^4)}{2\gamma^3}
=\frac{\Delta^3}{3}+O(\gamma)\to\frac{\Delta^3}{3}.
\]
三個 scalar 極限即 \(Q_{M1}(\Delta)\) 的 blocks（各 \(\times D\)），故
\(Q_\Delta\to Q_{M1}(\Delta)\)。∎(covariance)

**Proof（\(\gamma\) 連續性 / 唯一延拓）.** \(a(\gamma),q_{xx}(\gamma),q_{xv}(\gamma),
q_{vv}(\gamma)\) 皆為 \(\gamma>0\) 上的解析函數且在 \(\gamma\to0^+\) 有有限極限（上證），
故各有唯一連續延拓到 \([0,\infty)\)，其 \(\gamma=0\) 值即上述極限。\(A_\Delta,d_\Delta,
Q_\Delta\) 為這些連續 scalar 的矩陣裝配，故亦連續。這把 D1 §4.6 從「介面定義閉合」
升格為 proven fact：\(\gamma=0\) 取值是**唯一**使 \(K_\Delta\) 在 \(\gamma\ge0\) total
且連續的選擇。∎

**Remark.** L2 只斷言 \(\gamma\to0\) 的 nesting；它**不**宣稱 exact-CV null 於
\(\Delta=g_{\mathrm{phys}}\) 的 zero-innovation（那是 D1 §2 row 9(i)/§4.4 的既有陳述），
也**不**涉 operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\)（D1 §4.4，operator
layer，非本檔對象）。

## §4 Lemma L3 — transition semigroup 與 covariance composition（Chapman–Kolmogorov）

**Composition convention（全程固定）.** 由 anchor 起，先走長度 \(s\) 的 interval、
再走長度 \(t\)，總長 \(s+t\)（\(s,t\ge0\)）。以 left-multiply 表示第二段作用於第一段
之上：\(z_s=A_s z_0+d_s+\eta_s\)，\(z_{s+t}=A_t z_s+d_t+\eta_t'\)，其中 \(\eta_s\sim
\mathcal N(0,Q_s)\)、\(\eta_t'\sim\mathcal N(0,Q_t)\)，且因 Brownian increments 於不相交
區間獨立、皆 \(\perp z_0,\bar v(c)\)（D1 §4.7 causal assumptions），\(\eta_s\perp\eta_t'\)。

> **Lemma L3.** 固定 \(\gamma,D,\bar v(c)\)。family \(\{(A_\Delta,d_\Delta,Q_\Delta)\}_{\Delta\ge0}\)
> 為 time-homogeneous 的一致 Markov transition semigroup：對所有 \(s,t\ge0\)，
> \[
> A_{s+t}=A_tA_s,\qquad
> d_{s+t}=A_t\,d_s+d_t,\qquad
> Q_{s+t}=A_t\,Q_s\,A_t^\top+Q_t,
> \]
> 等價地 kernel 滿足 Chapman–Kolmogorov：
> \[
> K_{s+t}(z_0,\cdot)=\int_{\mathbb R^{2d}}K_t(z_1,\cdot)\,K_s(z_0,\mathrm dz_1).
> \]

**Proof（deterministic semigroup \(A_{s+t}=A_tA_s\)）.** 由 \(A_\Delta=e^{F\Delta}\)
與 one-parameter matrix-exponential semigroup，\(e^{F(s+t)}=e^{Ft}e^{Fs}\)。直接 block
驗證（記 \(b_r=e^{-\gamma r},\ a_r=(1-b_r)/\gamma\)）：
\[
A_tA_s=\begin{bmatrix}I&a_tI\\0&b_tI\end{bmatrix}
\begin{bmatrix}I&a_sI\\0&b_sI\end{bmatrix}
=\begin{bmatrix}I&(a_s+a_tb_s)I\\0&b_tb_sI\end{bmatrix}.
\]
\(b_tb_s=e^{-\gamma(s+t)}=b_{s+t}\)；且
\(a_s+a_tb_s=\dfrac{1-b_s}{\gamma}+\dfrac{1-b_t}{\gamma}b_s=\dfrac{1-b_sb_t}{\gamma}
=a_{s+t}\)。故 \(A_tA_s=A_{s+t}\)（同理 \(A_sA_t=A_{s+t}\)，因同一生成元 commute；
本檔採 \(A_tA_s\) 以與 composition convention 一致）。∎

**Proof（affine drift \(d_{s+t}=A_td_s+d_t\)）.** deterministic mean 部分：
\(m_{s+t}=A_{s+t}z_0+d_{s+t}\)；另一方面由 composition，
\(m_{s+t}=A_t(A_sz_0+d_s)+d_t=A_{s+t}z_0+(A_td_s+d_t)\)。對所有 \(z_0\) 相等迫使
\(d_{s+t}=A_td_s+d_t\)。（亦可 block 代入 \(d_\Delta=[(\Delta-a)\bar v;(1-b)\bar v]\)
直接驗證，與此一致。）∎

**Proof（covariance composition \(Q_{s+t}=A_tQ_sA_t^\top+Q_t\)）.**
_(隨機層)_ 由 convention，\(\eta_{s+t}=A_t\eta_s+\eta_t'\)，兩項獨立，故
\(\operatorname{Cov}(\eta_{s+t})=A_t\operatorname{Cov}(\eta_s)A_t^\top
+\operatorname{Cov}(\eta_t')=A_tQ_sA_t^\top+Q_t\)。
_(積分層，直接驗證等式)_ 由 noise-integral 形，
\[
Q_{s+t}=\int_0^{s+t}e^{F\tau}\Sigma e^{F^\top\tau}\mathrm d\tau
=\underbrace{\int_0^{t}e^{F\tau}\Sigma e^{F^\top\tau}\mathrm d\tau}_{=Q_t}
+\int_{t}^{s+t}e^{F\tau}\Sigma e^{F^\top\tau}\mathrm d\tau .
\]
第二項令 \(\tau=\sigma+t,\ \sigma\in[0,s]\)：
\[
\int_0^{s}e^{F(\sigma+t)}\Sigma e^{F^\top(\sigma+t)}\mathrm d\sigma
=e^{Ft}\Big(\int_0^{s}e^{F\sigma}\Sigma e^{F^\top\sigma}\mathrm d\sigma\Big)e^{F^\top t}
=A_t\,Q_s\,A_t^\top .
\]
故 \(Q_{s+t}=A_tQ_sA_t^\top+Q_t\)。∎

**Proof（Chapman–Kolmogorov，measure form）.** 內層 \(K_s(z_0,\cdot)=\mathcal N(A_sz_0+d_s,Q_s)\)。
把此 Gaussian measure 經 affine map \(z_1\mapsto A_tz_1+d_t\) push-forward 得
\(\mathcal N(A_t(A_sz_0+d_s)+d_t,\ A_tQ_sA_t^\top)\)，再與獨立 \(\mathcal N(0,Q_t)\)
卷積（Gaussian 卷積 = mean 相加、covariance 相加）得
\[
\mathcal N\!\big(A_t(A_sz_0+d_s)+d_t,\ A_tQ_sA_t^\top+Q_t\big)
=\mathcal N\!\big(A_{s+t}z_0+d_{s+t},\ Q_{s+t}\big)=K_{s+t}(z_0,\cdot),
\]
最後一步用上三個 composition 等式。此即 CK。論證全程以 Gaussian measure / 卷積
進行，**不**要求 \(Q_s,Q_t\) 可逆，故對 L1 的 degenerate（\(D\) rank-deficient）情形
同樣成立。∎

**Consequence（well-posedness，不選 terminal）.** L3 證明 \(\{K_\Delta\}\) 為
compositionally consistent 的 time-homogeneous Markov transition semigroup：任意
gap 切分下 transition law 一致，無 compositional ill-posedness。此內容**支持** charter
terminal 2（`GCTM_TRANSITION_FAMILY_NOT_WELL_POSED`）不因 composition 性質被觸發，但
terminal 選擇僅於 terminal review 依 frozen decision procedure 進行；本檔不代行。

## §5 Lemma L4 — short-gap / long-gap asymptotics

**型別提醒：** 以下 \(q_{\bullet\bullet}\) 為 **scalar** 係數；covariance **block**
為 \(q_{\bullet\bullet}D\)。兩者分開陳述。

### §5.1 Short gap（\(\Delta\to0^+\)，等價 \(\gamma\Delta\ll1\)）

固定 \(\gamma\)，對 \(\Delta\) 展開（等價地固定 \(\Delta\) 令 \(\gamma\to0\)，見 L2；此處
以 \(\gamma\Delta\) 為小量）：
\[
q_{vv}=\Delta+O(\gamma\Delta^2),\quad
q_{xv}=\frac{\Delta^2}{2}+O(\gamma\Delta^3),\quad
q_{xx}=\frac{\Delta^3}{3}+O(\gamma\Delta^4),
\]
\[
a=\Delta-\frac{\gamma\Delta^2}{2}+O(\gamma^2\Delta^3),\qquad
d_\Delta(c)=\begin{bmatrix}\tfrac{\gamma\Delta^2}{2}\bar v+O(\Delta^3)\\[2pt]
\gamma\Delta\,\bar v+O(\Delta^2)\end{bmatrix},\qquad d_\Delta(c)=O(\Delta)
\]
（drift 的 velocity 分量本身是 \(O(\Delta)\)，故整體 \(d_\Delta(c)=O(\Delta)\)，
**不是** \(O(\Delta^2)\)；只有 position 分量是 \(O(\Delta^2)\)。）

**covariance leading blocks 即 M1**（\(Q_{M1}\) blocks \(\tfrac{\Delta^3}{3}D,
\tfrac{\Delta^2}{2}D,\Delta D\)），與 \(\gamma\) 無關。**詮釋（限縮至 covariance
leading blocks）：** 就 process-noise 的 covariance leading blocks 而言，
mean-reversion（\(\gamma\) 效應）為高階小量，M1 是普適短-gap 極限；OU 與 CV 的
process noise 在短 gap 不可區分。

**此結論不涵蓋 state mean。** 完整 M2 mean 相對 M1 的 velocity 差為
\[
m_\Delta^v-v_0=-\gamma\Delta\big(v_0-\bar v(c)\big)+O(\Delta^2),
\]
即 velocity mean 的 mean-reversion 是**一階**效應（position mean 差為
\(-\tfrac{\gamma\Delta^2}{2}\big(v_0-\bar v(c)\big)+O(\Delta^3)\)，二階）。故「短 gap
不可區分」僅適用於 covariance leading blocks，**不**延伸到 velocity state mean。

### §5.2 Long gap（\(\gamma>0\) 固定，\(\Delta\to\infty\)，\(b=e^{-\gamma\Delta}\to0\)）

**Scalar 係數：**
\[
q_{vv}=\frac{1-b^2}{2\gamma}\to\frac{1}{2\gamma},\qquad
q_{xv}=\frac{(1-b)^2}{2\gamma^2}\to\frac{1}{2\gamma^2},\qquad
q_{xx}=\frac{2\gamma\Delta-3+4b-b^2}{2\gamma^3}=\frac{\Delta}{\gamma^2}+O(1).
\]
**對應 covariance blocks：**
\[
q_{vv}D\to\frac{D}{2\gamma},\qquad
q_{xv}D\to\frac{D}{2\gamma^2},\qquad
q_{xx}D=\frac{\Delta}{\gamma^2}D+O(1)\ \text{（block，隨 }\Delta\text{ 線性成長）}.
\]
**Mean：** \(b\to0,\ a=1/\gamma+O(e^{-\gamma\Delta})\to1/\gamma\)，故 \(A_\Delta\) 收斂
（entrywise，指數快）：
\[
A_\Delta\to\begin{bmatrix}I&\tfrac1\gamma I\\0&0\end{bmatrix}.
\]
drift **不**收斂為有限向量（position 分量在 \(\bar v(c)\neq0\) 時線性發散），只能寫成
漸近式：
\[
d_\Delta(c)=\begin{bmatrix}\big(\Delta-\tfrac1\gamma\big)\bar v(c)+O(e^{-\gamma\Delta})\\[2pt]
\bar v(c)+O(e^{-\gamma\Delta})\end{bmatrix}.
\]
即 **velocity 分量收斂**至 \(\bar v(c)\)（\(v_0\) 記憶被 \(b\to0\) 抹去、reversion 至
context mean）；**position 分量隨 \(\Delta\) 線性成長，不收斂**（\(\bar v(c)\neq0\) 時
無有限極限）。

**詮釋與 M1 對比：** velocity covariance 飽和到 OU stationary 值 \(D/(2\gamma)\)；
position covariance block \(\sim(\Delta/\gamma^2)D\) 隨 \(\Delta\)**線性**成長
（diffusive）。相對地 M1 的 \(q_{xx}D=\tfrac{\Delta^3}{3}D\) 為**三次**成長
（super-diffusive）。此即 M2 與 M1 在長 gap 的實質分歧：

| 量（block） | M2（\(\gamma>0\)，\(\Delta\to\infty\)） | M1（\(\gamma=0\)） |
|:--|:--|:--|
| velocity cov \(q_{vv}D\) | \(\to D/(2\gamma)\)（飽和／stationary） | \(\Delta D\)（線性發散） |
| position cov \(q_{xx}D\) | \(\sim(\Delta/\gamma^2)D\)（線性，diffusive） | \(\tfrac{\Delta^3}{3}D\)（三次，super-diffusive） |
| velocity mean | 忘記 \(v_0\)，revert \(\bar v(c)\) | 保留 \(v_0\) |

**Crossover scale** \(\Delta\sim1/\gamma\)：\(\gamma\Delta\ll1\) 時 M1-like（§5.1），
\(\gamma\Delta\gg1\) 時 OU-saturated（本節）。此 scale 是 M1/M2 可區分性的分界，屬
diagnostic 觀察，不含任何 fitting/threshold claim。

## §6 Dimensional consistency（cross-reference，不重證）

Dimensional / unit consistency 已在 frozen **D1 §4.7** 以 sanity 形式陳述
（\(A_\Delta\) 的 \(aI\) block 把 \(\ell/\mathrm{frame}\to\ell\)；\(q_{xx}D\sim\ell^2\)、
\(q_{xv}D\sim\ell\cdot\ell/\mathrm{frame}\)、\(q_{vv}D\sim(\ell/\mathrm{frame})^2\)）。
本檔的 L1–L4 全部保持該量綱（PSD/nesting/composition/asymptotics 皆不改變 units），
故不重複證明，僅 cross-reference D1 §4.7。此為 charter D2 列「dimensional consistency」
一項的落點。

## §7 Lemma L5 — \(q\) / NLL ranking equivalence under shared covariance（WP-A5 增量）

**Status.** 本節此前為 WP-A3 的 reserved dependency boundary（無 theorem）；\(S_\Delta\)
現由 frozen D1 §5（WP-A4）定義、claim-space 由 frozen D1 §6（WP-A5）定義，故本次
**WP-A5** 於此填入 lemma。charter D2 列項「`q`/NLL ranking equivalence under shared
covariance」的落點即此。**本節不選 terminal、不引入 D1 §4/§5/§6 以外的物件、不修改
frozen D1。**

**Imported notation（沿用 frozen D1 §5/§6，不重定義）.** Candidate event
\(\mathcal E\)（D1 §6.2）內 candidate \(i\)：innovation \(r_i\)（D1 §5.3）、total
innovation covariance \(S_{\Delta,i}\)（D1 §5.4）、\(k_i=\dim r_i\)、standardized
innovation \(q_i=r_i^\top S_{\Delta,i}^{-1}r_i\)、Gaussian NLL
\(E_i=\tfrac12 q_i+\tfrac12\log\det S_{\Delta,i}+\tfrac{k_i}{2}\log2\pi\)（D1 §6.1）。
全程假設所涉 \(S_{\Delta,i}\succ0\)（D1 §6.1 regime：canonical \(C=0\)、\(R_1\succ0\)；
退化情形不在本節範圍）。

> **Lemma L5（shared-covariance ordering equivalence）.** 固定一個 candidate event
> \(\mathcal E\)。若所有 candidate 共用同一 covariance \(S_{\Delta,i}=S\succ0\)（此即
> 唯一前提；共用同一矩陣 \(S\) 已蘊含共用維度 \(k\)。**不**要求同 \(\Delta,c,H\) 或同
> \(R_1\)：\(S_\Delta=HP^-_\Delta H^\top+R_1\) 含 entry-time 的 \(R_1\)、且 \(c\) 只入
> drift mean 不入 covariance，故那些來源既不必要也不充分），則對任意 \(i,j\in\mathcal E\)：
> \[
> q_i\le q_j\iff E_i\le E_j .
> \]
> 即在 event 內，\(q\) 與 NLL 誘導**相同**的 total preorder（ranking-equivalent）。

**Proof.** 由 \(S_{\Delta,i}=S\)，log-volume 項 \(\tfrac12\log\det S\) 與 additive
常數 \(\tfrac k2\log2\pi\) 皆與 \(i\) **無關**。記
\(\kappa:=\tfrac12\log\det S+\tfrac k2\log2\pi\)（candidate-independent），則
\[
E_i=\tfrac12 q_i+\kappa\qquad\forall i\in\mathcal E,
\]
故 \(E_i-E_j=\tfrac12(q_i-q_j)\)，兩差同號。因 \(\tfrac12>0\)，\(q_i\le q_j\iff
E_i\le E_j\)（嚴格不等亦然）。∎

> **Corollary L5.1（shared／isotropic gap-scaling ranking invariance）.** 承 L5 的
> shared-\(S\) 設定：對任意 event-level scalar \(\alpha>0\)，把 \(S\mapsto\alpha S\)
> **不改變** event 內 \(q\)- 與 NLL-order；特別地 \(S=\alpha I\)（isotropic）時
> \(q_i=\lVert r_i\rVert^2/\alpha\)，order \(=\lVert r_i\rVert^2\)-order，與
> \(\alpha\) 無關。故 gap-conditioned shared／isotropic scaling 對 candidate-local
> ranking **不變**，只**可能影響** calibration 而**不**影響 ranking（\(\alpha=1\) 為
> identity，什麼都不改；D1 §6.3(I)）。

**Proof.** \(S\mapsto\alpha S\Rightarrow q_i=r_i^\top S^{-1}r_i\mapsto q_i/\alpha\)：
因 \(1/\alpha>0\) 且與 \(i\) 無關，\(q_i/\alpha\) 與 \(q_i\) 同 order。NLL 變為
\(\tfrac12 q_i/\alpha+\kappa(\alpha S)\)，仍為 \(q_i\) 的 candidate-independent 遞增
仿射函數，order 不變。isotropic \(S=\alpha I\Rightarrow S^{-1}=\alpha^{-1}I\Rightarrow
q_i=\lVert r_i\rVert^2/\alpha\)。∎

> **Boundary L5.2（tightness：candidate-specific covariance 可改變 order）.** 若
> candidates **不**共用 covariance（\(S_{\Delta,i}\) 隨 \(i\) 變，如 candidate-specific
> \(\Delta_i,c_i\)，D1 §6.4），則
> \[
> E_i-E_j=\tfrac12(q_i-q_j)+\tfrac12\big(\log\det S_{\Delta,i}-\log\det S_{\Delta,j}\big),
> \]
> 第二項 candidate-dependent，可與第一項反號，故 \(q\)- 與 NLL-order **可不同**。因此
> **shared-\(S\) 前提不能從 universal ordering-equivalence guarantee 移除**（此
> counterexample 只證「移除 shared-\(S\) 後，同序不再對所有情形成立」；它**不**宣稱
> \(S_i\neq S_j\) 就**必然**異序——特定 \(S_i\neq S_j\) 下 \(q\) 與 NLL 仍可能碰巧同序）。

**Proof（constructive counterexample，\(k=1\)）.** 取兩 candidate：
\((r_1,S_1)=(1,\,1)\Rightarrow q_1=1,\ \log\det S_1=0\)；
\((r_2,S_2)=(1.2,\,4)\Rightarrow q_2=0.36,\ \log\det S_2=\log4\)。去掉共同常數
\(\tfrac k2\log2\pi\) 後比較 \(\tfrac12 q+\tfrac12\log\det S\)：candidate 1 得 \(0.5\)、
candidate 2 得 \(0.18+\tfrac12\log4\approx0.873\)。故 \(q_2<q_1\)（\(q\) 偏好 candidate
2）但 \(E_1<E_2\)（NLL 偏好 candidate 1）——orders 相反。∎

**Region-probability 註（型別提醒，非本 lemma 的一部分）.** D1 §6.1 的 candidate-region
probability \(\Pi_i(\Omega_i)\) 依賴 region volume／geometry，故即使 shared \(S\)，
**非全等** regions 下 \(\Pi\)-order 可與 \(q\)-order 不一致（較大 region 因 probability
mass 勝出）。此為 D1 §6.3 caveat 的重述，不改 L5（L5 只斷言 \(q\) 與 NLL 同序）。

**Consequence（不選 terminal）.** L5＋L5.1 給出 D1 §6.3(I)「calibration gain ⇏
ranking gain」在 \(q\)/NLL 上的精確形式；L5.2 給出其 tightness。此為 charter obligation 2
的 proof 面，**支持** obligation 2 於 terminal review 被記為 complete，但 terminal
選擇僅於 terminal review 依 charter frozen decision procedure 進行，本檔不代行。

其餘 D1 §4.8/§5.7/§6.6 typed deferrals（identifiability／leakage、B1/O1、
reverse-time atom、runtime、online、production、data、fitting）不在本檔範圍。

## §8 Scope / terminal note

本檔**不選** terminal。L1（PSD、rank 刻畫）與 L3（compositional consistency）的
well-posedness 內容**支持** charter terminal 1/2 不因這些性質被觸發；L2（nesting）滿足
charter §M2 的 \(\gamma\to0\) mean+covariance nesting 要求；L4 為 diagnostic
asymptotics；L5（§7）為 charter obligation 2 `q`/NLL ranking-equivalence 的 proof 面
（**支持** obligation 2 被記為 complete）。terminal 選擇僅於 terminal review 依 charter
frozen decision procedure 進行（**後續 terminal-review packet**，非本檔亦非 WP-A5），
本檔不代行，也不修改 frozen D1 interface。

## Appendix A — 符號／數值驗證

WP-A3 的 closed-form 與極限恆等式以 symbolic algebra 交叉驗證（獨立於 D1 §4 的
WP-A2 驗證），全部通過：

- scalar Gram 積分 \(\int_0^\Delta g^2,\int_0^\Delta gh,\int_0^\Delta h^2\) 與 D1 §4.5
  closed form \((q_{xx},q_{xv},q_{vv})\) 恆等；
- \(\gamma\to0\) 極限 \((a,q_{vv},q_{xv},q_{xx})\to(\Delta,\Delta,\Delta^2/2,\Delta^3/3)\)（L2）；
- semigroup \(a\)-composition \(a_{s+t}=a_s+a_tb_s\)（L3）；
- \(\det M_\Delta=q_{xx}q_{vv}-q_{xv}^2\ge0\)，且 \(\gamma\to0\) 時 \(\to\Delta^4/12\)（L1）。

**WP-A5 增量（§7 L5）的數值 sanity（elementary algebra）：** shared-\(S\) 下
\(\operatorname{argsort}(q)=\operatorname{argsort}(E)\) 且 \(E-\tfrac12 q\) 為常數
（L5）；isotropic \(S=\alpha I\) 下 \(q\)-order \(=\lVert r\rVert^2\)-order，與
\(\alpha\in\{0.1,1,7.3,100\}\) 無關（L5.1）；\((r,S)=(1,1),(1.2,4)\) 給 \(q\)-order 與
NLL-order 相反（L5.2 counterexample）；non-congruent region 下 \(\Pi\)-order 與
\(q\)-order 可相反（§6.3 caveat）。另（#255 review 對應）：CAL Gaussian working null
下 Gaussian \(r\) 給 \(q\sim\chi^2_k\)（KS 一致），**同 covariance 但 non-Gaussian**
（\(t_3\)）則被 \(\chi^2_k\) 拒斥——確認 \(q\sim\chi^2_k\) 需 Gaussianity 而非僅正確
covariance；event 內**統一**單調 \(\varphi_{\mathcal E}\) 保序、**per-candidate 不同**
\(\varphi_i\) 可改序。全部通過。

驗證為 appendix-level sanity，不改任何 frozen 定義；proof 的權威來自 §2–§5、§7 的推導。

## History

- 2026-07-22 — D2 created by **WP-A3**: proof appendix for the frozen D1 §4
  kernel \(K_\Delta(z_0,c)=\mathcal N(A_\Delta z_0+d_\Delta(c),Q_\Delta)\).
  Proves L1 (\(Q_\Delta\succeq0\ \forall\Delta\ge0\); for \(\Delta>0\),
  \(M_\Delta\succ0\) and \(Q_\Delta\) singular \(\iff D\) singular with
  \(\operatorname{rank}Q_\Delta=2\operatorname{rank}D\); \(\Delta=0\) degenerate
  boundary \(Q_0=0\)); L2 (\(\gamma\to0\) nesting of mean **and** covariance to
  M1, plus proven continuity of \(\gamma\mapsto(A_\Delta,d_\Delta,Q_\Delta)\) on
  \([0,\infty)\), upgrading D1 §4.6 from interface closure to lemma); L3
  (deterministic semigroup \(A_{s+t}=A_tA_s\), affine drift
  \(d_{s+t}=A_td_s+d_t\), covariance composition
  \(Q_{s+t}=A_tQ_sA_t^\top+Q_t\), and Chapman–Kolmogorov measure form, valid for
  degenerate \(Q\)); L4 (short-gap universal M1 limit; long-gap OU-saturated
  velocity cov \(D/(2\gamma)\), linear-in-\(\Delta\) position cov
  \((\Delta/\gamma^2)D\) vs M1 cubic, crossover \(\Delta\sim1/\gamma\)). §6
  dimensional consistency cross-references frozen D1 §4.7 (no re-proof); §7 marks
  the \(q\)/NLL ranking-equivalence dependency (\(S_\Delta\) from WP-A4; claim in
  WP-A5) as a reserved boundary with no theorem/proof/acceptance obligation.
  Selects no terminal; introduces no WP-A4/A5 object; makes no interface,
  runtime, data, or production change. Frozen D1 §2–§4 untouched (byte-frozen;
  no append-only correction). Symbolic cross-checks in Appendix A all pass.
- 2026-07-22 — L4 bounded corrections per #253 owner review (COMMENT verdict
  = changes required; pre-merge, nothing frozen, WP-A3 scope unchanged, frozen
  D1 untouched): (1) **§5.1 short-gap drift order** fixed — the velocity
  component of \(d_\Delta(c)\) is \(O(\Delta)\), so \(d_\Delta(c)=O(\Delta)\)
  (not \(O(\Delta^2)\); only the position component is \(O(\Delta^2)\)); the
  "OU/CV indistinguishable at short gap" statement is now restricted to the
  covariance leading blocks, and the first-order velocity-mean reversion
  \(m_\Delta^v-v_0=-\gamma\Delta(v_0-\bar v(c))+O(\Delta^2)\) is stated
  explicitly (excluded from the indistinguishability claim). (2) **§5.2
  long-gap drift** changed from an (illegitimate) limit arrow to an asymptotic:
  \(d_\Delta(c)=[(\Delta-1/\gamma)\bar v(c)+O(e^{-\gamma\Delta});\,\bar v(c)
  +O(e^{-\gamma\Delta})]\), noting the velocity component converges to
  \(\bar v(c)\) while the position component grows linearly and does not
  converge. \(A_\Delta\) limit unchanged (entrywise convergent). Verified
  symbolically.
- 2026-07-22 — **§7 filled by WP-A5** (charter obligation 2, proof side): added
  Lemma **L5** (\(q\)/NLL ranking equivalence under shared covariance — within an
  event with \(S_{\Delta,i}=S\succ0\), \(E_i=\tfrac12 q_i+\kappa\) with \(\kappa\)
  candidate-independent, so \(q\) and NLL induce identical orderings), Corollary
  **L5.1** (shared/isotropic gap-scaling \(S\mapsto\alpha S\) is ranking-invariant
  — calibration-only), and Boundary **L5.2** (candidate-specific covariance can
  flip \(q\) vs NLL order; \(k=1\) counterexample \((r,S)=(1,1),(1.2,4)\)), plus
  the region-probability caveat. Uses only frozen D1 §5 \(S_\Delta\) (canonical
  \(C=0\), \(S_\Delta\succ0\) regime) and D1 §6 score definitions; introduces no
  new transition object; selects no terminal. §7 was a reserved boundary (no
  theorem) under WP-A3, so this is a fill, not a modification of an accepted
  lemma. Also: **one owner-flagged non-blocking nit fixed** from the #253 review
  — §5.2 "drift 不收斂為有限矩陣" → "有限向量" (\(d_\Delta\) is a vector
  \(\in\mathbb R^{2d}\)), applied under review as the owner directed. §1.1's
  WP-A3 "不證明 q/NLL... 見 §7" is now completed by WP-A5 (scope note §1.3); §8
  terminal-review pointer corrected (terminal review is a later packet, not
  WP-A5). L1–L4 mathematically byte-unchanged; frozen D1 §2–§5 untouched.
  Symbolic/numeric cross-checks (L5/L5.1/L5.2 + region-probability) in Appendix A
  all pass.
- 2026-07-22 — bounded corrections per #255 owner review (pre-merge; §7 not yet
  frozen; L1–L4/§2–§5 byte-unchanged): (1) **L5 condition** — dropped the wrong
  "等價地共用 \(\Delta,c,k,H\)" equivalence; L5's sole premise is
  \(S_{\Delta,i}=S\) (sharing the matrix \(S\) already fixes the dimension \(k\);
  \(R_1\) is entry-time and \(c\) enters drift mean, not covariance, so those
  sources are neither necessary nor sufficient). (2) **L5.2 necessity softened** —
  the counterexample shows shared-\(S\) cannot be removed from the *universal*
  ordering-equivalence guarantee; it does **not** claim \(S_i\neq S_j\) forces a
  different order (specific unequal \(S\) may still coincide in order). Core L5/L5.1
  algebra and the \(k=1\) counterexample unchanged.
- 2026-07-22 — second bounded correction per #255 owner re-review (pre-merge;
  §7 not yet frozen; L1–L4/§2–§5 byte-unchanged): L5.1 wording softened —
  shared/isotropic rescaling "may affect calibration, not ranking" (identity
  \(\alpha=1\) changes nothing), matching D1 §6.3(I). L5's "唯一前提" wording kept
  (it names the premise list, not a mathematical necessity), and the L5 statement
  already flags \(S_{\Delta,i}=S\) as sufficient (sharing \(S\) fixes \(k\); \(R_1\)
  entry-time, \(c\)→drift mean). Core L5/L5.1/L5.2 unchanged.
