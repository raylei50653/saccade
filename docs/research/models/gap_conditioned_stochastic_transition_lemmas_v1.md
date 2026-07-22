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
drift **不**收斂為有限矩陣（position 分量在 \(\bar v(c)\neq0\) 時線性發散），只能寫成
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

## §7 Reserved dependency boundary — **not part of WP-A3 acceptance**

本節**沒有** theorem、proof 或 acceptance obligation，僅記錄一個依賴邊界，避免空白
reserved section 在 terminal review 被誤判為未完成 deliverable。

**\(q\) / NLL ranking equivalence under shared covariance（charter D2 列項）：**

- 依賴 \(S_\Delta\)（total innovation covariance），而 \(S_\Delta\) 由
  \(P_0,Q_\Delta,R_1\)（及 independence／cross-covariance \(C\)）組成——這些是
  **WP-A4**（obligation 3）才定義的物件，**不在** frozen D1 §4、也**不在** WP-A3 範圍。
- 其**實質 claim**（calibration-only gain vs candidate-local ranking gain 為不同
  capability；shared \(S_\Delta\) 下 \(q\) 與 NLL 同序）由 **WP-A5 / obligation 2**
  承接。
- 因此本節在 WP-A3 **無** theorem/proof/acceptance obligation；待 WP-A4 定義
  \(S_\Delta\) 後，於後續 D2 增量填入。WP-A3 的 acceptance **不**包含本節。

其餘 D1 §4.8 typed deferrals（\(P_0/R_1/S_\Delta/C\)、reverse-time atom、B1/O1、
runtime、online、production、data、fitting）同樣不在本檔範圍。

## §8 Scope / terminal note

本檔**不選** terminal。L1（PSD、rank 刻畫）與 L3（compositional consistency）的
well-posedness 內容**支持** charter terminal 1/2 不因這些性質被觸發；L2（nesting）滿足
charter §M2 的 \(\gamma\to0\) mean+covariance nesting 要求；L4 為 diagnostic
asymptotics。terminal 選擇僅於 terminal review 依 charter frozen decision procedure
進行（WP-A5 收尾），本檔不代行，也不修改 frozen D1 interface。

## Appendix A — 符號／數值驗證

WP-A3 的 closed-form 與極限恆等式以 symbolic algebra 交叉驗證（獨立於 D1 §4 的
WP-A2 驗證），全部通過：

- scalar Gram 積分 \(\int_0^\Delta g^2,\int_0^\Delta gh,\int_0^\Delta h^2\) 與 D1 §4.5
  closed form \((q_{xx},q_{xv},q_{vv})\) 恆等；
- \(\gamma\to0\) 極限 \((a,q_{vv},q_{xv},q_{xx})\to(\Delta,\Delta,\Delta^2/2,\Delta^3/3)\)（L2）；
- semigroup \(a\)-composition \(a_{s+t}=a_s+a_tb_s\)（L3）；
- \(\det M_\Delta=q_{xx}q_{vv}-q_{xv}^2\ge0\)，且 \(\gamma\to0\) 時 \(\to\Delta^4/12\)（L1）。

驗證為 appendix-level sanity，不改任何 frozen 定義；proof 的權威來自 §2–§5 的推導。

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
