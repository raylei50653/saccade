---
doc-status: active
doc-promotion: research-line normative doc; not evidence_ledger
owner-module: semantic
created: 2026-07-11
---

# 布林閉包域研究線：從 GT placement morphology 到結構約束 reject-domain 抽取

> **One-line:** 把「布林超立方體能量地形」收斂為一條可執行、可驗證、可停止的研究線：**資料決定合法偏序，偏序限制域搜尋，統計界決定域是否可接受**。PR ladder = **PR-B**（本文檔）；上游 = [PR-A #100](https://github.com/raylei50653/saccade/pull/100) · [PR-C #104](https://github.com/raylei50653/saccade/pull/104) · [PR-D gate #106](https://github.com/raylei50653/saccade/issues/106) `GLOBAL_PARTIAL_ORDER_READY`；下游 = separate restricted-closure prototype → PR-E nested validation。

Thread: [gt_support_morphology_20260711.md](../../../research/threads/gt_support_morphology_20260711.md) ·
Procedure: [framework §19](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（terminal 判準的唯一權威）·
Step-0: [gt_support_morphology_step0_20260711.md](gt_support_morphology_step0_20260711.md)

## 0. 文件定位

本文件將「布林超立方體能量地形」方向收斂為一條可在現有 MOT association-recovery 資料上執行、可驗證、可停止的研究線。

本研究不再假設能從目前的稀疏 cell table 完整估計 per-cell risk landscape，也不把 graph trend filtering、persistent homology、QPBO 或邏輯壓縮串成必經的七步管線。

目前的核心問題是：

> 先由 GT placement morphology 判定哪些 atom 維度具備合法的單調偏序，再在該偏序下使用 maximum-weight closure 產生結構一致的 reject-domain candidate frontier，最後以真實 pooled GT-UCB 與 nested held-out folds 決定候選域是否成立。

核心原則：

$$
\boxed{\text{資料決定合法偏序，偏序限制域搜尋，統計界決定域是否可接受。}}
$$

**權威分界（one fact one home）：** morphology 的 terminal 判準、ε_morph 界線、UCB 方法、escape-tail 保護規則的家在 [framework §19](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)（PR-A seal）。本文檔擁有的是**研究線本身**：正式問題、atom 偏序資格分類、MWC pipeline、Verdict A–E 完成條件、任務包與 PR ladder。本文檔不得重定義 §19 的 terminal 或界線。

## 1. 研究動機

現有 gate-rule search 可以找到有效的 Boolean clauses，但無法直接回答：

- gate 是否只是單一離散切點的偶然結果；
- reject region 是否具有可解釋的單調結構；
- 哪些 atoms 可以作為全域排序維度；
- 哪些 atoms 在特定 regime 中發生 role reversal；
- 現有 OR-tail 是否已接近最佳結構域，還是仍有更大的安全 reject closure。

[Step-0 identifiability audit](gt_support_morphology_step0_20260711.md)（procedure verdict = **UNRESOLVED**；以下為 bounded descriptive hypothesis，無 terminal 效力）已表明：

- 完整 per-cell GT risk landscape 在目前資料密度下不可識別；
- GT placement distribution 可識別；
- 描述層假說：GT 質量 corner-concentrated，另有極薄的 far-Hamming、motion-violation-enriched descriptive tail（4/209，**4/4 集中 MOT17-10-SDP** —— sequence clustering 為實據，nominal CP 不得跨 ε_morph 界線）；
- 假說層面：motion 類 atoms 可能非全域單調；height／scale 類 atoms 較接近全域單調候選（log_h_ratio 0/4 違反）——均待 PR-C forensic 與 Phase B partial-order audit 確認。

因此研究物件應由：

$$
P(\text{GT hurt}\mid Z=z)
$$

改為：

$$
P(Z=z\mid \text{GT track})
$$

以及對整個 reject domain 的 unit-level 非對稱損失約束：

$$
P_u(Z_u\subseteq D).
$$

## 2. 正式研究問題

令：

- $Z\in\{0,1\}^k$：Boolean atom state；
- $z_i=1$：該 atom 位於宣告的較安全側；
- $F_z$：cell $z$ 中可被 reject 的 FP 質量；
- $G_z$：cell $z$ 的 GT placement mass（僅 descriptive／MWC 線性化 surrogate；同一 track 可跨多個 cell，不可作正式 hurt 計數）；
- $Z_u$：GT trial unit $u$ 的所有有效 GT cells；
- $P$：由 morphology-supported atoms 定義的偏序；
- $\mathcal C(P)$：滿足該偏序的 downward-closed reject domains。

研究目標為：

$$
\begin{aligned}
\max_D\quad
& P_{\mathrm{FP}}(Z\in D)\\
\text{s.t.}\quad
& \operatorname{UCB}\!\left[P_u(Z_u\subseteq D)\right]\le \epsilon,\\
& D\in\mathcal C(P),\\
& P\text{ 僅由 morphology-supported atoms 定義。}
\end{aligned}
$$

（$\epsilon$ 為該 study 宣告的 GT-hurt budget；它與 §19.5 的形態分類預算 $\varepsilon_{\mathrm{morph}}=5\%$ 是**不同**的量，per-study 宣告。）

reject domain $D$ 的 downward-closed 條件與 §19.5 的 core（argmin retained-FP 的 monotone **upper** closure $C$）是同一結構的兩面：$D=\Omega\setminus C$，$z\in D,\ z'\le z \Rightarrow z'\in D$。**Track-level hurt 採 §19.4 set-valued semantics**：$H_C(u)=\mathbf 1[Z_u\cap C=\varnothing]$（closure 未保留該 track 任何有效 GT candidate 才算 hurt）；因此 $P_u(Z_u\subseteq D)=P(H_{\Omega\setminus D}(u)=1)$。min-d_H representative 不得作 closure 驗證的 trial 表示。

本問題包含兩個相互分離的研究責任：

### 2.1 統計／機制問題

判定哪些 atoms 足以合法定義偏序 $P$。

### 2.2 組合最佳化問題

在固定偏序 $P$ 下，尋找 FP removal 最大、GT harm 受控的 reject closure $D$。

**不得以第二步的最佳化結果反向證明第一步的 morphology 假設。**

## 3. 已知可識別邊界

### 3.1 目前可識別

- corner mass；
- Hamming shell／tail mass；
- per-atom violation profile；
- pairwise violation profile；
- reject domain 的 pooled track-level GT exposure；
- FP mass 在 Boolean cells／shells 中的分布。

### 3.2 目前不可識別

- 完整 per-cell conditional risk；
- basin depth；
- barrier height；
- saddle points；
- persistent homology lifetime；
- per-fold morphology field；
- 由無 exposure cells 推出的 unsafe claim。

硬限制（= framework §19.2）：

$$
\boxed{\text{no GT exposure} \neq \text{unsafe}}
$$

因此 graph trend filtering 或其他平滑器不能被用來「補出」缺失的 GT risk landscape。

## 4. Morphology 描述層

### 4.1 固定形態特徵

同 framework §19.4：$M_0$ · $M_r$ · $T_{\ge r}$ · $V_i$ · $V_{ij}$ · $d_H = d_{\mathrm{structural}} + d_{\mathrm{motion}}$ 分解（用於辨識 GT escape tail 是否主要沿 motion 維度延伸）。

### 4.2 Morphology 描述子類與 §19.5 terminal 對映

**Binding terminal 只有 §19.5 的四個**（`MONOTONE_CORE` / `CORE_PLUS_CONDITIONAL_ESCAPE_TAIL` / `DIFFUSE_OR_NONMONOTONE` / `UNRESOLVED`）。本線在描述層額外使用六個子類，報告時必須同時標注其 binding terminal：

| 描述子類 | binding terminal（§19.5） |
|:--|:--|
| 1. monotone core | `MONOTONE_CORE` |
| 2. corner core + conditional escape tail | `CORE_PLUS_CONDITIONAL_ESCAPE_TAIL` |
| 3. diffuse support | `DIFFUSE_OR_NONMONOTONE` |
| 4. multiple supported regimes（兩組均有足夠 exposure、機制可區分的 clusters） | `DIFFUSE_OR_NONMONOTONE`（全域偏序不成立；後續走 conditional closure，見 Verdict C） |
| 5. threshold-sensitive morphology（形態對二值 threshold 高度敏感） | `UNRESOLVED`（sealed thresholds 下 terminal 不穩） |
| 6. unresolved / non-identifiable | `UNRESOLVED` |

目前 Step-0 的 **procedure verdict = `UNRESOLVED`**（無 valid cluster-aware UCB、core 未求解、forensic 未跑、nested 未重跑；見 [Step-0 note §4](gt_support_morphology_step0_20260711.md)）。其 bounded descriptive hypothesis：

> corner-concentrated placement 為主體，另有 far-Hamming、motion-violation-enriched 的 descriptive tail（違反 motion continuity 而非 height consistency；**4/4 集中單一序列 MOT17-10-SDP**）。

此假說僅屬 pooled、in-sample 描述層，**不佔用任何 §19.5 terminal token**；正式 trial 表示須用 §19.4 set-valued semantics（$Z_u$ 全集 + $H_C(u)$），min-d_H representative 只作描述。

## 5. Atom 角色與偏序資格

每個 atom 必須在正式 closure search 前被歸類為：

| 類型 | 定義 | 在 closure 中的角色 |
|---|---|---|
| `global_orderable` | 跨主要 regime 均支持同一安全方向 | 可建立全域 closure arcs |
| `conditional_orderable` | 只在特定 regime 中支持單調方向 | 僅能建立 regime-specific closure arcs |
| `context_only` | 有判別訊號，但不具全域單調性 | 可影響權重或 regime，不建立硬偏序 |
| `unresolved` | support／threshold／fold 證據不足 | 不得進入正式偏序 |

關鍵理念：

> 非單調 atom 應退出偏序，但不一定退出模型。

例如 `speed_mismatch`、`dir_cos`、`resid_mean` 可作為：

- long-gap re-entry regime 指標；
- node utility 的條件訊號；
- short-gap conditional closure 的排序維度；

但不得直接作為 8-D global closure 的硬偏序。

可將 atoms 分為：

$$
A=A_{\mathrm{order}}\cup A_{\mathrm{context}}
$$

其中 $A_{\mathrm{order}}$ 定義偏序；$A_{\mathrm{context}}$ 定義權重、條件或 regime。

## 6. Maximum-weight closure 的正確角色

### 6.1 Downward-closed reject domain

若 $z_i=1$ 表示較安全側，reject domain 應滿足：

$$
z\in D,\ z'\le z \Longrightarrow z'\in D.
$$

意思是：若某狀態應被 reject，逐座標更不安全的狀態也應被 reject。

### 6.2 線性化候選問題

以 placement mass 建立每個 cell 的**候選權重**：

$$
w_\lambda(z)=F_z-\lambda G_z
$$

則：

$$
D_\lambda=
\arg\max_{D\in\mathcal C(P)}
\sum_{z\in D}w_\lambda(z)
$$

是標準 maximum-weight closure problem，可化為一次 $s$-$t$ min-cut／max-flow。

主要參考：

- Hochbaum, *A New–Old Algorithm for Minimum-cut and Maximum-flow in Closure Graphs*: <https://hochbaum.ieor.berkeley.edu/html/pub/HPF-closure-Net2001.pdf>
- Hochbaum, *Minimizing a Convex Cost Closure Set*: <https://hochbaum.ieor.berkeley.edu/html/pub/CCC-SIAM-2003.pdf>

### 6.3 與真實 GT-UCB 問題的差異

真實安全條件為：

$$
\operatorname{UCB}\!\left[P_u(Z_u\subseteq D)\right]
\le\epsilon,
$$

等價地，令 $C=\Omega\setminus D$ 後評估
$\operatorname{UCB}[P(H_C(u)=1)]$。此 unit-level UCB 是 domain-level 非線性函數，且同一 track 可跨多個 cell；$G_z$ 因而不能直接作為正式 hurt 的可加總計數，也不能把安全條件直接分解為單次 MWC 的節點權重。因此正確流程是：

1. 掃描 $\lambda$；
2. 用 MWC 產生 closure candidate frontier；
3. 對每個 $D_\lambda$ 重新計算真實 pooled track-level UCB；
4. 保留滿足 $\epsilon$ 且 FP removal 最大的候選；
5. 再進行 nested held-out validation。

即：

$$
\boxed{
\text{MWC 產生結構候選前沿}
\rightarrow
\text{exact UCB 做最終安全判定}
}
$$

**MWC 是候選域最佳化器，不是 safety proof。**

由於 MWC 的 $G_z$ 僅是 placement surrogate，lambda-parametric frontier **不保證覆蓋** non-linear、set-valued GT-UCB 約束下的全部可行 closure。除非另有 exhaustive enumeration、exact constrained optimizer，或 completeness／upper-bound certificate，研究不得把「frontier 中無增益」表述為整個 closure family 的近似最佳性。

## 7. 與現有 OR-tail 的關係

現有 OR-tail policy 通常形式為：

$$
D=A_1\lor A_2\lor\cdots\lor A_m
$$

若各 atoms 都具合法 unsafe direction，這類 policy 對應一種低複雜度 downward-closed reject region。

因此：

$$
\boxed{\text{OR-tail = 低複雜度 closure family}}
\qquad
\boxed{\text{MWC = restricted candidate-frontier optimizer}}
$$

正式比較問題：

> 在相同 pooled GT-UCB 與 held-out 約束下，**已搜尋的 MWC-generated candidate frontier** 是否能比 frozen OR-tail 額外移除有實質意義的 FP 質量？

若不能，則 OR-tail 是該已搜尋 frontier 內較佳的工程表示；若能，才有必要研究 closure 壓縮。這不對未被 MWC frontier 覆蓋的 closure 作全域最優性主張。

## 8. 正式實驗管線

### Phase A — Placement morphology seal

**已由 PR-A（[#100](https://github.com/raylei50653/saccade/pull/100)，framework §19 + Step-0 note）承擔**；本線直接消費 sealed procedure，不重定義。

輸入：frozen event pool、frozen track unit、predeclared atom families、nested fold 內產生的方向與 thresholds。
輸出：corner mass、shell／tail profile、per-atom／pairwise violation profile、support completeness、morphology verdict、例外 track 清單。
限制：無 exposure 不得判為 unsafe；median thresholds 只可作 audit；正式 claim 必須在 nested fold 內重做 atom selection、direction 與 binarization。

### Phase B — Partial-order audit

輸出每個 atom 的 `global_orderable` / `conditional_orderable` / `context_only` / `unresolved` 分類，以及：

- global closure graph；
- conditional closure graphs；
- 被禁止的 closure arcs 與原因；
- role-reversal event evidence。

### Phase C — Restricted／conditional MWC

至少比較：

1. frozen OR-tail baseline；
2. restricted global closure；
3. short-gap conditional closure；
4. long-gap re-entry protected variant。

對每個候選輸出：selected cells、FP removed、GT tracks touched、pooled exact UCB、per-sequence exposure、closure complexity、與 OR-tail 的增量。

### Phase D — Nested held-out validation

每個 outer fold 內必須重新執行：atom selection、atom direction、thresholding、morphology verdict、orderability classification、MWC candidate generation、candidate selection。

外層只用於：GT hurt／UCB 評估、FP removal retention、topology／closure retention、failure attribution。

### Phase E — Rule compression（僅在 held-out 成立後）

可將 validated closure domain 壓縮為：singleton OR-tail、compact DNF、decision list、lookup table（若 $k$ 很小）。

可用工具：

- Espresso heuristic logic minimizer background: <https://si2.epfl.ch/demichel/publications/mcgraw/reductions/twolevel2.4.pdf>
- Berkeley Espresso code port: <https://github.com/Gigantua/Espresso>

壓縮後必須重新驗證 $D_{\mathrm{compressed}}=D_{\mathrm{validated}}$，或明確報告 $\Delta\text{GT hurt}$、$\Delta\text{FP removed}$。**壓縮不能作為 domain validity 的證據。**

## 9. 完成條件與 bounded verdict

本研究線不以「成功找到更複雜 gate」為完成條件，而要求輸出以下其中一種 bounded verdict：

```text
Verdict A — OR-tail sufficient within searched frontier
  在預先宣告、已搜尋且逐一經 exact track-level UCB 驗證的
  MWC-generated candidate frontier 內，MWC 對 frozen OR-tail 無實質
  FP removal 增益；OR-tail 是該 frontier 內較佳的低複雜度工程表示。
  這不是對整個 allowed closure family 的 near-optimal certificate。

Verdict B — Restricted closure advantage
  只用 morphology-supported global-orderable atoms 時，
  已搜尋 MWC frontier 的候選在 held-out folds 穩定優於 OR-tail。

Verdict C — Conditional closure required
  全域 closure 失敗，但分 regime closure 在 held-out 中成立，
  支持 role reversal／conditional monotonicity。

Verdict D — Closure hypothesis rejected
  即使限制偏序後，closure topology 或效益仍無法跨 fold 保留；
  不得升格為 domain model。

Verdict E — Non-identifiable
  GT support／fold exposure 不足以驗證 closure family；
  只保留描述性 morphology，不做最佳化 claim。
```

（此 A–E 是**研究線完成 terminal**，與 §19.5 的 morphology terminal 屬不同軸，兩者都必須 bounded 報告。）

## 10. 不進入目前主線的方法

### 10.1 Graph trend filtering — future exploratory tool

- Wang et al., *Trend Filtering on Graphs*: <https://www.jmlr.org/papers/volume17/15-147/15-147.pdf>
- Multivariate Trend Filtering for Lattice Data: <https://www.stat.berkeley.edu/~ryantibs/papers/kroneckertf.pdf>

理由：能平滑有噪聲的 graph signal，但不能解決無 GT exposure 導致的 non-identifiability，也不能把平滑結果當成 statistical significance。

### 10.2 Pseudo-Boolean／Möbius transform — secondary interaction diagnostic

- Ren et al., *Identifying Interactions via the Möbius Transform*: <https://proceedings.neurips.cc/paper_files/paper/2024/file/520b379123d16e41f85472e766846486-Paper-Conference.pdf>

可回答 interaction order，但高階係數弱不自動代表單調、線性可分或單盆地 morphology。

### 10.3 Persistent homology／merge tree — blocked by substrate

- *Representations of Energy Landscapes by Sublevelset Persistent Homology*: <https://arxiv.org/abs/2011.00918>

理由：不存在可信的完整 per-cell scalar risk field，basin、barrier 與 persistence 不可識別。

### 10.4 Isotonic regression — future partial-order diagnostic

- Stout, *Fast, Provable Algorithms for Isotonic Regression in all Lp-norms*: <http://papers.neurips.cc/paper/5824-fast-provable-algorithms-for-isotonic-regression-in-all-L_p-norms.pdf>
- GIRP correctness: <https://arxiv.org/pdf/2401.04847>

只有在局部節點 response 足夠可估時，才考慮用 isotonic residual 判斷偏序誤設。

### 10.5 QPBO — not needed

- Kolmogorov & Rother, *Minimizing non-submodular functions with graph cuts*: <https://pub.ista.ac.at/~vnk/papers/KR-PAMI07.pdf>

理由：QPBO 解 roof-dual relaxation，可能只產生 partial labeling；非一般非單調 morphology 的精確求解器。$k=8$ 的小狀態空間也無必要優先引入。

## 11. 最小可執行任務包（= PR ladder）

| Task | 內容 | PR |
|:--|:--|:--|
| Task 1 — Seal morphology procedure | unit of analysis、atom family eligibility、nested threshold generation、corner／shell／tail 指標、orderability 分類規則、verdict 邊界、無 exposure 處理規則 | **PR-A [#100](https://github.com/raylei50653/saccade/pull/100)（done pending merge）** |
| Task 2 — Four-track forensic | Step-0 四條 $d\ge3$ GT tracks（**4/4 在 MOT17-10-SDP**，見 packet `tail_tracks.json`）逐件落入五個預宣告類別；判定 motion atom role reversal 是否具真實機制來源，並分辨單場景 vs 通用機制 | **PR-C / #102 · [PR #104](https://github.com/raylei50653/saccade/pull/104)** · aggregate `ROLE_REVERSAL_SUPPORTED` · **`ACCEPTED_WITH_LIMITS`** · [note](escape_tail_forensic_20260711.md) · [packet](evidence/escape_tail_forensic_20260711/manifest.json) |
| Task 2b — Partial-order audit gate | 8 frozen atoms → `global_orderable` / `conditional_orderable` / `context_only` / `unresolved`；allowed/forbidden graph contract；**no** MWC | **PR-D gate / [#106](https://github.com/raylei50653/saccade/issues/106)** · [PR #107](https://github.com/raylei50653/saccade/pull/107) · terminal **`GLOBAL_PARTIAL_ORDER_READY`** · global=`{dist_h, log_h_ratio}` · `bridge_dist` demoted (motion-extrapolation) · [note](boolean_atom_partial_order_20260711.md) · [packet](evidence/boolean_atom_partial_order_20260711/manifest.json) |
| Task 3 — Restricted closure prototype | 只用 `global_orderable` atoms 建 closure graph，parametric MWC，vs frozen OR-tail；read-only、offline、candidate-only，不改 production preset | **separate task**（authorized only by #106 `GLOBAL_PARTIAL_ORDER_READY`；不得與 audit 同 PR） |
| Task 4 — Conditional closure probe | 若 Task 2 支持 re-entry mechanism：short-gap／re-entry conditional closure，檢查保護 GT tail 並維持 FP removal | **separate conditional-representation task**（motion arcs = proposal-only per #106） |
| Task 5 — Nested validation | 只有 Task 3／4 出現實質增益後，才進 outer-fold full replay | **PR-E** |

（本文檔 = **PR-B**，在 Task 1 seal 與 Task 2 forensic 之間入庫，不含任何 forensic 結果或新數字。）

## 12. 研究與工程責任邊界

**工程負責：** 可重現的 cell／track ledger、atom generation 與 nested thresholding、closure graph construction、min-cut／max-flow 求解、exact UCB 計算、evidence packet、baseline invariance。

**研究 review 負責：** morphology claim 是否超出 identifiability、closure arcs 是否有資料與機制依據、role reversal 是否被正確條件化、candidate selection 是否洩漏 outer-fold、OR-tail 與 MWC 比較是否公平、verdict 是否 bounded、是否可 promotion 到 evidence ledger。

**工程 merge 與 research acceptance 必須分離**（同 DEVELOPMENT.md §6 lock）。

## 13. 最終研究理念

本研究線不再以「估出完整能量地形」為前提，也不以「找到最低能量點」為目標。

$$
\boxed{
\text{placement morphology}
\rightarrow
\text{partial-order audit}
\rightarrow
\text{maximum-weight closure}
\rightarrow
\text{exact GT-UCB validation}
\rightarrow
\text{optional gate compression}
}
$$
