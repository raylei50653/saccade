<!-- doc-status: draft -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-22 -->
<!-- doc-module: semantic -->

# GCTM 範圍與方案 — primary object = A

> **HISTORICAL — pre-activation planning memo（2026-07-22 撰寫；2026-07-23 起
> superseded）。** 本 memo 的 **primary-object = A 決策**與 **existing-online
> §§9.2–9.4 conditional scope amendment** 已被 owner 接受並落地（PR #248 acceptance
> ／PR #249 landing），GCTM 隨後於 2026-07-22 activate（WP-A0）、2026-07-23 關閉於
> owner-accepted ordered terminal **`GCTM_MODEL_SPEC_SEALABLE`**。**其餘平行建議
> 不屬於 GCTM closure，也未因此執行或取得 authority**——特別是 §4 的 `WP-L2∥`
> （registry L2 contract）與 `WP-B1`（B1 declaration）：**registry-owned L2
> score-layer contract 仍不存在**（[registry §7 架構缺口](../../../research/contracts/claim_state_registry.md)），
> B1 仍是 proposed。因此文中一切「parked」「keep GCTM
> parked」「activation 前」的敘述都是**當時快照**，**不是現況**；現況與 terminal
> 由下列 owner 擁有：
> [closed charter *Final status*](../../../research/threads/closed/gap_conditioned_stochastic_transition_model_task.md) ·
> [D1 spec v1](../../../research/models/gap_conditioned_stochastic_transition_spec_v1.md) ·
> [D2 lemmas v1](../../../research/models/gap_conditioned_stochastic_transition_lemmas_v1.md) ·
> [terminal review](../../../research/models/gap_conditioned_stochastic_transition_terminal_review_v1.md) ·
> [claim-state registry](../../../research/contracts/claim_state_registry.md)。
> 本 memo 保留原文作為決策脈絡（primary object = A 的選擇理由與被否決的替代），
> 不重寫、不追認。
>
> **Planning memo · not active · not sealed · no execution authority（原文保留）**
>
> 本文件是單一範圍／方案稿：收斂先前 GCTM 評語的修正，固定 **primary object = A
> （latent state transition）**，並重排 activation 前工作順序。
>
> 它**不**取代：
>
> - [GCTM task charter（撰寫時 parked；現為 **CLOSED**）](../../../research/threads/closed/gap_conditioned_stochastic_transition_model_task.md)
> - [H0→GCTM consumer compatibility](h0_gctm_consumer_compatibility_requirements_20260718.md)
> - [B1/O1 shared semantics](gctm_b1_o1_task_objectives_and_semantics_20260716.md)
> - [existing online object analysis](existing_online_object_analysis_for_gctm_alignment_20260718.md)
> - [claim-state registry](../../../research/contracts/claim_state_registry.md)
>
> 若本 memo 的 proposed scope 與上述 authority 的現行文字不一致，**現行
> authority 仍優先**；本 memo 只記錄待 owner 接受並由後續 docs PR 落地的
> normative amendment，不得以「解讀」方式靜默改寫既有義務。
>
> 它**不**啟動 GCTM、不選 terminal、不授權 data / fitting / capture / online /
> production、不修改 registry 或 sole-active WIP。

---

## 0. 一句話

GCTM 的 claim identity 應固定為：

\[
\boxed{
\text{A — latent state transition }\;
K_\Delta : s_0 \mapsto \mathcal P(\mathcal S_\Delta)
}
\]

B / C / D 是沿

\[
A \;\longrightarrow\; B \;\longrightarrow\; C \;\longrightarrow\; D
\]

逐層增加 substrate 與 consumer 假設的 **downstream constructible objects**，不得在
GCTM model-spec seal 前全部吞入理論層。

---

## 1. 對先前評語的三點修正

先前評語正確抓到三條真正危險的接縫（時間座標、機率 object 層級、
pair→event→score 語義），但下列三點必須修正。

### 1.1 H0 狀態不得簡化成「已死」

**錯誤簡化：**

> H0 已關於 `H0_PROVENANCE_INVALID` ⇒ GCTM 只能 indefinitely parked。

**應採用的 lifecycle 敘述：**

> 先前授權的 H0 輪次因 provenance invalid 關閉（route-1 永久留帳：無 faithful
> capture、無 accepted runtime-fidelity edge、無 actual H0 guarantee envelope）。
> 任何後續 H0 路徑都是 **separate re-entry**：必須先過 qualification，再經
> fresh I→F→S、exactly-once 授權與 owner 接受，才可能產生新的 ordered terminal。
> **在新的 owner-accepted positive terminal 與 guarantee registration 完成前，
> usable envelope 仍為空，不能做 bridge-runtime B1 claim。**

契約本身已明確區分：

| 路徑 | H0 是否阻擋 |
|:--|:--|
| Substrate-agnostic GCTM 數學（well-posedness、affine、PSD、nesting） | **否** |
| Bridge-runtime GCTM/B1 claim | **是** — 需要 accepted substrate / fidelity edge + compatibility verdict |

因此：H0 負終端不自動否定抽象 M1/M2；也不自動啟動 GCTM；更不授權 B1。

三個 gate 必須分開讀：

| Gate | H0 要求 |
|:--|:--|
| GCTM lifecycle activation | 任一 owner-accepted ordered terminal（可為負）+ separate owner scheduling |
| Substrate-agnostic A-layer model seal | 不要求 positive H0 substrate；仍須先依法 activation |
| Bridge-runtime B1 | accepted substrate / fidelity edge + guarantee registration + B1-owner-accepted compatibility verdict |

**Lifecycle authority：** registry / H0 declaration / owner acceptance 仍是唯一
狀態來源。本節只修正 *planning language*，不宣稱任何尚未 owner-accepted 的
新終端。

### 1.2 「未決 obligation」≠「邊界模糊」

先前評語把下列項目稱為「最模糊邊界」是誤標：

| 項目 | 權責邊界現狀 | 真正缺的是 |
|:--|:--|:--|
| \(g_{\mathrm{phys}}\) 不得默認等於 \(\Delta_{\mathrm{on}}\) | **清楚** — 映射由 GCTM 定義 | 內容決策（convention + map） |
| L2 contract 不存在時 B1/O1 score-ranking blocked | **清楚** — registry fail-closed | 另建 registry-owned L2 契約 |
| prediction / observation error 獨立或顯式 \(C\) | **清楚** — 二選一義務 | 內容決策 |
| M2 必須 canonical affine，不能只講 residual narrative | **清楚** — seal 條件 | 完整 interface + proofs |

更精確的名稱：

> **已明確定位、但尚未作出內容決策的 activation blockers。**

這反而是現有架構的優點：即使內容未解，錯誤路徑已 fail-closed。

真正概念上危險、且會改變 *claim identity* 的，是另一類問題：

- 是否把 A/B/C/D 揉成單一「transition likelihood」；
- 是否把 pair→event competition 吸進 GCTM core；
- 是否把 bridge-runtime 宣稱掛在空的 H0 envelope 上。

### 1.3 Pair→event competition 不必在 GCTM core 解完

這是與先前評語的最大實質差異。

Existing-online 分析 §9.1 要求 GCTM **先選 primary object**，並禁止把四層揉成
單一 likelihood；但其 §§9.2–9.4 與 `Required composition declaration` 目前仍以
無條件的「GCTM 必須」要求 native/event correspondence、competition distinction
與 composition choice。這些現行文字可被合理讀成 GCTM seal obligation，故不能
只靠本 memo 宣稱它們已經下放。

本方案提出的 **normative scope qualification** 是：

| Claim scope | Applicable obligation |
|:--|:--|
| A-layer model-spec seal | 完成 A、observation/time interface，以及指向 B/C/D 的 typed boundary；不必完成 event competition |
| 宣稱 B/C/D 或 existing-online correspondence | 必須滿足 existing-online §§9.2–9.4 中與該 claim 有關的 state/output/context correspondence |
| `replace-ranking`、decision probability 或 commit-related claim | 必須另有 frozen L2、event operator 與相應 B1/online charter；不得由 A seal 推出 |

此 qualification 必須經 owner 接受，並在後續 docs PR 中修改／註明
existing-online authority；在落地前，既有文件的現行文字仍控制。

若把 pair→event operator 列為 GCTM seal 必要組件，會把 B1 / L2 決策語義吸進
理論層。現有文件已把

```text
augment / calibrate / replace pair score / replace ranking / shadow-only
```

放在 future **B1 declaration**，且 replacement 必須受 frozen L2 contract 約束。

本方案因此**提議**把 competition / ranking / claim composition 下放到 B1（與
registry L2），不放進 primary=A 的 GCTM model-spec seal checklist。

---

## 2. Primary object 決策

### 2.1 選定

| Layer | Object | GCTM role under this plan |
|:--|:--|:--|
| **A** | Latent state transition \(K_\Delta\) | **Primary claim identity** |
| **B** | Native event-state law \(\mathcal L_{\mathrm{native},\boldsymbol\Delta}\) | Downstream constructible — needs event lift, dependence structure, and state/observation mapping |
| **C** | Score distribution \((F_{\mathrm{event}}^{(c)})_\#\mathcal L_{\mathrm{native},\boldsymbol\Delta}\) | Downstream constructible — enters production score map |
| **D** | Decision probability over discrete online decision | Downstream constructible — needs competitor universe, mask, margin, claim composition |

### 2.2 為何選 A

1. **A 才是 M0/M1/M2 真正定義的數學 object** — affine transition、\(Q_\Delta\)、
   nesting、PSD 都落在 latent state 層。
2. **B 已依賴 event lift + mapping** — 必須先把 pair-level \(K_{\Delta_i}\)
   組成 event-indexed joint law（明定 conditional independence 或 dependence），
   再經 \(\phi\) 或 \(\Phi\) 到 native state；這超出 pure transition family，
   但仍可比 A 更早以 *interface obligation* 形式預留。
3. **C 進入 production score map** — 牽涉 `bdist` / score atoms，屬
   runtime-aligned derived claim，不是 transfer family 本體。
4. **D 需要 competitor universe 與 claim composition** — 這是 B1/L2/online
   contract 的核心，不是 GCTM 數學 seal 的必要條件。

### 2.3 Pushforward chain（語意，非自動 handoff）

```text
A  latent transition law
   ↓  (event-indexed lift + dependence + state/observation mapping)
B  native event-state law
   ↓  (production continuous score map F^{(c)})
C  score distribution (pair scores, margins as continuous objects)
   ↓  (eligibility, ranking rule, claim composition under L2 + online contract)
D  discrete decision probability
```

規則：

- 每一支箭頭都是 **typed construction**，需額外假設；上游 sealable **不**推出下游有效。
- A 本身不決定 B：若不能宣告 event-indexed joint kernel、competitor assembly 與
  dependence structure，就只能保留 pair-level A，不得寫成完整 event law。
- GCTM 可 *define interfaces* 供 B 使用（canonical state、observation mode、
  time map），但 **GCTM seal 只對 A 負責**。
- B 的 runtime correspondence 由 mapping validation 負責；C 的 calibration／ranking
  value 由 B1 負責。D 的 distributional construction 與 O1 的 causal system efficacy
  是不同 claim，不得把「可計算 decision probability」直接視為 online value。

### 2.4 禁止的混層說法

在 primary = A 下，下列敘述 inadmissible，除非另開 object 與 charter：

1. 「GCTM transition likelihood = relink / commit probability」；
2. 「M1/M2 sealable ⇒ event ranking gain」；
3. 「pair \(p(i\leftrightarrow C)\) 自動給出 best/second/margin/claim」；
4. 「production bridge 是 intrinsic Markov kernel \(P_\Delta\)」；
5. 用單一 scalar 同時取代 bdist gain 與 decision boundary。

合法說法形式：

> Under model \(M\) and gap \(\Delta\), the latent transition law induces
> innovation residuals whose **calibration** may be studied separately from any
> **candidate-local ranking** score derived for a future B1 declaration.

---

## 3. 範圍邊界

### 3.1 本方案的 in-scope（GCTM core / theory path）

在 owner 另行 scheduling 啟動 GCTM charter 之後，理論路徑只負責：

```text
1. claim identity = A
2. canonical state, coordinates, units
3. time / gap conventions and g_phys ↔ Δ_on mapping ownership
4. M0 baseline + M1/M2 admitted family
5. M2 canonical affine form, Q_Δ, domains, nesting M2→M1
6. observation modes H_x vs H_xv and causal availability declarations
7. P0 / QΔ / R1 / SΔ uncertainty objects (and independence or cross-covariance)
8. quantity split: standardized innovation q, logdet S, NLL — as
   specification-level diagnostics induced by A + declared observation
   interface, not as ranking policies
9. calibration vs ranking as different claim spaces (definitions + nulls shape)
10. provisional → sealed terminal partition for GCTM_MODEL_SPEC_SEALABLE path
```

輸出類別固定為 **diagnostic-only model specification**（charter 既有
`output-class: diagnostic-only`）。

### 3.2 平行但非 GCTM 數學內容

| 工作 | Owner | 與 GCTM 關係 |
|:--|:--|:--|
| Registry-owned **L2 score-layer contract** | claim-state / research contracts | B1/O1 前置；**不是** GCTM 數學 |
| H0 qualification / re-entry / guarantee registration | H0 declaration + owner | Bridge-runtime enabler only |
| H0→GCTM consumer compatibility verdict | compatibility doc + owner | B1 bridge-runtime gate |
| B1 declaration（score insertion policy） | B1 charter | Freezes how A-derived scores enter ranking |
| O1 online intervention | O1 charter | After B1 design candidate |

### 3.3 Explicit out-of-scope（本 memo 與 GCTM seal 皆不做）

```text
no data execution / parameter fitting
no GT/FP reveal protocol
no frozen-pair-table empirical ranking study
no CUDA / runtime hook
no H0 ABI change
no online score write
no production default change
no pair→event competition operator as GCTM seal requirement
no claim arbitration / loser fallback / commit mutation
no automatic mainline transition H0→GCTM→B1→O1
```

### 3.4 GCTM seal 最小充分條件（under primary = A）

`GCTM_MODEL_SPEC_SEALABLE` 只要求 A-layer 完備：

| Obligation | Seal 需要？ | 不需要？ |
|:--|:--|:--|
| Canonical latent state + M2 affine + \(Q_\Delta\) + units | 是 | — |
| \(M2\to M1\) nesting + PSD arguments | 是 | — |
| Observation interface + causal availability | 是 | 已驗證 runtime 對齊 |
| \(g_{\mathrm{phys}}\leftrightarrow\Delta_{\mathrm{on}}\) **定義** | 是 | 已 empirically 校準 |
| Independence or explicit \(C\) | 是 | 資料擬合 |
| Calibration vs ranking claim-space **definitions** | 是 | ranking 實證有效 |
| Typed pointer：B/C/D 為 separately constructible, not primary | 是 | B/C/D 已實作 |
| L2 contract frozen | **否**（平行 blocker for B1） | — |
| Event competition operator | **否** | B1/L2 |
| H0 full-faithful terminal | **否** for abstract math；**是** for bridge-runtime B1 | — |

即使 sealable，仍 **不**授予 B1/O1/online/production 權限（charter 既有規則）。

---

## 4. 方案：重排後的優先順序

> 最先決定的不是 gap mapping，而是 **claim identity = A**。
> 一旦選 A，許多看似 GCTM 的問題自然下放到 B1，不必在 model seal 前全部吞掉。

### Step 0 — Claim identity（本 memo 採用）

- [x] 選定 primary object = **A**
- [ ] 在 future active GCTM declaration 中 **明文**寫入：B/C/D = downstream constructible
- [ ] 禁止混層 likelihood 語言（§2.4）

*Status of checkboxes above: planning adoption in this memo only; not landed into
charter/declaration until a separate docs PR after owner scheduling.*

### Step 1 — Freeze geometry of A

Freeze, as interface definitions (not empirical results):

1. canonical state \(z = [x; v]\) (or sealed alternative with migration note);
2. coordinate substrate id;
3. frame time unit / continuous \(\mathrm{d}t\) conversion;
4. physical gap definition \(g_{\mathrm{phys}}\);
5. online horizon \(\Delta_{\mathrm{on}}\) / `bridge_at` convention;
6. owned map \(g_{\mathrm{phys}}\leftrightarrow\Delta_{\mathrm{on}}\) (may be identity
   only if **explicitly** declared and justified — never silent);
7. production-CV null-offset treatment.

### Step 2 — Complete M2 as affine transition family

Deliver reviewable:

\[
z_\Delta = A_\Delta z_0 + d_\Delta(c) + \eta_\Delta,
\qquad
\eta_\Delta \sim \mathcal N(0, Q_\Delta)
\]

with parameter domains, units, \(Q_\Delta\) construction, and required proofs:

- \(M2 \to M1\) as \(\gamma \to 0\) (mean + covariance);
- PSD of \(Q_\Delta\);
- short- / long-gap asymptotics (diagnostic lemmas).

M0 remains comparison baseline only; M1 remains nested white-acceleration limit.

### Step 3 — Uncertainty composition

先固定符號：令 \(m^-_\Delta\) 為 prediction mean、\(z_\Delta\) 為 true state，

\[
e^- = z_\Delta-m^-_\Delta,
\qquad
y_1=Hz_\Delta+\epsilon_1,
\qquad
r=y_1-Hm^-_\Delta=He^-+\epsilon_1.
\]

再 decide exactly one of:

- independence of prediction error and entry-observation error; or
- explicit cross-covariance \(C=\operatorname{Cov}(e^-,\epsilon_1)\) and expanded

\[
S_\Delta = H P^-_\Delta H^\top + R_1 + HC + C^\top H^\top.
\]

Keep objects separate: \(P_0\), \(Q_\Delta\), \(R_1\), \(S_\Delta\).
若採相反的 prediction-error 定義，cross terms 的符號也必須一起改，不得只保留
上述公式。

### Step 4 — Split calibration vs ranking **as claims**, not as one metric

At GCTM specification + observation-interface layer, define quantities and
null shapes only:

| Quantity | Calibration space | Ranking space |
|:--|:--|:--|
| \(q = r^\top S^{-1} r\) | residual size vs declared uncertainty | 可形成 candidate-local order；是否改變 baseline order 取決於 declared covariance geometry |
| \(\log\det S\) | predictive volume | often shared → no event-local order change |
| NLL | joint fit + volume | same-order as \(q\) when all candidates share \(S\), dim, gap, mode |

Rule retained from charter: shared isotropic \(S_\Delta = \alpha_\Delta I\) cannot
change the ordering induced by the same residuals' squared Euclidean norm — it
only rescales calibration. This does not mean \(q\) has no ordering; it means the
gap-dependent scalar alone adds no new candidate-local order information.

Empirical ranking value is **B1**, not GCTM seal.

### Step 5 — Parallel: L2 contract（not GCTM math）

Independently (registry-owned):

- freeze cutoff / ranking / margin / top-1 / candidate-universe semantics;
- keep all `layer: L2 score` objects fail-closed until then;
- do **not** encode L2 policy inside GCTM equations.

### Step 6 — B1 declaration only: pair score → event ranking

Only after (or gated on) sealable A + L2 + H0 substrate/fidelity as required:

- freeze score policy class:
  `shadow-only | augment | calibrate | replace-pair-score | replace-ranking`;
- freeze insertion surface (GPU foot-bridge stage-1 candidate-local ranking);
- freeze candidate universe, margin, claim boundary non-mutation rules;
- never smuggle claim/commit changes without re-charter.

### Step 7 — O1 and production evaluation

Unchanged evidence chain: B1 design candidate → O1 online retention/efficacy →
separate production evaluation. No automatic handoff.

---

## 5. 與既有文件的對齊關係

```text
existing_online_object_analysis
  owns production operator object + A/B/C/D menu + compatibility constraints
  → this memo selects A and proposes a conditional scope amendment:
    §§9.2–9.4 / composition apply when claiming B/C/D or online correspondence,
    not as unconditional A-seal obligations
  → until that amendment lands, existing unconditional wording still controls

gap_conditioned_stochastic_transition_model_task (當時 parked charter；現 CLOSED，見 threads/closed/)
  owns stable research question, M0/M1/M2 family boundary, activation gates
  → this memo proposes primary-object resolution + reordered blockers
  → does NOT activate the charter

h0_gctm_consumer_compatibility_requirements
  owns R_obs ⊆ Γ_H0 registration protocol
  → unchanged; still blocks bridge-runtime B1 while Γ usable = ∅

gctm_b1_o1_task_objectives_and_semantics
  owns cross-layer typed mapping + claim spaces + forbidden shortcuts
  → consistent with primary A; B1 owns score policy under L2

claim_state_registry
  owns object rungs / admissibility
  → this memo proposes no registry mutation
```

---

## 6. Work packages（啟用後才開，本 memo 不開）

啟用前提仍是 charter 既有 gate：

```text
accepted H0 ordered terminal   (scheduling prerequisite language; not logical
                                necessity for pure math)
∧ separate owner scheduling decision
∧ explicit activation of GCTM theory work
```

啟用後建議拆成可審查的小包（示意）：

| WP | Content | Exit |
|:--|:--|:--|
| WP-A0 | Primary = A 寫入 active declaration; B/C/D construction map | declaration text sealed for identity |
| WP-A1 | State / time / gap / coordinate interface freeze | observation-interface checklist pass |
| WP-A2 | M2 affine + \(Q_\Delta\) + domains | transition family well-posed draft |
| WP-A3 | Nesting + PSD + asymptotics lemmas | proof appendix draft |
| WP-A4 | Independence / \(C\) decision + \(S_\Delta\) form | uncertainty composition sealed |
| WP-A5 | Calibration vs ranking claim-space defs + terminal partition | ready for `GCTM_MODEL_SPEC_SEALABLE` review |
| WP-L2∥ | Registry L2 contract (parallel) | B1 score path unblocked at registry layer |
| WP-B1 | B1 declaration after gates | score insertion policy frozen |

本 memo **不建立** 這些檔案或 PR。

---

## 7. Risks if primary A is not frozen first

| Failure mode | Symptom | Mitigation in this plan |
|:--|:--|:--|
| Likelihood soup | One scalar claimed as motion, score, and commit prob | Primary A + §2.4 bans |
| Theory absorbs L2 | GCTM seal blocked on ranking policy fights | Proposed scope amendment defers competition to B1 |
| Proxy time | Silent \(g_{\mathrm{phys}}=\Delta_{\mathrm{on}}\) | Step 1 explicit map ownership |
| Residual-only M2 | Narrative without affine interface | Step 2 seal condition |
| Bridge claim on empty Γ | “Runtime-grounded” without H0 guarantee | §1.1 + compatibility doc |
| Calibration≠ranking collapse | Isotropic \(S\) sold as ranking gain | Step 4 claim-space split |

---

## 8. Success criteria for *this* memo

This document is successful if and only if a reader can answer:

1. What is GCTM’s claim identity? → **A**
2. Are B/C/D core seal requirements? → **No under the proposed A-scope amendment; they are separately constructed downstream objects**
3. Is pair→event competition a GCTM seal blocker? → **No under the proposed amendment; it is required when a downstream B1/L2/decision claim needs it**
4. Does empty H0 envelope block pure math? → **No**
5. Does empty H0 envelope block bridge-runtime B1? → **Yes**
6. What is the first content decision after identity? → **canonical state + time/gap map**
7. What remains parallel non-math? → **L2 contract**

---

## 9. Recommended next actions（human / owner only）

**Not authorized by this memo.** Listed for navigation:

1. Owner review of both: primary object = A **and** the conditional scope amendment
   to existing-online §§9.2–9.4 / composition requirements (accept / amend / reject).
2. If accepted: later docs PR must first land that qualification in the
   existing-online authority, then land A into future active GCTM declaration
   language (still requires GCTM activation gate).
3. Keep GCTM charter **parked** until owner scheduling. *(已發生：owner scheduling 2026-07-22 ⇒ activate；2026-07-23 closed。)*
4. Continue H0 re-entry / qualification on its own authority chain; do not
   couple its WIP to GCTM math drafting unless a bridge-runtime claim is intended.
5. Optionally open a **separate** registry L2 contract draft as non-WIP
   governance work — still not GCTM content.

---

## 10. Source map

| Topic | Authority |
|:--|:--|
| Parked GCTM problem boundary | `docs/research/threads/closed/gap_conditioned_stochastic_transition_model_task.md` |
| A/B/C/D menu + online operator | `docs/modules/semantic/research/existing_online_object_analysis_for_gctm_alignment_20260718.md` |
| H0 consumer compatibility | `docs/modules/semantic/research/h0_gctm_consumer_compatibility_requirements_20260718.md` |
| B1/O1 shared semantics | `docs/modules/semantic/research/gctm_b1_o1_task_objectives_and_semantics_20260716.md` |
| B1 / O1 charters | `docs/research/threads/gctm_b1_*`, `gctm_o1_*` |
| Transition panel (projection only) | `docs/research/threads/README.md` |
| Lifecycle / envelope state | `docs/research/contracts/claim_state_registry.md` |

---

## 11. Change log

| Date | Change |
|:--|:--|
| 2026-07-22 | Initial planning memo: correct H0 framing; rename blockers; select primary A; reorder plan; propose deferring competition to B1. |
| 2026-07-22 | Review repair: make the authority conflict explicit; add conditional scope amendment, event-level lift/dependence, distinct native-law notation, H0 gate split, covariance sign convention, and ranking clarification. |
| 2026-07-22 | Owner accepted both scope decisions (primary object = A; conditional scope qualification for existing-online §§9.2–9.4). The qualification is landed in-force in `existing_online_object_analysis_for_gctm_alignment_20260718.md` §§9.2–9.4 via PR #249. Acceptance authorizes only the authority qualification; it does not activate GCTM, select a terminal, modify lifecycle/sole-active WIP, or authorize data/fitting/H0/B1/O1/runtime/online/production work. |
