# Research threads（navigation-only）

**定位：** 跨子類連續任務的**導航卡 / 母線**，不是新的事實家。

```text
DEVELOPMENT.md       = stable global action router (D0–D4)
TODO.md              = mainline-charter WIP lock / pointer（sole active 一句）
threads/             = 進行中 charter（proposed · active · parked）
Expected state       = charter 內可替換的 planning lease（非 accepted state）
Current step / PR    = fast probes（可停止、替換、丟棄）
threads/closed/       = 已結案 thread 檔案家（歷史導航；非 evidence）
implementation PR    = engineering delivery authority (head/base/SHA/CI/files)
module research      = 事實與分析
ledger/report_data/no_go = 升格後的正式事實
```

```text
DEVELOPMENT.md → module TODO sole-active → active thread / contract → PR
```

## Mainline charter / expected state / probe

| 層 | 作用 | 更新代價 |
|:--|:--|:--|
| Mainline charter | 定義 decision question、boundary 與下一個 commit point；只有它可取得 `sole-active` | 中；改主線才 park / close / replace charter |
| Expected state lease | 寫「預計抵達什麼」與 disconfirm / discard condition | 低；可在同一 charter 內替換或刪除，不改 registry |
| Probe | `Current step` 或 implementation PR 中的執行嘗試 | 最低；可停止、替換、丟棄；無可重用 evidence 就不建 formal note |

WIP=1 **只綁 mainline charter**。工程 follow-up、資料補件、文件收尾與 probe 可同時以
`non-wip` 執行；它們只有在 owner 指定為新的 decision-changing charter 時才取得 WIP 鎖。

**不做：**

- 不放長表、完整分析、可引用數字正文
- 不取代 `evidence_ledger` / `signal_analysis_ledger` / module research / `report_data`
- **不**取代 module TODO 的 WIP=1 鎖（thread 可多張；sole active 仍以 TODO 為準）
- **不**使用 direct-agent `*.dispatch.yaml` sidecars（retired; do not recreate）

**結構契約：** [../../ownership/doc_structure_contract.md](../../ownership/doc_structure_contract.md)（O1.5）  
**行動入口：** [../../../DEVELOPMENT.md](../../../DEVELOPMENT.md#agent-action-cards)

---

<a id="current-transition-panel"></a>

## Current transition panel — H0 → GCTM

**Manual navigation projection · reconciled 2026-07-25.** This panel owns no
terminal, evidence, or WIP state: `ACTIVE` is projected from the module TODO;
task lifecycle and gates remain owned by the linked charters/contracts. If a
row conflicts with its owner, the owner wins. Update this panel in the same
change as any terminal acceptance, WIP-pointer switch, owner scheduling
decision, blocker change, or consumer-compatibility verdict.

| Field | Current projection | Owner / read first |
|:--|:--|:--|
| **ACTIVE** | **none** — semantic sole-active WIP is empty (`active_wip: []`). | [semantic TODO](../../modules/semantic/TODO.md) · [claim-state registry](../contracts/claim_state_registry.md) · [machine identity decision](../contracts/gctm_b1_slot_identity_decision_v1.json) |
| **NEXT** | No automatic next. Registration-v3 closed at `H0_REGISTRATION_V3_CONTRACT_SEALABLE` for consumer `gctm_runtime_native_candidate_universe_v1`. An owner may design **H0 re-entry and actual baseline capture** under the sealed contract, define an exact minimal ABI delta if a later producer gap appears, or re-scope the B1 runtime hook. Both runtime slots remain **proposed / non-WIP** and both compatibility gates remain independently `missing`. | [registration-v3 charter (closed)](closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md) · [runtime universe charter (closed)](closed/gctm_runtime_native_candidate_universe_task.md) · [claim-state registry](../contracts/claim_state_registry.md) · [B1 charter](gctm_b1_runtime_grounded_offline_attribution_task.md) · [O1 charter](gctm_o1_online_intervention_efficacy_task.md) |
| **SUCCESSOR DESIGN** | **H2 — proposed / non-WIP. Current edge: W5a–W5e owner-accepted; successor implementation/evidence closure is finished.** The successor now binds a 659-member `include/` + `scripts/` + `src/` execution-code closure, verifies per-run observed-import containment, keeps the resolved RunSpec namespace alias-free and immutable, and separates independently valid archives from canonical corpus admission. Only fully closed, valid `exactly_once_measurement` archives may be admitted. **W5e confirmed the boundary on final merged projection `e94407f2…`: exactly one non-qualifying diagnostic completed 4/4 runs with five predicates passing, remained `diagnostic_complete` / `terminal: null`, independently verified `valid: true` on all seven checks, and was refused by the canonical corpus owner because diagnostic authority is never measurement authority. Owner verdict at reviewed `main` `136b9eea…`: `ACCEPT — W5 implementation/evidence closure`.** External self-sealed custody is `/home/ray/h2_w5e_diagnostic_20260803T132933Z`, bound here by full packet-inventory digest `c25baf48b8425543a1f430b83b0381bcc2dc72386398d54acf22e41103bea268`; this is not a measurement, qualification, authorization, seal or equivalence claim, and W5 has no automatic next. The owner also recorded one separate non-blocking plumbing follow-up: the rehearsal witness catches a narrower exception surface than the corpus CLI; it remains unfixed and authorizes neither a projection-moving edit nor a re-run. Review Correction 11 records the implementation, re-pins the pre-seal body and republishes runtime identity without rerunning its probe. Replaces H0's **identity mechanism only** (declaration §0.1); H0's sealed history, five spent `S`, and permanent ledger entry are unchanged. Four owner decisions **accepted 2026-07-25** ([#286](https://github.com/raylei50653/saccade/issues/286)) — partition, `captured_under` schema, `H2` slot/prefix, MOT17-09 fixture — and acceptance **authorizes nothing**: no `I`/`F`/`S`, no capture, no exactly-once grant, no registry state write, no re-entry. Layer P (S1–S3) landed and green; Layer M (S4) is implemented, contract-tested, **reviewed and merged** ([#295](https://github.com/raylei50653/saccade/pull/295), reviewed head `4c78b962…`, landed as `b2f3c23f…`) — and still unsealed. Review Correction 3 published the Phase-B chain form. **The single authorized Phase-A invocation was spent at head `0a5dffe9…` on 2026-07-27** — all three head-bound gates were met there (controlled-host green, certificate `d95859cb…` 37/37, freeze `a03fc459…` 22/22) — and it reached terminal 1 `H2_INPUT_MUTATED_DURING_MEASUREMENT` with **0/4 ordered runs started, zero capture, no seal**; the archive was refused by the independent verifier. Root cause is inside the controller (it creates its own evidence root in the working tree, then requires that tree clean). **That repair landed ([#299](https://github.com/raylei50653/saccade/pull/299), `7646f421…`), the whole chain was rebuilt there (run `30334080842` green, certificate `266f4b4c…` 65/65, freeze `f0d1b02e…` 51/51), a second authorization was issued, and it was spent on 2026-07-28** — terminal 4 `H2_MEASUREMENT_EXECUTION_INVALID`, **1/4 ordered runs started, still zero capture, no seal**. Every controller defect from the first attempt is closed and this archive is complete and **accepted** by the independent verifier; the cause is one layer deeper, in the child, which re-applies its ingress environment contract after importing the eval stack while `cv2` 4.11.0 rewrites the environment on import — so it fails on any host with OpenCV installed. Items 4–5 are satisfied-then-void a second time, both authorizations remain permanently spent (each replaced only by a new, separately issued one, never rebuilt), and registering that archive exposed a third defect of the same shape — archive verification recomputed the authorization execution domain from the *verifying* host, so a committed archive verified nowhere else. **That repair landed on 2026-07-29** as two independent commits (`cc02a0b0…` child ingress authority, `7cae46d8…` archive verifier plus the corpus checker wired back into CI over full history), each held closed by a test that fails unmodified at `c2d1c58f…`; it authorizes nothing, seals nothing and restores no authorization. Historical execution/rehearsal details remain in the H2 charter and registry; they do not authorize another attempt. | [H2 charter](h2_behavioral_identity_capture_task.md) · [H2 declaration](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md) · [claim-state registry](../contracts/claim_state_registry.md) |
| **READINESS** | H0 remains closed at owner-accepted `H0_PROVENANCE_INVALID` (five spent `S`; no faithful capture, accepted runtime-fidelity edge, or actual guarantee envelope). Static audit `H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT` still forbids unchanged-interface capture. Consumer universe and registration-v3 contracts are sealable but do **not** establish actual completeness guarantees, substrate, compatibility, or B1 activation. H2's accepted decisions and green Layer P change none of this. No H0 re-entry, exactly-once authority, runtime verdict, or B1/O1 activation follows. | [registration-v3 packet](../../modules/semantic/research/evidence/h0_gctm_guarantee_registration_v3_20260724/) · [runtime universe packet](../../modules/semantic/research/evidence/gctm_runtime_native_candidate_universe_20260724/) · [static audit packet](../../modules/semantic/research/evidence/h0_gctm_interface_static_feasibility_20260723/) · [claim-state registry](../contracts/claim_state_registry.md) |

### Blockers and transition gates

```text
H0 → GCTM (history; both gates consumed)
  H0 closed at owner-accepted `H0_PROVENANCE_INVALID` (route 1; not full-faithful)
  ∧ separate owner scheduling decision (#175, 2026-07-22) activated the GCTM charter
  → GCTM ran WP-A0…WP-A8 and closed at `GCTM_MODEL_SPEC_SEALABLE` (2026-07-23)
  → substrate-agnostic A-layer only; H0 state unchanged by it

GCTM → bridge-runtime B1
  accepted H0 runtime substrate or fidelity edge          ← still absent
  ∧ B1 declaration owner accepts an H0→GCTM consumer compatibility verdict
  ∧ sealable GCTM specification and proofs                ← satisfied 2026-07-23
  ∧ owner-accepted score-layer L2 contract             ← satisfied 2026-07-23
  ∧ owner resolves the B1-slot identity                ← coexist, satisfied 2026-07-23; no authority granted
  ∧ a sealed B1 declaration and separate owner scheduling
  → B1 may become active

GCTM_D1 diagnostic line
  closed at owner-accepted mechanical terminal GCTM_D1_INTERFACE_READY
  ∧ declaration_owner_acceptance + owner_scheduling both historical/satisfied
  ∧ one canonical execution consumed (gctm_d1_canonical_execution_20260723)
  ∧ active_wip: []; decision_relevant_candidates: []
  ∧ accepted packet identities remain PR #265 sealed (not rewritten)
  ∧ diagnostic evidence cannot satisfy H0 substrate/provenance/identity/checksum
  ∧ missing H0 compatibility verdict fails closed
  → no B1/O1 candidate, no H0 re-entry, no production claim

H0 → GCTM static feasibility line
  path/hash-frozen D1 interface + trace-v2 + registration-v2 audit
  → H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT
  ∧ g_phys/residual/operator-offset/GCTM covariance structurally derivable
  ∧ candidate_universe unresolved under D1 synthetic identity
  ∧ event_membership completeness not registrable in registration-v2
  ∧ both runtime compatibility gates remain independently missing
  → no unchanged-interface capture/re-entry

GCTM runtime-native candidate-universe consumer line
  consumer re-charter over frozen trace-v2 + score contract + static-audit terminal
  → GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE
  ∧ identity gctm_runtime_native_candidate_universe_v1
  ∧ event_key / candidate_key / pre_score inclusion / composition frozen
  ∧ completeness semantics defined (not an H0 guarantee)
  ∧ registration-v3 requirements-only surface published
  ∧ both runtime compatibility gates remain independently missing
  → suitable consumer target for a separate minimal H0 registration-v3 delta;
    not H0 implementation authority; no B1/O1 activation

H2 successor identity line (proposed; authorizes nothing)
  H0's identity mechanism could not transfer across invocations (R5 parity audit)
  → H2 replaces identity only: coordinate / bounded probe / equivalence=unproven
  ∧ Layer P is pre-seal and retryable; Layer M keeps the exactly-once budget
  ∧ four owner decisions accepted 2026-07-25 (#286) — partition, captured_under
    schema, H2 slot/prefix, MOT17-09 fixture
  ∧ acceptance grants no I/F/S, no capture, no authorization, no registry write
  seal additionally requires
    a declared Phase-B chain form                        ← published (Correction 3)
    ∧ S4 Layer-M plumbing implemented                    ← reviewed; merged (#295)
  then, in this order, at one exact seal-candidate head
    the published coordinate/probe current + controlled host green
    ∧ a Layer-P pass certificate there, independently verified
    ∧ a separate owner exactly-once authorization
  all three met at 0a5dffe9 on 2026-07-27, authorization spent, executed once
    → terminal 1 H2_INPUT_MUTATED_DURING_MEASUREMENT; 0/4 ordered runs started
    → no capture, no seal, equivalence still unproven; archive verifier refused it
    → cause is inside the controller (self-created evidence root vs its own
      clean-checkout invariant), not the head, the bindings or the ruler
  ⇒ items 4–5 satisfied-then-void (rebuild at the successor head);
    item 6 consumed, permanently spent (a new authorization must be issued,
    it is not rebuilt); next = controller repair on a successor head,
    then a completely new acceptance and authorization cycle
  controller repaired (#299, 7646f421), whole chain rebuilt there
    (run 30334080842 green, cert 266f4b4c… 65/65, F64 f0d1b02e… 51/51),
    second authorization issued and spent 2026-07-28, executed once
    → terminal 4 H2_MEASUREMENT_EXECUTION_INVALID; 1/4 ordered runs started
    → still no capture, no seal, equivalence still unproven
    → but the archive is complete and the independent verifier accepts it
      (valid, complete), and every 0a5dffe9 controller defect is closed
    → cause is inside the child: it re-applies its ingress environment
      contract after importing the eval stack, and cv2 4.11.0 rewrites the
      environment on import — fails on any host with OpenCV installed
  ⇒ items 4–5 satisfied-then-void a second time; both authorizations
    permanently spent; next = execution-and-archive-verifier repair
    (child ingress authority + verifier execution domain) on a further head,
    gated behind a non-evidence full run that consumes no authorization
  → no H0 re-entry, no substrate, no B1/O1 activation follows

H0 registration-v3 universe-completeness contract line
  additive schema/validator over frozen consumer universe + trace-v2 envelope
  → H0_REGISTRATION_V3_CONTRACT_SEALABLE
  ∧ identity h0_gctm_guarantee_registration_v3
  ∧ guarantee_class universe_completeness
  ∧ consumer objects runtime_candidate_universe / runtime_event_membership
  ∧ event_universe_sidecar binds frozen envelope fields only
  ∧ trace-v2 ABI change not required
  ∧ actual guarantee / substrate / compatibility / re-entry remain false
  → permits owner consideration of future H0 re-entry design;
    not actual guarantee registration; no B1/O1 activation
```

For any bridge-runtime consumer, the required compatibility check is limited to
runtime-observable inputs:

$$
R_{\mathrm{consumer,obs}} \subseteq \Gamma_{\mathrm{H0}}.
$$

H0's guarantee envelope does not establish GCTM model adequacy, physical-time
mapping, likelihood/ranking value, or L2 insertion semantics. Those remain
GCTM/B1-owned blockers. The required consumer registration and compatibility
verdict are defined in [H0→GCTM consumer compatibility requirements](../../modules/semantic/research/h0_gctm_consumer_compatibility_requirements_20260718.md).
A full-faithful H0 terminal makes a separately declared B1 consumer study a
candidate; it is never a direct handoff.

**Sources for the blockers and satisfied contract binding:** [H0 terminal boundary](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md#7-terminal-and-post-terminal-boundary) · [GCTM activation boundary](closed/gap_conditioned_stochastic_transition_model_task.md#activation-boundary) · [B1 activation gate](gctm_b1_runtime_grounded_offline_attribution_task.md#activation-gate) · [registry §7 score-layer binding](../contracts/claim_state_registry.md#7-架構缺口顯式化而不是假裝可編排).

---

## 統一工作／交接匯報

任何會改變 thread 狀態、WIP 指向或 handoff 的更新，先用下列四段
匯報；**先說現有工作怎麼收尾，再談是否另開新題**。

```text
全域 WIP／既有工作帳
- sole active：<以 module TODO 為準；可為 none>
- active but non-WIP：<工程 follow-up / 維護 / 治理卡>
- parked：<暫停中的既有母線>
- closed：<本輪剛結案的母線>

交接帳（只記 direct handoff disposition）
來源／terminal → direct receiver（或 no receiver） → 被交接的工作／class → 現在狀態

跨 thread consequence（非 handoff）
<terminal 對其他既有卡造成的 constraint / closure；既有卡仍留在全域工作帳>

本輪唯一具體工作
- thread：
- 工作：
- 完成條件：

不在本輪
- 未授權的新研究、擴 class、或候選方向；不得寫成既有 handoff 或 current step。
```

**判讀規則：**

- `closed` 的交接帳只記 terminal **直接**交出的工作或 class：交給 direct
  receiver，或 **no receiver / no continuation**；不能只寫抽象的「下一步」。
- 已存在於其他卡的工程債、maintenance、parked branch 一律留在「全域 WIP／
  既有工作帳」。terminal 對它們的影響只能寫為 **cross-thread consequence**，
  不得冒充為 handoff。
- `active` 不等於 sole-active。active 卡必須標出工作類別與 WIP 角色，
  讓工程收尾、維護、治理不會被誤讀為科學主線。
- 同一 charter 內替換 expected-state lease 或 probe 不是另開主線；不得因此更新 registry 或製造 handoff。
- `parked` 的既有分支只記 pause／resume 條件；它不是本輪可執行工作。
- 新研究候選只有在 owner 明示要排下一個 charter 時才另列；在此之前不進
  交接帳、Active 表或 `Current step`。

**卡片標記：** proposed／active／parked 卡都必須以 frontmatter 補充
`work-class: mainline-study | engineering-follow-up | maintenance | governance`
與 `wip-role: sole-active | non-wip | parked`。**proposed 一律必填
`work-class` 並使用 `wip-role: non-wip`，直到 owner activation；**不得在
proposed 階段使用 `sole-active` 或 `parked`。`TODO.md` 仍是 sole-active 的
唯一權威；這些標記只消除導航歧義。

---

## Lifecycle（thread 狀態機）

狀態以 frontmatter `doc-status` + 本 README 分表為準。  
**進行中**卡在 `threads/`；**結案**卡移入 [`closed/`](closed/)。關閉 ≠ 刪檔。  
只有不再需要 threads 導航的 one-shot 才進 `docs/archive/`。

```text
proposed  →  任務已成文、未開跑 / 未授權 sole-active；`wip-role: non-wip` →  threads/
active    →  進行中（可多張；sole active 以 module TODO） →  threads/
parked    →  有意暫停；不佔 WIP；可再激活                 →  threads/
closed    →  結案；terminal + handoff 導航               →  threads/closed/
archived  →  不再當現況導航                               →  docs/archive/
```

| status | 本 README 表 | 檔案位置 | frontmatter |
|:--|:--|:--|:--|
| `proposed` | **Proposed** | `threads/` | `doc-status: proposed` |
| `active` | **Active** | `threads/` | `doc-status: active` |
| `parked` | **Parked** | `threads/` | `doc-status: parked` |
| `closed` | **Closed** | **`threads/closed/`** | `doc-status: closed` + `closed: YYYY-MM-DD` |
| `archived` | Archive 附註或移除 | `docs/archive/` | `doc-status: archived` |

**歷史狀態放哪：**

| 層 | 放什麼 |
|:--|:--|
| 單張 thread 的 `## History` | 逐步時間線（開線、PR、gate、handoff） |
| **`threads/closed/<card>.md`** | 結案卡本體（Final status · terminal · History） |
| **本 README Closed 表** | 全域索引：terminal one-liner + close date |
| module research / evidence / ledger | 可引用事實（thread **不**當第二真相） |

---

## When to create a thread

```text
跨 2 個以上文檔家 / 子類
或連續 3 步以上
或會產生可引用數字 / policy / hook / audit
→ 建 thread
```

**不建：** 單次 bug fix、單次 ablation、一步能在 module research 結案的工作。
沒有可重用 evidence 的 disposable probe 也不建 thread / research note；直接從 `Current step` 或 PR 移除即可。

**Same-PR：** 新增 / 改狀態 thread → 更新本 README 對應表（Proposed / Active / Parked / Closed）。

---

## How to close a thread

先完成 [通用研究收尾卡](../../../DEVELOPMENT.md#研究收尾卡)；本節只補 thread 特有的
frontmatter、搬移與索引要求。

只有整個 charter terminal 才關 thread。丟棄 probe 或替換 expected-state lease 不走此流程。

關閉時 **同一變更** 做齊：

1. **frontmatter**
   - `doc-status: closed`
   - `closed: YYYY-MM-DD`
   - 可選：`closed-verdict: <token>`（如 `A1_ACCEPTED_WITH_LIMITS` · `V5` · `SUPERSEDED`）
2. **thread body**
   - 頂部 one-liner 標 **CLOSED** / **SUPERSEDED** / **PARKED→CLOSED**
   - 補或改寫 `## Final status`（或等價 terminal 表）：verdict · **direct handoff（direct receiver 或 no receiver）** · cross-thread consequence（若有，須明標非 handoff）· preset 未改 · 不再授權 next work
   - `## History` 末行記關閉事件 + 指向事實 note / PR / packet
   - `## Current step` 改為 **none — closed**（勿留假 next step）
3. **移動檔案**
   - `git mv docs/research/threads/<card>.md docs/research/threads/closed/<card>.md`
   - 修正相對連結深度（`../eval` → `../../eval`；`../../modules` → `../../../modules`；進行中 sibling → `../<active>.md`）
   - 全庫把指向舊路徑的 link 改成 `threads/closed/<card>.md`
4. **本 README**
   - 從 Proposed / Active / Parked **移出**
   - 加入 **Closed** 表一行：連結 `closed/<card>.md` · terminal one-liner · **direct handoff disposition** · close date；cross-thread consequence 另欄記錄
5. **下游**
   - module `TODO.md` sole-active 若仍指此 thread → 改指 handoff 目標或清掉
   - 若有 superseding thread → 雙方交叉連結
6. **不要**
   - 刪 thread 檔（要用 `archived` + `docs/archive/`）
   - 把結案長表貼進 thread（寫進 module research / ledger）
   - 結案卡繼續留在 `threads/` 根目錄

**Park（非關閉）：** 檔案**仍在** `threads/`；`doc-status: parked` + 本 README Parked 表；History 記 pause reason + 再開條件。

**Archive（很少用）：** 不再需要 threads 導航時 → `docs/archive/` + 本 README Closed 改為 archive 指標或移除。

---

## Proposed threads

| Thread | Work class / WIP role | Current concrete work / Intent | Owner |
|:--|:--|:--|:--|
| [gctm_b1_runtime_grounded_offline_attribution_task.md](gctm_b1_runtime_grounded_offline_attribution_task.md) | mainline-study · **non-WIP** | Proposed task charter — GCTM B1 runtime-grounded offline attribution and score-ranking evaluation (split from B1/O1 synthesis §37) | semantic |
| [gctm_o1_online_intervention_efficacy_task.md](gctm_o1_online_intervention_efficacy_task.md) | mainline-study · **non-WIP** | Proposed task charter — GCTM O1 online score intervention and system-efficacy evaluation (split from B1/O1 synthesis §37) | semantic |
| [h2_behavioral_identity_capture_task.md](h2_behavioral_identity_capture_task.md) | mainline-study · **non-WIP** | H2 bridge-decision capture under behavioral runtime identity (successor to H0's identity layer; five spent `S` unchanged). Four owner decisions **accepted 2026-07-25** ([#286](https://github.com/raylei50653/saccade/issues/286)); S1–S3 landed, S4 Phase-A controller implemented, contract-tested, reviewed and merged. Two authorized Phase-A invocations are **spent with zero capture**: `0a5dffe9…` (2026-07-27) → terminal 1, 0/4 runs, archive refused; `7646f421…` (2026-07-28, after the controller repair) → terminal 4, 1/4 runs, archive complete and verifier-accepted. Cause was a child environment-validation ordering defect plus a host-coupled archive verifier; the **execution-and-archive-verifier repair landed 2026-07-29** (`cc02a0b0…`, `7cae46d8…`) and authorizes nothing. The **rehearsal harness landed 2026-07-29** (`h2_rehearse_measurement.py`, with a canonical-corpus provenance guard committed ahead of it) and **ran once, at `ba40b3f8`, and failed — spending no authorization**: terminal `H2_MEASUREMENT_EXECUTION_INVALID`, `00_capture_off`'s child refused, `01/02/03` never started. Every isolation guarantee held, so this is a gate failure, not a harness failure. Cause = the H2 fixed-A5 invocation adapter passes only `--sequences`/`--output`, so the preset-driven `configure_runtime_env` rewrites the frozen A5 environment; H0 had already solved this through `EVALUATOR_ARGV_PREFIX`. Acceptance items 4–5 and `F` were rebuilt and independently verified at `ba40b3f8` (71/71, 69/69) and are historically valid there, stale for every descendant head. The repair landed the same day (`C_reg → B → A → C_close`): the child now sends the frozen A5 choices and the no-metrics boundary through the authoritative parser surface, and the harness derives run completion from the child's lifecycle record. Current work = rebuild items 4–5 and `F` at the repair head, then rehearse once more; both authorizations remain permanently spent — the owner must separately issue a third | semantic |

## Active threads

| Thread | Work class / WIP role | Current concrete work | Owner |
|:--|:--|:--|:--|
| [association_recovery_registry_20260709.md](association_recovery_registry_20260709.md) | maintenance · **non-WIP** | Keep R/H ownership and path-health registry current | semantic |
| [doc_structure_o15_followup_20260709.md](doc_structure_o15_followup_20260709.md) | governance · **non-WIP** | Pay down research-index debt; optional structure checks remain non-blocking | ownership |

## Parked threads

| Thread | Work class / WIP role | Pause / resume boundary | Owner |
|:--|:--|:--|:--|
| [score_temporal_to_stable_domain_20260712.md](score_temporal_to_stable_domain_20260712.md) | mainline-study · **parked** | `R1_FAITHFUL` closed (it is what made runtime coordinates auditable); discrete-\(M\) follow-on **reclassified as a score-ranking feature, not a gate** → parked unsealed. | semantic |
| [gt_support_morphology_20260711.md](gt_support_morphology_20260711.md) | mainline-study · **parked** | Score continuation is closed for Door 0's tested class; gate direction remains parked pending explicit re-charter and WIP authorization | semantic |
| [occ_exit_audit_20260709.md](occ_exit_audit_20260709.md) | mainline-study · **parked** | WP1–WP3 complete; waits for a RegionAsset producer/intervention consumer after the assetization gate | semantic |

## Closed threads（檔案在 [`closed/`](closed/)）

| Thread | Closed | Terminal (one-line) | Direct handoff disposition | Cross-thread consequence (not handoff) | Owner |
|:--|:--|:--|:--|:--|:--|
| [h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md](closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md) | 2026-07-24 | **`H0_REGISTRATION_V3_CONTRACT_SEALABLE`** — additive registration contract `h0_gctm_guarantee_registration_v3` for `universe_completeness` over `gctm_runtime_native_candidate_universe_v1`; no trace-v2 ABI delta | **no actual-guarantee handoff** — may only inform a separately scheduled H0 re-entry / baseline-capture design | seals registration-v3 contract only; does **not** establish actual H0 guarantee, substrate, compatibility, capture, or B1 activation; both runtime gates remain `missing` | semantic |
| [gctm_runtime_native_candidate_universe_task.md](closed/gctm_runtime_native_candidate_universe_task.md) | 2026-07-24 | **`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`** — freezes runtime consumer identity `gctm_runtime_native_candidate_universe_v1` with pre-score lost-centric event/candidate keys, composition/completeness semantics, and registration-v3 requirements-only surface | **no H0 implementation handoff** — may only inform a separately chartered minimal registration-v3 delta | closes consumer-owned universe/membership gap from the static audit; does **not** establish H0 completeness, substrate, compatibility, or B1 activation; both runtime gates remain `missing` | semantic |
| [h0_gctm_interface_static_feasibility_audit_20260723.md](closed/h0_gctm_interface_static_feasibility_audit_20260723.md) | 2026-07-23 | **`H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT`** — path/hash-frozen static audit; physical gap/residual/covariance path is structurally expressible, but D1's synthetic candidate universe and registration-v2's missing native-universe completeness binding stop the end-to-end evidence path | **no receiver** — owner must separately choose a new delta or re-charter; no implementation is auto-authorized | unchanged-interface H0 capture/re-entry is ineligible; both runtime gates remain independently `missing`; no B1/O1/GCTM_D1 state change; consumer re-charter later closed 2026-07-24 without reopening this terminal | semantic |
| [gctm_d1_substrate_agnostic_ranking_diagnostic_task.md](closed/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md) | 2026-07-23 | **`GCTM_D1_INTERFACE_READY` owner-accepted** (mechanical three-way terminal; canonical execution `gctm_d1_canonical_execution_20260723`; [execution witness](../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/execution_witness.json) · [terminal acceptance](../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/terminal_acceptance.json)) · diagnostic-only; I1–I12 pass; consumer interface complete | **no receiver / no continuation** — interface-ready ≠ runtime-compatible; unlocks no B1/O1/H0 | runtime compatibility gates remain `missing`; `active_wip: []`; `decision_relevant_candidates: []`; H0 re-entry still unauthorized; B1/O1 remain proposed behind runtime substrate | semantic |
| [gap_conditioned_stochastic_transition_model_task.md](closed/gap_conditioned_stochastic_transition_model_task.md) | 2026-07-23 | **`GCTM_MODEL_SPEC_SEALABLE` owner-accepted** (ordered terminal 5; selected by WP-A8 terminal review, [checklist artifact](../models/gap_conditioned_stochastic_transition_terminal_review_v1.md)) · diagnostic-only A-layer model-spec seal · D1 §2–§8 + D2 L1–L5 frozen | **no receiver / no continuation** — the seal grants no B1/O1/online/production authority; B1 and O1 need separate activation | H0 state unchanged (no faithful capture, no fidelity edge, empty guarantee set); the at-close L2 contract absence is **superseded** by the 2026-07-23 owner-accepted [v1 binding](../contracts/score_ranking_evidence_contract.md), but substrate still does not inherit and all other B1/O1 gates remain | semantic |
| [bridge_frozen_evidence_o0_routing_20260716.md](closed/bridge_frozen_evidence_o0_routing_20260716.md) | 2026-07-20 | **route 1 `H0_PROVENANCE_INVALID` owner-accepted**（[#209](https://github.com/raylei50653/saccade/issues/209)，2026-07-19）· H0 = CLOSED（diagnostic-only）· controller retry / Phase B forbidden | **no receiver / no continuation** — NEXT = none automatically | GCTM [#175](https://github.com/raylei50653/saccade/issues/175) was parked **at close time**（activation required a separate owner scheduling decision）——**superseded**: that decision was taken 2026-07-22 and GCTM then closed 2026-07-23 at `GCTM_MODEL_SPEC_SEALABLE`（see its Closed row above）; the O0 card's *Final status* keeps the at-close snapshot. Provenance gap becomes permanent registry `open_limits`（unchanged） | semantic |
| [gap_conditioned_probabilistic_motion_probe_20260711.md](closed/gap_conditioned_probabilistic_motion_probe_20260711.md) | 2026-07-13 | **`V5 ACCEPTED_WITH_LIMITS`** · D0 follow-up closed at `T2_PROXY_UNFAITHFUL` | **no receiver / no continuation** | H0 observability is an independent proposed task, not a handoff | semantic |
| [runtime_faithful_safe_domain_20260712.md](closed/runtime_faithful_safe_domain_20260712.md) | 2026-07-13 | **S0 `S0_UNDECIDABLE` ACCEPTED** · V7 has no offline-safe grid point, so runtime transfer is not assessed · [PR #152](https://github.com/raylei50653/saccade/pull/152) | **no receiver / no continuation** — wider runtime join requires a new decision-relevance and O0 decision | offline partial-order state unchanged; runtime transfer unaccepted and closure remains inadmissible | semantic |
| [ambiguous_band_ranking_power_probe_20260712.md](closed/ambiguous_band_ranking_power_probe_20260712.md) | 2026-07-12 | **T2 `NO_USABLE_RANKING_POWER_IN_CLASS` ACCEPTED**（12-member class-scoped；step ⑤ 生效；step ④ 未開）· [PR #135](https://github.com/raylei50653/saccade/pull/135) seal／[PR #136](https://github.com/raylei50653/saccade/pull/136) acceptance | tested 12-member score class → **no receiver / no continuation** | morphology score branch is closed for this class; its gate branch remains parked in its own card | semantic |
| [safe_region_assetization_20260710.md](closed/safe_region_assetization_20260710.md) | 2026-07-11 | **A1 CLOSED**（`A1_ACCEPTED_WITH_LIMITS`）· R2–R4 fail-closed | → [gt-support morphology](gt_support_morphology_20260711.md) (parked) | — | semantic |
| [composition_grammar_coverage_program_20260710.md](closed/composition_grammar_coverage_program_20260710.md) | 2026-07-10 | **SUPERSEDED** · coverage map absorbed into assetization R2–R4 · no C1–C6 execution | → assetization closed line; **no further continuation** | — | semantic |
| [composition_grammar_safe_region.md](closed/composition_grammar_safe_region.md) | 2026-07-10 | **CLOSED A0 baseline** · T0-A/B/R1 · radius≥1=0 · terminal B | → assetization closed line | — | semantic |
| [m_b1_online_hook_20260709.md](closed/m_b1_online_hook_20260709.md) | 2026-07-10 | **CLOSED** · S1+S2 Q4.5 B complete | → composition T0/A0 closed line; ranking deferred, **no active receiver** | — | semantic |

> 目錄索引見 [closed/README.md](closed/README.md)。數字與 claim 在 module research / ledger。

---

## File shape

### Active / proposed / parked

```md
---
doc-status: proposed | active | parked
doc-promotion: navigation-only; not evidence
owner-module: <module|cross|ownership>
work-class: mainline-study | engineering-follow-up | maintenance | governance
wip-role: sole-active | non-wip | parked
created: YYYY-MM-DD
---

# <short title>

## Status                 # work class + WIP role; proposed=non-wip; sole-active links to module TODO
## Current boundary
## Expected state (lease) # target only;not accepted state;replaceable
## Commit point           # when owner reviews whether state changed
## Discard when           # disconfirm / expiry / replacement condition
## Read first
## Artifacts
## Current step           # disposable probe(s);may be replaced without close / registry update
## Acceptance
## Must not
## History
```

### Closed（結案形）

```md
---
doc-status: closed
doc-promotion: navigation-only; not evidence
owner-module: <module|cross|ownership>
created: YYYY-MM-DD
closed: YYYY-MM-DD
closed-verdict: <optional token>
---

# <short title>

> **One-line (CLOSED):** …

## Final status          # terminal table · direct handoff (receiver or no receiver) · cross-thread consequence (if any) · preset unchanged
## Read first            # 結案後仍需要的入口
## Final evidence state  # optional；只放 pointer / 一句，不放長表
## Must not              # 關閉後仍禁止的 claim
## History               # 含關閉事件
```

閉合後可刪或折疊 `Current step` / `Acceptance` 中未完成 checklist；**不可**刪掉使 handoff 斷鏈的 Read first / History。

**命名：** `<topic>_<YYYYMMDD>.md`（母線主題，不是每步一個檔）。

---

## Role split

| 層 | 負責 |
|:--|:--|
| [DEVELOPMENT.md](../../../DEVELOPMENT.md) | D0–D4 action cards；模組入口不承載 live status |
| module `TODO.md` | **mainline-charter WIP 鎖**（sole active + links）；不放 expected state / probe / 任務敘事 |
| module / research README | 局部入口與檔案索引 |
| **threads/** | 進行中母線（proposed · active · parked） |
| Expected state / Current step | charter 內 planning lease / probe；可替換，不是 accepted state |
| **threads/closed/** | 已結案 thread 檔案家；本 README Closed 表 = 索引 |
| **implementation PR** | engineering delivery：branch tips、SHA、CI、changed files、review findings |
| research contracts / notes | normative inputs · claim boundaries · research acceptance · stage authorization |
| ledger / report_data / out/ | 事實與數字升格 |

```text
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
thread closed ≠ evidence promoted
```
