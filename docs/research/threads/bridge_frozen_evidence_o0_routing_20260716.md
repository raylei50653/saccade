---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: sole-active
created: 2026-07-16
---

# O0 — bridge frozen-evidence routing（薄編排 charter）

## Status

**ACTIVE · sole-active**（WIP 鎖以 [semantic TODO](../../modules/semantic/TODO.md) 為準）。
本卡是 O0 對 bridge-fidelity 線既有 terminal 狀態的**薄編排**：一個 sole-active
decision、一張薄狀態地圖、一張 exhaustive terminal routing map。本卡**不擁有任何
verdict、不放任何數字**；所有 label 逐字照 owner doc（link-don't-relabel，
[C5.1](../../ownership/doc_structure_contract.md)）。

本輪輸入：contract v1.2 §20.9（PR #169）· [reconciled flagship map](../../modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md) ·
[terminal-slot schema v0](../../ownership/terminal_slot_fixtures.yaml)（[ADR 020](../../decisions/020-doc-lifecycle-new-nogo.md)）·
[old-flagship per-study inventory](../../ownership/old_flagship_per_study_inventory.yaml)。

本輪明確不做：新 capture、新統計公式、ε-bound 定義、bridge 實驗設計、舊資產
disposal、contract / registry 語義改寫。

## Thin status map（labels only；數字家＝reconciled map 與各 owner doc）

空間符號與 typed-failure 語義家＝
[contract v1.2 §20.9](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)。

```text
offline trajectory ─R_offline─> s0
      │ J_v（partial exact-key join；只在 M^evt 上存在）
      ▼
runtime trajectory ─R_kernel──> bdist        κ_D0：FALSIFIED on M^evt [D0]

captured causal state ─C_R1─> replayed bdist  κ_R1：R1_FAITHFUL（scoped）[R1]

U^evt = M^evt ⊍ G^evt ⊍ E^evt                 partition 由 [D0] 擁有
```

Sealed per-study terminals（labels verbatim；軸值照 reconciled map 的 slot；
`study_id` 照 inventory）：

| study_id | owner terminal | claim_verdict · decision_outcome | owner doc |
|:--|:--|:--|:--|
| `kappa_d0_proxy_fidelity` | `T2_PROXY_UNFAITHFUL` | FALSIFIED · NOT_ASSESSED | [D0 results](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md) |
| `kappa_r1_runtime_replay` | `R1_FAITHFUL` | VERIFIED · NOT_ASSESSED | [R1 results](../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md) |
| `rho_s0_safe_axis_transfer` | `S0_UNDECIDABLE` | NOT_IDENTIFIABLE · NOT_ASSESSED | [S0 results](../../modules/semantic/research/closed/safe_domain_runtime_transfer_results_20260713.md) |
| `ek0_exact_key_recoverability` | `EK0_NO_RECOVERABLE_SUPPORT` | FALSIFIED · NOT_ASSESSED | [EK0 results](../../modules/semantic/research/frozen_packet_exact_key_recoverability_results_20260713.md) |
| `p0_decision_path_identifiability` | `P0_CAPTURE_SEMANTICS_UNVERIFIABLE` | NOT_IDENTIFIABLE · NOT_ASSESSED | [registry](../contracts/claim_state_registry.md) |
| `door0_t2_ranking_power` | `T2_NO_USABLE_RANKING_POWER_IN_CLASS` | FALSIFIED · NET_NEGATIVE | [T2 results](../../modules/semantic/research/door0_ranking_probe_results_20260712.md) |

§20.9 typing（routing 註記，非新 verdict）：κ_D0 的否證屬
**transport-noncommuting**；P0 屬 **not-identifiable**；EK0 的否證位於
**assignment / keying** 軸。三者不得互換（§20.9：keying ≠ not-identifiable；
non-commuting ≠ not-exchangeable）。S0 / T2 不指派 §20.9 failure type。

Live（依 ADR 020 §S1 不發 slot）：discrete-\(M\)
[parked-unsealed](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md)；
H0 [proposed-unsealed](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
——其 pre-terminal 狀態不得被寫成 sealed terminal。

## Current boundary

在 frozen bridge evidence 上：offline→runtime transport 在 matched domain 已被
否證（κ_D0）；packet 能否自證其 policy＝not identifiable（P0，cause＝provenance
incompleteness）；frozen packet 無 exact-key recoverable support（EK0）。本線在
frozen evidence 上沒有第三種讀法：**要嘛取得可自證 provenance 的新證據基礎
（H0），要嘛把 provenance 缺口留在帳上作為合法終局**。

## Sole-active decision（O0）

依 [registry §8](../contracts/claim_state_registry.md)：合法候選集恰有一個成員＝
H0（§4.3 dependency-typed，非 inadmissibility）。O0 決定：**取 H0，授予 WIP 鎖**。
範圍嚴格＝H0 declaration 已宣告的 pre-seal freeze artifact（target＝**m**，
Amendment 5）→ owner seal。

- **seal 仍是唯一權威事件**（declaration §8 的 literal `SEALED` review）；本
  charter 不是 seal，不授權任何 capture / 執行。
- 未 seal 前：execution prohibited；`H0_PRESEAL_COVERAGE_INCOMPLETE`＝engineering
  status，禁 seal 禁 capture，**不是** H0 terminal、不供給任何 result。
- registry 不因本決定改寫（無 object state 轉移；registry §9）。

## Expected state (lease)

pre-seal freeze artifact 對 **m** 完成 → owner literal `SEALED` → Phase A（單序列
preflight，只可能出 negative terminals）→ Phase B（七序列）→ H0 ordered
terminal → owner acceptance 時同 PR 回寫 registry（§9；不在本卡先寫）。

## Commit point

兩個：① owner 在 declaration §8 記 literal `SEALED`；② owner 接受某個 H0
terminal。只有 ② 造成 object state 轉移。

## Terminal routing map（exhaustive；無第三扇門，contract §20.7）

Routing 對象＝本 charter 的 decision question：「bridge-fidelity 線是否取得可自證
provenance 的證據基礎？」每格語義逐字歸
[H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
的 **Amendment 2 §A2.4 sealed terminal partition**（先滿足者為準，不得後移；
`H0_CAPTURE_PARTIAL` 已被 A2.4 退役——sealed invocation 下 there is no
partial-capture reinterpretation；後續 amendments 依其 supersession 條款優先，
如 Amendment 3 supersedes A2.1–A2.4 where they conflict），本表只做
charter-level 處置：

| # | 事件（labels verbatim） | charter 處置 | 下游 |
|:--|:--|:--|:--|
| 0 | owner 不 seal／Discard 條件成立 | charter CLOSED（declined） | 候選集回空；provenance 缺口以 registry `open_limits` 形式**永久留帳＝合法終局**，非待辦 |
| 0′ | `H0_PRESEAL_COVERAGE_INCOMPLETE` 持續 | 非 terminal；pre-seal engineering status，禁 seal 禁 capture | 修 instrumentation，或走 route 0 |
| 1 | `H0_PROVENANCE_INVALID` | charter CLOSED（diagnostic-only） | 同 route 0 下游；重進須 amendment＋owner reseal |
| 2 | `H0_EXECUTION_INVALID` | charter CLOSED（diagnostic-only；不得重讀為 partial capture） | 同上 |
| 3 | `H0_CAPTURE_PERTURBS_POLICY` | charter CLOSED（diagnostic-only） | 同上 |
| 4 | `H0_PACKET_INVALID` | charter CLOSED（diagnostic-only） | 同上 |
| 5 | `H0_FULL_COMMIT_CAPTURE_FAITHFUL` ＋ owner acceptance | charter CLOSED（enabler delivered） | **B1 consumer-faithful operating-curve study 僅成為 candidate，絕非 direct handoff**（declaration §7）；任何 re-audit 須新 §20.2 宣告 |

為 exhaustiveness 必須顯式排除的兩條「母體層」路（皆不由本卡授權）：

- **representation-domain expansion**（把 `G^evt` / `E^evt` 類事件變
  representable）：EK0 已否證 frozen artifacts 的 exact-key recoverable support
  ⇒ 此路必然＝新 capture／新 identity observability，目前**不存在宣告** ⇒ 不在
  候選集；走它＝新 §20.2 宣告＋新 decision relevance。
- **quantifier downgrade 到「commutes on representable subdomain」：不可用**——
  κ_D0 在 matched（representable）domain 上本身已 FALSIFIED（[reconciled map](../../modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md)
  Discrepancies）。殘餘的 downgrade 只剩 registry 既有 inadmissibility（s0 永不
  代表 production `bdist`）——那是既有狀態，不是新動作。

⇒ 全部未來狀態 ∈ { route 0／0′，routes 1–4（typed negative），route 5
（positive）}；無第三扇門。

## Discard when

owner 明示不 seal、H0 declaration 被 supersede、或 O0 重新排程候選集（route 0）。
Discard 走 threads README 收尾流程，terminal＝declined，缺口留帳。

## Read first

- [H0 declaration（含 Amendments 1–5）](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
- [claim-state registry（§8 候選集）](../contracts/claim_state_registry.md)
- [reconciled flagship map（數字家）](../../modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md)
- [contract v1.2 §20.7 / §20.8 / §20.9](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
- [ADR 020](../../decisions/020-doc-lifecycle-new-nogo.md) ＋ [terminal-slot fixtures](../../ownership/terminal_slot_fixtures.yaml)
- [old-flagship per-study inventory](../../ownership/old_flagship_per_study_inventory.yaml)
- [P0 declaration ＋ Correction 1](../../modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md)

## Artifacts

本卡＝orchestration only；**永不產生 evidence artifact**。

## Current step

H0 pre-seal freeze artifact（對 **m**）完成度檢查 → owner seal review。
（engineering、non-evidence；可替換、可丟棄。）

## Acceptance

route 5 或任一 typed negative route（1–4，含 route 0 declined）被 owner 接受 →
本卡照 threads README 收尾流程 close。

## Must not

- seal 前執行任何 H0 capture / Phase A / Phase B；**PR merge ≠ seal**。
- 把 H0 pre-terminal 狀態寫成 sealed terminal。
- 在本卡放數字、結果表、統計理由（數字家＝reconciled map / owner docs）。
- 改寫 contract / registry 語義；本卡只**消費** §20.9 typing。
- 把 B1 寫成 handoff；把 route 0 的永久留帳寫成待辦。
- 用 s0 代表 production `bdist`（registry inadmissibility）。
- 從本卡改 preset、production 行為或 no-go registry。

## History

- 2026-07-16: Opened as O0 thin orchestration over existing terminal state.
  Inputs: contract v1.2 §20.9 (PR #169) · reconciled flagship map ·
  terminal-slot schema v0 (ADR 020) · old-flagship inventory (#168).
  O0 took the sole registry candidate (H0 pre-seal, target m) as sole-active;
  thin map and exhaustive terminal routing recorded. No execution authorized;
  owner seal remains the single authoritative event.
- 2026-07-16: Review fix (blocking): the routing map had reintroduced the
  retired `H0_CAPTURE_PARTIAL` and cited the historical §7/A3 partition.
  Authority corrected to Amendment 2 §A2.4 (four negative terminals, then
  `H0_FULL_COMMIT_CAPTURE_FAITHFUL`; later amendments supersede on conflict);
  routes renumbered to 1–4 negative / 5 positive.
