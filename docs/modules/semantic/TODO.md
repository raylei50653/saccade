# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **無 active**
- H0 R4 repair closed at `H0_R4_REPAIR_QUALIFIED_SEALABLE`
  （`h0_authority_overlay_runtime_binding_split_v1` / Amendment 10）；WIP released.
  Seal PR #277 landed exact S=`a76efffa…`. See
  [repair evidence](research/evidence/h0_r4_authority_overlay_runtime_binding_split_20260724/).
- **H0-R4 Phase-A executed once under sealed S（2026-07-24）** — owner-accepted
  truthful-negative terminal only（PR #279 merge `55d2da47…`）:
  - `I=2a233387…` / `F=ced4a4cc…` / `S=a76efffa…`
  - authorization `h0_r4_phase_a_exactly_once_authorization_20260724`（Issue #278；consumed）
  - controller `result=provenance_invalid` → `H0_PROVENANCE_INVALID`; verifier `valid=true`
  - evidence:
    [h0_phase_a_2a233387…](research/evidence/h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4/)
    + [execution witness](research/evidence/h0_r4_phase_a_execution_witness_20260724/)
  - S4 permanently spent; no faithful capture; no actual H0 guarantee; Phase B forbidden;
    no future reentry authorization from R4 alone.
  Detail: [claim-state registry](../../research/contracts/claim_state_registry.md).
- H0 closure baseline（**five** mechanical `H0_PROVENANCE_INVALID` terminals including
  R5 S5; no faithful capture; no actual H0 guarantee; Phase B forbidden）:
  state fact-owner 見
  [claim-state registry](../../research/contracts/claim_state_registry.md)。
- **H0-R5 Repair closed at `H0_R5_ATTESTATION_QUALIFIED_SEALABLE`**
  （`h0_extension_plugin_runtime_attestation_closure_v1` / Issue #280）。
  Controlled-host qualification passed; later tool_runtime independent-expansion parity
  repair landed at `I=524f7e3b…`. See
  [repair evidence](research/evidence/h0_r5_extension_plugin_attestation_closure_20260724/).
- **H0-R5 Phase-A executed once under sealed S（2026-07-25）** — mechanical
  truthful-negative terminal:
  - `I=524f7e3b…` / `F=6e425dc6…` / `S=6fdb060c…`
  - authorization `h0_r5_phase_a_exactly_once_authorization_20260725`（Issue #283；consumed）
  - controller `result=provenance_invalid` → `H0_PROVENANCE_INVALID`; verifier `valid=true`
  - failure: `extension/plugin load is absent from runtime attestation` at `extension_load`
  - evidence:
    [h0_phase_a_524f7e3b…](research/evidence/h0_phase_a_524f7e3b88f73bc366d467d53a2c393a7d3ba937/)
    + [execution witness](research/evidence/h0_r5_phase_a_execution_witness_20260725/)
  - S5 permanently spent; no faithful capture; no actual H0 guarantee; Phase B forbidden;
    no future reentry authorization from R5 alone.
  Detail: [claim-state registry](../../research/contracts/claim_state_registry.md).
- 2026-07-24 consumer re-charter closed at
  `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`（`gctm_runtime_native_candidate_universe_v1`）；
  WIP released at terminal；**not** H0 implementation authority. See
  [closed charter](../../research/threads/closed/gctm_runtime_native_candidate_universe_task.md).
- 2026-07-24 registration-v3 closed at
  `H0_REGISTRATION_V3_CONTRACT_SEALABLE`（`h0_gctm_guarantee_registration_v3` /
  `quantity.h0_native_universe_completeness_registration`）；
  WIP released at terminal；**not** actual guarantee / capture / re-entry.
  H0-R4 Phase-A did **not** establish an actual registration-v3 guarantee.
  See [closed charter](../../research/threads/closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md).

## Proposed（non-WIP）

- **H0 route-5 B1 — runtime-grounded consumer-faithful operating curve** —
  `H0_ROUTE5_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it is distinct
  from and coexists with `GCTM_B1` → [machine identity decision](../../research/contracts/gctm_b1_slot_identity_decision_v1.json)
- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** —
  `GCTM_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it does not alias
  or supersede `H0_ROUTE5_B1` → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- **GCTM O1 — online score intervention and system-efficacy evaluation** → [task charter](../../research/threads/gctm_o1_online_intervention_efficacy_task.md)
- **H2 — bridge-decision capture under behavioral runtime identity** — `proposed`,
  non-WIP; successor to H0's **identity layer only**（declaration §0.1 boundary;
  H0 sealed history / 五個 spent `S` / permanent ledger 全部不變）；authorizes no
  capture, no `I`/`F`/`S`, no re-entry；Phase-A Layer-M controller implemented,
  contract-tested, reviewed and merged（[PR #295](https://github.com/raylei50653/saccade/pull/295)，
  merge `b2f3c23f…`）⇒ S4 code-review gate closed；**Acceptance items 4→5→6 已被
  走過兩次、兩份授權皆 spent、兩次皆 zero capture**：
  2026-07-27 在 `0a5dffe9…` → terminal 1 `H2_INPUT_MUTATED_DURING_MEASUREMENT`、
  0/4 ordered runs started、archive 被 verifier 拒收（根因＝controller self-mutation）；
  controller repair 落地後（[PR #299](https://github.com/raylei50653/saccade/pull/299)，`7646f421`）
  於該 head 重建整條鏈（run 30334080842 綠、cert `266f4b4c…` 65/65、`F64 f0d1b02e…` 51/51），
  2026-07-28 再執行一次 → terminal 4 `H2_MEASUREMENT_EXECUTION_INVALID`、
  1/4 ordered runs started、no capture、no seal、equivalence 仍 unproven。
  第二次的 archive **完整且通過 independent verifier**（valid=true、complete），
  第一次登記的四項 controller 缺陷全部關閉；新根因＝child 在 `_import_eval_stack()`
  之後重套自己的 ingress environment contract，而 cv2 4.11.0 於 import 時改寫環境
  ⇒ 任何裝 OpenCV 的 host 必失敗，與 head/binding/ruler 無關；
  items 4–5 第二次 satisfied-then-void（須在新 successor head 再重建）；
  item 6 兩份皆 consumed、永久 spent（不是重建，而是 owner 另行簽發第三份授權）；
  **execution-and-archive-verifier repair 已於 2026-07-29 落地**（commit `cc02a0b0` child
  ingress authority＋`7cae46d8` archive verifier／CI）：授權環境＝import 前的 immutable launch
  snapshot、只在 `execute_child` 判定一次、import delta 僅記 key 名稱作 diagnostic（見 declaration
  Review Correction 4）；archive 驗證改為只讀 archive 位元組，launch-time host binding 未放寬，
  corpus checker 以完整 git history 接回 CI；兩個缺陷各有一支在 `c2d1c58f` 上未改即失敗的測試。
  repair **不授權、不 seal、不恢復任何授權**。**rehearsal harness 已於 2026-07-29 落地**
  （`scripts/tools/h2_rehearse_measurement.py`，四個 commit 依 `P → B → A → C`：issuer 正規化 →
  canonical corpus provenance/admission guard → harness → governance；guard 必須先於入口，
  否則 non-squash merge 之後 harness commit 是一個可被 checkout 執行的 head）；harness 自簽
  grant、自帶可丟棄 ledger，走真正的 admission 與 consumption，產出在 repo 外且被 corpus guard 拒收。
  **2026-07-29 在 `ba40b3f8` 重建 items 4–5 與 F（71/71、69/69 獨立驗證）並首次執行 rehearsal
  ⇒ FAILED**：terminal `H2_MEASUREMENT_EXECUTION_INVALID`、`00_capture_off` 的 child 被拒、
  01/02/03 未起、**未耗任何授權**；隔離、receipt、archive binding、hygiene 與 corpus refusal
  全部成立 ⇒ gate failure 而非 harness failure。根因＝H2 fixed-A5 invocation adapter 只傳
  `--sequences`／`--output`，而 A5 preset 沒有 `double_buffer`／`detect_barrier`，於是
  `configure_runtime_env` 依其宣告的 args authority 把凍結環境改寫成 `full`／`0`；H0 早以
  `EVALUATOR_ARGV_PREFIX` 解決，H2 未沿用。items 4–5 與 F 在 `ba40b3f8` 上 historically valid、
  對每個 descendant head stale、不追溯無效。**repair 已 landed**（`C_reg → B → A → C_close`：登記失敗／harness 改由 child lifecycle
  record 判 run completion／child 以 `FIXED_EXECUTION_ARGV` 送出四個 A5 環境旗標與
  `--latency-only`／governance closeout；mutation gate、STATIC_ENV、generic parser 與 preset
  皆未動）。下一步＝在 merge head 重建 items 4–5 與 F →
  再跑一次 rehearsal → 綠了才由 owner 另發第三份授權。**此舊路徑已被
  2026-07-30 Review Correction 5 supersede**：PR #305 已落地 execution-integrity
  artifact schemas、454-key frozen RunSpec authority 與 runtime projection；
  successor 不再以 cross-host reconstruction、published coordinate/probe equality、
  Layer-P certificate 或 `F` 作 validity gate。Review Correction 8 進一步把
  `.github/workflows/runtime_identity.yml` 降為 `workflow_dispatch`-only、
  non-qualifying diagnostic；不再對 PR／`main` 自動重建。仍保留 per-execution
  build binding／extension load／identity run 與 foreign-host-independent
  archive verification。**2026-07-30 已在 `290fd0c1` 機械確認現況**（contract tests
  1158 passed、corpus checker PASS、staleness rc=0 且三個 legacy 軸仍 unresolved）：
  落地的只有 contracts 與 configuration authority，**沒有任何可執行碼讀 successor
  vocabulary**；且 `h2_execution_result_v1` 與 ruler `h2_terminal_partition.py`
  的 predicate 名／polarity／state space／result token 互不相符，`terminal: null`
  在 measurement authority 下仍 schema-legal（＝§C3.5.1 要讓它形不成的那個形狀），
  `valid` 與 `checks` 之間無任何約束。因此順序固定為
  **W1 治理 → W2 verdict algebra（Review Correction 9＋ruler＋republish，**已落地**：
  三個 rename／極性反轉／`certificate_mismatch` 被 `runtime_binding_mismatch` supersede、
  4 態 predicate 的兩條規則、六項 schema joint constraint、`valid ⇔ checks`；
  `build_failed`／`extension_load_failed` 重新入列而非併入 `runner_nonzero`）
  → W3 archive-only verifier／checksum closure（**已落地**：`verify_h2_execution.py`
  只組合不持有規則 ⇒ 命中 `plumbing_only`、**不需 ruler edit／re-pin／republish**；
  verdict 由 predicates 重算後才餵給 cross-artifact checker；
  `verify_projection=False` 是 execution integrity 與已 retire 的 environment
  reproducibility 之間那條線，並以 monkeypatch 機械證明；
  可形成的 archive 之瑕疵記為 `valid: false`，填不出必填欄位者不寫紀錄並非零退出）
  → W4 producer（**已落地,code only**:`run_h2_execution.py` 只寫三份 artifact、從不寫 verification.json／checksums.sha256;verdict 由 ruler 選出後轉錄,連「可不可以指名 terminal 4 cause」都先問 ruler;四個 ordered run 沿用 Layer-M runner ⇒ executed_surfaces 不動、不需 re-pin;Stages／Runs 為 Protocol ⇒ 控制流不需 build 即可測;W3 verifier 當場抓到 producer 的 stage-failure 藏住 mutation 的 bug）
  → W4b 綁 driver 並完成 diagnostic（修正 child 自造的 cross-member equality 後，
  4/4 runs completed、五項 predicates pass、仍只得 `diagnostic_complete`）
  → W5a–W5e closure **已完成並經 owner review ACCEPT**（verdict:
  `ACCEPT — W5 implementation/evidence closure`）：三-root 659-member
  execution-code closure、1414
  observed imports / zero unbound、alias-free immutable RunSpec namespace、successor
  canonical corpus admission 與 rehearsal witness v3、Review Correction 11／re-pin／
  republish／governance closeout，以及 final-mainline W5e diagnostic。W5e 在
  `e94407f2…` 精確執行一次：4/4 runs completed、五項 predicates pass、
  `diagnostic_complete`、`terminal: null`；independent verifier `valid: true`、七項
  checks 全綠。canonical corpus owner 以唯一理由拒收：
  `non_qualifying_diagnostic` 不是 exactly-once measurement authority。Evidence:
  `/home/ray/h2_w5e_diagnostic_20260803T132933Z` (512-member self-sealed packet;
  inventory
  `c25baf48b8425543a1f430b83b0381bcc2dc72386398d54acf22e41103bea268`)。
  Owner review 的 exception-surface observation 已納入後續 W6 projection-moving
  implementation：witness catch 面與 corpus CLI 對齊；這不改寫 W5 verdict，也不授權第二次 W5 diagnostic。W5 不消耗或產生 authorization、不是 measurement、
  不建 F/S、不 seal、不證明 equivalence，也不恢復兩份 spent
  authorization。**W5 本身沒有自動 next**。Owner 已另外選定 W6 formal-measurement
  plan：新增 successor-specific request/grant/durable-receipt envelope、v2 inner
  archive、outer verifier 與 controlled-domain corpus guard，沿用 MOT17-04-SDP／4 runs／
  454-key RunSpec；先在 final committed projection 走一次 disposable-ledger rehearsal，
  必須得到 `measurement_pass`／`terminal: null`、inner+outer valid，且 corpus 僅因 domain
  mismatch 拒收。之後 exact clean merged head 只可 emit request 並停止，等待 owner 另行
  簽發第三份 matching grant。Plan/implementation/rehearsal 都不是 authorization；正式
  success disposition 是 owner-reviewed measurement closure + canonical corpus admission，
  不含 Phase B、equivalence、I/F/S、seal 或自動 next。**W6 implementation 與 final
  disposable rehearsal 已在 feature head `f0ee5da3…` 完成**：前兩次 rehearsal review
  分別抓到 double-`runs/` custody 與 outer verifier 未獨立重驗 import/A7.6 的
  self-consistency 缺口；修後 final packet 為 `measurement_pass`／`terminal: null`、4/4、
  五 predicates pass、inner 7/7、outer 7/7、每 run 3355 imports / zero unbound，corpus
  僅以 disposable-domain mismatch 拒收。Custody
  `/home/ray/h2_w6_successor_rehearsal_20260803T163028Z`，inventory
  `bc748a7a4659ee0aaaeec7e3408cdc54cc3ec21b5a845f3a83c7ad503f8a3e41`。
  下一步不是 execution：先 review/merge，exact clean merged head 才 emit request 並停在 owner
  matching grant gate。**W7 已走完那條鏈並於 2026-08-04 被 owner 接受 ⇒ Phase A 結束。**
  在 exact clean merged head `c570dd92…` emit 一份 request、owner 另行簽發並消耗第三份授權
  （`c2f11ab1…`，永久 spent），formal measurement `h2-w6-formal-20260804T052122Z` 得到
  `measurement_pass`／`terminal: null`、4/4 runs、inner verdict 與獨立執行的 outer envelope
  verdict 各 7/7、canonical corpus `PASS (2 roots)` ⇒ v2 envelope 家族第一次被收
  （[admitted packet](research/evidence/h2_measure_envelope_c570dd9202498f390083dd02503d5675f900e027/)，
  [PR #321](https://github.com/raylei50653/saccade/pull/321)，merge `e4879d5f`）。
  Owner verdict＝**`ACCEPT WITH NAMED LIMITS — H2 measurement closure`**。
  接受只終結 Phase A：不建立 Phase B、equivalence 仍 `unproven`、無 I/F/S、無 seal、無 H0 re-entry、
  **沒有自動 next**；任何後續執行都是另一次 owner 決策。
  具名 limits 與 adjudication 由 registry
  [`successor_formal_measurement_executed.owner_review`](../../research/contracts/claim_state_registry.md)
  獨家持有。
  W3 先於 W4 的理由＝#294→#295 的契約先行與 #302／#304 的
  guard-before-entry，且 W2–W3 不需 build ⇒ item 5 的 `build/h2_layer_p`
  位元不動。不是重建舊 items 4–5／`F` →
  [second failure evidence](research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/)
  · [controller archive](research/evidence/h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/)
  · [first failure evidence](research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/)
  ·
  [task charter](../../research/threads/h2_behavioral_identity_capture_task.md)
  · [declaration](research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)
  · owner decision surface [#286](https://github.com/raylei50653/saccade/issues/286)（四項決策；不授權任何執行）
- **(next only via separate owner decision)** any future H0 re-entry requires a new
  repair/seal/authorization chain; exact S=`6fdb060c…` is permanently spent.

## Parked

- Score temporal-to-stable-domain → [charter](../../research/threads/score_temporal_to_stable_domain_20260712.md)
- GT-support morphology → [charter](../../research/threads/gt_support_morphology_20260711.md)
- Occ-exit intervention modeling → [charter](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank → [research note](research/sparse_key_embedding_bank_20260704.md)

## Navigation（不佔 WIP）

- [Research threads index](../../research/threads/README.md)
- [Module research index](README.md)
- [Claim-state registry](../../research/contracts/claim_state_registry.md)
- [NO-GO registry](../../reference/no_go_registry.md)

## Done / closed

See the module research index, closed threads index, evidence ledger, and NO-GO registry; terminal details do not live in TODO.
