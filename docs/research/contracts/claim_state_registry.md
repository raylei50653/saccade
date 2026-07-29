<!-- doc-status: active -->
<!-- doc-promotion: none; STATE fact-owner only — never an evidence duplicate -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: cross -->
<!-- fact-owner: research-object current state = this file -->

# Claim-State Registry（研究對象的當前狀態）

**這張表解決的是「可判定性」，不是「任務選擇」。** 它不決定要做什麼；它決定**什麼是合法的**。

state、intent 與 execution 必須分開，否則這張表會退化成一個會自行膨脹的規劃系統：

```text
registry       已接受的「現在是什麼」 —— state（本檔；慢）
contracts      「可以去哪裡」          —— admissible transitions
O0 / TODO      選一個 mainline charter —— WIP=1
linked charter expected-state lease    —— 預計去哪；可替換，不是 state
Current step / PR                       —— probes；可丟棄，不是 authority
DEVELOPMENT                             —— 穩定路由，不承載 live state
```

> **`next admissible unit` ≠ `next task`。**
> 一個 object 可以同時有兩個合法後續（補 L4 transfer／因 decision relevance 消失而停止研究）。
> **兩者都 admissible，但值不值得做不由 ladder 自己決定。** registry 只產生**合法候選集**；
> 從候選集裡選出唯一 decision-changing mainline charter 是 O0 的事（§ 5）。否則只是把「機械爬梯」換成
> 「機械執行 next_admissible_unit」。

---

## 1. 邊界：這是 state 的 fact-owner，不是 evidence 的副本

| 誰是 authority | 管什麼 |
|---|---|
| **contracts/**（[本目錄](README.md)） | **規則**：什麼是合法狀態、合法轉移、證據型別、claim ladder |
| **declaration / evidence / review 文件** | **證據與裁決**：數字、統計理由、terminal 為何被接受 |
| **本表** | **當前有效狀態**：這個 object 目前最高被接受到哪一格，並**指回**上面兩者 |

本表**可以**權威地說「這個 object 目前最高被接受到 L1，terminal 是 `ACCEPTED_WITH_LIMITS`」。
本表**不得**重新描述統計理由、不得抄結論、不得放數字。每一列必須附
`supporting_declaration` / `accepting_review` / `blocking_clause` / `last_transition`，
否則它就是一份會漂移的摘要表（[C5](../../ownership/doc_structure_contract.md)：不得有第二真相）。

本表也**不得**記錄 expected / hoped-for state、正在跑的 probe 或僅因工作開始／停止而更新
`last_transition`。這些只屬於 linked charter / `Current step`；只有 owner 接受 object state 改變才回寫本表。

---

## 2. Object identity（最需要被釘死的一件事）

**一個 object ＝ 一個 production claim 所作用的決策對象。**
**不是**一份 artifact，**不是**一次實驗實例，**不是**一個 substrate 上的版本。

```text
✅  gate.safe_region.dist_h_log_h_ratio          ← 語義穩定;substrate 是狀態,不是身分
❌  safe_region                                   ← 太粗:不同座標族會被錯誤合併
❌  safe_region_on_offline_proxy                  ← 太細:每換一次 substrate 就生一個新 object,
                                                     從此追不到「同一對象跨 substrate 遷移」
```

`substrate` 是**狀態欄**。同一個 object 在 offline 座標上是 L1、在 runtime 座標上未證，這正是
registry 要能表達的東西——而不是兩個 object。

命名：`<layer>.<decision-object>.<axes|quantity>`。

---

## 3. State record schema

```yaml
object:                  # §2 identity,語義穩定
layer:                   # L0 gate | L2 score | quantity（非決策層,如忠實性）
ladder:                  # 這個 object 的 rung 由哪份契約定義
transition_semantics:    # defined | unavailable   ← 缺契約時是合法狀態,不是 schema 缺陷
state:                   # 已被 owner 接受的 rung/terminal;draft 宣告不是 state
substrate:               # 這個 state 是在哪組座標/底材上證的
target_substrate:        # 這個 claim 最終要作用的底材（≠ substrate ⇒ 需要 transfer）
open_limits:             # 已記錄的限制（指回來源,不複述）
blockers:                # 見 §4:必須標型別
dependencies:            # 合法但需先完成的其他 object/unit
decision_relevance:      # 見 §5:必須通過反事實測試,否則為 zero
supporting_declaration:  # 證據/裁決來源
accepting_review:        #
last_transition:         # 日期 + PR/commit
# ---- 以下為 derived,非原始事實 ----
admissible_units:        # [reviewed derived state / cached derivation]
derived_from:            # 推導所依據的契約條文與 state
last_reviewed_at:        #
```

`admissible_units` 是**快取的推導值**，不是事實。契約條文改動、substrate 裁決更新、或依賴 object
完成時，它就可能過期——所以必須帶 `derived_from` 與 `last_reviewed_at`。長期它應該由規則推導，
第一版允許人工填，但**必須標示為 derived**。

---

## 4. Admissibility（registry 只產生候選集）

### 4.1 Layer / evidence 型別必須匹配

| Layer | 只接受 | 出現即**錯層** |
|---|---|---|
| `L0 gate` | GT retention / hurt / safe prune / coverage（單向、非補償） | margin、rank、top-1、AUC、「贏過 baseline X %」 |
| `L2 score` | event-local rank / margin / top-1 / calibration | coverage、prune mass、safety bound |

錯層的單元 **INADMISSIBLE**，不論宣告寫得多嚴謹。
*（實例：discrete-\(M\) 以「贏過 CV 10 %」判定卻自稱 gate → 錯層。修法是**改層**，不是修統計。）*

### 4.2 Substrate 不繼承

`substrate ≠ target_substrate` ⇒ **所有消費該 state 的單元 INADMISSIBLE**，直到取得 **L4**。
依據：framework § 9.3 + § 13（*offline safe \(\not\Rightarrow\) online effective*）＋
[fidelity protocol](runtime_quantity_fidelity_protocol.md) core lemma。
**「已被授權」≠「admissible」。**

### 4.3 Blocker 必須分型（編排語義不同）

| 型別 | 意義 | 編排器該做什麼 |
|---|---|---|
| `inadmissibility` | 在當前狀態下這個單元**不合法** | **直接排除**（不是排隊等待） |
| `dependency` | 合法，但必須先完成另一單元 | **展開依賴**，把依賴項放進候選集 |

不分型的話，registry 只能說「被擋住」，卻不知道該**停止**還是該**往前追依賴**。

### 4.4 `transition_semantics: unavailable` 是合法狀態

某個 layer 的契約若不存在，該 layer 的 object **無法自動判定 admissibility**。這**不是** schema 缺陷——
它把架構缺口**顯式化**，而不是假裝所有研究都已可編排。這類 object 一律**不進候選集**，直到契約補上。

---

## 5. Decision relevance（硬性；反事實測試）

一列的 `decision_relevance` 必須能回答**兩件事**，否則視為 **zero**：

1. **哪個決策變數會改變？** gate membership rule／threshold／substrate selection／default-off hook 能否升格？
2. **什麼裁決結果才會造成改變？**（正反都要寫）

> **反事實測試：假如下一格通過**或**失敗，哪個已知決策會因此不同？**
> 兩種結果都不改變任何決策 → **形式相關、決策無效** → **不得取得 WIP 鎖**。

很多研究表面上「與 production 有關」，實際上不論正反都不會改任何決策。這條就是用來擋它們的。

---

## 6. Records

> `state` 只填**已被 owner 接受**的東西。draft 宣告不是 state。理由不在這裡——點連結。

### `gate.safe_region.dist_h_log_h_ratio`

```yaml
layer: L0 gate
ladder: feasible-set L0–L6            transition_semantics: defined
state: L1 in-sample region (GLOBAL_PARTIAL_ORDER_READY / ACCEPTED_WITH_LIMITS)
substrate: offline proxy coordinates (ensure_prod_proxy_scores; 高度比用原始框高)
target_substrate: offline proxy coordinates (coordinate transfer to runtime is not yet established)
open_limits: [axes 建於 offline 座標, track-level CP UCB 為 nominal 非 cluster-adjusted,
              restricted closure 未 solve]
blockers:
  - type: inadmissibility
    what: restricted-closure solve
    clause: framework §9.3 + §13; fidelity_protocol core lemma
    because: the underlying offline proxy coordinates are unfaithful (T2_PROXY_UNFAITHFUL) and no coordinate transfer has been accepted
dependencies: []
decision_relevance:
  variable: bridge candidate 的 gate membership rule
            （production 粗篩層現幾乎全關:max_speed=0, spatial_gate=0,僅 h_lo/h_hi
              ⇒ 目前是 bdist 這個 score 在兼任 gate）
  if_pass: N/A (no transfer is currently active)
  if_fail: N/A
supporting_declaration: ../../modules/semantic/research/boolean_atom_partial_order_20260711.md
accepting_review: PR #107 (ACCEPTED_WITH_LIMITS)
last_transition: 2026-07-12 — offline-coordinate limit 追記（append-only）
admissible_units: []
derived_from: §4.2 substrate
last_reviewed_at: 2026-07-12
```

**Stable limit:** runtime-coordinate transfer 尚未被接受；任何消費此 offline state
的 closure solve 仍 inadmissible。執行 authority 與 lifecycle 只見
[declaration](../../modules/semantic/research/safe_domain_runtime_transfer_declaration_20260712.md) /
[thread](../threads/closed/runtime_faithful_safe_domain_20260712.md)，不在 registry 鏡射。

### `gate.safe_region.region_asset_pack`

```yaml
layer: L0 gate
ladder: RegionAsset maturity           transition_semantics: defined
state: A1 (A1_ACCEPTED_WITH_LIMITS)    substrate: offline proxy
target_substrate: runtime
open_limits: [5 條:no D1 trace / no second consumer / event-mass 等查詢需 raw …]
blockers:
  - type: dependency                   # ← 合法,但要等
    what: transfer / intervention / production promotion
    clause: safe_region_asset_contract（transfer 尚未授權）
    depends_on: gate.safe_region.dist_h_log_h_ratio (unfaithful offline substrate)
decision_relevance: 隨上游 object;本身不獨立驅動決策
supporting_declaration: safe_region_asset_contract.md
last_transition: 2026-07-11
admissible_units: []                   # [cached derivation] 依賴未解
derived_from: §4.3 dependency blocker
last_reviewed_at: 2026-07-12
```

### `quantity.bdist_temporal_reduction_R`

```yaml
layer: quantity
ladder: fidelity 四分 terminal          transition_semantics: defined
state: R1_FAITHFUL                     # 未降級:P0 從未證偽此 state,見下方 open_limits
substrate: runtime CUDA（封印於 headline-**m** preset 的 adaptive-anchor 設定 + 七序列 support）
target_substrate: same
open_limits: [不推廣至其他 anchor mode / preset / detector / substrate,
              **capture provenance 不完整（P0 2026-07-14 重發 cause）**:
                h_lo / h_hi / spatial_gate / max_speed 未蓋章,
                且無 capture-time tracker_gpu.cu file hash（僅 git_commit）
                ⇒ packet 無法自證其所跑的 policy;此為記錄缺口,非 state 反證]
blockers: []
decision_relevance:
  variable: 無直接決策變數
  role: **enabler** — 它是「runtime 座標可稽核」的前提,safe domain 與任何 score 研究得以
        建在 runtime 座標上,靠的就是它
supporting_declaration: ../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md
accepting_review: P0 Correction 1 §C1.6（2026-07-14 owner 重發 cause;僅授權本表更新）
last_transition: 2026-07-12 — owner accepted R1_FAITHFUL
                 2026-07-14 — open_limits append-only 追記 provenance 缺口（state 不變）
admissible_units: []                   # 已達成
last_reviewed_at: 2026-07-14
```

> **不要把 P0 讀成「D0/R1/S0 證據無效」。** P0 原本公布的 cause（外來 capture config）
> 已於 2026-07-14 撤回:它比對的 `px=0.4 / dir_bonus=0.0` 正是 headline-**m** 的正確值,
> 而 P0 凍結的是 **s**。真因是上面的 provenance 缺口,**在任何 preset 下都成立**,
> 補救方向是補蓋章而非重捕證據——但補救動作 H0 已於 2026-07-19 以
> `H0_PROVENANCE_INVALID`（route 1,owner-accepted）關閉,缺口現為永久留帳,
> 見 `quantity.bridge_capture_provenance`。

### `quantity.s0_offline_proxy_of_bdist`

```yaml
layer: quantity
ladder: fidelity 四分 terminal          transition_semantics: defined
state: T2_PROXY_UNFAITHFUL (= not_fidelity_aligned)
substrate: offline reconstruction
open_limits: [GT 邊界被扭曲（存在 offline-safe / online-unsafe 質量）]
blockers:
  - type: inadmissibility
    what: 任何「s0 代表 production bdist」的宣稱
    clause: fidelity_protocol — 不可轉移
decision_relevance:
  role: **反向約束** — 它讓一整類宣稱失效;純 offline 訊號仍可用,但須標示不可轉移
supporting_declaration: ../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md
last_transition: 2026-07-12
admissible_units: []
last_reviewed_at: 2026-07-12
```

### `quantity.bridge_capture_provenance`

```yaml
layer: quantity                        # 非決策層:它管「證據能否自證」,不管任何 gate/score
ladder: P0 ordered terminal            transition_semantics: defined（該研究自帶封印判準）
state: P0_CAPTURE_SEMANTICS_UNVERIFIABLE (scope-corrected 2026-07-14, owner review)
       # terminal 現為**推導值**:contradiction（有蓋章且矛盾）> absence（未蓋章）> clean。
       # 對 m（證據實際封存的 preset）稽核 ⇒ 零矛盾、四欄位缺席 ⇒ UNVERIFIABLE。
prior_terminal: P0_CAPTURE_SEMANTICS_INVALID (sealed 2026-07-13)
       # **不是被推翻,是 out of scope**:P0 §1 凍結 s,而對 s 而言 px/dir_bonus 確實是
       # 已蓋章的矛盾 ⇒ INVALID 在其宣告的 scope 內成立。錯的只有 scope。
       # sealed packet 保留該 label,不得改。
cause: capture-provenance incompleteness（h_lo / h_hi / spatial_gate / max_speed 未蓋章;
       無 capture-time tracker_gpu.cu hash）
substrate: D0/R1/S0 的 shadow capture provenance（runtime CUDA）
target_substrate: same
open_limits: [**永久留帳（route-1 合法終局,2026-07-20）:未取得 faithful capture;
              沒有 accepted runtime-fidelity edge;沒有 actual H0 guarantee envelope**
              —— 唯一授權的 H0 sealed invocation 於 ordered terminal
              `H0_PROVENANCE_INVALID` 關閉（owner-accepted 2026-07-19,#209;
              controller retry / Phase B forbidden;NEXT none automatically）;
              缺口內容不變:
              h_lo / h_hi / spatial_gate / max_speed 未蓋章 ⇒ packet 無法自證其 policy
              （**在任何 preset 下**;此為記錄缺口,非證據矛盾）;
              無 capture-time tracker_gpu.cu hash;
              replay 封頂 L1（D0 v2 export 缺 frame / slot / det score
              ⇒ margin、atomicMax claim、commit 不可重建）;
              **本研究的 seal 不可稽核**——宣告/runner/results/packet 同一 commit 落地
              （b136437f）⇒ 見 declaration §C1.9 與 tests/contract/test_declaration_seal_order.py]
blockers: []
decision_relevance:
  role: **反向約束 + enabler 前提** — 它不改任何 production 決策,但它決定
        D0/R1/S0 的 packet 能否自證所跑的 policy;缺口未補前,任何「這份證據代表
        headline runtime」的宣稱都只能靠**外部**宣告支撐,不能由 packet 自證
  variable: 無直接決策變數
supporting_declaration: ../../modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md
accepting_review: 同上 §C1.6（2026-07-14 owner 重發 cause）;
                  H0 route-1 terminal acceptance＝#209 owner comment（2026-07-19）,
                  charter 收尾見 threads/closed/bridge_frozen_evidence_o0_routing_20260716.md
last_transition: 2026-07-14 — cause 由 foreign-capture 改為 provenance incompleteness;
                 terminal **重判型別** INVALID → UNVERIFIABLE（owner review）
                 2026-07-20 — H0 admissible unit 依 route 1 關閉（object state 不變）;
                 open_limits 改寫為永久留帳形式
                 2026-07-21 — #224/#227 re-entry 第二個 owner-accepted `H0_PROVENANCE_INVALID`
                 （object state 不變;詳見 reentry_terminal_history）
                 2026-07-21 — #235 re-entry #3 第三個 owner-accepted `H0_PROVENANCE_INVALID`
                 （object state 不變;死因不同＝extension_load confinement-plan construction;
                 詳見 reentry_terminal_history re-entry #3）
                 2026-07-24 — H0-R4 Phase-A sole authoritative invocation under sealed S=a76efffa
                 mechanical disposition `H0_PROVENANCE_INVALID`（controller literal
                 `provenance_invalid`; independent verifier valid=true）; authorization
                 consumed; exact S permanently spent; owner acceptance of the truthful
                 negative terminal is the surface of this evidence PR merge only
                 （object state 不變;詳見 reentry_terminal_history re-entry #4 / H0-R4）
                 2026-07-27 — **H2**（successor unit,非 H0 re-entry）Phase-A single invocation
                 at head 0a5dffe9 消耗一次授權,controller terminal
                 `H2_INPUT_MUTATED_DURING_MEASUREMENT`,0/4 ordered runs started,no capture,
                 archive 被 independent verifier 拒收;adjudicated root cause＝controller
                 self-mutation（見 reentry_terminal_history 2026-07-27 條）
                 2026-07-28 — **H2** 第二次 Phase-A single invocation at head 7646f421 消耗第二份授權,
                 controller terminal `H2_MEASUREMENT_EXECUTION_INVALID`,1/4 ordered runs started,
                 no capture;archive 通過 independent verifier（valid=true, complete）;
                 adjudicated root cause＝child 在 import 之後重套 ingress environment contract
                 （見 reentry_terminal_history 2026-07-28 條）
                 （object state 不變;H0 五個 spent S 與 sealed history 全部不變）
admissible_units: []                   # H0 unit 已消費並於 route 1 關閉;現無宣告的補救動作。
                                       # 重進=append-only declaration amendment＋新 I→F→S
                                       # owner reseal＋owner scheduling（§9 重新 review 時再推導）
derived_from: §4.3 dependency 已消費:H0 exactly-once sealed invocation terminal=
              `H0_PROVENANCE_INVALID`（owner-accepted,#209）⇒ 候選集回空,
              缺口以 open_limits 永久留帳（O0 route-1 下游;合法終局,非待辦）
reentry_terminal_history:                 # append-only;不改上面 route-1 永久留帳結論
  - date: 2026-07-21
    scheduling: separate owner-scheduled re-entry（非本 registry 自動發生;O0 charter 仍 closed,候選集仍空）
    scope: "#224 append-only amendment（build-tool binding 進 h0_bound_inputs_v1）→ #227 repair（canonical CONTROLLER_INPUT_MEMBERS 14 單源 across full/landing/discovery verifiers）"
    chain:
      I2: 31c9eee83fc46f34ab0fd9218c4e1ba2ed545636
      F2: 46539a2d490aeed63b7c9cea8a10e9bf2819a364
      S2: 0da082a9c5366092334bd140c76f54dc4b904423
    invocation: authoritative_count=1; exactly_once_authorization=consumed; retry/second_invocation_under_S2=permanently_forbidden
    terminal: H0_PROVENANCE_INVALID       # controller literal `provenance_invalid`;A2.4 第一 ordered terminal
    independent_verifier: '{"result":"provenance_invalid","valid":true}' ; rc=0
    factual_boundary:                      # 接受為 truthful negative ordered terminal
      failure_stage: preflight
      failure_reason: "build/h0_phase_a exists at controller launch"
      checkpoints_T0_T4: not_reached
      monitor: not_started
      capture_child_runs: NOT_RUN（00_capture_off / 01_capture_on_1 / 02_capture_on_2 / 03_capture_on_3）
      build_runtime_gpu_identity_and_comparison: not_produced
      capture_evidence: none
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_31c9eee83fc46f34ab0fd9218c4e1ba2ed545636/
    digests:
      manifest_json_sha256: ff9fea3e9150ca90da9cba21064ca4428b6d9ebcf5fda4a35168d7365e29b578
      result_json_sha256: 2c1cfa17c977ad02c6c1dee335810b9ee7ff37f1cbba1382d41a00f06b96529a
      checksums_sha256_digest: c1e9d2a4fc06f8a6bad12152c6833f29729e0865eeb9697b0a7d99a06705549a
      verifier_report_aggregate_json_sha256: 0be12cc292773239d75604e9f2496787387ee8874395f4a3a9d723c615fe3f2e
    owner_acceptance: 2026-07-21 — accepted as truthful negative ordered terminal;
                      **not** `H0_FULL_COMMIT_CAPTURE_FAITHFUL`;不成立 actual H0 guarantee;不構成 guarantee registration 基礎
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge / 無 actual H0 guarantee envelope）;候選集仍空;Phase B / GCTM / B1 / O1 未啟動
    future_reentry_precondition: 另一 separate owner-scheduled task;須先把 launch hygiene 做成 machine-checked non-authoritative pre-authorization gate（複用 controller 真實 preflight predicate,在授權 exactly-once 前 fail-closed 拒 pre-existing `build/h0_phase_a` tree）;再 fresh qualified I→F→S 鏈 + 另行 exactly-once 授權
    issues: '#224 (closed on this closeout landing) / #227 (repair delivered; closed after linking accepted terminal)'
  - date: 2026-07-21
    scheduling: separate owner-scheduled re-entry #3（滿足前次 future_reentry_precondition:launch-hygiene gate #234 先行;非本 registry 自動發生;O0 charter 仍 closed,候選集仍空）
    declaration_amendment: "Amendment 9（headline_bridge_full_decision_capture_declaration_20260713.md;append-only;owner-sealed 進 S=3a6a9ec6）"
    scope: "re-admit 既有 sealed unit h0_build_tool_provenance_closure 供單一 fresh I→F→S attempt;非第二 unit;acceptance matrix/checker/workflow tuple/qualification 語義/歷史 declaration/歷史 sealed evidence 全 byte 不變"
    chain:
      I3: 5a2d1de509fa64f2e5ce9a4db8182337da215968
      F3: 7895704c298504b279ae8e1febf19ca2a715637f
      S3: 3a6a9ec6348f1dccca6acabef8025159c3bec1d3
    invocation: authoritative_count=1; exactly_once_authorization=consumed; exact_S_permanently_spent; retry/resume/second_invocation_under_S3=permanently_forbidden
    terminal: H0_PROVENANCE_INVALID       # controller literal `provenance_invalid`;A2.4 第一 ordered terminal
    independent_verifier: '{"result":"provenance_invalid","valid":true}' ; rc=0
    factual_boundary:                      # 接受為 truthful negative ordered terminal
      failure_stage: extension_load confinement-plan construction（capture 前）
      failure_reason: "seal-event/runtime-binding identity incompatibility：declaration 既是 F 凍結的 runtime-bound input 又是 S 要 append SEALED 的目標 ⇒ seal mutation 被判 provenance mismatch"
      failing_predicate: provenance_ok=false（唯一 false;build_ok/extension_ok/artifacts_ok/classified_execution/packets_valid/policy_equal/runners_ok/serialization_ok 皆 true;timed_out=false）
      checkpoints: T0/T1 completed（inventory_equal;t0 bound_inputs_digest 4e3eb01f）; T2a_0→T4 not_reached
      capture_child_runs: NOT_RUN（00_capture_off / 01_capture_on_1 / 02_capture_on_2 / 03_capture_on_3;confinement_plan_digest=null）
      build_runtime_gpu_identity: complete（build/runtime/GPU identity 已產生;四個 child runtime_inputs 因 blocking result 均為 not_produced）
      comparison: not_produced
      capture_evidence: none
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_5a2d1de509fa64f2e5ce9a4db8182337da215968/
    digests:
      inventory_digest: c797de0e28d3f325ecb2a7ae06f74a9169dfe2208426c434218e224fab76def9   # 25 members
      manifest_json_sha256: 51ec6d0a223f378fd40aed71b9e9582afb14e071c0de78491e0663318816706d
      result_json_sha256: 2c1cfa17c977ad02c6c1dee335810b9ee7ff37f1cbba1382d41a00f06b96529a
      checksums_sha256_digest: 85dbca11a80691c78ba13eddaa385a2564a0b2811f2ced71d4738f68350dd600
      verifier_report_aggregate_json_sha256: 0be12cc292773239d75604e9f2496787387ee8874395f4a3a9d723c615fe3f2e
    owner_acceptance: 2026-07-21 — accepted as truthful negative ordered terminal（PR #235 comment 5032610430）;
                      **not** `H0_FULL_COMMIT_CAPTURE_FAITHFUL`;不成立 actual H0 guarantee;不構成 guarantee registration 基礎
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge / 無 actual H0 guarantee envelope）;候選集仍空;guarantee set 空;Phase B / GCTM / B1 / O1 未啟動
    future_reentry_precondition: NONE — 本 acceptance 不授權任何 repair / reseal / 新 re-entry;exact S=3a6a9ec6 permanently spent
    issues: '#235 (owner acceptance surface; remains UNMERGED; exact S=3a6a9ec6 immutable spent history; closeout landed via separate PR; no closing keyword)'
  - date: 2026-07-24
    scheduling: owner-scheduled H0-R4 Phase-A execution under sealed S（post #277 Seal; post Amendment 10 repair qualification; non-automatic; O0 charter 仍 closed,候選集仍空）
    declaration_amendment: "Amendment 10 authority overlay + owner SEALED row at S=a76efffa（PR #277）"
    scope: "sole authoritative Phase-A controller invocation under exact sealed S; Phase B forbidden; no retry/resume; no automatic guarantee registration"
    chain:
      I4: 2a233387a6a321dd43570e2e30dc718571b3b4f4
      F4: ced4a4cc6a71473dcb1225203e6d59df0437d976
      S4: a76efffa01a6fb731218150c355f5859bb8e6dd4
    authorization:
      identity: h0_r4_phase_a_exactly_once_authorization_20260724
      surface: "https://github.com/raylei50653/saccade/issues/278"
      authorized_invocation_count: 1
      authorization_consumed: true
      consumed_at: controller_process_launch
    invocation: authoritative_count=1; exactly_once_authorization=consumed; exact_S_permanently_spent; retry/resume/second_invocation_under_S4=permanently_forbidden
    controller_result: provenance_invalid   # controller literal; A2.4 first ordered terminal
    mechanical_disposition: H0_PROVENANCE_INVALID
    independent_verifier: '{"document_type":"aggregate_verification","result":"provenance_invalid","schema":"h0_phase_a_verifier_v1","valid":true}' ; rc=0
    factual_boundary:
      failure_stage: extension_load
      failure_reason: "extension/plugin load is absent from runtime attestation"
      failing_predicate: provenance_ok=false（唯一 false;build_ok/extension_ok/artifacts_ok/classified_execution/packets_valid/policy_equal/runners_ok/serialization_ok 皆 true;timed_out=false）
      checkpoints: T0/T1 completed（inventory_equal;t0 bound_inputs_digest 2e5e11a4）; T2a_0→T4 not_reached
      capture_child_runs: NOT_RUN（00_capture_off / 01_capture_on_1 / 02_capture_on_2 / 03_capture_on_3;confinement_plan_digest=null）
      build_runtime_gpu_identity: complete（build/runtime/GPU identity 已產生;四個 child runtime_inputs 因 blocking_result=provenance_invalid 均為 not_produced）
      comparison: not_produced
      capture_evidence: none
      note: "DriftError during extension_load runtime attestation maps to ordered terminal provenance_invalid before extension_load_failed; extension_ok predicate remains true"
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4/
    witness_bundle: docs/modules/semantic/research/evidence/h0_r4_phase_a_execution_witness_20260724/
    digests:
      inventory_digest: a0527d314051b9aa96660ce57c7a1c5478aa17950db5533549deca48b4f52b4e   # 25 members
      manifest_json_sha256: c3cf4bd8bdfbf0fc2dc500b982ceca8136913d7381b185a4b9506e724e903cf0
      result_json_sha256: 2c1cfa17c977ad02c6c1dee335810b9ee7ff37f1cbba1382d41a00f06b96529a
      checksums_sha256_digest: 3ff0ac6087a989265d6d8f73b0a2af903b446d03c24a41891576b9a4241a4a04
      verifier_report_aggregate_json_sha256: 0be12cc292773239d75604e9f2496787387ee8874395f4a3a9d723c615fe3f2e
    owner_acceptance: "accepted: 2026-07-24; acceptance_surface: PR #279 merge; acceptance_commit: 55d2da47c5538004e254830a273a2ac4362d8eae; scope: truthful-negative H0_PROVENANCE_INVALID only"
    owner_acceptance_not: "**not** `H0_FULL_COMMIT_CAPTURE_FAITHFUL`; actual guarantee = none; runtime compatibility = none; Phase B = forbidden; future reentry authorization = none; does not authorize repair / reseal / new re-entry by itself"
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge / 無 actual H0 guarantee envelope）;候選集仍空;guarantee set 空;Phase B / GCTM / B1 / O1 未啟動; registration-v3 仍為 contract-sealable only; R4 chain permanently spent; R5 repair authority is a separate owner decision (#280) only
    future_reentry_precondition: NONE from this mechanical terminal alone — exact S=a76efffa permanently spent; any future work requires a separate owner decision
    issues: '#277 (Seal landed) / #278 (exactly-once authorization surface; consumed at launch; closed after PR #279 acceptance) / #280 (H0-R5 repair owner decision; not execution authority)'
  - date: 2026-07-25
    scheduling: owner-scheduled H0-R5 Phase-A exactly-once execution under sealed S（post tool_runtime independent-expansion parity repair 524f7e3; non-automatic; O0 charter 仍 closed,候選集仍空）
    repair_unit: h0_r5_preseal_verifier_tool_runtime_independent_expansion_parity_v1
    scope: "sole authoritative Phase-A controller invocation under exact sealed S; Phase B forbidden; no retry/resume; no automatic guarantee registration"
    chain:
      I5: 524f7e3b88f73bc366d467d53a2c393a7d3ba937
      F5: 6e425dc6f89a15d4eb43d3889a517d632f0ee39e
      S5: 6fdb060c50c9ed784a3fa2229b1ea2514dd1af5e
    authorization:
      identity: h0_r5_phase_a_exactly_once_authorization_20260725
      surface: "https://github.com/raylei50653/saccade/issues/283"
      authorized_invocation_count: 1
      authorization_consumed: true
      consumed_at: controller_process_launch
    invocation: authoritative_count=1; exactly_once_authorization=consumed; exact_S_permanently_spent; retry/resume/second_invocation_under_S5=permanently_forbidden
    controller_result: provenance_invalid   # controller literal; A2.4 first ordered terminal
    mechanical_disposition: H0_PROVENANCE_INVALID
    independent_verifier: '{"document_type":"aggregate_verification","result":"provenance_invalid","schema":"h0_phase_a_verifier_v1","valid":true}' ; rc=0
    factual_boundary:
      failure_stage: extension_load
      failure_reason: "extension/plugin load is absent from runtime attestation"
      failing_predicate: provenance_ok=false（唯一 false;build_ok/extension_ok/artifacts_ok/classified_execution/packets_valid/policy_equal/runners_ok/serialization_ok 皆 true;timed_out=false）
      checkpoints: T0/T1 completed（inventory_equal;t0 bound_inputs_digest a8e4bece）; T2a_0→T4 not_reached
      capture_child_runs: NOT_RUN（00_capture_off / 01_capture_on_1 / 02_capture_on_2 / 03_capture_on_3;confinement_plan_digest=null）
      build_runtime_gpu_identity: complete（build/runtime/GPU identity 已產生;四個 child runtime_inputs 因 blocking_result=provenance_invalid 均為 not_produced）
      comparison: not_produced
      capture_evidence: none
      note: "Same ordered terminal surface as R4: DriftError during extension_load runtime attestation maps to provenance_invalid before extension_load_failed; extension_ok remains true. Preseal verifier tool_runtime independent-expansion parity held (valid=true); failure is post-build extension_load attestation, not freeze assembly."
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_524f7e3b88f73bc366d467d53a2c393a7d3ba937/
    witness_bundle: docs/modules/semantic/research/evidence/h0_r5_phase_a_execution_witness_20260725/
    digests:
      inventory_digest: 800eb83952ee416c40d3ababc1acee6e9a770db267173b1d60adf4a2d1937f44   # 25 members
      manifest_json_sha256: 8b96269bdfb25f6f7bcfd8368db0a46c724cdd2f1225c19b1210d43f2eff5ace
      result_json_sha256: 2c1cfa17c977ad02c6c1dee335810b9ee7ff37f1cbba1382d41a00f06b96529a
      checksums_sha256_digest: 9e69184f064ae66c115d4e4f58fea88b822847034456a3256066fce03c9dacfe
      verifier_report_aggregate_json_sha256: 0be12cc292773239d75604e9f2496787387ee8874395f4a3a9d723c615fe3f2e
    owner_acceptance: "mechanical closeout recorded 2026-07-25; owner acceptance of truthful-negative terminal remains a separate evidence-landing surface; scope: H0_PROVENANCE_INVALID only"
    owner_acceptance_not: "**not** `H0_FULL_COMMIT_CAPTURE_FAITHFUL`; actual guarantee = none; runtime compatibility = none; Phase B = forbidden; future reentry authorization = none; does not authorize repair / reseal / new re-entry by itself"
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge / 無 actual H0 guarantee envelope）;候選集仍空;guarantee set 空;Phase B / GCTM / B1 / O1 未啟動; registration-v3 仍為 contract-sealable only; R5 S5 chain permanently spent
    future_reentry_precondition: NONE from this mechanical terminal alone — exact S=6fdb060c permanently spent; any future work requires a separate owner decision
    issues: '#283 (exactly-once authorization surface; consumed at launch) / prior R5 repair chain including 524f7e3 tool_runtime parity + extension/plugin attestation closure'
  - date: 2026-07-27
    unit: H2                                # **不是 H0 re-entry**:successor unit,identity layer only;
                                            # H0 的五個 spent S / sealed history / permanent ledger 全部不變
    scheduling: owner-scheduled H2 Phase-A single-invocation execution（非本 registry 自動發生;O0 charter 仍 closed,候選集仍空）
    scope: "sole authorized H2 Phase-A controller invocation at the exact seal-candidate head; Phase B forbidden; no retry/resume"
    predecessor_head: 0a5dffe921d78fce8e525baf8b4b624fc9ab957c   # source_tree 5530be2d67b8e7c83a7a858a44a2b11a1c347927
    binding:
      f64: a03fc4590ca931435fde4a93f28bec8ed156fe852718cd214e780e002d97fd8b
      layer_p_certificate_digest: d95859cb3cc27eeadb72b0f94fdcf45107c590058dd2c288e14bcd47c3e24802
      layer_p_selected_base: b2f3c23f419cb03cf89eae677bdf9262a8dd3634
      bounded_probe: 2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5
      controlled_host_reattestation: "green at this exact head — run 30276844285"
    authorization:
      authorized_invocation_count: 1
      authorization_consumed: true          # spent
      consumed_at: controller_process_launch
    invocation: authoritative_count=1; retry/resume/second_invocation_at_this_head=forbidden
    controller_terminal: H2_INPUT_MUTATED_DURING_MEASUREMENT   # order 1, phase a; controller literal `input_mutated`
    adjudicated_result: no capture
    ordered_runs_started: 0/4               # 00_capture_off / 01 / 02 / 03_capture_on — 全未啟動;archive 無 runs/
    faithful_capture: 0
    measurement_claim: 未成立                # observation 的 capture_off_on_equal / packets_valid 為未經執行的預設值,不得引用
    equivalence: unproven                   # 未改動
    seal: 未完成
    archive_verifier: 拒收 — "recorded Layer-P certificate match disagrees with the archived
                       freeze/certificate/content bindings and independent Git-tree recomputation";
                       controller rc=2;本次無 verifier report
    adjudicated_root_cause: controller self-mutation / 不可滿足的 checkout invariant
    label_vs_cause: "`H2_INPUT_MUTATED_DURING_MEASUREMENT` 是 controller 記錄的 terminal label;
                     它**不是**對實際根因的正確語義描述。實際事件＝controller 在 repo 內建立自身
                     evidence root,再把該自身產物判定為 execution checkout mutation。"
    defect_sites:
      - "scripts/tools/run_h2_measurement.py:1050-1057 — clean-checkout gate 之後立即在同一 checkout 建立未被 gitignore 的 evidence root（EVIDENCE_REL,h2_measurement_evidence.py:103）"
      - "scripts/tools/run_h2_measurement.py:1140-1145 — checkout hygiene 理由被折進 layer_p_certificate_matches_freeze;archived certificate_mismatch_reasons 僅一句 checkout 字串、零個真正 certificate 不符"
      - "scripts/tools/run_h2_measurement.py:1291-1293 — stop boundary 重複同一不可滿足要求"
    pre_launch_conditions: 成立且已獨立驗證（Layer-P certificate 37/37;freeze 22/22,含 controller 自身
                           terminal-1 predicate 乾跑零 reason）⇒ 失敗不在 binding,而在 controller 時序
    evidence_packet: docs/modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge /
                   無 actual H0 guarantee envelope）;候選集仍空;guarantee set 空;Phase B / GCTM / B1 / O1 未啟動
    required_successor_action: controller repair on a successor head，followed by a **completely new**
                               acceptance and authorization cycle — acceptance gate 2、Acceptance items 4/5
                               與 F 全部須對 successor head 重建;item 6 不在「重建」之列 —
                               該 exactly-once authorization 已 consumed 且永久 spent,
                               successor cycle 必須由 owner 另行簽發一份新授權;repair 會移動 execution-relevant code,
                               `0a5dffe9` 的任何 binding 皆不轉移;repair PR 不得宣稱延續本次 seal attempt
  - date: 2026-07-28
    unit: H2                                # 第二次 Phase-A attempt;仍非 H0 re-entry,H0 五個 spent S 全部不變
    scheduling: owner-scheduled H2 Phase-A single-invocation execution（第二份授權,與 2026-07-27 那份無關）
    scope: "sole authorized H2 Phase-A controller invocation at the exact seal-candidate head; Phase B forbidden; no retry/resume"
    predecessor_head: 7646f421a85a580e37e457def5e8ddc7c4bfa0ab   # source_tree 79ea5ae0ca6c69d7273d558dfaae9e08d6e1a64f
    binding:
      f64: f0d1b02e5a162d4949bb2db00f30d73242e7c4a8a833400b712f378c91d31ce4
      layer_p_certificate_object_digest: e60b98e6f7a2823e9921eac1b2f374d7391c686c433602b1dd41c2c04e1c1618
      layer_p_certificate_file_digest: 266f4b4ca5b891639d885f795f77ef603bb0b6877990a29922054af06e63d3e2
      layer_p_selected_base: 7646f421a85a580e37e457def5e8ddc7c4bfa0ab   # changed_count 0
      bounded_probe: 2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5
      controlled_host_reattestation: "green at this exact head — run 30334080842"
    authorization:
      authorized_invocation_count: 1
      authorization_consumed: true          # spent;authorization_id 342416678caa…
      consumed_at: controller_process_launch
    invocation: authoritative_count=1; retry/resume/second_invocation_at_this_head=forbidden
    controller_terminal: H2_MEASUREMENT_EXECUTION_INVALID   # order 4, phase a; controller literal `runner_nonzero`
    adjudicated_result: no capture
    ordered_runs_started: 1/4               # 00_capture_off 啟動並非零退出;01/02/03_capture_on 未達
    faithful_capture: 0                     # 無 packet / inventory / MOT 輸出
    measurement_claim: 未成立
    equivalence: unproven                   # 未改動
    seal: 未完成
    archive_verifier: 接受 — valid=true, verify_class=complete, file_count=28;
                      corpus checker PASS (1 roots; complete=1)。接受的是 archive 自洽性,
                      **不是** measurement——terminal 為負且 capture 為零。
                      **註冊當時的限定條件:兩項結果都只在 execution host 上成立**（見 defect_sites
                      verify_h2_measurement.py:207-280）;該限制已由 2026-07-29 repair 解除
                      （見 repair_landed.b），corpus checker 已以完整 git history 接回 CI,
                      host-independent inventory contract 仍並存
    adjudicated_root_cause: child 在 import eval stack 之後重新套用 ingress environment contract;
                            cv2 4.11.0 於 import 時新增 QT_QPA_FONTDIR / QT_QPA_PLATFORM_PLUGIN_PATH
                            並在 LD_LIBRARY_PATH 前綴自身 lib 目錄
    label_vs_cause: "`H2_MEASUREMENT_EXECUTION_INVALID` **是**對根因的正確語義描述
                     （與 2026-07-27 那次相反）:execution 確實 invalid,partition 依 execution
                     catch-all 選出該 terminal。"
    defect_sites:
      - "scripts/tools/run_h2_measurement_child.py:298 — repository_runner 在 :271 的 _import_eval_stack() 之後重跑整份 ingress predicate"
      - "scripts/tools/run_h2_measurement_child.py:176-200 — 同一 predicate 有 key-set 與 environment digest 兩個獨立分支;只修 key set 會在 digest 分支得到同一 terminal"
      - "scripts/tools/run_h0_phase_a_child.py:372 — 同形狀 latent（frozen ruler,不得修改;H0 五次執行都在此行之前終止）"
      - "scripts/tools/verify_h2_measurement.py:207-280 — archive 驗證時以**驗證主機**的 /etc/machine-id 與 os.getuid()
         重算 authorization execution domain 並要求與 archived record 相等 ⇒ 已 commit 的 Phase-A archive
         只能在產生它的那台機器上通過驗證,independent reviewer 與 CI 都不行。grant 綁 host 在 launch 時正確且須保留,
         但把該 live 重算搬進 archive 驗證是同一種結構錯誤。由本次註冊在 CI 上首次暴露;
         verify 檔是 F 綁定的 executed surface,修復歸同一個 repair PR 的第二個 commit,不在本次註冊內;
         修復時應一併補 H2 版 host-independence test（H0 有 test_h0_phase_a_archive_verification_is_execution_host_independent,H2 從來沒有）"
    sound_and_unchanged:
      - "scripts/tools/run_h2_measurement.py:602-629 — controller 從零建構 child environment 並在 launch 前斷言 key set（實測恰 17 個 expected keys）"
      - "scripts/tools/run_h2_measurement_child.py:683 — child ingress 驗證位置正確且通過 ⇒ launch 授權判定本身健全"
    pre_launch_conditions: 成立且已獨立驗證（Layer-P certificate 65/65;freeze 51/51;controlled host 綠）
                           ⇒ 失敗不在 binding,而在 child 的環境驗證時序
    predecessor_defects_closed: 2026-07-27 登記的四項 controller 缺陷在本 head 全部關閉
                                （checkout hygiene、predicate ownership、stop boundary、archive finalize）;
                                controller 首次抵達 child_launch
    why_review_missed_it: child 只以 source review 與合成環境 unit test 檢查;launch probe 雖 import 同一
                          eval stack,卻是以 operator 繼承環境執行（run_h2_measurement.py:654）,
                          從不使用那份 sanitized 17-key 環境 ⇒ probe 轉綠不帶任何關於 child 環境契約的資訊
    evidence_packet: docs/modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/
    controller_archive: docs/modules/semantic/research/evidence/h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/
    ledger_effect: 上方 route-1 永久留帳結論不變（仍無 faithful capture / 無 accepted runtime-fidelity edge /
                   無 actual H0 guarantee envelope）;候選集仍空;guarantee set 空;Phase B / GCTM / B1 / O1 未啟動
    repair_scope: H2 Phase-A execution-and-archive-verifier repair — 兩個必修 executed surface:
                  (a) child ingress-authority repair（run_h2_measurement_child.py）
                  (b) archive-verifier execution-domain repair（verify_h2_measurement.py）
                  兩者皆使既有 F/cert stale,故必須在**同一個**最終 successor head 上完成;
                  一個 repair PR、兩個獨立 commit 即可,不必拆成兩個 PR
    required_successor_action: 上述 repair scope on a successor head，followed by a **completely new**
                               acceptance and authorization cycle。owner 已裁定修法:保留 ingress gate、
                               不再從 import 之後的 live `os.environ` 重推 ingress predicate、
                               pre_import→post_import delta 僅作 diagnostic 記錄且不得重新參與授權判定;
                               授權環境＝third-party import 前捕獲的 immutable launch snapshot,
                               成文為 declaration Review Correction 4（只約束語義,不規定檢查次數與位置）;
                               frozen H0 child 不得修改。
                               修復落地後 `F64 f0d1b02e…` 與 certificate `266f4b4c…` 即 stale,不得沿用;
                               item 6 亦不在「重建」之列——第二份授權同樣已 consumed 且永久 spent
    required_successor_action_status: FULFILLED 2026-07-29 — repair_scope (a)(b) 兩個 executed surface
                               皆已修復並落地,見下方 repair_landed。**只履行 repair,未履行 acceptance /
                               authorization cycle**:未 seal、未恢復任何授權、equivalence 仍 unproven、
                               兩份授權仍永久 spent、item 4/5 仍待在新 head 重建
    repair_landed:
      date: 2026-07-29
      repair_implementation_head: 7cae46d8      # commit (b);commit (a) = cc02a0b0
                                                # 此欄指「修復實作所在的 commit」,不是重建 item 4/5 的 head——
                                                # 後者是本工作 merge 之後的最終 head,另行記錄
      unit: H2 Phase-A execution-and-archive-verifier repair（一個 PR、四個 commit:(a) child、
            (b) archive verifier + CI、(c) governance transition、(d) 審查中補上的
            production-runner regression;(d) 在 (c) 之後才加入,故 (c) 的 commit message 寫「三個」）
      merged_head: bb98dd61                     # merge commit（非 squash,四個 commit 全在 main）
                                                # 這才是重建 item 4/5 所用的最終 head
      a_child_ingress_authority: launch snapshot 於 execute_child 恰一次消費;import 之後不再從 live
            os.environ 重推 ingress predicate;pre_import→post_import delta 僅記 key 名稱
            （environment_import_delta.json,authority=diagnostic_only,不含任何 value 或 value 指紋）;
            configure_runtime_env 的 mutation gate 改用 **post-import 的具名 baseline**（用 ingress
            snapshot 當 baseline 會把 cv2 delta 誤算成 repo-owned mutation,即第三次同型失敗）;
            REPOSITORY_OWNED_ENV_KEYS 以 equality（非 subset）綁 producer 實際行為;
            EXPECTED_ENV_KEYS / STATIC_ENV / run_h0_phase_a_child.py 一 byte 未改,H0 :372 twin 仍 latent
      b_archive_verifier_execution_domain: archive 驗證只讀 archive 位元組(member set、host_identity、
            operator_uid、canonical absolute POSIX ledger_root 以純字串判定 + 原有 digest binding);
            launch-time host binding 未放寬(controller 於 admission/consumption 仍 live 推導且
            無 machine identity 時 fail-close);corpus checker 以完整 git history 接回 CI
      old_head_reproduction: 兩個缺陷各有一支在 c2d1c58f 上未經修改即失敗的測試
            （child = AST reproducer;verifier = live-derivation sentinel,失敗於 verify_h2_measurement.py:235）
      not_established: 無 seal、無 equivalence、無 capture、無新授權、無 registry object state 改變
      successor_work_item: h2_phase_a_rehearsal_harness — non-evidence full run 目前沒有入口
            （run_h2_measurement.py 僅單一路徑,--authorization required,ledger 為 default);
            **不得**在 repair 內新增 production rehearsal mode（admission 前分支等於沒測到 admission,
            admission 後跳過 consumption 等於改變 production authorization invariant);
            「不耗授權」＝不動 owner 第三份 grant,而非完全沒有 authorization artifact 走過 admission。
            harness 契約:不改 production controller / 完全隔離可丟棄 ledger + synthetic grant /
            走原本 admission 與 consumption 路徑 / 不接觸 owner ledger /
            產出永不進 canonical corpus,亦不得充當 item 4、item 5、F 或 S 的證據。另行審查
  rehearsal_harness_landed:
    date: 2026-07-29
    harness_landed: true                      # 入口存在
    rehearsal_passed: false                   # gate 未通過:harness 尚未被執行過一次
    unit: h2_phase_a_rehearsal_harness（一個 PR、四個 commit,順序 P → B → A → C:
          (P) authorization issuer 正規化、(B) canonical corpus provenance/admission guard、
          (A) rehearsal harness、(C) 本 governance transition。guard 必須先於入口落地——
          非 squash merge 之後 (A) 那個 head 可被 checkout 並執行,
          「PR 末端有 guard」不構成原子安全）
    p_authorization_issuer: `issued_by == "research_owner"` 原本在 controller 與 archive verifier
          各寫一份字面值 ⇒ 「誰可以授權」有兩個互不相干的答案（§C3.9）。改為
          `h2_measurement_evidence.AUTHORIZATION_ISSUER` 單一 authoritative constant,值未變;
          測試以行為(移動 authority ⇒ 兩個 validator 的判斷同步反向)而非字串比對來綁定
    b_corpus_admission_guard: rehearsal archive 與 canonical archive **同形且自洽**,
          archive verifier(修復後只讀 archive 位元組)必然判它 valid ⇒ 只有 corpus 層擋得住。
          新增 tracked anchor `docs/research/contracts/h2_controlled_host_execution_domain_v1.json`
          （內容即已在主線的 7646f421 archive 之 authorization_execution_domain.json,零新增揭露）;
          `check_h2_measure_archives.execution_domain_admission_reasons` 比較 **parsed object**
          （member set + 逐欄位值,格式化不是 identity）,兩邊都是 archive 位元組、**不觀察驗證主機**,
          host-independence 未被回退;僅限 Phase A（§C3.5.1 step 5 的 Phase-B consumption 形狀
          尚未被規範,在此判它等於發明契約);`test_research_packet_schema` 原本把 `valid=True`
          單獨當 canonical acceptance,已改為斷言 conjunction
    b_threat_boundary: 這是 provenance/admission guard,**不是** authority proof。它擋得住
          未修改的 rehearsal archive 與任何 execution domain 不同的自洽 archive;
          擋不住偽造——能重寫 grant/receipt/digest chain 的人也能寫入 anchor 的內容。
          不可偽造的簽發需要簽章機制,本 repo 不提供
    a_harness: `scripts/tools/h2_rehearse_measurement.py`（新檔,零 production 檔案修改;
          只用既有 seam `evidence_parent` / `authorization_ledger`);無 `--authorization`、
          無 `--invocation-id`（沒有任何參數可以餵進 owner 的 grant);grant 全部欄位在 call time
          由 authority 取得;隔離用 filesystem identity(resolve 既存 symlink、component containment
          非字串前綴、目標必須不存在、exclusive mkdir 0700、執行前/後/寫 witness 前重驗 pathname→inode)
    a_path_classification: harness 落地時的檔名不匹配 `h2_path_partition` 任何 plumbing prefix
          ⇒ 被判 `unclassified`,而 `unclassified` 在 Layer-P retry admissibility 是 fail-closed。
          `--base == head` 時 `changed_count == 0` 所以當下擋不到任何事,但只要日後某次
          Layer-P 的 base 早於一次 harness 修改,就會被一支只產診斷的檔案 block。
          2026-07-29 改名為 `scripts/tools/h2_rehearse_measurement.py`(匹配既有 `scripts/tools/h2_`
          prefix),**不動 `h2_path_partition.py`** —— 那是 ruler 檔,改它要 republish identity
          並重開 acceptance gate 2;改名只動 plumbing、測試與索引,不動任何發布座標軸。
          分類由 `test_h2_rehearsal_harness.test_the_harness_is_classified_plumbing_only` 綁住
    a_threat_boundary: 只承諾「啟動時的 lexical/symlink/ancestor alias 與意外重用會被拒絕,
          執行期間被抽換會被**檢出**」;不承諾抵抗同 UID 惡意並行程序的 rename/mount/symlink
          substitution——controller 透過 pathname 寫入,持有 dir fd 無法阻止,
          要真正阻止需要 openat-relative I/O 或 mount namespace,超出本工作項
    a_success_predicate: terminal is None **且** verifier valid **且** disposable ledger 內恰一份
          receipt 且其 id/digest 對得上 synthetic grant **且** 四個 ordered run 皆完成
          （由 archive 投影,不用 harness 自己的計數器)**且** 執行後 checkout hygiene 為空
          **且** witness 落地;harness invariant violation 與 rehearsal terminal 在 witness 中
          以 failure_class 區分
    a_witness: schema `h2_phase_a_rehearsal_witness_v1`;開跑前 exclusive-create `status: started`,
          結束時原子替換為 completed/failed ⇒ crash 不會留下「archive 看似完整卻無 rehearsal 標記」;
          安全輸出位置成立**之前**的拒絕(pre-witness refusal)一律 exit 2、不建立 witness、
          回滾已建立的目錄
    not_established: 無 seal、無 equivalence、無 capture、無新授權、無 registry object state 改變;
          harness 落地只代表入口存在,不代表 gate 已通過
    successor_work_item: h2_phase_a_rehearsal_execution — 在本 PR merge 後的 head 依序:
          重建 item 4（controlled-host run）與 item 5（Layer-P certificate)、建 F、
          執行一次 rehearsal（disposable ledger 與 evidence parent 皆在 repo 外,
          witness 只做 repo 外唯讀 custody copy;**不得在 Phase A 之前 commit witness**——
          任何 commit 都會移動 head 而使 F 與 certificate stale)、
          綠了才由 owner 另行簽發第三份授權。兩份既有授權仍永久 spent
    orphan_closure: 若 rehearsal 跑完而 owner 最終未簽發第三份授權,witness 不得無限期停在 repo 外
          custody:此時已無 head-bound 授權需要維持,改以 rehearsal-only registration 或
          abandonment 收編,兩條路徑擇一,不留無閉包的 custody
pending_reentry:                          # append-only; pre-seal, no terminal claimed; route-1 永久留帳結論不變
  - date: 2026-07-21
    scheduling: owner-scheduled re-entry #3（滿足 line-337 future_reentry_precondition:launch-hygiene gate 先行）
    declaration_amendment: "Amendment 9（headline_bridge_full_decision_capture_declaration_20260713.md;pre-seal, append-only）"
    scope: "re-admit 既有 sealed unit h0_build_tool_provenance_closure 供單一 fresh I→F→S attempt（前次於 #224/#227 I=31c9eee8 被 PROVENANCE_INVALID 消費,capture 前失敗）;非第二 unit;acceptance matrix/checker/workflow tuple/qualification 語義/歷史 declaration/歷史 sealed evidence 全 byte 不變"
    launch_hygiene_gate: "scripts/tools/h0_launch_hygiene_gate.py（非授權;複用單源 predicate run_h0_phase_a.assert_no_preexisting_build_tree）;mandatory:授權前與 sealed checkout controller launch 前皆須報 clear"
    status: "RESOLVED 2026-07-21 — owner sealed（S=3a6a9ec6）＋ exactly-once authorized ＋ scheduled ＋ executed ⇒ 到達 owner-accepted ordered terminal H0_PROVENANCE_INVALID（見 reentry_terminal_history re-entry #3;PR #235 comment 5032610430）;exactly-once authorization consumed;exact S permanently spent;retry/resume/second invocation forbidden;無 repair / 新 re-entry 授權"
  - date: 2026-07-24
    scheduling: owner-scheduled H0 R4 Repair（re-entry #4 pre-seal engineering; not an execution authorization）
    declaration_amendment: "Amendment 10（headline_bridge_full_decision_capture_declaration_20260713.md;append-only; authority-overlay/runtime-binding separation）"
    scope: "sole repair unit h0_authority_overlay_runtime_binding_split_v1 — remove declaration from runtime-bound repository inventory; bind it only via h0_owner_authority_overlay_v1 with S-byte continuous monitoring; re-admit exactly one future fresh I→F→S after qualification; does not select I / create F or S / authorize execution / establish guarantee"
    spent_chain_boundary: "S3=3a6a9ec6 permanently spent; no retry/resume/re-interpretation"
    registration_v3_downstream: "h0_gctm_guarantee_registration_v3 / gctm_runtime_native_candidate_universe_v1 structurally reachable after successful Phase-A; no actual guarantee in this Repair"
    launch_hygiene_gate: "scripts/tools/h0_launch_hygiene_gate.py retained; single-source predicate run_h0_phase_a.assert_no_preexisting_build_tree"
    status: "RESOLVED 2026-07-24 — mechanical terminal H0_R4_REPAIR_QUALIFIED_SEALABLE; controlled-host qualification passed; WIP released; exact qualified head eligible for separate Seal PR only; no I/F/S, no execution, no actual guarantee"
    qualified_head_binding: "see docs/modules/semantic/research/evidence/h0_r4_authority_overlay_runtime_binding_split_20260724/qualification_report.json"
  - date: 2026-07-24
    scheduling: owner-scheduled H0-R4 Phase-A exactly-once execution under sealed S（post #277 Seal; authorization #278）
    chain:
      I4: 2a233387a6a321dd43570e2e30dc718571b3b4f4
      F4: ced4a4cc6a71473dcb1225203e6d59df0437d976
      S4: a76efffa01a6fb731218150c355f5859bb8e6dd4
    authorization_identity: h0_r4_phase_a_exactly_once_authorization_20260724
    launch_hygiene_gate: "scripts/tools/h0_launch_hygiene_gate.py; first+second gate clear before launch"
    status: "EXECUTED + OWNER-ACCEPTED 2026-07-24 — controller launched once under exact S; authorization consumed; controller_result=provenance_invalid; mechanical_disposition=H0_PROVENANCE_INVALID; independent verifier valid=true; exact S permanently spent; retry/resume/second invocation forbidden; Phase B not authorized; actual guarantee not established; owner acceptance of truthful-negative terminal only via PR #279 merge (55d2da47…); no repair/reseal/new re-entry authorized by this record"
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4/
    witness_bundle: docs/modules/semantic/research/evidence/h0_r4_phase_a_execution_witness_20260724/
  - date: 2026-07-24
    scheduling: owner-scheduled H0-R5 Repair（extension/plugin runtime-attestation closure; not an execution authorization）
    owner_decision_identity: h0_r5_extension_plugin_attestation_closure_authorization_20260724
    owner_decision_surface: "https://github.com/raylei50653/saccade/issues/280"
    repair_unit: h0_extension_plugin_runtime_attestation_closure_v1
    spent_chain_boundary: "S4=a76efffa permanently spent; no retry/resume/re-interpretation of S4; no reuse of F4/S4; no Phase B from R4 packet"
    authorized: "one repair unit; controlled-host non-authoritative qualification; fresh future Seal eligibility if qualification passes"
    not_authorized: "F/S creation; Phase-A execution; exactly-once authorization; Phase B; actual registration-v3 guarantee; runtime compatibility; H0_ROUTE5_B1 / GCTM_B1 / O1 activation"
    status: "RESOLVED 2026-07-24 — mechanical terminal H0_R5_ATTESTATION_QUALIFIED_SEALABLE; controlled-host qualification passed; exact qualified head eligible for separate Seal PR only; no I/F/S, no execution, no actual guarantee"
    qualified_head_binding: "see docs/modules/semantic/research/evidence/h0_r5_extension_plugin_attestation_closure_20260724/qualification_report.json"
  - date: 2026-07-25
    scheduling: owner-scheduled H0-R5 Phase-A exactly-once execution under sealed S（authorization #283）
    chain:
      I5: 524f7e3b88f73bc366d467d53a2c393a7d3ba937
      F5: 6e425dc6f89a15d4eb43d3889a517d632f0ee39e
      S5: 6fdb060c50c9ed784a3fa2229b1ea2514dd1af5e
    authorization_identity: h0_r5_phase_a_exactly_once_authorization_20260725
    launch_hygiene_gate: "scripts/tools/h0_launch_hygiene_gate.py; first+second gate clear before launch"
    status: "EXECUTED 2026-07-25 — controller launched once under exact S; authorization consumed; controller_result=provenance_invalid; mechanical_disposition=H0_PROVENANCE_INVALID; independent verifier valid=true; exact S permanently spent; retry/resume/second invocation forbidden; Phase B not authorized; actual guarantee not established; no repair/reseal/new re-entry authorized by this record"
    evidence_packet: docs/modules/semantic/research/evidence/h0_phase_a_524f7e3b88f73bc366d467d53a2c393a7d3ba937/
    witness_bundle: docs/modules/semantic/research/evidence/h0_r5_phase_a_execution_witness_20260725/
last_reviewed_at: 2026-07-25
```

#### ADR 020 terminal slot (per-study owner)

```yaml
study_id: p0_decision_path_identifiability
line_type: scoped-empirical
claim_verdict: NOT_IDENTIFIABLE
decision_outcome: NOT_ASSESSED
lifecycle_disposition: SEALED
verdict_locus:
  assumptions: scope-corrected target is headline-m; provenance lacks h_lo, h_hi, spatial_gate, max_speed, and a capture-time kernel hash
  domain: the frozen D0/R1/S0 packets; not an assertion that their capture semantics are invalid
  protocol_ref: P0 decision-path identifiability declaration (runtime_bridge_decision_path_identifiability_declaration_20260713)
evidence_owner: docs/research/contracts/claim_state_registry.md
process_disposition: retained
```

### `score.ambiguous_band_ranking_class`

```yaml
layer: L2 score
ladder: study-sealed legacy terminal; future transitions use score_ranking_evidence_contract_v1
                                         transition_semantics: defined
state: T2_NO_USABLE_RANKING_POWER_IN_CLASS (accepted)
substrate: s0 proxy 空間                 target_substrate: runtime
open_limits: [class-scoped:只封 12 members;AND pair / 連續訊號 / 有限 λ / learned score 未耗盡]
blockers: []
decision_relevance:
  role: **反向約束** — proxy 空間的 ranking headroom 已封;score 線若開,必須在 runtime 座標上
supporting_declaration: ../../modules/semantic/research/door0_ranking_probe_results_20260712.md
last_transition: 2026-07-12 (#136 accepted)
admissible_units: []                     # class 已封;重開需新宣告
derived_from: accepted legacy terminal + score_ranking_evidence_contract_v1 §6/§7（不追溯改寫既有 terminal）
last_reviewed_at: 2026-07-23
```

### `score.anchor_propagation`（前 discrete-\(M\)）

```yaml
layer: L2 score                          # 曾被誤標為 gate
ladder: score_ranking_evidence_contract_v1
                                         transition_semantics: defined
state: none                              # 未 seal,無被接受的 state
substrate: —
open_limits: [宣告的 horizon {1,2,4,8} 與 consumer 脫節（la median 12、p90 26）]
blockers:
  - type: inadmissibility
    what: 以 gate 身分執行
    clause: §4.1 evidence 型別錯層（margin/rank ≠ gate 證據）
  - type: dependency
    what: 作為 score feature 執行
    clause: score_ranking_evidence_contract_v1 §2.3/§5 — 必須先凍結 candidate universe 並通過 decision-relevance 反事實測試
    depends_on: [保留域先被定義]
decision_relevance:
  status: **zero（目前）** — 保留域尚未定義,排序無意義;正反結果都不改變任何 production 決策
  # ⇒ §5 反事實測試不通過 ⇒ 不得取得 WIP 鎖
supporting_declaration: ../../modules/semantic/research/discrete_m_capability_declaration_20260712.md（parked, unsealed）
last_transition: 2026-07-12 — reclassified gate → score feature;parked
admissible_units: []                     # [cached derivation]
derived_from: §4.1 錯層 + score_ranking_evidence_contract_v1 §2.3/§5 + registry §5 relevance zero
last_reviewed_at: 2026-07-23
```

### `quantity.h0_gctm_interface_static_feasibility`

```yaml
layer: quantity（cross-module static producer→consumer feasibility；非 runtime compatibility object，非 B1 candidate）
ladder: ordered H0_GCTM static-feasibility terminal family
transition_semantics: defined
lifecycle_state: terminal
state: H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT
terminal_acceptance:
  owner_acceptance_id: h0_gctm_static_audit_terminal_owner_acceptance_20260723
  acceptance_date: 2026-07-23
  acceptance_mechanism: PR merge of the exact mechanically validated packet
substrate: none（static ABI/schema/interface identities only；no H0 capture）
target_substrate: future H0→GCTM bridge-runtime evidence path
authority_class: static_feasibility_only
frozen_inputs:
  - GCTM_D1_INTERFACE_READY / gctm_d1_terminal_owner_acceptance_20260723
  - gctm_d1_substrate_agnostic_ranking_v1_consumer_interface @ 45c16a6c8cf50d098b12cc8e4f1acbdcc846d0d431ad77f9fbe64bdb60bb57ce
  - h0_bridge_decision_trace_v2 @ 5ea8bea16ab6f6916baebd486c3256100427ae823e2f7bac021ff1284254f9a2
  - h0_gctm_guarantee_registration_v2 @ 63af2eba012a248d769cde225ce28db404cefe592edc995a12d86af307419ff2
result:
  physical_gap: legal GCTM_DERIVED relation g_phys = la - bridge_at + 1；la / g_phys / bridge_at / operator offset 保持分離
  residual: legal GCTM_DERIVED M1/Hx construction with M2 inactive；fwd_r/bwd_r 不作 proxy
  covariance: GCTM_DERIVED from declared P0/gamma/D/R1 artifacts；非 H0 guarantee
  context: M2 inactive / null drift only；nonzero context source unavailable
  unresolved_runtime_objects: [candidate_universe, event_membership]
  terminal_reason: accepted D1 universe 是 synthetic_event_candidate_set_v1；registration-v2 無 envelope/native-universe completeness binding
blockers:
  - type: inadmissibility
    what: 以 unchanged ABI + registration-v2 重複 H0 capture 來解鎖 runtime B1
    clause: static producer→consumer evidence path 在 candidate universe / event membership 前已中斷；增加 capture rows 不補 semantic/registration edge
  - type: dependency
    what: 未來重新評估 H0 runtime-substrate re-entry
    clause: owner 先選 minimal H0 registration/fidelity-edge delta 或 GCTM runtime consumer re-charter
decision_relevance:
  status: terminal decision-changing mainline — feasible 才可考慮新 H0 runtime-substrate path；insufficient ⇒ unchanged-interface re-entry 不可排程
  selected_consequence: do not authorize new capture under unchanged ABI/registration-v2
runtime_consumer_gates:
  H0_ROUTE5_B1: missing（unchanged）
  GCTM_B1: missing（unchanged；與前者獨立）
not_established:
  - accepted H0 baseline
  - field-level runtime guarantees
  - runtime fidelity / compatibility / substrate
  - H0_ROUTE5_B1 / GCTM_B1 / O1 activation
wip_history: 2026-07-23 sole-active acquired for bounded audit → terminal selected → WIP released
supporting_charter: ../threads/closed/h0_gctm_interface_static_feasibility_audit_20260723.md
supporting_packet: ../../modules/semantic/research/evidence/h0_gctm_interface_static_feasibility_20260723/
accepting_review: h0_gctm_static_audit_terminal_owner_acceptance_20260723
last_transition: 2026-07-23 — mechanical static terminal structurally insufficient；no capture/re-entry/registration/compatibility/gate/activation effect
admissible_units: []
derived_from: frozen responsibility matrix + immutable derivations + coverage conservation + ordered terminal
last_reviewed_at: 2026-07-23
```

### `quantity.gctm_runtime_native_candidate_universe`

```yaml
layer: quantity（runtime-consumer candidate-universe / event-composition contract；非 H0 guarantee、非 runtime substrate、非 B1 activation）
ladder: ordered GCTM_RUNTIME_UNIVERSE terminal family
transition_semantics: defined
lifecycle_state: terminal
state: GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE
terminal_acceptance:
  owner_acceptance_id: gctm_runtime_universe_terminal_owner_acceptance_20260724
  acceptance_date: 2026-07-24
  acceptance_mechanism: PR merge of the exact mechanically validated packet
substrate: none（consumer contract over frozen trace-v2 identities only；no H0 capture）
target_substrate: future GCTM_B1 runtime candidate-universe binding after separate gates
authority_class: runtime_consumer_contract_only
runtime_consumer_identity: gctm_runtime_native_candidate_universe_v1
frozen_inputs:
  - H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT / h0_gctm_static_audit_terminal_owner_acceptance_20260723
  - score_ranking_evidence_contract_v1 @ 7dbc2d965079fa3fc13f7802a4a083b1c4cbf49d658ffe3728b6c405364a13b4
  - h0_bridge_decision_trace_v2 @ 5ea8bea16ab6f6916baebd486c3256100427ae823e2f7bac021ff1284254f9a2
  - GCTM_D1_INTERFACE_READY (closed read-only; synthetic universe not replaced)
result:
  event_key: gctm_runtime_event_key_v1 = (seq, frame, lost_slot, lost_instance_uid, event_key_version)
  candidate_key: gctm_runtime_candidate_key_v1 = (event_key, cand_slot, cand_instance_uid)
  inclusion_stage: pre_score_eligible_v1 (score-independent)
  composition_and_completeness: defined as consumer semantics only
  registration_requirements: h0_native_universe_completeness_registration_requirements_v1 (requirements-only)
  maximum_conclusion: suitable consumer target for a separate minimal H0 registration/fidelity-edge delta
blockers:
  - type: inadmissibility
    what: treat this packet as H0 completeness guarantee / runtime substrate / B1 activation
    clause: consumer contract only; fixed non-authority outputs remain false
  - type: dependency
    what: producer-side completeness registration
    clause: registration-v2 cannot bind envelope/native-universe completeness; producer path now owns quantity.h0_native_universe_completeness_registration (v3 contract sealed 2026-07-24; actual guarantee still absent)
decision_relevance:
  status: terminal decision-changing mainline for consumer universe definition — positive path enables a precisely scoped H0 registration-v3 delta; negative would have ruled out the current bridge hook as a score-ranking universe
  selected_consequence: freeze consumer target; do not auto-authorize H0 implementation
runtime_consumer_gates:
  H0_ROUTE5_B1: missing（unchanged）
  GCTM_B1: missing（unchanged；與前者獨立）
not_established:
  - H0 completeness guarantee
  - runtime fidelity / compatibility / substrate
  - actual H0 guarantee registration
  - H0_ROUTE5_B1 / GCTM_B1 / O1 activation
wip_history: 2026-07-24 sole-active acquired for consumer re-charter → terminal selected → WIP released
supporting_charter: ../threads/closed/gctm_runtime_native_candidate_universe_task.md
supporting_packet: ../../modules/semantic/research/evidence/gctm_runtime_native_candidate_universe_20260724/
accepting_review: gctm_runtime_universe_terminal_owner_acceptance_20260724
last_transition: 2026-07-24 — mechanical runtime-universe terminal sealable as consumer contract；no capture/re-entry/registration/compatibility/gate/activation effect
admissible_units: []
derived_from: frozen score-policy spaces + pre-score inclusion proof + composition/completeness contract + registration requirements surface + ordered terminal
last_reviewed_at: 2026-07-24
```

### `quantity.h0_native_universe_completeness_registration`

```yaml
layer: quantity（H0 registration contract for native-universe completeness；非 actual guarantee、非 runtime substrate、非 compatibility verdict、非 B1 candidate）
ladder: ordered H0_REGISTRATION_V3 terminal family
transition_semantics: defined
lifecycle_state: terminal
state: H0_REGISTRATION_V3_CONTRACT_SEALABLE
terminal_acceptance:
  owner_acceptance_id: h0_registration_v3_terminal_owner_acceptance_20260724
  acceptance_date: 2026-07-24
  acceptance_mechanism: PR merge of the exact mechanically validated registration-v3 contract
substrate: none（registration contract over frozen trace-v2 + consumer universe identities only；no H0 capture）
target_substrate: future owner-accepted H0 baseline that may register universe_completeness guarantees
authority_class: registration_contract_only
registration_identity: h0_gctm_guarantee_registration_v3
consumer_universe_id: gctm_runtime_native_candidate_universe_v1
guarantee_class: universe_completeness
consumer_objects: [runtime_candidate_universe, runtime_event_membership]
frozen_inputs:
  - GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE / gctm_runtime_universe_terminal_owner_acceptance_20260724
  - h0_native_universe_completeness_registration_requirements_v1 @ 2b4086e4aff4f185a995220201c36e4c19b94dc752d8382107ade15ac69e967f
  - h0_bridge_decision_trace_v2 @ 5ea8bea16ab6f6916baebd486c3256100427ae823e2f7bac021ff1284254f9a2
  - h0_gctm_guarantee_registration_v1 / v2 (frozen; not modified)
result:
  schema: scripts/tools/h0_gctm_guarantee_registration_schema_v3.json
  validator_dispatch: independent v3 path in verify_h0_gctm_guarantee_registration.py
  event_universe_sidecar: registration-level source type binding frozen trace-v2 envelope fields only
  completeness_predicate: h0_native_universe_completeness_predicate_v1 (15 mechanical checks)
  trace_v2_abi_change_required: false
  maximum_conclusion: registration-v3 is structurally capable of registering future H0 native-universe and event-membership completeness guarantees for gctm_runtime_native_candidate_universe_v1
blockers:
  - type: inadmissibility
    what: treat sealable registration-v3 as an actual H0 guarantee / accepted baseline / capture authorization
    clause: contract only; authority_verified and structurally_usable semantics remain non-authoritative until a separate owner-controlled acceptance registry/packet binding
  - type: dependency
    what: actual H0 baseline capture and owner-accepted completeness guarantee registration
    clause: next owner design for H0 re-entry + actual baseline capture; candidate-source fixture remains fixture_only
decision_relevance:
  status: terminal decision-changing mainline for the registration contract — sealable permits owner consideration of a future H0 runtime-substrate re-entry design; requires-ABI-delta would forbid capture until producer delta accepted; invalid would require contract repair
  selected_consequence: seal registration-v3 contract; do not authorize capture/re-entry/actual guarantee
runtime_consumer_gates:
  H0_ROUTE5_B1: missing（unchanged）
  GCTM_B1: missing（unchanged；與前者獨立）
not_established:
  - actual H0 guarantee
  - accepted runtime baseline
  - runtime substrate
  - runtime compatibility
  - H0 re-entry authority
  - H0_ROUTE5_B1 / GCTM_B1 / O1 activation
  - production claim
wip_history: 2026-07-24 sole-active acquired for registration-v3 seal → terminal selected → WIP released
supporting_charter: ../threads/closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md
supporting_packet: ../../modules/semantic/research/evidence/h0_gctm_guarantee_registration_v3_20260724/
accepting_review: h0_registration_v3_terminal_owner_acceptance_20260724
last_transition: 2026-07-24 — mechanical registration-v3 terminal sealable without trace-v2 ABI delta；no capture/actual-guarantee/substrate/compatibility/gate/activation effect
admissible_units: []
derived_from: frozen consumer universe + requirements surface + trace-v2 envelope coverage + additive schema/validator + ordered terminal
last_reviewed_at: 2026-07-24
```

### `quantity.gap_conditioned_transition_model.a_layer_spec`

```yaml
layer: quantity（非決策層 — A 層 latent transition specification;不是 gate,也不是可插入的 score）
ladder: study-sealed terminal（GCTM charter WP-A0 凍結的 5-terminal ordered partition）
                                         transition_semantics: defined（該 charter 自帶機械判準）
state: GCTM_MODEL_SPEC_SEALABLE (accepted)   # diagnostic-only spec seal;非 production claim
substrate: substrate-agnostic A 層數學（canonical state \(\mathbb R^{2d}\);無任何 runtime 擷取值）
target_substrate: runtime bridge `S_A`（active CUDA bridge geometry）
open_limits:
  - identifiability **specified ≠ established**（\(\gamma\) regime 未證;\(P_{xx}\leftrightarrow R_1\) gauge 與 \(H_x\) 下 \(\mathrm{asym}(P_{xv})\) 結構性不可識別）
  - 無 runtime substrate／accepted fidelity edge（H0 envelope 空）;所有 production-facing 欄位仍是 declared-target
  - CAL／RANK 只有定義、null 與 metric family,**無任何 measured gain**
  - canonical \(C=0\) 是宣告決定;dependent-error path 的可逆性需額外假設
  - D1 §8 只列 requirement,不宣稱 runtime availability
  - dimensional consistency 由 cross-reference 而非獨立 lemma（terminal review J-1）
blockers:
  - type: inadmissibility
    what: 以 bridge-runtime claim 消費本 spec
    clause: §4.2 substrate 不繼承（substrate ≠ target_substrate,且無 accepted fidelity edge / L4）
  - type: dependency
    what: 作為 score-ranking 層模型進入 B1/O1
    clause: score_ranking_evidence_contract_v1 已可提供 transition semantics;仍需 accepted runtime substrate／fidelity edge ＋ consumer compatibility verdict ＋ B1-slot identity ＋ sealed B1 declaration ＋ owner scheduling
decision_relevance:
  status: **zero（目前）** — 無 runtime substrate;terminal 5 與 terminal 3/4 在當前狀態下**都不**改變任何 production 決策 ⇒ §5 反事實測試不通過 ⇒ 不得取得 WIP 鎖
  role: enabling precondition only — 它固定的是「未來若被授權,B1 可引用哪一份 frozen 模型介面與其 regime」,不是 production 行為
supporting_declaration: ../models/gap_conditioned_stochastic_transition_spec_v1.md（D1,§2–§8 frozen）· ../models/gap_conditioned_stochastic_transition_lemmas_v1.md（D2,L1–L5）
accepting_review: ../models/gap_conditioned_stochastic_transition_terminal_review_v1.md（WP-A8 checklist ＋ 機械 selection）· charter *Final status*: ../threads/closed/gap_conditioned_stochastic_transition_model_task.md
last_transition: 2026-07-23 — 本 object 第一個被接受的 terminal（WP-A8;Issue #175）
admissible_units: []                     # [cached derivation]
derived_from: §4.2（substrate 不繼承,無 L4）＋ score_ranking_evidence_contract_v1 §10 consumer boundary ＋ §5（relevance zero）
last_reviewed_at: 2026-07-23
```

### `score.h0_route5_runtime_b1`

```yaml
slot_id: H0_ROUTE5_B1
layer: L2 score
ladder: score_ranking_evidence_contract_v1
transition_semantics: defined
lifecycle_state: proposed
state: none                              # proposed 不是 accepted rung
substrate: none
target_substrate: H0 runtime capture
authority_class: runtime_grounded
blocked_by: h0_runtime_substrate
blockers:
  - type: inadmissibility
    what: activation 或 decision-relevant candidate transition
    clause: 缺 valid H0 runtime substrate、stable evidence identity、canonical checksum 與 owner-accepted H0→GCTM consumer compatibility verdict
activation_forbidden_until:
  - valid_h0_runtime_substrate
  - stable_evidence_identity
  - canonical_checksum
  - h0_gctm_consumer_compatibility_verdict
  - observation_freeze
  - parameterization_freeze
  - sealed_b1_declaration
  - owner_scheduling
decision_relevance:
  status: zero（目前）— route-5 positive substrate 不存在;score contract acceptance 不建立 substrate 或候選
supporting_declaration: ../threads/closed/bridge_frozen_evidence_o0_routing_20260716.md（route 5）
accepting_review: gctm_b1_slot_identity_decision_v1.json（identity/coexist only;非 charter acceptance）
last_transition: 2026-07-23 — owner 接受 slot identity/coexist 關係;slot lifecycle 不變
admissible_units: []
derived_from: §4.2 substrate 不繼承 + score_ranking_evidence_contract_v1 §10 + machine identity record
last_reviewed_at: 2026-07-23
```

### `score.gctm_b1_runtime_grounded_ranking`

```yaml
slot_id: GCTM_B1
layer: L2 score
ladder: score_ranking_evidence_contract_v1
transition_semantics: defined
lifecycle_state: proposed
state: none
substrate: none
target_substrate: H0 runtime capture
authority_class: runtime_grounded
blocked_by: h0_runtime_substrate
blockers:
  - type: inadmissibility
    what: activation 或 decision-relevant candidate transition
    clause: 與 H0_ROUTE5_B1 coexist 但不共享 authority;缺 runtime substrate、identity、checksum 與 compatibility verdict
decision_relevance:
  status: zero（目前）— theory seal 與 score contract 均不提供 runtime substrate
supporting_declaration: ../threads/gctm_b1_runtime_grounded_offline_attribution_task.md
accepting_review: gctm_b1_slot_identity_decision_v1.json（identity/coexist only）
last_transition: 2026-07-23 — identity ambiguity resolved;proposed lifecycle 不變
admissible_units: []
derived_from: machine identity record + §4.2 + score_ranking_evidence_contract_v1 §10
last_reviewed_at: 2026-07-23
```

### `diagnostic.gctm_d1_substrate_agnostic_ranking`

```yaml
slot_id: GCTM_D1
layer: diagnostic-only（非 runtime decision layer）
ladder: ordered GCTM_D1 terminal family
transition_semantics: defined（local diagnostic terminal only；no cross-slot unlock）
lifecycle_state: terminal
state: GCTM_D1_INTERFACE_READY
declaration_acceptance:
  terminal: GCTM_D1_DECLARATION_ACCEPTED
  owner_acceptance_id: gctm_d1_declaration_owner_acceptance_20260723
  acceptance_date: 2026-07-23
  activation_requirement_id: declaration_owner_acceptance
  evidence_class: owner_accepted_governance
  freezes: sealed declaration + diagnostic policy + synthetic input + I1–I12 + consumer interface + compatibility-requirements identity + exhaustive terminal procedure
  does_not_equal_execution: true
  does_not_create_decision_relevant_candidate: true
  does_not_promote_provisional_terminal: true
  note: declaration acceptance id is not slot.owner_acceptance_id
owner_scheduling:
  scheduling_id: gctm_d1_owner_scheduling_20260723
  evidence_class: owner_scheduling_decision
  owner_acceptance_id: gctm_d1_activation_owner_acceptance_20260723
  acceptance_date: 2026-07-23
  decision: schedule GCTM_D1 for one canonical execution
  scope: one canonical execution of the exact declaration and packet identities accepted by gctm_d1_declaration_owner_acceptance_20260723
  artifact: ../../modules/semantic/research/gctm_d1_owner_scheduling_20260723.json
  artifact_sha256: 17c534c5dc25ad0318d26b0208a7f9dceedc603ec1f81dc29fe27b49bc2607c8
  consumed_by_execution: gctm_d1_canonical_execution_20260723
  does_not_create_decision_relevant_candidate: true
supporting_execution:
  execution_id: gctm_d1_canonical_execution_20260723
  execution_commit: d80f53389a4b3f4a9f8ea83f80c5133ba9602451
  artifact: ../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/execution_witness.json
  bit_identical_to_accepted_packet: true
  invariants: 12/12
  mechanical_selected_terminal: GCTM_D1_INTERFACE_READY
terminal_acceptance:
  selected_terminal: GCTM_D1_INTERFACE_READY
  owner_acceptance_id: gctm_d1_terminal_owner_acceptance_20260723
  acceptance_date: 2026-07-23
  acceptance_mechanism: PR merge
  artifact: ../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/terminal_acceptance.json
  terminal_procedure_id: gctm_d1_mechanical_three_way_terminal_v1
  maximum_supported_claim: >-
    The frozen substrate-agnostic diagnostic family is internally
    machine-checkable and exposes a complete consumer interface suitable
    for a separate runtime compatibility review.
  blocked_claims:
    - runtime compatibility
    - runtime fidelity
    - H0 substrate
    - H0 re-entry authority
    - H0_ROUTE5_B1 activation
    - GCTM_B1 activation
    - O1 eligibility
    - decision-relevant candidate
    - production claim
  note: terminal acceptance id is independent of slot.owner_acceptance_id (activation)
seal_candidate:
  status: SEAL_CANDIDATE_GENERATED
  generation_kind: pre_activation_synthetic_seal_candidate
  provisional_terminal: GCTM_D1_INTERFACE_READY
  authority: sealed packet identities frozen by declaration acceptance; promoted to canonical terminal only via supporting_execution + terminal_acceptance
  packet: ../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/
  declaration: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md
  terminal_report: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md
substrate: synthetic fixture pack gctm_d1_synthetic_fixture_pack_v1（non-runtime）
target_substrate: none
authority_class: diagnostic_only
blocked_by: []
blockers:
  - type: inadmissibility
    what: runtime claim、runtime B1 transition、O1 unlock 或 decision-relevant candidate
    clause: diagnostic evidence 不可滿足 runtime substrate/provenance/identity/checksum/compatibility/activation authority
decision_relevance:
  status: zero as decision-relevant candidate — diagnostic terminal only；does not enter decision_relevant_candidates；does not unlock B1/O1/H0
supporting_declaration: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md
supporting_terminal: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md
supporting_packet: ../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/
supporting_scheduling: ../../modules/semantic/research/gctm_d1_owner_scheduling_20260723.json
supporting_execution: ../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/execution_witness.json
supporting_terminal_acceptance: ../../modules/semantic/research/evidence/gctm_d1_canonical_execution_20260723/terminal_acceptance.json
charter_ref: ../threads/closed/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md
accepting_review: gctm_d1_terminal_owner_acceptance_20260723
last_transition: 2026-07-23 — canonical execution completed; mechanical terminal GCTM_D1_INTERFACE_READY owner-accepted; lifecycle terminal; WIP released; no B1/O1/H0/runtime-compatibility change
admissible_units: []
derived_from: gctm_b1_slot_identity_decision_v1.json terminal policy + owner-accepted declaration + owner scheduling + canonical execution witness + terminal acceptance + §5 relevance
last_reviewed_at: 2026-07-23
```
---

## 7. 架構缺口（顯式化，而不是假裝可編排）

| 缺口 | 影響 |
|---|---|
| **score-layer transition semantics** | **RESOLVED 2026-07-23.** [`score_ranking_evidence_contract_v1`](score_ranking_evidence_contract.md) 已由 owner 接受並凍結；`owner_acceptance_id: score_ranking_contract_owner_acceptance_20260723`；`registry_binding_id: claim_state_registry_score_ranking_v1`；`contract_sha256: 7dbc2d965079fa3fc13f7802a4a083b1c4cbf49d658ffe3728b6c405364a13b4`。本 binding 只使 L2 admissibility 可判定；不自動推進 object、產生候選、啟動 B1/O1 或授權 runtime 行為。 |
| **B1-slot identity** | **RESOLVED 2026-07-23.** [`gctm_b1_slot_identity_decision_v1`](gctm_b1_slot_identity_decision_v1.json) 固定 `GCTM_B1 != H0_ROUTE5_B1`、`relation: coexist`、非 alias、非 supersede、不可共享 activation authority；未來改寫關係須另開 owner-accepted transition。Identity resolution 不解除任何 runtime gate。 |
| **H0 runtime substrate / compatibility** | **OPEN / fail closed；unchanged-interface re-entry blocked.** 五個 mechanical `H0_PROVENANCE_INVALID` terminals（含 R4 via PR #279 與 R5 S5 via #283）後仍無 valid runtime substrate、stable evidence identity、canonical checksum 或 owner-accepted H0→GCTM compatibility verdict。R5 於 extension_load 重現 `extension/plugin load is absent from runtime attestation`（preseal tool_runtime parity 通過）。2026-07-23 bounded static audit 選出 `H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT`。2026-07-24 consumer re-charter 凍結 `gctm_runtime_native_candidate_universe_v1`（`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`）。同日 additive registration-v3 選出 **`H0_REGISTRATION_V3_CONTRACT_SEALABLE`**（`quantity.h0_native_universe_completeness_registration` / `h0_gctm_guarantee_registration_v3`）：可 fail-closed 描述 future `universe_completeness` for `runtime_candidate_universe` + `runtime_event_membership`，且 **不需** trace-v2 ABI delta。registration-v1/v2 保持 frozen；**仍無** actual H0 guarantee、runtime substrate 或 compatibility verdict。`H0_ROUTE5_B1` 與 `GCTM_B1` 均保持 proposed；兩個 compatibility gate 各自 `missing`。下一個 owner 決策優先為 **separate repair design for extension/plugin runtime attestation**（不得重試 S5）；不得把 registration-v3 seal 直接交接成 accepted guarantee / capture authority。 |

---

## 8. 候選集 → O0 選擇（registry 到此為止）

**registry 產生的合法候選集（重推於 2026-07-23；前次 2026-07-20）：**

| 候選 | Object | 為何合法 |
|---|---|---|
| *(全空)* | 所有 object | 已達 terminal／被 inadmissibility 排除／依賴未解／relevance zero。前次唯一成員（H0 pre-seal，`quantity.bridge_capture_provenance` 的 §4.3 dependency）已被 O0 取用並執行完畢：exactly-once sealed invocation 的 ordered terminal＝`H0_PROVENANCE_INVALID`，owner-accepted（#209，2026-07-19）⇒ unit 消費、候選集回空。 |
| *(仍全空，2026-07-23 L2 binding 後重推)* | `score.anchor_propagation` | Transition semantics 現已 defined，但保留域／candidate universe 尚未定義，且 §5 decision relevance 仍為 zero；不得因契約生效而取得 WIP 鎖。 |
| *(仍全空，2026-07-23 L2 binding 後重推)* | `quantity.gap_conditioned_transition_model.a_layer_spec` | Accepted state 仍是 diagnostic-only spec seal。L2 contract blocker 已解除，但 consumers 仍受 §4.2（substrate 不繼承、無 fidelity edge）、consumer compatibility、B1-slot、declaration／seal／scheduling 與 §5 relevance zero 阻擋。GCTM charter 已關閉，semantic WIP 鎖維持空。 |
| *(仍全空，2026-07-23 identity decision 後重推)* | `H0_ROUTE5_B1` / `GCTM_B1` | Relation 已固定為 coexist，但兩者仍各自 `proposed`，不共享 authority，且明確 `blocked_by: h0_runtime_substrate`。缺 substrate／identity／checksum／compatibility verdict／freeze／declaration／scheduling；identity 與 L2 contract acceptance 均不產生候選。 |
| *(仍全空，2026-07-23 D1 charter 建立後重推)* | `GCTM_D1` | 新 charter 仍 `proposed`、diagnostic-only。其 terminal policy 不解鎖 runtime B1/O1、不產生 decision-relevant candidate。 |
| *(仍全空，2026-07-23 D1 seal-candidate 後重推)* | `GCTM_D1` | 僅有 pre-activation synthetic **seal-candidate**（provisional terminal string `GCTM_D1_INTERFACE_READY`）；canonical `state` 仍 `none`。**不**滿足 runtime gates；**不**進入決策候選集。 |
| *(仍全空，2026-07-23 D1 declaration acceptance 後重推)* | `GCTM_D1` | Declaration 已 `GCTM_D1_DECLARATION_ACCEPTED`（`gctm_d1_declaration_owner_acceptance_20260723`）；execution contract 凍結；`blocked_by: owner_scheduling`；canonical `state` 仍 `none`。**declaration acceptance ≠ execution**；**不**產生 decision-relevant candidate；**不**取得 WIP；runtime gates 仍 `missing`。 |
| *(仍非 decision-relevant，2026-07-23 D1 owner scheduling 後重推)* | `GCTM_D1` | Owner scheduling 已接受（`gctm_d1_owner_scheduling_20260723` / `gctm_d1_activation_owner_acceptance_20260723`）；slot **active**；`active_wip: [GCTM_D1]`；one canonical execution authorized。canonical `state` 仍 `none`；**不**進入 `decision_relevant_candidates`；runtime gates 仍 `missing`；execution / terminal / closure 屬後續獨立 PR。 |
| *(仍全空，2026-07-23 D1 canonical execution closure 後重推)* | `GCTM_D1` | Canonical execution 完成；mechanical terminal **`GCTM_D1_INTERFACE_READY`** owner-accepted（`gctm_d1_terminal_owner_acceptance_20260723`）；`lifecycle_state: terminal`；`active_wip: []`。**不**進入 `decision_relevant_candidates`；runtime gates 仍 `missing`；**不**授權 H0 re-entry 或 B1/O1 activation。 |
| *(仍全空，2026-07-23 H0→GCTM static audit closure 後重推)* | `quantity.h0_gctm_interface_static_feasibility` | Decision-changing mainline 已完成並選出 **`H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT`**；WIP 已釋放，object 本身不是 runtime compatibility object 或 B1 candidate。結論阻止 unchanged-interface H0 capture/re-entry 自動排程，但不產生新的 decision-relevant implementation candidate；owner 必須另選 registration/fidelity-edge delta 或 consumer re-charter。 |
| *(仍全空，2026-07-24 runtime universe contract closure 後重推)* | `quantity.gctm_runtime_native_candidate_universe` | Consumer re-charter 完成並選出 **`GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`**；凍結 `gctm_runtime_native_candidate_universe_v1`（pre-score lost-centric event/candidate keys + composition/completeness semantics + registration-v3 requirements-only surface）。WIP 已釋放。此 object **不是** H0 guarantee、runtime substrate、compatibility verdict 或 B1 candidate；**不**進入 `decision_relevant_candidates`；**不**授權 capture/re-entry/actual guarantee。 |
| *(仍全空，2026-07-24 registration-v3 contract closure 後重推)* | `quantity.h0_native_universe_completeness_registration` | Additive registration-v3 完成並選出 **`H0_REGISTRATION_V3_CONTRACT_SEALABLE`**；凍結 `h0_gctm_guarantee_registration_v3`（`universe_completeness` + `runtime_candidate_universe` / `runtime_event_membership` + `event_universe_sidecar` + 15-point completeness predicate；trace-v2 ABI change **not** required）。WIP 已釋放。此 object **不是** actual H0 guarantee、accepted baseline、runtime substrate、compatibility verdict 或 B1 candidate；**不**進入 `decision_relevant_candidates`；**不**授權 capture/re-entry。下一個 owner 決策為 design H0 re-entry + actual baseline capture。 |

**Machine projection:** `gctm_b1_slot_identity_decision_v1.json` 的
`registry_projection` 已重推為 `decision_relevant_candidates: []`、
`active_wip: []`、`o1_state: proposed`、`h0_reentry_authorized: false`；
`GCTM_D1` slot `state: terminal`。該 projection 由
`validate_research_slot_governance.py` fail closed 驗證。

**O0 的選擇（歷史）：** O0 於 2026-07-16 取 H0 為唯一 active，2026-07-20 依
route 1 關閉（[closed charter](../threads/closed/bridge_frozen_evidence_o0_routing_20260716.md)）。
候選集回到空集，D0/R1/S0 的 provenance 缺口依 route-1 下游以 `open_limits` 的形式
**永久留在帳上**（這是合法終局，不是待辦）；重進須新的 declaration amendment＋
新 owner reseal＋owner scheduling，**不因本表自動發生**。
若候選集日後有多個成員，**由 O0 依 decision relevance、依賴關係與 WIP=1 選出唯一 active**，
並在 module TODO 記錄唯一 charter pointer；預計狀態與 probe 放 linked charter，DEVELOPMENT 只提供穩定入口。

**入口：** [semantic TODO](../../modules/semantic/TODO.md)

---

## 9. 維護規則

- owner 接受一個 terminal 時，**同一個 PR** 更新對應 record（`state` / `open_limits` /
  `last_transition`），並**重新 review** `admissible_units`（它是 derived，會過期）。
- 本表與 module TODO 不一致 ＝ 治理錯誤：**以本表的 state + §4 推導為準**，先修 TODO。
- 本表不得增生數字、統計理由或結論摘要。要理由 → 點 `supporting_declaration`。
