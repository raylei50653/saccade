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
pending_reentry:                          # append-only; pre-seal, no terminal claimed; route-1 永久留帳結論不變
  - date: 2026-07-21
    scheduling: owner-scheduled re-entry #3（滿足 line-337 future_reentry_precondition:launch-hygiene gate 先行）
    declaration_amendment: "Amendment 9（headline_bridge_full_decision_capture_declaration_20260713.md;pre-seal, append-only）"
    scope: "re-admit 既有 sealed unit h0_build_tool_provenance_closure 供單一 fresh I→F→S attempt（前次於 #224/#227 I=31c9eee8 被 PROVENANCE_INVALID 消費,capture 前失敗）;非第二 unit;acceptance matrix/checker/workflow tuple/qualification 語義/歷史 declaration/歷史 sealed evidence 全 byte 不變"
    launch_hygiene_gate: "scripts/tools/h0_launch_hygiene_gate.py（非授權;複用單源 predicate run_h0_phase_a.assert_no_preexisting_build_tree）;mandatory:授權前與 sealed checkout controller launch 前皆須報 clear"
    status: "RESOLVED 2026-07-21 — owner sealed（S=3a6a9ec6）＋ exactly-once authorized ＋ scheduled ＋ executed ⇒ 到達 owner-accepted ordered terminal H0_PROVENANCE_INVALID（見 reentry_terminal_history re-entry #3;PR #235 comment 5032610430）;exactly-once authorization consumed;exact S permanently spent;retry/resume/second invocation forbidden;無 repair / 新 re-entry 授權"
last_reviewed_at: 2026-07-21
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
ladder: proposed charter terminal family
transition_semantics: defined（只允許 local diagnostic terminal；canonical transition 需 owner scheduling 後 execution）
lifecycle_state: proposed
state: none
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
  does_not_satisfy_owner_scheduling: true
  note: slot.owner_acceptance_id remains null until activation; it is not this declaration acceptance id
seal_candidate:
  status: SEAL_CANDIDATE_GENERATED
  generation_kind: pre_activation_synthetic_seal_candidate
  provisional_terminal: GCTM_D1_INTERFACE_READY
  authority: sealed packet identities frozen by declaration acceptance; not charter execution; not canonical state transition
  packet: ../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/
  declaration: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md
  terminal_report: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md
substrate: synthetic fixture pack gctm_d1_synthetic_fixture_pack_v1（non-runtime）
target_substrate: none
authority_class: diagnostic_only
blocked_by: owner_scheduling
blockers:
  - type: inadmissibility
    what: runtime claim、runtime B1 transition、O1 unlock 或 decision-relevant candidate
    clause: diagnostic evidence 不可滿足 runtime substrate/provenance/identity/checksum/compatibility/activation authority
  - type: dependency
    what: charter activation / WIP / canonical terminal acceptance
    clause: declaration_owner_acceptance 已滿足（owner_accepted_governance）；owner_scheduling 仍缺且必須是 owner_scheduling_decision；canonical state 仍 none
decision_relevance:
  status: zero — declaration acceptance does not equal execution；does not create decision-relevant candidate；不取得 WIP
supporting_declaration: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_declaration_20260723.md
supporting_terminal: ../../modules/semantic/research/gctm_d1_ranking_diagnostic_terminal_20260723.md
supporting_packet: ../../modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723/
charter_ref: ../threads/gctm_d1_substrate_agnostic_ranking_diagnostic_task.md
accepting_review: gctm_d1_declaration_owner_acceptance_20260723
last_transition: 2026-07-23 — GCTM_D1_DECLARATION_ACCEPTED; requirement declaration_owner_acceptance bound; owner_scheduling still requires owner_scheduling_decision; canonical state remains none; no B1/O1/H0 change
admissible_units: []
derived_from: gctm_b1_slot_identity_decision_v1.json terminal policy + owner-accepted declaration + seal-candidate packet + §5 relevance
last_reviewed_at: 2026-07-23
```
---

## 7. 架構缺口（顯式化，而不是假裝可編排）

| 缺口 | 影響 |
|---|---|
| **score-layer transition semantics** | **RESOLVED 2026-07-23.** [`score_ranking_evidence_contract_v1`](score_ranking_evidence_contract.md) 已由 owner 接受並凍結；`owner_acceptance_id: score_ranking_contract_owner_acceptance_20260723`；`registry_binding_id: claim_state_registry_score_ranking_v1`；`contract_sha256: 7dbc2d965079fa3fc13f7802a4a083b1c4cbf49d658ffe3728b6c405364a13b4`。本 binding 只使 L2 admissibility 可判定；不自動推進 object、產生候選、啟動 B1/O1 或授權 runtime 行為。 |
| **B1-slot identity** | **RESOLVED 2026-07-23.** [`gctm_b1_slot_identity_decision_v1`](gctm_b1_slot_identity_decision_v1.json) 固定 `GCTM_B1 != H0_ROUTE5_B1`、`relation: coexist`、非 alias、非 supersede、不可共享 activation authority；未來改寫關係須另開 owner-accepted transition。Identity resolution 不解除任何 runtime gate。 |
| **H0 runtime substrate / compatibility** | **OPEN / fail closed.** 三次 `H0_PROVENANCE_INVALID` 後仍無 valid runtime substrate、stable evidence identity、canonical checksum 或 owner-accepted H0→GCTM compatibility verdict。`H0_ROUTE5_B1` 與 `GCTM_B1` 均保持 proposed；`GCTM_D1` diagnostic evidence 不可補此缺口。 |

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

**Machine projection:** `gctm_b1_slot_identity_decision_v1.json` 的
`registry_projection` 已重推為 `decision_relevant_candidates: []`、
`active_wip: []`、`o1_state: proposed`、`h0_reentry_authorized: false`。
該 projection 由 `validate_research_slot_governance.py` fail closed 驗證。

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
