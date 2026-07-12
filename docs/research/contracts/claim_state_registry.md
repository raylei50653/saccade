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

**S0 宣告（draft，未 seal）：** [safe_domain_runtime_transfer_declaration](../../modules/semantic/research/safe_domain_runtime_transfer_declaration_20260712.md) · [thread](../threads/runtime_faithful_safe_domain_20260712.md)

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
state: R1_FAITHFUL
substrate: runtime CUDA（封印於 headline adaptive-anchor 設定 + 七序列 support）
target_substrate: same
open_limits: [不推廣至其他 anchor mode / preset / detector / substrate]
blockers: []
decision_relevance:
  variable: 無直接決策變數
  role: **enabler** — 它是「runtime 座標可稽核」的前提,safe domain 與任何 score 研究得以
        建在 runtime 座標上,靠的就是它
supporting_declaration: ../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md
last_transition: 2026-07-12 — owner accepted R1_FAITHFUL
admissible_units: []                   # 已達成
last_reviewed_at: 2026-07-12
```

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

### `score.ambiguous_band_ranking_class`

```yaml
layer: L2 score
ladder: study-sealed terminal            transition_semantics: defined（該研究自帶封印判準）
state: T2_NO_USABLE_RANKING_POWER_IN_CLASS (accepted)
substrate: s0 proxy 空間                 target_substrate: runtime
open_limits: [class-scoped:只封 12 members;AND pair / 連續訊號 / 有限 λ / learned score 未耗盡]
blockers: []
decision_relevance:
  role: **反向約束** — proxy 空間的 ranking headroom 已封;score 線若開,必須在 runtime 座標上
supporting_declaration: ../../modules/semantic/research/door0_ranking_probe_results_20260712.md
last_transition: 2026-07-12 (#136 accepted)
admissible_units: []                     # class 已封;重開需新宣告
last_reviewed_at: 2026-07-12
```

### `score.anchor_propagation`（前 discrete-\(M\)）

```yaml
layer: L2 score                          # 曾被誤標為 gate
ladder: —                                transition_semantics: unavailable   # ← §4.4
state: none                              # 未 seal,無被接受的 state
substrate: —
open_limits: [宣告的 horizon {1,2,4,8} 與 consumer 脫節（la median 12、p90 26）]
blockers:
  - type: inadmissibility
    what: 以 gate 身分執行
    clause: §4.1 evidence 型別錯層（margin/rank ≠ gate 證據）
  - type: dependency
    what: 作為 score feature 執行
    clause: §4.4 — score layer 無契約,transition semantics unavailable
    depends_on: [score-layer contract（不存在）, 保留域先被定義]
decision_relevance:
  status: **zero（目前）** — 保留域尚未定義,排序無意義;正反結果都不改變任何 production 決策
  # ⇒ §5 反事實測試不通過 ⇒ 不得取得 WIP 鎖
supporting_declaration: ../../modules/semantic/research/discrete_m_capability_declaration_20260712.md（parked, unsealed）
last_transition: 2026-07-12 — reclassified gate → score feature;parked
admissible_units: []                     # [cached derivation]
derived_from: §4.1 錯層 + §4.4 契約缺席 + §5 relevance zero
last_reviewed_at: 2026-07-12
```

---

## 7. 架構缺口（顯式化，而不是假裝可編排）

| 缺口 | 影響 |
|---|---|
| **score-layer 契約不存在** | 所有 `layer: L2 score` 的 object 只能是 `transition_semantics: unavailable`；**score 半邊無法自動判定 admissibility**。這是合法狀態，不是 schema 缺陷。score 線開啟**前**必須補上（rank/margin/top-1 的證據語義與 claim ladder）；**現在不憑空寫**。 |

---

## 8. 候選集 → O0 選擇（registry 到此為止）

**registry 產生的合法候選集（2026-07-12）：**

| 候選 | Object | 為何合法 |
|---|---|---|
| *(空)* | 所有 object | 已達 terminal／被 inadmissibility 排除／依賴未解／relevance zero |

**O0 的選擇（不是 registry 的）：** 候選集目前為空。目前除維護、治理與工程收尾外，無科學主線 WIP 鎖被授權。
若候選集有多個成員，**由 O0 依 decision relevance、依賴關係與 WIP=1 選出唯一 active**，
並在 module TODO 記錄唯一 charter pointer；預計狀態與 probe 放 linked charter，DEVELOPMENT 只提供穩定入口。

**入口：** [semantic TODO](../../modules/semantic/TODO.md)

---

## 9. 維護規則

- owner 接受一個 terminal 時，**同一個 PR** 更新對應 record（`state` / `open_limits` /
  `last_transition`），並**重新 review** `admissible_units`（它是 derived，會過期）。
- 本表與 module TODO 不一致 ＝ 治理錯誤：**以本表的 state + §4 推導為準**，先修 TODO。
- 本表不得增生數字、統計理由或結論摘要。要理由 → 點 `supporting_declaration`。
