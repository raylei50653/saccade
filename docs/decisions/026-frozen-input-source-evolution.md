<!-- doc-status: accepted -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-20 -->
<!-- doc-module: cross -->

# ADR 026: CLOSED packet frozen input 的 source evolution policy（historical immutability ≠ production freeze）

## Status

**Accepted** (2026-09-20) — [issue #436](https://github.com/raylei50653/saccade/issues/436)

**Terminal verdict: `historical_immutability_not_production_freeze`** — 一個 CLOSED
research packet 以 `path + sha256` 凍結的是**它自己的 referent**（那組 bytes 的身分），
不是 HEAD 上那個 path 的未來。packet 永遠不改、永遠可逐位元驗證；production source 可以
演進，代價是**在同一個 PR 內把「HEAD 已離開這個座標」記錄成機械可判定的事實**，而不是
悄悄讓舊 packet 的結論看起來仍描述 HEAD。

對照時點 2026-09-20，`main` = `ea9af2eb`。本文是 [ADR 022](022-check-taxonomy-and-publication-lag.md)
四類檢查在「packet frozen input」上的套用，不建立新的 check 家族。

本文**不**重開或重新詮釋 H0/GCTM 任何結果、**不**更新任何 frozen hash、**不**授權新的
capture / Phase A/B / research claim、**不**重新設計 tracker 行為。

---

## 1. Context —— #434 是第一次碰撞

### 1.1 什麼被凍住

兩個 CLOSED packet 以 strict `path + sha256` 綁住 tracker substrate：

| packet | terminal | 綁的 tracker 檔 | role |
|---|---|---|---|
| [`h0_gctm_interface_static_feasibility_20260723`](../modules/semantic/research/evidence/h0_gctm_interface_static_feasibility_20260723/) | `H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT` | `include/tracking/tracker_gpu.hpp` @ `898fa631…`、`src/tracking/tracker_gpu.cu` @ `e89934f6…` | `h0_capture_record_types` / `h0_capture_writer` |
| [`gctm_runtime_native_candidate_universe_20260724`](../modules/semantic/research/evidence/gctm_runtime_native_candidate_universe_20260724/) | `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE` | 同上兩檔、同一組 digest | 同上 |

兩包在 [claim-state registry](../research/contracts/claim_state_registry.md) 都是 `substrate: none`
（static ABI / schema / interface identities only；no H0 capture）。也就是說 tracker bytes 被綁進去
的身分是 **interface-identity witness**，不是執行過的量測底材——packet 的結論是「*這組*
ABI / registration-v2 結構上不足、*unchanged interface* 下禁止重複 capture」。

凍結是遞迴的：兩包的 `manifest.json[tooling]` 還把各自的 validator
（`validate_h0_gctm_static_feasibility.py`、`validate_gctm_runtime_universe.py`）與 targeted
tests（`test_h0_gctm_static_feasibility_v1.py`、`test_gctm_runtime_universe_v1.py`）也釘了
sha256，而那些 pinned tests 會拿 packet digest 去比**工作樹**。第三包
`h0_gctm_guarantee_registration_v3_20260724` 不直接綁 tracker，只綁前一包的
`frozen_input_identities.json`（一個 packet artifact）。

### 1.2 實測 blast radius（2026-09-20，對 `main` 加一個 byte 到 `tracker_gpu.hpp`）

13 個 contract test 紅：

- **7 個**來自上面兩包的 pinned targeted tests（frozen-input hash、canonical declaration、fixture catalog）；
- **6 個**來自 runtime-identity currency（`test_runtime_identity_staleness`、`test_research_lock`、
  `test_h2_measurement_controller`）——因為 `tracker_gpu.{hpp,cu}` 是 `decision_relevant` path。

第二組**不是本 ADR 的對象**：那是 ADR 022 已治理的 runtime coordinate publication，唯一受支援
的補救是 [republication runbook](../reference/runbooks/runtime_identity_republication.md)（#434 就是
走 PR #435）。本 ADR 只解第一組。**改 tracker 會同時碰到這兩道獨立的 gate，本 ADR 不合併它們。**

### 1.3 #434 怎麼過的，以及為什麼那不是 precedent

PR #434 是 2026-07-13 之後第一次需要在 `run_stage` 內加（default-off）association 計數器的改動。
當時沒有政策，owner 的選擇是**選項 1：把 tracker 端 instrumentation 從落地內容拿掉、hpp/cu 還原
`main` bytes**，`PerceptionPipeline` 那半搬到 `src/tracking/private_workload_stats.cu`（不在任何
packet 內），D2 的 per-stage 欄位降級為 branch-only evidence（commit `e03f7d81`，不可從 `main`
重現）。owner 同時明說**不可直接更新 frozen hash**——那會破壞 freeze 語義。

這是正確的一次性處置：在沒有政策時保住 CLOSED evidence。它**不是**「所有 tracker 改動都要繞開
frozen files」的先例；在本 ADR 之下，同一個改動的正規路徑是 §5.1 的 historicization entry。

---

## 2. Problem —— 兩個問題共用一個 exit code

ADR 022 §2 把檢查分成四類。packet 的 frozen-input 檢查其實橫跨兩類：

| 問題 | ADR 022 類別 | 失敗策略 | 今天誰在做 |
|---|---|---|---|
| packet 的 bytes（artifacts、tooling digest、frozen digest 值）還是當初那些嗎？ | 歷史完整性 | 永遠 fail-closed | pinned tests 的 manifest / artifact 比對 |
| HEAD 上的 `tracker_gpu.cu` 還等於 packet 凍結的 bytes 嗎？ | 證據適用 / 消費 | **只有當有人把 packet 當 HEAD 為真時**才 fail | pinned tests 對工作樹的 sha256 比對——**無條件** |

第二列被當成第一列執行，於是「research evidence freeze」在 2026-07-13 之後 de facto 變成
「production source freeze」：`tracker_gpu.{hpp,cu}`、`tracker_gpu_python.cpp`、`pipeline.{hpp,cpp}`
在 `main` 上兩個月零 commit。

pinned tests 自己不能改（它們被 manifest 釘住，改了就是另一種 drift）。所以政策與機制必須
**整個放在 pinned files 之外**。

---

## 3. Decision —— 名詞、狀態、authority

### 3.1 Packet identity 不可變

packet 目錄（`docs/modules/semantic/research/evidence/<packet_id>/`）內的每個檔案是
**packet artifact**：manifest、terminal report、declaration、`frozen_input_identities.json`……
永遠不改、永遠不得被 §3.3 的 ledger 標為 historical。它們的 digest 由 pinned tests 與本 ADR
的 checker **雙重**比對工作樹，任何 drift 都是 `unrecorded_drift`（兩臂都 fail）。

唯一例外維持不變：H0 owner-event declaration
（`headline_bridge_full_decision_capture_declaration_20260713.md`）只允許純尾端 `SEALED` row
append（`h0_declaration_frozen_identity.py`，Amendment 10）。它同樣**不可 historicize**。

### 3.2 Binding status 是算出來的，不是存起來的

一個 **binding** = 某個 packet 的一列 `frozen_input_identities.json[inputs]`（`inputs:<role>`）
或 `manifest.json[tooling]`（`tooling:<key>`），指向 `(path, sha256)`。每個 binding 在某個 HEAD
上恰有一個狀態：

| 狀態 | 定義 | 意義 |
|---|---|---|
| `current` | 工作樹 `path` 的 sha256 == 凍結值 | packet 對這個 path 的結論可被當成描述 HEAD（仍受 packet 自己的 non-authority flags 約束） |
| `historical` | 工作樹不等，**且** ledger 有一筆合法 entry，其 `last_current_ref` 上的 blob 重算 == 凍結值 | packet 仍 CLOSED、仍逐位元可驗證；結論描述**那個座標**，不描述 HEAD |
| `unrecorded_drift` | 工作樹不等，沒有合法 entry | 有人在沒宣告的情況下讓 HEAD 離開座標——**每一臂都 fail** |

status 由 [`scripts/tools/frozen_source_status.py`](../../scripts/tools/frozen_source_status.py)
從 git objects 重算；ledger 只記錄**轉移**。bytes 若回到凍結值，binding 自動回到 `current`
（entry 變 dormant，只 warn）。

path 依規則分五類：`source`（`include/`、`src/`…）、`tooling`（`scripts/`、`tests/`、`tools/`）、
`document`（`docs/` 但不在 evidence 目錄）三類可 historicize；`packet_artifact` 與
`owner_declaration` 不可（§3.1）。

### 3.3 Ledger = historical / current 的 mechanical authority

[`docs/research/contracts/frozen_source_supersession_ledger_v1.json`](../research/contracts/frozen_source_supersession_ledger_v1.json)
（schema：[`scripts/tools/frozen_source_supersession_ledger_v1.schema.json`](../../scripts/tools/frozen_source_supersession_ledger_v1.schema.json)）
是 append-only sidecar。定位與 [`runtime_identity_bindings_v1.json`](../research/contracts/runtime_identity_bindings_v1.json)
相同：**registry 仍是 object state 的唯一寫入者（C5.1），ledger 只持有 digest 與座標**。
registry record 裡的 `frozen_inputs` 列出的是 digest，本來就不會因 HEAD 演進而錯，所以不改。

每筆 entry 以 `(path, frozen_sha256)` 為 key，欄位：

| 欄位 | 規則（checker 逐條驗證） |
|---|---|
| `kind` | `historicization` 或 `supersession`（§3.4） |
| `bound_by[]` | 必須**恰好**等於所有釘住這組 `(path, sha256)` 的 packet binding；多列、少列都拒絕 |
| `last_current_ref` | 40-hex **commit**（不得是 branch tip，與 packet 的 `path_plus_sha256_only_no_mutable_branch_tip` 同精神）；必須是 HEAD 祖先；`git cat-file blob <ref>:<path>` 重算 == `frozen_sha256` |
| `claims_inherited` | 永遠 `false`——繼承是 successor packet 自己要建立的 claim，不是 ledger 能宣告的 |
| `successor_packet_id` / `owner_authorization` | historicization 必須為 `null`；supersession 必須齊備（§3.4） |
| `recorded_by_pr`、`recorded_on`、`rationale` | 出處 |

缺檔 = 被刪除的 guard，不是空 ledger。這是 structural provenance，不是密碼學簽章（與
`h2_controlled_host_execution_domain_v1` 的自述一致）。

### 3.4 兩種轉移、兩級 authorization

| kind | 情境 | 誰授權 | 附帶條件 |
|---|---|---|---|
| **historicization** | production source / tooling / document 正常演進；**沒有任何 research claim** 說「舊結論對新 bytes 仍成立」 | **engineering**：同一 PR、正常 review。owner 不需逐案簽 | packet 對該 path 變 `historical`；其 pinned targeted tests 移入 attested arm（§4） |
| **supersession** | 新研究需要演進後的 substrate，並主張與 CLOSED terminal 的關係（successor / 承接 retained conclusions） | **owner**：successor packet 的 `owner_acceptance_id`（沿用既有 packet terminal acceptance 機制） | successor packet 必須：(a) 以**新** digest 重新凍結該 path；(b) 在自己的 `frozen_input_identities.json` 帶 `supersedes[]`，逐一列出每個被取代的 packet 與同一個 `owner_acceptance_id`；(c) ledger entry 指向它 |

為什麼 historicization 不需要 owner：它**不主張任何事**。它只把「HEAD 不再是那個座標」寫成
可判定事實，而這件事本來就已經是真的。要求 owner 逐案簽等於把 owner 變成 production
merge gate，那正是 #436 要拆掉的隱性 freeze。

owner 若要對特定 path 收緊（例如 tracker 在某段期間需要逐案簽），做法是在 ledger 之外另立
owner-event，不是改本規則；本 ADR 不預設那種期間。

### 3.5 什麼**不會**因 historicization 發生

- **retained conclusions 不轉移。** `H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT` 仍是對
  `898fa631…/e89934f6…` 那組介面的裁決；對新 bytes 什麼都沒說。
- **禁令不解除。** 「repeat capture under unchanged interface is forbidden」在介面變了之後
  **不會**自動變成「可以 capture」——interface 變了只代表那條禁令的前提不再成立，而不是
  授權；任何 capture 仍需 successor packet + owner 決策（registry §7 的 H0 缺口原樣）。
- **registry state 不變。** `lifecycle_state: terminal`、`state`、`substrate: none` 全部原樣；
  registry 若要標註「packet 對 path X 已 historical」是 owner 的事，ledger 是它的 evidence。
- **pinned files 不動。** 兩包的 v1 validator / tests 保持 byte-identical，在 attested arm
  照常有效。

---

## 4. CI 行為 —— 三臂

| 臂 | 觸發 | 檢查 | `unrecorded_drift` | `historical` |
|---|---|---|---|---|
| **ordinary development PR** | 每個 PR：`pytest tests/`（`test_frozen_source_evolution_policy.py`，pre-push §5 與 CI 都跑）＋ CI contracts job 的 step `frozen_source_status.py --mode development` | packet artifacts 不變、ledger 合法、每個 binding 是 `current` 或 `historical` | **fail**，錯誤訊息指回 §5 | warn；該 packet 的 `tooling:targeted_tests` 檔由 `tests/contract/conftest.py` **整檔 skip 並附理由** |
| **attested / current consumer** | `SACCADE_ATTESTED_CONSUMER=1 pytest …` 或 `frozen_source_status.py --mode attested`；任何把 packet 當 HEAD 為真的東西都應跑這臂（successor packet 的 CI、H0 re-entry 準備、把 packet 列為 substrate 的 `research_lock open`） | 同上，外加「packet 描述 HEAD」 | fail | **fail**；pinned targeted tests 也照常執行並（正確地）紅 |
| **successor work** | 研究單元要用演進後的 substrate | `frozen_source_status.py --replay <packet_id>`：在 `last_current_ref` 的 detached worktree 重跑該 packet 的 pinned tests（證明歷史 packet 在自己的座標上仍通過自己的檢查）；successor packet 自己則在 attested arm 必須全 `current` | fail | 舊 packet `historical` 是**預期狀態**；successor 的 binding 必須 `current` |

skip 的範圍刻意是「整個 targeted test 檔」而不是逐條挑：那個檔就是 packet 的 currency
suite，且它被 manifest 釘住不能拆。skip 只在狀態確為 `historical` 時發生；`unrecorded_drift`
永遠不 skip（pinned tests 照紅，加上本 ADR 的 test 點名缺哪筆 entry）。

runtime-identity gate（§1.2 第二組）與本表**獨立並存**，各走各的 runbook。

---

## 5. Procedure —— 一個 PR 要改到 frozen path 時

### 5.1 Historicization（一般工程改動）

1. 改之前：`uv run python scripts/tools/frozen_source_status.py --json`，看該 path 被哪些
   binding 釘住（`bound_by` 就從這裡抄）。
2. 做改動。
3. 在**同一個 PR** 對 `frozen_source_supersession_ledger_v1.json` **append** 一筆：

   ```json
   {
     "entry_id": "tracker_gpu_cu_e89934f6_historical_20260921",
     "kind": "historicization",
     "path": "src/tracking/tracker_gpu.cu",
     "frozen_sha256": "e89934f6…（完整 64 hex）",
     "bound_by": [
       {"packet_id": "gctm_runtime_native_candidate_universe_20260724", "binding": "inputs:h0_capture_writer"},
       {"packet_id": "h0_gctm_interface_static_feasibility_20260723",  "binding": "inputs:h0_capture_writer"}
     ],
     "last_current_ref": "<git merge-base origin/main HEAD 的 40-hex>",
     "recorded_on": "2026-09-21",
     "recorded_by_pr": 4xx,
     "rationale": "一句話：改了什麼、為何不主張任何舊結論對新 bytes 成立",
     "claims_inherited": false,
     "successor_packet_id": null,
     "owner_authorization": null
   }
   ```

   `hpp` 與 `cu` 各一筆（key 是 `(path, sha256)`）。`last_current_ref` 用分支點的 `main`
   commit——那是最後一個工作樹仍等於凍結值的座標，checker 會 `git cat-file` 重算。
4. `uv run python scripts/tools/frozen_source_status.py` exit 0、只剩 `historical` warnings；
   `pytest tests/contract` 看到該 packet 的 targeted tests 以 ADR 026 理由 skip。
5. 另外走 runtime-identity republication runbook（§1.2）——那是獨立 gate。
6. PR 描述寫明「packet X 對 path Y 自本 PR 起 historical；不主張任何繼承」。

### 5.2 Supersession（successor packet）

只在**新的研究 claim 需要演進後的 substrate 並要與 CLOSED terminal 建立關係**時走：

1. 先完成 §5.1 的工程改動（可同 PR，也可先行）。
2. 建 successor packet 目錄 `docs/modules/semantic/research/evidence/<successor_id>/`，
   沿用既有 packet 形式（manifest / frozen_input_identities / terminal_report…）；
   `frozen_input_identities.json` 以 **HEAD 的** digest 重新凍結該 path，並加：

   ```json
   "supersedes": [
     {"packet_id": "h0_gctm_interface_static_feasibility_20260723", "owner_acceptance_id": "<successor 的 owner acceptance id>"},
     {"packet_id": "gctm_runtime_native_candidate_universe_20260724", "owner_acceptance_id": "<同一個 id>"}
   ]
   ```

   `prerequisite_terminals[].retained_conclusions`（既有欄位）照舊列出它**承接**的結論——
   承接的意思是「successor 要重新證明它們對新 bytes 成立」，不是自動繼承。
3. ledger entry `kind: "supersession"`、`successor_packet_id`、`owner_authorization:
   {owner_acceptance_id, date}`；`claims_inherited` 仍是 `false`。
4. successor 自己的 CI 在 attested arm 必須全 `current`；舊 packet 在 development arm 是
   `historical`，在 attested arm 紅是預期。
5. registry 新 record 由 owner 寫（C5.1）。

### 5.3 Pinned tooling（validator / schema / fixture / targeted tests）

同一套規則（它們就是 `tooling:*` binding），但**建議做 successor tooling（v2）而不是改 v1**：
改 v1 會讓那個 packet 在 attested arm 永久紅，而 v2 讓 v1 在 replay 時仍可自證。

---

## 6. 多個 CLOSED packet 綁同一個 path 的組合規則

- entry 的 key 是 `(path, frozen_sha256)`，`bound_by` 必須枚舉**所有**釘住這組值的 packet
  binding（checker 重算並拒絕不完整的列表）。所有 binder **一起**轉為 historical——同一組
  bytes 不可能對 packet A 是 current、對 packet B 是 historical。
- 同一個 path 的**不同 digest**（例如 successor 重新凍結後）是獨立 entry；在任一 HEAD 上
  至多一個 digest 是 `current`。
- 鏈式 packet（`h0_gctm_guarantee_registration_v3` 綁 `gctm_runtime_native_candidate_universe`
  的 `frozen_input_identities.json`）不受影響：它綁的是 packet artifact，永遠不變。

---

## 7. Tracker substrate 的 migration rule

- `include/tracking/tracker_gpu.hpp`、`src/tracking/tracker_gpu.cu`：一般工程改動 ⇒ §5.1，
  兩檔各一筆 entry（如只改一檔，只需一筆；另一檔仍 `current`）。
- `tracker_gpu_python.cpp`、`pipeline.{hpp,cpp}` 等鄰檔**沒有**被任何 packet 綁——它們的
  兩個月零 commit 是連坐，不是規則。它們只受 runtime-identity gate。
- **[#438](https://github.com/raylei50653/saccade/issues/438)**（portable OR-tail hook accept /
  dispose）是第一個應走本路徑的單元：dispose 與 accept 的 instrumentation 都是 §5.1；
  accept 若要產出 acceptance packet 並主張與 H0/GCTM 的關係，才進 §5.2。
- **Instrumentation / extension seams**（如 #434 的 `private_workload_stats.cu`）**允許但非義務**。
  本 ADR 不把「量測必須在 frozen code 之外」立為架構約束；seam 是設計選擇，不是政策要求。
- #434 的 branch-only evidence（`e03f7d81`）維持 branch-only；本 ADR 不回填。

---

## 8. Issue #436 的問題逐條回答

| 問題 | 回答 |
|---|---|
| CLOSED packet 是永久凍結 source bytes，還是只凍結 packet 的 historical referent？ | **只凍結 referent**（§3.1–3.2）。 |
| production code 可以演進而舊 packet 保持 immutable 且明確 historical？ | **可以**；historical 是算出來的狀態，由 ledger entry 使其可判定（§3.2–3.3）。 |
| 什麼把 downstream claims 標成不再 current？ | binding status `historical` + attested arm fail；registry state 不動，由 owner 決定是否註記（§3.5、§4）。 |
| 最小 successor / supersession 機制？ | successor packet 重新凍結 + `supersedes[]` + owner acceptance id + ledger `supersession` entry（§5.2）。 |
| strict frozen-path check 全域跑，還是只對 attested consumer？ | **歷史完整性**（packet artifacts、ledger 合法、digest 可從 git 重算）全域跑；**currency**（工作樹 == 凍結值）只在 attested arm 是 failure（§4）。 |
| 多個 CLOSED packet 綁同一 path 如何組合？ | 以 `(path, sha256)` 為單位一起轉移（§6）。 |
| instrumentation seam 能不能不變成永久架構約束？ | 能：允許但非義務（§7）。 |
| `tracker_gpu.{hpp,cu}` 與鄰檔的 migration rule？ | §7；#438 是第一個 case。 |

---

## 9. 限制與不做

- ledger 是 structural provenance，不是簽章；能改 packet 的人也能改 ledger。這與 repo 內其他
  authority sidecar 的限制相同，本 ADR 不新增簽章機制。
- `last_current_ref` 的祖先與 blob 檢查需要完整 git 歷史（CI pytest job 已 `fetch-depth: 0`；
  shallow clone 會 fail-closed 成「不可驗證」）。
- 不改 `h0_declaration_frozen_identity.py` 的 SEALED-append 規則。
- 不動任何 pinned validator / test / fixture / schema 的 bytes。
- 不定義 successor packet 的完整 schema——只定義它與舊 packet 之間**機械可驗的最小連結**
  （`supersedes[]` + 重新凍結）；其餘沿用該研究線自己的 packet 形式。
- 不決定 #438 走 accept 還是 dispose。

---

## 10. 落地清單

| 項目 | 位置 |
|---|---|
| 本 ADR | `docs/decisions/026-frozen-input-source-evolution.md` |
| ledger（初始空） | `docs/research/contracts/frozen_source_supersession_ledger_v1.json` |
| ledger schema | `scripts/tools/frozen_source_supersession_ledger_v1.schema.json` |
| checker / CLI（`--mode development\|attested`、`--json`、`--replay`） | `scripts/tools/frozen_source_status.py` |
| development-arm skip guard | `tests/contract/conftest.py` |
| contract tests（live-tree gate + worktree scenarios） | `tests/contract/test_frozen_source_evolution_policy.py` |
| CI step | `.github/workflows/ci.yml` — "Frozen-source evolution status (ADR 026, development arm)" |
| pre-push | 經 `scripts/pre_push.sh` §5 的 pytest；**不**加專用行——`pre_push.sh` 是 `identity_semantics` path，改它就得 republish runtime coordinate（ADR 022 §8） |

驗證（2026-09-20，`main` = `ea9af2eb`）：live tree 49 個 binding 全 `current`；scratch worktree
內對 `tracker_gpu.hpp` 加一 byte ＋ 一筆 historicization entry ⇒ development arm 37 個 pinned
tests 以 ADR 026 理由 skip、checker exit 0 帶 2 個 warning；`SACCADE_ATTESTED_CONSUMER=1` ⇒
同樣 7 個 pinned tests 紅、checker exit 1。
