<!-- doc-status: accepted -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-29 -->
<!-- doc-module: cross -->

# ADR 027: 歷史檢查按用途分流（HEAD 不再欠舊研究 ledger）

## Status

**Accepted** (2026-09-29) — [issue #493](https://github.com/raylei50653/saccade/issues/493) PR-1

**Partially supersedes [ADR 026](026-frozen-input-source-evolution.md)**：§3.2 的 `historical`
定義、§3.4 的 historicization 列、§4 development arm 對 `unrecorded_drift` 的定義、§5.1 整節。
ADR 026 的其他部分（packet identity 不可變、ledger schema 與 append-only、supersession
的 owner 授權、attested / scoped / replay 三臂、§3.5 什麼不會因 historicization 發生）**原樣
有效**。依 [decisions README](README.md) 的規則，ADR 026 原文不改，只加 superseded 標註。

本文**不**重開或重新詮釋任何 CLOSED 研究的 verdict、**不**更新任何 frozen hash、**不**動
runtime identity 的觸發點、**不**動任何 pinned validator / test / fixture / schema 的 bytes。

---

## 1. Context

ADR 026 已經說對了一件事：CLOSED packet 凍結的是**它自己的 referent**，不是 HEAD 的未來。
但它把「HEAD 已離開這個座標」的記錄成本放在每一個一般工程 PR 上：每次改到被綁的
source / tooling / document，都得在同一 PR 手寫一筆 historicization entry，否則
development arm 與 pytest 都 fail。那筆 entry **不主張任何事**（ADR 026 §3.4 自己這麼說），
它記錄的 `last_current_ref` 也完全可以從 git 算出來。

實測（2026-09-29，`main` = `8415b4ac`，scratch worktree，只加註解型 byte）：

| mutation | CI step 失敗 | pytest 失敗中屬「HEAD 必須等於舊 source」的 |
|---|---|---|
| `tracker_gpu.hpp` 加一行 | `frozen_source_status.py --mode development` | `test_frozen_source_evolution_policy`（11，live-tree + 以 HEAD 為基線的 scenarios）、兩包 pinned targeted tests（7） |
| `src/`、`include/`、`docs/`（evidence 目錄與 H0 declaration 除外）全部各加一行 | 同上 | 上列 + `test_gctm_d1_ranking_diagnostic_v1::test_sealed_packet_is_bit_identical_to_fresh_emit` |
| `scripts/**/*.py` 全部各加一行 | 同上 | 上列 pinned tests + `tests/unit/eval/test_capture_race_incidence_20260909.py`（整檔，preflight 拒絕漂移的 control source） |

同一組 mutation 下 **不**失敗的 CI step：`check_h0_repair_acceptance_matrix.py`、
`check_h0_phase_a_archives.py`、`check_h2_measure_archives.py`、`check_headline_decision_contract.py`、
`check_runtime_identity_staleness.py`（CLI）。它們驗的是封存 evidence 本身，不要求 HEAD 等於舊
source，所以本 ADR 不動它們（owner 決定：留在預設 gate）。

另外失敗、但**屬於其他飄移類型**、本 ADR 不動的：

- runtime identity currency（`test_runtime_identity_staleness`、`test_research_lock`、
  `test_h2_measurement_controller` 各 2）——結論飄移防護，走
  [republication runbook](../reference/runbooks/runtime_identity_republication.md)，觸發點不變；
- #465 active line 的 runner 宣告 blob / construction source pin——active 研究的凍結座標；
- `test_score_ranking_declaration_v1::test_active_binding_identity_matches_frozen_contract`——
  accepted contract 的 bytes，改它就是改契約；
- `check_math_model_source_attestation.py`——[ADR 022](022-check-taxonomy-and-publication-lag.md) 已治理；
- master map、scripts/tests index、training freshness 等衍生檔——文件飄移，歸 #493 PR-2。

## 2. Decision —— 按檢查用途分三類

| 檢查用途 | 處理 | 本 ADR 落地 |
|---|---|---|
| 要求現行 HEAD 永遠等於舊研究所用的 source | **退出一般開發 gate** | 見 §3 |
| 驗證封存 packet／舊 validator 在原始座標上仍然正確 | 按需執行：`SACCADE_ATTESTED_CONSUMER=1` 或 `--replay` | pinned targeted tests（既有 ADR 026 skip guard）、GCTM D1 fresh-emit、#340 2026-09-09 campaign harness tests |
| 防止封存證據被改寫，或防止舊證據被當成 HEAD 的證據 | **保留在預設 gate** | packet artifact / owner declaration drift、ledger 合法性與 append-only、attested arm、runtime identity、archive checkers |

判準一句話：**一般開發不必反覆證明舊研究仍描述 HEAD；但證據不能被改寫、適用座標要
明確、正式結論不能超出有效量測。**

## 3. `historical` 改為從 git 推導

binding 狀態（取代 ADR 026 §3.2 表中 `historical` / `unrecorded_drift` 兩列）：

| 狀態 | 定義 |
|---|---|
| `current` | 工作樹 bytes == 凍結值（不變） |
| `historical` | 工作樹不等，且凍結 bytes 可在某個 `last_current_ref` 重算：有合法 ledger entry 時用 entry 的值；**沒有 entry 時從 git 推導**——HEAD 仍帶凍結 bytes（只有工作樹漂移）則為 HEAD；否則為「把 bytes 改走的那個 commit」的 parent |
| `unrecorded_drift` | packet artifact 或 H0 owner declaration 的任何漂移；或凍結 bytes 不在任何可達 commit 上（shallow clone、改寫歷史）——**不可驗證即 fail-closed** |

後果：

- **一般 source / tooling / document 演進不再需要 ledger entry**。development arm 以
  `historical (derived from git at <ref>)` warning 報告；該 packet 的 pinned targeted tests
  由 `tests/contract/conftest.py` 以既有理由 skip。
- **ledger 只在 supersession 時必填**（ADR 026 §3.4 第二列、§5.2 不變）。historicization entry
  仍被接受並逐條驗證（寫錯仍 fail），但不再是義務；既有 entries 依 append-only 保留。
- attested arm 不變：任何 `historical`（不論 recorded 或 derived）在 global attested 都 fail；
  scoped attested 只看被 attest 的 packet。
- `--replay <packet>` 不再需要 ledger：預設座標 = packet 仍 current 時的 HEAD，否則是其
  historical bindings 的 `last_current_ref` 中**同時帶齊該 packet 所有凍結 bytes 的最新一個**；
  找不到單一座標就要求 `--at`。

## 4. 其他 currency 斷言的分流

兩個 probe 找到的 currency 斷言不在 frozen-source binding 內，照同一規則處理：
**凍結輸入仍 current 時照跑；漂移後在 development arm skip 並寫明理由，
`SACCADE_ATTESTED_CONSUMER=1` 時照跑（並正確地紅）。**

| 測試 | 漂移判定 | 按需重現 |
|---|---|---|
| `tests/contract/test_gctm_d1_ranking_diagnostic_v1.py::test_sealed_packet_is_bit_identical_to_fresh_emit` | packet `identities.json` 記錄的 GCTM theory / lemmas / score contract sha256 ≠ HEAD | 在最後一個三檔仍等於記錄值的 commit 上跑該測試 |
| `tests/unit/eval/test_capture_race_incidence_20260909.py`（整檔） | `capture_attribution/analyze.py`、`observer.cpp` ≠ harness 的 §2.2 凍結值 | `git worktree add --detach <wt> 276d8d74`（`OBSERVER_CONTROL_COMMIT`）後在該 worktree 跑 |

同檔其餘測試（封存 artifact bytes、fixture digest、terminal 內容）不受影響，仍在預設 gate。

## 5. 阻擋入口

| 入口 | 變化 |
|---|---|
| `scripts/pre_push.sh` | **不改**（它是 `identity_semantics` path）；經 §5 的 pytest 自動生效 |
| pytest（`tests/contract/`、`tests/unit/eval/`） | 上述 skip；`test_frozen_source_evolution_policy.py` 的 scenarios 改以固定 anchor `8415b4ac` 為基線，不再假設 HEAD 全 current |
| `.github/workflows/ci.yml` `Frozen-Source Evolution Status` | 指令不變，語義改為本 ADR；`H0 Repair Acceptance Matrix`、`H0 Immutable Archive Corpus`、`H2 Measurement Archive Corpus` 不動 |
| `.github/workflows/h0_qualification.yml` | 不動（repair matrix 不擋一般開發） |

## 6. 驗收（反向測試）

| 驗收 | 由誰證明 |
|---|---|
| 改 `tracker_gpu.{hpp,cu}` 或其他 frozen input、不補 ledger，本地 hook / pytest / CI 通過 | `test_source_drift_without_entry_is_derived_historical`；PR 描述附 probe worktree 全 pytest + CI step 結果 |
| 改封存 packet 或 H0 declaration，會 fail | `test_packet_artifact_drift_fails_without_any_entry`、`test_owner_declaration_drift_fails_without_any_entry`、`test_packet_artifacts_cannot_be_historicized` |
| 凍結 bytes 不可驗證時 fail | `test_frozen_bytes_absent_from_history_stay_unrecorded` |
| 把 closed packet 當 current evidence 仍被拒 | `test_source_drift_without_entry_is_derived_historical`（attested 分支）、`test_scoped_attestation_of_a_historical_packet_fails`；runtime identity consumer gate 不變 |
| 移出預設 gate 的驗證可用 commit + 指令重現 | `frozen_source_status.py --replay <packet>`（`test_replay_runs_a_packet_suite_at_a_frozen_coordinate`）；§4 表 |
| 不需要每次 push 重跑整套歷史研究 | 漂移後 currency suites skip；replay 只在按需時跑 |

## 7. 新增閘門的準則

新增任何 gate 時，必須說明它防的是哪一種飄移（結論 / 行為 / 文件 / 歷史），以及擋在哪些
入口（hook / pytest / CI）。「HEAD 必須等於某個 CLOSED 研究的輸入」不是可接受的預設 gate；
那屬於 attested arm。

## 8. 限制

- 推導只看 HEAD 可達歷史；shallow clone 會把 derived historical 變成 `unrecorded_drift`
  （fail-closed）。CI 的 contracts 與 pytest job 都是 `fetch-depth: 0`。
- 推導出的 `last_current_ref` 是「最後仍帶凍結 bytes 的 commit」，不是「packet 被 seal 的
  commit」；兩者只在 replay 需要時有差，而 replay 以「帶齊全部凍結 bytes」為條件選座標。
- ledger 仍是 structural provenance，不是簽章（ADR 026 §9 不變）。
