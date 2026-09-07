<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-07 -->
<!-- doc-module: cross -->

# ADR 022: 檢查分類法與出版落後預設 (check taxonomy + publication lag)

## Status

**Proposed** (2026-09-07)

這是一份**計畫文檔**（同 [ADR 020](020-doc-lifecycle-new-nogo.md) / [ADR 021](021-asset-provenance-and-progress-reporting.md) 的性質）：定義問題、設計理由與規格需求，作為後續 PR 的依據。本文**不**宣告任何 claim 的 state，也**不**建立新的 verdict 家。

- 來源：[issue #334](https://github.com/raylei50653/saccade/issues/334)
- 對照時點 2026-09-07，`main` = `0fe937b4`
- 本 ADR **不**授權繞過任何現有 hook，**不**引入 digest 自動刷新（`--fix`）路徑

---

## 1. Context —— 解決什麼問題

日常開發被兩份「歷史出版／全檔 attestation」綁住：

1. H2 runtime publication — `docs/reference/runtime_identity.generated.json`
2. math-model 全檔 attestation — `docs/reference/math_model_source_attestation_v1.json`

改 `src/` 或被 attest 的 source anchor、且**不主張繼承任何舊 research evidence** 時，pre-push 與 CI 仍要求先做 publication 級維護（controlled-host 重捕捉 + math-model 重審計）。

觸發案例：`feat/postproc-chain-order` 的 `f60e814b`（postprocess 順序／stage diagnostics）。單 stage 相容性通過，實驗也沒證明 chaining 勝過 tracklet merge，但 pre-push 卡在 runtime-coordinate staleness 與 math-model attestation。該 commit 當時未 push；開 issue 時未獨立重現。

### 1.1 實際擋人的是什麼（對過程式，非政策想像）

**bindings 幾乎是空的。** `docs/research/contracts/runtime_identity_bindings_v1.json` 只有一列：

- `object`: `quantity.bridge_capture_provenance`
- `captured_under: null` → `classify_binding` 回 `"unattested"`

亦即**今天沒有任何 accepted evidence 被當成 current**。所以日常 `src/` 編輯卡關，**不是**「消費了舊證據」，而是「出版座標與 HEAD 的 byte digest 不一致」。

| 關卡 | 實際機制 | 卡什麼 |
|---|---|---|
| 本地 pre-push §4.11 | `scripts/pre_push.sh:131` 無條件跑 `check_runtime_identity_staleness.py` | `compare_publication` 只要 `decision_surface` / `implementation` / `identity_semantics` 與出版不符就 exit 1（`check_runtime_identity_staleness.py:151-156`），**即使所有 binding 都是 unattested** |
| CI merge gate | 不跑 staleness CLI，跑 pytest | `test_the_published_coordinate_is_complete_and_static_axes_are_current`（runtime 座標須 current）＋ `test_checked_in_attestation_is_current`（math-model 工作樹 bytes = attested SHA） |
| CI contracts | 再跑一次 math-model checker | 同一套全檔 SHA，雙閘 |

### 1.2 既有但解不了 chicken-and-egg 的東西

- `.github/workflows/runtime_identity.yml` 是 `workflow_dispatch`、唯讀，且**先要求「Coordinate must already be current」**才能產 probe。能驗證舊出版，**不能**收集更新座標所需的證據。
- Layer P `preflight()`（`scripts/tools/run_h2_layer_p.py:200-240`）也是驗證、不是 capture；成功還會發 `h2_layer_p_certificate_v2`。它與 workflow 的 "already current" **不是同一組檢查**（一個看 runtime-inputs，一個看 environment）。
- `build_runtime_identity.py` 已能標 `publication_complete: false`，但**沒有**「禁止用不完整檔覆蓋完整 canonical」的守衛。
- `research_lock.open` 已經**正確地不走 bindings**（`research_lock.py:314-327` 的 docstring 寫明：過期 closed study 不該擋新 instance）。日常 development 卻把這條 walk 又加了回來。
- **probe 相等不是 semantic equivalence**；`equivalence.state` 釘死 `unproven`，本 ADR 不動。

### 1.3 math-model checker 把三件事混成一個失敗清單

1. `math_model.md` 工作樹 bytes（歷史紀錄完整性）
2. `math_model_drift_2026-08-30.md` 工作樹 bytes（同上）
3. 八個 source anchor 的 **HEAD** bytes（「文件描述當前 HEAD」的主張）

改 `tracker_gpu.cu` 會因 (3) 失敗，**即使歷史 ref 與文件／audit 檔都沒動**。

---

## 2. Decision —— 四類檢查

工具可兼多類，但**每個 check 只能屬一類**。

| 類別 | 問什麼 | 失敗策略 |
|---|---|---|
| 日常開發 | 工程改動合不合法（測試、lint、非繼承主張的合約） | 工程缺陷 fail-closed；**歷史出版落後本身不擋** |
| 證據適用／消費 | 有沒有人把舊證據當成 HEAD 為真 | fail-closed |
| 出版完整性 | 要當 publication 時，source/config/inputs/env/probe 是否綁齊 | promote 時 fail-closed；candidate 收集時可暫不完整 |
| 歷史完整性 | 已接受的 bytes／紀錄是否還是當初那些 | 永遠 fail-closed |

### 2.1 核心定義

**「consumed as current」= `classify_binding(...) == "current"`**，不是「任何非 null `captured_under`」。

否則第一次真的 bind、再 promote 一份 P2 之後，`stale` 列會再次擋住所有 `src/` 改動 —— 等於 #334 復發。

---

## 3. Owner decisions (2026-09-07)

| # | 題目 | 決定 |
|---|---|---|
| 1 | Lag 預設 | **無 current binding 時 warn、不擋開發** |
| 2 | Republication 能否重用舊 MOT17-09 probe | **永不重用，每次 fresh capture** |
| 3 | math-model current-HEAD 何時關掉 | **math-model 先翻，獨立 PR**（它不在 `identity_semantics` 軸上，不受 republication chicken-and-egg 綁住） |
| 4 | `research_lock open` 是否仍要求出版 current | **維持是**（理由已寫在 `research_lock.publication_precondition` docstring） |
| 5 | Archive 放哪 | **`docs/reference/runtime_identity/archive/`**，不只靠 git history |

---

## 4. 仍必須 exit 1 的剩餘集合

- `current` binding ＋ source lag（假 current-attestation）
- 把列改寫成對一份落後出版為 `current`
- 用不完整 publication 覆蓋完整 canonical
- `research_lock` open（量測＝把出版當 substrate；本 ADR 不改，見 §3 決定 4）
- math-model **歷史完整性**（文件＋當前 audit 檔案 bytes、code-owned digest、`read_at_ref`）
- 畸形 publication / bindings / `equivalence` 被改成不是 `unproven`

---

## 5. Candidate capture

新增 sibling workflow `.github/workflows/runtime_identity_candidate.yml`：

- `plumbing_only`；**不改**現有 `runtime_identity.yml`，因此不動 `identity_semantics`
- **不**要求座標已經 current
- 只上傳 artifact；`contents: read`
- promote 必須另開 review PR ＋ `--promote-complete` ＋ 先 archive 到 §3 決定 5 的路徑
- 每次 promote 都要 fresh probe（§3 決定 2）

Layer P **不是** capture 路徑，這一系列不擴它，不改 `run_h2_layer_p.py`。

---

## 6. Math-model

繼續用全檔 SHA，**不做新 hasher**。把 §1.3 的三件事拆成兩個具名臂 —— `--mode {development,attested}`：

- `--mode development`（**新預設**）：歷史完整性。仍鎖 `math_model.md` 與當前 audit 檔案 bytes、code-owned digest、以及 `read_at_ref` 對 audited ref 的八份 source bytes 比對。這些只能靠改寫歷史才會壞，永遠 fail-closed。
- `--mode attested`：再加上 §1.3 第 3 項 —— **八個 source anchor 的工作樹比對**，即「文件描述當前 HEAD」的主張。行為與翻轉前一致，在 re-audit 或要把文件當成 current 出版時跑。

未知 mode 一律回 failure，不靜默退回較弱的一臂。`development` 的 PASS 訊息不得繼承 `attested` 的措辭，須明寫它對「文件是否仍描述 HEAD」無主張。

> 命名說明：草稿原本寫 `--historical-only`。實作時改為具名 mode 對 —— 一旦預設翻成 development，否定語氣的旗標讀起來是反的（預設就已經 historical-only，旗標會變成永遠開著）。

沒有 `--fix`，沒有靜默刷新。依 §3 決定 3，這件事走**獨立 PR**（從 `main` 開，不疊在本 ADR 上），先於 runtime default 翻轉。

---

## 7. 實作限制（review 收斂結果）

### 7.1 不得更動 `compare_publication` 的契約

`staleness.compare_publication` 有兩個外部消費者，都解 2-tuple：

- `scripts/tools/research_lock.py:326` — `failures, _ = compare_publication(published, probe=None)`；其 docstring 明寫**刻意不走 binding walk**
- `scripts/tools/run_h2_layer_p.py:216` — `failures, warnings = ...(verify_environment=True)`，而 `warnings` 被寫進 certificate 的 `static_axis_warnings` 欄位

動 return shape 或 warnings 語義會同時弄壞 research_lock 與 Layer P certificate 內容。

**但**：今天的 binding walk 本來就只在 `main()`（`check_runtime_identity_staleness.py:228-255`），而日常開發真正被擋的三條 static-axis failure 在 `compare_publication` **函式內部**（151-156）。因此「lag 在 `main` / `development_verdict` 重分類」要能實作，前提是：

> 先把 axis 重算抽成共用 helper（例如 `static_axis_lag(published) -> dict[axis, tuple[published, recomputed]]`），讓 `compare_publication` 的簽名、回傳形狀與 failure 字串**逐字不變**；`--mode development` 在 `main()` 裡改用該 helper 自行組合，**不經過** `compare_publication`。

否則 `main()` 只拿得到一串 failure 字串，只能靠 string-match 重分類。

### 7.2 `--strict` 的語義不得被偷換

`--strict` 今天的定義是 `verify_environment=True` **且**「unresolved ⇒ exit 1」（`check_runtime_identity_staleness.py:262`）。development 模式下 `runtime_inputs` 與 `probe` 兩條 warning 在沒給 `--probe-from` / `--runtime-inputs-from` 時**必然存在**（173-178、190-193 行）。

因此 fixtures **不得**把 `--mode development --strict` 釘成 exit 0 —— 那等於重新定義 `--strict`。注意 environment 在 `--strict` 下是被**比對**而非 unresolved，所以「environment 可能 unresolved」不是這條的理由。

### 7.3 section map 測試不得只認 `##`

`docs/reference/math_model.md` 的 `##` 只有整數節（`## 0.`、`## 1.`…），小節全部是 `###`（`### 0.1`、`### 3.1`、`### 4.2`、`### 5.3`…），而 §4.2 正是文件自己 preamble（第 19 行）引用的 anchor。section map 測試須匹配 `^#{2,4}\s`。

---

## 8. PR 序

前四項不改 default、不繞 hook。

| # | 內容 | 動 `identity_semantics`？ |
|---|---|---|
| 1 | 本 ADR —— 文件化四類檢查與 lag 預設 | 否 |
| 2 | Fixtures —— 四個代表案例的測試（live-tree 斷言暫時維持今天） | 否 |
| 3 | Candidate workflow —— 新檔，可收集完整 candidate，不 promote | 否 |
| 4 | Math-model 雙臂命名 ＋ **翻 default**（§3 決定 3：獨立、先行） | 否 |
| 5 | 改 runtime default ＋ 完整 republication | **是** —— 唯一動的一支；必須帶上由 PR 3 在該 head 跑出的完整出版 |
| 6 | 可選 —— 把 candidate 折回原 workflow 當 mode | 是 |

**卡點**：改 `pre_push.sh` / staleness checker / builder 會動 `identity_semantics` digest，今天的 live-tree pytest 會紅。所以必須**先**有 candidate 路徑，才能在 PR 5 合法 republish。PR 4 因為 math-model 不在該軸上，不受此限。
