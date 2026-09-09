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
- `build_runtime_identity.py` 已能標 `publication_complete: false`，但**沒有**「禁止用不完整檔覆蓋完整 canonical」的守衛，而且缺口在**兩個獨立位置**：
  - `check_runtime_identity_staleness.load_published()` 從不讀 `publication_complete`，不完整出版照樣通過。
  - 既有的 `--require-complete` 旗標**排在 `--emit` 寫檔之後**（`build_runtime_identity.py:469-479`）：檔案已經被覆蓋，才回非零。指向 canonical 路徑時，它會先摧毀一份完整出版再回報失敗。實測見 `tests/contract/test_adr_022_check_taxonomy.py::test_case4_require_complete_reports_after_it_has_already_written`。
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
- 用不完整 publication 覆蓋完整 canonical —— 須在**寫檔前**拒絕，並由 `load_published` 一併把關（§1.2 的兩個缺口）
- `research_lock` open（量測＝把出版當 substrate；本 ADR 不改，見 §3 決定 4）
- math-model **歷史完整性**（文件＋當前 audit 檔案 bytes、code-owned digest、`read_at_ref`）
- 畸形 publication / bindings / `equivalence` 被改成不是 `unproven`

---

## 5. Candidate capture

新增 sibling workflow `.github/workflows/runtime_identity_candidate.yml`：

- `plumbing_only`；**不改**現有 `runtime_identity.yml`，因此不動 `identity_semantics`
- **不**要求座標已經 current
- 只上傳 artifact；`contents: read`
- promote 必須另開 review PR ＋ `--require-complete`（**既有旗標，須先依 §4 改成寫檔前拒絕**）＋ 先 archive 到 §3 決定 5 的路徑
- 每次 promote 都要 fresh probe（§3 決定 2）

Layer P **不是** capture 路徑，這一系列不擴它，不改 `run_h2_layer_p.py`。

---

## 6. Math-model

繼續用全檔 SHA，**不做新 hasher**。把 §1.3 的三件事拆成兩個具名臂：

- `--historical-only`：仍鎖 `math_model.md` 與當前 audit 檔案 bytes、code-owned digest、`read_at_ref` 比對
- 只有**八個 source anchor 的工作樹比對**（§1.3 的第 3 項）變成可關的 current-HEAD 主張

沒有 `--fix`，沒有靜默刷新。依 §3 決定 3，這件事走**獨立 PR**，先於 runtime default 翻轉。

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

| # | 內容 | 動 `identity_semantics`？ | PR |
|---|---|---|---|
| 1 | 本 ADR —— 文件化四類檢查與 lag 預設 | 否 | [#359](https://github.com/raylei50653/saccade/pull/359) MERGED |
| 2 | Fixtures —— 四個代表案例的測試（live-tree 斷言暫時維持今天） | 否 | [#360](https://github.com/raylei50653/saccade/pull/360) MERGED |
| 3 | Candidate workflow —— 新檔，可收集完整 candidate，不 promote | 否 | [#382](https://github.com/raylei50653/saccade/pull/382) MERGED |
| 4 | Math-model 雙臂命名 ＋ **翻 default**（§3 決定 3：獨立、先行） | 否 | [#361](https://github.com/raylei50653/saccade/pull/361) MERGED |
| 5 | 改 runtime default ＋ 完整 republication | **是** —— 唯一動的一支；必須帶上由 PR 3 在該 head 跑出的完整出版 | 本 PR |
| 6 | 可選 —— 把 candidate 折回原 workflow 當 mode | 是 | 未開工（runner 目前不存在，見下） |

**卡點**：改 `pre_push.sh` / staleness checker / builder 會動 `identity_semantics` digest，今天的 live-tree pytest 會紅。所以必須**先**有 candidate 路徑，才能在 PR 5 合法 republish。PR 4 因為 math-model 不在該軸上，不受此限。

---

## 9. 執行後的兩項修訂（2026-09-09）

本 ADR 是計畫文檔；下列兩點是執行時實測推翻的前提，記錄於此以免後人照字面重建。

### 9.1 §5 的 candidate workflow 目前跑不動

repo 上註冊的 self-hosted runner 數為 **0**（`gh api repos/raylei50653/saccade/actions/runners` → `total_count: 0`）。`runtime_identity.yml` 最後一次成功是 2026-07-30，之後四次 dispatch 都排隊 24h 後自動取消。

而 §1.2 的 chicken-and-egg 是 **workflow 的性質、不是 builder 的**：`build_runtime_identity.py --run-probe --build-dir` 從來不要求座標 current。因此 PR 3 交付的**受支援路徑是本機 runbook**（`docs/reference/runbooks/runtime_identity_republication.md`），workflow 照 §5 寫好但等 runner 回來才可用。

「Controlled host」真正需要的是那十二份 runtime input、CUDA/TensorRT closure 與 build 資源，**不是某個 GitHub runner 身分**。

### 9.2 environment 軸混了兩種可檢性

`environment_axis()` 把 **recipe 半邊**（`CMakeLists.txt`/`pyproject.toml`/`uv.lock` 的 git blob，任何 host 都能重算）和 **observed toolchain**（Torch/CUDA/TensorRT/device，只有 controlled host 能比）hash 成同一個 digest，而整個 axis 只在 `verify_environment` / `--strict` 下比對。

後果：`89515241`（移除 Optuna，merge=`c5e22bab`）改了 `pyproject.toml` 與 `uv.lock` 之後，canonical 的 environment 軸就落後了，**而 `pre_push.sh` 與 `research_lock` 都沒發現**——observed toolchain 完全沒變，落後的純粹是可攜的 git object。

PR 5 因此把 recipe 半邊移進 portable 層（`static_axis_lag()`），兩條臂都會算；observed toolchain 維持只在 `--strict` 下比對。§2 的四類檢查沒有改變，這是把既有分類正確地套用到一個先前被錯置的量。
