<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-01 -->
<!-- doc-module: cross -->

# ADR 021: 資產身分層與生成式進度報告 (asset provenance + generated status)

## Status

**Proposed** (2026-09-01)

這是一份**計畫文檔**（同 [ADR 020](020-doc-lifecycle-new-nogo.md) 的性質）：定義問題、設計理由與完整規格需求，作為後續 PR 的依據。本文**不**宣告任何 claim 的 state，也**不**建立新的 verdict 家。

- 診斷來源：2026-09-01 對 `docs/` 與 workspace 資產目錄的一次盤點（量測方法見 §1.1）
- 治理前提：[doc_structure_contract.md](../ownership/doc_structure_contract.md)（C4 index / C5.1 單一寫入者 / C6 lifecycle / C8 WIP=1）、[ADR 020](020-doc-lifecycle-new-nogo.md)（typed terminal + seal-triggered disposal）
- 本 ADR **不取代**上述兩者：它補的是 ADR 020 沒有覆蓋的**另一半資產**（實驗產物目錄），以及**狀態的呈現面**（如何一行指令看到主線在哪）

---

## 1. Context —— 解決什麼問題

### 1.1 量測：兩層資產的成熟度差了一個數量級

量測時點 2026-09-01，`main` = `77464885`。

**claim / doc 層 —— 治理成熟，但 ADR 020 只落地了前半：**

| 項目 | 實測 |
|---|---|
| `claim_state_registry.md` | 存在且為 state 的 fact-owner，schema 完整 |
| ADR 020 | 仍為 `proposed`；umbrella issue #164 自 2026-07-16 開啟後未關 |
| terminal slot | schema (`scripts/docs/terminal_slot_schema.py`) + fixtures + `tests/contract/test_terminal_slot_schema.py` 已在；以 `line_type:` 字面搜尋，`docs/` 下帶 slot 的**非 schema/非 ADR** 文件僅 **6 份**（bridge-fidelity 那批） |
| ADR 020 S3（seal fail-closed 觸發）/ S4（dispose flow）/ S6（drift reconciliation） | 未落地 |
| migration manifest | 9 個 cluster **全部** `migration_state: quarantined`，0 個到達 `disposable`；`snapshot.resolved_at: 2026-07-16` |
| doc-status | `docs/**/*.md` 共 290 檔，**187 檔無 `doc-status`（64%）**；enum 漂移仍在（`sealed-for-execution` 與 `sealed-execution` 各 1） |
| 進度快照 | `docs/research/tracker-decision/status_2026-07-09.md` 為**手寫**、日期 2026-07-09 |

**asset 層 —— 無任何治理：**

| 目錄 | `du -sh` | 頂層條目數 | provenance |
|---|---:|---:|---|
| `runs/` | 82 G | 229 | 無 manifest |
| `results/` | 3.1 G | 681 | 無 manifest |
| `out/` | 1.3 G | 174 | 無 manifest |
| `output/` | 282 M | 106 | 無 manifest |
| `scratch/` | 794 M | 42 | 無 manifest |
| `logs/` | 16 M | 88 | 無 manifest |

抽樣的 `results/` 子目錄內容為 `MOT17-*.txt` / `_fps_summary.txt` / `_latency_profile.json` / `_global_id_map.txt`：**沒有 commit SHA、沒有 preset、沒有 config、沒有 host**。

**可達性量測（方法：以目錄名做字面字串 grep 掃 `docs/`）：**

- `results/` 前 200 個目錄名 → **15 個**能被搜到
- `runs/` 前 100 個目錄名 → **42 個**能被搜到

> **這個量測只支持一件事**：這些目錄**無法由任何 doc 依名字到達**。它**不**證明未被搜到的目錄無價值，也**不**是「該刪」的判據（`runs/` 命中率較高多半只因 run 名常等同 preset 名）。

### 1.2 兩個缺口

**缺口 A —— 資產無身分。** 一個 `results/` 目錄無法回答「誰跑的、什麼 commit、什麼 preset、支撐哪個 claim」。因此它**既不能被安全引用，也不能被安全刪除**。這正是 82 G 只增不減的機制成因，且與 ADR 020 §1.2 的診斷是**同一個病的另一半**：缺 disposal，而混亂本身抹掉了清理所需的資訊。

`evidence_ledger.md` 引用的是**文件**，文件引用的是**數字**；「這個數字由哪個目錄、哪個 commit 產生」這一節是斷的。

**缺口 B —— 進度只能靠人腦。** 沒有任何一條指令能回答「現在主線在哪、下一步的合法候選是什麼」。registry / `MEMORY.md` / `TODO.md` / `threads/README.md` 四個導覽面各自為政，每次開工都要重讀。唯一的一頁式快照是手寫且已過期的 `status_2026-07-09.md`。

### 1.3 為什麼「刪一刪」與「再寫一份總結」都不是解

- **刪一刪無判據**：沒有 provenance 就沒有 orphan 的定義，刪除等於賭。先給身分，才談處置。
- **再寫一份總結 = 第二真相**：state 已有 fact-owner（registry）、terminal 已有 slot、quarantine 已有 manifest。任何**手寫**的進度頁都會漂移，正是 ADR 020 §1.3 明確否決的做法。因此本 ADR 的報告面**必須是生成的**。

---

## 2. Decision —— 為什麼這樣設計

核心：**把 ADR 020 的「typed terminal + 觸發即處置」紀律，延伸到實驗產物；並讓「現在在哪」成為既有 fact-owner 的一個生成投影，而不是一份新的手寫文件。**

```
asset manifest  = 機械事實（誰跑的/什麼 commit/什麼 preset），寫在產出當下，fail-closed
asset inventory = 生成的查詢視圖（cited / manifested / orphan），不是存起來的分類
status report   = 既有 fact-owner 的生成投影（registry ∪ slots ∪ manifest ∪ gh），只讀不寫
```

| 選擇 | 理由 | 被否決的替代 |
|---|---|---|
| manifest 寫在**產出當下**，且 fail-closed | 唯一能讓問題**停止變大**的動作；事後考古不可靠（config 已隨 commit 漂走） | 事後掃描推斷 provenance —— 會 fabricate |
| 舊資產 **pay-on-use** 回填 | 同 ADR 020 決定不回填 177 舊檔的理由：存量債延後、不阻塞 | 一次回填 681 個目錄 —— 任務邊界不合理的信號 |
| inventory / status 皆為 **generated** | C5.1 單一寫入者；手寫必漂 | 手維一頁 status —— 即 §1.3 的第二真相 |
| manifest 只放**機械事實**，不放 verdict | verdict 的家是 slot / registry；link-don't-relabel | 在 manifest 裡寫結論 —— 第四套 archive |
| disposal 需**人工核可** | 82 G 內含不可重建的訓練產物；fail-closed 應偏向保留 | 依 mtime 自動刪 —— 不可逆且無 owner |
| 一個目錄 = 一次 run；**已有內容即拒絕 claim** | 產出端不會先清空目錄，覆寫 manifest 會讓新 run 的身分蓋在它沒覆寫到的舊檔上 ⇒ 產生**看起來可信的錯誤 provenance**，比沒有 provenance 更難發現 | `overwrite=True` / resume —— 那是 run continuation 語義，未設計前不得由預設值代答 |
| 進度 = **主線狀態轉移** | §20.7；artifact 數 / GB / PR 數都不是進度 | 以檔案數或釋出 GB 當 KPI |

---

## 3. Specification —— 完整規格需求

一條 line：`asset_provenance_and_progress_reporting`（下稱 **AP 線**），三個 workstream，WIP=1（§5）。

### W-A —— 資產身分層

> 目標：給每個實驗產物一個 machine-readable 身分，**寫在產出當下**。

**AP-1 · manifest schema**

定義 `run_manifest.json`（落在每個產物目錄根）：

```yaml
run_id:        # 目錄名
commit:        # 產出當下的 HEAD
dirty:         # working tree 是否髒（true 即宣告不可重現）
preset:        #
detector:      #
dataset:       # 含 split（如 MOT17 train-half 7seq）
host:          # hostname + GPU
cmdline:       #
started_at:    #
produced_by:   # eval | train | diagnostic | ad-hoc
claims: []     # 選填；指回 registry object id。只 link，不複寫 verdict
```

- **unknown-field fail-closed**（同 terminal slot schema 的理由：擋語意欄位漂移）。
- `claims` 為**選填**且**只放 id**。manifest 不得承載結論、數字或 terminal。

**AP-2 · 產出端落地（本 workstream 的止血點）**

eval / train 入口在寫結果時自動寫入 manifest；**寫不出 manifest 就不准寫結果**（fail-closed）。
自此刻起所有新產物都有 provenance —— 這條**單獨**就讓問題停止變大。

**AP-3 · 生成式 inventory**

`scripts/docs/asset_inventory.py` → `docs/ownership/asset_inventory.generated.md`，掃 `runs|results|out|output`，分三類**查詢視圖**（可重疊，不是存起來的欄位）：

- `cited` —— 目錄名被 `docs/` 字面引用
- `manifested` —— 有合法 manifest 但未被引用
- `orphan` —— 兩者皆無

接 `scripts/pre_push.sh`，pattern 對齊既有的 `scripts_inventory.generated.md` / `tests_inventory.generated.md`，**不發明新機制**。

**AP-4 · 存量 pay-on-use 回填**

只有當某目錄被 `evidence_ledger.md` 或 registry row 引用時，才補 manifest（且**只補可從 git / 日誌確證的欄位**，不可確證者留空，不得推測）。**不回填 681 個目錄。**

**AP-5 · disposal policy**

`orphan` ∧ mtime > N 天 ∧ 不在任何 cited 集合 → 進 `asset_disposal_candidates.generated.md`；**由 owner 人工核可才刪**。這是 82 G 的唯一合法出口。

**Exit criteria（W-A）：** 新產出 100% 帶 manifest；orphan 集合有機械定義且可重生；至少完成一輪人工核可的 disposal。
*釋出的 GB 數是副產品，不是驗收指標。*

> ⚠️ **第一條 exit criterion 目前不成立，且不由 W-A 自己解除**——見 §4.3 的 named limit。
> `scripts/eval/mot17.py` 與 `_per_seq/` 子目錄仍未覆蓋，因此 **AP-2 的狀態是 partial coverage，不是 complete**。
> PR #330 merged **不等於** AP-2 完成。

### W-B —— 關掉 ADR 020 的後半（issue #164）

**AP-6 · S3 seal 觸發（機制的心臟）**
seal 一個 study 時未填 terminal slot 即 fail-closed。沒有這條，新研究會繼續沉積，W-A 的資產債也會跟著長回來。

**AP-7 · S4 dispose flow**
compress + dispose 一個動作，讓 9 個 quarantined cluster 有出口。

**AP-8 · enum 釘死**
修 `sealed-execution` → `sealed-for-execution`，並讓 validator 把 `doc-status` enum 釘成封閉集合。

**Exit criteria（W-B）：** 至少 **1 個** cluster 走完 `quarantined → disposable → entry removed`（證明流程可執行，而非只有規格）；#164 可關閉或明確縮 scope。

### W-C —— 生成式進度報告

**AP-9 · `scripts/docs/status_report.py` → `docs/ownership/status.generated.md`**

**硬約束（違反任一條即退回 §1.3 的第二真相）：**

1. **只讀不寫**：每一格都由既有 fact-owner 推導 —— registry（state）、terminal slots（terminal）、migration manifest（quarantine）、`gh pr/issue`（in-flight）、asset inventory（debt）。
2. **不複寫 verdict、不放數字、不發明名詞**（link-don't-relabel）；每格必附來源連結。
3. **進度 = 主線狀態轉移**，此定義須寫在報告 header（§20.7）。

六個區塊：

| # | 區塊 | 來源 fact-owner |
|---|---|---|
| 1 | Sole-active charter（無則顯示 `NONE`，這是合法狀態） | O0 / `docs/TODO.md` |
| 2 | State transitions（近 30 天） —— **唯一進度指標** | registry `last_transition` |
| 3 | Open blockers by type | registry `blockers` |
| 4 | Next admissible units（合法候選集，**≠ next task**） | registry `admissible_units` |
| 5 | In-flight（open PR / issue / 未合分支） | `gh` |
| 6 | Debt（無 terminal 的 sealed study、quarantined cluster、orphan assets） | slots / manifest / AP-3 |

**AP-10 · 降級舊快照**
`status_2026-07-09.md` 標為歷史快照（`doc-status: closed` + 指向生成報告），不再假裝是 living status。

**Exit criteria（W-C）：** 一行指令產出；區塊 2 與 4 的內容 100% 可回溯到 registry 列。

---

## 4. Scope —— 自我約束

### 4.1 三條紀律（皆源自本 repo 已踩過的坑）

1. **AP 線自己要有 terminal slot。** 否則它就是第 10 個 quarantined cluster。dogfood 才能證明 S1/S3 可用。
2. **每新增/刪除 `docs/**.md` 或 `.yaml` 必須重生 `master_map.generated.md`**，否則 `tests/contract/test_migration_manifest_v0.py` fail-closed。
3. **不新增第四套 archive。** inventory 是生成的查詢視圖；manifest 是 ephemeral 的機械事實。兩者都不承載 verdict。

### 4.3 Named limit —— protected-path remainder（結構性，非一次性失誤）

> **AP-2 rollout crossing `decision_relevant` / runtime-identity protected paths requires controlled
> re-attestation and cannot be performed as ordinary provenance plumbing.**

`scripts/tools/h2_path_partition.py` 的 `decision_relevant` 分區（含整個 `src/saccade/**` 與
`_POLICY_SURFACE` 明列的 `scripts/eval/mot17.py`）只要內容變動——**新增檔案也算**——
`docs/reference/runtime_identity.generated.json` 的 `implementation` digest 就漂，
`check_runtime_identity_staleness` fail-closed。checker 的語義是 **re-attestation required**，
**沒有**「behavior probe 沒變所以視為 equivalent」的逃生門。

重新發布需要 controlled-host 的 identity-probe 與 runtime-inputs capture。因此：

- **不為一個 provenance hook 消耗一次 controlled-host republication。** 那會把 AP-2 的工程 hygiene
  與 H2 的 runtime identity authority 綁進同一個 PR，邊界反而更差。
- **後果（必須明講）：** standalone `scripts/eval/mot17.py` run 與 `_per_seq/` 子目錄**仍匿名**；
  diagnostics / sweeps / caches 本就不在 W-A 範圍。已覆蓋的是 batch eval 的 `output_root`
  （evidence_ledger 實際引用的那層）與已接線的 training run root。
- **解除條件：** 等某個**真正需要**更新 runtime coordinate 的 decision-relevant 變更出現時，
  把這個 hook 搭同一次合法 republication；或當 standalone `mot17.py` 的匿名真的成為 W-A 的主要
  blocker 時，另開一個 attestation PR。**兩者都不是本線可以順手做掉的事。**

**紀錄用途：** 避免日後有人看到 PR #330 merged 就誤認 AP-2 已 complete。

### 4.2 明確不做

- 不回填 187 份無 `doc-status` 的舊檔（pay-on-use）。
- 不對 681 個既有 `results/` 目錄做全量考古。
- 不自動刪除任何東西。
- 不移動既有 doc 的 home（C1 路由不變）。
- 不在本 ADR 內宣告任何 claim 的 state。

---

## 5. 順序與 WIP

建議 **W-A → W-C(v0) → W-B**：

- `AP-2` 是唯一能**立刻止血**的動作，成本也最低。
- W-C 可先出 v0（只吃 registry + `gh` + migration manifest，暫不吃 asset inventory），讓「一行指令看進度」即刻可用。
- W-B 最重、可分批，其價值在**新研究**而非存量。

反序（W-B 優先）同樣 admissible：若判斷「接下來還會開很多新 study」，seal 觸發的邊際價值更高。**這是 owner 的 `decision_relevance` 判斷，本 ADR 不代選**（registry 只產生合法候選集）。

**WIP=1**：AP 線在任一時點只有一個 workstream 為 sole-active charter，於 `docs/TODO.md` 掛一行並回連本 ADR。

**第一個 PR**：`AP-1` + `AP-2` + contract test = PR #330（batch eval `output_root` + 六個 training entry；protected-path remainder 見 §4.3）。

---

## 6. Consequences

**得：**
- 實驗產物從**匿名**變**可回溯**；「這個數字哪來的」有機械答案。
- 82 G 有了**合法出口**（機械候選 + 人工核可），而非只能無限增長或盲刪。
- 「現在在哪、下一步合法候選是什麼」從人腦記憶變成一行指令。
- 新債歸零：`AP-2` 之後產生的資產不再進入舊債池。

**失 / 風險：**
- 產出端多一個 fail-closed 步驟；若 manifest 寫入本身壞掉，會**擋住 eval**（這是刻意的取捨，但需要好的錯誤訊息）。
- 生成報告若被允許出現一格手寫例外，會立刻退化成第二真相；審查時對此**零容忍**。
- 存量債（187 無 status、681 個無 manifest 目錄）仍在，只是延後、pay-on-use，非消失。
- `orphan` 的定義依賴字面 grep，**會低估可達性**（語義引用抓不到）。因此它只能餵人工核可流程，不得直接授權刪除。

---

## 7. 相關

- 前半機制（doc 側 typed terminal + seal 觸發）：[ADR 020](020-doc-lifecycle-new-nogo.md)、issue #164
- 治理契約：[doc_structure_contract.md](../ownership/doc_structure_contract.md)（C4 / C5.1 / C6 / C8）
- state fact-owner：[claim_state_registry.md](../research/contracts/claim_state_registry.md)
- 可引用數字的家：[evidence_ledger.md](../research/evidence_ledger.md)
- 既有生成索引 pattern：[scripts_inventory.generated.md](../ownership/scripts_inventory.generated.md)、[tests_inventory.generated.md](../ownership/tests_inventory.generated.md)、[master_map.generated.md](../ownership/master_map.generated.md)
