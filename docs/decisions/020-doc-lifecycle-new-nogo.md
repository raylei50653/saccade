<!-- doc-status: proposed -->

# ADR 020: Doc-Lifecycle 管理策略 —「新 NO-GO」(typed terminal + versioned model + seal-triggered disposal)

## Status

**Proposed** (2026-07-15)

這是一份**計畫文檔**:定義問題、設計理由、與完整規格需求,作為後續 PR 的依據。落地 PR 只需含「管理策略 + 預計改動清單」;本文即該策略的完整上下文,避免日後抓不到點。

- 診斷來源:`../modules/semantic/research/` 膨脹事件 + docs 全域 lifecycle 稽核(2026-07-15)
- 相關治理契約:[doc_structure_contract.md](../ownership/doc_structure_contract.md)(本 ADR 不取代它,是在其「when to archive」欄補上**觸發機制**)

---

## 1. Context —— 解決什麼問題

### 1.1 現象

- docs/ 252 檔,**只 75 有 doc-status(177 無 = 70%)**。
- 膨脹集中在 `docs/modules/semantic/research/`:**56 檔**(第二名 detection 僅 8),其中 **48 檔擠在 2026-07-09~07-13** 五天內。
- doc-status enum 已漂移:`sealed-for-execution` vs `sealed-execution` 同義不同拼;56 檔裡 **33 個標 `active` 但實際多已 sealed**(A1 已 closed、gap-motion phase-B 已 sealed、D0 已 sealed,標籤沒翻)。
- 三處 archive 位置(`docs/archive`、`docs/research/threads/closed`、`docs/modules/semantic/research/closed`)+ in-place header,兩套封存機制未收斂。
- 整堆幾乎全互連(223 檔僅 19 真 orphan)→ **無明顯冗餘可剪**,病不是「數量」。

### 1.2 根因(retrospective)

> **真正的根 = 文檔生命週期一直缺「seal 即壓縮 + 處置」這一步。**

這缺口一直存在,只是被老 NO-GO 紀律**掩蓋**:殺一個實驗 → 寫一行進 no_go_registry → 過程不留檔。disposal 隱式、免費、當下完成,所以幾個月沒暴露。放大它的三件事:

1. **產檔率暴增(seal-bar 協議)。** 預宣告協議(§20.2 / §20.8)為承重的形式 claim 把「每 study 產檔數」從 1 拉到 4(declaration + results + amendment + preflight)。這對「not-identifiable ≠ 不交換 ≠ falsified」這種一行壓不住的 nuance 是**正確的**,但協議**只有前半**(predeclare → execute → reveal),**沒有後半**(seal → compress → dispose)。verdict 有壓進 registry/memory,process 檔沒人處置。
2. **快 pivot 沉積。** 兩週內 gate/score → safe-region → D0 → bridge → headline 連續轉向,前一個還沒壓縮封存就開新 cluster,每次轉向留一層沉積。pivot 快是研究該有的,但它**預設有一個便宜的 disposal 步**,而那步不存在。
3. **新 doc-org 疊加不取代。** modules / EvalConfig views / registry / experiments 四套分解並存,無一權威 → 量的混亂上再疊檢索混亂。

**惡性循環(= 為何「感覺無解」):** 沒 dispose → sealed 檔仍掛 `active` → 機械分不出哪個死 → 更不敢 dispose → 沉積更多。**混亂本身抹掉了清理所需的資訊。**

### 1.3 為什麼「剪枝」與「重寫摘要」都不是解

- **剪枝無槓桿**:223 檔僅 19 真 orphan;已封證據被活 thread 引用(承重的 anti-re-litigation 紀錄),刪不得。
- **另寫摘要 = second truth**:結論其實很少(no_go_registry ~數十 + registry 23 objects + threads 19 + 手維 MEMORY.md 111 行,約 100 行量級),已存在三四份。再寫一份 restate verdict 的總結,只會製造會漂移的第二真相,正是本病的病根。

---

## 2. Decision —— 為什麼這樣設計

核心:**把老 NO-GO「殺即壓縮即處置」的收尾紀律,重新機制化,並升級成能承載形式 claim nuance 的 typed 形式。**

```
typed terminal = ( versioned_model @ coordinate  |  engineering empirical verdict )
   · seal 時 fail-closed 觸發(沒填 slot 不准封)
   · git 管 model 版本 + dead process
   · master map = 生成骨架 + 手維壓縮原子
   · 人手只對帳「版本落後集」
   · 一線一模型,跨 repo 靠 slot 統一
```

設計選擇與理由:

| 選擇 | 理由 | 被否決的替代 |
|---|---|---|
| **typed terminal slot**(固定小 tuple)取代 1 格 GO/NO-GO | 承載 epistemic 軸上 `FALSIFIED≠NOT_IDENTIFIABLE≠NET_NEGATIVE` 的區別,且與 lifecycle 軸正交;座標/enum 不會像自由字串漂移 | 自由文字 terminal(會漂,如 sealed 兩拼);單一 enum 混兩軸(如舊 `active`) |
| **slot 必須便宜(~1 分鐘填完)** | 老 NO-GO 之所以能當 disposal,唯一原因是 30 秒寫完;貴到快 pivot 會跳過 = 重演堆積 | 每 study 都畫完整模型圖(貴,必被跳過) |
| **一線一模型,非一 repo 一模型** | 專案不是一個數學物件(89 條 NO-GO 多是異質工程 ablation);且有「機制全成立、端到端淨負」的**不可定位** failure | 全 repo 統一主模型(過度一般化) |
| **主模型 versioned + git**;terminal pin 版本 | model 改了是新版本,舊 terminal 準確指舊版座標,永不錯位;靜默飄移 → 可見版本落後 | 不版本化(model 一改,所有 terminal misplace) |
| **seal 觸發 dispose**(填 slot = 授權處置) | 補回缺席的後半步;compress 與 dispose 綁成一個動作,不會漏做 | 定期巡檔清理(一定落後 = 飄移源) |
| **git 當死 process 的庫** | 已 commit 的檔刪掉零內容損失;「archive 放哪」問題直接蒸發 | 第 N 個 archive 目錄 |
| **master map = 生成骨架 + 手維壓縮原子** | 生成器信任標籤、會原樣傳播謊言;**只有人拿 map 對現實才抓得到語意飄移**。手維可持續的前提 = 壓縮夠小(111 行 MEMORY.md 正證,250 doc labels 反證) | 純生成(抓不到 label-vs-現實 落差)/ 純手維(surface 太大必落後) |

---

## 3. Specification —— 完整規格需求

### S1. Typed terminal slot(新 NO-GO 記錄)

每個 study / 探究線在 **seal 時必須** emit 一個 terminal slot;固定 schema、機械可檢:

```yaml
study_id:              <stable id>
line_type:             math-closed | engineering-ablation | local-math-claim

# --- 三條正交 typed 軸(勿合併) ---
epistemic_verdict:     VERIFIED | FALSIFIED | NOT_IDENTIFIABLE | NET_NEGATIVE
                       | INCONCLUSIVE | NOT_EVALUATED       # empirical 線(math-closed / engineering-ablation)
                       | PROVED | REFUTED                   # 純演繹線(local-math-claim only)
lifecycle_disposition: PROPOSED | ACTIVE | PARKED | SEALED | CLOSED   # 生命週期 / 調度;single writer
model_relation:        current | superseded                 # 與當前 master model 的關係(SUPERSEDED 從 lifecycle 移來此軸)

# --- verdict 的證據定位(依 line_type) ---
verdict_locus:
  # 兩維(數學命題的有效性只沿此二維移動)—— math-closed 與 local-math-claim 皆用:
  assumptions:         <在什麼前提下 — 假設 / substrate / 門檻 / held-fixed>
  domain:              <對哪些對象成立或失效 — 量化域 / representability>
  # + line_type=math-closed(已有 versioned 旗艦模型):
  model_ref:           <path to model doc>
  model_version:       vMAJOR.MINOR
  # + line_type=local-math-claim(尚無旗艦模型、但已有局部數學命題):
  claim:               <局部命題>
  # + line_type=engineering-ablation(改用一行歸因,即上二維的非形式影子):
  attribution:         <one-line 歸因,指向證據,不複寫數字>

evidence_owner:        <link 到 fact-owner doc(registry / results)>   # link-not-copy
process_disposition:   retained | deleted-to-git@<sha> | folded-to-workspace@<path>
# migration_state:     quarantined  —— 遷移期 manifest-only,不進本 slot(見 §4.5)
```

- **三軸正交,不可合併**:`epistemic_verdict`(證據上如何)/ `lifecycle_disposition`(調度上如何)/ `model_relation`(與當前 master model 的關係)。同一 study 可 `lifecycle=SEALED` 且 `epistemic=NOT_IDENTIFIABLE`(封了但沒判出來),或 `lifecycle=SEALED` 且 `model_relation=superseded`(當年封的、如今被新模型取代)。把它們塞一個 enum 正是舊 `active` 同時扛多義而漂移的病根。**`SUPERSEDED` 從 lifecycle 移到 `model_relation`** 就是這道理——它講的是「與模型的關係」,不是「調度狀態」。
- **`VERIFIED`(epistemic)= 通過預宣告經驗檢查、在 scope 內成立的正面判決**(如 `R1_FAITHFUL`)——這是 empirical protocol 判決。**純演繹的 `local-math-claim` 用獨立值 `PROVED`/`REFUTED`,不與 `VERIFIED`/`FALSIFIED` 過載**:後者代表「過了 fidelity / 預宣告協議」,前者代表「被推導證明」;混用會讓 verdict 必須先看 `line_type` 才能解碼,正是本 ADR 要殺的 cross-axis 解碼依賴。`NET_NEGATIVE`/`NOT_EVALUATED` 對演繹線通常 N/A。相容性由 enum guard 的 `line_type → 允許 verdict` 矩陣強制(§S3),順帶抓「在 empirical ablation 上填 `PROVED`」這類錯。正面**生產晉升**走 registry accepted-state,不佔 epistemic 軸。**`PROPOSED`(lifecycle)= 已宣告未 seal**(如 H0),與 `ACTIVE` 區分。(`PROVED`/`REFUTED` + `PROPOSED` + `model_relation` + `local-math-claim` 皆為 2026-07-15 以旗艦線實測本 schema 撞出缺口後補入。)
- **三種 `line_type`**:`math-closed`(有 versioned 旗艦模型,verdict 以 `model@version` 定位於 assumptions/domain 兩維)/ `local-math-claim`(**尚無旗艦模型、但已有局部數學命題**——同用 assumptions/domain 兩維但無 model;將來模型出現時**加上** `model_ref/version` 即升格,不必原地改寫 terminal)/ `engineering-ablation`(用一行 attribution)。
- slot 存放位置:study 的 terminal owner doc(如 registry 條目或 workspace entry),**單一寫入者**。

### S2. Master model + 版本(僅 math-closed 線)

- 數學封閉線**可**宣告一份 master model doc,帶語意版本 `vMAJOR.MINOR`。
- terminal 以 `(model_ref@version, assumptions/domain)` 定位。`local-math-claim` 無此段(無 model);升格為 `math-closed` = 補上 `model_ref/version`。
- **Fail-closed 規則:對 model doc body 的任何語意改動,必須 bump 版本。** CI 檢查:model body diff 存在但版本未變 → fail(防止「v1.1 悄悄換意思」= enum 飄移升到模型層)。
- git 保有所有版本;無需獨立 archive。

### S3. Seal 觸發(fail-closed gate)

在 `scripts/pre_push.sh` / CI 增設:

1. **Slot presence**:任何 study 的 `disposition` 轉入 terminal(`sealed`/`closed`)時,必須存在合法 terminal slot;缺失或欄位不合 schema → **阻擋 push**。
2. **Enum guard**:`epistemic_verdict` / `lifecycle_disposition` / `model_relation` / `doc-status` 各自只能取其封閉詞彙;未知、錯拼、或**跨軸誤用**(例如把 `NOT_IDENTIFIABLE` 填進 lifecycle,含既有 `sealed-for-execution`↔`sealed-execution`)→ fail。**另加 `line_type → 允許 verdict` 相容矩陣**:`PROVED`/`REFUTED` 僅 `local-math-claim`;`VERIFIED`/`FALSIFIED`/`NOT_IDENTIFIABLE`/`NET_NEGATIVE` 僅 empirical 線(`math-closed`/`engineering-ablation`)——錯配即 fail。本 PR 一次性收斂既有漂移值。(`migration_state` 屬 manifest,由 §4.5 的 manifest 工具守,不在此 doc-slot guard。)
3. **Version-bump guard**:見 S2。
4. 填了合法 slot,才**授權** S4 的 dispose。

### S4. Dispose flow(compress + dispose 一個動作)

- seal 通過後,該 study 的 process 檔(declaration/results/amendment/preflight)成為 **dispose-eligible**。
- 兩種模式:
  - `delete-to-git`(死 process 預設):git history 保有內容,工作區移除。
  - `fold-to-workspace`:搬進 `docs/experiments/<study>/`,適用可能 reopen 的線。
- **入鏈處理(維持 `check_doc_links` 綠)**:活檔指向被 dispose 檔的連結必須改寫——fold 改指新 workspace 路徑;delete 改成 `sealed — see <registry slot> / git@<sha>` stub。
- **keep-in-tree 資格規則**:一份檔留在工作區,當且僅當**有活的 artifact 需要碰到它**(verdict 索引 / 活引用)。死 process 無活物可達 → `delete-to-git`。

### S5. Master map(生成骨架 + 手維原子)

- **Generated skeleton**(`scripts/docs/build_master_map.py`,產物 generated、禁手改):由 repo 機械組裝——檔清單、doc-status 有無、link 圖、日期、terminal-slot inventory、model 版本。結構性飄移(缺 status/slot)由此 fail-closed。
- **Hand-maintained atoms**:只有那 ~N 個 terminal slot(壓縮原子)由人維護 / 對帳現實。
- 生成 map 需標出 **version-lag flags**:`model_version < 當前 model 版本` 的 terminal → 標記待人對帳。**這就是飄移偵測器。**

### S6. Drift reconciliation

- 人手對帳範圍 = **版本落後集**(非全 repo)。
- **Residual(接受、不可約)**:版本號說 verdict 是對哪個 model 版本下的,**不**斷言它在當前 model 版本下還成不成立——那是研究判斷,人做。版本制只讓它**有界、可見**。

---

## 4. Scope —— 預計 PR 改動(lean)

**首個 PR 含:**
- 本策略文檔(即本 ADR)+ 預計改動清單。
- 機械件:
  - `docs/ownership/` 或本 ADR 附錄定義 terminal slot schema(S1)。
  - `scripts/pre_push.sh` / CI 增 S3 三道 fail-closed 檢查。
  - 一次性收斂既有 enum 漂移值(S3.2)。
  - `build_master_map.py` 生成骨架(S5),先只做 inventory + version-lag flag。

**Non-goals(明確不做):**
- 一次回填 177 份無 status 舊檔。
- 一次搬遷 / 折疊全部 sealed studies。
- 追溯補齊所有歷史 terminal slot。

> **Pay-on-use**:一個 sealed study 只在**下次被碰到(reopen)**時才折成 workspace / 補 slot。舊債逐次償還,不做大 bang。

---

## 4.5 過渡期:邏輯半封存(migration manifest)

舊旗艦核心被推翻後(見 reconciled map),既不能整批立即搬/刪,也不能等全部舊實驗重新定性完才恢復研究。過渡順序:

```text
先止血隔離  →  按依賴分批定性  →  定性完才正式 dispose
```

半封存是**邏輯隔離,不是物理搬檔**——文件互相引用太多,搬檔 = 又一次整體 link 改寫(見 S4 的 159 入鏈教訓)。

### 機制:ephemeral、machine-consumed 的 migration manifest

一份很小的 manifest(`docs/ownership/doc_migration_manifest.yaml`),cluster 級 ~10 條,**只記機械需要的事實**:

```yaml
clusters:
  old-flagship:
    migration_state: quarantined        # 唯一狀態;`frozen` 由此推導,不另存欄位
    process_globs: [ docs/modules/semantic/research/d0_*, ... ]
    terminal_owner: <ref | null>        # null = 尚未定性
    premise_refs: [ d0/core-claim ]      # 只記「失效傳播型」依賴
```

規則:
- **frozen(= quarantined)cluster 不再被當成當前架構依據**;除定性 PR 外不改其 process 檔;搜尋 / 交接**預設排除**;git 與原路徑**暫留**(避免 churn);terminal 提取完才進正式 disposal。
- manifest **被工具消費**(`build_master_map` 讀它 gray-out;`pre_push` 讀它做 freeze guard),**不是人讀的散文**——否則凍結規則 = honor-system = 正在修的病。
- manifest **ephemeral**:遷移抽乾後**刪除**,不留成第四套 archive。

### A/B/C 是查詢視圖,不是存起來的分類

manifest 只存事實(`terminal_owner`、`premise_refs`);A/B/C 由查詢生成,且**可重疊**(PR #165 證明同一 cluster 可同時 A 又 C):

```text
A = terminal_owner 存在且合法
B = terminal_owner 缺失
C = premise_refs 命中「registry state = refuted」的 claim
```

**只有 `premise_refs`(失效傳播型依賴)參與 C**;普通 links / 歷史來源 / 證據引用 / 比較對象**不參與**(否則最小事件依賴會長成含義不明的全圖)。refuted 狀態由 `claim_state_registry` 擁有,manifest **不複寫**(no second truth)。`C` 不等於結果全廢,而是「承重模型失效,需重判它保留了什麼局部結論」——不急著逐個改寫。

### 過程層三態(遷移期,manifest-only)

`active`(仍執行)/ `quarantined`(停了但 terminal 未提取 = 債)/ `disposable`(terminal 完成,可 fold/delete)。**這三態是遷移標記,不進永久 `lifecycle_disposition` enum**。quarantined 是**排水口不是家**:

```text
quarantined + terminal missing
→ quarantined + terminal present
→ disposable
→ manifest entry removed
```

**freeze guard 要求「改 frozen process 必伴隨一個合法 migration transition(上鏈)」**,不是只要求「順手碰一下 terminal owner」——避免為繞檢查而亂動。

### 分批定性:按依賴,不按日期 / 資料夾

1. **批 1 —— 仍會影響下一個實驗的 cluster**:回答局部結論 / 前提 / 作用面 / 哪些推論依賴已死核心 → 提取 terminal → 立即 dispose。（old-flagship 已由 reconciled map 部分完成。）
2. **批 2 —— 已關閉但仍被大量引用的 cluster**(最污染檢索):先把引用改到 terminal owner,再 dispose process。
3. **批 3 —— 無活依賴的歷史 cluster**:維持隔離,pay-on-use,下次真被碰到才補 terminal + dispose。

### 兩條不混的 lane

- **研究 lane**:新任務照常,每個**當場閉環**(開題 = 局部 object + 前提 + 作用面 + 預期 terminal;關閉 = terminal record + process disposition)。無新旗艦模型時用 `local-math-claim` slot,**不強掛** master map。
- **恢復 lane**:一次只抽乾**一個**舊 cluster。

**不開「整理全部 56 份」的大任務**——跨整體修正本身就是任務邊界不合理的信號。新任務即刻閉環讓**新債歸零**,舊債才會慢慢下降。

---

## 5. Consequences

**得:**
- 飄移從**靜默**變**機械可見**(version-lag flag),解「飄移了都不知道」。
- disposal 有了**觸發**(seal gate),不再靠人記得清。
- 工作區按需縮小;手維面維持在可持續的小壓縮層(no second truth,link-not-copy)。

**失 / 風險:**
- 旗艦線要付**建模型 + 維護模型**成本。
- **version-bump 紀律必須守住**,否則病往模型層高一階復發(S2 的 CI 檢查是防線)。
- 舊債(177 無 status、既有 sealed 未折)仍是 slog,只是**延後、pay-on-use**,非消失。

---

## 6. 相關

- 診斷:doc-lifecycle 痛點(memory `project_doc_lifecycle_missing_top_layer`)
- 定案策略摘要:memory `project_doc_lifecycle_new_nogo_strategy`
- 旗艦 typed 模型範本:owner charter preamble 的帶型別表示—化約—驗證圖(memory `project_charter_preamble_status_map`)
- 治理不重複:[doc_structure_contract.md](../ownership/doc_structure_contract.md)、seal-bar 前半([feedback] 預宣告協議 §20.2/§20.8)
