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
| **typed terminal slot**(固定小 tuple)取代 1 格 GO/NO-GO | 承載 `claim_verdict`(`FALSIFIED≠NOT_IDENTIFIABLE≠PROVED`)與 `decision_outcome`(`NET_NEGATIVE≠NO_PRODUCTION_ADVANTAGE`)兩條正交軸的區別;座標/enum 不會像自由字串漂移 | 自由文字 terminal(會漂,如 sealed 兩拼);單一 enum 混多軸(如舊 `active`、或把命題真假與決策效用塞一格) |
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
line_type:             math-closed | local-math-claim | scoped-empirical | engineering-ablation

# --- 四條正交 typed 軸(勿合併) ---
claim_verdict:         # 認識論:命題本身如何(依 line_type 限定子集,§S3)
                       VERIFIED | FALSIFIED | NOT_IDENTIFIABLE | INCONCLUSIVE | NOT_EVALUATED   # empirical 組
                       | PROVED | REFUTED                                                        # deductive 組
decision_outcome:      POSITIVE | NET_NEGATIVE | NO_PRODUCTION_ADVANTAGE | NOT_ASSESSED           # 生產決策效用:與命題真假正交
lifecycle_disposition: SEALED           # 生命週期:只在 seal 時 emit 的終端標記(單值);live 態(proposed/active/parked)不進 slot
model_relation:        current | superseded    # 僅 line_type=math-closed(有 model 才有語意),其餘省略

# --- verdict 的證據定位(依 line_type) ---
verdict_locus:
  # assumptions/domain 兩維 —— math-closed / local-math-claim / scoped-empirical 皆用:
  assumptions:         <在什麼前提下 — 假設 / substrate / 門檻 / held-fixed>
  domain:              <對哪些對象成立或失效 — 量化域 / representability>
  # + line_type=math-closed(有 versioned 旗艦模型):
  model_ref:           <path to model doc>
  model_version:       vMAJOR.MINOR
  # + line_type=local-math-claim(純演繹命題、無 model):
  claim:               <局部演繹命題>
  # + line_type=scoped-empirical(empirical、有預宣告協議、無 model):
  protocol_ref:        <link 到預宣告 declaration / seal 契約>
  # + line_type=engineering-ablation(一行歸因,即上二維的非形式影子):
  attribution:         <one-line 歸因,指向證據,不複寫數字>

evidence_owner:        <link 到 fact-owner doc(registry / results)>   # link-not-copy
process_disposition:   retained | deleted-to-git | folded-to-workspace@<path>
                       # deleted-to-git 不寫 inline sha;刪除 commit 由 `git log -- <path>` 機械解析(見 §S4)
# migration_state:     quarantined  —— 遷移期 manifest-only,不進本 slot(見 §4.5)
```

- **四軸正交,不可合併**:`claim_verdict`(命題證據上如何)/ `decision_outcome`(生產決策效用如何)/ `lifecycle_disposition`(調度上如何)/ `model_relation`(與當前 master model 的關係)。**`claim_verdict` 與 `decision_outcome` 必須分開**:一條線可 `claim_verdict=VERIFIED`(機制忠實、局部 claim 成立)且 `decision_outcome=NET_NEGATIVE`(端到端淨負)——這正是本 repo 反覆出現的「機制全成立、端到端有害」型 NO-GO(如 T2:類內無可用 ranking power **且** motion 條件實測有害)。舊版把二者塞進單一 `epistemic_verdict`(含 `NET_NEGATIVE`)會逼二選一,並造成不對稱(正面走 registry、負面卻佔 epistemic 軸)。
- **`lifecycle_disposition` 收成單值 `SEALED`**:slot 只在 seal 時存在,live 調度態(`proposed`/`active`/`parked`)**不進 slot**(留在 Issue / module TODO);「是否仍承重 / 可 reopen」由 `process_disposition` + `premise_refs` 依賴圖**推導**,不另存 enum——少一個飄移面,並取代舊 `sealed-for-execution`↔`sealed-execution` 兩拼。`model_relation` 只對 `math-closed`(有 model)有語意,其餘 line_type 省略(不再對「根本沒有 model」的線硬填)。
- **`claim_verdict` 依 line_type 分 empirical / deductive 兩組**:`scoped-empirical` / `engineering-ablation` / empirical 的 `math-closed` 用 `VERIFIED`/`FALSIFIED`/`NOT_IDENTIFIABLE`/`INCONCLUSIVE`/`NOT_EVALUATED`(過了 / 沒過預宣告經驗協議、或無法識別);純演繹的 `local-math-claim` 用 `PROVED`/`REFUTED`(被推導證明 / 反證)。不混用——否則 verdict 要先看 line_type 才能解碼,正是本 ADR 要殺的 cross-axis 依賴。相容性由 §S3 的 `line_type → 允許 claim_verdict` 矩陣強制,順帶抓「在 empirical 線填 `PROVED`」這類錯。正面**生產晉升**走 registry accepted-state。
- **四種 `line_type`**(依 verdict 如何定位):`math-closed`(有 versioned 旗艦模型,定位於 `model@version` + assumptions/domain)/ `local-math-claim`(純演繹、無 model,assumptions/domain + `claim`;將來模型出現**加上** `model_ref/version` 即升格,不必原地改寫)/ `scoped-empirical`(**empirical、有完整 assumptions/domain locus + 預宣告協議、但尚無 versioned model**——D0/R1/S0/EK0/P0/T2 這類 fidelity / probe 線;帶 `protocol_ref`,claim_verdict 用 empirical 組)/ `engineering-ablation`(empirical,一行 attribution,89 條異質工程 NO-GO 的形態)。
- slot 存放位置:study 的 **per-study** terminal owner doc(如 registry 條目或 workspace entry),**單一寫入者**;混合 cluster 沒有單一 cluster-level owner(見 §4.5)。

### S2. Master model + 版本(僅 math-closed 線)

- 數學封閉線**可**宣告一份 master model doc,帶語意版本 `vMAJOR.MINOR`。
- terminal 以 `(model_ref@version, assumptions/domain)` 定位。`local-math-claim` / `scoped-empirical` 無此段(無 model),故 `model_relation` 省略;升格為 `math-closed` = 補上 `model_ref/version`。
- **Fail-closed 規則:對 model doc body 的任何語意改動,必須 bump 版本。** CI 檢查:model body diff 存在但版本未變 → fail(防止「v1.1 悄悄換意思」= enum 飄移升到模型層)。
- git 保有所有版本;無需獨立 archive。

### S3. Seal 觸發(fail-closed gate)

在 `scripts/pre_push.sh` / CI 增設:

1. **Slot presence**:任何 study 被 `SEALED`(唯一終端態)時,必須存在合法 terminal slot;缺失或欄位不合 schema → **阻擋 push**。（`proposed`/`active`/`parked` 是 live 態、不 emit slot,故不觸發此檢查。）
2. **Enum guard**:`claim_verdict` / `decision_outcome` / `lifecycle_disposition` / `model_relation` / `doc-status` 各自只能取其封閉詞彙;未知、錯拼、或**跨軸誤用**(例如把 `NOT_IDENTIFIABLE` 填進 `decision_outcome`,或把 `NET_NEGATIVE` 填進 `claim_verdict`,或既有 `sealed-for-execution`↔`sealed-execution`)→ fail。**另加 `line_type → 允許 claim_verdict` 相容矩陣**:`PROVED`/`REFUTED` 僅 `local-math-claim`;empirical 組(`VERIFIED`/`FALSIFIED`/`NOT_IDENTIFIABLE`/`INCONCLUSIVE`/`NOT_EVALUATED`)僅 `scoped-empirical`/`engineering-ablation`/empirical `math-closed`——錯配即 fail。`model_relation` 僅 `math-closed` 可出現。本 PR 一次性收斂既有漂移值。(`migration_state` 屬 manifest,由 §4.5 的 manifest 工具守,不在此 doc-slot guard。)
3. **Version-bump guard**:見 S2。
4. 填了合法 slot,才**授權** S4 的 dispose。

### S4. Dispose flow(compress + dispose 一個動作)

- seal 通過後,該 study 的 process 檔(declaration/results/amendment/preflight)成為 **dispose-eligible**。
- 兩種模式:
  - `delete-to-git`(死 process 預設):git history 保有內容,工作區移除。
  - `fold-to-workspace`:搬進 `docs/experiments/<study>/`,適用可能 reopen 的線。
- **入鏈處理(維持 `check_doc_links` 綠)**:活檔指向被 dispose 檔的連結必須改寫——fold 改指新 workspace 路徑;delete 改成 `sealed — see <registry slot>(deleted; 由 git log -- <path> 復原)` stub。
- **`deleted-to-git` 無 commit 循環**:process_disposition 只記 `deleted-to-git`,**不 pin 刪除 commit 的 SHA**(該 SHA 在 commit 建立前不存在,pin 會逼兩次 commit/amend)。刪除 commit 事後由 `git log -- <path>` 機械解析;若要更緊,可記 deleted blob SHA(pre-commit 即存在)。
- **keep-in-tree 資格規則**:一份檔留在工作區,當且僅當**有活的 artifact 需要碰到它**(verdict 索引 / 活引用)。死 process 無活物可達 → `deleted-to-git`。

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
  - **標準 fixture `docs/ownership/terminal_slot_fixtures.yaml`**(valid / invalid + `expect_error`):validator 先對它跑;reconciled map 的 6 個 slot 必須全屬 `valid`(`scoped-empirical`)型,避免「照 ADR 寫就打爆範例」或「為範例放寬 schema」。
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
    frozen_at_commit: <sha>             # 凍結時點;process_globs 只解析到「此 commit 當時存在」的檔
    process_globs: [ docs/modules/semantic/research/d0_*, ... ]   # 開放 prefix,但於 frozen_at_commit 解析成固定清單
    terminal_owner: null                # cluster 級恆 null(混合 cluster 無單一 owner,見下);per-study terminal 由恢復批次逐項 inventory
    navigation_ref: <非規範導覽圖 | null>   # 只導覽、不擁有 verdict(如 reconciled map);≠ terminal owner
    premise_refs: [ d0/core-claim ]      # 只記「失效傳播型」依賴
```

規則:
- **cluster 級 `terminal_owner` 恆 `null`,不指向導覽圖。** 混合 cluster(old-flagship 含 D0/R1/S0/EK0/P0/T2/H0,部分已 terminal、部分仍 proposed)無法用單一 owner 表示;明示 `navigation-only / not evidence` 的 reconciled map **不得**充當 terminal owner(它自己也聲明沒有 terminal-slot ownership)。真正的 terminal 是**per-study** 的(各 study 的 `evidence_owner` doc),由恢復批次逐項 inventory,不在此事先臆測。
- **`frozen_at_commit` + resolved snapshot**:`process_globs` 是開放 prefix,但只解析成 `frozen_at_commit` 當時存在的具體檔;**之後新增的檔不繼承 glob**(否則未來 lane 沿用 `d0_*`/`runtime_bridge_*`/`headline_bridge_*` 命名會被自動誤凍)。工具須檢查**一個檔不得匹配 >1 cluster**(重疊 = manifest 錯誤)。
- **frozen(= quarantined)cluster 不再被當成當前架構依據**;除定性 PR 外不改其 process 檔;搜尋 / 交接**預設排除**;git 與原路徑**暫留**(避免 churn);terminal 提取完才進正式 disposal。
- manifest **被工具消費**(`build_master_map` 讀它 gray-out;`pre_push` 讀它做 freeze guard),**不是人讀的散文**——否則凍結規則 = honor-system = 正在修的病。
- manifest **ephemeral**:遷移抽乾後**刪除**,不留成第四套 archive。

### A/B/C 是查詢視圖,不是存起來的分類

manifest 只存事實(per-study terminal 完整度、`premise_refs`);A/B/C 由查詢生成,且**可重疊**(old-flagship 同時 B 又 C:H0 尚無 terminal → 未達 A,且 premise 依賴已被推翻的 D0 → C):

```text
A = 該 cluster 全部 matched process objects 都有合法 per-study terminal(⇒ disposable 前提)
B = 尚有 matched process object 缺 terminal
C = premise_refs 命中「registry state = refuted」的 claim
```

**只有 `premise_refs`(失效傳播型依賴)參與 C**;普通 links / 歷史來源 / 證據引用 / 比較對象**不參與**(否則最小事件依賴會長成含義不明的全圖)。refuted 狀態由 `claim_state_registry` 擁有,manifest **不複寫**(no second truth)。`C` 不等於結果全廢,而是「承重模型失效,需重判它保留了什麼局部結論」——不急著逐個改寫。

### 過程層三態(遷移期,manifest-only)

`active`(仍執行)/ `quarantined`(停了但 terminal 未提取 = 債)/ `disposable`(terminal 完成,可 fold/delete)。**這三態是遷移標記,不進永久 `lifecycle_disposition` enum**。quarantined 是**排水口不是家**:

```text
quarantined + 尚有 study 缺 terminal
→ quarantined + 全部 matched study 皆有合法 per-study terminal
→ disposable
→ manifest entry removed
```

**freeze guard 要求「改 frozen process 必伴隨一個合法 migration transition(上鏈)」**,不是只要求「順手碰一下 terminal owner」——避免為繞檢查而亂動。

### 分批定性:按依賴,不按日期 / 資料夾

1. **批 1 —— 仍會影響下一個實驗的 cluster**:回答局部結論 / 前提 / 作用面 / 哪些推論依賴已死核心 → 提取 terminal → 立即 dispose。（old-flagship 已由 reconciled map 部分完成。）
2. **批 2 —— 已關閉但仍被大量引用的 cluster**(最污染檢索):先把引用改到 terminal owner,再 dispose process。
3. **批 3 —— 無活依賴的歷史 cluster**:維持隔離,pay-on-use,下次真被碰到才補 terminal + dispose。

### 兩條不混的 lane

- **研究 lane**:新任務照常,每個**當場閉環**(開題 = 局部 object + 前提 + 作用面 + 預期 terminal;關閉 = terminal record + process disposition)。無新旗艦模型時:純演繹命題用 `local-math-claim`、有預宣告協議的經驗線用 `scoped-empirical` slot,**不強掛** master map。
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
