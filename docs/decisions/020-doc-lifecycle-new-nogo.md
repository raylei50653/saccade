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
line_type:             math-closed | engineering-ablation

# --- 兩條正交 typed 軸(勿合併成一個 enum) ---
epistemic_verdict:     VERIFIED | FALSIFIED | NOT_IDENTIFIABLE | NET_NEGATIVE
                       | INCONCLUSIVE | NOT_EVALUATED       # 認識論 / 實證判決
lifecycle_disposition: PROPOSED | ACTIVE | PARKED | SUPERSEDED | SEALED | CLOSED  # 生命週期 / 調度;single writer

# --- verdict 的證據定位(依 line_type) ---
verdict_locus:
  # line_type=math-closed:
  model_ref:           <path to model doc>
  model_version:       vMAJOR.MINOR
  coordinate:          <該版模型內的位置指標>
  # line_type=engineering-ablation:
  attribution:         <one-line 歸因,指向證據,不複寫數字>

evidence_owner:        <link 到 fact-owner doc(registry / results)>   # link-not-copy
process_disposition:   retained | deleted-to-git@<sha> | folded-to-workspace@<path>
```

- **兩軸正交,不可合併**:`epistemic_verdict` 答「這命題證據上如何」,`lifecycle_disposition` 答「這條線現在調度上如何」。同一 study 可以是 `lifecycle=SEALED` 且 `epistemic=NOT_IDENTIFIABLE`(封了但沒判出來),或 `lifecycle=PARKED` 且 `epistemic=INCONCLUSIVE`。把兩者塞一個 enum 正是舊 `active` 標籤同時扛「還在做」與「還沒定論」而漂移的病根。
- **`VERIFIED`(epistemic)= 通過預宣告檢查、在 scope 內成立的正面判決**(如 `R1_FAITHFUL`);它**不同於**晉升生產——生產晉升走 registry accepted-state,不佔 epistemic 軸。**`PROPOSED`(lifecycle)= 已宣告但未授權執行 / 未 seal**(如 H0),與 `ACTIVE`(執行中)區分。此二值為 2026-07-15 以旗艦線(D0/R1/S0/…)實測本 schema 時補入:`R1_FAITHFUL` 無正面值可表達、H0 無 `PROPOSED` 可表達(見 ADR 相關的 reconciled map)。
- 工程線填 `epistemic_verdict + attribution` 即結案(~1 分鐘)。旗艦數學線才用 `model_ref@version + coordinate`。
- slot 存放位置:study 的 terminal owner doc(如 registry 條目或 workspace entry),**單一寫入者**。

### S2. Master model + 版本(僅 math-closed 線)

- 數學封閉線**可**宣告一份 master model doc,帶語意版本 `vMAJOR.MINOR`。
- terminal 以 `(model_ref@version, coordinate)` 引用。
- **Fail-closed 規則:對 model doc body 的任何語意改動,必須 bump 版本。** CI 檢查:model body diff 存在但版本未變 → fail(防止「v1.1 悄悄換意思」= enum 飄移升到模型層)。
- git 保有所有版本;無需獨立 archive。

### S3. Seal 觸發(fail-closed gate)

在 `scripts/pre_push.sh` / CI 增設:

1. **Slot presence**:任何 study 的 `disposition` 轉入 terminal(`sealed`/`closed`)時,必須存在合法 terminal slot;缺失或欄位不合 schema → **阻擋 push**。
2. **Enum guard**:`epistemic_verdict` / `lifecycle_disposition` / `doc-status` 各自只能取其封閉詞彙;未知、錯拼、或**跨軸誤用**(例如把 `NOT_IDENTIFIABLE` 填進 lifecycle,含既有 `sealed-for-execution`↔`sealed-execution`)→ fail。本 PR 一次性收斂既有漂移值。
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
