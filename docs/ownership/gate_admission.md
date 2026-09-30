# Gate admission

**中文：** 閘門准入準則——新增或修改一道檢查時必須回答的問題  
**Status:** normative (docs governance) — 本檔是飄移分類、核心判準與閘門准入欄位的**唯一** owner；其他入口只連結，不重述  
**Issue:** [#493](https://github.com/raylei50653/saccade/issues/493)（PR-1 [ADR 027](../decisions/027-historical-checks-by-purpose.md)、PR-2 規則作用分流、PR-3 [§20.11](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) 分級研究）

**本檔不授權增加任何閘門。** 新增或加嚴一道檢查仍然需要它自己的變更、理由與證據；本檔只規定
那個變更必須填什麼。「符合下方分類」不等於「因此可以直接加 enforcement」。

---

## 1. 核心判準

> **要退休的是「一般開發必須反覆證明舊研究仍描述 HEAD」的負擔；要保留的是「證據沒有被改寫、
> 適用座標明確、正式結論沒有超出有效量測」這三項最低防護。**

治理存在是因為**飄移的代價很高**。執行成本可以接受；不防飄移、只製造同步工作的檢查不該存在。
去留按**檢查的用途**與**規則的作用**決定，不按研究線名稱或 checker 檔名。

## 2. 飄移分類

| 飄移類型 | 例子 | 代價 | 預設處理 |
|---|---|---|---|
| **結論飄移** | 舊數字被當成現況、offline 量被當成 runtime 量、宣稱強於量測、exploratory 被正式鏈引用 | 高 | fail-closed |
| **行為飄移** | 改了程式碼，結果悄悄跟著變 | 高 | fail-closed |
| **文件飄移** | 索引、header、manifest、衍生清單不同步 | 低，重生即可 | 能生成的就生成（CI 重生、checked-in 版為可落後 snapshot）；其餘 warn。**例外**：把狀態投影錯（closed 被列成 active）或規則層 authority 錯置屬於結論飄移，照結論飄移處理 |
| **歷史飄移** | 已結案證據包依賴的程式碼後來被改 | 近乎零，結論已綁在 commit / tag 上 | 不在一般開發 gate 裡要求 HEAD 等於舊 source；封存完整性與防冒用仍 fail-closed（[ADR 027 §2](../decisions/027-historical-checks-by-purpose.md)） |

同一個 checker 可能同時含多種用途的規則；分類以**單條規則**為單位，不以檔案為單位
（例：[C6.4](doc_structure_contract.md) 的 L1／L5 是 warn、L2–L4 是 fail）。

## 3. 准入欄位

新增一道檢查，或把既有檢查從 warn 改 fail、擴大它擋住的入口時，在 PR 描述填：

| 欄位 | 內容 |
|---|---|
| **防的飄移** | §2 的哪一類；說得出「沒有這道檢查時，哪一種錯誤結論或行為會悄悄進 main」 |
| **擋住的入口** | 列出實際執行它的入口：pre-commit hook（`.githooks/`）、`scripts/pre_push.sh`、pytest 預設收集、CI workflow（寫 job／step 名）。沒有列出的入口視為不擋 |
| **fail-closed 或 warn 的理由** | fail-closed 需對應結論飄移或行為飄移，或封存完整性／防冒用；文件飄移預設 warn 或自動生成 |
| **失效後的處理** | 紅燈時的正確動作（修正、重生、republish、supersession…），以及誰有權做 |

移除或降級一道檢查時，填同樣欄位說明它為什麼不防（或不再防）上述飄移。

## 4. 不因本檔改變的事

- runtime identity 的觸發點與 [republication runbook](../reference/runbooks/runtime_identity_republication.md)；
- 已結案研究的 verdict、封存 packet bytes、pinned validator／fixture；
- 已凍結研究的執行契約（新制度不回頭套用）；
- 不新增 gate 統計或治理清冊：本檔不列舉現有閘門，各閘門的用途寫在它自己的 PR／ADR／契約裡。

## 5. 相關規則的 owner

| 主題 | Owner |
|---|---|
| 歷史檢查按用途分流（replay／attested／預設 gate） | [ADR 027](../decisions/027-historical-checks-by-purpose.md) |
| 文件結構規則作用（warn／strict）、衍生索引 | [doc_structure_contract C6.4 · C9](doc_structure_contract.md) |
| 研究分級、freeze-gated 執行、attempts | [contract §20.11](../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)、[studies/](../research/studies/README.md) |
| PR 前檢查清單 | [`scripts/pre_push.sh`](../../scripts/pre_push.sh) |

---

## One-liner

> 每道閘門都要說得出它防哪一種飄移、擋在哪個入口；說不出來的，就不該是閘門。
