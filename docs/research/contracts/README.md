# Research Contracts（跨研究規範層）

**這裡是規範，不是筆記。** 本目錄的文件定義**方法、證據語義與宣稱邊界**：它們**先於**任何研究單元存在，研究單元只能**引用與實例化**，不得改寫、不得繞過、**不得自造平行的統計機器**。

> **開任何 gate / safe-region / reject-rule / fidelity 研究之前先讀這裡。**
> 若你正在自己定義 ε、UCB、independence unit、claim level 或 terminal 分類，**停下來**——那些已經有 owner。

實驗筆記與 ablation 在 [../eval/](../eval/README.md)；文檔治理（檔案家、索引、promotion）在 [../../ownership/](../../ownership/README.md)。本目錄**不放數字、不放結果、不放 study note**。

## Index

| 文件 | 角色 |
|------|------|
| **[claim_state_registry.md](claim_state_registry.md)** | **當前狀態的 fact-owner（先查這個）。** 每個研究對象站在哪一格（layer／ladder／state／**substrate**／open limits／typed blockers／dependencies／decision relevance），以及**合法候選集**如何被推導（錯層 → 改層；substrate 未證 → 先做 L4 transfer；relevance 過不了反事實測試 → 不得取 WIP 鎖）。**它只產生候選集，不選任務**——`next admissible unit ≠ next task`，選擇是 O0 的事。 |
| **[statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)** | **最上層數學契約。** 非對稱損失下的安全域：\(\max_\theta G_{\mathrm{FP}}(\theta)\) s.t. \(L_{\mathrm{GT}}(\theta)\le\varepsilon\) · feasible / productive-safe / robust-feasible set · region geometry（thickness、boundary distance、interior）· **有限樣本與 independence unit 強制宣告（§8.1）** · robustness 軸（sequence / fold / **substrate** / perturbation / execution，§9）· **claim ladder L0–L6（§10）** · **§13 forbidden shortcuts** |
| **[runtime_quantity_fidelity_protocol.md](runtime_quantity_fidelity_protocol.md)** | **runtime 量的忠實性協議。** 任何聲稱代表 production runtime quantity 的 offline 量，不得因公式同形/同名而繼承語義。core lemma：**同一個 \(f\)、不同的時域化約算子 \(R\)** → shadow capture → 版本化 key/ID universe → partition 守恆 → 封門檻 → 五項驗證 → 四分 terminal → append-only amendment |
| **[signal_table_schema.md](signal_table_schema.md)** | **分層契約。** A/B1/B2 訊號表；**§0.4 L0 safe-reject**；**§0.5 Gate vs Score**（membership vs ordering — 決定一個問題屬於哪一層） |
| **[boolean_composition_semantics_contract.md](boolean_composition_semantics_contract.md)** | **組合語義。** Ω/Θ 分型、三值 predicate、universe identity、threshold edge、role closure、canonical grammar、closed-loop firewall |
| **[safe_region_asset_contract.md](safe_region_asset_contract.md)** | **打包契約（R0-B RegionAsset）。** 把已封存的 evidence 決定性地打包成 region asset；claim level 與成熟度；**transfer / intervention / production 皆尚未授權** |

執行層 procedure（study-specific）：[../eval/procedures/](../eval/procedures/)。

## Precedence（衝突時誰說了算）

```text
statistical_robust_feasible_set  ──┬─►  一切 gate / safe-region / reject 研究的數學語言與 claim ladder
（最上層；ε、UCB、independence  │
  unit、L0–L6、forbidden        │
  shortcuts 都由它定義）         │
                                 ├─►  boolean_composition_semantics   （組合式規則的語義）
                                 ├─►  safe_region_asset_contract       （把結果打包成資產）
                                 └─►  signal_table_schema §0.5         （這問題是 gate 還是 score）

runtime_quantity_fidelity_protocol ──►  任何「這個 offline 量代表 production 量」的宣稱
```

- **層級衝突**：分層歸屬（gate vs score）以 `signal_table_schema §0.5` 為準；統計與 claim level 以 feasible-set 框架為準；量的忠實性以 fidelity protocol 為準。
- **研究單元不得**在 declaration 內重新定義上述任何一項；只能**引用**（並註明出處），或**明確宣告偏離**並說明理由。

## 兩份契約的接合處（**重要，且曾經缺席**）

feasible-set 框架的 **§9.3 substrate robustness** 與 fidelity protocol 的 **core lemma** 是**同一件事的兩個視角**：

- 框架說：substrate 包含 **score definitions / feature extraction / hook placement / online state**；policy 若跨不過 substrate 變更，就不是 substrate-robust，**claim ladder 停在 L4 以下**。
- protocol 說：offline 重建的量 \(s = f(R_{\text{off}}(x))\) 與 runtime 的 \(f(R_{\text{ker}}(x))\) **不是同一個量**，即使 \(f\) 同形。

合起來的推論（**兩份文件單獨都推不出來，這是本目錄存在的理由之一**）：

> **一個在 offline 座標上證明安全的域，其 \(L_{\mathrm{GT}}\) 界不會自動轉移到 runtime 座標。**
> 座標來源的改變**就是**一次 substrate 變更 → 需要 **L4 portability 審計**，而不是繼承。
> 對照框架 §13 的禁則：**offline safe \(\not\Rightarrow\) online effective**。

現行實例：[S0 safe-domain axis transfer](../../modules/semantic/research/safe_domain_runtime_transfer_declaration_20260712.md)（[thread](../threads/closed/runtime_faithful_safe_domain_20260712.md)）。

## 已知缺口

| 缺口 | 狀態 |
|---|---|
| **score-ranking 層沒有對應的規範契約** | gate 側齊備（數學框架＋分層＋組合語義＋資產打包），但 **score 側是空的**：event-local rank / margin / top-1 的證據語義、calibration 語義、與 score 的 claim ladder 目前**沒有 owner**。score 線開啟前必須補；在那之前，任何 ranking 研究只能引用 `signal_table_schema §0.5` 的邊界，不得自造 claim level。 |
