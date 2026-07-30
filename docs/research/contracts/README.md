# Research Contracts（跨研究規範層）

**這裡是規範，不是筆記。** 本目錄的文件定義**方法、證據語義與宣稱邊界**：它們**先於**任何研究單元存在，研究單元只能**引用與實例化**，不得改寫、不得繞過、**不得自造平行的統計機器**。

> **開任何 gate / safe-region / reject-rule / fidelity 研究之前先讀這裡。**
> 若你正在自己定義 ε、UCB、independence unit、claim level 或 terminal 分類，**停下來**——那些已經有 owner。

實驗筆記與 ablation 在 [../eval/](../eval/README.md)；文檔治理（檔案家、索引、promotion）在 [../../ownership/](../../ownership/README.md)。本目錄**不放數字、不放結果、不放 study note**。

## Index

| 文件 | 角色 |
|------|------|
| **[claim_state_registry.md](claim_state_registry.md)** | **當前狀態的 fact-owner（先查這個）。** 每個研究對象站在哪一格（layer／ladder／state／**substrate**／open limits／typed blockers／dependencies／decision relevance），以及**合法候選集**如何被推導（錯層 → 改層；substrate 未證 → 先做 L4 transfer；relevance 過不了反事實測試 → 不得取 WIP 鎖）。**它只產生候選集，不選任務**——`next admissible unit ≠ next task`，選擇是 O0 的事。 |
| **[statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)** | **最上層數學契約。** 非對稱損失下的安全域：\(\max_\theta G_{\mathrm{FP}}(\theta)\) s.t. \(L_{\mathrm{GT}}(\theta)\le\varepsilon\) · feasible / productive-safe / robust-feasible set · region geometry（thickness、boundary distance、interior）· **有限樣本與 independence unit 強制宣告（§8.1）** · robustness 軸（sequence / fold / **substrate** / perturbation / execution，§9）· **claim ladder L0–L6（§10）** · **§13 forbidden shortcuts** · **role-aligned experiment contract（§20）＋ declaration seal bar（§20.8 — 宣告邊界規則的唯一 owner：typed κ、凍結自由度、機械可判定、exhaustive terminal、scoped exhaustion naming、blind→reveal hash binding）＋ dual-space accounting / reduction typing（§20.9 — 第四宣告座標 substrate、ρ_v/a_{v,t} 型別介面、conservation identities、跨空間推論義務、typed failure semantics）＋ online / research mutual exclusion（§20.10 — `ONLINE_OPEN → RESEARCH_OPEN → RESEARCH_CLOSED → ONLINE_OPEN` 狀態機、預設凍結軸與 per-instance 升級、`sealed`/`voided` 兩種 disposition、鎖必須在它所凍結的軸之外、以及明列的 non-goals）** |
| **[runtime_quantity_fidelity_protocol.md](runtime_quantity_fidelity_protocol.md)** | **runtime 量的忠實性協議。** 任何聲稱代表 production runtime quantity 的 offline 量，不得因公式同形/同名而繼承語義。core lemma：**同一個 \(f\)、不同的時域化約算子 \(R\)** → shadow capture → 版本化 key/ID universe → partition 守恆 → 封門檻 → 五項驗證 → 四分 terminal → append-only amendment |
| **[signal_table_schema.md](signal_table_schema.md)** | **分層契約。** A/B1/B2 訊號表；**§0.4 L0 safe-reject**；**§0.5 Gate vs Score**（membership vs ordering — 決定一個問題屬於哪一層） |
| **[score_ranking_evidence_contract.md](score_ranking_evidence_contract.md)** | **Owner-accepted L2 score-ranking 契約 v1。** 定義 event-local rank／margin／top-1、candidate-universe identity、calibration 分離與 `SR0`–`SR6` claim ladder；v1 schema／validator／正負 fixtures 提供 fail-closed declaration surface。契約 acceptance 只補上 transition semantics，不自動啟動研究或推進任何 object。 |
| **[gctm_b1_slot_identity_decision_v1.json](gctm_b1_slot_identity_decision_v1.json)** | **Owner-accepted、machine-readable B1-slot identity 決策。** 固定 `GCTM_B1 != H0_ROUTE5_B1` 與 `relation: coexist`，隔離 `GCTM_D1` diagnostic authority，記錄 runtime compatibility fail-closed gate，並投影空候選集／空 WIP。通用 schema 與 validator 位於 `scripts/tools/research_slot_governance_*`。 |
| **[research_lock_v1.json](research_lock_v1.json)** | **online 與 research 互斥的當前狀態（machine-readable）。** 規則在 §20.10、執行在 `tests/contract/test_research_lock.py`，本檔只承載 state：`ONLINE_OPEN` / `RESEARCH_OPEN` / `RESEARCH_CLOSED`、開啟中實例所凍結的座標軸、以及 append-only 的轉移紀錄。轉移只由 `scripts/tools/research_lock.py` 執行；**缺檔 = 被刪除的 guard，不是 `ONLINE_OPEN`**。 |
| **[h2_controlled_host_execution_domain_v1.json](h2_controlled_host_execution_domain_v1.json)** | **H2 canonical corpus 的 admission anchor。** 唯一被 canonical measurement corpus 接納的 authorization execution domain（controlled host／operator／ledger namespace）。規則在 `check_h2_measure_archives.execution_domain_admission_reasons`，比對的是 archive 位元組與本檔位元組、**不觀察驗證主機**。它是 provenance/admission guard,**不是** authority proof：能重寫 grant／receipt／digest chain 的人也能寫入本檔內容,不可偽造的簽發需要簽章機制,本 repo 不提供。 |
| **[H2 execution-integrity artifact suite](h2_phase_a_run_spec_v1.json)** | **H2 successor execution 的 machine-readable artifact contract。** [`h2_phase_a_authoring_profile_v1`](h2_phase_a_authoring_profile_v1.json) 是完整 454-key frozen authoring authority（非 runtime preset）；`h2_phase_a_run_spec_v1` 固定其發行出的 resolved namespace 與 declared content projection；[`h2_runtime_binding_v1`](h2_runtime_binding_v1.json) 綁實際 code/input/build bytes；[`h2_execution_result_v1`](h2_execution_result_v1.json) 區分 non-qualifying diagnostic 與 exactly-once measurement；[`h2_execution_verification_v1`](h2_execution_verification_v1.json) 只容許獨立 process 產生 foreign-host-independent verdict。 |
| **[boolean_composition_semantics_contract.md](boolean_composition_semantics_contract.md)** | **組合語義。** Ω/Θ 分型、三值 predicate、universe identity、threshold edge、role closure、canonical grammar、closed-loop firewall |
| **[safe_region_asset_contract.md](safe_region_asset_contract.md)** | **打包契約（R0-B RegionAsset）。** 把已封存的 evidence 決定性地打包成 region asset；claim level 與成熟度；**transfer / intervention / production 皆尚未授權** |

執行層 procedure（study-specific）：[../eval/procedures/](../eval/procedures/)。

## H2 execution-integrity authority

四份 H2 schema 固定兩條不可由 producer 改寫的規則：

1. **Configuration authority。** [`h2_phase_a_authoring_profile_v1.json`](h2_phase_a_authoring_profile_v1.json)
   是完整 454-key、owner-adjudicated、byte-frozen 的 authoring profile，
   不是 runtime preset；其 [schema](h2_phase_a_authoring_profile_v1.schema.json)
   與 [owner decision](h2_phase_a_run_spec_authoring_decision_v1.json) 必須先
   驗證通過，resolver 才能機械發行 `run_spec.json`。RunSpec 內 `preset`
   固定為 `null`，一般 preset 與 parser defaults 只存在於 profile 的一次性
   authoring lineage，runtime 不重讀。`run_spec.json` 必須包含 evaluator
   parser 暴露的完整 canonical resolved namespace；runtime parser defaults、
   environment 與 argv 都只是由 RunSpec 導出的 transport，parse 後每個 key
   都必須與 RunSpec 相等，任何一處不得另行 originate value。RunSpec object
   digest 的 canonical JSON bytes 不含 trailing LF；`run_spec.json` artifact
   serialization 則在相同 bytes 後恰加一個 LF，兩個 byte domain 以獨立必填
   identifier 固定。
2. **Projection authority。** `execution_semantics_projection` 是 declared
   content set 的 byte digest equality，成員固定包含 executed surfaces、
   capture ABI、`mot17_args.py`、frozen authoring profile、profile schema、
   owner decision、RunSpec schema 與 resolver。它不讀
   `h2_path_partition` 的分類結果；分類器只保護 ruler edit，不能反過來決定
   哪些 bytes 具有 execution semantics。

Producer 只寫 `run_spec.json`、`runtime_binding.json`、`result.json`。
`verification.json` 必須由獨立 command/process 僅從 archive bytes 與 checksum
closure 寫出；缺欄位不得透過重跑 producer 補齊。獨立 verifier 寫完
`verification.json` 後才原子提交涵蓋四份 JSON 的最終
`checksums.sha256`；`verification.json` 不反向包含該 checksum-file digest，
避免自我雜湊循環。

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

signal_table_schema §0.5 ──► score_ranking_evidence_contract v1
（gate / score 分層）         （L2 rank / margin / top-1 / calibration / SR0–SR6；
                               owner-accepted 2026-07-23）
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
| **score-ranking 規範契約** | **已解決（2026-07-23）。** [`score_ranking_evidence_contract_v1`](score_ranking_evidence_contract.md) 已 owner-accepted 並綁入 registry；L2 transition semantics 現為 defined。這不解除各 object 自己的 substrate、relevance、declaration、seal 或 scheduling blockers。 |
