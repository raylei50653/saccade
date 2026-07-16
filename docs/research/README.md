# Experiments

跨模組實驗、決策語義、全局 eval / training 與可引用 evidence。  
**結構契約：** [../ownership/doc_structure_contract.md](../ownership/doc_structure_contract.md)  
一次性數據報告結案後歸 [../archive/](../archive/)。

---

<a id="research-control-plane"></a>

## 🧭 Research control plane

這裡是研究任務的總入口。本圖只擁有**路由拓撲、型別與轉移邊**；
每個可點節點都連到狀態或事實 owner，不在入口複寫 terminal、數字或 live state。
宣告邊界規則的唯一 owner 是 [experiment contract §20.2 / §20.8](contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)：
每個可判定任務宣告 typed κ（§20.2），宣告可封印性由 §20.8 seal bar 判定。

### Typed route — runtime representation → reduction → validation

下圖按**量化空間**分層；各節點完整的 κ（comparison relation 與 decision rule）
以 linked owner document 為準，此處不複寫判準或數字。

- **Captured runtime event space** — partition 依 owner 定義:`matched ˙∪ cohort_gap ˙∪ unemitted`
  - κ quantification space = captured events:
    [R1 temporal-reduction capture — owner terminal results](../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md)
    — captured runtime ↔ canonical replay
  - **Joined `matched` pair space**(event-level fidelity 帳目)
    - κ quantification space = matched joined pairs；relation / rule = owner-defined fidelity boxes:
      [D0 — runtime shadow bridge fidelity: results](../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md)
      — offline proxy ↔ runtime quantity
    - `ρ_v: event → trial unit`(typing、conservation 與 cross-space 義務:[experiment contract §20.9](contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md))↓
    - **Trial-unit claim space**(track-level independence unit)
      - κ quantification space = trial units:
        [S0 — safe-domain axis transfer to runtime coordinates: results](../modules/semantic/research/closed/safe_domain_runtime_transfer_results_20260713.md)
        — runtime-coordinate transfer / claim support
  - **Frozen-packet bookkeeping**(κ quantification space = frozen unjoined runtime events)
    - [EK0 — frozen-packet exact-key recoverability — results](../modules/semantic/research/frozen_packet_exact_key_recoverability_results_20260713.md)
      — consistency audit
  - **Runtime decision-path observability**(κ quantification space = frozen evidence under examination — capture 與 provenance records)
    - [P0 — runtime bridge decision-path identifiability: results](../modules/semantic/research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md)
      → [H0 — headline-m full bridge-decision trace capture](../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)（native capture 宣告）
      → downstream claim 必須另行宣告。

### State resolution

| 圖上要讀的東西 | 唯一 owner |
|---|---|
| research-object accepted state / substrate / limits / admissible units | [claim-state registry](contracts/claim_state_registry.md) |
| closed diagnostic（S0 / EK0 / P0）的 terminal | 圖上 linked results note；registry 只承接其誘發的 object limits，不鏡射 diagnostic terminal |
| sole-active selection / module transition pointer | 對應 [module TODO](../modules/)；本線見 [semantic TODO](../modules/semantic/TODO.md) |
| charter lifecycle / sequence / expected-state lease / handoff | [research threads](threads/README.md) 與 linked declaration |
| verdict / statistics / reproducible evidence | 圖上 linked fact owner；可引用數字見 [evidence ledger](evidence_ledger.md) |

關閉一個節點不會自動授權下一個節點。任務只能由 owner-accepted terminal
觸發新的 admissible transition，再由 TODO 的 WIP 鎖選定是否執行。

治理（O-series，非實驗正文）：[../ownership/README.md](../ownership/README.md) · [DOC_MAINTENANCE § WIP](../DOC_MAINTENANCE.md) · [DEVELOPMENT.md](../../DEVELOPMENT.md)

---

## 🔒 Closed (read-only)

| 文件 | 內容 |
|------|------|
| [tracker-decision/status_2026-07-09.md](tracker-decision/status_2026-07-09.md) | P0–P8 closed — production / contract / dual-stab / NO-GO；**勿 drive-by reopen** |
| [tracker-decision/README.md](tracker-decision/README.md) | 決策層範圍、文件索引、與 pipeline 分工 |

---

## Paper & evidence

兩條敘事線**互指、不互相覆寫**（契約 C5）：

| 線 | 入口 | 用途 |
|------|------|------|
| Decision / production | [paper_outline.md](paper_outline.md) · [evidence_ledger.md](evidence_ledger.md) | Geometry-first + whole-graph；可引用 metrics / 決策列 |
| Mamba method assets | [../../report_data/README.md](../../report_data/README.md) | Curriculum / detector thesis、可重建 tables/figures |

負結果總表：[../reference/no_go_registry.md](../reference/no_go_registry.md)

---

## Subdirectories（全局 research）

子目錄 **完整檔案表** 以各子目錄 README 為準（契約 C4）；此處只列入口。

| 目錄 | 入口 | 內容 |
|------|------|------|
| [threads/](threads/README.md) | [threads/README.md](threads/README.md) | **連續任務母線**（navigation-only；不放長表 / 不取代 ledger） |
| [pipeline/](pipeline/) | 見下表（本目錄無獨立 README） | Runtime 路徑、perf、sync、CPU |
| **[contracts/](contracts/README.md)** | **[contracts/README.md](contracts/README.md)** | **跨研究規範層（先讀，勿自造統計）**：feasible-set 數學框架（ε／independence unit／claim ladder L0–L6／forbidden shortcuts）· runtime-quantity fidelity protocol · gate-vs-score 分層 · Boolean 組合語義 · RegionAsset 打包契約 |
| [eval/](eval/README.md) | [eval/README.md](eval/README.md) · **[signal_analysis_ledger](eval/signal_analysis_ledger.md)** | Eval / ablation 筆記；**深度訊號總帳**（規範層見 contracts/） |
| [training/](training/README.md) | [training/README.md](training/README.md) | 訓練實驗 |
| [reid/](reid/) | 見下表 | 外觀能力上限等跨模組 reid 筆記 |
| [tracker-decision/](tracker-decision/README.md) | [tracker-decision/README.md](tracker-decision/README.md) | 決策語義（closed 線為主） |

### pipeline/（全表）

| 文件 | 內容 |
|------|------|
| [mot17_mamba_whole_graph_m_sdp_double_buffer.md](pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md) | `mamba_whole_graph_m` + SDP + double-buffer 路徑 runbook |
| [gpu_pipeline_m4b_identity_resolver.md](pipeline/gpu_pipeline_m4b_identity_resolver.md) | M4b native identity resolver 設計 |
| [perf_attribution_whole_graph_m.md](pipeline/perf_attribution_whole_graph_m.md) | whole_graph_m 每幀開銷歸因 + backlog |
| [CPU_BOUND_ANALYSIS.md](pipeline/CPU_BOUND_ANALYSIS.md) | CPU-bound 分析 |
| [cpu_overhead_analysis_20260707.md](pipeline/cpu_overhead_analysis_20260707.md) | CPU overhead（2026-07） |
| [optimization_redundant_computations_20260620.md](pipeline/optimization_redundant_computations_20260620.md) | 冗餘計算優化筆記 |
| [sync_audit_20260706.md](pipeline/sync_audit_20260706.md) | Sync audit |

### reid/（本樹）

| 文件 | 內容 |
|------|------|
| [appearance_ceiling_mot17.md](reid/appearance_ceiling_mot17.md) | MOT17 appearance 能力上限 |

模組側 reid / trigger research 見下方 Related module research。

---

## Related module research（pointers only）

| 模組 | 入口 |
|------|------|
| [semantic](../modules/semantic/README.md) | Semantic relink / runtime bridge / safe-domain research index |
| [detection](../modules/detection/README.md) | T3→T1、CUDA graph、kernel fusion 等 |
| [geometry](../modules/geometry/README.md) | [fp_fn_recovery_and_gmc](../modules/geometry/research/fp_fn_recovery_and_gmc.md) |
| [lifecycle](../modules/lifecycle/README.md) | [tentative_confirmed_state](../modules/lifecycle/research/tentative_confirmed_state.md) |
| [reid](../modules/reid/README.md) | LaSt-ViT、semantic_relink_and_crop |
| [trigger](../modules/trigger/README.md) | [dynamic_trigger](../modules/trigger/research/dynamic_trigger.md) |

---

## 規範（摘要）

- 新增檔 → 同一 PR 更新 owning README 索引（[契約 C4](../ownership/doc_structure_contract.md)）
- 跨子類連續任務 → [threads/](threads/README.md) navigation-only 母線（不放長表）
- 可引用數字 → [evidence_ledger](evidence_ledger.md) 或 [report_data](../../report_data/README.md)
- 結案 one-shot → [../archive/](../archive/)
- Closed tracker-decision：只讀，勿並開
