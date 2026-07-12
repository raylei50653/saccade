# Experiments

跨模組實驗、決策語義、全局 eval / training 與可引用 evidence。  
**結構契約：** [../ownership/doc_structure_contract.md](../ownership/doc_structure_contract.md)  
一次性數據報告結案後歸 [../archive/](../archive/)。

---

## 🔄 現在在做什麼

**這裡不列狀態**（列了就會漂移——實測如此）。三個入口，各有唯一 owner：

| 我要知道 | 去 |
|------|------|
| 每個研究對象**現在站在哪一格**（state / substrate / limits / 合法候選集） | **[claim_state_registry](contracts/claim_state_registry.md)** |
| 哪一個被選為 **sole active**（WIP 鎖） | 對應的 [module TODO](../modules/) |
| 這條線**怎麼接續**（敘事導覽） | [threads/](threads/README.md) |

治理（O-series，非實驗正文）：[../ownership/README.md](../ownership/README.md) · [DOC_MAINTENANCE § WIP](../DOC_MAINTENANCE.md) · [DEVELOPMENT.md 模組現狀總覽](../../DEVELOPMENT.md)

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
| [semantic](../modules/semantic/README.md) | 12 notes — offline relink hub、Cheb-GR bank、**occ-exit active** |
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
