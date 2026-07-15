# Semantic Relink Module (語意關聯與重排)

## 📐 模組職責
負責基於外觀相似度的匈牙利算法關聯匹配、重排 (Rerank) 以過濾 False-Accept，以及跨鏡頭/長失聯身份關聯。

## 🟢 目前現況
* 實現 Sinkhorn Auction 混合關聯機制與 Rerank Phase 3 重排；appearance / ReID relink 能力保留，但現行 `mamba_whole_graph` headline baseline 使用 `reid_mode: "off"`，主要 identity 修復來自 GPU 雙向橋接 relink 與 tracker core。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `bg_relink_wait` + `relink_write`（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | track candidates + embeddings（來自 [reid](../reid/README.md)）+ motion snapshots |
| **輸出** | resolved stable identities（local track id → 穩定 identity output） |
| **上游 → 下游** | `materialize → IdentityResolver.resolve_pass (semantic relink + lifecycle merge) → emit`；GMC ON 下 semantic relink 基本冗余 |

> 職責分界：用 [reid](../reid/README.md) 的 embedding 做匈牙利關聯 / rerank / false-accept 過濾，是本模組；特徵抽取本身在 reid。

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| — | Sinkhorn Auction 混合關聯 + Rerank Phase 3 | ✅ module-level GO；appearance path 非 current headline preset（`mamba_whole_graph` ReID off） |
| 2026-04-27 | Relink threshold 調優 | ✅ thr=0.90 Pareto 最優 |
| 2026-05-03 | Post-merge（A5 soft appearance cost + gap uncertainty） | ⚠️ 有害→中性偏正；default off |
| 2026-06-03 | Cheb-GR re-ranking — standalone 方法 gate（Market-1501 / SigLIP2） | ✅ 方法成立：GPU Cheb-GR k-reciprocal +9.56pp（純自適應 λ=4 +8.76pp），但**不優於**經典 fixed-k（+10.03pp）；feature-propagation / Jaccard-w/o-QE 變體負向 |
| 2026-06-03 | Cheb-GR 路徑2 — offline tracklet merge（MOT17 mamba_whole_graph） | ❌ NO-GO：safe 操作點（max_cost 0.20–0.25）IDs 536→527 但 **AssA/IDF1/HOTA 全 0.0pp**；放寬即過度合併傷 IDF1。強偵測+GMC 下無 appearance headroom；code 保留 default off |
| 2026-06-03 | Birth-time lost-bank relink（C++ GPU，含 Cheb-GR 自適應門檻 + 速度搜捕圈） | ❌ NO-GO：無 λ 能讓復活降 IDs（高→0、中→白做、低→誤接）；根因＝長 gap embedding rank-1 僅 13–33%，接到 look-alike。基建保留 default off（`--relink-enabled`）。共因見 [appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md) |
| 2026-06-04 | Kalman 物理重連門控（正向卡方 + cosθ 方向 + 各向異性雪茄雲 + 速度上限） | 🚧 Phase 0 實作完成（default off）；純幾何路線，修好 custom_seq ID4 假併吞；待 MOT17 ablation。見 [bidirectional_relink_roadmap](research/bidirectional_relink_roadmap.md) |
| 2026-06-11 | GPU 雙向中點橋接 relink（px=0.25 + scale gate） | ✅ GO，**preset 預設開**；MOT17-SDP on/off IDF1 +2.1 / AssA +2.8 / IDs −13.6% / FP −14% 全指標嚴格優勢 |
| 2026-06-11 | Relink bridge scale gate（speed 方向擴展） | ❌ NO-GO（SDP 小幅正向但速度方向全線死；P0 L_med 復核不重現），registry [#31](../../reference/no_go_registry.md) |
| 2026-06-11 | occ_cover live relink（gap-path 占用門） | ❌ NO-GO（live accepts 全 gap≤1；長 gap 被 track_buffer=30 結構性消滅；tb90 解鎖反 −0.8），registry [#33](../../reference/no_go_registry.md) |
| 2026-07-03 | Cheb-GR causal online handover — **live streaming claims**（每幀 Cheb-GR k-reciprocal newborn→dead 手遞，不重標已輸出幀） | ❌ NO-GO（**全網格單調有害，7/7 序列負**；min_head=2 揭露 budget 下 head 恆=1；mainline 每幀抽取 h≥2/h≥3 也 −2.4；live 回饋迴路複利懲罰 ~4.3）。C++ streaming port 本身已驗證正確（含重大 fix：Eigen::Map column-major→row-major，kernel vs torch 6e-8），code 保留 default-off + `SACCADE_HO_DEBUG_LEVEL` gating。**可用路徑 = offline Cheb-GR merge (已 GO) + delayed-claim 臂重掃（kernel 修復紅利）**。registry [#56](../../reference/no_go_registry.md#56) |
| 2026-07-04 | Sync online ReID in tracker critical path（mnv4 dynamic / bridge appearance veto） | ❌ NO-GO（cost-benefit 不成立）。無 ReID `mamba_whole_graph_m` 79.5 / IDs 335 / ~241 FPS；修掉 live handover 污染後同步 `reid_mnv4_dynamic` 79.7 / IDs 333 / ~193 FPS，吞吐約 −20% 只換 +0.2 IDF1。`relink_bridge_app_veto=0.20` 無實際 veto；`veto=1.0` 僅 33 veto 且 IDs 341。**主線：geometry + conservative relink；ReID 只能 async sidecar / offline，不得阻塞 double-buffer critical path。** registry [#57](../../reference/no_go_registry.md#57) |
| 2026-07-04 | Cheb-GR offline handover（output-layer，mnv4 h2 margin0.05） | ✅ GO as post-hoc cleanup；`--cheb-gr-offline-handover --cheb-gr-model mobilenetv4_reid --cheb-gr-offline-min-head 2 --cheb-gr-offline-margin 0.05`：IDF1 80.3 / IDs 311 / ~219 FPS；先輸出 MOT 再 crop/extract/relabel，不進 tracker critical path |

### Cheb-GR Door D owner status (2026-07-12)

| Field | Value |
|---|---|
| `research_status` | **closed** |
| `support_status` | **cold-supported**（非 retire） |
| `production_status` | default-off, non-headline |
| `maintenance_scope` | reproducibility + diagnostic CLI/schema |
| `active_optimization` | **no** |

- Offline post-hoc GO 仍成立（上表）；live/online claims 仍 NO-GO（#56 / #57）。
- Tool registry Door D = **P2**（結案重跑／旁線診斷），不再是 P0 現用開工鏈。
- 詳見 fact-owner [research/chebgr_handover_signal_map_20260704.md](research/chebgr_handover_signal_map_20260704.md) · registry [research/association_tools.yaml](research/association_tools.yaml)。

## 📚 研究 / 設計

> **這是目錄，不是狀態表。** 每列只說「這份文件是什麼」。
> **裁決 / terminal / 數字不寫在這裡**——狀態的家是
> [claim_state_registry](../../research/contracts/claim_state_registry.md)，
> 數字的家是 [evidence_ledger](../../research/evidence_ledger.md) 與各 note 的 evidence packet
> （[契約 C5](../../ownership/doc_structure_contract.md)：不得有第二真相）。
> WIP 鎖見 [TODO.md](TODO.md)。

### 安全域 / gate

| 文件 | 這是什麼 |
|------|------|
| [research/frozen_packet_exact_key_recoverability_results_20260713.md](research/frozen_packet_exact_key_recoverability_results_20260713.md) | EK0：frozen packet exact-key 可恢復性 consistency audit（結果） |
| [research/frozen_packet_exact_key_recoverability_declaration_20260713.md](research/frozen_packet_exact_key_recoverability_declaration_20260713.md) | EK0 預宣告（sealed，rev.3 pure consistency audit） |
| [research/closed/safe_domain_runtime_transfer_results_20260713.md](research/closed/safe_domain_runtime_transfer_results_20260713.md) | S0：offline→runtime safe-axis transfer audit（closed results） |
| [research/safe_domain_runtime_transfer_declaration_20260712.md](research/safe_domain_runtime_transfer_declaration_20260712.md) | S0 declaration：已接受的 safe 軸能否從 offline 座標轉移到 runtime 座標 |
| [research/boolean_atom_partial_order_20260711.md](research/boolean_atom_partial_order_20260711.md) | Boolean atom 偏序稽核 — 全域可排序軸的認定 |
| [research/boolean_closure_domain_line_20260711.md](research/boolean_closure_domain_line_20260711.md) | 布林閉包域研究線的 normative doc |
| [research/gt_support_morphology_step0_20260711.md](research/gt_support_morphology_step0_20260711.md) | GT-support morphology Step-0：placement 分佈與 escape tail |
| [research/escape_tail_forensic_20260711.md](research/escape_tail_forensic_20260711.md) | escape-tail 四軌 forensic |
| [research/safe_region_a1_audit_20260711.md](research/safe_region_a1_audit_20260711.md) | Safe-region A1 acceptance 稽核 |
| [research/safe_region_asset_r1_conversion_20260710.md](research/safe_region_asset_r1_conversion_20260710.md) | RegionAsset R1 轉換（sealed evidence → pack） |
| [research/safe_region_r0_asset_contract_preflight_20260710.md](research/safe_region_r0_asset_contract_preflight_20260710.md) | RegionAsset 契約 preflight |
| [research/safe_region_assetization_r1_20260710.md](research/safe_region_assetization_r1_20260710.md) | Assetization R1 診斷 overlay（非 A1 物件） |
| [research/safe_region_assetization_r11_20260710.md](research/safe_region_assetization_r11_20260710.md) | Assetization R1.1 診斷 overlay |
| [research/composition_grammar_safe_region_coverage_audit_20260710.md](research/composition_grammar_safe_region_coverage_audit_20260710.md) | 組合文法 × safe-region 覆蓋稽核 |
| [research/composition_grammar_t0_region_interpretation_20260710.md](research/composition_grammar_t0_region_interpretation_20260710.md) | 組合文法 T0 區域解讀 |
| [research/composition_grammar_t0_artifact_preflight_20260710.md](research/composition_grammar_t0_artifact_preflight_20260710.md) | 組合文法 T0 artifact preflight |
| [research/m_gate_h_ratio_signal_7seq_20260709.md](research/m_gate_h_ratio_signal_7seq_20260709.md) | h-ratio gate 訊號的七序列分析 |

### runtime 量的忠實性（safe domain 的前提）

| 文件 | 這是什麼 |
|------|------|
| [research/bridge_fidelity_reconciled_map_20260715.md](research/bridge_fidelity_reconciled_map_20260715.md) | 🗺️ **總覽地圖**（draft, navigation-only）：D0/R1/S0/EK0/P0/T2 現況以 ADR 020 typed-terminal schema 重建；校正舊圖（D0=FALSIFIED 非 bit-exact）|
| [research/headline_bridge_full_decision_capture_declaration_20260713.md](research/headline_bridge_full_decision_capture_declaration_20260713.md) | H0：headline-m full bridge-decision trace capture 宣告 |
| [research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md](research/closed/runtime_bridge_decision_path_identifiability_results_20260713.md) | P0：既有 frozen capture 的 decision-path 可識別性結果 |
| [research/runtime_bridge_decision_path_identifiability_declaration_20260713.md](research/runtime_bridge_decision_path_identifiability_declaration_20260713.md) | P0：decision-path identifiability 預宣告 |
| [research/d0_runtime_shadow_fidelity_results_20260712.md](research/d0_runtime_shadow_fidelity_results_20260712.md) | D0：以真實 runtime CUDA `bdist` 認證 offline proxy 的忠實性（結果） |
| [research/d0_runtime_shadow_fidelity_declaration_20260712.md](research/d0_runtime_shadow_fidelity_declaration_20260712.md) | D0 預宣告（sealed） |
| [research/s0_proxy_validity_amendment_20260712.md](research/s0_proxy_validity_amendment_20260712.md) | `s0` 自此為 offline-only 量的效力修訂（append-only） |
| [research/r1_temporal_reduction_capture_results_20260712.md](research/r1_temporal_reduction_capture_results_20260712.md) | R1：時域化約算子 \(R\) 的捕獲／重播認證（結果） |
| [research/r1_temporal_reduction_capture_declaration_20260712.md](research/r1_temporal_reduction_capture_declaration_20260712.md) | R1 預宣告（sealed） |
| [research/d0_bridge_estimator_fidelity_20260711.md](research/d0_bridge_estimator_fidelity_20260711.md) | ⚠️ legacy v1 reconstruction packet（已被 v2 取代，保持凍結） |
| [research/production_substrate_mapping_20260711.md](research/production_substrate_mapping_20260711.md) | production consumer / substrate 對照（binding precondition） |

### score 線（parked：保留域建立前不開）

| 文件 | 這是什麼 |
|------|------|
| [research/score_temporal_to_stable_domain_20260712.md](research/score_temporal_to_stable_domain_20260712.md) | score 時域→穩定域建模的 charter |
| [research/discrete_m_capability_declaration_20260712.md](research/discrete_m_capability_declaration_20260712.md) | discrete-\(M\) anchor propagation 宣告（parked、未 seal） |
| [research/door0_ranking_probe_results_20260712.md](research/door0_ranking_probe_results_20260712.md) | Door 0 ambiguous-band ranking-power probe（結果） |
| [research/ambiguous_band_ranking_power_probe_declaration_20260712.md](research/ambiguous_band_ranking_power_probe_declaration_20260712.md) | Door 0 預宣告（sealed） |

### gap-conditioned motion

| 文件 | 這是什麼 |
|------|------|
| [research/gap_conditioned_motion_e0_20260711.md](research/gap_conditioned_motion_e0_20260711.md) | E0 可識別性 |
| [research/gap_conditioned_motion_e1_m0_20260711.md](research/gap_conditioned_motion_e1_m0_20260711.md) | E1 M0 baseline |
| [research/gap_conditioned_motion_e2_family_20260711.md](research/gap_conditioned_motion_e2_family_20260711.md) | E2 position-only 模型族 |
| [research/gap_conditioned_motion_e3_signals_20260711.md](research/gap_conditioned_motion_e3_signals_20260711.md) | E3 LOO fold 訊號（sealed cube） |
| [research/gap_conditioned_motion_phase_b_design_20260711.md](research/gap_conditioned_motion_phase_b_design_20260711.md) | Phase B 預宣告設計（A1–A8） |
| [research/gap_conditioned_motion_phase_b_20260711.md](research/gap_conditioned_motion_phase_b_20260711.md) | Phase B 執行結果 |

### occ-exit / m_b1 hook（closed 線的參考）

| 文件 | 這是什麼 |
|------|------|
| [../../experiments/occ_exit_audit_p55/](../../experiments/occ_exit_audit_p55/README.md) | occ-exit 稽核工作區（WP1–WP3；prototype workspace） |
| [research/m_b1_research_history_20260709_20260710.md](research/m_b1_research_history_20260709_20260710.md) | 📜 M-B1 offline 階段的歷史彙整 |
| [research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md](research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | M-B1 候選規則 |
| [research/m_b1_portable_or_tail_hook_contract_20260709.md](research/m_b1_portable_or_tail_hook_contract_20260709.md) | M-B1 portable OR-tail hook 契約 |
| [research/m_b1_stage1_online_hook_final_20260710.md](research/m_b1_stage1_online_hook_final_20260710.md) | M-B1 Stage-1 online hook 收尾 |
| [research/m_b1_5_stage2_entry_contract_20260710.md](research/m_b1_5_stage2_entry_contract_20260710.md) | M-B1.5 Stage-2 進入契約 |
| [research/m_b1_5_stage2_d_online_final_20260710.md](research/m_b1_5_stage2_d_online_final_20260710.md) | M-B1.5 Stage-2D online 收尾 |
| [research/m_b1_to_m_b1_5_two_stage_plan_20260710.md](research/m_b1_to_m_b1_5_two_stage_plan_20260710.md) | M-B1 → M-B1.5 兩階段計畫 |
| [research/m_b1_doc_consolidation_report_20260710.md](research/m_b1_doc_consolidation_report_20260710.md) | M-B1 文檔彙整報告 |
| [research/m_b2_reconnect_bridge_ab_20260709.md](research/m_b2_reconnect_bridge_ab_20260709.md) | m B2 bridge on/off 重連 A/B |

### relink / bank / Cheb-GR（基礎與工具）

| 文件 | 這是什麼 |
|------|------|
| 🧭 [research/offline_relink_candidate_analysis.md](research/offline_relink_candidate_analysis.md) | relink 候選池的離線分析（bridge score 的來源設計） |
| [research/bidirectional_relink_roadmap.md](research/bidirectional_relink_roadmap.md) | 雙向時空收斂幾何重連路線圖 |
| [research/bidir_relink_data_analysis.md](research/bidir_relink_data_analysis.md) | 線上 bridge 候選 per-attempt 分析 |
| [research/relink_normalization_gate_analysis.md](research/relink_normalization_gate_analysis.md) | scale / normalization gate 分析 |
| [research/depth_ordering_crossing_swap.md](research/depth_ordering_crossing_swap.md) | 深度排序與交錯換位 |
| [research/sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md) | 稀疏 key-embedding bank |
| [research/clean_fifo_bank_substrate_20260704.md](research/clean_fifo_bank_substrate_20260704.md) | CleanFIFO bank substrate |
| [research/chebgr_handover_signal_map_20260704.md](research/chebgr_handover_signal_map_20260704.md) | Cheb-GR handover 訊號圖（Door D fact-owner） |
| [research/online_sparse_reid_handoff_20260704.md](research/online_sparse_reid_handoff_20260704.md) | 線上 sparse ReID handoff |
| 🗺️ [research/association_recovery_crosswalk_20260709.md](research/association_recovery_crosswalk_20260709.md) | 關聯回復研究的 crosswalk |
| 📇 [research/association_recovery_scripts_index_20260709.md](research/association_recovery_scripts_index_20260709.md) | 關聯回復腳本索引 |
| 📜 [research/association_recovery_info_source_contract_20260709.md](research/association_recovery_info_source_contract_20260709.md) | 關聯回復資訊來源契約 |
| 📋 [research/association_tools.yaml](research/association_tools.yaml) | 工具 registry |
| 📇 深度訊號總帳 | [signal_analysis_ledger](../../research/eval/signal_analysis_ledger.md) |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
