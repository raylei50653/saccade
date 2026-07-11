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

## 📚 研究 / 設計

> 全表義務見 [Doc Structure Contract C4](../../ownership/doc_structure_contract.md)。Active 對齊 [TODO.md](TODO.md) sole active（WIP=1）。

### 🔄 Active

| 文件 | 內容 |
|------|------|
| [research/d0_bridge_estimator_fidelity_20260711.md](research/d0_bridge_estimator_fidelity_20260711.md) | **D0 bridge fidelity `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE`** · terminal **`not_fidelity_aligned`** · **primary=`runtime_capture_unavailable`** · Issue #112 **incomplete** · reconstruction diagnostics only (not runtime CA capture) · \(S_A\) coverage FAIL · GT boundary distorted · single-factor S0–S6 decomp · headline preset file hash · Phase-B V5 remains representation-level · production unchanged · packet [evidence/](research/evidence/d0_bridge_estimator_fidelity_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) · [Issue #112](https://github.com/raylei50653/saccade/issues/112) |
| [research/gap_conditioned_motion_phase_b_20260711.md](research/gap_conditioned_motion_phase_b_20260711.md) | **Gap-conditioned motion Phase B `V5 ACCEPTED_WITH_LIMITS`** · sealed A1–A8 execution · no five-box representation/attribution contract · claim ceiling = representation / level 1 · no tracker/preset/hook change · packet [evidence/](research/evidence/gap_conditioned_motion_phase_b_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/production_substrate_mapping_20260711.md](research/production_substrate_mapping_20260711.md) | **Production substrate mapping（Step-0 abstraction-chain audit · binding precondition for E3/A1–A8）** · terminal **`CONSUMER_SPLIT`**：consumer A＝active tracker-core bridge（連續 speed-weighted `bdist` 聚合＋距離排序＋margin）；B＝optional semantic relinker（Boolean gate chain）；C1＝semantic live claim（繼承 B pre-gates）／C2＝evfifo live-bank output handover（僅 gap window，無 motion pre-gates），皆 pooled-low-mean `c_app` · **A governs deployment claims** · atoms 對 A 為 production-native counterparts（同名／同式／同 ABI，estimator-shifted）；kernel 內 default-off portable OR-tail hook＝level-3 plumbing available，acceptance pending（`ONLINE_BAUDIT_IMPLEMENTED=False`） · lifecycle-derived support `S_A=[1,26]`／`S_C2=[1,60]`／`S_B=[2,45]`（row-level 切面） · deployment claim 須在 named consumer support 單獨成立 · claim ladder 四級 · D0 bridge-score estimator fidelity gate（三值 verdict；**executed → `not_fidelity_aligned`**） · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/gap_conditioned_motion_phase_b_design_20260711.md](research/gap_conditioned_motion_phase_b_design_20260711.md) | **Gap-conditioned motion Phase B predeclared design** · frozen A1–A8 numeric criteria / support / V1–V5 partition · seal consumed without deviation by the Phase-B `V5` run · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/gap_conditioned_motion_e3_signals_20260711.md](research/gap_conditioned_motion_e3_signals_20260711.md) | **Gap-conditioned motion E3 LOO fold signals `E3_SIGNALS_SEALED`** · 7 folds · 28 parameter + 7 selection · full fold×pair×model cube 679,952 rows (`evaluation_role=held_out|train` · A6 train-side surface) · energy terms split · no winner-only filter · Phase B design seal recorded · **no A1–A8/V1–V5 inside E3** · downstream Phase B recorded `V5` · packet [evidence/](research/evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/gap_conditioned_motion_e2_family_20260711.md](research/gap_conditioned_motion_e2_family_20260711.md) | **Gap-conditioned motion E2 position-only family `ACCEPTED_WITH_LIMITS`** · `GCM-E2-POSITION-ONLY-v1` · `FROZEN_ACCEPTED_WITH_LIMITS` · global M1-P + three predeclared M2-P half-lives · finite/support + per-fold LOO lineage gates PASS · E3 sealed downstream · Phase-B `V5 ACCEPTED_WITH_LIMITS` (representation / level 1) · packet [evidence/](research/evidence/gap_conditioned_motion_e2_family_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/gap_conditioned_motion_e1_m0_20260711.md](research/gap_conditioned_motion_e1_m0_20260711.md) | **Gap-conditioned motion E1 M0 baseline `ACCEPTED_WITH_LIMITS`** · 0/20 aggregate reversal cells under frozen reporting criterion · bridge/residual AUC erodes with gap · later Phase-B `V5 ACCEPTED_WITH_LIMITS` (representation / level 1) · packet [evidence/](research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/gap_conditioned_motion_e0_20260711.md](research/gap_conditioned_motion_e0_20260711.md) | **Gap-conditioned probabilistic motion E0** · `PARTIALLY_IDENTIFIABLE` · M0 + position-only observation available · joint/velocity/context claims fail-closed · packet [evidence/](research/evidence/gap_conditioned_motion_e0_20260711/manifest.json) · [thread](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [research/boolean_atom_partial_order_20260711.md](research/boolean_atom_partial_order_20260711.md) | **PR-D gate partial-order audit（#106 / [PR #107](https://github.com/raylei50653/saccade/pull/107)）** · **`ACCEPTED_WITH_LIMITS`** · terminal **`GLOBAL_PARTIAL_ORDER_READY`** · global=`{dist_h, log_h_ratio}` · packet [evidence/](research/evidence/boolean_atom_partial_order_20260711/manifest.json) · [thread](../../research/threads/gt_support_morphology_20260711.md) |
| [research/escape_tail_forensic_20260711.md](research/escape_tail_forensic_20260711.md) | **PR-C four-track escape-tail forensic（#102 / [PR #104](https://github.com/raylei50653/saccade/pull/104)）** · 3×TRUE + 1×UNRESOLVED · aggregate `ROLE_REVERSAL_SUPPORTED` · **`ACCEPTED_WITH_LIMITS`** (L1 single-seq; partial-order audit only) · packet [evidence/](research/evidence/escape_tail_forensic_20260711/manifest.json) · [thread](../../research/threads/gt_support_morphology_20260711.md) |
| [research/boolean_closure_domain_line_20260711.md](research/boolean_closure_domain_line_20260711.md) | **布林閉包域研究線（normative doc, PR-B）** · placement morphology → partial-order audit → MWC → exact GT-UCB validation → optional compression · atom orderability 四分類 · Verdict A–E 完成條件 · terminal 權威 = framework §19 · [thread](../../research/threads/gt_support_morphology_20260711.md) |
| [research/gt_support_morphology_step0_20260711.md](research/gt_support_morphology_step0_20260711.md) | **GT-Support Morphology Step-0 audit** · per-cell risk field 不可識別（任何 k 僅 1 cell 達 ε≤0.05）· **verdict `UNRESOLVED`** + descriptive hypothesis：corner-concentrated（M₀=97.1%）+ far-Hamming motion tail 4/209（**4/4 在 MOT17-10**；log_h_ratio 0/4；nominal CP 4.33% 不得跨界）· packet [evidence/](research/evidence/gt_support_morphology_step0_20260711/manifest.json) · [thread](../../research/threads/gt_support_morphology_20260711.md) · [procedure §19](../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) |
| [research/safe_region_r0_asset_contract_preflight_20260710.md](research/safe_region_r0_asset_contract_preflight_20260710.md) | **R0-A Region Asset Contract Preflight** · CR1–CR9 **ACCEPTED** · [thread](../../research/threads/closed/safe_region_assetization_20260710.md) · [math](../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) |
| [research/safe_region_a1_audit_20260711.md](research/safe_region_a1_audit_20260711.md) | **A1 acceptance-unit lock + read-only audit · 26/26 PASS + mutation 5/5** · unit = conversion pack `1a180620bc…` · S0/S1/Q1/N1 · **terminal recorded: `A1_ACCEPTED_WITH_LIMITS` → maturity A1 (gate closed)** · [thread](../../research/threads/closed/safe_region_assetization_20260710.md) |
| [research/safe_region_assetization_r11_20260710.md](research/safe_region_assetization_r11_20260710.md) | **R1.1 attribution · DOWNGRADED to diagnostic overlay** · 2 unique harmful AND events + 3 descriptive symptoms · "primary F3" rejected (post-hoc floors / K-duplicated count / alias-ambiguous predicate) · study `out/signal_study/safe_region_assetization_r11_20260710/` · [thread](../../research/threads/closed/safe_region_assetization_20260710.md) |
| [research/safe_region_assetization_r1_20260710.md](research/safe_region_assetization_r1_20260710.md) | **R1 capacity probe · DOWNGRADED to diagnostic overlay** · V-C = heuristic-specific descriptive failure (LOO pool global-label-screened; class null retracted) · 15 unique masks / 34 grid-local assets · study `out/signal_study/safe_region_assetization_r1_20260710/` · R2 blocked · [thread](../../research/threads/closed/safe_region_assetization_20260710.md) |
| [research/safe_region_asset_r1_conversion_20260710.md](research/safe_region_asset_r1_conversion_20260710.md) | **R1 conversion note** · A0 pack candidate under `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` · eng. MERGED ([PR #95](https://github.com/raylei50653/saccade/pull/95) history) · [contract](../../research/eval/safe_region_asset_contract.md) · [thread](../../research/threads/closed/safe_region_assetization_20260710.md) · [math](../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) · [boolean](../../research/eval/boolean_composition_semantics_contract.md) |
| [research/occ_exit_audit_p55_scope_20260709.md](research/occ_exit_audit_p55_scope_20260709.md) | **#55 occ-exit audit** 範圍與 substrate |
| [research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md](research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md) | WP2 序列條件化標註 |
| [research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md](research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md) | WP3 promotion 決策（`split_feat_pr`；runtime 未開） |

### 收成 / 結案參考

| 文件 | 內容 |
|------|------|
| 🧭 **[research/offline_relink_candidate_analysis.md](research/offline_relink_candidate_analysis.md)** | **relink / AssA 調查 hub（s 歷史）** |
| 📜 **[research/m_b1_research_history_20260709_20260710.md](research/m_b1_research_history_20260709_20260710.md)** | **M-B1 offline history** (mine→region→LOO→freeze) · phase CLOSED |
| [research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md](research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | **Freeze identity**（LOO_pass_region_candidate · offline smoke pass） |
| [research/m_b1_portable_or_tail_hook_contract_20260709.md](research/m_b1_portable_or_tail_hook_contract_20260709.md) | Stage 1 hook **ABI contract**（default-off；preset NO） |
| [research/m_b1_stage1_online_hook_final_20260710.md](research/m_b1_stage1_online_hook_final_20260710.md) | Stage 1 **CLOSED** · wire + e2e · B-audit 244 · A0/det/runtime |
| [research/m_b1_5_stage2_entry_contract_20260710.md](research/m_b1_5_stage2_entry_contract_20260710.md) | Stage 2 entry · G0–G4 claim firewall |
| [research/m_b1_5_stage2_d_online_final_20260710.md](research/m_b1_5_stage2_d_online_final_20260710.md) | Stage 2 **final** · Q1–Q3 mass · Q4 grade C · Q4.5 v4 atlas B (154/0) |
| [research/composition_grammar_safe_region_coverage_audit_20260710.md](research/composition_grammar_safe_region_coverage_audit_20260710.md) | **G1–G7 × R1–R6 coverage audit**（recon closed）· Q4.5 terminal B unchanged · next T0 interpretation pack · [thread](../../research/threads/closed/composition_grammar_safe_region.md) |
| [research/composition_grammar_t0_artifact_preflight_20260710.md](research/composition_grammar_t0_artifact_preflight_20260710.md) | **T0-A preflight** · schema/key map · 7-output derivability · G7 N · T0-B surface proposed only · [thread](../../research/threads/closed/composition_grammar_safe_region.md) |
| [research/composition_grammar_t0_region_interpretation_20260710.md](research/composition_grammar_t0_region_interpretation_20260710.md) | **T0-B interpretation** · 154 PS geometry · radius≥1=0 · G7 contract-gap · [evidence](research/evidence/m_b1_5_t0_region_interpretation_20260710/) · [thread](../../research/threads/closed/composition_grammar_safe_region.md) |
| [research/m_b1_to_m_b1_5_two_stage_plan_20260710.md](research/m_b1_to_m_b1_5_two_stage_plan_20260710.md) | Stage 1+2 plan body（runner/contract ref） |
| [research/m_b1_doc_consolidation_report_20260710.md](research/m_b1_doc_consolidation_report_20260710.md) | Doc consolidation + information-preservation report |
| [research/m_b2_reconnect_bridge_ab_20260709.md](research/m_b2_reconnect_bridge_ab_20260709.md) | **m B2** production-like reconnect A/B（未來 e2e baseline） |
| 📇 depth index | [signal_analysis_ledger](../../research/eval/signal_analysis_ledger.md) · offline history § tools |
| 🗺️ **[research/association_recovery_crosswalk_20260709.md](research/association_recovery_crosswalk_20260709.md)** | **D1 對照圖（research-synthesis）**：實驗前導航；production stack 薄摘要 + door/knobs/NO-GO/substrate。**非** sole active、非第二 baseline |
| 📇 **[research/association_recovery_scripts_index_20260709.md](research/association_recovery_scripts_index_20260709.md)** | **腳本查找表**：task→script、door 分區、wrapper→canonical、R-A/D/F recipes。結論仍手動 |
| 📜 **[research/association_recovery_info_source_contract_20260709.md](research/association_recovery_info_source_contract_20260709.md)** | **資訊源契約**：disk / registry / no_go / preset / ledger 誰當 truth；腳本只 check·render·print |
| 📋 **[research/association_tools.yaml](research/association_tools.yaml)** | **R registry populated（Step 1B）**：~64 tools · R-A/B/D/E/F；禁 metrics/verdict/knobs。Checker：`scripts/tools/check_association_tools.py` |
| [research/depth_ordering_crossing_swap.md](research/depth_ordering_crossing_swap.md) | foot_y front/back AUC 0.898；same-height gate GO default-on |
| [research/bidirectional_relink_roadmap.md](research/bidirectional_relink_roadmap.md) | 雙向時空收斂幾何重連路線圖（Phase 0 已落地） |
| [research/bidir_relink_data_analysis.md](research/bidir_relink_data_analysis.md) | 線上 bridge 候選 per-attempt（hard-case AUC≈0.55） |
| [research/relink_normalization_gate_analysis.md](research/relink_normalization_gate_analysis.md) | Scale / normalization gate ablation |
| [research/sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md) | Cheb-GR 稀疏 bank 結案（#58）；clean-FIFO-20；Python 層已落地 |
| [research/clean_fifo_bank_substrate_20260704.md](research/clean_fifo_bank_substrate_20260704.md) | CleanFifoBank substrate contract（occ-audit / handover 共用） |
| [research/chebgr_handover_signal_map_20260704.md](research/chebgr_handover_signal_map_20260704.md) | Offline handover 訊號地圖；best_cost frontier |
| [research/online_sparse_reid_handoff_20260704.md](research/online_sparse_reid_handoff_20260704.md) | Sparse key-crop / async sidecar 接續摘要 |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
