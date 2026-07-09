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
| [research/occ_exit_audit_p55_scope_20260709.md](research/occ_exit_audit_p55_scope_20260709.md) | **#55 occ-exit audit** 範圍與 substrate |
| [research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md](research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md) | WP2 序列條件化標註 |
| [research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md](research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md) | WP3 promotion 決策（`split_feat_pr`；runtime 未開） |

### 收成 / 結案參考

| 文件 | 內容 |
|------|------|
| 🧭 **[research/offline_relink_candidate_analysis.md](research/offline_relink_candidate_analysis.md)** | **relink / crossing-swap / AssA 調查主入口（hub）**。bridge 全池 AUC 0.895、門作用區 ~0.65；面積率閘；reach-gate 速度項死重 |
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
