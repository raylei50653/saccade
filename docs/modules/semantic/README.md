# Semantic Relink Module (語意關聯與重排)

## 📐 模組職責
負責基於外觀相似度的匈牙利算法關聯匹配、重排 (Rerank) 以過濾 False-Accept，以及跨鏡頭/長失聯身份關聯。

## 🟢 目前現況
* 實現 Sinkhorn Auction 混合關聯機制與 Rerank Phase 3 重排，有效解決相似衣著行人的 ID 混淆（default=0.91）。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `[13] bg_relink_wait` + `[14] relink_write`（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | track candidates + embeddings（來自 [reid](../reid/README.md)）+ motion snapshots |
| **輸出** | resolved stable identities（local track id → 穩定 identity output） |
| **上游 → 下游** | `[12] materialize → IdentityResolver.resolve_pass (semantic relink + lifecycle merge) → emit`；GMC ON 下 semantic relink 基本冗余 |

> 職責分界：用 [reid](../reid/README.md) 的 embedding 做匈牙利關聯 / rerank / false-accept 過濾，是本模組；特徵抽取本身在 reid。

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| — | Sinkhorn Auction 混合關聯 + Rerank Phase 3 | ✅ GO，default `semantic_threshold=0.91` |
| 2026-04-27 | Relink threshold 調優 | ✅ thr=0.90 Pareto 最優 |
| 2026-05-03 | Post-merge（A5 soft appearance cost + gap uncertainty） | ⚠️ 有害→中性偏正；default off |
| 2026-06-03 | Cheb-GR re-ranking — standalone 方法 gate（Market-1501 / SigLIP2） | ✅ 方法成立：GPU Cheb-GR k-reciprocal +9.56pp（純自適應 λ=4 +8.76pp），但**不優於**經典 fixed-k（+10.03pp）；feature-propagation / Jaccard-w/o-QE 變體負向 |
| 2026-06-03 | Cheb-GR 路徑2 — offline tracklet merge（MOT17 mamba_whole_graph） | ❌ NO-GO：safe 操作點（max_cost 0.20–0.25）IDs 536→527 但 **AssA/IDF1/HOTA 全 0.0pp**；放寬即過度合併傷 IDF1。強偵測+GMC 下無 appearance headroom；code 保留 default off |
| 2026-06-03 | Birth-time lost-bank relink（C++ GPU，含 Cheb-GR 自適應門檻 + 速度搜捕圈） | ❌ NO-GO：無 λ 能讓復活降 IDs（高→0、中→白做、低→誤接）；根因＝長 gap embedding rank-1 僅 13–33%，接到 look-alike。基建保留 default off（`--relink-enabled`）。共因見 [appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md) |
| 2026-06-04 | Kalman 物理重連門控（正向卡方 + cosθ 方向 + 各向異性雪茄雲 + 速度上限） | 🚧 Phase 0 實作完成（default off）；純幾何路線，修好 custom_seq ID4 假併吞；待 MOT17 ablation。見 [bidirectional_relink_roadmap](research/bidirectional_relink_roadmap.md) |

## 📚 研究 / 設計

| 文件 | 內容 |
|------|------|
| [research/bidirectional_relink_roadmap.md](research/bidirectional_relink_roadmap.md) | 雙向時空收斂幾何重連長期路線圖（Phase 0 已落地，1–4 規劃） |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
