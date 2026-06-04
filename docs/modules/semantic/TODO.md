# Semantic Relink — 模組 TODO

> 全局進度矩陣與 Baseline 見 [docs/TODO.md](../../TODO.md)。

🟢 目前無 active 待辦（appearance 關聯全面結案，見下）。完整脈絡：[appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md)。

## ✅ 已結案（2026-06-03，全 NO-GO，code 保留 default off）

- [x] **Cheb-GR 路徑2 — offline tracklet merge** → ❌ NO-GO（AssA 0.0pp）。
  實作：`src/saccade/perception/eval/cheb_gr_merge.py`（時間分散取樣 N=20~100 + cheb_gr_kreciprocal 樣本級圖 + robust-min pool + greedy/frame-set 不相交 UnionFind）+ adapter `extract_tracklet_embeddings` + evaluator `run_eval` 掛載 + config `cheb_gr_merge_*`；7 unit tests。啟用 `--cheb-gr-merge-enabled`。
- [x] **Birth-time lost-bank relink（C++ GPU）** → ❌ NO-GO（無 λ 能降 IDs）。
  實作：`tracker_gpu.cu` GPU ring-buffer lost bank + Cheb-GR 兩階段 relink kernel（自適應門檻 + 速度搜捕圈）+ spawn 復活；`set_relink_params` + `--relink-*` flags。啟用 `--relink-enabled`。
- 診斷工具保留：`scripts/eval/reid_id_benchmark.py`（embedding 區分力 gate）、`scripts/eval/reconnect_rate.py`（重連率）、`scripts/train/reid_domain_probe.py`（MOT-域 head probe）。

> 註：Cheb-GR / relink core 程式碼物理上放在 `perception/reid/` 與 `tracking/`，但功能（re-ranking / 身份關聯）歸此模組；[reid](../reid/README.md) 維持暫緩。重啟條件＝MOT-域 ReID 特徵（見 [reid TODO](../reid/TODO.md)）。
