# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦與近期方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/home/ray/developer/ai/saccade/docs/TODO_history.md:1)。

---

## Current Focus

- [ ] **Rerank Phase 2：Bank Inject（C）+ Reciprocal Margin（D）— 收斂結論**
  - **Phase 1 結論（2026-04-29）**：`buffer_size × rerank_mode` 多樣本 scoring 無效——11 的 IDF1 系統性下降（-2.8～-3.6pp），02 的 FP/IDs 未改善。問題根源不在 appearance scoring，而在 reference 品質與 false accept 過濾。
  - **實作已完成（2026-04-29）**：
    - C++ `SemanticRelinker` 新增 `rerank_mode`、`reciprocal_margin`、`inject_reference()`、`alias`/`features` 屬性
    - `runner.py` 修正：`semantic_bank_inject` 與 `reciprocal_margin > 0` 不再強制 Python fallback（僅 non-mean rerank_mode 才用 Python）
    - `relink.py` 修正：Python `inject_reference()` 強制 `.cpu()` 防止 device mismatch
    - `ablation_rerank.py` 修正：base 加 `--no-semantic-bank-inject --semantic-buffer-size 1`；C/D/CD 均加 `--semantic-buffer-size 1`
  - **結果輸出**：`scripts/eval/output/ablation_rerank.txt` 與 `scripts/eval/output/current_margin_m*`
  - **Phase 2 結論**：
    - `buf=1 EMA` 基線上，`C+D: inject+margin=0.05` 最強，可作為 stripped reference path 的最佳組合。
    - 但在 current documented base 上，沒有任何 reciprocal margin 能乾淨取代 default。
    - `margin=0.15` 只適合保留為 `IDs/MOTA` 傾向的可選 variant，不應進 default。
  - **下一步**：停止 reciprocal margin default tuning，轉向 Phase 3 reference-quality / false-accept filtering。

## Active Context

- 目前 `siglip2` 仍是最高 IDF1 ceiling。
- `transreid` 與 `osnet` 已完成對比，但都未超過當前 `siglip2` base。
- OSNet engine 已建立，可用 `uv run python scripts/model/build_osnet.py` 重建。
- Phase 1 multi-sample rerank 結果已存 `results/ablation_rerank/`，可用 `--skip-run` 重新讀取。
- Phase 2 ablation script 已修正，直接 `--fast` 執行即可（舊 `base/` 結果仍有效）。

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖： [docs/TODO_history.md](/home/ray/developer/ai/saccade/docs/TODO_history.md:1)
- Tracking base 與 relink sweep： [docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/home/ray/developer/ai/saccade/docs/experiments/tracking/fp_fn_recovery_and_gmc.md:1)
- ReID backbone refresh 歸檔： [docs/experiments/reid/semantic_relink_and_crop.md](/home/ray/developer/ai/saccade/docs/experiments/reid/semantic_relink_and_crop.md:252)

---

## 🎯 專案里程碑

最後更新：2026-04-29

下一步：保留 current documented base 為 default；若需要比較變體，可留 `--semantic-reciprocal-margin 0.15` 作為 `IDs/MOTA`-leaning profile，主線轉進 Phase 3。
