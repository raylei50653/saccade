# m B1 substrate smoke — offline_relink data-spec on `mamba_whole_graph_m`

**Date:** 2026-07-09  
**Status:** **D1 煙測通過**（工具 + 真 m 資料）  
**Preset:** `mamba_whole_graph_m` · detector SDP · double-buffer · **bridge off · interpolate off**  
**Master artifacts:** [`out/signal_study/m_b1_smoke_20260709T092543Z/`](../../../out/signal_study/m_b1_smoke_20260709T092543Z/)  
（`context.json` · `metrics_auc.json` · `metrics_thr.csv` · `pairs.csv` · study `README.md`）

> 本檔 **只 pointer + 驗收一句**。AUC / thr / base rate **以 study_dir 為準**，不在此嵌 master 表。  
> 方法樣式與注意事項： [signal_table_schema.md](signal_table_schema.md) **§0.2–0.3**。  
> s 歷史 hub： [offline_relink_candidate_analysis.md](../../modules/semantic/research/offline_relink_candidate_analysis.md)（數字 as-of 該文，**非** m 現況）。

## Verdict

在 `mamba_whole_graph_m` 上可完整跑出 offline_relink / B1 需要的資料規格：

1. substrate MOT dump（7-seq SDP）  
2. `build_relink_candidates.py` → `U_relink_pair` 齊欄  
3. `summarize_relink_pairs.py` → 契約三檔；`n_pos>0`、hard≤full、`citation_ok=true`

**一句：** B1 管道在真 m 資料上驗收通過；D2 研究 note 另開。

## Recipe（摘要）

完整命令與 preset→CLI 對照見 study `README.md`。核心覆寫：

```text
--preset mamba_whole_graph_m --detector SDP
--double-buffer --detect-barrier event
--no-interpolate-tracklets --no-relink-bridge-enabled
```

（m yaml 預設 bridge/interp **ON**；B1 必須 CLI 關掉。）

## Paths

| 角色 | 路徑 |
|:--|:--|
| study（數字 master） | `out/signal_study/m_b1_smoke_20260709T092543Z/` |
| substrate MOT | `results/MOT17_eval_m_b1_substrate_20260709T092543Z/` |
| eval log | `logs/m_b1_substrate_20260709T092543Z.log` |

## Out of scope

- 不當 D2 解讀 headline  
- 不改 production preset  
- 不重寫 s 文內嵌表  
- B2 reconnect 另見 D3
