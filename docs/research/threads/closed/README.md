# Closed research threads

**定位：** 已結案 thread 卡的**檔案家**。仍是 navigation-only，不是 evidence home。

```text
threads/         = proposed · active · parked
threads/closed/   = closed（本目錄）
docs/archive/    = archived（很少用；不再需要 threads 導航）
```

**索引權威：** 父層 [../README.md](../README.md) 的 **Closed** 表（terminal one-liner + close date）。  
本 README 只列檔名，避免第二真相。

## Cards

| File | Closed | Terminal (one-line) |
|:--|:--|:--|
| [safe_region_assetization_20260710.md](safe_region_assetization_20260710.md) | 2026-07-11 | A1 CLOSED · handoff → gt_support_morphology |
| [composition_grammar_coverage_program_20260710.md](composition_grammar_coverage_program_20260710.md) | 2026-07-10 | SUPERSEDED · absorbed into assetization |
| [composition_grammar_safe_region.md](composition_grammar_safe_region.md) | 2026-07-10 | CLOSED A0 · handoff → assetization |
| [m_b1_online_hook_20260709.md](m_b1_online_hook_20260709.md) | 2026-07-10 | CLOSED · S1+S2 Q4.5 B · ranking deferred |

## On move-in

見父層 [How to close a thread](../README.md#how-to-close-a-thread)：

1. frontmatter `doc-status: closed` + `closed: YYYY-MM-DD`
2. body Final status / History close line
3. `git mv` 進本目錄 + 修相對連結 + 全庫 path rewrite
4. 父 README Closed 表加行；本表加行

**Same-PR：** 結案移動必須連同 link 修復一併提交。
