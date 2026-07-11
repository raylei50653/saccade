# Association recovery — scripts lookup index

<!-- doc-status: research-synthesis -->
<!-- doc-promotion: not-for-report-citation-yet -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Purpose:** 可查找的 **腳本地圖**（path → door → 角色 → 優先級）。  
給 AssA / identity 開工與維護用，**不**寫 GO/NO-GO 結論、**不**嵌 baseline 數字。

**Sibling maps:**

| 檔 | 管什麼 |
|:--|:--|
| [association_recovery_info_source_contract_20260709.md](association_recovery_info_source_contract_20260709.md) | **誰是 truth**；腳本可生成什麼（Step 0） |
| [association_tools.yaml](association_tools.yaml) | **R** populated（Step 1B）；權威 door/role/recipe。Checker：`scripts/tools/check_association_tools.py` |
| [association_recovery_crosswalk_20260709.md](association_recovery_crosswalk_20260709.md) | 實驗 ↔ knobs / code / NO-GO / substrate |
| [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) | Door A 研究 hub（結論在那） |
| **this file** | 腳本 inventory + 查找（目前 = curated **H** snapshot，尚非 registry **R**） |

**Not sole active.** Status = D1 docs-only synthesis. 實驗 sole active 仍見 [TODO.md](../TODO.md)。

**Honesty:** 本檔表格來自 2026-07-09 清點（disk + research 引用），是 **Step 1 `association_tools.yaml` 的種子**，不是長期 source of truth。長期：door/role/fact-owner → **R**；path 健康 → checker(**D**)；結論 → research/ledger。

---

## 0. How to find a script (30s)

```text
1. 知道門？     → §2 按 Door 表
2. 知道任務？   → §1 任務 → 腳本捷徑
3. 只記得舊 path？ → §3 wrapper 表（轉 canonical）
4. 要開工鏈？   → §4 recipes（只組 CLI，不代寫結論）
5. 要全量掃？   → §5 分區全表
```

**Canonical 優先：** 有 wrapper 時改/讀 **canonical** 本體；wrapper 僅相容入口。

**Live eval 共用入口：** `scripts/eval/mot17.py`  
**CLI 旋鈕家：** `scripts/eval/config/lifecycle.py` · `geometry.py`  
**Preset 生產值：** `configs/presets/mamba_whole_graph.yaml` · `_m.yaml`（見 crosswalk §2）

---

## 1. Task → script 捷徑

| 我要… | 先跑 / 先開 | Door |
|:--|:--|:--|
| signal-study 表契約 / B1 宇宙 | [signal_table_schema.md](../../../research/eval/signal_table_schema.md)（`U_relink_pair`） | A |
| 建 offline 候選池 CSV | `scripts/tools/build_relink_candidates.py` | A |
| pairs → context/AUC/thr（B1 輸出） | `scripts/tools/summarize_relink_pairs.py` | A |
| B1 L0 gate coverage（prod-shaped + by-seq） | `audit_relink_safe_reject.py --by-gap --by-seq` | A |
| B1 offline **全訊號 auto-mine** | `mine_relink_signals.py --all` | A |
| energy **transform separability**（d′/Fisher，非 AUC 比 log） | `energy_transform_separability.py --all` | A |
| **combo AND safe region**（2D thr · 撈門檻） | `combo_gate_safe_region.py --all-default-pairs` | A |
| **gate rule search**（Pareto·itemset·greedy OR） | `gate_rule_search.py` | A |
| **gate rule LOO**（strict 6+1） | `gate_rule_search_loo.py` | A |
| **GT-safe region area**（GT_tail_mass，非 raw） | `gt_safe_region_area.py` | A |
| **Weight methods × productive plateau**（多核） | `weight_method_safe_region.py` | A |
| **LOO hurt attribution + atom repair re-LOO** | `loo_hurt_attribution.py` | A |
| **Repaired all-tail OR region (shared-q / 2D)** | `repaired_tail_or_safe_region.py` | A |
| **B2/e2e smoke (candidate_id only)** | `smoke_repaired_candidate_b2e2e.py` | A |
| **M-B1 offline phase hub (CLOSED nav)** | — · [hub note](m_b1_research_history_20260709_20260710.md) | A |
| 面積 / 轉向前 loss | `analyze_preloss_motion.py` · `analyze_turn_baseline.py` | A |
| speed×turn 分布 | `scripts/tools/sweep_speed_turn.py` | A |
| 否證 reach / s·G | `scripts/tools/validate_reach_gate.py` | A |
| 調 speed-weight score | `scripts/tools/optimize_relink_weight.py` | A |
| 線上 bridge dump 分析 | `scripts/tools/analyze_bidir_relink.py` | A |
| front/back depth 探針 | `scripts/tools/depth_ordering_probe.py` (+ `_auc` / `_gate_sweep`) | B |
| 分類 crossing-swap | `scripts/eval/diagnostics/analyze_crossing_swaps.py` | B |
| oracle 遮擋 hold 上界 | `scripts/eval/experiments/oracle_occlusion_hold.py` | B |
| embedding 區分力 | `scripts/eval/appearance/reid_id_benchmark.py` | C |
| 重連率 | `scripts/eval/diagnostics/reconnect_rate.py` | C |
| offline handover 標註 | `scripts/eval/diagnostics/cheb_gr_offline_handover_report.py` | D |
| handover 跨 run 比較 | `compare_handover_summaries.py` · `synthesize_handover_applicability.py` | D |
| 防假 FRCNN/SDP 條件 | `scripts/eval/diagnostics/compare_detector_suffix_runs.py` | D |
| substrate 上 replay handover 變體 | `scripts/eval/run_offline_handover_ablation.py` | D |
| 稀疏 bank ≡ dense | `scripts/eval/diagnostics/probe_sparse_bank_equivalence.py` | E |
| bank vs post-hoc occ-audit ref | `probe_occ_audit_bank_reference.py` | F |
| occ-exit 序列適用圖 | `analyze_occ_audit_seq_conditioning.py` | F |
| WP3 promotion 管線 | `run_occ_audit_wp3_promotion.py` | F |
| 全量 MOT 評測 A/B | `scripts/eval/mot17.py` | all |

路徑未寫全時，預設在 `scripts/tools/` 或 `scripts/eval/diagnostics/`（見 §5）。

---

## 2. By door (P 級 = 開工/維護優先)

| P | 含義 |
|:--|:--|
| **P0** | 現用開工鏈；壞了會卡住 AssA 實驗 |
| **P1** | 常用探針 / 第二門 |
| **P2** | 結案重跑、cold-supported / diagnostic 旁線（可重播，非現用開工） |
| **P3** | 冷存 / 歷史 occ 調參 |

### Door A — Bridge offline geometry

| P | Path | Role |
|:--|:--|:--|
| P0 | `scripts/tools/build_relink_candidates.py` | no-relink/no-interp → candidate CSV |
| P0 | `scripts/tools/analyze_preloss_motion.py` | pre-loss area / turn |
| P0 | `scripts/tools/analyze_turn_baseline.py` | pre-loss vs interior control |
| P0 | `scripts/tools/sweep_speed_turn.py` | speed×turn + npz |
| P0 | `scripts/tools/validate_reach_gate.py` | reach / drift vs spatial |
| P0 | `scripts/tools/optimize_relink_weight.py` | speed-weight grid + LOSO |
| P1 | `scripts/tools/analyze_bidir_relink.py` | live bridge dump per-attempt |
| P2 | `scripts/tools/analyze_missed_relinks.py` | GT missed-relink features |
| P2 | `scripts/tools/analyze_relink_stats.py` | gap-bin separation stats |
| P2 | `scripts/tools/remap_gpu_relinks.py` | GPU relink id remap |
| P2 | `scripts/tools/color_relink_features.py` | color hist on candidates |
| P2 | `scripts/tools/osnet_relink_features.py` | OSNet upper bound |
| P2 | `scripts/tools/mamba_relink_features.py` | mamba-head features |
| P2 | `scripts/tools/probe_relink_occlusion_signal.py` | occ signal vs true bridge |
| P2 | `scripts/tools/gap_occupancy_features.py` | gap occupancy features |
| P1 | `scripts/eval/diagnostics/relink_bridge_guard_report.py` | guarded bridge run summary |

Fact-owner: [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md).

### Door B — Crossing-swap / depth

| P | Path | Role |
|:--|:--|:--|
| P1 | `scripts/tools/depth_ordering_probe.py` | front/back geometry probe |
| P1 | `scripts/tools/depth_ordering_auc.py` | discrimination AUC |
| P1 | `scripts/tools/depth_ordering_gate_sweep.py` | same-height gate sweep |
| P1 | `scripts/eval/diagnostics/analyze_crossing_swaps.py` | swap event attribution |
| P1 | `scripts/eval/diagnostics/analyze_front_flag_exposure.py` | front-flag exposure |
| P1 | `scripts/eval/experiments/oracle_occlusion_hold.py` | oracle hold ceiling |
| P2 | `scripts/eval/probe_occ_swap_disambiguation.py` | occ signal at ABSORB swaps |

Fact-owner: [depth_ordering_crossing_swap.md](depth_ordering_crossing_swap.md). Re-read no_go **#39** before reopening hooks.

### Door C — Appearance / reconnect (sync critical path: no_go #57)

| P | Path | Role |
|:--|:--|:--|
| P2 | `scripts/eval/appearance/reid_id_benchmark.py` | embedding discriminability |
| P2 | `scripts/eval/diagnostics/reconnect_rate.py` | lost→recover rate |
| P2 | `scripts/eval/probe_assoc_appearance_veto.py` | appearance at primary assoc |
| P2 | `scripts/train/reid_domain_probe.py` | MOT-domain head probe |

### Door D — Cheb-GR offline handover (**P2 cold-supported / closed**)

> **Owner status (2026-07-12):** `research_status=closed`, `support_status=cold-supported`,
> `production_status=default-off non-headline`, `active_optimization=no`.
> Keep replay + diagnostic CLI; demote from P0 active work queue. See fact-owner status block.
> Authority for door/role/priority: [association_tools.yaml](association_tools.yaml).

| P | Path | Role |
|:--|:--|:--|
| P2 | `scripts/eval/run_offline_handover_ablation.py` | historical / diagnostic / replay |
| P2 | `scripts/eval/diagnostics/cheb_gr_offline_handover_report.py` | canonical / diagnostic label + registry |
| P2 | `scripts/eval/diagnostics/compare_handover_summaries.py` | diagnostic cross-run summary diff |
| P2 | `scripts/eval/diagnostics/synthesize_handover_applicability.py` | diagnostic applicability map |
| P2 | `scripts/eval/diagnostics/compare_detector_suffix_runs.py` | diagnostic detector-suffix guard |
| P2 | `scripts/eval/appearance/cheb_gr_osnet_gate.py` | probe / historical OSNet gate |

Fact-owner: [chebgr_handover_signal_map_20260704.md](chebgr_handover_signal_map_20260704.md).

### Door E — CleanFifo / sparse bank

| P | Path | Role |
|:--|:--|:--|
| P1 | `scripts/eval/diagnostics/probe_sparse_bank_equivalence.py` | sparse ≡ dense bank |
| P1 | `scripts/eval/diagnostics/probe_track_bank_fifo_replacement.py` | Track/Output bank FIFO |
| P1 | `scripts/eval/diagnostics/probe_forwarded_embedding_assoc_cost.py` | forwarded emb assoc cost |
| P2 | `scripts/tools/bench_bank_scatter.py` | bank scatter microbench |

Fact-owner: [clean_fifo_bank_substrate_20260704.md](clean_fifo_bank_substrate_20260704.md) · sparse bank note.

### Door F — Occ-exit audit (#55 sole active)

| P | Path | Role |
|:--|:--|:--|
| P0 | `scripts/eval/diagnostics/probe_occ_audit_bank_reference.py` | bank ref vs post-hoc |
| P0 | `scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py` | WP2 seq map |
| P0 | `scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py` | WP3 control/treatment |
| P1 | `scripts/eval/run_occ_audit_offline.py` | earlier offline A/B |

Fact-owner: occ_exit_audit_p55_*.md series.

### Occ-signal / FN peripheral (mostly closed probes)

| P | Path | Role |
|:--|:--|:--|
| P3 | `scripts/eval/probe_occ_separability.py` | score vs occlusion |
| P3 | `scripts/eval/probe_occ_activation_separability.py` | head activation vs occ |
| P3 | `scripts/eval/probe_occ_pairwise_confound.py` | non-geo residual confound |
| P3 | `scripts/eval/probe_lowiou_occ_gate.py` | occ relax low-IoU gate? |
| P3 | `scripts/tools/occ_event_values.py` | GT crossing event values |
| P3 | `scripts/tools/occ_candidate_analyze.py` | occ candidate analyze |
| P3 | `scripts/eval/analyze_occlusion_events.py` | occlusion events overview |
| P3 | `scripts/eval/analyze_occ_size.py` | occ size |
| P3 | `scripts/eval/occ_rank.py` · `occ_tune.py` | historical occ tune |
| P2 | `scripts/eval/diagnostics/analyze_assoc_fn.py` | assoc-failure FN reasons |
| P2 | `scripts/eval/diagnostics/analyze_fn.py` | FN analysis |
| P2 | `scripts/eval/diagnostics/analyze_near_miss_offsets.py` | near-miss offsets |
| P2 | `scripts/eval/diagnostics/analyze_near_miss_final_output.py` | near-miss final |
| P2 | `scripts/eval/diagnostics/analyze_near_miss_stage_attribution.py` | near-miss stage |
| P2 | `scripts/eval/diagnostics/label_boosted_birth_rows.py` | birth boost labeling |

### Substrate helpers

| P | Path | Role |
|:--|:--|:--|
| P0 | `scripts/eval/mot17.py` | full MOT eval entry |
| P2 | `scripts/tools/add_occlusion_to_seq.py` | inject demo occlusion bar |

---

## 3. Wrappers → canonical

Do not edit logic in the wrapper; follow the target.

| Wrapper (compat entry) | Canonical |
|:--|:--|
| `scripts/eval/reconnect_rate.py` | `scripts/eval/diagnostics/reconnect_rate.py` |
| `scripts/eval/reid_id_benchmark.py` | `scripts/eval/appearance/reid_id_benchmark.py` |
| `scripts/eval/analyze_crossing_swaps.py` | `scripts/eval/diagnostics/analyze_crossing_swaps.py` |
| `scripts/eval/oracle_occlusion_hold.py` | `scripts/eval/experiments/oracle_occlusion_hold.py` |
| `scripts/eval/cheb_gr_osnet_gate.py` | `scripts/eval/appearance/cheb_gr_osnet_gate.py` |

Redirect helper: `scripts/eval/_redirect.py`.

---

## 4. Recipes (CLI assembly only — conclusions stay manual)

**能用 / 缺什麼 + Gate vs Score 分工（不在本檔重複維護）：**  
[signal_table_schema §8](../../../research/eval/signal_table_schema.md) · [§0.5](../../../research/eval/signal_table_schema.md)（L0 gate / L1–L2 score）。

These are **suggested command chains**, not automated GO gates. Fill `<…>` from crosswalk §5 substrates.

### R-A — Offline bridge pool + kinematics

```bash
# 1) substrate dump (relink-off / interp-off as in offline hub §1)
#    s historical (offline hub as-of):
uv run python scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP \
  --no-interpolate-tracklets --output results/<substrate_no_interp>

#    m mainline B1 (must also force bridge OFF — m preset defaults bridge ON):
uv run python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --detect-barrier event \
  --no-interpolate-tracklets --no-relink-bridge-enabled \
  --output results/<m_b1_substrate>

# 2) candidate table → B1 study_dir (see signal_table_schema §0.2)
uv run python scripts/tools/build_relink_candidates.py \
  --mot-dir results/<m_b1_substrate> --gt-root datasets/MOT17/train \
  --out out/signal_study/<id>/pairs
uv run python scripts/tools/summarize_relink_pairs.py \
  --pairs out/signal_study/<id>/pairs.csv \
  --study-dir out/signal_study/<id> --hard-dist 1.0 \
  --preset mamba_whole_graph_m --mot-dir results/<m_b1_substrate> \
  --relink off --interpolate off --double-buffer true --copy-pairs

# 3) optional analyses (pick as needed; s-era kinematics helpers)
uv run python scripts/tools/analyze_preloss_motion.py --window 8
uv run python scripts/tools/analyze_turn_baseline.py --min-speed 0.03
uv run python scripts/tools/sweep_speed_turn.py
uv run python scripts/tools/validate_reach_gate.py
uv run python scripts/tools/optimize_relink_weight.py
```

Verified m smoke (pointer only): [m_b1_substrate_smoke_20260709.md](../../../research/eval/m_b1_substrate_smoke_20260709.md) → `out/signal_study/m_b1_smoke_*`.  
Artifacts often under `out/signal_study/` · `scripts/tools/out/` · `docs/modules/semantic/research/figures/`.

### R-B2 — Online reconnect (bridge on/off, B2)

```bash
# production-like m: interpolate ON; ablate only bridge
uv run python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --detect-barrier event --no-relink-bridge-enabled \
  --output results/<m_b2_off>
uv run python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --detect-barrier event --relink-bridge-enabled \
  --output results/<m_b2_on>

uv run python scripts/eval/diagnostics/reconnect_rate.py \
  --pred-dir results/<m_b2_on> --baseline-dir results/<m_b2_off> \
  --label bridge_on --baseline-label bridge_off \
  --json-out out/signal_study/<id>/metrics_reconnect.json \
  --events-out out/signal_study/<id>/events_bridge_on.csv \
  --baseline-events-out out/signal_study/<id>/events_bridge_off.csv
```

Live pointer: [m_b2_reconnect_bridge_ab_20260709.md](m_b2_reconnect_bridge_ab_20260709.md).

### R-B1s — Safe-reject audit (constrained FP pruning)

```bash
# After pairs exist (R-A / B1). Does NOT tune thr F1 — reports FP_removed @ GT_hurt<=ε.
uv run python scripts/tools/audit_relink_safe_reject.py \
  --pairs out/signal_study/<id>/pairs.csv \
  --study-dir out/signal_study/<id> \
  --write-study --by-gap
```

Contract: [signal_table_schema.md](../../../research/eval/signal_table_schema.md) §0.4.  
Live: [m_b1_research_history_20260709_20260710.md](m_b1_research_history_20260709_20260710.md) §3d.

### R-B — Depth / swap probe

```bash
uv run python scripts/tools/depth_ordering_probe.py ...
uv run python scripts/tools/depth_ordering_auc.py ...
uv run python scripts/eval/diagnostics/analyze_crossing_swaps.py ...
```

### R-D — Offline handover signal pipeline

```bash
# produce handover log via mot17 + module YAML (see chebgr signal map / TODO)
uv run scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer \
  --module-lifecycle configs/modules/cheb_gr_offline_mnv4.yaml \
  --cheb-gr-offline-log --output results/<run>

uv run python scripts/eval/diagnostics/cheb_gr_offline_handover_report.py \
  --handover-log results/<run>/_cheb_gr_offline_handover.csv \
  --baseline-dir results/<no_handover> --pred-dir results/<run> \
  --out-csv results/<run>/_cheb_gr_offline_handover_labeled.csv \
  --registry-md results/<run>/parameter_registry.md \
  --summary-json results/<run>/parameter_summary.json

uv run python scripts/eval/diagnostics/compare_handover_summaries.py \
  results/<run_a>/parameter_summary.json results/<run_b>/parameter_summary.json

uv run python scripts/eval/diagnostics/synthesize_handover_applicability.py \
  results/<run_a>/parameter_summary.json results/<run_b>/parameter_summary.json \
  --out-md results/<map>.md --out-json results/<map>.json

# if claiming detector-suffix conditions:
uv run python scripts/eval/diagnostics/compare_detector_suffix_runs.py ...
```

### R-E — Sparse bank equivalence

```bash
uv run python scripts/eval/diagnostics/probe_sparse_bank_equivalence.py ...
# optional: run_offline_handover_ablation.py for end-to-end bank mode
```

### R-F — Occ-exit (#55)

```bash
uv run python scripts/eval/diagnostics/probe_occ_audit_bank_reference.py ...
uv run python scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py ...
uv run python scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --out-dir results/occ_exit_p55_wp3 ...
```

Exact flags: owning research notes (WP2/WP3) and script `--help`.

---

## 5. Layout on disk (mental model)

```text
scripts/eval/mot17.py                 # live / full eval entry
scripts/eval/config/{lifecycle,geometry}.py
scripts/eval/diagnostics/             # AssA diagnostics + probes (preferred home)
scripts/eval/appearance/              # ReID benchmarks / gates
scripts/eval/experiments/             # oracle / one-shot experiments
scripts/eval/*.py                     # many = thin wrappers → above
scripts/tools/                        # offline candidate pool + depth + features
scripts/train/reid_domain_probe.py    # domain probe (train tree)
configs/presets/mamba_whole_graph*.yaml
configs/modules/cheb_gr_offline_*.yaml
```

**Typical artifacts (local / often gitignored):**

```text
scripts/tools/out/relink_candidates.csv
scripts/tools/out/speed_turn_sweep.npz
results/diag_*_no_reid_*              # frozen MOT substrates
results/*/_cheb_gr_offline_handover.csv
results/*/_occ_audit.csv
results/occ_exit_p55_wp3/
docs/modules/semantic/research/figures/
```

---

## 6. What scripts should vs should not own

**Full matrix:** [info source contract](association_recovery_info_source_contract_20260709.md) §2–§3.

| Scripts **may** automate | Stay **manual** (research notes / human) |
|:--|:--|
| Path existence / inventory refresh (**D**) | GO / NO-GO verdict text (**V**) |
| CLI recipe *print* from registry (**R**) | door / role / fact-owner assignment (**R** human) |
| Artifact *name* presence checks (**R**+**D**) | ledger / report_data promotion (**M**/**V**) |
| NO-GO **id** existence (**N**) | NO-GO **verdict** body (**N**/**V** human) |
| Preset vs schema knob extract (**C**) | hardcoding knob masters in this MD |

**Order:** contract → `association_tools.yaml` → checker → optional MD render.  
Checker: `uv run python3 scripts/tools/check_association_tools.py` (`--list`, `--print-recipe R-A`).  
**Do not** land a second script that hardcodes this snapshot as inventory truth.

---

## 7. Count snapshot (2026-07-09; Door D priority refreshed 2026-07-12)

| Cluster | Approx. paths (incl. wrappers) | P0 canonical |
|:--|:--|:--|
| A offline + features | ~16 | 6 tools chain |
| B depth / swap | ~9 | — (P1 set) |
| C appearance | ~6 | — |
| D Cheb-GR | ~7 | — (**all P2 cold-supported**; was 5×P0) |
| E bank | 4 | — (P1) |
| F occ-exit | 4 | 3 diagnostics |
| occ-signal / FN / misc | ~15 | — |
| **Total tracked** | **~60 paths** | **~15–20** active-work P0 (Door A/F) |

Re-count after large moves; prefer updating tables over inventing a second index.

---

## 8. Related doc index

| Doc | Link |
|:--|:--|
| Experiment crosswalk | [association_recovery_crosswalk_20260709.md](association_recovery_crosswalk_20260709.md) |
| Offline hub | [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) |
| Module card | [../README.md](../README.md) |
| Sole active | [../TODO.md](../TODO.md) |
| NO-GO | [../../../reference/no_go_registry.md](../../../reference/no_go_registry.md) |

---

## 9. Maintenance checklist

When adding an AssA-related script:

1. **After Step 1:** add the tool to `association_tools.yaml` (**R**) first — door, role, fact-owner, recipes.  
2. **Until Step 1:** add a row under the correct **Door** in §2 (P level + one-line role) and treat it as temporary **H**.  
3. If it is a **new task entry point**, add §1 shortcut.  
4. If old path remains, add §3 wrapper row (and later **R** `role: wrapper` + canonical id).  
5. If it is a **standard chain step**, extend §4 recipe (commands only; no auto-exec).  
6. Do **not** paste metrics or promotion decisions here — link the research note (**M**/**V**).  
7. Path health after Step 2: run checker; do not hand-assert “exists” without **D**.
