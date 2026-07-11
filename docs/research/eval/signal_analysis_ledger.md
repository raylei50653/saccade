# Signal Analysis Ledger（深度訊號分析總帳）

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- fact-owner: signal-depth-analysis = this file (index) + out/signal_study/<id>/ (numbers) -->

**Purpose:** **一次一個 gate / 一個連續訊號** 做深度分析時，**只在這裡掛一列**。  
這是「分析過哪些訊號、結論 pointer、下一步」的**統一入口**；不是 e2e metrics 百科，也不是 production GO 表。

| 誰管什麼 | 家 |
|:--|:--|
| **本檔** | 深度分析**索引**（一訊號一列；狀態；一句 verdict；連 note + study） |
| **數字 master** | `out/signal_study/<study_id>/`（json/csv；禁止把大表嵌死 markdown） |
| **契約 / recipe** | [signal_table_schema.md](signal_table_schema.md)（A/B1/B2、L0/L1/L2、§0.5 Gate vs Score） |
| **可引用 e2e / 決策數字** | [evidence_ledger.md](../evidence_ledger.md)（升格後才抄一行） |
| **NO-GO 結案** | [no_go_registry.md](../../reference/no_go_registry.md) |
| **長 note 正文** | 通常 `docs/modules/semantic/research/`（relink/gate）或本目錄 eval note |

**協議（強制）：**

1. **一次一個 gate / 一個 score 欄**（可同 note 裡對照 hard pool，但不要一次掃十個 rule 當「深度分析」）。  
2. 新分析 → **本表加一列** + study_dir + 短 note；重測 → **新 `study_id`**，改 pointer，不改舊 study。  
3. 結論分層標清：**L0 gate 覆蓋** vs **L1 term 可分性** vs **L2 online**（見 schema §0.5）。  
4. 7-seq MOT 便宜（~30s）— 需要就重跑；不必為省 eval 縮序列。  
5. 本檔**不**嵌 master 表；orient 數字 as-of 一句即可，裁決以 study 為準。

開發入口：[DEVELOPMENT.md](../../../DEVELOPMENT.md) §3「數據驅動 gate / relink」→ 契約 schema + **本 ledger**。

---

## 0. 狀態圖例

| Status | 含義 |
|:--|:--|
| `🔄 analyzing` | 正在挖；note 可 WIP |
| `✅ depth-done` | 單訊號深度讀完；有 study + note；**未**主張上線 |
| `⏸ parked` | 有意暫停；理由在 notes |
| `⬆ promoted` | 已進 ledger / no_go / preset 討論（另列 promotion） |
| `∅ not-started` | 排隊；尚無 study |

**Layer：** `L0` support/coverage · `L1` ranking term · `L2` online state（B2）· `mix` 僅當 note 明確分節。

---

## 1. 總表（唯一索引）

> 新列插在表**頂**（最新在上）。`signal_id` 穩定、可 grep。

| signal_id | 物理量 / gate | Layer | Substrate | Status | Study (master) | Note | One-line verdict (as-of) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `m.gate.portable_or_tail_hook` | **research default-off OR-tail hook** | eng / L2 path | freeze + B-audit | ✅ **Stage 1 CLOSED** | [`m_b1_hook_ab_20260710T071001Z_stage1_close`](../../../out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/) | [e2e](../../modules/semantic/research/m_b1_stage1_online_hook_final_20260710.md) · [thread](../threads/closed/m_b1_online_hook_20260709.md) | **1a** elig244；freeze null；**1b** P/F pass；**B-audit** 244 zero-fire full table recon ok；rebased strict A0；det B≡B_repeat；runtime named；**overall CLOSED**；freeze online relevance NULL；preset NO |
| `m.gate.repaired_b2e2e_smoke` | **B2/e2e smoke**（candidate_id only） | validation | m B1 offline + B2 ref | ⚠ **offline_pass / online_blocked** | [`m_b2e2e_smoke_m_b1_repaired_eps0_loo_pass_20260709T151000Z`](../../../out/signal_study/m_b2e2e_smoke_m_b1_repaired_eps0_loo_pass_20260709T151000Z/) | [B2/e2e smoke](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | offline GT0 FP8721；**online_blocked** 是正確邊界非失敗；**historical** next at this checkpoint = default-off hook · **superseded by** Stage 1 CLOSED + Stage 2 Q4.5；preset NO |
| `m.gate.repaired_tail_region` | **all-tail OR shared-q / 2D ε=0 region** | validation | m B1 7-seq | ✅ **LOO_pass_region_candidate** | [`m_repaired_tail_region_20260709T150000Z`](../../../out/signal_study/m_repaired_tail_region_20260709T150000Z/) | [tail OR region](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | ε0 shared-q safe **56%** · p80 **14%** · best_q**0.83**≈q85；LOO p80 **15%** freeze 7/7；2D pairs **broad**；仍 offline |
| `m.gate.repaired_eps0_loo_pass` | **Freeze repaired ε=0 all-tail OR candidate** | L0 policy card | m B1 7-seq | ✅ **LOO_pass_region_candidate**（≠ production） | [`m_b1_repaired_eps0_loo_pass_20260709`](../../../out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/) | [repaired candidate card](../../modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md) · [region](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | OR-5 **all tails**；FP **8721** GT0；LOO **7/7** retained **97%**；region audit **厚平台 near q85**；**historical** next at freeze = B2/e2e · **current lifecycle** → Stage 1/2 finals；preset NO |
| `m.gate.loo_atom_repair` | **LOO hurt 歸因 → atom repair → re-LOO** | validation | m B1 7-seq | ✅ **loo_pass under repair**（≠ production） | [`m_loo_attr_20260709T143000Z`](../../../out/signal_study/m_loo_attr_20260709T143000Z/) | [LOO atom repair](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | 漏點=`zone_q50/70`+`gap_61_150`；**ban_gap+ban_zone → 7/7 GT0**，teFP 1244（**retained 97%**）；speed_mismatch 會在半修時竄入；preset 仍 NO |
| `m.weight.safe_region` | **加權方法 × productive plateau**（非 best FP） | meta / L1→L0 | m B1 7-seq | ✅ depth-done | [`m_weight_safe_20260709T142000Z`](../../../out/signal_study/m_weight_safe_20260709T142000Z/) | [weight method safe region](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **無厚 ε=0 production plateau**；ε0.01 clipped_logz 最厚/最高 FP（relaxed frontier）；GT-CDF/soft-AND 較乾淨但薄；**research-only** → atom repair 主線 |
| `m.gt.safe_region_area` | **GT_tail_mass** 安全域面積（非 raw thr） | meta / L0 | m B1 7-seq | ✅ 量測完成 | [`m_gt_safe_area_20260709T125933Z`](../../../out/signal_study/m_gt_safe_area_20260709T125933Z/) | [GT safe region area](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | ε0 safe% **&lt;1%**（isolated）；ε0.01 ~2–5% thin；**productive@80 極薄**；非 production-promising |
| `m.gate.rule_search` | atoms→AND→OR in-sample | L0–L3 | m B1 7-seq | ✅ **in-sample candidate**（≠ production） | [`m_gate_rule_search_20260709T124534Z`](../../../out/signal_study/m_gate_rule_search_20260709T124534Z/) | [policy card](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) · [architecture](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **2026-07-09T124534Z** ε=0 OR-5 FP **9130** GT_hurt **0**；status=in-sample only；blocked: LOO/B2/e2e；preset unchanged |
| `m.gate.rule_search.loo` | strict LOO（train 6 搜、held-out 套 thr） | validation | m B1 7-seq | ⚠ **loo_partial**（baseline） | [`m_gate_rule_loo_20260709T125245Z`](../../../out/signal_study/m_gate_rule_loo_20260709T125245Z/) | [LOO note](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) · [repair](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | baseline **5/7**；repair 後見 `m.gate.loo_atom_repair` |
| `m.combo.safe_region` | 2D thr surface AND：**safe region area** vs best | L0 combo | m B1 7-seq | ✅ depth-done | [`m_combo_safe_20260709T124215Z`](../../../out/signal_study/m_combo_safe_20260709T124215Z/) | [combo gate safe region](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **recoverability 主收益**：nv 9→436（33×）；ta 單軸≥10 可放到 4.8；**無** marginal FP gain；勿只報 best FP |
| `m.energy.transform_separability` | raw/log1p/sqrt/rank：**AUC vs d′/Fisher/logloss** | meta | m B1 7-seq | ✅ depth-done | [`m_energy_xform_20260709T123727Z`](../../../out/signal_study/m_energy_xform_20260709T123727Z/) | [energy transform separability](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **禁止 AUC 比 transform**；幾何族 log_linear_good（d′ 1.0→1.4）；dir=raw_linear；speed=rank_only/hard no；加權前必須 compressive |
| `m.dist.stability` | 分布尾部/跨 seq thr 穩定性（實作） | meta | m B1 7-seq | ✅ depth-done | [`m_b1_dist_stability_20260709T124000Z`](../../../out/signal_study/m_b1_dist_stability_20260709T124000Z/) | [distribution stability](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | linear 重尾 kurt~11；px=0.4 在 GT CDF 30%→跨 seq hurt std **20pp**；log≠修固定 thr；融合須 z-score；h-gate 尾部才穩 |
| `m.scale.linear_vs_log` | 同物理量 × linear/log1p/sqrt/band | meta | m B1 7-seq | ✅ depth-done | [`m_b1_scale_compare_20260709T123000Z`](../../../out/signal_study/m_b1_scale_compare_20260709T123000Z/) | [scale linear/log note](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **單調變換 AUC 不變**；log 只改 thr 單位/skew；h-ratio band 已近 log-sym；raw ratio 不當 ranker |
| **batch** `m.b1.offline_catalog` | 8 連續訊號 auto-mine（**linear 默認**） | L0+L1 | m B1 7-seq pairs | ✅ depth-done | [`m_b1_signal_mine_20260709T122534Z`](../../../out/signal_study/m_b1_signal_mine_20260709T122534Z/) | [m_b1_signal_mine_batch_20260709](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | hard AUC 冠軍 score_m_bridge 0.80；幾何族 0.75–0.80；dir/speed hard≈0.53；無閉合 ID |
| `m.score_m_bridge.px` | speed-weighted score；prod `px=0.4` | L0+L1 | ↑ batch `signals/` | ✅ depth-done | 同上 | 同上 + Gate 細節在 json | full 0.87 / hard **0.80**；prod GT_hurt **70%** FP_rm 99%（操作區） |
| `m.fwd_bwd_resid` | ½(fwd+bwd) residual | L1 | ↑ | ✅ depth-done | 同上 | batch note | full 0.86 / hard **0.79**；與 score 同族 |
| `m.h_ratio.scale` | \|log h\|；prod `[0.6,1.7]` | L0+L1 | ↑ + 專深 study | ✅ depth-done | [h_ratio 專深](../../../out/signal_study/m_gate_h_ratio_7seq_20260709T122056Z/) · batch | [h_ratio note](../../modules/semantic/research/m_gate_h_ratio_signal_7seq_20260709.md) | full 0.86 / hard 0.78；prod hurt **3.2%** FP_rm **54%**（最佳稅比） |
| `m.bridge_dist.midpoint` | builder mid-point | L1 (+L0 thr) | ↑ + B1 smoke | ✅ depth-done | [B1 smoke](../../../out/signal_study/m_b1_smoke_20260709T092543Z/) · batch | [discriminability](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | full 0.87 / hard 0.76；base-rate wall |
| `m.dist_h` | foot dist / h | L1 | ↑ batch | ✅ depth-done | batch | batch note | full 0.84 / hard 0.75；弱於 residual blend |
| `m.gap` | time gap | context | ↑ batch | ✅ depth-done | batch | batch note | full 0.73 / hard 0.65；**context 非 ID** |
| `m.dir_cos` | direction cosine | L0/L1 | ↑ batch | ✅ depth-done | batch | batch note | full 0.69 / hard **0.54 ~random**；難池不可用 |
| `m.speed_mismatch` | \|exit−entry\| speed | L0/L1 | ↑ batch | ✅ depth-done | batch | batch note | full 0.61 / hard **0.53**；ε0 僅 ~3% FP |
| `m.prod_shaped.bulk_cover` | m px∨h 批量地圖 | L0 map | pairs | ✅ depth-done | [gate_coverage](../../../out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/) | [coverage note](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | 地圖；分訊號見上 |
| `m.reconnect.bridge_onoff` | online reconnect | L2 | m 7-seq | ✅ depth-done | [m_b2](../../../out/signal_study/m_b2_bridge_ab_20260709T094646Z/) | [m_b2 note](../../modules/semantic/research/m_b2_reconnect_bridge_ab_20260709.md) | B2；≠ B1 AUC |
| `frame.iou_maha_cost` | 幀內 support gates | L0 | U_cand | ∅ not-started | — | — | **需 cand dump**；下一宇宙 |
| `live.bridge_fire` | attempts/accepts | L0 live | MOT bridge ON | ∅ not-started | — | — | 需 live log 產物 |
| `score.audit.margin` | GT vs best FP margin | L1→L2 | pairs/cand | ∅ not-started | — | schema §0.5 | **工具待建** |
| `s.bridge_dist.historical` | s offline hub | L1 | s historical | ⏸ parked | s hub as-of | [offline_relink](../../modules/semantic/research/offline_relink_candidate_analysis.md) | 方法可引；數字不當 m |

---

## 2. 單訊號深度分析 — 必報清單（copy 到 note）

每個 `✅ depth-done` 列對應的 note **至少**覆蓋：

```text
[ ] signal_id + 拒絕/通過定義（或 score 方向）
[ ] substrate：preset · 7-seq · relink/interp flags · pairs/mot path
[ ] 連續量：pos/neg median · p05/p90 · full+hard AUC（若可排序）
[ ] L0 工作點：production 帶寬或 thr 的 GT_hurt / FP_removed（full + 建議 hard）
[ ] by gap · by seq（覆蓋偏差）
[ ] band 或 thr 小掃（可選但推薦）
[ ] ε∈{0,0.1%,1%} 1D frontier（若 reject-high score）
[ ] hurt GT vs kept GT 側寫（gap / seq）
[ ] surviving FP 體量（過閘後還剩多少負例）
[ ] 一句：有/無訊號 · 能不能當 safe_reject · 解不開什麼
[ ] study_dir 路徑；禁止「只存在 chat 的數字」
```

**禁止：** 把 thr F1 當 headline；只報全池 AUC；L0/L1/L2 混稱；一次 note 結十個 gate。

---

## 3. 新開一列的流程

```bash
# 1) 選 signal_id（本表未佔用）
# 2) 數字 → 新 study（可重用既有 pairs；7-seq MOT 需要就重跑）
STUDY=out/signal_study/<signal_id_or_stamp>/

# 3) 短 note：docs/modules/semantic/research/<signal>_signal_7seq_<date>.md
#    正文 = 解讀 + pointer；master = study

# 4) 本檔 §1 插一列 Status=✅ 或 🔄

# 5) owning README 索引一行（semantic 或 eval README）
```

**與 bulk audit 分工：**  
`audit_relink_safe_reject.py --by-gap --by-seq` = **地圖 / 多 rule 掃描**。  
**深度** = 本 ledger 單列 + 專用 study json（如 `signal_gate_h_ratio.json`）。地圖跑完後仍要對優先訊號做 §2 清單。

---

## 4. 自動挖掘 vs 人工

| 自動（已落地） | 人工 / 未建 |
|:--|:--|
| `mine_relink_signals.py --all` 挖完整 B1 offline catalog | 升格 evidence_ledger / 改 preset |
| 每訊號 json + hard AUC 排名 + auto_verdict | Score Audit margin 工具（§0.5） |
| 重跑 7-seq pairs 後一鍵再 mine | U_cand / live fire 產物管線 |

```bash
# 閉合 B1 offline 訊號數字（完整 catalog）
uv run python scripts/tools/mine_relink_signals.py \
  --pairs out/signal_study/m_b1_smoke_*/pairs.csv \
  --all --study-dir out/signal_study/m_b1_signal_mine_<stamp>
```

## 5. 排隊

### M-B1 offline gate / signal / safe-region research phase — **CLOSED**

**Phase hub (nav + maintenance rules):**  
[m_b1_research_history_20260709_20260710.md](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md)

```text
phase: M-B1 offline gate / signal / safe-region research
status: closed successfully (2026-07-09)
deliverable:
  - frozen offline region candidate m_b1_repaired_eps0_loo_pass_20260709
  - LOO_pass_region_candidate + offline_smoke_pass + online_blocked
  - next-phase contract: research default-off portable OR-tail hook
NOT delivered: production gate / preset change

maintenance:
  - status/verdict churn → candidate card + this ledger + hook contract only
  - Tier-B as-of notes → closed; re-run study_id if numbers must change
  - do not re-edit intermediate method notes for “latest status”
```

**Offline phase closed；Stage 1 online hook eng CLOSED (2026-07-10)。production 仍 NO。**

### Next phase (Stage 2 domain — separate)

```text
1. Stage 1 portable OR-tail hook eng: CLOSED  ← B-audit 244-row table ready
2. Stage 2 entry contract: m_b1_5_stage2_entry_contract_20260710.md (G0–G4)
3. Stage 2 Q1–Q3: DONE — study m_b1_5_stage2_q1q3_20260710
     Q1 PASSED · Q2 PASSED · Q3 SUFFICIENT (safe_removable=23)
4. Stage 2 Q4: DONE — study m_b1_5_stage2_q4_20260710
     q4_separability_grade: C (separability_weak_or_unstable; best AUC_oriented=0.588)
     → stage2_entry_terminal_after_q4: B
5. Stage 2 Q4.5: DONE — study m_b1_5_stage2_q45_20260710 (evaluator v4)
     q45_atlas_terminal B isolated_safe_points_only · 154 productive-safe · 0 region candidates
6. Next authorized: ranking / assignment-relative decision modeling
7. production preset: still NO
```

Canonical docs:  
[Stage 2 final](../../modules/semantic/research/m_b1_5_stage2_d_online_final_20260710.md) ·  
[entry contract](../../modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md) ·  
[Stage 1 final](../../modules/semantic/research/m_b1_stage1_online_hook_final_20260710.md) ·  
[offline history](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) ·  
[consolidation report](../../modules/semantic/research/m_b1_doc_consolidation_report_20260710.md)

### Other universes

```text
- frame.iou_maha_cost
- score.audit.margin
```

semantic **WIP sole active** 仍是 #55 occ-exit；本線 = D1 RESEARCH 旁線已結案，不佔 sole active。

---

## 6. Related

| 資源 | 角色 |
|:--|:--|
| [signal_table_schema.md](signal_table_schema.md) | 宇宙 / L0–L2 / study_dir 契約 |
| [eval/README.md](README.md) | 本目錄索引 |
| [association_recovery_scripts_index](../../modules/semantic/research/association_recovery_scripts_index_20260709.md) | 腳本查找 |
| `scripts/tools/mine_relink_signals.py` | **B1 offline 全 catalog 自動深度 mine** |
| `scripts/tools/energy_transform_separability.py` | **raw/log/sqrt/rank**：d′/Fisher/logloss（**非 AUC 比 transform**） |
| `scripts/tools/combo_gate_safe_region.py` | **2D AND/OR surface** + safe_region_area / recoverability |
| `scripts/tools/gate_rule_search.py` | **L1–L3**：atoms · Pareto · AND mine · greedy OR |
| `scripts/tools/audit_relink_safe_reject.py` | L0 multi-rule / prod-shaped map |
| `scripts/tools/summarize_relink_pairs.py` | B1 AUC + thr |
| `out/signal_study/` | 全部 study 落盤根 |
