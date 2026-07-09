# M-B1 offline gate / safe-region research phase — hub

<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->
<!-- fact-owner: m-b1-offline-phase = this hub (nav) + signal_analysis_ledger (index) + candidate card (freeze identity) -->

**Status:** **CLOSED successfully** (2026-07-09)  
**Production:** not started · **preset:** unchanged  

```text
offline research candidate 已成立
online/e2e 邊界尚未打通
下一步只補 default-off hook
preset 不動
```

---

## Maintenance contract (why this hub exists)

This phase produced many dated notes. **Do not treat each note as a living status page.**

| Role | Update when… | Do not… |
|:--|:--|:--|
| **This hub** | Phase open/close; rare renames | Copy large metric tables here |
| **[signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md)** | New `signal_id` / phase queue | Embed master numeric tables |
| **[candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)** | Freeze identity / validation_status change | Re-open rule search in-place |
| **[hook contract](m_b1_portable_or_tail_hook_contract_20260709.md)** | Next eng phase only | Expand into production design |
| **Tier-B as-of notes** | Almost never (re-run → new study_id + optional new note) | Patch status/verdict after freeze |

**Numbers master:** `out/signal_study/<id>/` (gitignored). Notes point; they do not re-own tables.

**Next status churn** belongs only on: hook contract → (later) e2e result note → candidate card `e2e_safe_for_default_off`.

---

## Canonical freeze

| Field | Value |
|:--|:--|
| **candidate_id** | `m_b1_repaired_eps0_loo_pass_20260709` |
| **validation_status** | `LOO_pass_region_candidate` |
| **offline_smoke** | pass (GT0 · FP=8721) |
| **online** | blocked |
| **e2e_safe_for_default_off** | no |
| **Card** | [m_b1_repaired_eps0_loo_pass_candidate_20260709.md](m_b1_repaired_eps0_loo_pass_candidate_20260709.md) |
| **Study (local)** | `out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/` |

---

## Document tiers

### A — keep open in your head (3)

| Doc | Why |
|:--|:--|
| **This hub** | Navigation + maintenance rules |
| [candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md) | Freeze identity |
| [hook contract](m_b1_portable_or_tail_hook_contract_20260709.md) | **Only** next-phase work |

Plus always: [signal_analysis_ledger](../../../research/eval/signal_analysis_ledger.md) · [scripts index](association_recovery_scripts_index_20260709.md)

### B — closed as-of method notes (read on demand)

Do not update verdicts; re-run study if numbers must change.

| Doc | One-line role |
|:--|:--|
| [architecture](m_b1_gate_rule_search_architecture_20260709.md) | atoms→AND→OR search design |
| [unrepaired policy card](m_b1_policy_card_eps0_or5_20260709.md) | **superseded** for LOO claims |
| [LOO baseline](m_b1_gate_rule_search_loo_20260709.md) | 5/7 partial before repair |
| [LOO atom repair](m_b1_loo_hurt_atom_repair_20260709.md) | ban_gap+ban_zone → 7/7 |
| [tail region](m_b1_repaired_tail_or_safe_region_20260709.md) | q85 productive band |
| [B2/e2e smoke](m_b1_repaired_candidate_b2e2e_smoke_contract_20260709.md) | offline_pass / online_blocked |
| [GT safe area](m_b1_gt_safe_region_area_20260709.md) | 2D GT-tail mass protocol |
| [weight methods](m_b1_weight_method_safe_region_20260709.md) | no thick ε=0 weight plateau |
| [combo safe region](m_b1_combo_gate_safe_region_20260709.md) | recoverability (raw grid) |
| [energy transform](m_b1_energy_transform_separability_20260709.md) | no AUC-for-transform |
| [dist stability](m_b1_signal_distribution_stability_20260709.md) | thr / seq stability |
| [scale linear/log](m_b1_signal_scale_linear_log_20260709.md) | monotone AUC invariant |
| [signal mine batch](m_b1_signal_mine_batch_20260709.md) | catalog ranking |
| [gate coverage](m_b1_gate_coverage_7seq_20260709.md) | L0 map |
| [h_ratio signal](m_gate_h_ratio_signal_7seq_20260709.md) | single-gate depth |
| [bridge discriminability](m_b1_bridge_discriminability_20260709.md) | B1 bridge AUC |

### C — substrate / B2 sibling (separate line)

| Doc | Role |
|:--|:--|
| [B2 reconnect A/B](m_b2_reconnect_bridge_ab_20260709.md) | production-like B2 baseline for future A/B |

---

## Tools (stable; maintain code not prose)

| Tool | Task |
|:--|:--|
| `mine_relink_signals.py` | catalog |
| `energy_transform_separability.py` | transform audit |
| `combo_gate_safe_region.py` | 2D recoverability |
| `gt_safe_region_area.py` | GT-tail area |
| `gate_rule_search.py` + `AtomRepairConfig` | search + repair flags |
| `gate_rule_search_loo.py` | LOO |
| `loo_hurt_attribution.py` | hurt → repair compare |
| `weight_method_safe_region.py` | weighting plateau |
| `repaired_tail_or_safe_region.py` | freeze region thickness |
| `smoke_repaired_candidate_b2e2e.py` | offline smoke + B2 ref attach |

Full task→script map: [association_recovery_scripts_index](association_recovery_scripts_index_20260709.md).

---

## Next phase (not this hub’s job to implement)

```text
research-only default-off portable OR-tail hook
→ baseline B2 vs B2 + hook
→ e2e_safe_for_default_off: yes / no
→ preset still NO
```

Contract: [m_b1_portable_or_tail_hook_contract_20260709.md](m_b1_portable_or_tail_hook_contract_20260709.md)
