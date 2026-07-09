# Association recovery — information source contract

<!-- doc-status: research-synthesis -->
<!-- doc-promotion: not-for-report-citation-yet -->
<!-- doc-date: 2026-07-09 -->
<!-- doc-module: semantic -->

**Purpose:** define **where AssA / identity navigation data comes from**, who may
generate what, and what automation must never invent. This is a **governance
contract** for maps and future tooling — not a research report, not a baseline,
not sole active.

**Why this exists first:** one-off inventories (chat, scripts_index tables,
crosswalk paste) are useful snapshots. Without a source contract, a helper script
that hardcodes that snapshot becomes a **third truth** and drifts from disk,
presets, and research notes.

**Sibling maps (consumers of this contract, not superseding it):**

| Map | Role under this contract |
|:--|:--|
| [association_recovery_crosswalk_20260709.md](association_recovery_crosswalk_20260709.md) | Human-readable experiment alignment (may later be partially *rendered*) |
| [association_recovery_scripts_index_20260709.md](association_recovery_scripts_index_20260709.md) | Human-readable script lookup (may later be partially *rendered*) |
| [association_tools.yaml](association_tools.yaml) | **R** registry — **partial_p0** (Step 1B-1); Door A/D/F P0 tools + R-A/R-D/R-F; P1+ deferred |
| Checker / printer scripts (planned) | **Check / render / print only** — no research judgment (Step 2 / 3) |

---

## 0. One-line rule

```text
disk / preset / config  = mechanical truth (scannable)
registry yaml           = curated navigation metadata (human-owned)
research / no_go / ledger = verdicts & citable claims (human-owned)
scripts                 = check · render · print  (never invent verdicts)
```

If a field is not listed as **generated**, treat it as **manual** until this
contract is amended.

---

## 1. Source classes

| Class | What it is allowed to assert | Typical location |
|:--|:--|:--|
| **D — Disk** | Path exists; file size / line count; thin-wrapper redirect target (when parseable); mtime | repo tree under `scripts/`, `docs/`, `configs/` |
| **R — Registry** | Door A–F; role tags; fact-owner path; recipe id; expected artifact *names*; priority P0–P3; canonical vs wrapper intent | [association_tools.yaml](association_tools.yaml) — partial_p0 (Step 1B-1); P0 A/D/F only |
| **N — NO-GO master** | Whether a registry id **exists**; body of settled negative results | [no_go_registry.md](../../../reference/no_go_registry.md) (+ details) |
| **V — Verdict masters** | GO / NO-GO / parked / promotion / narrative mechanism | owning research note · module README GO table · ledger · TODO sole active |
| **C — Config truth** | Actual knob values for a named preset; schema defaults when no preset | `configs/presets/*.yaml` · `scripts/eval/config/*.py` |
| **M — Metric masters** | Absolute or citable numbers | [docs/TODO.md](../../../TODO.md) · [evidence_ledger.md](../../../research/evidence_ledger.md) · owning research note |
| **H — Human maps** | Prose synthesis, protocol, stack narrative | crosswalk · scripts_index · offline hub (until regenerated under rules below) |

Maps and scripts **must name their class** when they display a field
(e.g. “path OK (D)”, “door A (R)”, “#57 exists (N)”, “IDF1 → TODO (M)”).

---

## 2. Field ownership matrix

| Field / concern | Source of truth | Script may | Script must not |
|:--|:--|:--|:--|
| Path existence | **D** | report missing / present | invent paths |
| Line count / mtime | **D** | compute and print | use as quality score |
| Wrapper → target path | **D** parse of wrapper + **R** declared canonical | verify redirect matches **R** | invent wrapper status without **R** or parse |
| Door A–F assignment | **R** | read & group by door | infer door from filename alone as final truth |
| Role: canonical / wrapper / probe / peripheral / core-pipeline | **R** | read tags | reclassify by heuristics as authority |
| P0–P3 priority | **R** | sort/print | auto-promote/demote by mtime |
| fact-owner doc path | **R** | check file exists (**D**) | guess owner from imports or proximity |
| Recipe / 開工鏈 steps | **R** + human description in **R** or map | print commands | execute eval by default; decide success |
| Expected artifact *names* (e.g. `relink_candidates.csv`) | **R** | check presence under declared roots | treat missing artifact as NO-GO |
| NO-GO **id** listed on a tool | **R** cites id; **N** owns existence | check id anchor/row exists in **N** | invent id; change verdict text |
| NO-GO **verdict / mechanism** | **V** / **N** body | link to **N** | summarize as new master |
| GO / promotion / sole active | **V** (research note, README, TODO, ledger) | link; optionally “doc-status comment present” | auto GO/NO-GO |
| Preset knob values | **C** preset YAML | extract & show for named preset | hardcode 0.25 / 0.4 into script or map as master |
| Schema defaults (no preset) | **C** `scripts/eval/config/*.py` | extract & contrast with preset | conflate schema with production |
| Baseline / metric tables | **M** | point to path; optional “file exists” | copy numbers into yaml/map/script |
| Production stack narrative | **H** (crosswalk §0.5) human | render if later generated from fixed template | invent new stack stages |
| Offline discriminability conclusions | owning research note (**V**) | link | restate AUC tables as second master |

---

## 3. What may be **generated** vs **manual**

### 3.1 Generated fields (automation-safe)

Once a checker exists, these are the **only** fields it may *author* into a
report or regenerated section:

```text
path_exists          # D
line_count           # D (optional)
wrapper_target       # D parse (best-effort)
redirect_matches_R   # D vs R
fact_owner_exists    # R path checked on D
no_go_id_exists      # R cited id checked in N
preset_knob_snapshot # C extract for named presets (optional report)
missing_R_entry      # D script on disk but not in R (warn)
stale_R_entry        # R path not on D (warn)
```

Generated output should be labeled, e.g.:

```text
<!-- generated: association-tools-check; do not hand-edit -->
```

or a clearly titled “Checker report” section — never silently mixed with verdicts.

### 3.2 Manual fields (human-only authority)

```text
door                 # A–F
role / tags          # canonical, wrapper, probe, …
priority             # P0–P3
fact_owner           # which research note owns conclusions
recipe semantics     # why these steps; success criteria in prose
expected_artifacts   # which names matter for a recipe
no_go_ids (list)     # which ids are relevant (citation list)
verdict              # GO / NO-GO / parked / conditional
promotion            # ledger / default-on / split_feat_pr / …
recipe narrative     # “when to use R-A”
stack prose          # geometry-first mainline paragraph
any metric number    # always M or V
```

A script may **display** manual fields by reading **R** / **V** / **M**; it may
not **derive** them from disk heuristics as the system of record.

### 3.3 Forbidden as second masters

| Anti-pattern | Why |
|:--|:--|
| Script embeds full Door A–F tables from a one-off inventory | Becomes third truth vs **R** |
| Crosswalk hardcodes `bridge_px=0.25` without pointing at preset | Drifts from **C** |
| scripts_index copies IDF1 / AUC into recipe section | Drifts from **M** / **V** |
| Checker prints “likely NO-GO” from missing artifact | Confuses **D** with **V** |
| Chat paste promoted to registry without review | Bypasses **R** ownership |

---

## 4. Layer diagram

```text
┌─────────────────────────────────────────────────────────────┐
│  V / M / N — research notes · TODO · ledger · no_go_registry │
│       verdicts · promotion · citable numbers · NO-GO body    │
└────────────────────────────▲────────────────────────────────┘
                             │ cite / link only
┌────────────────────────────┴────────────────────────────────┐
│  R — association_tools.yaml (planned)                        │
│       door · role · fact-owner path · recipe · no_go id list │
└────────────────────────────▲────────────────────────────────┘
                             │ read metadata
┌────────────────────────────┴────────────────────────────────┐
│  Script layer — check / render / print                       │
│       never writes V/M; never invents R; may write reports   │
└───────────────┬─────────────────────────────┬───────────────┘
                │ scan                        │ extract
                ▼                             ▼
         D — disk                      C — preset YAML
         scripts/ docs/ …              + eval config schema
```

**Human maps** (crosswalk, scripts_index) sit beside **R**: either hand-maintained
under this contract, or later **rendered from R + generated checks**. They are
never the source for knob values or metrics.

---

## 5. Artifacts and step order

| Step | Artifact | Status | Allowed contents |
|:--|:--|:--|:--|
| **0** | This contract | **sealed** | process rules only |
| **1A** | [association_tools.yaml](association_tools.yaml) | **schema skeleton** | enums, field schemas, `tools: []`, `recipes: []`; no inventory fill |
| **1B-1** | same yaml, P0 fill | **partial_p0** | Door A/D/F P0 tools + R-A/R-D/R-F only; seed from scripts_index **H** |
| **1B** | same yaml, full inventory | not yet | remaining P1+ tools/recipes (**R** only) |
| **2** | Checker CLI | not yet | compare **R ↔ D ↔ N** (optional **C**); no verdicts |
| **3** | Optional MD render / recipe printer | not yet | print **R** + generated check footnotes; do not auto-run MOT eval |

**Step order is mandatory.**

```text
No checker before R schema exists.
No full inventory in Step 1A.
No render that invents doors not in R.
No script that writes verdicts / metrics / promotion.
```

Schema / field authority for **R** lives in the yaml file itself
(`tool_field_schema`, `recipe_field_schema`, `forbidden_fields`,
`registry_status: partial_p0` after Step 1B-1; was `schema_skeleton` at 1A).

Rules for **R** (unchanged intent):

- Every `path` must be intended to exist on **D** (checker enforces, Step 2).
- Every `fact_owner` must exist on **D** (warn if missing).
- Every `no_go_ids[]` entry must resolve in **N** (warn/fail).
- No `metrics:` block. No `verdict: GO`. No duplicated preset knobs.
- Step 1B-1 may populate P0 tools/recipes; remaining inventory stays empty until full 1B.

---

## 7. Checker (**Step 2**) non-goals

A compliant checker:

| Does | Does not |
|:--|:--|
| Fail if **R** path missing on **D** | Decide GO/NO-GO |
| Fail if **R** cites unknown no_go id | Rank tools by “usefulness” |
| Warn if disk script looks AssA-related but absent from **R** | Auto-add doors |
| Optionally dump preset vs schema knob diff (**C**) | Edit presets |
| Print recipe command skeletons from **R** | `mot17.py` full grid search |

---

## 8. Map maintenance under this contract

| Map | How to update (now) |
|:--|:--|
| crosswalk | Manual **H**; knobs must **point at** preset/schema paths (**C**), not invent masters |
| scripts_index | Manual **H** snapshot until Step 1B; seed for **R** fill, not master |
| offline hub | Fact-owner for Door A conclusions (**V**); hub rows may point to maps |
| association_tools.yaml | Step 1B-1 partial_p0 (A/D/F P0); remaining inventory by hand from scripts_index **H** |

After Step 3 (optional):

- Generated “path health” tables may be inserted under a marked section.
- Door/role columns must still come from **R**, not from the generator’s opinions.

---

## 9. Amendment process

1. Propose field ownership change in a PR that edits **this contract first**.  
2. Then update **R** schema / maps / checker in the same or follow-up PR.  
3. Do not “fix” by teaching the script a new heuristic that bypasses **R** or **V**.

---

## 10. Status of current maps (honest)

| Artifact | Status under contract |
|:--|:--|
| This contract | **Authoritative for process** (Step 0 sealed) |
| scripts_index (2026-07-09) | **Curated H snapshot**; seed for Step 1B; not yet **R**-backed |
| crosswalk | **Curated H**; knob table must stay subordinate to **C** |
| association_tools.yaml | **partial_p0** (Step 1B-1); Door A/D/F P0 tools + recipes; P1+ empty |
| Full inventory | **Not yet** (rest of Step 1B) |
| list_association_tools / checker | **Not yet** (Step 2) |

The 2026-07-09 inventory is **seed material for Step 1B**, not a substitute for **R**.

---

## 11. Related entry points

| Need | Go to |
|:--|:--|
| Experiment protocol / NO-GO levers | [association_recovery_crosswalk_20260709.md](association_recovery_crosswalk_20260709.md) |
| Script path lookup (current H) | [association_recovery_scripts_index_20260709.md](association_recovery_scripts_index_20260709.md) |
| Door A conclusions | [offline_relink_candidate_analysis.md](offline_relink_candidate_analysis.md) |
| Module sole active | [../TODO.md](../TODO.md) |
| WIP / D-level docs | [DEVELOPMENT.md](../../../../DEVELOPMENT.md) |
