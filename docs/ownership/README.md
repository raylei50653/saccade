# O-series: Ownership / Objective Isolation

**中文：** 模組目標隔離階段  
**Status:** O0 sealed on `main` · **O1 Module Objective Map** (this tree)  
**Not this series:** tracker-decision P0–P8 (closed), dual-stability reopen, “P9”

---

## Why this exists

Large modules hurt review not only because of LOC, but because **one owner / file
carries multiple competing objectives** at once:

```text
RUNTIME  CORRECTNESS  PERF  CONFIG  RESEARCH  BRIDGE  DEBUG  LEGACY
```

Reviewers then judge **multiple responsibility lines**, not just “which file changed.”

O-series isolates objectives so each module (and, where needed, each hot file)
has **one primary job**, explicit secondaries, and a clear **should-not-own** list.

---

## Phase map

| Phase | Name | Deliverable | Behavior? |
|:--|:--|:--|:--|
| **O0** | Workstream WIP Seal | WIP=1 per module owner | **No** (docs) — [DOC_MAINTENANCE § Workstream WIP](../DOC_MAINTENANCE.md) |
| **O1** | Module Objective Map | this directory | **No** (annotate only) |
| **O1.5** | Doc Structure Contract | [doc_structure_contract.md](doc_structure_contract.md) | **No** (homes / indexes / promotion) |
| **O2** | Ownership notes for top modules | deeper notes per module | No |
| **O3** | Low-risk extraction plan | ordered extract plan | No |
| **O4** | First mechanical split | code move | Yes — only with checker/smoke |

**O0 entry / dashboard:** [DEVELOPMENT.md 模組現狀總覽](../../DEVELOPMENT.md)  
**Dev entry (need levels D0–D4):** [DEVELOPMENT.md](../../DEVELOPMENT.md)  
**Closed decision line (read-only):** [tracker-decision/status_2026-07-09.md](../research/tracker-decision/status_2026-07-09.md)  
**Doc homes / research indexes:** [doc_structure_contract.md](doc_structure_contract.md)

---

## O1 files

| File | Role |
|:--|:--|
| [objective_template.md](objective_template.md) | Objective type catalog + card schema |
| [module_objective_map.md](module_objective_map.md) | Primary / secondary / should-not-own per module & hot file |
| [change_routing_matrix.md](change_routing_matrix.md) | Objective touched → required checks |
| [extraction_candidates.md](extraction_candidates.md) | What to extract later (reasons only; no moves) |
| [doc_structure_contract.md](doc_structure_contract.md) | **O1.5** write-where / index / promotion / lifecycle |

---

## O1 completion definition

- [x] Objective types defined  
- [x] Large modules / hot files have primary + should-not-own  
- [x] Change routing by objective  
- [x] Extraction candidates with rationale  

### O1 invariants (must hold for this PR)

- [x] **No** behavior change, code movement, runtime/default flip

---

## Non-goals (O1)

- Not P9; not continuation of tracker-decision phases  
- Do not split `evaluator.py` / `stages.py` / `tracker_gpu.cu` in this PR  
- Do not “inventory while editing runtime”  
- LOC is **not** the only criterion (objective coupling is)  
- Paper / evidence workstreams stay separate from tracker behavior lines  

---

## How to use in review

1. Identify **which objective** the PR primarily serves (use routing matrix).  
2. Check the file’s card in [module_objective_map.md](module_objective_map.md): does the PR expand a **should-not-own** surface?  
3. Require checks from [change_routing_matrix.md](change_routing_matrix.md).  
4. If the PR grows a second objective on the same owner, apply **O0 WIP=1** (park or finish first).  

---

## One-liner

> Module-objective isolation: each module owns **one** primary goal so runtime, perf, research, config, debug, and legacy do not pull the same owner at once.
