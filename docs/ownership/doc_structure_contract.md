# Doc Structure Contract

**中文：** 研究 / 模組文檔結構契約  
**Status:** annotate-only (docs governance)  
**Series:** O-series companion (**O1.5**) — not tracker-decision P9  
**Companion:** [README.md](README.md) · [DOC_MAINTENANCE.md](../DOC_MAINTENANCE.md) · [change_routing_matrix.md](change_routing_matrix.md)

**Purpose:** one place for *where to write*, *how to index*, *when to archive*, and
*where numbers promote* — so research and module docs stay discoverable under WIP=1
and objective isolation.

**Non-goals:** mass file moves, runtime/default flips, full historical backfill of
frontmatter, topic-hub pages (optional later).

---

## C0 — Homes (layering)

```text
[治理]           docs/ownership/                    O-series · WIP=1 · this contract
[跨模組研究]     docs/research/                     cross-module experiments, decision semantics, global eval/training
[模組]           docs/modules/<m>/                  module card + design + module research
[可引用資產]     report_data/                       paper-rebuild tables/figures + method thesis assets
[歷史]           docs/archive/                      closed one-shots; not current direction
[數字 master]    docs/research/evidence_ledger.md   citable decision / baseline rows
[負結果 master]  docs/reference/no_go_registry.md   default-off / NO-GO register
```

Rules of thumb:

- **One home per note.** Do not duplicate long reports across `modules/` and `docs/research/`.
- **Pointers are free; second truths are not.** Entry READMEs link; they do not restate full metrics without promotion rules (C5).

---

## C1 — Writing decision tree

| I did… | Home | Must also… |
|:--|:--|:--|
| Single-module experiment / ablation | `docs/modules/<m>/research/` | Index row in parent module `README.md`; numbers must be source-traceable |
| Cross-module / global eval / pipeline / shared training | `docs/research/<area>/` | Index row in subdir README **or** top `docs/research/README.md` |
| Decision-layer *why* (association / gates / knobs) | `docs/research/tracker-decision/` | **Closed line is read-only** (P0–P8); reopen only as a *named* new line with evidence |
| Citable baseline / decision outcome number | `docs/research/evidence_ledger.md` | One ledger row + link to source doc; no chat-only numbers |
| Deep **per-signal / per-gate** analysis progress | `docs/research/eval/signal_analysis_ledger.md` | One row per `signal_id`; numbers master = `out/signal_study/`; contract = `signal_table_schema` |
| Paper claim / rebuildable tables & figures | `report_data/` | Link from `source_map.md` or `report_data/README.md` back to research |
| Finished one-shot / abandoned design | `docs/archive/` | Drop from “active” indexes or mark historical |
| Stable module design | `docs/modules/<m>/architecture*.md` (or detection top-level protocol docs) | Link from module README “設計入口” |
| Ops steps | `docs/modules/<m>/runbooks/` or `docs/reference/runbooks/` | Link from module / reference README |
| Cheb-GR / bank / offline identity / occ-exit | **`docs/modules/semantic/`** | Doc home is semantic even if code lives under `perception/reid/` |
| Feature extract / bank implementation (not association policy) | `docs/modules/reid/` | Keep association / handover policy out |

Also see the shorter tree in [docs/README.md](../README.md).

---

## C2 — Module package schema

Every `docs/modules/<m>/` **must** have:

```text
README.md
TODO.md
```

Optional packages — **README must say whether each exists**:

```text
architecture*.md | research/ | runbooks/
```

### README four blocks

1. **Card** — 職責 / 現況 / I/O / GO·NO-GO (existing template).
2. **Design entry** — `architecture*` and/or “see tracker-decision / ADR …”.
3. **Research index** (if `research/` exists) — **list every** `research/*.md` with status; put `🔄 Active` first.
4. **TODO link** — `TODO.md` only.

### Detection exception

`docs/modules/detection/README.md` may keep **index library + module card** dual structure.
Label the dual layout at the top of that README (“detection 特例”).

### TODO rules (with C7)

- WIP=1: at most one unchecked sole active (see [DOC_MAINTENANCE § WIP](../DOC_MAINTENANCE.md)).
- If no active work: explicit `⏸️` or one line “無 active”.
- TODO holds **one-line** active + checkboxes + **links** to research files — not full reports.

---

## C3 — Research note status markers (new notes required)

**Style: HTML comments** (same family as fact-owner; easy to grep).

Place near the top of each new note under:

- `docs/modules/*/research/*.md`
- `docs/research/{pipeline,eval,training,reid}/*.md`
- other dated research notes under `docs/research/` as applicable

```html
<!-- doc-status: active | parked | closed | archived -->
<!-- doc-promotion: none | ledger | report_data | archive | no_go -->
<!-- doc-date: YYYY-MM-DD -->
```

Optional:

```html
<!-- doc-module: semantic | detection | cross | … -->
```

| Field | Meaning |
|:--|:--|
| `doc-status` | Lifecycle (C6) |
| `doc-promotion` | Where outcomes must land if cited outside the note |
| `doc-date` | Note date (not a freshness proof for entry docs) |

**Legacy notes:** backfill gradually; do not block merges solely on missing markers on old files.
**New notes (after this contract):** required in the same PR as the note.

---

## C4 — Index freshness (entry = catalog)

| Entry | Obligation |
|:--|:--|
| `docs/research/README.md` | Active workstreams (including **pointers** into module research), Closed lines, Paper → `report_data`, subdir entry points; **no phantom paths** |
| `docs/research/<sub>/README.md` | Index **all** `.md` in that subdir (except the README itself), **or** state “no index; filenames only” and do not claim a table elsewhere |
| `docs/modules/<m>/README.md` | If `research/` exists, index **all** research notes |
| `report_data/README.md` | Start-here list; **one-line** link to decision paper outline |
| `docs/research/paper_outline.md` | Links to `evidence_ledger.md` **and** `report_data/README.md` |

**Same-PR rule:** adding a research file **must** add the matching README index row.

Checker: `scripts/tools/check_doc_structure.py` (warn-only) flags research notes not mentioned by the owning README.

---

## C5 — Evidence & promotion

```text
Research body
  ├─ cited as decision / production baseline  → evidence_ledger row + source link
  ├─ paper claim / rebuildable table-figure → report_data (+ source_map)
  ├─ negative / default-off outcome         → no_go_registry (+ optional ledger decision row)
  └─ engineering-only process                → stay in research; promotion: none
```

### Dual paper lines (must cross-link; do not overwrite each other)

| Line | Owner | Use |
|:--|:--|:--|
| Decision / production narrative | `docs/research/paper_outline.md` + `evidence_ledger.md` | Geometry-first + whole-graph engineering story |
| Mamba method paper assets | `report_data/paper_direction.md` + `tables/` | Curriculum / detector thesis |

Numbers: each line keeps its own master. Entry docs that quote baselines still follow fact-owner for `current-baseline`.

---

## C6 — Lifecycle

| status | Meaning | Entry behavior |
|:--|:--|:--|
| `active` | In progress; should align with module sole active or a named cross-module line | README Active section |
| `parked` | Intentionally paused | Parked section; does not consume WIP |
| `closed` | Done but still citable | May stay in place; mark closed; ledger / no_go as needed |
| `archived` | One-shot / not current | Move to `docs/archive/` or archive index only |

**Closed decision line:** [tracker-decision/status_2026-07-09.md](../research/tracker-decision/status_2026-07-09.md) (P0–P8) is read-only; no drive-by reopen (WIP rules).

---

## C7 — TODO vs research role split

| Artifact | Role |
|:--|:--|
| `TODO.md` | Sole active one-liner, checkboxes, links |
| `research/*.md` | Full method, commands, tables, conclusions |
| `evidence_ledger.md` / `report_data/` | Citable aggregates |

**Do not** paste full closed-out reports into `TODO.md` for new work. Historical dense TODOs may remain until cleaned; new closes write a research file first.

---

## C8 — Relationship to O0 / O1

| Layer | Owns |
|:--|:--|
| **O0 WIP=1** | At most one concurrent active goal per module owner |
| **O1 objectives** | Primary / should-not-own; RESEARCH PRs must not flip production defaults ([routing](change_routing_matrix.md)) |
| **This contract** | File homes, index discoverability, promotion paths |

Not P9; not dual-stability reopen.

---

## C9 — Enforcement

| Layer | Mechanism |
|:--|:--|
| Human | [DOC_MAINTENANCE PR checklist](../DOC_MAINTENANCE.md) — index row, promotion, no phantom paths |
| Machine (existing) | `check_doc_links.py` hard · `check_doc_stale_paths.py` hard · `check_doc_freshness.py` warn |
| Machine (this contract) | `check_doc_structure.py` **warn-only** — research file not referenced by owning README |

```bash
uv run python3 scripts/tools/check_doc_structure.py
uv run python3 scripts/tools/check_doc_links.py
uv run python3 scripts/tools/check_doc_stale_paths.py
```

---

## Minimal good examples (after contract landing)

- Active module research discoverable from `docs/research/README.md` → module README → note (≤3 hops).
- Geometry module README links design + research + tracker-decision.
- `paper_outline.md` ↔ `report_data/README.md` mutual links.

## Explicit backlog (not blocking this contract)

- Full frontmatter backfill on pre-contract notes
- Semantic `TODO.md` slim-down (move prose into research files)
- Topic hubs (relink / GMC / whole_graph)
- Tighten `check_doc_structure.py` to `--strict` after index debt is paid down

---

## One-liner

> Every research note has one home, one index row, and a promotion path — so WIP=1 work stays findable without a second truth.
