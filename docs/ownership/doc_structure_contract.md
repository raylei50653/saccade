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
[研究規範]       docs/research/contracts/           method / evidence semantics / claim ladders（規則）
[研究狀態]       docs/research/contracts/claim_state_registry.md   每個 object 現在站在哪一格（狀態）
[跨模組研究]     docs/research/                     cross-module experiments, decision semantics, global eval/training
[任務母線]       docs/research/threads/             navigation-only continuous-task cards (not evidence)
[模組]           docs/modules/<m>/                  module card + design + module research
[可引用資產]     report_data/                       paper-rebuild tables/figures + method thesis assets
[歷史]           docs/archive/                      closed one-shots; not current direction
[數字 master]    docs/research/evidence_ledger.md   citable decision / baseline rows
[負結果 master]  docs/reference/no_go_registry.md   default-off / NO-GO register
```

Rules of thumb:

- **One home per note.** Do not duplicate long reports across `modules/` and `docs/research/`.
- **Pointers are free; second truths are not.** Entry READMEs link; they do not restate full metrics without promotion rules (C5).

### C0.1 — 決策層不出文檔（decision layer carries state, not prose）

**決策層只有兩個 artifact，兩者都是狀態，不是散文：**

| 決策層 artifact | 內容 | 明確**不是** |
|:--|:--|:--|
| [claim_state_registry](../research/contracts/claim_state_registry.md) | 每個 research object 的當前狀態 + 合法候選集 | 證據、統計理由、數字 |
| `docs/modules/<m>/TODO.md` | WIP=1 鎖（sole active 一行 + link） | 任務敘事、進度報告 |

**規則：** 決策層**不得**新增 prose 檔。要解釋 → research note；要導覽 → thread；要規則 → `contracts/`。
`DEVELOPMENT.md` 與各 README 對決策層**只做投影**（公告選擇結果），**不得**重述任何 object 的
rung / limits / substrate — 那是 registry 的 fact-ownership（C5 的「不得有第二真相」在狀態上的推論）。

---

## C1 — Writing decision tree

| I did… | Home | Must also… |
|:--|:--|:--|
| Single-module experiment / ablation | `docs/modules/<m>/research/` | Index row in parent module `README.md`; numbers must be source-traceable |
| Multi-home / multi-step research chain (≥2 homes or ≥3 steps or citable policy/hook/audit) | `docs/research/threads/` | Navigation card only; index in `threads/README.md`; **no** long tables / no second evidence home |
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

- WIP=1: at most one sole active (see [DOC_MAINTENANCE § WIP](../DOC_MAINTENANCE.md)).
- If no active work: explicit `⏸️` or one line “無 active”.
- TODO is a **WIP register only**: sole active one-liner + link(s) to thread card and/or research note; optional parked one-liners.
- **No** long prose, metrics, command dumps, or closed-report bodies in TODO (those live in research / threads / ledger).

---

## C3 — Research note status markers (new notes required)

**Style: HTML comments** (same family as fact-owner; easy to grep).

Place near the top of each new note under:

- `docs/modules/*/research/*.md`
- `docs/research/{pipeline,eval,training,reid}/*.md`
- other dated research notes under `docs/research/` as applicable

```html
<!-- doc-status: proposed | active | parked | closed | archived -->
<!-- doc-promotion: none | ledger | report_data | archive | no_go -->
<!-- doc-date: YYYY-MM-DD -->
```

Optional:

```html
<!-- doc-module: semantic | detection | cross | … -->
```

Thread cards under `docs/research/threads/` may use YAML frontmatter instead of HTML comments; same status vocabulary. Closed threads also set `closed: YYYY-MM-DD` (optional `closed-verdict`).

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
| `docs/research/README.md` | Active workstreams (including **pointers** into module research), Closed lines, Paper → `report_data`, subdir entry points（含 `threads/`）; **no phantom paths** |
| `docs/research/threads/README.md` | Index **all** thread cards by lifecycle table (Proposed / Active / Parked / Closed); closed cards live under `threads/closed/`; close = move + row + frontmatter; navigation-only |
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

## C6 — Lifecycle（適用**所有** doc class，不只 threads）

歷史上 C6 只寫了 threads 的 close 協議：research note 拿到 `doc-status` 欄位卻**沒有轉移**——
closed 的 note 不搬家、不離開索引、沒有觸發條件。note 只增不減**是這個機制的產物，不是紀律問題**。

| status | Meaning | Entry behavior |
|:--|:--|:--|
| `proposed` | Spec / mother-line written; not started or not authorized as sole active | Proposed section (threads); does not consume WIP |
| `active` | In progress; should align with module sole active or a named cross-module line | README Active section |
| `parked` | Intentionally paused | Parked section; does not consume WIP |
| `closed` | Done but still citable as navigation | **Move** into the owner's `closed/`; Closed index row + `closed:` date; ledger / no_go only if claims promote |
| `archived` | One-shot / not current | Move to `docs/archive/` or archive index only |

### 關閉一個研究單元：三條規則（沒有第四條）

**1 · 關閉必須產出一份高密度結論。**
說清楚**裁決**、**適用範圍**、**限制**、**證據在哪裡**。一份，不是三份。

**2 · 細節退出 active 視野，但內容不改。**
封存 ＝ 移出索引、移入 `closed/`／`archive/`、降低可見性。
**不需要**也**不得**重寫、壓縮、合併——sealed declaration 與 evidence packet 的價值就是
「封的時候寫了什麼，事後不能改」。整理只動**位置與可見性**，不動**內容**。

**3 · 關閉流程必須同時完成整理。**
**不得**先宣布實驗關閉、之後再開一個「整理文檔」任務。之後不會來——那正是 doc 只增不減的原因。
owner 接受 terminal 的**同一個 PR** 內：結論就位 → 細節搬家 → 移出 active 索引 →
[registry](../research/contracts/claim_state_registry.md) 狀態更新。

**實作細節（不是規則）：** 檔名保持穩定語義、不寫 terminal（否則收單改名會斷連結）；
生命週期由**目錄**表達。thread 的既有 close checklist：
[threads/README.md § How to close](../research/threads/README.md)。

### Enforcement

`check_doc_structure.py --strict`（pre_push 執行）**紅燈**：

| | 擋什麼 | 對應 |
|:--|:--|:--|
| **L1** | `doc-status: closed` 卻仍在 active 路徑 | 規則 2 · 3 |
| **L2** | closed note 仍佔用 owning README 的 Active 區塊 | 規則 2 · 3 |
| **L3** | 決策層（`research/contracts/`）長出 prose | C0.1（**與這三條無關**，是另一條規則） |
| **L4** | thread 的 `wip-role` 與 threads 索引列不一致 | C5.1（投影不得與 owner 矛盾） |

**規則 1 不機械化**：結論夠不夠高密度，checker 判不了——它由 review 擋。假的牙齒比沒有牙齒更糟。
既有 7 份 closed-in-active note 已 allowlist：**回填是清潔工作，不阻擋主線；但新違規一律擋。**

### C5.1 — 狀態只有一個寫入者（推論自 C5）

**狀態只能寫在它的 owner；其他表面只能連結或投影，不得複述。**

| 狀態 | 唯一寫入者 |
|:--|:--|
| 研究對象的 rung / substrate / limits / 候選集 | [claim_state_registry](../research/contracts/claim_state_registry.md) |
| 模組的 sole active（WIP 鎖） | `docs/modules/<m>/TODO.md` |
| thread 自身的 wip-role | thread frontmatter（索引列只是投影，**L4 檢查一致性**） |

**手寫的投影必然漂移**——實測：`DEVELOPMENT.md` dashboard 與 `research/README.md` 在被發現時都還停在
一個**已關閉**的單元上。因此索引與入口表只列「這是什麼 + 去哪裡」：**不列裁決、不列數字、不列狀態**。
module README 的 research 索引尤其如此（它一度整段抄錄裁決與指標，那是 C5 違規）。

**Closed decision line:** [tracker-decision/status_2026-07-09.md](../research/tracker-decision/status_2026-07-09.md) (P0–P8) is read-only; no drive-by reopen (WIP rules).

---

## C7 — TODO vs research / threads role split

```text
TODO.md              = WIP lock / active pointer   (not task narrative)
docs/research/threads/ = continuous-task mother line (navigation-only)
module|global research = facts, methods, commands, tables
ledger/report_data/no_go = promoted formal facts
conversation hook    = short-lived handoff (prefer promote into a thread)
```

| Artifact | Role | Must not |
|:--|:--|:--|
| `TODO.md` | WIP=1 lock: sole active **one-liner** + links to thread and/or research; parked one-liners; explicit `⏸️` / 無 active when idle | Long prose, metrics tables, reasoning logs, full closed reports |
| `docs/research/threads/` | Multi-step / multi-home **navigation** card | Evidence tables / second truth |
| `research/*.md` | Full method, commands, tables, conclusions | Act as WIP lock |
| `evidence_ledger.md` / `report_data/` / `no_go_registry` | Citable / formal aggregates | |

**TODO template (target ≤ ~20 lines):**

```md
## Sole active
🔄 <one-liner>
- Thread: docs/research/threads/<card>.md   # if multi-step
- Canonical: docs/modules/<m>/research/<note>.md

## Parked
- <one-liner> → link

## Done / closed
See research index / no_go / evidence_ledger.   # no embedded reports
```

**Do not** paste full closed-out reports into `TODO.md`. New closes write a research file (or ledger/no_go row) first; TODO only points.

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
- Optional `check_doc_structure` warn: TODO overlong / missing sole-active or ⏸️ / missing link
- Topic hubs (relink / GMC / whole_graph)
- Tighten `check_doc_structure.py` to `--strict` after index debt is paid down

---

## One-liner

> Every research note has one home, one index row, and a promotion path — so WIP=1 work stays findable without a second truth.
