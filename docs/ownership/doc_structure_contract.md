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

歷史上 C6 只寫了 threads 的 close 協議，research note 拿到了 `doc-status` 欄位卻**沒有轉移協議**：
closed 的 note 不搬家、不離開索引、沒有觸發條件、checker 不擋。結果是 note 只增不減——
**那是定義域的洞，不是紀律問題。** 本節補完它。

| status | Meaning | Entry behavior |
|:--|:--|:--|
| `proposed` | Spec / mother-line written; not started or not authorized as sole active | Proposed section (threads); does not consume WIP |
| `active` | In progress; should align with module sole active or a named cross-module line | README Active section |
| `parked` | Intentionally paused | Parked section; does not consume WIP |
| `closed` | Done but still citable as navigation | **Move** into the owner's `closed/`; Closed index row + `closed:` date; ledger / no_go only if claims promote |
| `archived` | One-shot / not current | Move to `docs/archive/` or archive index only |

### C6.1 — 檔名承擔語義；目錄承擔生命週期

- **檔名**：穩定的 `<object|unit>_<YYYYMMDD>.md` 語義名。**不得**把 terminal / 狀態寫進檔名
  （否則收單時要改名，連結全斷）。檔名是**抽象語意連接**——看名字就知道它屬於哪個 object。
- **目錄**：`<home>/` = active；`<home>/closed/` = 已收單；`docs/archive/` = 一次性 / 不再是方向。
  **狀態的可見性由目錄表達，不由檔名。**

### C6.2 — 細節與總結解耦（且細節**不可改寫**）

一個研究單元收單後，**只留三種東西**：

```text
state          → registry 一列（無散文、無數字）
terminal record→ 恰好一份高密度總結：裁決 / 範圍 / 限制 / 指回證據
detail         → declaration + evidence packet + 中間 note:內容 byte 不變,移出 active 視野
```

**硬約束：** sealed declaration 與 evidence packet **不得**為了「整理」而被壓縮、改寫或合併。
封印的價值就在於「封的時候寫了什麼，事後不能改」。**「封存細節」只能是索引層與目錄層的動作**
（移出視野、留一行 pointer），**不能是內容層的動作**。為了乾淨而砍掉稽核性，是把成本轉嫁給未來。

### C6.3 — 觸發器：研究狀態轉移**驅動**文檔收單（這是本契約唯一有牙齒的地方）

收單不是自律，是**合入條件**。觸發事件在研究層，執行動作在文檔層：

```text
觸發:  owner 接受一個 terminal（registry 的 state transition）
         │
         ▼
同一個 PR 內必須完成（缺一項 = pre_push 紅燈）:
  1. registry 對應 object 的 state / last_transition 更新
  2. 恰好一份 terminal record（高密度;不複述統計 → 指回 declaration）
  3. 該單元的 note / declaration `git mv` 進 owner 的 closed/（內容不變）
  4. 從所有 active 索引移除,只留一行 pointer
  5. 若該單元有 thread → thread 移入 threads/closed/（既有協議）
```

**不得**：在 owner 接受 terminal 的 PR 裡「之後再整理」。之後不會來——這正是 doc 只增不減的機制。

**Threads close protocol (summary):** same change updates (1) thread frontmatter `doc-status: closed` + `closed: YYYY-MM-DD`, (2) body `Final status` / terminal + History close line, (3) **`git mv` into `docs/research/threads/closed/`** + fix relative links / repo pointers, (4) `docs/research/threads/README.md` Closed row + `closed/README.md` index. Do not delete the card; do not leave closed cards in `threads/` root. Full checklist: [threads/README.md § How to close](../research/threads/README.md).

### C6.4 — Enforcement（fail-closed）

`scripts/tools/check_doc_structure.py --strict` 對以下情形**紅燈**（其餘維持 warn-only）：

| 規則 | 條件 |
|:--|:--|
| **L1 closed 必須搬家** | `doc-status: closed` 的 note 仍在 active 路徑（不在 `closed/` 或 `archive/` 下） |
| **L2 closed 不得佔用 active 索引** | closed note 仍被 owning README 的 active 區塊索引 |
| **L3 決策層不得長 prose** | `docs/research/contracts/` 下新增非契約 prose 檔（C0.1） |

pre_push 以 `--strict` 執行。舊有未遷移的 note **豁免**（allowlist），新違規一律擋——
**回填是清潔工作，不阻擋主線；但不得再製造新的違規。**

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
