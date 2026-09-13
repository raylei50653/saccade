# Phase 1 Classification Protocol — Module Doc Question Map

**中文：** `docs/modules/` 問題導向重整（#368）Phase 1 執行協議——只分類，不綜合  
**Status:** execution authority for Phase 1（由「加入本檔的 commit」pin 住；見 § Protocol pin）  
**Issue:** [#368](https://github.com/raylei50653/saccade/issues/368) — Phase 2 direction 由 issue body 擁有，本檔**不**重述  
**Provenance:** materialized from the reviewed Phase 1 protocol comment (2026-09-10, [#issuecomment-5615730650](https://github.com/raylei50653/saccade/issues/368#issuecomment-5615730650)); that comment is downgraded by this file to navigation / historical context. The activation gate requiring this materialization is [#issuecomment-5616701795](https://github.com/raylei50653/saccade/issues/368#issuecomment-5616701795).  
**Home:** `docs/ownership/`, not `docs/modules/` — this governs *how module docs are classified*; it answers no module's durable question, and filing it under `docs/modules/` would create exactly the extra first-class module document that #368 acceptance criterion 5 exists to prevent (C0.1).  
**Related:** [doc_structure_contract.md](doc_structure_contract.md) (O1.5) · [DOC_MAINTENANCE.md](../DOC_MAINTENANCE.md) · [README.md](README.md)

**Purpose:** produce a reviewed mapping from every in-scope module document to the durable question it answers — a new axis that no current owner provides — **without** synthesizing the subsystem's current answer.

**Guiding principle (from the issue):** module documentation should optimize for *"what can I learn / decide from here now?"*, not *"what work has been recorded here over time?"*.

---

## Protocol pin

- Execution pins the commit that adds this file. Step 0 records that SHA as `protocol_commit`; classification must not begin before it is pinned.
- The committed file is the execution authority. Later edits to this file do not retroactively change what a pinned run executed; a re-run is a new pinned execution.
- The original GitHub protocol comment is navigation / historical context only.

---

## Goal

```text
document
→ question / knowledge domain   (new axis — no owner exists today)
→ lifecycle marker              (copied verbatim from the existing owner)
→ confidence
→ unresolved classification questions
```

The question axis reflects **what the document actually answers**, not which task, phase, experiment, or historical research line produced it.

---

## Step 0 — freeze the scope (before batch 1)

"All in-scope documents" is unfalsifiable without a fixed list, so Phase 1 opens by producing one. Copy the SNAPSHOT contract from [`doc_migration_manifest.yaml`](doc_migration_manifest.yaml): the consumer reads a frozen `resolved_files` list; the glob is only a regeneration recipe, and files added later do not join by inheritance.

```bash
git rev-parse HEAD                       # pin
find docs/modules -name '*.md' -not -path '*/evidence/*' | sort
```

Draft-time baseline, measured at `953f0e56`: **121** human-facing `.md` under `docs/modules/`, of which `semantic/` = **70** (every other module 2–16). The live tree drifts, so the rule is: **re-measure at the pinned protocol commit, re-pin, and re-freeze; if a count differs from this baseline, record the re-pin rather than proceeding on the stale expectation.**

Scope is **all of `docs/modules/`**, not `semantic/` alone — the issue's acceptance criteria are module-wide, and the remaining modules are ~50 files combined. Batch order runs `semantic/` first because it is the failure case that motivates the work.

---

## Step 1 — the two axes, and which one is new

### Axis A — lifecycle / role: already owned. Copy, never re-derive.

`doc-status` exists today (contract C3 vocabulary: `proposed | active | parked | closed | archived`), and C5.1 gives state exactly one writer, with "Link, don't relabel" requiring a projection to copy the owner's self-naming. Role-like values — *terminal result*, *failed / withdrawn line*, *research declaration*, *historical context* — are **state** claims. A separate role vocabulary would become a second truth about lifecycle, so none is used.

Instead, each record carries the marker **verbatim** as found, plus where it was found:

```yaml
lifecycle_marker: sealed-for-execution   # verbatim; null if absent
lifecycle_marker_carrier: html | frontmatter | none
lifecycle_marker_in_c3_vocabulary: false
```

### Axis B — durable question / knowledge domain: genuinely unowned. This is what Phase 1 adds.

Provisional categories: module responsibility / architecture · association behavior · candidate generation / observability · ranking / scoring · relink / handover · ReID / semantic identity signal · motion information · runtime / proxy fidelity · production substrate · safe-domain / operating envelope · qualification / verification · research methodology / declaration · historical execution record · failure / withdrawal · index / router · other.

Provisional means provisional: if full reading shows the taxonomy is wrong, record proposed additions / splits / merges rather than forcing documents into a bad list. One primary category, zero or more secondary.

---

## Step 2 — marker drift is a Phase 1 output, not a nuisance to route around

Measured in `semantic/` (70 human-facing docs) at `953f0e56`:

| | |
|---|---|
| carry a marker | 58 (**12 carry none**) |
| observed values | `active` 35 · `closed` 9 · `draft` 5 · `proposed` 4 · `research-synthesis` 3 · `sealed-for-execution` 1 · `sealed-execution` 1 |
| outside the C3 vocabulary | **10 / 58** |
| C3 values never used | `parked`, `archived` |
| carriers in use | 48 HTML comment / 10 YAML frontmatter |

Adding a fresh 12-value role vocabulary on top of an unenforced 5-value one is how a third taxonomy is born. Phase 1 therefore reports the same table for the full frozen scope as a deliverable, and proposes reconciliation — it does **not** fix markers in flight (that is a state edit, and state has an owner).

---

## Required reading rule

Every assigned document must be read before it is classified. Do not classify from: filename, title, README entry, grep/search snippet, task identity, phase name, labels such as D0 / H0 / H2 / GCTM / M-B1, or another document's short description of it. If classification depends on directly referenced context, read that context too.

**Carve-out (C5.1).** This rule governs Axis B only. It does not override fact ownership: where a state fact already has a single writer (`doc-status` marker, thread frontmatter, `docs/research/contracts/claim_state_registry.md`), the owner's self-naming is authoritative and is copied verbatim, never re-derived from the body — even when the body appears to disagree. A body/owner disagreement is recorded as a question, not resolved by the classifier.

---

## Boundary

Reading is used to verify whether the classification is correct. Do **not** start synthesis in this phase: no deciding the subsystem's current answer, no rewriting architecture, no merging research lines, no proposing canonical contents, no declaring supersession unless classification requires it, no turning a batch into a research summary, no moving files, no editing markers.

---

## Confidence scale (anchored)

| value | meaning |
|:--|:--|
| `high` | The primary category follows from the document body alone; no unread document could change it. |
| `medium` | The body supports the category, but a **named** referenced document could plausibly narrow or move it. Name it in `questions`. |
| `low` | Classification rests on context not yet read, or the document genuinely spans domains with no clear primary. Always pairs with a blocking question. |

Unanchored `high/medium/low` is not comparable across batches; a record whose confidence is not justified by this table is not accepted.

---

## Questions and ambiguities

Uncertainty is recorded explicitly, never silently resolved by guessing. Per question: file · question · uncertainty type · why it matters · blocking or non-blocking · related files needed to resolve it.

**Blocking** = prevents reliable classification (e.g. unclear whether the doc is about runtime fidelity or only offline proxy behavior; scope cannot be established without another referenced result; task name suggests one domain but the body supports another).

**Non-blocking** = classification is reliable but the uncertainty matters later (possible supersession, unclear authority relationship, scope later narrowed, unknown whether a terminal result remained current, possible conflict with a later document).

**Resolution owner.** The classifier may resolve a blocking question only by reading more in-scope documents. Any question that turns on **state** — supersession, authority, whether a terminal still holds — is registry-owned and goes to the repo owner for adjudication; the classifier may not settle it, and the affected document stays explicitly unverified until it is settled. A blocking question with no reader-resolvable path is escalated in the batch report rather than absorbed.

---

## Per-document record

```yaml
file: docs/modules/semantic/research/<doc>.md
primary_category: runtime_proxy_fidelity
secondary_categories:
  - observability
lifecycle_marker: active            # verbatim from the doc; null if absent
lifecycle_marker_carrier: html
lifecycle_marker_in_c3_vocabulary: true
classification_confidence: high
classification_basis:
  - short reason grounded in the document body
  - enough detail to show why the category is correct
  - why an obvious alternative category was rejected, if useful
questions:
  - type: possible_supersession
    blocking: false
    owner: registry            # registry | classifier
    question: >
      Does the later reconciled map formally replace this framing?
    why_it_matters: >
      Affects later current-answer / historical-context analysis,
      but not the present classification.
```

`classification_basis` justifies classification only; it is not a document summary, and it must not copy verdicts, metrics or accepted state (C5.1).

---

## Where the output lives

One machine-consumed YAML: **`docs/ownership/module_doc_question_map.yaml`**, following the precedent and the stated properties of `doc_migration_manifest.yaml` — EPHEMERAL (drained and deleted once Phase 2 promotes it; it must not grow into a fourth archive) and MACHINE-CONSUMED (not human prose).

It must **not** be a hand-written note under `docs/modules/`: acceptance criterion 5 in the issue body exists precisely to stop finished work from spawning another first-class module document, and Phase 1 would otherwise be the first violation of the rule it is meant to enable.

Two mechanical consequences of adding it: adding any `.md`/`.yaml` under `docs/` requires regenerating [`master_map.generated.md`](master_map.generated.md) or `tests/contract/test_migration_manifest_v0.py::test_checked_in_master_map_is_current` fails closed; and the file must not restate verdicts or state, per its own precedent's rule (「只記機械需要的事實，不存分類、不複寫 verdict」) — Axis A is a verbatim copy and Axis B is not a verdict, so both stay on the right side of that line.

Whether the map is later promoted into per-document self-doc headers plus a generated index — the shape already used by `scripts_inventory.generated.md` and `tests_inventory.generated.md` — is a **Phase 2** decision, not a Phase 1 one.

---

## Batch handling

Process in batches, grouping strongly related / cross-referencing files rather than splitting alphabetically. Even then: **classify only; do not synthesize the research line yet.**

Each batch reports:

```text
Assigned files:
Fully read:
Classified:
High confidence:
Medium confidence:
Low confidence:
Multi-category:
Blocking questions:      (classifier-resolvable / registry-owned)
Non-blocking questions:
Unclassified:
Skipped:
```

Rules: no silently skipped files; `Assigned files` must reconcile with reviewed/exception counts; every unclassified file needs an explicit reason; a file with an unresolved blocking question is not fully verified.

Plus per batch: **category inventory** (categories encountered + counts), **question ledger**, **marker drift** (new out-of-vocabulary values or missing markers seen in this batch), and **classification anomalies** — filename ≠ content, task identity ≠ durable question, scope narrower than title, unclear runtime/substrate boundary, likely-but-unverified supersession, genuinely multi-domain.

---

## Phase 1 completion criteria

Complete when every file in the frozen `resolved_files` list has a reviewed record carrying primary category, secondary categories, verbatim lifecycle marker, confidence, and unresolved questions — reconciling to the frozen count with no unexplained residue — plus a final category inventory, documents per category, multi-category list, low-confidence list, blocking-question ledger split by owner, non-blocking-question ledger, the full marker-drift table, and proposed taxonomy changes.

Do not proceed to content synthesis while classification-blocking uncertainty remains unresolved **unless** it is unambiguously isolated to named documents.

---

## Non-goal (unchanged from the issue body)

`docs/modules/semantic/research/evidence/` is out of scope: evidence packets, generated artifacts, retention rules, artifact layout, figures, score/policy files, and evidence governance belong to a separate issue. Phase 1 covers only the human-facing documentation surface.

---

## One-liner

> Phase 1 maps every in-scope module document to the durable question it answers — verbatim lifecycle marker, anchored confidence, explicit questions — and produces no new knowledge model.
