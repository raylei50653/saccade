# tests/research — research test quarantine

> **Governance rule**
> Dated research tests may live in `tests/research/` only while the study is
> active; sealing triggers promotion, consolidation, or deletion. Sealed
> packets are preserved by artifacts and generic packet validation, not
> packet-specific regression tests.

## What lives here

Study-scoped tests: they validate a specific research proposition, read a
sealed evidence packet, exec a runner frozen inside a dated evidence
directory, or pin frozen study numbers. They are **excluded from the default
pytest collection** (`addopts = -m 'not research'` in pyproject.toml) and from
pre_push / CI. The `conftest.py` here auto-applies `@pytest.mark.research` to
everything in this tree, plus `@pytest.mark.packet_bound` for files listed in
`PACKET_BOUND_FILES`.

Run them explicitly:

```bash
.venv/bin/python -m pytest -m research tests/research/          # all
.venv/bin/python -m pytest tests/research/safe_region -m research
```

(The CLI `-m` overrides the addopts default.)

## What must NOT live here

- Behavior tests of live modules (evaluator wiring, determinism tooling,
  tracker code paths) — those belong in `tests/unit/` etc., even if the
  feature is default-off.
- Permanent packet integrity validation — that is `tests/contract/`
  (`test_research_packet_schema.py`, `test_research_packet_manifest.py`),
  which parameterizes over every sealed packet generically.

## Lifecycle

When a study seals, each of its tests must reach one of four terminal states
(tracked in `DISPOSITION.md`):

1. **Promote** to `tests/contract/` — the study surfaced a durable
   engineering contract.
2. **Consolidate** into a parameterized behavior test — it is really a case
   of generic module behavior.
3. **Covered by generic packet checkers** — it only verified packet
   completeness; delete it.
4. **Delete** — it only reproduced a one-shot research result; the recipe,
   artifacts, and git history preserve it.

A test failing here does **not** mean a historical research conclusion is
invalid — the substrate evolves. Do not contort mainline code to keep a
sealed study's test green; dispose of the test instead.
