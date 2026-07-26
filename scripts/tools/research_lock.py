#!/usr/bin/env python3
"""Online modification and research measurement are mutually exclusive states.

`check_runtime_identity_staleness.py` guards the online -> research direction: a
coordinate that moved without a republish is caught. Nothing guarded the other
direction. A research instance could be measuring against a coordinate while
that coordinate was edited underneath it, and the loss surfaced only afterwards,
as evidence that had already been collected against a substrate that no longer
existed.

This lock makes the two states named, exclusive and fail-closed:

    ONLINE_OPEN --open--> RESEARCH_OPEN --close--> RESEARCH_CLOSED --release--> ONLINE_OPEN

Opening an instance freezes the runtime coordinate it binds to. While the
instance is open, the frozen axes may not move. The default frozen set is the
two axes the accepted `runtime_coordinate_bindings_v1` consumption rule already
classifies as `stale` (conclusion-invalidating) rather than
`re_attestation_required`; a study that needs more freezes more, per instance,
instead of everyone paying for it permanently.

The lock file is deliberately outside every axis it freezes. H0 re-entry #3 died
because its declaration was simultaneously a frozen runtime-bound input and the
target of the seal that mutated it. A lock whose own transitions moved a frozen
digest would reproduce that defect exactly.

Enforcement does not live here. It lives in `tests/contract/test_research_lock.py`
so that the guard runs under the existing fail-closed pytest step without
editing `scripts/pre_push.sh`, which is itself an `identity_semantics` file.

Usage:
  uv run python scripts/tools/research_lock.py status
  uv run python scripts/tools/research_lock.py verify
  uv run python scripts/tools/research_lock.py open --instance-id <id> \
      --declaration <path> --evidence-root <path> [--freeze-axes a,b]
  uv run python scripts/tools/research_lock.py close --disposition sealed|voided \
      --note <text> [--registry-pointer <object>]
  uv run python scripts/tools/research_lock.py release --note <text>
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import check_runtime_identity_staleness as staleness  # noqa: E402

LOCK_REL = "docs/research/contracts/research_lock_v1.json"
LOCK_SCHEMA = "research_lock_v1"

ONLINE_OPEN = "ONLINE_OPEN"
RESEARCH_OPEN = "RESEARCH_OPEN"
RESEARCH_CLOSED = "RESEARCH_CLOSED"
STATES = (ONLINE_OPEN, RESEARCH_OPEN, RESEARCH_CLOSED)

# The transition graph is total: anything not named here is refused.
TRANSITIONS = {
    "open": (ONLINE_OPEN, RESEARCH_OPEN),
    "close": (RESEARCH_OPEN, RESEARCH_CLOSED),
    "release": (RESEARCH_CLOSED, ONLINE_OPEN),
}

# Only source-derived axes can be recomputed on every host, so only they can be
# enforced on every push. `environment` and `runtime_inputs` are excluded by
# construction, not by preference.
LOCKABLE_AXES = ("decision_surface", "identity_semantics", "implementation")
DEFAULT_FROZEN_AXES = ("decision_surface", "identity_semantics")

DISPOSITIONS = ("sealed", "voided")

INSTANCE_REQUIRED = (
    "instance_id",
    "declaration",
    "evidence_root",
    "opened_at",
    "frozen_axes",
    "frozen",
    "disposition",
    "registry_pointer",
)
HISTORY_REQUIRED = ("from", "to", "at", "instance_id", "note")

# Every use of this sits inside a message about drift that has already happened,
# so it must not offer `sealed` (which that drift has already made illegal) or
# imply that `release` gates editing (it gates the next instance).
DRIFT_EXIT_HINT = (
    "the frozen coordinate moved while a research instance is open — either "
    "revert the edit, or close the instance as voided "
    "(`research_lock.py close --disposition voided --note ...`); `release` is "
    "required only before another research instance can open"
)


class ResearchLockError(RuntimeError):
    pass


def lock_path() -> Path:
    return REPO_ROOT / LOCK_REL


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_lock(path: Path | None = None) -> dict[str, Any]:
    """Read and fully validate the lock. A missing lock is never ONLINE_OPEN."""
    path = path or lock_path()
    if not path.is_file():
        raise ResearchLockError(
            f"no research lock at {path} — the lock is committed state; its "
            "absence is a deleted guard, not an open online surface"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchLockError(f"{path}: invalid JSON: {exc}") from exc
    validate_lock(payload)
    return payload


def validate_lock(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ResearchLockError("lock payload is not a mapping")
    if payload.get("schema") != LOCK_SCHEMA:
        raise ResearchLockError(f"lock is not a {LOCK_SCHEMA} payload")
    state = payload.get("state")
    if state not in STATES:
        raise ResearchLockError(f"unknown lock state {state!r}; want one of {STATES}")

    history = payload.get("history")
    if not isinstance(history, list):
        raise ResearchLockError("history is not a list")
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise ResearchLockError(f"history[{index}] is not a mapping")
        missing = [key for key in HISTORY_REQUIRED if key not in record]
        if missing:
            raise ResearchLockError(f"history[{index}] is missing {missing}")
        if record["to"] not in STATES or record["from"] not in STATES:
            raise ResearchLockError(f"history[{index}] names an unknown state")
    if history and history[-1]["to"] != state:
        raise ResearchLockError(
            f"state {state} does not match the last transition "
            f"{history[-1]['from']} -> {history[-1]['to']}"
        )

    instance = payload.get("instance")
    if state == ONLINE_OPEN:
        if instance is not None:
            raise ResearchLockError(
                "ONLINE_OPEN carries an instance — a released lock keeps no "
                "ghost freeze; the version binding lives in the closed "
                "instance's own evidence"
            )
        return
    if not isinstance(instance, Mapping):
        raise ResearchLockError(f"{state} without an instance")
    missing = [key for key in INSTANCE_REQUIRED if key not in instance]
    if missing:
        raise ResearchLockError(f"instance is missing {missing}")

    axes = instance["frozen_axes"]
    if not isinstance(axes, list) or not axes:
        raise ResearchLockError("frozen_axes is empty — an instance freezes something")
    unknown = [axis for axis in axes if axis not in LOCKABLE_AXES]
    if unknown:
        raise ResearchLockError(
            f"frozen_axes names non-recomputable axes {unknown}; want a subset "
            f"of {list(LOCKABLE_AXES)}"
        )

    frozen = instance["frozen"]
    if not isinstance(frozen, Mapping):
        raise ResearchLockError("instance.frozen is not a mapping")
    coordinate = frozen.get("coordinate")
    if not isinstance(coordinate, Mapping):
        raise ResearchLockError("instance.frozen.coordinate is not a mapping")
    absent = [axis for axis in axes if axis not in coordinate]
    if absent:
        raise ResearchLockError(f"instance.frozen.coordinate is missing {absent}")
    if not isinstance(frozen.get("probe"), str):
        raise ResearchLockError("instance.frozen.probe is missing")

    disposition = instance["disposition"]
    if state == RESEARCH_OPEN and disposition is not None:
        raise ResearchLockError("an open instance already carries a disposition")
    if state == RESEARCH_CLOSED and disposition not in DISPOSITIONS:
        raise ResearchLockError(
            f"closed instance disposition {disposition!r}; want one of {DISPOSITIONS}"
        )


def recompute(axes: tuple[str, ...] | list[str]) -> dict[str, str]:
    computed = {
        "decision_surface": identity.decision_surface_axis,
        "identity_semantics": identity.identity_semantics_axis,
        "implementation": identity.implementation_axis,
    }
    return {axis: computed[axis]()["digest"] for axis in axes}


def verify(
    lock: Mapping[str, Any],
    published: Mapping[str, Any],
    *,
    measured: Mapping[str, str] | None = None,
) -> list[str]:
    """Return hard failures. Only RESEARCH_OPEN enforces the freeze.

    RESEARCH_CLOSED keeps the frozen coordinate as the sealed version binding of
    a finished study; it no longer holds the online surface shut.
    """
    if lock["state"] != RESEARCH_OPEN:
        return []

    instance = lock["instance"]
    axes = list(instance["frozen_axes"])
    frozen = instance["frozen"]["coordinate"]
    instance_id = instance["instance_id"]
    failures: list[str] = []

    if measured is None:
        measured = recompute(axes)
    for axis in axes:
        if measured.get(axis) != frozen[axis]:
            failures.append(
                f"{axis} moved while research instance {instance_id!r} is open: "
                f"frozen {frozen[axis]}, recomputed {measured.get(axis)}. "
                f"{DRIFT_EXIT_HINT}"
            )

    # A republish is an online move too: it is how a coordinate edit is made
    # official, so it must not be a way around the freeze.
    published_coordinate = published["coordinate"]
    for axis in axes:
        if published_coordinate.get(axis) != frozen[axis]:
            failures.append(
                f"the published runtime identity was re-published on {axis} while "
                f"research instance {instance_id!r} is open: frozen "
                f"{frozen[axis]}, published {published_coordinate.get(axis)}. "
                f"{DRIFT_EXIT_HINT}"
            )
    published_probe = published["probe"]["digest"]
    if published_probe != instance["frozen"]["probe"]:
        failures.append(
            f"the published identity probe moved while research instance "
            f"{instance_id!r} is open: frozen {instance['frozen']['probe']}, "
            f"published {published_probe}. {DRIFT_EXIT_HINT}"
        )
    return failures


def transition(
    lock: dict[str, Any],
    action: str,
    *,
    instance: dict[str, Any] | None,
    note: str,
    at: str | None = None,
) -> dict[str, Any]:
    """Apply a named transition. History is append-only; nothing is rewritten."""
    origin, target = TRANSITIONS[action]
    if lock["state"] != origin:
        raise ResearchLockError(
            f"illegal transition: `{action}` runs {origin} -> {target}, but the "
            f"lock is {lock['state']}"
        )
    instance_id = None
    if action == "open":
        instance_id = instance["instance_id"]  # type: ignore[index]
    elif lock["instance"] is not None:
        instance_id = lock["instance"]["instance_id"]

    updated = dict(lock)
    updated["state"] = target
    updated["instance"] = instance
    updated["history"] = [
        *lock["history"],
        {
            "from": origin,
            "to": target,
            "at": at or _now(),
            "instance_id": instance_id,
            "note": note,
        },
    ]
    validate_lock(updated)
    return updated


def write_lock(payload: Mapping[str, Any], path: Path | None = None) -> None:
    """Key order is the file's own; a transition must not reflow the document."""
    path = path or lock_path()
    rendered = json.dumps(payload, indent=2) + "\n"
    path.write_text(rendered, encoding="utf-8")


def _published() -> dict[str, Any]:
    return staleness.load_published(REPO_ROOT / staleness.PUBLISHED_REL)


def publication_precondition(published: Mapping[str, Any]) -> list[str]:
    """Is the publication itself current? Nothing about who already consumed it.

    Opening an instance must not depend on global consumer state. The staleness
    CLI additionally walks every `runtime_identity_bindings_v1` row and fails on
    any `stale` or `re_attestation_required` binding, which is the right verdict
    for a consumer and the wrong precondition for establishing a new lock: a
    closed study going stale as online advances is the expected end of its life
    (§20.10.5), not a reason to refuse the next study. Requiring it to be
    current again would smuggle in exactly the cross-version maintenance the
    contract excludes.
    """
    failures, _ = staleness.compare_publication(published, probe=None)
    return failures


def _do_open(args: argparse.Namespace) -> int:
    lock = load_lock()
    axes = [axis.strip() for axis in args.freeze_axes.split(",") if axis.strip()]
    unknown = [axis for axis in axes if axis not in LOCKABLE_AXES]
    if unknown:
        raise ResearchLockError(
            f"--freeze-axes names non-recomputable axes {unknown}; want a subset "
            f"of {list(LOCKABLE_AXES)}"
        )

    # Freezing a coordinate that has already drifted from source would bind the
    # instance to a publication that no longer describes anything.
    published = _published()
    stale = publication_precondition(published)
    if stale:
        raise ResearchLockError(
            "the published runtime coordinate has drifted from source; republish "
            "before opening a research instance: " + "; ".join(stale)
        )

    binding = staleness._published_binding(published)
    instance = {
        "instance_id": args.instance_id,
        "declaration": args.declaration,
        "evidence_root": args.evidence_root,
        "opened_at": _now(),
        "frozen_axes": axes,
        "frozen": {
            "coordinate": {axis: binding["coordinate"][axis] for axis in axes},
            "probe": binding["probe"],
        },
        "disposition": None,
        "registry_pointer": None,
    }
    write_lock(transition(lock, "open", instance=instance, note=args.note))
    print(f"{ONLINE_OPEN} -> {RESEARCH_OPEN}: {args.instance_id} froze {axes}")
    return 0


def _do_close(args: argparse.Namespace) -> int:
    lock = load_lock()
    if lock["state"] != RESEARCH_OPEN:
        raise ResearchLockError(
            f"nothing to close: the lock is {lock['state']}, not {RESEARCH_OPEN}"
        )
    if args.disposition == "sealed":
        # Enforcement stops at RESEARCH_CLOSED, so sealing is the last moment at
        # which the freeze can be checked at all. Without this, drifting the
        # coordinate and then sealing would launder a broken instance into a
        # finished one and every check downstream would agree.
        failures = verify(lock, _published())
        if failures:
            raise ResearchLockError(
                "cannot seal an instance whose frozen binding moved — a sealed "
                "instance asserts that its coordinate held for the whole "
                "measurement; revert the drift, or close it as `voided`: "
                + "; ".join(failures)
            )
    instance = dict(lock["instance"])
    instance["disposition"] = args.disposition
    instance["registry_pointer"] = args.registry_pointer
    instance["closed_at"] = _now()
    write_lock(transition(lock, "close", instance=instance, note=args.note))
    print(
        f"{RESEARCH_OPEN} -> {RESEARCH_CLOSED}: "
        f"{instance['instance_id']} {args.disposition}"
    )
    return 0


def _do_release(args: argparse.Namespace) -> int:
    lock = load_lock()
    write_lock(transition(lock, "release", instance=None, note=args.note))
    print(
        f"{RESEARCH_CLOSED} -> {ONLINE_OPEN}: instance released; "
        "a new research instance may now open"
    )
    return 0


def _do_status(_: argparse.Namespace) -> int:
    lock = load_lock()
    print(lock["state"])
    instance = lock["instance"]
    if instance is not None:
        print(f"  instance      {instance['instance_id']}")
        print(f"  declaration   {instance['declaration']}")
        print(f"  evidence      {instance['evidence_root']}")
        print(f"  frozen axes   {', '.join(instance['frozen_axes'])}")
        if instance["disposition"] is not None:
            print(f"  disposition   {instance['disposition']}")
    return 0


def _do_verify(_: argparse.Namespace) -> int:
    lock = load_lock()
    failures = verify(lock, _published())
    if lock["state"] != RESEARCH_OPEN:
        print(f"{lock['state']}: no freeze is enforced")
    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    if failures:
        return 1
    if lock["state"] == RESEARCH_OPEN:
        print(f"{RESEARCH_OPEN}: frozen axes are unmoved")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status").set_defaults(handler=_do_status)
    sub.add_parser("verify").set_defaults(handler=_do_verify)

    opener = sub.add_parser("open")
    opener.add_argument("--instance-id", required=True)
    opener.add_argument("--declaration", required=True)
    opener.add_argument("--evidence-root", required=True)
    opener.add_argument("--freeze-axes", default=",".join(DEFAULT_FROZEN_AXES))
    opener.add_argument("--note", default="research instance opened")
    opener.set_defaults(handler=_do_open)

    closer = sub.add_parser("close")
    closer.add_argument("--disposition", required=True, choices=DISPOSITIONS)
    closer.add_argument("--registry-pointer", default=None)
    closer.add_argument("--note", required=True)
    closer.set_defaults(handler=_do_close)

    releaser = sub.add_parser("release")
    releaser.add_argument("--note", required=True)
    releaser.set_defaults(handler=_do_release)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ResearchLockError,
        staleness.StalenessError,
        identity.IdentityError,
        OSError,
    ) as exc:
        print(f"research lock: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
