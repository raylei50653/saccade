#!/usr/bin/env python3
"""Historical-vs-current status of CLOSED packets' frozen inputs (ADR 026).

A CLOSED research packet pins its inputs by ``path + sha256``.  That identity
is immutable and stays byte-verifiable forever; it does **not** by itself say
that HEAD still carries those bytes.  This tool separates the two questions
that the packets' own pinned validators conflate (they compare the packet's
digest against the working tree):

* **historical integrity** -- the packet directory is untouched and every
  frozen digest is still recoverable from git objects (from HEAD when the
  binding is ``current``, from ``last_current_ref`` when it is ``historical``);
* **currency** -- HEAD still equals the frozen bytes, so a consumer may read
  the packet's conclusions as describing HEAD.

Per binding the status is ``current`` (disk == frozen), ``historical`` (disk
differs and the supersession ledger records the transition with a re-hashable
``last_current_ref``) or ``unrecorded_drift`` (disk differs and nothing records
it).  Packet artifacts and the H0 owner-event declaration are never ledgerable:
drift there is always unrecorded.

``--mode development`` (default) fails on unrecorded drift and on an invalid
ledger; ``historical`` bindings are reported as warnings.  ``--mode attested``
adds the claim that the packets describe HEAD and also fails on ``historical``.
``--replay <packet_id>`` runs a historical packet's pinned targeted tests in a
detached worktree at its ``last_current_ref``: the strongest form of "the
evidence is still verifiable" without touching HEAD.

Discovery is generic: every ``docs/modules/semantic/research/evidence/*/`` that
has ``frozen_input_identities.json`` (rows ``inputs[].{role,path,sha256}``) or a
``manifest.json`` with ``tooling.<key>.{path,sha256}`` contributes bindings.
"""
# status: stable

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
if TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, TOOLS.as_posix())

import h0_declaration_frozen_identity as decl_id  # noqa: E402

EVIDENCE_REL = "docs/modules/semantic/research/evidence"
LEDGER_REL = "docs/research/contracts/frozen_source_supersession_ledger_v1.json"
SCHEMA_REL = "scripts/tools/frozen_source_supersession_ledger_v1.schema.json"
POLICY_REL = "docs/decisions/026-frozen-input-source-evolution.md"
LEDGER_SCHEMA_ID = "frozen_source_supersession_ledger_v1"
ATTESTED_ENV = "SACCADE_ATTESTED_CONSUMER"

STATUS_CURRENT = "current"
STATUS_HISTORICAL = "historical"
STATUS_UNRECORDED = "unrecorded_drift"

KIND_SOURCE = "source"
KIND_TOOLING = "tooling"
KIND_DOCUMENT = "document"
KIND_PACKET_ARTIFACT = "packet_artifact"
KIND_OWNER_DECLARATION = "owner_declaration"
NON_LEDGERABLE_KINDS = frozenset({KIND_PACKET_ARTIFACT, KIND_OWNER_DECLARATION})

ENTRY_HISTORICIZATION = "historicization"
ENTRY_SUPERSESSION = "supersession"


class FrozenSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Binding:
    packet_id: str
    binding: str  # "inputs:<role>" | "tooling:<key>"
    path: str  # repo-relative posix
    sha256: str
    kind: str


@dataclass
class BindingStatus:
    binding: Binding
    status: str
    disk_sha256: str | None
    entry_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self.binding)
        data.update(
            status=self.status, disk_sha256=self.disk_sha256, entry_id=self.entry_id
        )
        return data


@dataclass
class Report:
    mode: str
    head: str | None
    bindings: list[BindingStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def historical_packets(self) -> dict[str, list[BindingStatus]]:
        out: dict[str, list[BindingStatus]] = {}
        for item in self.bindings:
            if item.status == STATUS_HISTORICAL:
                out.setdefault(item.binding.packet_id, []).append(item)
        return out

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "frozen_source_status_report_v1",
            "policy": POLICY_REL,
            "mode": self.mode,
            "head": self.head,
            "ok": self.ok,
            "bindings": [item.as_dict() for item in self.bindings],
            "historical_packets": sorted(self.historical_packets),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- io


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenSourceError(f"cannot load JSON {path}: {exc}") from exc


def _normalize(path: str) -> str:
    text = path.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def classify_path(path: str) -> str:
    """Which evolution rule governs *path* (repo-relative posix)."""
    rel = _normalize(path)
    if decl_id.is_h0_capture_declaration_path(rel):
        return KIND_OWNER_DECLARATION
    if rel.startswith(EVIDENCE_REL + "/"):
        return KIND_PACKET_ARTIFACT
    if rel.startswith(("scripts/", "tests/", "tools/")):
        return KIND_TOOLING
    if rel.startswith("docs/"):
        return KIND_DOCUMENT
    return KIND_SOURCE


# ---------------------------------------------------------------------- discovery


def discover_bindings(root: Path = ROOT) -> list[Binding]:
    """Every (packet, binding) -> (path, sha256) pin under the evidence root."""
    evidence = root / EVIDENCE_REL
    found: list[Binding] = []
    if not evidence.is_dir():
        return found
    for packet_dir in sorted(p for p in evidence.iterdir() if p.is_dir()):
        packet_id = packet_dir.name
        frozen_path = packet_dir / "frozen_input_identities.json"
        if frozen_path.is_file():
            frozen = load_json(frozen_path)
            if not isinstance(frozen, Mapping):
                raise FrozenSourceError(f"{frozen_path} must be an object")
            for index, row in enumerate(frozen.get("inputs") or []):
                if not isinstance(row, Mapping):
                    raise FrozenSourceError(
                        f"{frozen_path}: inputs[{index}] must be an object"
                    )
                found.append(
                    Binding(
                        packet_id=packet_id,
                        binding=f"inputs:{row.get('role', index)}",
                        path=_normalize(str(row["path"])),
                        sha256=str(row["sha256"]),
                        kind=classify_path(str(row["path"])),
                    )
                )
        manifest_path = packet_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            tooling = manifest.get("tooling") if isinstance(manifest, Mapping) else None
            if isinstance(tooling, Mapping):
                for key, row in tooling.items():
                    if not (
                        isinstance(row, Mapping) and "path" in row and "sha256" in row
                    ):
                        continue
                    found.append(
                        Binding(
                            packet_id=packet_id,
                            binding=f"tooling:{key}",
                            path=_normalize(str(row["path"])),
                            sha256=str(row["sha256"]),
                            kind=classify_path(str(row["path"])),
                        )
                    )
    return found


def packet_targeted_tests(root: Path = ROOT) -> dict[str, str]:
    """packet_id -> repo-relative path of the packet's pinned targeted tests."""
    out: dict[str, str] = {}
    for binding in discover_bindings(root):
        if binding.binding == "tooling:targeted_tests":
            out[binding.packet_id] = binding.path
    return out


# ---------------------------------------------------------------------------- git


def _git(root: Path, *args: str, binary: bool = False) -> bytes | str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", root.as_posix(), *args],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace").strip()


def git_head(root: Path = ROOT) -> str | None:
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    return head if isinstance(head, str) and head else None


def git_blob_sha256(root: Path, ref: str, path: str) -> str | None:
    data = _git(root, "cat-file", "blob", f"{ref}:{path}", binary=True)
    if not isinstance(data, bytes):
        return None
    return sha256_bytes(data)


def git_is_ancestor(root: Path, ref: str, of: str = "HEAD") -> bool | None:
    try:
        proc = subprocess.run(
            ["git", "-C", root.as_posix(), "merge-base", "--is-ancestor", ref, of],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


# ------------------------------------------------------------------------- ledger


def load_ledger(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FrozenSourceError(
            f"missing ledger {path}: a deleted guard is not an empty ledger"
        )
    ledger = load_json(path)
    if not isinstance(ledger, Mapping):
        raise FrozenSourceError(f"{path} must be an object")
    return ledger


def validate_ledger_schema(ledger: Mapping[str, Any], schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - contract env always has it
        return ["jsonschema is not importable; ledger schema not verified"]
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"ledger schema: {'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(
            validator.iter_errors(ledger), key=lambda e: list(e.absolute_path)
        )
    ]


def _binding_set(
    bindings: Iterable[Binding], path: str, sha256: str
) -> set[tuple[str, str]]:
    return {
        (b.packet_id, b.binding)
        for b in bindings
        if b.path == path and b.sha256 == sha256
    }


def _successor_checks(
    entry: Mapping[str, Any],
    bound_packets: set[str],
    root: Path,
) -> list[str]:
    errors: list[str] = []
    eid = entry["entry_id"]
    successor = entry.get("successor_packet_id")
    auth = entry.get("owner_authorization")
    if not successor:
        errors.append(f"{eid}: supersession requires successor_packet_id")
        return errors
    if not isinstance(auth, Mapping):
        errors.append(f"{eid}: supersession requires owner_authorization")
        return errors
    packet_dir = root / EVIDENCE_REL / str(successor)
    manifest = packet_dir / "manifest.json"
    frozen = packet_dir / "frozen_input_identities.json"
    if not (manifest.is_file() and frozen.is_file()):
        errors.append(
            f"{eid}: successor packet {successor!r} must exist with "
            "manifest.json and frozen_input_identities.json"
        )
        return errors
    record = load_json(frozen)
    rows = [
        r
        for r in (record.get("inputs") or [])
        if isinstance(r, Mapping)
        and _normalize(str(r.get("path", ""))) == entry["path"]
    ]
    if not rows:
        errors.append(
            f"{eid}: successor packet {successor!r} must re-freeze {entry['path']}"
        )
    elif any(str(r.get("sha256")) == entry["frozen_sha256"] for r in rows):
        errors.append(
            f"{eid}: successor packet {successor!r} pins {entry['path']} at the "
            "superseded digest; a successor must freeze the evolved bytes"
        )
    supersedes = record.get("supersedes")
    if not isinstance(supersedes, Sequence) or isinstance(supersedes, str):
        errors.append(
            f"{eid}: successor packet {successor!r} frozen_input_identities.json "
            "must carry a `supersedes` list"
        )
        return errors
    declared = {
        (str(s.get("packet_id")), str(s.get("owner_acceptance_id")))
        for s in supersedes
        if isinstance(s, Mapping)
    }
    for packet_id in sorted(bound_packets):
        if (packet_id, str(auth.get("owner_acceptance_id"))) not in declared:
            errors.append(
                f"{eid}: successor packet {successor!r} must declare "
                f"supersedes[{{packet_id: {packet_id!r}, owner_acceptance_id: "
                f"{auth.get('owner_acceptance_id')!r}}}]"
            )
    return errors


def validate_ledger(
    ledger: Mapping[str, Any],
    bindings: Sequence[Binding],
    root: Path = ROOT,
    schema_path: Path | None = None,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[str]]:
    """Return (valid entries keyed by (path, frozen_sha256), errors).

    An entry with any error is excluded from the valid map, so drift it was
    meant to cover degrades to ``unrecorded_drift`` rather than being trusted.
    """
    errors = validate_ledger_schema(ledger, schema_path or root / SCHEMA_REL)
    if errors:
        return {}, errors
    valid: dict[tuple[str, str], Mapping[str, Any]] = {}
    seen_ids: set[str] = set()
    head = git_head(root)
    for entry in ledger.get("entries") or []:
        eid = str(entry["entry_id"])
        entry_errors: list[str] = []
        if eid in seen_ids:
            entry_errors.append(f"{eid}: duplicate entry_id")
        seen_ids.add(eid)
        path = _normalize(str(entry["path"]))
        digest = str(entry["frozen_sha256"])
        key = (path, digest)
        if key in valid:
            entry_errors.append(f"{eid}: duplicate (path, frozen_sha256) {key}")
        kind = classify_path(path)
        if kind in NON_LEDGERABLE_KINDS:
            entry_errors.append(
                f"{eid}: {path} is a {kind}; packet artifacts and the H0 owner-event "
                "declaration are immutable and cannot be historicized"
            )
        actual = _binding_set(bindings, path, digest)
        if not actual:
            entry_errors.append(f"{eid}: no packet binds {path} at {digest}")
        declared = {
            (str(b.get("packet_id")), str(b.get("binding")))
            for b in entry.get("bound_by") or []
        }
        if actual and declared != actual:
            entry_errors.append(
                f"{eid}: bound_by must list exactly the packets pinning "
                f"({path}, {digest[:12]}); expected {sorted(actual)}, got {sorted(declared)}"
            )
        ref = str(entry["last_current_ref"])
        if head is None:
            entry_errors.append(
                f"{eid}: git is unavailable; cannot verify last_current_ref"
            )
        else:
            ancestor = git_is_ancestor(root, ref, head)
            if ancestor is None:
                entry_errors.append(
                    f"{eid}: last_current_ref {ref} is not a commit reachable in this clone"
                )
            elif not ancestor:
                entry_errors.append(
                    f"{eid}: last_current_ref {ref} is not an ancestor of HEAD {head}"
                )
            else:
                blob = git_blob_sha256(root, ref, path)
                if blob != digest:
                    entry_errors.append(
                        f"{eid}: blob at {ref[:12]}:{path} hashes to {blob}, "
                        f"not the frozen {digest}"
                    )
        if entry.get("claims_inherited") is not False:
            entry_errors.append(
                f"{eid}: claims_inherited must be false; inheritance is the successor "
                "packet's claim to establish, never the ledger's"
            )
        kind_of_entry = str(entry["kind"])
        if kind_of_entry == ENTRY_HISTORICIZATION:
            if entry.get("successor_packet_id") is not None:
                entry_errors.append(
                    f"{eid}: historicization must not name a successor packet; use kind "
                    f"{ENTRY_SUPERSESSION!r}"
                )
        elif kind_of_entry == ENTRY_SUPERSESSION:
            entry_errors.extend(_successor_checks(entry, {p for p, _ in actual}, root))
        errors.extend(entry_errors)
        if not entry_errors:
            valid[key] = entry
    return valid, errors


# ----------------------------------------------------------------------- evaluate


def _disk_sha256(root: Path, path: str) -> str | None:
    try:
        return sha256_bytes((root / path).read_bytes())
    except OSError:
        return None


def evaluate(
    root: Path = ROOT,
    ledger_path: Path | None = None,
    schema_path: Path | None = None,
    mode: str = "development",
) -> Report:
    if mode not in ("development", "attested"):
        raise ValueError(f"unknown mode {mode!r}")
    report = Report(mode=mode, head=git_head(root))
    try:
        bindings = discover_bindings(root)
        ledger = load_ledger(ledger_path or root / LEDGER_REL)
    except FrozenSourceError as exc:
        report.errors.append(str(exc))
        return report
    entries, ledger_errors = validate_ledger(ledger, bindings, root, schema_path)
    report.errors.extend(ledger_errors)

    for binding in bindings:
        disk = _disk_sha256(root, binding.path)
        entry = entries.get((binding.path, binding.sha256))
        if binding.kind == KIND_OWNER_DECLARATION:
            raw = (root / binding.path).read_bytes() if disk is not None else b""
            current = disk is not None and decl_id.frozen_path_hash_ok(
                path=binding.path, disk_bytes=raw, expected_sha256=binding.sha256
            )
        else:
            current = disk == binding.sha256
        if current:
            status = STATUS_CURRENT
            if entry is not None:
                report.warnings.append(
                    f"{binding.packet_id} {binding.binding}: ledger entry "
                    f"{entry['entry_id']} is dormant (HEAD equals the frozen bytes again)"
                )
        elif entry is not None and binding.kind not in NON_LEDGERABLE_KINDS:
            status = STATUS_HISTORICAL
        else:
            status = STATUS_UNRECORDED
        report.bindings.append(
            BindingStatus(
                binding=binding,
                status=status,
                disk_sha256=disk,
                entry_id=str(entry["entry_id"])
                if entry is not None and status != STATUS_CURRENT
                else None,
            )
        )

    for item in report.bindings:
        b = item.binding
        if item.status == STATUS_UNRECORDED:
            what = (
                "missing from HEAD"
                if item.disk_sha256 is None
                else f"hashes to {item.disk_sha256}"
            )
            if b.kind in NON_LEDGERABLE_KINDS:
                report.errors.append(
                    f"{b.packet_id} {b.binding}: {b.path} {what}, frozen {b.sha256}; "
                    f"{b.kind} bytes are immutable (ADR 026 §3)"
                )
            else:
                report.errors.append(
                    f"{b.packet_id} {b.binding}: {b.path} {what}, frozen {b.sha256}; "
                    f"unrecorded drift -- append a ledger entry ({LEDGER_REL}) in the "
                    f"same PR, see {POLICY_REL} §5"
                )
        elif item.status == STATUS_HISTORICAL:
            line = (
                f"{b.packet_id} {b.binding}: {b.path} is historical "
                f"(entry {item.entry_id}); packet conclusions describe the frozen "
                "coordinate, not HEAD"
            )
            if mode == "attested":
                report.errors.append(line)
            else:
                report.warnings.append(line)
    return report


def targeted_tests_to_skip(report: Report, root: Path = ROOT) -> dict[str, str]:
    """Pinned targeted-test files whose packet is historical at HEAD.

    Those files are the packet's own currency assertions (they compare frozen
    digests against the working tree).  Under ADR 026 they belong to the
    attested arm once the packet is historical; the development arm skips them
    with a reason instead of failing.  Unrecorded drift is never skipped.
    """
    targeted = packet_targeted_tests(root)
    out: dict[str, str] = {}
    for packet_id, items in report.historical_packets.items():
        test_path = targeted.get(packet_id)
        if test_path is None:
            continue
        paths = ", ".join(sorted({i.binding.path for i in items}))
        out[test_path] = (
            f"packet {packet_id} is historical at HEAD for {paths} (ADR 026); "
            f"these are its currency assertions -- set {ATTESTED_ENV}=1 to run them"
        )
    return out


def attested_consumer_requested(environ: Mapping[str, str] | None = None) -> bool:
    value = (environ if environ is not None else os.environ).get(ATTESTED_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------------------------- replay


def replay_packet(packet_id: str, root: Path = ROOT, ref: str | None = None) -> int:
    """Run a packet's pinned targeted tests in a detached worktree at *ref*.

    *ref* defaults to the ``last_current_ref`` recorded for that packet (any of
    its historical entries; they must agree).  Returns the pytest exit code.
    """
    targeted = packet_targeted_tests(root).get(packet_id)
    if targeted is None:
        raise FrozenSourceError(
            f"packet {packet_id!r} has no tooling:targeted_tests binding"
        )
    if ref is None:
        ledger = load_ledger(root / LEDGER_REL)
        refs = {
            str(e["last_current_ref"])
            for e in ledger.get("entries") or []
            if any(b.get("packet_id") == packet_id for b in e.get("bound_by") or [])
        }
        if not refs:
            raise FrozenSourceError(
                f"packet {packet_id!r} has no ledger entry; pass --at <commit>"
            )
        if len(refs) > 1:
            raise FrozenSourceError(
                f"packet {packet_id!r} has entries at several refs {sorted(refs)}; pass --at"
            )
        ref = refs.pop()
    worktree = Path(tempfile.mkdtemp(prefix="frozen-replay-"))
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                root.as_posix(),
                "worktree",
                "add",
                "--detach",
                worktree.as_posix(),
                ref,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise FrozenSourceError(f"git worktree add failed: {proc.stderr.strip()}")
        test_file = worktree / targeted
        if not test_file.is_file():
            raise FrozenSourceError(f"{targeted} does not exist at {ref}")
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_file.as_posix(),
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=worktree.as_posix(),
            check=False,
        ).returncode
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                root.as_posix(),
                "worktree",
                "remove",
                "--force",
                worktree.as_posix(),
            ],
            capture_output=True,
            check=False,
        )
        shutil.rmtree(worktree, ignore_errors=True)


# ----------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--mode",
        choices=("development", "attested"),
        default="development",
        help="development: unrecorded drift and ledger defects fail, historical warns; "
        "attested: historical also fails (the consumer claims packets describe HEAD)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--ledger", type=Path, default=None, help="override ledger path"
    )
    parser.add_argument(
        "--replay",
        metavar="PACKET_ID",
        help="run PACKET_ID's pinned targeted tests in a worktree at its last_current_ref",
    )
    parser.add_argument(
        "--at", metavar="COMMIT", help="commit for --replay (default: ledger)"
    )
    args = parser.parse_args(argv)

    if args.replay:
        try:
            return replay_packet(args.replay, ROOT, args.at)
        except FrozenSourceError as exc:
            print(f"replay failed: {exc}", file=sys.stderr)
            return 2

    report = evaluate(ROOT, args.ledger, mode=args.mode)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        counts: dict[str, int] = {}
        for item in report.bindings:
            counts[item.status] = counts.get(item.status, 0) + 1
        summary = (
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no bindings"
        )
        print(
            f"frozen-source status [{args.mode}] @ {report.head or 'no-git'}: {summary}"
        )
        for line in report.warnings:
            print(f"  warning: {line}")
        for line in report.errors:
            print(f"  error: {line}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
