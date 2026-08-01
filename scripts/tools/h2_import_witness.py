#!/usr/bin/env python3
"""Observe every repository-local file an execution loads, then classify it.

The RunSpec declares which repository source bytes are *allowed* to take part in
an execution.  This module records which ones actually did, and checks the one
direction Correction 5 needs: observed repository code lies inside the declared
authority domains.  Not equality — a run that imported the whole repository is
not the run anyone wants, and requiring it would fail every honest execution.

Two rules shape the implementation, and both were paid for.

Observe first, classify second.  Nothing is filtered at the recorder's entrance,
not third-party packages under `.venv/` and not native objects under `build/`.
A filter there would delete the evidence before anything judged it, and the file
worth seeing is precisely the one nobody predicted would load — including a
misplaced, untracked repository file, which is the case the declared roots'
`--others` selector exists to catch.

Nothing at module scope imports repository code.  The bootstrap must import this
module before the recorder can exist, so anything this module pulls in at import
time is unwitnessed by construction.  Keeping that set to this file alone is what
lets the verifier bound the preloaded list exactly instead of trusting it.
"""
# status: stable

from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

WITNESS_SCHEMA = "h2_import_witness_v1"
BOOTSTRAP_SCHEMA = "h2_import_witness_bootstrap_v1"
WITNESS_ALGORITHM = "sha256_canonical_json_observed_repo_local_files_v1"
WITNESS_AUTHORITY = "execution_integrity_gate"
WITNESS_NAME = "import_witness.json"

DOMAIN_BUILD = "build_witness"
DOMAIN_CLOSURE = "declared_execution_code_closure"
DOMAIN_ENVIRONMENT = "environment_external"
DOMAIN_NAMED = "named_execution_semantics"

# Where native build products live.  A `.so` under the invocation's *bound* build
# directory is carried by the build witness; one under any other build directory
# is not carried by anything, so it must land unbound rather than be waved
# through as an external dependency.
NATIVE_BUILD_PARENT = "build/"

# The only repository files that may already be loaded when the recorder starts:
# this module, and the bootstrap that imports it.  Anything else in the preloaded
# set means the recorder was installed too late to witness it.
BOOTSTRAP_SELF_PATHS: frozenset[str] = frozenset(
    {
        "scripts/tools/h2_child_bootstrap.py",
        "scripts/tools/h2_import_witness.py",
    }
)


class WitnessError(RuntimeError):
    """The import witness is unbuildable, or the execution left its namespace."""


def _origin_kind(relative: str) -> str:
    return (
        "extension"
        if any(relative.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
        else "source"
    )


class ImportRecorder:
    """A meta-path finder that watches resolution without performing it.

    It delegates to the finders behind it, then digests whatever file the winning
    spec resolved to.  Digesting here rather than after the run matters: a file
    rewritten mid-execution is recorded as the bytes that were about to be
    executed, not as the bytes that happen to be on disk at the end.
    """

    def __init__(self, entry_module: str = "") -> None:
        self.entry_module = entry_module
        self._records: dict[str, dict[str, Any]] = {}
        self._preloaded: set[str] = set()
        self._lock = threading.RLock()
        self._reentry = threading.local()

    # -- observation ----------------------------------------------------

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if getattr(self._reentry, "busy", False):
            return None
        self._reentry.busy = True
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None:
                    self._note_spec(fullname, spec)
                    return spec
            return None
        finally:
            self._reentry.busy = False

    def _note_spec(self, fullname: str, spec: Any) -> None:
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
            return
        loader = getattr(spec, "loader", None)
        self.note_file(fullname, origin, type(loader).__name__ if loader else "unknown")

    def note_file(self, module_name: str, origin: str, loader: str) -> None:
        """Record one resolved origin, if it is repository-local.

        `resolve()` first: a module reached through a symlink and the same module
        reached directly are one file, and recording them as two would let the
        same bytes appear under two identities.
        """
        try:
            resolved = Path(origin).resolve(strict=True)
            relative = resolved.relative_to(REPO_ROOT).as_posix()
        except (OSError, ValueError):
            return
        with self._lock:
            record = self._records.get(relative)
            if record is not None:
                record["module_names"].add(module_name)
                return
            try:
                payload = resolved.read_bytes()
            except OSError as exc:
                raise WitnessError(f"loaded file is unreadable: {relative}") from exc
            self._records[relative] = {
                "length": len(payload),
                "loader": loader or "unknown",
                "module_names": {module_name},
                "origin_kind": _origin_kind(relative),
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

    def snapshot_preloaded(self) -> tuple[str, ...]:
        """Record repository files already imported before the recorder existed."""
        with self._lock:
            before = set(self._records)
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if not isinstance(origin, str):
                continue
            loader = getattr(getattr(module, "__spec__", None), "loader", None)
            self.note_file(name, origin, type(loader).__name__ if loader else "unknown")
        with self._lock:
            self._preloaded = set(self._records) - before
            return tuple(sorted(self._preloaded))

    @property
    def preloaded(self) -> tuple[str, ...]:
        return tuple(sorted(self._preloaded))

    def observations(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                {**record, "module_names": sorted(record["module_names"])}
                for _, record in sorted(self._records.items())
            )


_ACTIVE: ImportRecorder | None = None


def install(
    recorder: ImportRecorder | None = None, *, entry_module: str = ""
) -> ImportRecorder:
    """Snapshot what is already loaded, then start watching.

    The recorder is published here rather than passed down through the child's
    call signatures.  The child is launched as a process, so there is no call to
    thread it through, and a recorder handed to `main()` would only be available
    after the imports worth witnessing had already happened.
    """
    global _ACTIVE
    started = ImportRecorder(entry_module) if recorder is None else recorder
    if entry_module and not started.entry_module:
        started.entry_module = entry_module
    started.snapshot_preloaded()
    sys.meta_path.insert(0, started)
    _ACTIVE = started
    return started


def uninstall(recorder: ImportRecorder) -> None:
    global _ACTIVE
    while recorder in sys.meta_path:
        sys.meta_path.remove(recorder)
    if _ACTIVE is recorder:
        _ACTIVE = None


def active() -> ImportRecorder | None:
    """The running recorder, or None when this process was not bootstrapped."""
    return _ACTIVE


# -- classification -----------------------------------------------------


def selectable_repository_paths() -> frozenset[str]:
    """Every repository file git would hand over: tracked, or present and unignored.

    The same selector the declared closure is built from, applied to the whole
    repository rather than to the roots.  Using one vocabulary for both is what
    makes "ignored, therefore external" a statement about the repository's own
    boundary instead of a hardcoded list of directories to wave through.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WitnessError("the repository file set is not enumerable") from exc
    return frozenset(
        entry for entry in completed.stdout.decode("utf-8").split("\0") if entry
    )


def _declared_digests(
    document: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], tuple[str, ...]]:
    try:
        projection = document["execution_semantics_projection"]
        closure = projection["execution_code_closure"]
        closure_members = {
            str(member["path"]): str(member["sha256"]) for member in closure["members"]
        }
        named_members = {
            str(member["path"]): str(member["sha256"])
            for member in projection["members"]
        }
        roots = tuple(str(root) for root in closure["roots"])
    except (KeyError, TypeError) as exc:
        raise WitnessError("RunSpec carries no usable declaration") from exc
    return closure_members, named_members, roots


def classify(
    observations: Sequence[Mapping[str, Any]],
    *,
    document: Mapping[str, Any],
    build_dir: Path,
    selectable: Iterable[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Assign every observation the set of authorities that bind its bytes.

    A set, not a kind.  Since `scripts/` became a declared root the domains
    overlap for real: the named execution-semantics tooling under `scripts/` is
    also a closure member, and both bindings apply.  Admission therefore may not
    read a precedence order — it asks whether the set is non-empty and whether
    every binding in it agrees with the observed bytes.
    """
    closure_members, named_members, roots = _declared_digests(document)
    known = frozenset(
        selectable_repository_paths() if selectable is None else selectable
    )
    try:
        build_prefix = f"{build_dir.resolve().relative_to(REPO_ROOT).as_posix()}/"
    except (OSError, ValueError):
        build_prefix = None

    classified: list[dict[str, Any]] = []
    for observation in observations:
        path = str(observation["path"])
        observed = str(observation["sha256"])
        domains: list[str] = []
        if closure_members.get(path) is not None:
            domains.append(DOMAIN_CLOSURE)
        if named_members.get(path) is not None:
            domains.append(DOMAIN_NAMED)
        if build_prefix is not None and path.startswith(build_prefix):
            domains.append(DOMAIN_BUILD)
        if (
            not domains
            and path not in known
            and not path.startswith(roots)
            and not path.startswith(NATIVE_BUILD_PARENT)
        ):
            # Ignored by the repository and outside every declared root: a
            # dependency, held by the environment axis.  The `build/` exclusion
            # is deliberate — a native object from a build directory this
            # invocation did not bind is carried by nothing, and must land
            # unbound rather than be excused as an external package.
            domains.append(DOMAIN_ENVIRONMENT)
        classified.append(
            {
                "authority_domains": sorted(domains),
                "length": int(observation["length"]),
                "loader": str(observation["loader"]),
                "module_names": sorted(
                    str(name) for name in observation["module_names"]
                ),
                "origin_kind": str(observation["origin_kind"]),
                "path": path,
                "sha256": observed,
            }
        )
    return tuple(classified)


def containment_failures(
    observations: Sequence[Mapping[str, Any]],
    *,
    document: Mapping[str, Any],
) -> tuple[str, ...]:
    """Every reason this execution left its declared namespace.

    Two failure kinds, reported together so one run surfaces both: code that no
    authority binds, and code some authority binds to different bytes than the
    ones that loaded.
    """
    closure_members, named_members, _ = _declared_digests(document)
    reasons: list[str] = []
    for observation in observations:
        path = str(observation["path"])
        observed = str(observation["sha256"])
        domains = set(observation.get("authority_domains") or ())
        if not domains:
            reasons.append(f"unbound repository code loaded: {path}")
            continue
        for domain, declared in (
            (DOMAIN_CLOSURE, closure_members.get(path)),
            (DOMAIN_NAMED, named_members.get(path)),
        ):
            if domain in domains and declared != observed:
                reasons.append(
                    f"{domain} byte identity differs from the declaration: {path}"
                )
    return tuple(sorted(reasons))


def bootstrap_failures(
    preloaded: Sequence[str], observations: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Check the recorder started before anything it was meant to witness.

    Judged by domain, not by path.  A virtualenv living inside the checkout puts
    third-party files under the repository root, and the interpreter loads two of
    them from `.pth` hooks before any code here can run — `_virtualenv` and
    `_distutils_hack`.  Calling that "repository code loaded too early" would
    make the check fire on every correct run, and a check that always fires is a
    check nobody keeps.  What the recorder must not miss is repository *source*;
    external dependencies are the environment axis's subject, whenever they load.
    """
    external = {
        str(observation["path"])
        for observation in observations
        if DOMAIN_ENVIRONMENT in (observation.get("authority_domains") or ())
    }
    return tuple(
        f"repository code loaded before the recorder installed: {path}"
        for path in sorted(set(preloaded) - BOOTSTRAP_SELF_PATHS - external)
    )


# -- the record ---------------------------------------------------------


def build_witness(
    recorder: ImportRecorder,
    *,
    document: Mapping[str, Any],
    build_dir: Path,
    entry_module: str | None = None,
) -> dict[str, Any]:
    """Assemble the witness document and gate the execution on it."""
    from h2_runtime_inputs import digest

    entry_module = recorder.entry_module if entry_module is None else entry_module
    if not entry_module:
        raise WitnessError("the import witness has no bootstrap entry point")

    observations = classify(
        recorder.observations(), document=document, build_dir=build_dir
    )
    if not observations:
        raise WitnessError("the import witness observed nothing")
    preloaded = recorder.preloaded
    reasons = bootstrap_failures(preloaded, observations) + containment_failures(
        observations, document=document
    )
    if reasons:
        raise WitnessError("; ".join(reasons))
    projection = document["execution_semantics_projection"]
    witness = {
        "algorithm": WITNESS_ALGORITHM,
        "authority": WITNESS_AUTHORITY,
        "bootstrap": {
            "entry_module": entry_module,
            "preloaded_repo_local_paths": list(preloaded),
            "recorder_installed_before_entry_import": True,
            "schema": BOOTSTRAP_SCHEMA,
        },
        "declared": {
            "execution_code_closure_digest": projection["execution_code_closure"][
                "digest"
            ],
            "execution_semantics_projection_digest": projection["digest"],
            "roots": list(projection["execution_code_closure"]["roots"]),
        },
        "observations": [dict(observation) for observation in observations],
        "schema": WITNESS_SCHEMA,
    }
    witness["digest"] = digest(witness["observations"])
    return witness


def validate_witness(witness: Mapping[str, Any]) -> None:
    """Re-derive the archive's own claims from its own bytes.

    Deliberately no checkout read.  Whether the recorded bytes still exist on
    some host is a different question from whether this record is internally
    honest, and only the second is answerable from an archive.
    """
    from h2_runtime_inputs import digest

    if witness.get("schema") != WITNESS_SCHEMA:
        raise WitnessError("import witness schema mismatch")
    if witness.get("algorithm") != WITNESS_ALGORITHM:
        raise WitnessError("import witness algorithm mismatch")
    if witness.get("authority") != WITNESS_AUTHORITY:
        raise WitnessError("import witness authority mismatch")
    observations = witness.get("observations")
    if not isinstance(observations, list) or not observations:
        raise WitnessError("import witness carries no observations")
    paths = [observation.get("path") for observation in observations]
    if paths != sorted(set(str(path) for path in paths)):
        raise WitnessError("import witness observations are unsorted or repeat")
    if witness.get("digest") != digest(observations):
        raise WitnessError("import witness digest mismatch")
    bootstrap = witness.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise WitnessError("import witness carries no bootstrap evidence")
    if bootstrap.get("recorder_installed_before_entry_import") is not True:
        raise WitnessError("import witness does not claim an early recorder")
    reasons = bootstrap_failures(
        bootstrap.get("preloaded_repo_local_paths") or (), observations
    )
    unbound = [
        str(observation.get("path"))
        for observation in observations
        if not observation.get("authority_domains")
    ]
    if unbound:
        reasons += tuple(f"unbound repository code loaded: {path}" for path in unbound)
    if reasons:
        raise WitnessError("; ".join(reasons))
