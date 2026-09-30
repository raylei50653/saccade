#!/usr/bin/env python3
"""Research study protocol v1 (experiment contract §20.11): tier, freeze, attempts.

A study under §20.11 is one directory, ``docs/research/studies/<study_id>/``:

    study.yaml        the §20.2 fields that decide the tier, in machine form
    declaration.md    the human declaration (never parsed, only pinned)
    results.md        optional; written after an attempt, never before
    attempts/NNN/     one directory per execution attempt, append-only

What this module enforces, and against which drift (#493 PR-4 rule: every gate
names the drift it prevents):

* **Tier is derived, not asserted** (conclusion drift). ``evidence_tier`` in
  study.yaml must equal :func:`derive_tier` of the §20.2 fields; the author's
  label is checked, never trusted.
* **Exploratory is not citable** (conclusion drift). A formal evidence chain
  that names an exploratory study fails; promotion is a new formal study
  (§20.5), never a relabel, so a study's tier can never change once merged.
* **Declaration before execution, in one PR** (conclusion drift). A runner
  gets data paths only from :func:`open_frozen_study`, which first verifies
  the freeze: clean tree, annotated tag == HEAD locally and on the remote, and
  every pinned declaration blob. There is no other constructor for the handle.
* **Attempts are append-only and adopted by rule** (conclusion drift). An
  invalid attempt must cite a predeclared validity criterion; the adopted
  terminal comes from the declared rule, never from "the latest attempt".

Limit: Python cannot sandbox file access. A runner that computes a data path
itself bypasses the handle; the static check rejects literal data paths and a
runner that never calls ``open_frozen_study``, not arbitrary computed paths.

Usage:
  .venv/bin/python scripts/tools/research_study.py [--base origin/main]
"""

# status: stable

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[2]

STUDIES_REL = "docs/research/studies"
STUDY_FILE = "study.yaml"
ATTEMPTS_DIR = "attempts"
ATTEMPT_FILE = "attempt.json"
STUDY_SCHEMA = "research_study_v1"
ATTEMPT_SCHEMA = "research_attempt_v1"
UNRESOLVED = "UNRESOLVED"

Tier = Literal["formal", "exploratory"]
TIERS: frozenset[str] = frozenset({"formal", "exploratory"})

# §20.7 transitions 1–3 plus `none`; only 2 and 3 make a study formal (owner, #493).
MAINLINE_TRANSITIONS: frozenset[str] = frozenset(
    {
        "closes_core_unknown",
        "adds_decision_capability",
        "changes_production_behavior",
        "none",
    }
)
FORMAL_TRANSITIONS: frozenset[str] = frozenset(
    {"adds_decision_capability", "changes_production_behavior"}
)
# §20.4 output classes.
OUTPUT_CLASSES: frozenset[str] = frozenset(
    {
        "design_candidate",
        "performance_upper_bound",
        "diagnostic",
        "unexplained_residual",
    }
)
ADOPTION_RULES: frozenset[str] = frozenset({"first_valid", "unanimous_valid"})

# Documents whose citations make a claim formal. Formal studies' own artifacts
# are added at check time.
FORMAL_CHAIN_PREFIXES: tuple[str, ...] = (
    "docs/research/contracts/",
    "docs/research/evidence_ledger.md",
    "docs/reference/no_go_registry.md",
)

# Literal path roots a runner may not name: data reaches it through the handle.
FORBIDDEN_RUNNER_PREFIXES: tuple[str, ...] = ("datasets/", "results/")

_STUDY_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TERMINAL = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CRITERION = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_BLOB = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_DIR = re.compile(r"^[0-9]{3}$")
_TIER_HEADER = re.compile(r"^<!-- evidence-tier: ([a-z]+) -->\s*$")
_HEADER_SCAN_LINES = 20

_STUDY_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "evidence_tier",
        "hypothesis",
        "section_20_2",
        "declaration",
        "results",
        "runner",
        "inputs",
        "validity_criteria",
        "attempt_policy",
    }
)
_REQUIRED_STUDY_KEYS = frozenset(
    {"schema", "study_id", "evidence_tier", "hypothesis", "section_20_2", "declaration"}
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "attempt",
        "runner",
        "freeze",
        "validity",
        "terminal",
        "invalid_criterion",
        "files",
    }
)


class StudyError(Exception):
    """A study, freeze or attempt problem; ``problems`` lists every finding."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


# --------------------------------------------------------------------------- git


def _git(root: Path, *args: str, binary: bool = False) -> Any:
    proc = subprocess.run(
        ["git", "-C", root.as_posix(), *args], capture_output=True, check=False
    )
    if proc.returncode != 0:
        return None
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace").strip()


def _blob_at(root: Path, commit: str, rel: str) -> str | None:
    out = _git(root, "rev-parse", "--verify", "--quiet", f"{commit}:{rel}")
    return out if isinstance(out, str) and _BLOB.match(out) else None


def _read_at(root: Path, commit: str, rel: str) -> bytes | None:
    out = _git(root, "cat-file", "blob", f"{commit}:{rel}", binary=True)
    return out if isinstance(out, bytes) else None


def _worktree_blob(root: Path, rel: str) -> str | None:
    if not (root / rel).is_file():
        return None
    out = _git(root, "hash-object", "--", rel)
    return out if isinstance(out, str) and _BLOB.match(out) else None


def _head(root: Path) -> str | None:
    return _git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")


def _tag_problems(root: Path, tag: str, commit: str) -> list[str]:
    ref = f"refs/tags/{tag}"
    if _git(root, "rev-parse", "--verify", "--quiet", ref) is None:
        return [f"freeze tag {tag!r} does not exist"]
    kind = _git(root, "cat-file", "-t", ref)
    problems = []
    if kind != "tag":
        problems.append(f"freeze tag {tag!r} is {kind!r}, not an annotated tag")
    peeled = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if peeled != commit:
        problems.append(f"freeze tag {tag!r} peels to {peeled}, not {commit}")
    return problems


def _remote_peeled(root: Path, remote: str, tag: str) -> str | None:
    out = _git(root, "ls-remote", remote, f"refs/tags/{tag}^{{}}")
    if not isinstance(out, str) or not out:
        return None
    return out.split()[0]


# ------------------------------------------------------------------------ schema


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated key instead of keeping the last one."""


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def parse_study(text: str | bytes) -> dict[str, Any]:
    try:
        doc = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise StudyError([f"study.yaml does not parse: {exc}"]) from exc
    if not isinstance(doc, dict):
        raise StudyError(["study.yaml is not a mapping"])
    return doc


def derive_tier(section_20_2: Mapping[str, Any]) -> Tier:
    """Formal iff a terminal moves §20.7 transition 2/3 or a design candidate may be claimed."""
    transitions = section_20_2.get("mainline_transition") or {}
    classes = section_20_2.get("output_class") or []
    if any(value in FORMAL_TRANSITIONS for value in transitions.values()):
        return "formal"
    if "design_candidate" in classes:
        return "formal"
    return "exploratory"


def _is_repo_rel(value: Any) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return False
    parts = PurePosixPath(value).parts
    return ".." not in parts and "." not in parts and "\\" not in value


def _is_leaf_name(value: Any) -> bool:
    return isinstance(value, str) and _is_repo_rel(value) and "/" not in value


def study_schema_problems(doc: Mapping[str, Any], study_id: str) -> list[str]:
    """Every schema finding for one parsed study.yaml living in ``<study_id>/``."""
    problems: list[str] = []
    unknown = sorted(set(doc) - _STUDY_KEYS)
    if unknown:
        problems.append(f"unknown keys {unknown}")
    missing = sorted(_REQUIRED_STUDY_KEYS - set(doc))
    if missing:
        return problems + [f"missing keys {missing}"]
    if doc["schema"] != STUDY_SCHEMA:
        problems.append(f"schema is {doc['schema']!r}, not {STUDY_SCHEMA!r}")
    if doc["study_id"] != study_id or not _STUDY_ID.match(str(study_id)):
        problems.append(
            f"study_id {doc['study_id']!r} must equal its directory {study_id!r}"
        )
    hypothesis = doc["hypothesis"]
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or "\n" in hypothesis.strip()
    ):
        problems.append("hypothesis must be one non-empty line")

    section = doc["section_20_2"]
    if not isinstance(section, dict) or set(section) != {
        "output_class",
        "mainline_transition",
    }:
        return problems + [
            "section_20_2 must hold exactly output_class and mainline_transition"
        ]
    classes = section["output_class"]
    if (
        not isinstance(classes, list)
        or not classes
        or len(set(map(str, classes))) != len(classes)
        or not set(classes) <= OUTPUT_CLASSES
    ):
        problems.append(
            f"output_class must be a non-empty, duplicate-free subset of {sorted(OUTPUT_CLASSES)}"
        )
    transitions = section["mainline_transition"]
    if not isinstance(transitions, dict) or not transitions:
        return problems + [
            "mainline_transition must map every terminal to its §20.7 transition"
        ]
    for terminal, transition in transitions.items():
        if not isinstance(terminal, str) or not _TERMINAL.match(terminal):
            problems.append(f"terminal {terminal!r} is not UPPER_SNAKE")
        if transition not in MAINLINE_TRANSITIONS:
            problems.append(
                f"terminal {terminal!r} maps to unknown transition {transition!r}"
            )

    tier = doc["evidence_tier"]
    derived = derive_tier(section)
    if tier not in TIERS:
        problems.append(f"evidence_tier {tier!r} is not one of {sorted(TIERS)}")
    elif tier != derived:
        problems.append(
            f"evidence_tier says {tier!r} but §20.2 fields derive {derived!r} (§20.11.1)"
        )

    if not _is_leaf_name(doc["declaration"]) or not str(doc["declaration"]).endswith(
        ".md"
    ):
        problems.append(
            "declaration must be a .md file name inside the study directory"
        )
    if "results" in doc and (
        not _is_leaf_name(doc["results"]) or not str(doc["results"]).endswith(".md")
    ):
        problems.append("results must be a .md file name inside the study directory")

    runner = doc.get("runner")
    if derived == "formal" and runner is None:
        problems.append("a formal study must name its runner (§20.11.3)")
    if runner is None:
        for key in ("inputs", "validity_criteria", "attempt_policy"):
            if key in doc:
                problems.append(f"{key} without a runner has nothing to govern")
        return problems

    if not _is_repo_rel(runner) or not str(runner).endswith(".py"):
        problems.append("runner must be a repo-relative .py path")
    inputs = doc.get("inputs")
    if not isinstance(inputs, dict):
        problems.append(
            "a study with a runner must declare inputs (a mapping, possibly empty)"
        )
    else:
        for name, path in inputs.items():
            if (
                not isinstance(name, str)
                or not _CRITERION.match(name)
                or not _is_repo_rel(path)
            ):
                problems.append(
                    f"input {name!r} -> {path!r} is not name -> repo-relative path"
                )
    criteria = doc.get("validity_criteria")
    if not isinstance(criteria, dict) or not criteria:
        problems.append(
            "a study with a runner must predeclare validity_criteria (§20.11.4)"
        )
    else:
        for key, text in criteria.items():
            if not isinstance(key, str) or not _CRITERION.match(key):
                problems.append(f"validity criterion id {key!r} is malformed")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"validity criterion {key!r} has no definition")
    policy = doc.get("attempt_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "adoption",
        "max_valid_attempts",
        "max_attempts",
    }:
        problems.append(
            "attempt_policy must hold exactly adoption, max_valid_attempts, max_attempts"
        )
    else:
        if policy["adoption"] not in ADOPTION_RULES:
            problems.append(f"adoption must be one of {sorted(ADOPTION_RULES)}")
        max_valid, max_total = policy["max_valid_attempts"], policy["max_attempts"]
        if not (
            isinstance(max_valid, int)
            and not isinstance(max_valid, bool)
            and max_valid >= 1
        ):
            problems.append("max_valid_attempts must be an integer >= 1")
        elif not (
            isinstance(max_total, int)
            and not isinstance(max_total, bool)
            and max_total >= max_valid
        ):
            problems.append("max_attempts must be an integer >= max_valid_attempts")
        if policy["adoption"] == "unanimous_valid" and UNRESOLVED not in transitions:
            problems.append(
                f"unanimous_valid needs a declared {UNRESOLVED} terminal for disagreement"
            )
    return problems


def _md_tier_header(text: str) -> str | None:
    for line in text.splitlines()[:_HEADER_SCAN_LINES]:
        match = _TIER_HEADER.match(line)
        if match:
            return match.group(1)
    return None


# ----------------------------------------------------------------- runner check


def runner_source_problems(source: str, study: Mapping[str, Any]) -> list[str]:
    """Static half of "no data before freeze": no literal data path, one freeze call."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"runner does not parse: {exc}"]
    inputs = [str(p) for p in (study.get("inputs") or {}).values()]
    forbidden_exact = {PurePosixPath(p).parts[0] for p in inputs} | {
        prefix.rstrip("/") for prefix in FORBIDDEN_RUNNER_PREFIXES
    }
    problems: list[str] = []
    calls_open = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip("./")
            if (
                value in forbidden_exact
                or value.startswith(FORBIDDEN_RUNNER_PREFIXES)
                or any(
                    value
                    and (value == p or p.startswith(value + "/") or value.startswith(p))
                    for p in inputs
                )
            ):
                problems.append(
                    f"runner line {node.lineno} names data path {node.value!r}; "
                    "data must come from FrozenStudy.input()"
                )
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if name == "open_frozen_study":
                calls_open = True
    if not calls_open:
        problems.append(
            "runner never calls open_frozen_study: it is not bound to its declaration"
        )
    return problems


# ---------------------------------------------------------------- freeze + API


@dataclass(frozen=True)
class StudyBinding:
    """What a runner pins about its own declaration, fixed in the runner source.

    ``pinned_blobs`` maps repo-relative paths to git blob ids and must cover the
    study's study.yaml and declaration. ``runner_file`` is the runner's
    ``__file__``; the freeze requires study.yaml to name that same file.
    """

    study_id: str
    runner_file: str | Path
    freeze_tag: str
    pinned_blobs: Mapping[str, str]


_OPEN_TOKEN = object()


class FrozenStudy:
    """The only source of data paths and the only writer of attempt records.

    Obtainable only from :func:`open_frozen_study`, i.e. after the freeze passed.
    """

    def __init__(
        self,
        token: object,
        *,
        root: Path,
        study: Mapping[str, Any],
        attempt: int,
        runner: str,
        tag: str,
        commit: str,
        pinned: Mapping[str, str],
    ) -> None:
        if token is not _OPEN_TOKEN:
            raise StudyError(["FrozenStudy is only issued by open_frozen_study()"])
        self._root = root
        self._study = dict(study)
        self.study_id = str(study["study_id"])
        self.attempt = attempt
        self._runner = runner
        self._tag = tag
        self.freeze_commit = commit
        self._pinned = dict(pinned)
        self._recorded = False

    @property
    def attempt_dir(self) -> Path:
        return (
            self._root
            / STUDIES_REL
            / self.study_id
            / ATTEMPTS_DIR
            / f"{self.attempt:03d}"
        )

    def input(self, name: str) -> Path:
        inputs = self._study.get("inputs") or {}
        if name not in inputs:
            raise StudyError([f"{name!r} is not a declared input of {self.study_id}"])
        return self._root / inputs[name]

    def payload_dir(self) -> Path:
        """Create (once) and return this attempt's directory for result files."""
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        return self.attempt_dir

    def record(
        self,
        validity: Literal["valid", "invalid"],
        *,
        terminal: str | None = None,
        invalid_criterion: str | None = None,
    ) -> Path:
        """Seal this attempt: hash every payload file and write attempt.json."""
        if self._recorded:
            raise StudyError(["this attempt is already recorded"])
        problems = _outcome_problems(self._study, validity, terminal, invalid_criterion)
        if problems:
            raise StudyError(problems)
        directory = self.payload_dir()
        record = {
            "schema": ATTEMPT_SCHEMA,
            "study_id": self.study_id,
            "attempt": self.attempt,
            "runner": self._runner,
            "freeze": {
                "tag": self._tag,
                "commit": self.freeze_commit,
                "pinned_blobs": dict(sorted(self._pinned.items())),
            },
            "validity": validity,
            "terminal": terminal,
            "invalid_criterion": invalid_criterion,
            "files": _payload_hashes(directory),
        }
        path = directory / ATTEMPT_FILE
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._recorded = True
        return path


def _outcome_problems(
    study: Mapping[str, Any], validity: Any, terminal: Any, criterion: Any
) -> list[str]:
    terminals = study["section_20_2"]["mainline_transition"]
    criteria = study.get("validity_criteria") or {}
    if validity == "valid":
        problems = (
            [] if terminal in terminals else [f"terminal {terminal!r} is not declared"]
        )
        if criterion is not None:
            problems.append("a valid attempt cites no invalidity criterion")
        return problems
    if validity == "invalid":
        problems = (
            []
            if criterion in criteria
            else [
                f"invalid attempt must cite a predeclared validity criterion, got {criterion!r}"
            ]
        )
        if terminal is not None:
            problems.append("an invalid attempt has no terminal")
        return problems
    return [f"validity {validity!r} is neither 'valid' nor 'invalid'"]


def _payload_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.relative_to(directory).as_posix() != ATTEMPT_FILE:
            hashes[path.relative_to(directory).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return hashes


def _study_rel(study_id: str, name: str) -> str:
    return f"{STUDIES_REL}/{study_id}/{name}"


def _existing_attempts(root: Path, study_id: str) -> list[Path]:
    base = root / STUDIES_REL / study_id / ATTEMPTS_DIR
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def freeze_problems(
    binding: StudyBinding, *, root: Path = ROOT, remote: str = "origin"
) -> tuple[list[str], dict[str, Any] | None, str | None]:
    """(problems, study at HEAD, HEAD) — every check that must pass before data."""
    problems: list[str] = []
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        return [f"{root} is not a git work tree"], None, None
    if status:
        problems.append(
            "work tree is not clean; the freeze is HEAD, not the files on disk"
        )
    head = _head(root)
    if head is None:
        return problems + ["HEAD does not resolve"], None, None

    problems += _tag_problems(root, binding.freeze_tag, head)
    remote_peeled = _remote_peeled(root, remote, binding.freeze_tag)
    if remote_peeled != head:
        problems.append(
            f"{remote} {binding.freeze_tag}^{{}} is {remote_peeled}, not HEAD {head}: "
            "an unpublished freeze is not a freeze"
        )

    study_rel = _study_rel(binding.study_id, STUDY_FILE)
    raw = _read_at(root, head, study_rel)
    if raw is None:
        return problems + [f"{study_rel} is not committed at HEAD"], None, head
    try:
        study = parse_study(raw)
    except StudyError as exc:
        return problems + exc.problems, None, head
    schema = study_schema_problems(study, binding.study_id)
    if schema:
        return problems + [f"{study_rel}: {p}" for p in schema], None, head

    declaration_rel = _study_rel(binding.study_id, study["declaration"])
    for required in (study_rel, declaration_rel):
        if required not in binding.pinned_blobs:
            problems.append(
                f"binding does not pin {required}: the runner is not bound to it"
            )
    for rel, blob in sorted(binding.pinned_blobs.items()):
        actual = _blob_at(root, head, rel)
        if actual != blob:
            problems.append(f"{rel} is {actual} at HEAD, runner pins {blob}")

    runner_rel = _repo_relative(root, binding.runner_file)
    if study.get("runner") != runner_rel:
        problems.append(
            f"study.yaml binds runner {study.get('runner')!r}, this runner is {runner_rel!r}"
        )
    source = _read_at(root, head, runner_rel) if runner_rel else None
    if source is None:
        problems.append(f"runner {runner_rel!r} is not committed at HEAD")
    else:
        problems += runner_source_problems(source.decode("utf-8", "replace"), study)

    attempts = _existing_attempts(root, binding.study_id)
    used_tags = {_attempt_field(a, "freeze", "tag") for a in attempts}
    if binding.freeze_tag in used_tags:
        problems.append(
            f"freeze tag {binding.freeze_tag!r} already froze an earlier attempt; "
            "every attempt gets its own tag (moving one would orphan that record)"
        )
    policy = study["attempt_policy"]
    valid = sum(1 for a in attempts if _attempt_validity(a) == "valid")
    if len(attempts) >= policy["max_attempts"]:
        problems.append(
            f"attempt budget spent ({len(attempts)}/{policy['max_attempts']})"
        )
    if valid >= policy["max_valid_attempts"]:
        problems.append(
            f"{valid} valid attempt(s) already recorded; a valid result is not rerun "
            "— write a new declaration (§20.11.4)"
        )
    return problems, study, head


def open_frozen_study(
    binding: StudyBinding, *, root: Path = ROOT, remote: str = "origin"
) -> FrozenStudy:
    """Verify the freeze, then hand out the only route to data. Raises StudyError."""
    problems, study, head = freeze_problems(binding, root=root, remote=remote)
    if problems or study is None or head is None:
        raise StudyError(problems or ["freeze could not be established"])
    attempt = len(_existing_attempts(root, binding.study_id)) + 1
    return FrozenStudy(
        _OPEN_TOKEN,
        root=root,
        study=study,
        attempt=attempt,
        runner=str(study["runner"]),
        tag=binding.freeze_tag,
        commit=head,
        pinned=binding.pinned_blobs,
    )


def _repo_relative(root: Path, path: str | Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _attempt_field(directory: Path, *keys: str) -> Any:
    try:
        value: Any = json.loads((directory / ATTEMPT_FILE).read_text(encoding="utf-8"))
        for key in keys:
            value = value[key]
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _attempt_validity(directory: Path) -> str | None:
    return _attempt_field(directory, "validity")


# ------------------------------------------------------------ attempt verifier


def attempt_problems(
    root: Path, study: Mapping[str, Any], directory: Path
) -> list[str]:
    """Whether one recorded attempt is a formally valid execution record."""
    name = directory.name
    path = directory / ATTEMPT_FILE
    if not _ATTEMPT_DIR.match(name):
        return [f"attempt directory {name!r} is not NNN"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{name}: attempt.json unreadable: {exc}"]
    if not isinstance(record, dict) or set(record) != _ATTEMPT_KEYS:
        return [f"{name}: attempt.json keys must be exactly {sorted(_ATTEMPT_KEYS)}"]
    problems: list[str] = []
    study_id = str(study["study_id"])
    if record["schema"] != ATTEMPT_SCHEMA or record["study_id"] != study_id:
        problems.append(f"{name}: schema/study_id mismatch")
    if record["attempt"] != int(name):
        problems.append(f"{name}: records attempt {record['attempt']!r}")

    freeze = record["freeze"]
    if not isinstance(freeze, dict) or set(freeze) != {"tag", "commit", "pinned_blobs"}:
        return problems + [f"{name}: freeze must hold tag, commit, pinned_blobs"]
    commit, tag, pinned = freeze["commit"], freeze["tag"], freeze["pinned_blobs"]
    if not (isinstance(commit, str) and _BLOB.match(commit)) or not isinstance(
        pinned, dict
    ):
        return problems + [f"{name}: freeze commit/pins malformed"]
    if _git(root, "cat-file", "-t", commit) != "commit":
        return problems + [f"{name}: freeze commit {commit} is not in this repository"]
    ancestor = subprocess.run(
        ["git", "-C", root.as_posix(), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        problems.append(
            f"{name}: freeze commit {commit[:12]} is not an ancestor of HEAD "
            "(squash or rebase lost the freeze; merge with a merge commit)"
        )
    problems += [f"{name}: {p}" for p in _tag_problems(root, str(tag), commit)]

    study_rel = _study_rel(study_id, STUDY_FILE)
    raw = _read_at(root, commit, study_rel)
    if raw is None:
        return problems + [f"{name}: {study_rel} absent at the freeze commit"]
    try:
        frozen_study = parse_study(raw)
    except StudyError as exc:
        return problems + [f"{name}: frozen {p}" for p in exc.problems]
    schema = study_schema_problems(frozen_study, study_id)
    if schema:
        return problems + [f"{name}: frozen study.yaml: {p}" for p in schema]
    declaration_rel = _study_rel(study_id, str(frozen_study.get("declaration")))
    for required in (study_rel, declaration_rel):
        if required not in pinned:
            problems.append(f"{name}: freeze does not pin {required}")
    for rel, blob in sorted(pinned.items()):
        if _blob_at(root, commit, str(rel)) != blob:
            problems.append(
                f"{name}: {rel} at the freeze commit is not the pinned blob"
            )
    if frozen_study.get("runner") != record["runner"]:
        problems.append(
            f"{name}: attempt runner {record['runner']!r} is not the study's runner"
        )
    source = _read_at(root, commit, str(record["runner"]))
    if source is None:
        problems.append(f"{name}: runner absent at the freeze commit")
    else:
        problems += [
            f"{name}: {p}"
            for p in runner_source_problems(
                source.decode("utf-8", "replace"), frozen_study
            )
        ]
    results = frozen_study.get("results")
    if (
        results
        and _read_at(root, commit, _study_rel(study_id, str(results))) is not None
    ):
        problems.append(
            f"{name}: results existed at the freeze commit, before execution"
        )

    # Nothing that decides the tier or the terminals may move after a freeze.
    if _worktree_blob(root, study_rel) != pinned.get(study_rel):
        problems.append(
            f"{name}: study.yaml changed after the freeze (relabel is not promotion, §20.5)"
        )
    frozen_declaration = _read_at(root, commit, declaration_rel)
    current = root / declaration_rel
    if (
        frozen_declaration is None
        or not current.is_file()
        or not current.read_bytes().startswith(frozen_declaration)
    ):
        problems.append(
            f"{name}: the declaration's frozen body changed (append below it only)"
        )

    problems += [
        f"{name}: {p}"
        for p in _outcome_problems(
            frozen_study,
            record["validity"],
            record["terminal"],
            record["invalid_criterion"],
        )
    ]
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            problems.append(
                f"{name}: symlink {entry.relative_to(directory)} in attempt"
            )
    files = record["files"]
    if not isinstance(files, dict) or files != _payload_hashes(directory):
        problems.append(f"{name}: payload files differ from the sealed inventory")
    return problems


def adopt(study: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str | None:
    """The adopted terminal under the declared rule; None while nothing is valid.

    Never "the last attempt": ``first_valid`` takes the earliest valid attempt,
    ``unanimous_valid`` requires every valid attempt to agree, else UNRESOLVED.
    """
    ordered = sorted(records, key=lambda r: int(r["attempt"]))
    valid = [str(r["terminal"]) for r in ordered if r["validity"] == "valid"]
    if not valid:
        return None
    rule = study["attempt_policy"]["adoption"]
    if rule == "first_valid":
        return valid[0]
    if rule == "unanimous_valid":
        return valid[0] if len(set(valid)) == 1 else UNRESOLVED
    raise StudyError([f"unknown adoption rule {rule!r}"])


# --------------------------------------------------------------- tree checks


def discover(root: Path = ROOT) -> list[Path]:
    base = root / STUDIES_REL
    return (
        sorted(p for p in base.iterdir() if (p / STUDY_FILE).is_file())
        if base.is_dir()
        else []
    )


def study_problems(
    root: Path, directory: Path
) -> tuple[list[str], dict[str, Any] | None]:
    """Everything checkable about one study directory on the working tree."""
    label = directory.name
    try:
        study = parse_study((directory / STUDY_FILE).read_bytes())
    except StudyError as exc:
        return [f"{label}: {p}" for p in exc.problems], None
    schema = study_schema_problems(study, label)
    if schema:
        return [f"{label}: {p}" for p in schema], None
    problems: list[str] = []
    tier = study["evidence_tier"]
    for key in ("declaration", "results"):
        if key not in study:
            continue
        doc = directory / study[key]
        if not doc.is_file():
            if key == "declaration":
                problems.append(f"{label}: declaration {study[key]} is missing")
            continue
        header = _md_tier_header(doc.read_text(encoding="utf-8"))
        if header != tier:
            problems.append(
                f"{label}: {study[key]} must carry '<!-- evidence-tier: {tier} -->', found {header!r}"
            )
    runner = study.get("runner")
    if runner and not (root / runner).is_file():
        problems.append(f"{label}: runner {runner} is missing")

    attempts = _existing_attempts(root, label)
    records = []
    for index, attempt in enumerate(attempts, start=1):
        if attempt.name != f"{index:03d}":
            problems.append(
                f"{label}: attempts must be 001..N without gaps, found {attempt.name}"
            )
            break
        found = attempt_problems(root, study, attempt)
        problems += [f"{label}: {p}" for p in found]
        if not found:
            records.append(
                json.loads((attempt / ATTEMPT_FILE).read_text(encoding="utf-8"))
            )
    if runner:
        policy = study["attempt_policy"]
        valid = sum(1 for r in records if r["validity"] == "valid")
        if (
            len(attempts) > policy["max_attempts"]
            or valid > policy["max_valid_attempts"]
        ):
            problems.append(f"{label}: attempts exceed the declared attempt_policy")
        results = study.get("results")
        if (
            tier == "formal"
            and results
            and (directory / results).is_file()
            and not records
        ):
            problems.append(f"{label}: formal results without a verified attempt")
    return problems, study


def citation_problems(
    root: Path, studies: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """A formal evidence chain may not name an exploratory study (§20.11.2)."""
    exploratory = sorted(
        sid for sid, s in studies.items() if s["evidence_tier"] == "exploratory"
    )
    if not exploratory:
        return []
    chain: list[Path] = []
    for prefix in FORMAL_CHAIN_PREFIXES:
        target = root / prefix
        if target.is_file():
            chain.append(target)
        elif target.is_dir():
            chain += sorted(p for p in target.rglob("*") if p.is_file())
    for sid, study in studies.items():
        if study["evidence_tier"] == "formal":
            directory = root / STUDIES_REL / sid
            chain += [directory / STUDY_FILE] + [
                directory / study[k] for k in ("declaration", "results") if k in study
            ]
    patterns = {
        sid: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(sid)}(?![A-Za-z0-9_])")
        for sid in exploratory
    }
    problems = []
    for path in chain:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for sid, pattern in patterns.items():
            if pattern.search(text):
                problems.append(
                    f"{path.relative_to(root).as_posix()} cites exploratory study {sid!r}; "
                    "promote by a new formal declaration and rerun (§20.5, §20.11.2)"
                )
    return problems


def history_problems(root: Path, base: str) -> list[str]:
    """Against merge-base(base, HEAD): no study vanishes or changes tier; attempts never change."""
    merge_base = _git(root, "merge-base", base, "HEAD")
    if not merge_base:
        return [f"cannot resolve merge-base of {base!r} and HEAD"]
    listing = _git(root, "ls-tree", "-r", "--name-only", merge_base, "--", STUDIES_REL)
    problems: list[str] = []
    for rel in (listing or "").splitlines():
        parts = PurePosixPath(rel).relative_to(STUDIES_REL).parts
        if len(parts) == 2 and parts[1] == STUDY_FILE:
            sid = parts[0]
            try:
                before = parse_study(_read_at(root, merge_base, rel) or b"")
            except StudyError:
                continue
            current = root / rel
            if not current.is_file():
                problems.append(f"{sid}: merged study removed")
                continue
            try:
                after = parse_study(current.read_bytes())
            except StudyError:
                continue  # reported by study_problems
            if isinstance(before.get("section_20_2"), dict) and isinstance(
                after.get("section_20_2"), dict
            ):
                if derive_tier(before["section_20_2"]) != derive_tier(
                    after["section_20_2"]
                ):
                    problems.append(
                        f"{sid}: tier changed after merge — relabel is not promotion (§20.5)"
                    )
        if len(parts) >= 3 and parts[1] == ATTEMPTS_DIR:
            if _worktree_blob(root, rel) != _blob_at(root, merge_base, rel):
                problems.append(
                    f"{rel}: merged attempt record changed or removed (append-only)"
                )
    return problems


def check_all(root: Path = ROOT, base: str | None = None) -> list[str]:
    problems: list[str] = []
    studies: dict[str, dict[str, Any]] = {}
    for directory in discover(root):
        found, study = study_problems(root, directory)
        problems += found
        if study is not None:
            studies[directory.name] = study
    problems += citation_problems(root, studies)
    if base is not None:
        problems += history_problems(root, base)
    return problems


def default_base(root: Path = ROOT) -> str | None:
    """``origin/main`` when it resolves (as frozen_source_status does), else None."""
    return (
        "origin/main"
        if _git(root, "rev-parse", "--verify", "--quiet", "origin/main")
        else None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--base", default=None, help="merge-base ref (default origin/main)"
    )
    args = parser.parse_args(argv)
    problems = check_all(ROOT, args.base or default_base(ROOT))
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{len(discover(ROOT))} studies, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
