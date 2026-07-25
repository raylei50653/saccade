#!/usr/bin/env python3
"""H2 Layer P: retryable pre-seal plumbing, with no epistemic budget attached.

Layer P covers build, build-tool binding, extension/plugin load, mutation
detection, and the identity run. It emits `plumbing_pass` or
`plumbing_blocked(<coordinate>)` and **never an H2 terminal** — the exactly-once
authorization belongs to the Layer-M measurement (declaration § 5.1 / § 5.2).

Three rules shape this controller, each answering a recorded H0 failure:

  * **Retries may only touch plumbing.** Admissibility is decided by
    `h2_path_partition`, which rejects `decision_relevant`, `invariant_authority`
    and unclassified paths. Retryability without that firewall would be the E1b
    circular oracle with extra steps.
  * **The invariant is behavioral, not physical.** Every run recomputes the
    `behavior` axis and compares it to the published identity. Physical artifact
    hashes are recorded as witness: the same source built in a different
    directory has a different `sha256` (R5 parity audit), so physical identity
    cannot be an invariant.
  * **A rejection persists its coordinate before it aborts.** R5's authoritative
    run died with `extension_load_record: not_recorded`, buying zero information
    for a whole authorization. Here every stage writes its record first.

What the reduced confinement role is: `BoundInputMonitor` (inotify) proves no
bound input was mutated during the run, and the identity tool proves the
extension under test was the one actually imported. Neither needs the loaded file
closure enumerated in advance, which is the assumption that failed five times.

Usage:
  uv run python scripts/tools/run_h2_layer_p.py
  uv run python scripts/tools/run_h2_layer_p.py --base main --emit out/h2_layer_p/cert.json
  uv run python scripts/tools/run_h2_layer_p.py --skip-build   # reuse the build dir
"""
# status: stable

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import check_runtime_identity_staleness as staleness  # noqa: E402
import h2_path_partition as partition  # noqa: E402
import run_h0_phase_a as h0_controller  # noqa: E402  (imported, never modified)
from h2_behavioral_identity import IDENTITY_SEQUENCE, canonical_json_bytes  # noqa: E402

LAYER_P_SCHEMA = "h2_layer_p_v1"
BUILD_DIR_REL = "build/h2_layer_p"
WORK_ROOT_REL = "out/h2_layer_p"

STAGES = (
    "retry_admissibility",
    "preflight",
    "build",
    "build_binding",
    "extension_load",
    "identity_run",
)

# Recorded as witness. These are the tools whose absence from the bound-input set
# ended re-entry #2 (`build_binding`) — kept, but as evidence rather than predicate.
BUILD_TOOLS = ("cmake", "nvcc", "c++")


class Blocked(RuntimeError):
    """A Layer-P stage refused. Retryable: no terminal, no budget consumed."""

    def __init__(self, coordinate: str, detail: str) -> None:
        super().__init__(f"{coordinate}: {detail}")
        self.coordinate = coordinate
        self.detail = detail


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LayerP:
    def __init__(
        self,
        *,
        base: str | None,
        skip_build: bool,
        fixture: str,
        build_dir: Path | None = None,
    ) -> None:
        self.base = base
        self.skip_build = skip_build
        self.fixture = fixture
        self.build_dir = (build_dir or (REPO_ROOT / BUILD_DIR_REL)).resolve()
        self.build_dir_rel = (
            self.build_dir.relative_to(REPO_ROOT).as_posix()
            if self.build_dir.is_relative_to(REPO_ROOT)
            else self.build_dir.as_posix()
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.work_dir = REPO_ROOT / WORK_ROOT_REL / stamp
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "schema": LAYER_P_SCHEMA,
            "authority": "non_authoritative_pre_seal_engineering",
            "started_utc": _utc(),
            "fixture": fixture,
            "stages": {},
            "result": "in_progress",
        }

    # -- record keeping ---------------------------------------------------- #
    def _persist(self) -> None:
        """Write the record after every stage, so an abort still leaves a coordinate."""
        path = self.work_dir / "layer_p.json"
        path.write_bytes(canonical_json_bytes(self.record) + b"\n")

    def _stage(self, name: str, state: str, **fields: Any) -> None:
        self.record["stages"][name] = {"state": state, "utc": _utc(), **fields}
        self._persist()

    def _block(self, name: str, detail: str, **fields: Any) -> Blocked:
        self._stage(name, "blocked", detail=detail, **fields)
        self.record["result"] = "plumbing_blocked"
        self.record["blocked_at"] = name
        self.record["blocked_detail"] = detail
        self._persist()
        return Blocked(name, detail)

    # -- stages ------------------------------------------------------------ #
    def retry_admissibility(self) -> None:
        if self.base is None:
            self._stage(
                "retry_admissibility",
                "skipped",
                note="no --base given; admissibility is only defined against a base",
            )
            return
        try:
            changed = partition.changed_paths_since(self.base)
        except subprocess.CalledProcessError as exc:
            raise self._block(
                "retry_admissibility", f"git diff vs {self.base} failed: {exc}"
            )
        verdict = partition.check_retry(changed)
        fields = {
            "base": self.base,
            "changed_count": len(changed),
            "decision_relevant": list(verdict.decision_relevant),
            "invariant_authority": list(verdict.invariant_authority),
            "unclassified": list(verdict.unclassified),
        }
        if not verdict.admissible:
            raise self._block("retry_admissibility", verdict.reason(), **fields)
        self._stage("retry_admissibility", "pass", **fields)

    def preflight(self) -> dict[str, Any]:
        # Launch hygiene: a pre-existing build tree ended one H0 re-entry
        # (`build/h0_phase_a exists at controller launch`). Here it is a cheap
        # precondition rather than a spent authorization.
        if self.build_dir.exists() and not self.skip_build:
            raise self._block(
                "preflight",
                f"{self.build_dir_rel} exists at launch; remove it or pass --skip-build",
            )
        if self.skip_build and not self.build_dir.is_dir():
            raise self._block(
                "preflight", f"--skip-build but {self.build_dir_rel} is absent"
            )
        try:
            published = staleness.load_published(REPO_ROOT / staleness.PUBLISHED_REL)
            failures, warnings = staleness.compare_publication(published, behavior=None)
        except (staleness.StalenessError, identity.IdentityError) as exc:
            raise self._block("preflight", f"published identity unusable: {exc}")
        if failures:
            raise self._block("preflight", "; ".join(failures))
        reference = published["identity"].get("behavior")
        if reference is None:
            raise self._block(
                "preflight",
                "published identity has no behavior axis; there is no invariant to hold",
            )
        self._stage(
            "preflight",
            "pass",
            published_identity=published["identity"],
            static_axis_warnings=warnings,
        )
        return published

    def build(self) -> None:
        if self.skip_build:
            self._stage("build", "skipped", build_dir=self.build_dir_rel)
            return
        logs = self.work_dir / "logs"
        logs.mkdir(exist_ok=True)
        vectors = (
            (
                "configure",
                [
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    self.build_dir_rel,
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
            ),
            ("build", ["cmake", "--build", self.build_dir_rel, "--parallel"]),
        )
        for name, vector in vectors:
            started = time.monotonic()
            completed = subprocess.run(
                vector, cwd=REPO_ROOT, capture_output=True, text=True
            )
            (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise self._block(
                    "build",
                    f"{name} exited {completed.returncode}; see {logs}/{name}.stderr.log",
                    vector=vector,
                    returncode=completed.returncode,
                )
            self.record.setdefault("build_timings", {})[name] = round(
                time.monotonic() - started, 3
            )
        self._stage("build", "pass", build_dir=self.build_dir_rel)

    def build_binding(self) -> None:
        """Record which build tools were actually resolved — witness, not predicate."""
        tools = []
        for command in BUILD_TOOLS:
            resolved = shutil.which(command)
            tools.append({"command": command, "path": resolved})
        missing = [item["command"] for item in tools if item["path"] is None]
        if missing:
            # Genuinely blocking: a build cannot be described if its tools are absent.
            raise self._block(
                "build_binding", f"build tools absent: {missing}", tools=tools
            )
        self._stage(
            "build_binding",
            "pass",
            tools=tools,
            note=(
                "Witness only. Build-tool provenance ended re-entry #2 as a predicate; "
                "under H2 the behavior digest carries the invariant instead."
            ),
        )

    def extension_load(self) -> dict[str, Any]:
        """Prove the extension in the build dir is the one an interpreter imports."""
        script = (
            "import json,sys;"
            "sys.path.insert(0, %r);"
            "import h2_behavioral_identity as b;"
            "d=b.resolve_build_dir();"
            "import importlib; importlib.import_module('saccade_tracking_ext');"
            "print(json.dumps(b._assert_extension_consumed(d)))"
        ) % (_TOOLS.as_posix(),)
        environment = {
            **os.environ,
            "SACCADE_BUILD_PATH": self.build_dir.as_posix(),
            "PYTHONPATH": os.pathsep.join(
                (
                    self.build_dir.as_posix(),
                    (REPO_ROOT / "src").as_posix(),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise self._block(
                "extension_load",
                f"confined load probe exited {completed.returncode}: "
                f"{completed.stderr.strip()[-2000:]}",
                returncode=completed.returncode,
            )
        try:
            witness = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise self._block(
                "extension_load", f"load probe emitted no witness record: {exc}"
            )
        self._stage("extension_load", "pass", witness=witness)
        return witness

    def identity_run(self, published: dict[str, Any]) -> str:
        reference = published["identity"]["behavior"]
        result_path = self.work_dir / "behavior.json"
        bound = [
            REPO_ROOT / path
            for path in identity.decision_relevant_files()
            if (REPO_ROOT / path).is_file()
        ]
        try:
            monitor = h0_controller.BoundInputMonitor(bound)
        except (h0_controller.DriftError, OSError) as exc:
            raise self._block("identity_run", f"mutation monitor unavailable: {exc}")

        environment = {
            **os.environ,
            "SACCADE_BUILD_PATH": self.build_dir.as_posix(),
        }
        completed = subprocess.run(
            [
                sys.executable,
                (_TOOLS / "h2_behavioral_identity.py").as_posix(),
                "--identity-mode",
                "--sequences",
                self.fixture,
                "--emit",
                result_path.as_posix(),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=environment,
        )
        (self.work_dir / "identity_run.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        # Mutation check first: a mutated input invalidates whatever the run said.
        events = monitor.drain()
        if events:
            raise self._block(
                "identity_run",
                f"bound input mutated during the run: {events!r}",
                mutation_events=[event.path for event in events],
            )
        if completed.returncode != 0:
            raise self._block(
                "identity_run",
                f"identity run exited {completed.returncode}; see identity_run.log",
                returncode=completed.returncode,
            )
        try:
            measured = identity.load_behavior(result_path)["digest"]
        except identity.IdentityError as exc:
            raise self._block("identity_run", f"identity result unusable: {exc}")

        if measured != reference:
            # This is the firewall firing: a plumbing-only change moved the
            # invariant, which means it was not plumbing-only.
            raise self._block(
                "identity_run",
                "H2_PLUMBING_CHANGED_BEHAVIOR — the behavior axis moved under a "
                f"plumbing-only change: published {reference}, measured {measured}. "
                "This chain is dead; only the owner may re-open it.",
                published_behavior=reference,
                measured_behavior=measured,
            )
        self._stage("identity_run", "pass", behavior=measured, sequence=self.fixture)
        return measured

    # -- driver ------------------------------------------------------------ #
    def run(self) -> int:
        try:
            self.retry_admissibility()
            published = self.preflight()
            self.build()
            self.build_binding()
            witness = self.extension_load()
            behavior = self.identity_run(published)
        except Blocked as exc:
            print(f"plumbing_blocked({exc.coordinate}): {exc.detail}", file=sys.stderr)
            print(f"record: {self.work_dir / 'layer_p.json'}", file=sys.stderr)
            self._append_retry_log()
            return 1

        self.record["result"] = "plumbing_pass"
        self.record["certificate"] = {
            "behavior": behavior,
            "build_dir": self.build_dir_rel,
            "extension_witness": witness,
            "identity": published["identity"],
            "note": (
                "A Layer-P pass certificate is a precondition for a Layer-M seal, "
                "never an authorization. It grants no execution and no terminal."
            ),
        }
        self.record["finished_utc"] = _utc()
        self._persist()
        self._append_retry_log()
        print(f"plumbing_pass  behavior={behavior}")
        print(f"record: {self.work_dir / 'layer_p.json'}")
        return 0

    def _append_retry_log(self) -> None:
        """Append-only: every attempt records what it resolved or hit (§ 5.3 rule 3)."""
        log = REPO_ROOT / WORK_ROOT_REL / "retry_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "utc": self.record.get("finished_utc") or _utc(),
            "result": self.record["result"],
            "coordinate": self.record.get("blocked_at"),
            "detail": self.record.get("blocked_detail"),
            "record": (self.work_dir / "layer_p.json")
            .relative_to(REPO_ROOT)
            .as_posix(),
        }
        with log.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json_bytes(entry).decode("utf-8") + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help="git base to check retry admissibility against (e.g. main)",
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse an existing build dir"
    )
    parser.add_argument("--fixture", default=IDENTITY_SEQUENCE, help="identity fixture")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help=f"build directory under test (default {BUILD_DIR_REL})",
    )
    parser.add_argument(
        "--emit", type=Path, default=None, help="write the pass certificate here"
    )
    args = parser.parse_args(argv)

    controller = LayerP(
        base=args.base,
        skip_build=args.skip_build,
        fixture=args.fixture,
        build_dir=args.build_dir,
    )
    code = controller.run()
    if code == 0 and args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_bytes(
            canonical_json_bytes(controller.record["certificate"]) + b"\n"
        )
        print(f"wrote {args.emit}")
    return code


if __name__ == "__main__":
    sys.exit(main())
