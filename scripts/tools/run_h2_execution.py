#!/usr/bin/env python3
"""Produce one H2 successor execution archive: three artifacts, no verdict about itself.

Review Correction 5 splits the successor path in two. This is the first half —
the producer — and the rule it exists to obey is what it may *not* do: it emits
`run_spec.json`, `runtime_binding.json` and `result.json`, and it must never
write `verification.json`. `verify_h2_execution.py` is the other half, and it
landed first so this file could be written against a contract that was already
enforced rather than against whatever this file happens to emit (§ 5.3).

**It decides no verdict.** `select_successor_result` selects the result and the
terminal from the recorded predicates; this module transcribes that selection. A
producer that wrote its own `result` field would be answering the question its
own archive exists to pose, and the § 20.8 two-implementer test would then be
comparing a copy against itself. The selection runs twice — unnamed, to ask
whether terminal 4's cause may be named at all, then carrying it — over one
frozen observation, so the second call cannot answer a question the first was
never asked.

**It executes nothing of its own.** The six retained stages come from
`run_h2_layer_p`, the four ordered runs from the Layer-M runner, the runtime
input manifest from `h2_runtime_inputs`, and the declared content set from the
RunSpec resolver. That is why the `executed_surfaces` the binding records are
exactly the seven paths the frozen schema names: they are the code that ran, and
this orchestrator — like `run_h2_layer_p.py` before it — is not among them.

**The sequencing is injected, not hard-wired.** `Stages` and `Runs` are
protocols, so the control flow this module owns — fail-fast ordering, a stage
failure leaving every measurement run `not_run`, the diagnostic/measurement
authority split — is exercised in tests without a build or a GPU. The real
implementations bind the retained modules; nothing here re-implements them.
"""
# status: stable

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_measurement_evidence as evidence  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_execution as archive  # noqa: E402

PHASE = archive.PHASE
BINDING_SCHEMA = "h2_runtime_binding_v1"
RESULT_SCHEMA = "h2_execution_result_v1"
PROBE_SCHEMA = "h2_behavior_probe_result_v1"
PROBE_ROLE = "recorded_observation_not_equivalence_or_gate"
PROBE_STATE = "computed"


class ProducerError(RuntimeError):
    """The producer refuses to emit. No archive is written."""


# -- what the frozen contracts say this execution must record --------------- #


def _binding_contract() -> Mapping[str, Any]:
    return archive._load_contract(archive.PRODUCER_ARTIFACTS["runtime_binding.json"])


def _result_contract() -> Mapping[str, Any]:
    return archive._load_contract(archive.PRODUCER_ARTIFACTS["result.json"])


def executed_surface_paths() -> tuple[str, ...]:
    """The seven surfaces the binding schema names, read from the schema itself."""
    blocks = _binding_contract()["properties"]["executed_surfaces"]["allOf"]
    return tuple(block["contains"]["properties"]["path"]["const"] for block in blocks)


def capture_abi_path() -> str:
    return str(
        _binding_contract()["properties"]["capture_abi"]["allOf"][1]["properties"][
            "path"
        ]["const"]
    )


def run_ids() -> tuple[str, ...]:
    return tuple(
        _result_contract()["properties"]["run_plan"]["properties"]["run_ids"]["const"]
    )


def sequence() -> str:
    return str(
        _result_contract()["properties"]["run_plan"]["properties"]["sequence"]["const"]
    )


def predicate_names() -> tuple[str, ...]:
    return tuple(_result_contract()["properties"]["predicate_results"]["required"])


# -- projecting the retained modules' evidence into the frozen shapes -------- #


def _content_member(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "length": member["length"],
        "path": member["path"],
        "sha256": member["sha256"],
    }


def declared_content(projection: Mapping[str, Any]) -> tuple[list[dict], dict]:
    """Split the declared content set into the two members the binding records.

    The bytes come from the RunSpec's own projection rather than from a second
    walk of the tree: the binding's claim is that the execution ran *these*
    bytes, and re-hashing them here would let the two documents disagree about
    what "these" means.
    """
    members = {str(item["path"]): item for item in projection["members"]}
    surfaces: list[dict[str, Any]] = []
    for relative in sorted(executed_surface_paths()):
        member = members.get(relative)
        if member is None:
            raise ProducerError(
                f"the declared content set does not name the executed surface {relative}"
            )
        surfaces.append(_content_member(member))
    abi = members.get(capture_abi_path())
    if abi is None:
        raise ProducerError("the declared content set does not name the capture ABI")
    return surfaces, _content_member(abi)


def build_artifacts_from_manifest(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The two roles `h2_runtime_inputs` already records, in the binding's shape."""
    artifacts = [
        {
            "length": item["length"],
            "path": _relative(item["resolved_path"]),
            "role": item["role"],
            "sha256": item["sha256"],
        }
        for item in section["files"]
    ]
    roles = sorted(item["role"] for item in artifacts)
    if roles != ["tensorrt_scan_plugin", "tracking_extension"]:
        raise ProducerError(f"build artifacts do not carry both roles: {roles}")
    return sorted(artifacts, key=lambda item: str(item["role"]))


def _relative(path: str) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return candidate.name


def extension_load_from_witness(witness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "length": witness["extension_length"],
        "loaded_path": witness["extension_path"],
        "sha256": witness["extension_sha256"],
    }


def identity_probe_record(
    probe: Mapping[str, Any], *, build_artifact_digest: str
) -> dict[str, Any]:
    """A recorded observation. Never an equivalence oracle — the role says so."""
    return {
        "build_artifact_digest": build_artifact_digest,
        "digest": probe["digest"],
        "role": PROBE_ROLE,
        "schema": PROBE_SCHEMA,
        "state": PROBE_STATE,
    }


def runtime_input_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    members = [
        {
            "length": item["length"],
            "path": _relative(item["resolved_path"]),
            "role": item["role"],
            "sha256": item["sha256"],
        }
        for section in manifest["sections"]
        for item in manifest[section]["files"]
    ]
    return {
        "manifest_digest": manifest["coordinate_digest"],
        "manifest_schema": runtime_inputs.SCHEMA,
        "members": sorted(members, key=lambda item: (item["role"], item["path"])),
    }


# -- the two producer artifacts this module assembles ----------------------- #


@dataclass(frozen=True)
class StageEvidence:
    """What the six retained stages produced, and where they stopped.

    A stage failure is not an error here: it is the observation. The schema
    forbids a binding that failed at `build` from carrying an extension load or a
    probe, which is the same rule stated as a data shape.
    """

    input_monitor: Mapping[str, Any]
    source_audit: Mapping[str, Any]
    runtime_inputs: Mapping[str, Any]
    failed_stage: str | None = None
    build_artifacts: Sequence[Mapping[str, Any]] | None = None
    extension_load: Mapping[str, Any] | None = None
    identity_probe: Mapping[str, Any] | None = None


def build_runtime_binding(
    *, execution_id: str, run_spec: Mapping[str, Any], stages: StageEvidence
) -> dict[str, Any]:
    """Assemble `runtime_binding.json` and refuse to emit one the schema rejects."""
    surfaces, abi = declared_content(run_spec["execution_semantics_projection"])
    document: dict[str, Any] = {
        "capture_abi": abi,
        "executed_surfaces": surfaces,
        "execution_id": execution_id,
        "execution_semantics_projection_digest": run_spec[
            "execution_semantics_projection_digest"
        ],
        "failed_stage": stages.failed_stage,
        "input_monitor": dict(stages.input_monitor),
        "resolved_run_spec_digest": run_spec["resolved_run_spec_digest"],
        "runtime_inputs": dict(stages.runtime_inputs),
        "schema": BINDING_SCHEMA,
        "source_audit": dict(stages.source_audit),
    }
    if stages.build_artifacts is not None:
        document["build_artifacts"] = [dict(item) for item in stages.build_artifacts]
    if stages.extension_load is not None:
        document["extension_load"] = dict(stages.extension_load)
    if stages.identity_probe is not None:
        document["identity_probe"] = dict(stages.identity_probe)
    _validate(document, "runtime_binding.json")
    return document


def build_result(
    *,
    execution_id: str,
    run_spec: Mapping[str, Any],
    authority: str,
    authorization_binding_digest: str | None,
    predicate_results: Mapping[str, Any],
    ordered_runs: Sequence[Mapping[str, Any]],
    execution_result: str | None = None,
) -> dict[str, Any]:
    """Assemble `result.json`. The verdict is selected by the ruler, not written here.

    `execution_result` is the one thing a producer may name that the predicates
    cannot derive: which cause put the execution in terminal 4. Everything else
    in the verdict — the result token and the terminal — comes back from
    `select_successor_result` and is transcribed.
    """
    selection = partition.select_successor_result(
        predicate_results,
        authority=authority,
        phase=PHASE,
        execution_result=execution_result,
    )
    document = {
        "authority": authority,
        "authorization_binding_digest": authorization_binding_digest,
        "execution_id": execution_id,
        "execution_semantics_projection_digest": run_spec[
            "execution_semantics_projection_digest"
        ],
        "ordered_runs": [dict(run) for run in ordered_runs],
        "predicate_results": {
            name: dict(record) for name, record in predicate_results.items()
        },
        "resolved_run_spec_digest": run_spec["resolved_run_spec_digest"],
        "result": selection.result,
        "run_plan": {"run_ids": list(run_ids()), "sequence": sequence()},
        "schema": RESULT_SCHEMA,
        "terminal": selection.terminal,
    }
    _validate(document, "result.json")
    return document


def _validate(document: Mapping[str, Any], name: str) -> None:
    import jsonschema

    schema = archive._load_contract(archive.PRODUCER_ARTIFACTS[name])
    try:
        jsonschema.validate(instance=document, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ProducerError(f"{name} would violate its own contract: {exc.message}")


# -- emission --------------------------------------------------------------- #


def emit_archive(
    root: Path,
    *,
    run_spec: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Write the three producer artifacts, and nothing else, exactly once.

    No `verification.json` and no `checksums.sha256`: the closure belongs to the
    independent verifier, and a producer that wrote either would be closing an
    archive over its own claim about itself.
    """
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(item.name for item in root.iterdir())
    if existing:
        raise ProducerError(
            f"execution archive root is not empty, so this execution would "
            f"overwrite another: {existing}"
        )
    written = []
    for name, document in (
        ("run_spec.json", run_spec),
        ("runtime_binding.json", runtime_binding),
        ("result.json", result),
    ):
        written.append(evidence.write_document_exclusive(root, name, document))
    return tuple(written)


# -- the sequencing this module owns ---------------------------------------- #


class Stages(Protocol):
    """The six retained stages, in order, as one injectable surface."""

    def run(self) -> StageEvidence: ...


class Runs(Protocol):
    """The four ordered measurement runs, and the predicates they decide."""

    def run(
        self, stages: StageEvidence
    ) -> tuple[list[dict], dict[str, Any], str | None]:
        """Return `(ordered_runs, predicate_results, execution_result)`."""


def unstarted_runs() -> list[dict[str, Any]]:
    """What the ordered runs are when a stage failure stopped the execution."""
    return [
        {"artifact_digest": None, "run_id": run_id, "state": "not_run"}
        for run_id in run_ids()
    ]


def undecided_predicates(*, decided: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every predicate `not_run` unless a stage decided it before the runs began."""
    states = dict(decided or {})
    return {
        name: {"reasons": [], "state": states.get(name, "not_run")}
        for name in predicate_names()
    }


@dataclass
class Execution:
    """One successor execution: stages, then runs, then three artifacts."""

    execution_id: str
    authority: str
    stages: Stages
    runs: Runs
    authorization_binding_digest: str | None = None
    run_spec: Mapping[str, Any] | None = None

    def produce(self, root: Path) -> dict[str, Any]:
        if self.authority not in partition.AUTHORITIES:
            raise ProducerError(f"unknown authority: {self.authority!r}")
        spec = (
            self.run_spec
            if self.run_spec is not None
            else run_spec_module.build_run_spec()
        )

        evidence_record = self.stages.run()
        if evidence_record.failed_stage is not None:
            # Fail-fast is not a shortcut: no run started, so nothing the runs
            # would have decided may be reported as decided.
            ordered, predicates, named = (
                unstarted_runs(),
                undecided_predicates(
                    decided=_stage_decided(evidence_record.input_monitor)
                ),
                _stage_result(evidence_record.failed_stage),
            )
        else:
            ordered, predicates, named = self.runs.run(evidence_record)

        observation = _snapshot(predicates)
        binding = build_runtime_binding(
            execution_id=self.execution_id, run_spec=spec, stages=evidence_record
        )
        result = build_result(
            execution_id=self.execution_id,
            run_spec=spec,
            authority=self.authority,
            authorization_binding_digest=self.authorization_binding_digest,
            predicate_results=observation,
            ordered_runs=ordered,
            execution_result=named
            if _may_name_a_cause(observation, authority=self.authority)
            else None,
        )
        emit_archive(root, run_spec=spec, runtime_binding=binding, result=result)
        return result


def _stage_decided(monitor: Mapping[str, Any]) -> dict[str, str]:
    """What a stage failure decides: the execution stopped, and what the monitor saw.

    `execution_complete` obviously. But the monitor is *stage-independent* — it
    starts before the binding by contract — so a change it recorded is decided
    too, and terminal 1 outranks the stage that failed. Reporting only the stage
    would emit an archive claiming `build_failed` beside a recorded mutation,
    which the ruler refuses and the verifier would reject. Every other predicate
    stays `not_run`: the runs that would have decided them never started.
    """
    decided = {"execution_complete": "fail"}
    mutated = bool(monitor.get("changed_count")) or not bool(
        monitor.get("final_drain_clean")
    )
    if mutated:
        decided[partition.SUCCESSOR_PREDICATES[0][0]] = "fail"
    return decided


def _snapshot(predicate_results: Mapping[str, Any]) -> Mapping[str, Any]:
    """One frozen observation, selected on twice.

    The verdict is selected twice — once unnamed, to ask the ruler whether a
    cause may be named at all, and once carrying that cause. Both calls must see
    the *same* observation, or the second could return a verdict the first never
    authorised. Passing the same object is already true of the call site above,
    but only as an accident of statement order; copying it here makes it a
    property of the value. The copy also drops the driver's aliases: whatever a
    `Runs` implementation still holds a reference to can no longer reach the
    artifact between the two selections, or after either of them.

    Records that are not mappings pass through untouched — the ruler decides
    what an unusable record is, and it says so in its own words.
    """
    return MappingProxyType(
        {
            name: copy.deepcopy(dict(record)) if isinstance(record, Mapping) else record
            for name, record in predicate_results.items()
        }
    )


def _may_name_a_cause(predicate_results: Mapping[str, Any], *, authority: str) -> bool:
    """Ask the ruler where this observation lands before offering it a cause.

    Naming terminal 4's cause when something else won is a caller defect the
    selector refuses outright, and the two ways to reach that mistake — a higher
    finding, or a diagnostic, which selects no terminal at all — are both
    answered by asking for the unnamed selection first. Deriving the answer here
    instead would be this file holding a rule about precedence.
    """
    selection = partition.select_successor_result(
        predicate_results, authority=authority, phase=PHASE
    )
    return selection.terminal == partition.EXECUTION_INVALID_TERMINAL


def _stage_result(failed_stage: str) -> str:
    """Which terminal-4 cause a failed stage names — the ruler owns the mapping."""
    for result, stage in partition.RESULT_REQUIRES_FAILED_STAGE.items():
        if stage == failed_stage:
            return result
    return "unclassified_execution_failure"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execution-id", required=True)
    parser.add_argument(
        "--authority", required=True, choices=sorted(partition.AUTHORITIES)
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--emit-run-spec-only",
        action="store_true",
        help="issue and print the RunSpec without executing anything",
    )
    args = parser.parse_args(argv)
    if args.emit_run_spec_only:
        print(json.dumps(run_spec_module.build_run_spec(), indent=2, sort_keys=True))
        return 0
    print(
        "no execution driver is bound: the stage and run surfaces are injected, "
        "and binding them to a build is a separate landing",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
