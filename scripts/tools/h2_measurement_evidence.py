#!/usr/bin/env python3
"""The H2 Layer-M evidence-root contract: names, records, and the inventory.

This module is `plumbing_only` (§ 5.3, § C3.9) and it must stay that way. It
holds **no** ruler: no terminal condition, no admission condition, no phase
completion count, no predicate name of its own. Every one of those is imported
from `h2_terminal_partition`, which is the identity-semantics authority for them
(`IDENTITY_SEMANTICS_PATHS` in `h2_path_partition.py`). C3.9 pins the trap this
avoids: a *new* `scripts/tools/h2_*.py` file classifies as plumbing, so ruler
logic placed here would move the ruler inside the frozen window with no check
firing.

What the module does own is the shape of an evidence root — which is bound into
`F_B` as executed code (C3.2 item 6) rather than published on an axis:

```text
h2_measure_<I40>                     phase A   (§ 9 item 3)
h2_measure_b_<I40_B>_<F64>           phase B   (§ C3.1; complete digest, never
                                               a truncation)
    manifest.json                    capture_phase, artifact inventory, digests
    checksums.sha256                 <sha256>  <relative path>, H0's format
    freeze.json                      the F record this root was launched against
    admission.json                   phase B: the § C3.6 verdict (pre-terminal),
                                     recomputed by the verifier, never trusted
    layer_p_certificate.json         the certificate F binds, archived so the
                                     § C3.6(d) condition can be recomputed
    reference_probe.json             the Layer-P probe-result file F binds
    launch_probe.json                the launch-time probe used by terminal 1
    runtime_inputs.json              the complete bound-input manifest
    published_identity.json          the coordinate/probe publication F binds
    checkout_identity_witness.json   source tree and all three content axes
    mutation_observation.json        the BoundInputMonitor record
    measurement_stop_boundary.json   monitor-stop and post-close revalidation
    authorization_consumed.json      phase B: the § C3.5.1 step-5 write that
                                     *is* the consumption of S_B
    observation.json                 exactly ORDERED_PREDICATES (+ optional
                                     execution_result)
    terminal.json                    the recorded selection
    runs/<sequence>/<run id>/        policy_inventory.json, packet.json,
                                     packet_verification.json
    runs/<sequence>/comparison.json  the A7.6 reconstruction for that sequence
```

The evidence prefix is deliberately not `h0_phase_a_*` (§ 9 item 3), so
`check_h0_phase_a_archives.py` keeps verifying the frozen v1 corpus under the v1
schema without ever seeing an H2 root.

Usage:
  uv run python scripts/tools/h2_measurement_evidence.py --describe
  uv run python scripts/tools/h2_measurement_evidence.py --inventory <root>
"""
# status: stable

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_terminal_partition as partition  # noqa: E402
from h2_behavioral_identity import (  # noqa: E402
    A76_POLICY_INVENTORY_SCHEMA,
    MEASUREMENT_SEQUENCE,
)

# One digest convention for the whole unit (§ 8.1, H0's
# `h0_phase_a_execution_v1` convention consumed unchanged). Imported rather than
# re-implemented so a second convention cannot appear by copy.
from h2_runtime_inputs import (  # noqa: E402
    MEASUREMENT_SEQUENCES,
    canonical_json_bytes,
    digest,
    sha256_file,
)

MANIFEST_SCHEMA = "h2_measurement_evidence_v1"
OBSERVATION_SCHEMA = "h2_measurement_observation_v1"
TERMINAL_SCHEMA = "h2_terminal_selection_v1"
FREEZE_SCHEMA = "h2_measurement_freeze_v1"
ADMISSION_SCHEMA = "h2_admission_verdict_v1"
AUTHORIZATION_SCHEMA = "h2_authorization_consumed_v1"
CONTROLLER_SCHEMA = "h2_measurement_controller_v1"
MUTATION_SCHEMA = "h2_bound_input_mutation_v1"
CHECKOUT_WITNESS_SCHEMA = "h2_checkout_identity_witness_v1"
STOP_BOUNDARY_SCHEMA = "h2_measurement_stop_boundary_v1"

# Re-exported, never redeclared: the schema identifier is a ruler fact and lives
# in `h2_behavioral_identity.py`, which owns the A7.6 member definitions (§ 4).
POLICY_INVENTORY_SCHEMA = A76_POLICY_INVENTORY_SCHEMA

EVIDENCE_REL = "docs/modules/semantic/research/evidence"

PHASE_A_ROOT_PREFIX = "h2_measure_"
PHASE_B_ROOT_PREFIX = "h2_measure_b_"

# § C3.2 item 9's vocabulary, in H0's own frozen terms. `capture_phase` is a
# required manifest field (§ C3.1).
CAPTURE_PHASE: dict[str, str] = {"a": "phase_a", "b": "phase_b"}
PHASE_BY_CAPTURE_PHASE: dict[str, str] = {
    value: key for key, value in CAPTURE_PHASE.items()
}

# The § 3.3 four-run block, in the declared order.
RUN_IDS: tuple[str, ...] = (
    "00_capture_off",
    "01_capture_on",
    "02_capture_on",
    "03_capture_on",
)
CAPTURE_OFF_RUN = RUN_IDS[0]
CAPTURE_ON_RUNS = RUN_IDS[1:]

# Phase A runs the § 3.3 measurement fixture alone; Phase B runs the § C3.2
# item 10 plan. Both come from modules that already own the value.
PHASE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "a": (MEASUREMENT_SEQUENCE,),
    "b": MEASUREMENT_SEQUENCES,
}

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
FREEZE_NAME = "freeze.json"
ADMISSION_NAME = "admission.json"
AUTHORIZATION_NAME = "authorization_consumed.json"
OBSERVATION_NAME = "observation.json"
TERMINAL_NAME = "terminal.json"
COMPARISON_NAME = "comparison.json"
POLICY_INVENTORY_NAME = "policy_inventory.json"
PACKET_NAME = "packet.json"
PACKET_VERIFICATION_NAME = "packet_verification.json"
CERTIFICATE_NAME = "layer_p_certificate.json"
REFERENCE_PROBE_NAME = "reference_probe.json"
LAUNCH_PROBE_NAME = "launch_probe.json"
RUNTIME_INPUTS_NAME = "runtime_inputs.json"
PUBLISHED_IDENTITY_NAME = "published_identity.json"
MUTATION_NAME = "mutation_observation.json"
CHECKOUT_WITNESS_NAME = "checkout_identity_witness.json"
STOP_BOUNDARY_NAME = "measurement_stop_boundary.json"
CONTROLLER_NAME = "controller.json"

RUNS_DIR = "runs"


class EvidenceError(RuntimeError):
    """The evidence root does not satisfy the contract. Always fail-closed."""


class RootName(NamedTuple):
    phase: str
    i40: str
    freeze_digest: str | None


def _hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def phase_a_root_name(i40: str) -> str:
    if not _hex(i40, 40):
        raise EvidenceError(f"phase-A head is not 40 lowercase hex: {i40!r}")
    return f"{PHASE_A_ROOT_PREFIX}{i40}"


def phase_b_root_name(i40: str, freeze_digest: str) -> str:
    if not _hex(i40, 40):
        raise EvidenceError(f"phase-B head is not 40 lowercase hex: {i40!r}")
    if not _hex(freeze_digest, 64):
        # § C3.1: the complete digest, never a truncation — an evidence root is
        # an identity, and shortening it trades collision probability for path
        # cosmetics.
        raise EvidenceError(
            f"F_B digest is not the complete 64 lowercase hex: {freeze_digest!r}"
        )
    return f"{PHASE_B_ROOT_PREFIX}{i40}_{freeze_digest}"


def parse_root_name(name: str) -> RootName:
    """Total over the `h2_measure_` family; anything else is not an H2 root."""
    if name.startswith(PHASE_B_ROOT_PREFIX):
        body = name[len(PHASE_B_ROOT_PREFIX) :]
        head, _, freeze = body.partition("_")
        if not _hex(head, 40) or not _hex(freeze, 64):
            raise EvidenceError(f"malformed phase-B evidence root name: {name!r}")
        return RootName("b", head, freeze)
    if name.startswith(PHASE_A_ROOT_PREFIX):
        head = name[len(PHASE_A_ROOT_PREFIX) :]
        if not _hex(head, 40):
            raise EvidenceError(f"malformed phase-A evidence root name: {name!r}")
        return RootName("a", head, None)
    raise EvidenceError(f"not an H2 measurement evidence root: {name!r}")


def freeze_digest(freeze_record: Mapping[str, Any]) -> str:
    """The F64 of § C3.1, over the canonical bytes of the freeze record."""
    return digest(freeze_record)


# -- the observation emitter (charter S4 item 8) --------------------------- #


def build_observation(
    predicates: Mapping[str, Any], *, execution_result: str | None = None
) -> dict[str, Any]:
    """Emit exactly `ORDERED_PREDICATES`, so no unexpressible terminal is claimed.

    Fail-closed in both directions: a missing predicate is not defaulted and an
    extra key is not carried. The controller therefore cannot record a
    predicate the partition does not read, and cannot omit one it does — the
    observation is the whole interface between execution and the ruler.
    """
    expected = tuple(key for key, _ in partition.ORDERED_PREDICATES)
    missing = [key for key in expected if key not in predicates]
    if missing:
        raise EvidenceError(f"observation is missing predicates: {missing}")
    extra = sorted(set(predicates) - set(expected))
    if extra:
        raise EvidenceError(
            f"observation carries keys the partition does not define: {extra}"
        )
    document: dict[str, Any] = {"schema": OBSERVATION_SCHEMA}
    for key in expected:
        value = predicates[key]
        if not isinstance(value, bool):
            raise EvidenceError(f"predicate {key} is not a bool: {value!r}")
        document[key] = value
    if execution_result is not None:
        # Every legal name maps to terminal 4, so naming the cause records it
        # without letting the name change the terminal.
        if (
            partition.RESULT_TO_TERMINAL.get(execution_result)
            != partition.EXECUTION_INVALID_TERMINAL
        ):
            raise EvidenceError(
                f"execution_result {execution_result!r} does not map to terminal 4"
            )
        document["execution_result"] = execution_result
    return document


def observation_predicates(document: Mapping[str, Any]) -> dict[str, Any]:
    """The partition's view of a recorded observation: predicates + cause only."""
    keys = {key for key, _ in partition.ORDERED_PREDICATES} | {"execution_result"}
    return {key: value for key, value in document.items() if key in keys}


# -- records and the checksum inventory ------------------------------------ #


def write_document(root: Path, name: str, payload: Mapping[str, Any]) -> Path:
    """Write one canonical record, flushed, and return its path.

    Flushing matters for exactly one record — § C3.5.1 step 5, where the durable
    write *is* the consumption of `S_B` — so every record takes the same path
    rather than leaving the load-bearing one as a special case.
    """
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_document(
    root: Path, name: str, *, schema: str | None = None
) -> dict[str, Any]:
    path = root / name
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"required record is unreadable: {name} ({exc})") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"record is not JSON: {name} ({exc})") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"record is not an object: {name}")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise EvidenceError(f"record is not in canonical form: {name}")
    if schema is not None and payload.get("schema") != schema:
        raise EvidenceError(
            f"record {name} declares schema {payload.get('schema')!r}, expected {schema!r}"
        )
    return payload


def evidence_files(root: Path) -> tuple[Path, ...]:
    """Every regular file in the root except the inventory itself, sorted.

    Symlinks are refused rather than followed: an evidence root that can point
    outside itself is not an immutable artifact.
    """
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode()):
        if path.is_symlink():
            raise EvidenceError(
                f"evidence root contains a symlink: {path.relative_to(root).as_posix()}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError(
                f"evidence root contains a non-regular file: "
                f"{path.relative_to(root).as_posix()}"
            )
        if path.name == CHECKSUMS_NAME and path.parent == root:
            continue
        files.append(path)
    return tuple(files)


def write_checksum_inventory(root: Path) -> Path:
    """H0's `<sha256>  <relative path>` format, sorted by encoded path."""
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in evidence_files(root)
    ]
    path = root / CHECKSUMS_NAME
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_checksum_inventory(root: Path) -> dict[str, str]:
    path = root / CHECKSUMS_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"checksum inventory is unreadable: {exc}") from exc
    inventory: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise EvidenceError(f"checksum inventory line {number} is empty")
        checksum, separator, relative = line.partition("  ")
        if not separator or not _hex(checksum, 64) or not relative:
            raise EvidenceError(f"checksum inventory line {number} is malformed")
        if relative in inventory:
            raise EvidenceError(f"checksum inventory names {relative} twice")
        inventory[relative] = checksum
    return inventory


def verify_checksum_inventory(root: Path) -> dict[str, str]:
    """Total both ways: every file inventoried, every inventoried file present."""
    inventory = read_checksum_inventory(root)
    present = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in evidence_files(root)
    }
    missing = sorted(set(inventory) - set(present))
    unlisted = sorted(set(present) - set(inventory))
    if missing:
        raise EvidenceError(f"checksum inventory names absent files: {missing}")
    if unlisted:
        raise EvidenceError(f"evidence files are absent from the inventory: {unlisted}")
    moved = sorted(name for name, value in present.items() if inventory[name] != value)
    if moved:
        raise EvidenceError(f"evidence bytes differ from the inventory: {moved}")
    return present


def checksum_inventory_digest(root: Path) -> str:
    """The inventory digest `F_B` binds (§ C3.2 item 1)."""
    return sha256_file(root / CHECKSUMS_NAME)


def run_dir(root: Path, sequence: str, run_id: str) -> Path:
    return root / RUNS_DIR / sequence / run_id


def sequence_dirs(root: Path) -> tuple[Path, ...]:
    runs = root / RUNS_DIR
    if not runs.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(runs.iterdir(), key=lambda item: item.name.encode())
        if path.is_dir()
    )


def expected_sequences(phase: str) -> tuple[str, ...]:
    if phase not in PHASE_SEQUENCES:
        raise EvidenceError(f"unknown phase: {phase!r}")
    return PHASE_SEQUENCES[phase]


def completion(phase: str) -> dict[str, int]:
    """The ruler's counts, read from the partition — never restated here."""
    if phase not in partition.PHASE_COMPLETION:
        raise EvidenceError(f"unknown phase: {phase!r}")
    return dict(partition.PHASE_COMPLETION[phase])


def describe() -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "capture_phase": CAPTURE_PHASE,
        "phase_completion": {
            phase: completion(phase) for phase in partition.PHASE_COMPLETION
        },
        "phase_sequences": {
            phase: list(value) for phase, value in PHASE_SEQUENCES.items()
        },
        "records": {
            "manifest": MANIFEST_NAME,
            "checksums": CHECKSUMS_NAME,
            "freeze": FREEZE_NAME,
            "admission": ADMISSION_NAME,
            "authorization_consumed": AUTHORIZATION_NAME,
            "controller": CONTROLLER_NAME,
            "launch_probe": LAUNCH_PROBE_NAME,
            "mutation_observation": MUTATION_NAME,
            "observation": OBSERVATION_NAME,
            "published_identity": PUBLISHED_IDENTITY_NAME,
            "reference_probe": REFERENCE_PROBE_NAME,
            "runtime_inputs": RUNTIME_INPUTS_NAME,
            "terminal": TERMINAL_NAME,
        },
        "run_ids": list(RUN_IDS),
        "root_prefixes": {"a": PHASE_A_ROOT_PREFIX, "b": PHASE_B_ROOT_PREFIX},
        "note": (
            "plumbing_only: this module holds no terminal, admission or phase "
            "completion definition of its own; all of them are imported from "
            "h2_terminal_partition."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--describe", action="store_true", help="print the evidence-root contract"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="verify a root's checksum inventory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.describe and args.inventory is None:
        parser.error("one of --describe or --inventory is required")
    if args.describe:
        print(canonical_json_bytes(describe()).decode("utf-8"))
    if args.inventory is not None:
        try:
            present = verify_checksum_inventory(args.inventory)
        except (EvidenceError, OSError) as exc:
            print(f"checksum inventory rejected: {exc}", file=sys.stderr)
            return 1
        print(f"checksum inventory: PASS ({len(present)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
