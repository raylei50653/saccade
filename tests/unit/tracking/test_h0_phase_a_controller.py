"""Hermetic A7/RC1 controller, child-contract, and verifier admissions."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import run_h0_phase_a as parent  # noqa: E402
import run_h0_phase_a_child as child  # noqa: E402
import h0_runtime_confinement as runtime_confinement  # noqa: E402
import verify_h0_phase_a as verifier  # noqa: E402
import export_headline_bridge_decision_trace as trace_export  # noqa: E402
import verify_headline_bridge_decision_trace as trace_verifier  # noqa: E402


def _sequence() -> dict[str, object]:
    content = b"fixture"
    files = [
        {
            "length": len(content),
            "path": "seqinfo.ini",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    ]
    return {
        "algorithm": "h0_sequence_inputs_v1",
        "digest": verifier.digest(
            {"algorithm": "h0_sequence_inputs_v1", "files": files}
        ),
        "files": files,
        "root": parent.SEQUENCE_REL,
    }


def _bound_inputs() -> dict[str, object]:
    models = [
        {
            "length": 1,
            "logical_path": logical_path,
            "realpath": f"/fixture/{index}",
            "sha256": hashlib.sha256(bytes([index])).hexdigest(),
            "symlink_chain": [],
        }
        for index, logical_path in enumerate(parent.MODEL_LOGICAL_PATHS, start=1)
    ]
    repository = []
    for path in parent.REQUIRED_REPOSITORY_INPUTS:
        data = path.encode("utf-8")
        repository.append(
            {
                "git_object": hashlib.sha1(data, usedforsecurity=False).hexdigest(),
                "git_type": "blob",
                "kind": "regular",
                "length": len(data),
                "mode": "100644",
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    value: dict[str, object] = {
        "digest": "0" * 64,
        "models_engines": models,
        "repository": repository,
        "schema": "h0_bound_inputs_v1",
        "sequence": _sequence(),
        "tool_runtime": [],
    }
    value["digest"] = parent.bound_inventory_digest(value)
    return value


def _controller() -> dict[str, object]:
    head = "a" * 40
    evidence = f"docs/modules/semantic/research/evidence/h0_phase_a_{head}"
    bound = _bound_inputs()
    return {
        "bound_inputs": bound,
        "document_type": "controller_input",
        "evidence_root": evidence,
        "execution_constants": parent.execution_constants(ROOT),
        "gpu": {
            "compute_capability": "9.0",
            "driver": "fixture-driver",
            "name": "fixture-gpu",
            "normalized_pci_bus_id": "0000:01:00.0",
            "total_memory": 1,
            "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "vbios": "fixture-vbios",
        },
        "incomplete_root": evidence + ".incomplete",
        "instrumentation_head": head,
        "library_dirs": {
            "cuda_library_dir": "/opt/cuda/lib64",
            "pytorch_library_dir": f"{ROOT}/.venv/lib/python3.12/site-packages/torch/lib",
            "tensorrt_library_dir": f"{ROOT}/.venv/lib/python3.12/site-packages/tensorrt_libs",
        },
        "repository_root": ROOT.as_posix(),
        "schema": "h0_phase_a_controller_v1",
        "sequence_input_digest": bound["sequence"]["digest"],
        "tool_paths": {
            "git": "/usr/bin/git",
            "ldd": "/usr/bin/ldd",
            "readelf": "/usr/bin/readelf",
            "uv": "/usr/bin/uv",
        },
    }


def _environment(controller: dict[str, object], run_id: str) -> dict[str, str]:
    root = controller["repository_root"]
    run_dir = f"{root}/{controller['incomplete_root']}/runs/{run_id}"
    temp = f"{run_dir}/_env"
    libraries = controller["library_dirs"]
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": controller["gpu"]["uuid"],
        "HOME": f"{temp}/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(
            (
                f"{root}/build/h0_phase_a",
                libraries["tensorrt_library_dir"],
                libraries["pytorch_library_dir"],
                libraries["cuda_library_dir"],
            )
        ),
        "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SACCADE_BUILD_PATH": f"{root}/build/h0_phase_a",
        "SACCADE_DETECT_BARRIER": "event",
        "SACCADE_DOUBLE_BUFFER": "1",
        "SACCADE_GPU_DECODE": "1",
        "SACCADE_MAIN_NMS_GRAPHED": "1",
        "TMPDIR": f"{temp}/tmp",
        "TZ": "UTC",
        "XDG_CACHE_HOME": f"{temp}/xdg-cache",
    }


def _children(controller: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for index, run_id in enumerate(parent.RUN_IDS):
        environment = _environment(controller, run_id)
        run_dir = f"{controller['repository_root']}/{controller['incomplete_root']}/runs/{run_id}"
        result.append(
            {
                "bound_inputs_digest": controller["bound_inputs"]["digest"],
                "capture_run_uuid": f"00000000-0000-4000-8000-{index:012d}",
                "confinement_backend": "landlock_seccomp_ptrace_v1",
                "confinement_plan_digest": "1" * 64,
                "confinement_probe_passed": True,
                "document_type": "child_invocation",
                "environment": environment,
                "environment_digest": verifier.digest(environment),
                "evaluator_argv": list(parent.EVALUATOR_ARGV_PREFIX)
                + [f"{run_dir}/_runtime"],
                "incomplete_root": controller["incomplete_root"],
                "instrumentation_head": controller["instrumentation_head"],
                "result": "run_completed",
                "run_id": run_id,
                "runtime_inputs_digest": "2" * 64,
                "schema": "h0_phase_a_child_v1",
                "state": "completed",
                "vector": list(parent.child_argv(ROOT, run_id)),
            }
        )
    return result


def _mark_not_run(invocation: dict[str, object]) -> None:
    invocation["confinement_plan_digest"] = None
    invocation["confinement_probe_passed"] = None
    invocation["result"] = None
    invocation["runtime_inputs_digest"] = None
    invocation["state"] = "not_run"


def _predicates() -> dict[str, bool]:
    return {
        "artifacts_ok": True,
        "build_ok": True,
        "classified_execution": True,
        "extension_ok": True,
        "packets_valid": True,
        "policy_equal": True,
        "provenance_ok": True,
        "runners_ok": True,
        "serialization_ok": True,
        "timed_out": False,
    }


def _binding(controller: dict[str, object]) -> dict[str, object]:
    digest_value = controller["bound_inputs"]["digest"]
    return {
        "algorithm": "h0_bound_inputs_v1",
        "checkpoints": [
            {
                "digest": digest_value,
                "events_after": [],
                "events_before": [],
                "inventory_equal": True,
                "monotonic_ns": index,
                "name": name,
                "state": "completed",
            }
            for index, name in enumerate(parent.CHECKPOINTS)
        ],
        "final_equal": True,
        "inotify_mask": list(parent.INOTIFY_MASK_NAMES),
        "monitor_state": "closed_clean",
        "mutation_events": [],
        "t0_digest": digest_value,
    }


def evidence_for(result: str = "phase_a_pass") -> dict[str, object]:
    controller = _controller()
    predicates = _predicates()
    if result == "provenance_invalid":
        predicates["provenance_ok"] = False
    elif result == "build_failed":
        predicates["build_ok"] = False
    elif result == "extension_load_failed":
        predicates["extension_ok"] = False
    elif result == "runner_nonzero":
        predicates["runners_ok"] = False
    elif result == "runner_timeout":
        predicates["timed_out"] = True
    elif result == "serialization_failed":
        predicates["serialization_ok"] = False
    elif result == "artifact_missing_or_unreadable":
        predicates["artifacts_ok"] = False
    elif result == "unclassified_execution_failure":
        predicates["classified_execution"] = False
    elif result == "capture_perturbs_policy":
        predicates["policy_equal"] = False
    elif result == "packet_invalid":
        predicates["packets_valid"] = False
    elif result != "phase_a_pass":
        raise AssertionError(result)
    matrix = parent.execution_constants(ROOT)["result_matrix"]
    policy = "not_produced"
    packets = ["not_produced"] * 3
    if result == "capture_perturbs_policy":
        policy = "unequal"
    elif result == "packet_invalid":
        policy, packets = "equal", ["pass", "fail", "pass"]
    elif result == "phase_a_pass":
        policy, packets = "equal", ["pass"] * 3
    children = _children(controller)
    if result in {"provenance_invalid", "build_failed", "extension_load_failed"}:
        for invocation in children:
            invocation["confinement_plan_digest"] = None
            invocation["confinement_probe_passed"] = None
            invocation["runtime_inputs_digest"] = None
            invocation["state"] = "not_run"
            invocation["result"] = None
    elif result in {"runner_nonzero", "runner_timeout"}:
        children[0]["state"] = "failed"
        children[0]["result"] = result
        for invocation in children[1:]:
            invocation["confinement_plan_digest"] = None
            invocation["confinement_probe_passed"] = None
            invocation["runtime_inputs_digest"] = None
            invocation["state"] = "not_run"
            invocation["result"] = None
    return {
        "artifact_inventory": list(matrix[result]["required"]),
        "child_invocations": children,
        "controller_input": controller,
        "decision_predicates": predicates,
        "document_type": "execution_evidence",
        "input_binding": _binding(controller),
        "packet_verification_states": packets,
        "policy_comparison": policy,
        "result": result,
        "result_matrix": matrix,
        "schema": "h0_phase_a_execution_v1",
    }


def test_canonical_dry_run_contract_passes_strict_schema() -> None:
    parent.validate_schema_document(_controller(), "controller_input")


def test_parent_and_child_vectors_are_unique_and_identical() -> None:
    controller = _controller()
    children = _children(controller)
    assert [entry["vector"] for entry in children] == controller["execution_constants"][
        "child_vectors"
    ]
    assert len({tuple(entry["vector"]) for entry in children}) == 4


def test_parent_parser_has_no_execution_options_or_positionals() -> None:
    assert parent._parse_no_options(()) is True
    assert parent._parse_no_options(("--help",)) is False
    for argv in (("--dry-run",), ("manifest.json",), ("--preset", "other")):
        with pytest.raises(parent.ContractError):
            parent._parse_no_options(argv)


def test_parent_rejects_working_directory_drift_before_host_inspection(
    tmp_path: Path,
) -> None:
    with pytest.raises(parent.ContractError, match="physical cwd"):
        parent.preflight_controller_input({}, tmp_path.resolve())


def test_environment_and_digests_rebuild() -> None:
    controller = _controller()
    parent.validate_bound_inventory(controller["bound_inputs"])
    for invocation in _children(controller):
        assert invocation["environment_digest"] == parent.environment_digest(
            invocation["environment"]
        )


def test_build_environment_is_exact_and_ignores_host_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    contract["tool_paths"]["uv"] = "/usr/bin/true"
    incomplete = tmp_path / contract["incomplete_root"]
    (incomplete / "logs").mkdir(parents=True)
    expected = parent._create_build_environment(contract)
    selectors = {
        "CC": "/host/cc",
        "CFLAGS": "-DHOST",
        "CMAKE_GENERATOR": "Host Generator",
        "CMAKE_PREFIX_PATH": "/host/prefix",
        "CMAKE_TOOLCHAIN_FILE": "/host/toolchain.cmake",
        "CUDA_HOME": "/host/cuda",
        "CUDACXX": "/host/nvcc",
        "CXX": "/host/cxx",
        "CXXFLAGS": "-DHOST_CXX",
        "LDFLAGS": "-L/host/lib",
        "LD_PRELOAD": "/host/inject.so",
        "NVCC_PREPEND_FLAGS": "--host-flag",
    }
    for key, value in selectors.items():
        monkeypatch.setenv(key, value)
    captured: dict[str, object] = {}

    class Process:
        pid = 999999

        def poll(self):
            return None

        def wait(self, timeout=None):
            captured["timeout"] = timeout
            return 0

    def factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Process()

    returncode = parent._run_build_vector(
        contract,
        parent.BUILD_VECTORS[0],
        incomplete / "logs/configure.stdout.log",
        incomplete / "logs/configure.stderr.log",
        started=parent.time.monotonic(),
        popen_factory=factory,
    )
    assert returncode == 0
    environment = captured["kwargs"]["env"]
    assert environment == expected
    assert tuple(environment) == parent.BUILD_ENVIRONMENT_KEYS
    assert not set(environment).intersection(selectors)
    assert verifier._expected_build_environment(contract) == environment
    assert parent.build_environment_digest(environment) == verifier.digest(environment)


def test_independent_verifier_rejects_self_consistent_build_environment_drift(
    tmp_path: Path,
) -> None:
    contract = _launch_contract(tmp_path)
    artifact_values = {
        "build/h0_phase_a/saccade_tracking_ext.so": b"extension",
        "build/h0_phase_a/libsaccade_scan_plugin.so": b"plugin",
    }
    for relative, payload in artifact_values.items():
        (tmp_path / relative).write_bytes(payload)
    cache = tmp_path / "build/h0_phase_a/CMakeCache.txt"
    cache.write_bytes(b"cache")
    (tmp_path / "uv.lock").write_bytes(b"lock")

    def tool_record(path: str) -> dict[str, object]:
        return {
            "length": 1,
            "path": path,
            "sha256": "1" * 64,
            "version": "fixture",
        }

    artifacts = [
        {
            "dynamic_dependencies": [],
            "elf_gnu_build_id": "a",
            "length": len(payload),
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in artifact_values.items()
    ]
    environment = parent.build_environment(contract)
    identity = {
        "artifacts": artifacts,
        "build_environment": environment,
        "build_environment_digest": verifier.digest(environment),
        "build_vectors": [list(vector) for vector in parent.BUILD_VECTORS],
        "cmake": {"generator": "fixture", **tool_record("/usr/bin/true")},
        "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "compilers": {
            "c": tool_record("/usr/bin/true"),
            "cxx": tool_record("/usr/bin/true"),
            "cuda": tool_record("/usr/bin/true"),
        },
        "cuda_toolkit_root": "/opt/cuda",
        "python": {
            "abi": "fixture",
            **tool_record("/usr/bin/true"),
        },
        "python_ext_suffix": ".so",
        "state": "complete",
        "uv_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
    }
    verifier._verify_complete_build_identity(identity, contract)
    drifted = dict(environment)
    drifted["CC"] = "/host/cc"
    identity["build_environment"] = drifted
    identity["build_environment_digest"] = verifier.digest(drifted)
    with pytest.raises(verifier.VerificationError, match="environment table/digest"):
        verifier._verify_complete_build_identity(identity, contract)


def test_gpu_selection_is_lexicographic_on_normalized_pci_bus_id() -> None:
    records = [
        {"normalized_pci_bus_id": "00000000:0a:00.0", "uuid": "GPU-b"},
        {"normalized_pci_bus_id": "0000:02:00.0", "uuid": "GPU-a"},
    ]
    selected = parent.select_gpu_record(records)
    assert selected["normalized_pci_bus_id"] == "0000:02:00.0"
    assert selected["uuid"] == "GPU-a"


def test_complete_untampered_synthetic_evidence_passes() -> None:
    report = verifier.verify_evidence(evidence_for())
    assert report == {
        "document_type": "aggregate_verification",
        "result": "phase_a_pass",
        "schema": "h0_phase_a_verifier_v1",
        "valid": True,
    }


def test_preflight_rejection_uses_explicit_not_reached_checkpoints() -> None:
    evidence = evidence_for("provenance_invalid")
    evidence["input_binding"] = {
        "algorithm": "h0_bound_inputs_v1",
        "checkpoints": [
            parent._not_reached_checkpoint(name) for name in parent.CHECKPOINTS
        ],
        "final_equal": None,
        "inotify_mask": list(parent.INOTIFY_MASK_NAMES),
        "monitor_state": "not_started",
        "mutation_events": [],
        "t0_digest": evidence["controller_input"]["bound_inputs"]["digest"],
    }
    assert verifier.verify_evidence(evidence)["result"] == "provenance_invalid"
    evidence["result"] = "build_failed"
    evidence["decision_predicates"]["provenance_ok"] = True
    evidence["decision_predicates"]["build_ok"] = False
    evidence["artifact_inventory"] = list(parent.C_PATHS)
    with pytest.raises(verifier.VerificationError, match="unstarted monitor"):
        verifier.verify_evidence(evidence)


@pytest.mark.parametrize("result", parent.RESULT_ENUM)
def test_every_legal_result_matrix_terminal_combination(result: str) -> None:
    assert verifier.verify_evidence(evidence_for(result))["result"] == result


@pytest.mark.parametrize("completed_count", range(5))
def test_runner_timeout_accepts_completed_prefix_then_not_run_suffix(
    completed_count: int,
) -> None:
    evidence = evidence_for("runner_timeout")
    children = _children(evidence["controller_input"])
    for child_invocation in children[completed_count:]:
        _mark_not_run(child_invocation)
    evidence["child_invocations"] = children
    assert verifier.verify_evidence(evidence)["result"] == "runner_timeout"


@pytest.mark.parametrize("failed_index", range(4))
def test_runner_timeout_accepts_completed_prefix_then_failed_child(
    failed_index: int,
) -> None:
    evidence = evidence_for("runner_timeout")
    children = _children(evidence["controller_input"])
    children[failed_index]["state"] = "failed"
    children[failed_index]["result"] = "runner_timeout"
    for child_invocation in children[failed_index + 1 :]:
        _mark_not_run(child_invocation)
    evidence["child_invocations"] = children
    assert verifier.verify_evidence(evidence)["result"] == "runner_timeout"


def test_runner_timeout_rejects_non_prefix_controller_timeout_states() -> None:
    evidence = evidence_for("runner_timeout")
    children = _children(evidence["controller_input"])
    _mark_not_run(children[1])
    evidence["child_invocations"] = children
    with pytest.raises(verifier.VerificationError, match="state order"):
        verifier.verify_evidence(evidence)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("result"), "schema rejection"),
        (lambda value: value.update({"unknown": True}), "schema rejection"),
        (lambda value: value.update({"result": "mystery"}), "schema rejection"),
        (
            lambda value: value["controller_input"]["execution_constants"][
                "ordered_run_plan"
            ].reverse(),
            "schema rejection",
        ),
        (
            lambda value: value["child_invocations"][0]["vector"].__setitem__(
                0, "/tmp/python"
            ),
            "command vector",
        ),
        (
            lambda value: value["child_invocations"][0]["vector"].append("--extra"),
            "command vector",
        ),
        (
            lambda value: value["child_invocations"][0]["vector"].__setitem__(
                4, "00_capture_off"
            ),
            "command vector",
        ),
        (
            lambda value: value["child_invocations"][0]["environment"].pop("TZ"),
            "schema rejection",
        ),
        (
            lambda value: value["child_invocations"][0]["environment"].update(
                {"EXTRA": "1"}
            ),
            "schema rejection",
        ),
        (
            lambda value: value["child_invocations"][0]["environment"].update(
                {"TZ": "Asia/Taipei"}
            ),
            "schema rejection",
        ),
        (
            lambda value: value["controller_input"].update(
                {"repository_root": "/tmp/drift"}
            ),
            "child vector",
        ),
        (
            lambda value: value["controller_input"]["bound_inputs"]["sequence"].update(
                {"digest": "b" * 64}
            ),
            "sequence-input digest",
        ),
        (
            lambda value: value["input_binding"].update(
                {"inotify_mask": list(reversed(parent.INOTIFY_MASK_NAMES))}
            ),
            "schema rejection",
        ),
        (
            lambda value: value["input_binding"]["checkpoints"][2].update(
                {"digest": "c" * 64}
            ),
            "bound input changed",
        ),
        (
            lambda value: value["input_binding"]["checkpoints"][2][
                "events_after"
            ].append({"path": "uv.lock"}),
            "schema rejection",
        ),
        (
            lambda value: value["child_invocations"][2].update(
                {
                    "runtime_inputs_digest": None,
                    "state": "running_interrupted",
                    "result": None,
                }
            ),
            "non-completed child",
        ),
        (
            lambda value: value["child_invocations"][1].pop("environment_digest"),
            "schema rejection",
        ),
        (lambda value: value.update({"valid": True}), "schema rejection"),
        (
            lambda value: value["result_matrix"]["phase_a_pass"].update(
                {"required": list(parent.C_PATHS)}
            ),
            "C/D/V matrix",
        ),
        (lambda value: value["artifact_inventory"].pop(), "artifact inventory"),
    ],
)
def test_adversarial_evidence_fails_closed(mutation, match: str) -> None:
    value = evidence_for()
    mutation(value)
    with pytest.raises(verifier.VerificationError, match=match):
        verifier.verify_evidence(value)


def test_result_enum_and_matrix_must_agree() -> None:
    value = evidence_for("packet_invalid")
    value["result"] = "phase_a_pass"
    with pytest.raises(verifier.VerificationError, match="decision predicates"):
        verifier.verify_evidence(value)


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--run-id",),
        ("--run", "00_capture_off"),
        ("--run-id=00_capture_off",),
        ("--run-id", "unknown"),
        ("--run-id", "00_capture_off", "extra"),
    ],
)
def test_child_parser_rejects_every_nonliteral_suffix(argv: tuple[str, ...]) -> None:
    with pytest.raises(child.ChildContractError):
        child._parse_argv(argv)


def test_child_environment_missing_extra_and_drift_fail_closed() -> None:
    environment = _environment(_controller(), "00_capture_off")
    child._initial_environment_gate(environment)
    for mutate in (
        lambda value: value.pop("TZ"),
        lambda value: value.update({"EXTRA": "1"}),
        lambda value: value.update({"SACCADE_GPU_DECODE": "0"}),
    ):
        changed = dict(environment)
        mutate(changed)
        with pytest.raises(child.ChildContractError):
            child._initial_environment_gate(changed)


def test_path_traversal_noncanonical_and_symlink_are_rejected(tmp_path: Path) -> None:
    for value in ("../x", "a/../x", "/absolute", "a\\b", "a//b"):
        with pytest.raises(parent.ContractError):
            parent.require_canonical_relative(value)
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(parent.ContractError, match="symlink-substituted"):
        parent.require_canonical_absolute(link.as_posix(), directory=False)


def test_sequence_digest_excludes_gt_and_det_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "sequence"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "det").mkdir()
    (sequence / "seqinfo.ini").write_bytes(b"seq")
    (sequence / "img1/000001.jpg").write_bytes(b"image")
    (sequence / "gt/gt.txt").write_bytes(b"label")
    (sequence / "det/det.txt").write_bytes(b"detector input excluded by A7 digest")
    inventory = parent.sequence_input_inventory(sequence)
    assert [record["path"] for record in inventory["files"]] == [
        "img1/000001.jpg",
        "seqinfo.ini",
    ]
    (sequence / "img1/link").symlink_to(sequence / "seqinfo.ini")
    with pytest.raises(parent.ContractError, match="non-symlink regular file"):
        parent.sequence_input_inventory(sequence)


def test_access_classifier_rejects_labels_traversal_and_unknown(tmp_path: Path) -> None:
    root = tmp_path / "root"
    run = root / "run"
    run.mkdir(parents=True)
    allowed = root / "input"
    allowed.write_bytes(b"x")
    allowed_set = frozenset({allowed.resolve()})
    assert (
        child.classify_access(allowed, root=root, run_dir=run, allowed=allowed_set)
        == "bound_input"
    )
    assert (
        child.classify_access(run / "out", root=root, run_dir=run, allowed=allowed_set)
        == "writable_output"
    )
    assert (
        child.classify_access(
            root / "datasets/x/gt/gt.txt", root=root, run_dir=run, allowed=allowed_set
        )
        == "forbidden_label"
    )
    assert (
        child.classify_access(
            root / "datasets/x/det/det.txt", root=root, run_dir=run, allowed=allowed_set
        )
        == "forbidden_label"
    )
    assert (
        child.classify_access("../escape", root=root, run_dir=run, allowed=allowed_set)
        == "non_canonical"
    )
    assert (
        child.classify_access(
            root / "other", root=root, run_dir=run, allowed=allowed_set
        )
        == "unexpected"
    )


def test_malformed_child_output_is_not_canonical(tmp_path: Path) -> None:
    path = tmp_path / "invocation.json"
    path.write_text('{"state":', encoding="utf-8")
    with pytest.raises(parent.ContractError, match="malformed"):
        parent.read_canonical_json(path)


def _launch_contract(tmp_path: Path) -> dict[str, object]:
    for relative in (".venv", "build/h0_phase_a", "trt", "torch", "cuda"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    controller = _controller()
    controller["repository_root"] = tmp_path.as_posix()
    controller["incomplete_root"] = "evidence.incomplete"
    controller["evidence_root"] = "evidence"
    controller["library_dirs"] = {
        "cuda_library_dir": (tmp_path / "cuda").as_posix(),
        "pytorch_library_dir": (tmp_path / "torch").as_posix(),
        "tensorrt_library_dir": (tmp_path / "trt").as_posix(),
    }
    controller["execution_constants"] = parent.execution_constants(tmp_path)
    return controller


class _FakeProcess:
    def __init__(
        self, invocation: Path, mode: str, captured: dict[str, object], *args, **kwargs
    ) -> None:
        self.pid = 999999
        self._invocation = invocation
        self._mode = mode
        self._captured = captured
        captured["args"] = args
        captured["kwargs"] = kwargs

    def poll(self):
        return None

    def wait(self, timeout=None):
        self._captured["timeout"] = timeout
        if self._mode == "complete":
            value = parent.read_canonical_json(self._invocation)
            value["confinement_plan_digest"] = "1" * 64
            value["confinement_probe_passed"] = True
            value["state"] = "completed"
            value["result"] = "run_completed"
            value["runtime_inputs_digest"] = "2" * 64
            parent._replace_canonical_fsync(self._invocation, value)
            return 0
        if self._mode == "malformed":
            self._invocation.write_bytes(b"{bad\n")
        return 17


def test_parent_launch_boundary_is_literal_and_collects_structured_child(
    tmp_path: Path,
) -> None:
    contract = _launch_contract(tmp_path)
    captured: dict[str, object] = {}
    invocation = tmp_path / "evidence.incomplete/runs/00_capture_off/invocation.json"

    def factory(*args, **kwargs):
        return _FakeProcess(invocation, "complete", captured, *args, **kwargs)

    code, result = parent.launch_child(
        contract,
        "00_capture_off",
        started=parent.time.monotonic(),
        popen_factory=factory,
    )
    assert code == 0 and result["state"] == "completed"
    kwargs = captured["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["close_fds"] is True
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == tmp_path
    assert set(kwargs["env"]) == set(parent.CHILD_ENV_KEYS)
    assert captured["args"][0] == parent.child_argv(tmp_path, "00_capture_off")


def test_parent_rejects_unexpected_exit_and_malformed_child_output(
    tmp_path: Path,
) -> None:
    contract = _launch_contract(tmp_path)
    invocation = tmp_path / "evidence.incomplete/runs/00_capture_off/invocation.json"

    def factory(*args, **kwargs):
        return _FakeProcess(invocation, "malformed", {}, *args, **kwargs)

    with pytest.raises(parent.ContractError, match="malformed"):
        parent.launch_child(
            contract,
            "00_capture_off",
            started=parent.time.monotonic(),
            popen_factory=factory,
        )


def test_confinement_setup_failure_is_provenance_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    monkeypatch.setattr(
        runtime_confinement,
        "build_plan",
        lambda **_kwargs: {"digest": "1" * 64},
    )

    def reject_spawn(*_args, **_kwargs):
        raise runtime_confinement.ConfinementError("forbidden inherited channel")

    monkeypatch.setattr(runtime_confinement, "spawn_confined", reject_spawn)
    with pytest.raises(parent.DriftError, match="runtime confinement setup failed"):
        parent.launch_child(
            contract,
            "00_capture_off",
            started=parent.time.monotonic(),
            build_identity={},
        )
    invocation = parent.read_canonical_json(
        tmp_path / "evidence.incomplete/runs/00_capture_off/invocation.json"
    )
    assert invocation["result"] == "provenance_invalid"
    assert invocation["state"] == "failed"


def test_unknown_result_is_never_classified() -> None:
    with pytest.raises(parent.ContractError, match="unrecognized"):
        parent.result_artifact_sets("unknown")


def test_result_selection_preserves_a7_first_applicable_order() -> None:
    predicates = _predicates()
    predicates["runners_ok"] = False
    predicates["timed_out"] = True
    assert parent.classify_result(**predicates) == "runner_nonzero"
    assert verifier.select_result(predicates) == "runner_nonzero"


def test_policy_projection_digest_and_unknown_fields_fail_closed() -> None:
    empty_proposal = {"candidates": [], "claims": []}
    empty_winner = {"commits": [], "winning_claims": []}
    inventory = {
        "active_tid_slot_pairs": [],
        "final_track_rows": [],
        "mot_output": {"length": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        "overflow_vector": [0] * 9,
        "proposal_projection": {
            "count": 0,
            "digest": verifier.digest(empty_proposal),
            "records": empty_proposal,
        },
        "relink_debug_raw": [0] * 13,
        "schema": "h0_phase_a_policy_inventory_v1",
        "winner_commit_projection": {
            "count": 0,
            "digest": verifier.digest(empty_winner),
            "records": empty_winner,
        },
    }
    parent._validate_policy_inventory("01_capture_on_1", inventory)
    verifier._verify_policy_inventory("01_capture_on_1", inventory)
    inventory["proposal_projection"]["digest"] = "0" * 64
    with pytest.raises(parent.ContractError, match="count/digest"):
        parent._validate_policy_inventory("01_capture_on_1", inventory)
    with pytest.raises(verifier.VerificationError, match="count/digest"):
        verifier._verify_policy_inventory("01_capture_on_1", inventory)


def _write_c_only_root(root: Path) -> None:
    evidence = evidence_for("build_failed")
    aggregate = verifier.verify_evidence(evidence)
    json_values = {
        "manifest.json": evidence,
        "build_identity.json": {
            "blocking_result": "build_failed",
            "state": "not_produced",
        },
        "runtime_identity.json": {
            "blocking_result": "build_failed",
            "state": "not_produced",
        },
        "gpu_identity.json": {
            "blocking_result": "build_failed",
            "state": "not_produced",
        },
        "input_binding.json": evidence["input_binding"],
        "comparison.json": {"blocking_result": "build_failed", "state": "not_produced"},
        "result.json": {"result": "build_failed", "schema": "h0_phase_a_execution_v1"},
        "verification/aggregate.json": aggregate,
    }
    for invocation in evidence["child_invocations"]:
        json_values[f"runs/{invocation['run_id']}/invocation.json"] = invocation
        json_values[f"runs/{invocation['run_id']}/runtime_inputs.json"] = {
            "blocking_result": "build_failed",
            "schema": "h0_runtime_inputs_v1",
            "state": "not_produced",
        }
    for relative in parent.C_PATHS:
        if relative == "checksums.sha256":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in json_values:
            path.write_bytes(parent.canonical_json_file_bytes(json_values[relative]))
        else:
            path.write_bytes(b"NOT_RUN\n")
    checksum_lines = []
    for relative in sorted(
        (path for path in parent.C_PATHS if path != "checksums.sha256"),
        key=lambda value: value.encode("utf-8"),
    ):
        checksum_lines.append(
            f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
        )
    (root / "checksums.sha256").write_text("".join(checksum_lines), encoding="ascii")


def _rewrite_c_checksums(root: Path) -> None:
    lines = [
        f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
        for relative in sorted(
            (path for path in parent.C_PATHS if path != "checksums.sha256"),
            key=lambda value: value.encode("utf-8"),
        )
    ]
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def _rewrite_root_checksums(root: Path) -> None:
    manifest = verifier.load_json(root / "manifest.json", canonical_file=True)
    lines = [
        f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
        for relative in sorted(
            (
                path
                for path in manifest["artifact_inventory"]
                if path != "checksums.sha256"
            ),
            key=lambda value: value.encode("utf-8"),
        )
    ]
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def _empty_policy_inventory(run_id: str, *, perturb_policy: bool) -> dict[str, object]:
    proposal_records = {"candidates": [], "claims": []}
    winner_records = {"commits": [], "winning_claims": []}
    capture_on = run_id != parent.RUN_IDS[0]
    relink_debug = [0] * 13
    if perturb_policy and run_id == parent.RUN_IDS[1]:
        relink_debug[0] = 1
    return {
        "active_tid_slot_pairs": [],
        "final_track_rows": [],
        "mot_output": {"length": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        "overflow_vector": [0] * 9,
        "proposal_projection": {
            "count": 0,
            "digest": verifier.digest(proposal_records),
            "records": proposal_records,
        }
        if capture_on
        else None,
        "relink_debug_raw": relink_debug,
        "schema": "h0_phase_a_policy_inventory_v1",
        "winner_commit_projection": {
            "count": 0,
            "digest": verifier.digest(winner_records),
            "records": winner_records,
        }
        if capture_on
        else None,
    }


def _synthetic_packet() -> dict[str, object]:
    return {
        "canonical": {
            "streams": {
                "candidate_records": [],
                "claim_records": [],
                "commit_records": [],
            }
        },
        "overflow_candidate_records": 0,
        "overflow_claim_records": 0,
        "overflow_commit_records": 0,
        "overflow_native_candidate_keys": 0,
        "overflow_native_claim_winner_keys": 0,
        "overflow_native_commit_keys": 0,
        "overflow_native_pair_keys": 0,
        "overflow_native_proposal_keys": 0,
        "overflow_pair_records": 0,
        "report": {"semantic_digest_sha256": "1" * 64},
    }


def _write_complete_synthetic_root(root: Path, result: str) -> None:
    assert result in {"capture_perturbs_policy", "phase_a_pass"}
    evidence = evidence_for(result)
    json_values: dict[str, object] = {}
    packet = _synthetic_packet()
    for run_id, invocation in zip(
        parent.RUN_IDS, evidence["child_invocations"], strict=True
    ):
        json_values[f"runs/{run_id}/invocation.json"] = invocation
        json_values[f"runs/{run_id}/runtime_inputs.json"] = {"state": "complete"}
        inventory = _empty_policy_inventory(
            run_id, perturb_policy=result == "capture_perturbs_policy"
        )
        json_values[f"runs/{run_id}/policy_inventory.json"] = inventory
        if run_id != parent.RUN_IDS[0]:
            json_values[f"runs/{run_id}/packet.json"] = packet
            if result == "phase_a_pass":
                json_values[f"runs/{run_id}/packet_verification.json"] = {
                    "report": packet["report"],
                    "state": "pass",
                }

    for relative, value in json_values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(parent.canonical_json_file_bytes(value))
    for run_id in parent.RUN_IDS:
        (root / "runs" / run_id / "MOT17-04-SDP.txt").write_bytes(b"")
    _, comparison = parent._compare_policy_inventories(root)

    controller = evidence["controller_input"]
    json_values.update(
        {
            "manifest.json": evidence,
            "build_identity.json": {"state": "complete"},
            "runtime_identity.json": {
                "bound_inputs_digest": controller["bound_inputs"]["digest"],
                "child_runtime_inputs": [
                    {
                        "confinement_plan_digest": invocation[
                            "confinement_plan_digest"
                        ],
                        "run_id": run_id,
                        "runtime_inputs_digest": invocation["runtime_inputs_digest"],
                        "state": "complete",
                    }
                    for run_id, invocation in zip(
                        parent.RUN_IDS,
                        evidence["child_invocations"],
                        strict=True,
                    )
                ],
                "confinement_backend": parent.RUNTIME_CONFINEMENT_BACKEND,
                "ingress_policy": parent.RUNTIME_INGRESS_POLICY,
                "library_dirs": controller["library_dirs"],
                "resolved_policy_fingerprint": parent.POLICY_FINGERPRINT,
                "runtime_trace_scope": list(parent.RUNTIME_TRACE_SCOPE),
                "state": "complete",
                "tool_runtime": controller["bound_inputs"]["tool_runtime"],
            },
            "gpu_identity.json": {**controller["gpu"], "state": "complete"},
            "input_binding.json": evidence["input_binding"],
            "comparison.json": comparison,
            "result.json": {
                "result": result,
                "schema": "h0_phase_a_execution_v1",
            },
            "verification/aggregate.json": verifier.verify_evidence(evidence),
        }
    )
    for relative in evidence["artifact_inventory"]:
        if relative == "checksums.sha256":
            continue
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in json_values:
            path.write_bytes(parent.canonical_json_file_bytes(json_values[relative]))
        else:
            path.write_bytes(b"SYNTHETIC\n")
    _rewrite_root_checksums(root)


def _patch_synthetic_domain_verifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verify_build(identity, _controller) -> None:
        assert identity == {"state": "complete"}

    def verify_runtime(value, _invocation, _controller, _identity) -> None:
        assert value == {"state": "complete"}

    monkeypatch.setattr(verifier, "_verify_complete_build_identity", verify_build)
    monkeypatch.setattr(verifier, "_verify_runtime_inputs", verify_runtime)
    monkeypatch.setattr(
        trace_export,
        "canonical_semantic_packet",
        lambda capture: capture["canonical"],
    )
    monkeypatch.setattr(
        trace_verifier,
        "verify_capture",
        lambda capture: capture["report"],
    )


def test_verifier_rebuilds_published_files_and_checksums(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    _write_c_only_root(evidence_root)
    assert verifier.verify_evidence_root(evidence_root)["result"] == "build_failed"
    (evidence_root / "logs/00_cmake_configure.stdout.log").write_bytes(b"tampered\n")
    with pytest.raises(verifier.VerificationError, match="checksum mismatch"):
        verifier.verify_evidence_root(evidence_root)


def test_verifier_rejects_unknown_status_field_even_with_valid_checksum(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    _write_c_only_root(evidence_root)
    status = verifier.load_json(
        evidence_root / "build_identity.json", canonical_file=True
    )
    status["unknown"] = True
    (evidence_root / "build_identity.json").write_bytes(
        parent.canonical_json_file_bytes(status)
    )
    _rewrite_c_checksums(evidence_root)
    with pytest.raises(
        verifier.VerificationError, match="not-produced identity status"
    ):
        verifier.verify_evidence_root(evidence_root)


@pytest.mark.parametrize(
    ("result", "expected_paths"),
    [
        ("build_failed", parent.C_PATHS),
        ("capture_perturbs_policy", parent.C_PATHS + parent.D_PATHS),
        ("phase_a_pass", parent.ALL_ARTIFACT_PATHS),
    ],
)
def test_untampered_synthetic_staged_roots_pass_their_exact_c_d_v_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: str,
    expected_paths: tuple[str, ...],
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _patch_synthetic_domain_verifiers(monkeypatch)
    if result == "build_failed":
        _write_c_only_root(root)
    else:
        _write_complete_synthetic_root(root, result)
    assert parent._artifact_inventory(root) == sorted(
        expected_paths, key=lambda value: value.encode("utf-8")
    )
    assert verifier.verify_evidence_root(root)["result"] == result


@pytest.mark.parametrize(
    "mutation",
    ["missing_file", "extra_file", "extra_directory", "forbidden_path"],
)
def test_staged_verifier_rejects_file_and_directory_universe_drift(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _write_c_only_root(root)
    if mutation == "missing_file":
        (root / "result.json").unlink()
    elif mutation == "extra_file":
        (root / "unknown.txt").write_bytes(b"unknown")
    elif mutation == "extra_directory":
        (root / "unknown").mkdir()
    else:
        forbidden = root / "runs/00_capture_off/gt/gt.txt"
        forbidden.parent.mkdir()
        forbidden.write_bytes(b"labels")
    with pytest.raises(verifier.VerificationError, match="universe"):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize("entry_type", ["symlink", "fifo", "socket"])
def test_staged_verifier_rejects_non_regular_entry_types(
    tmp_path: Path, entry_type: str
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _write_c_only_root(root)
    target = root / "x"
    open_socket: socket.socket | None = None
    if entry_type == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"NOT_RUN\n")
        target.symlink_to(outside)
    elif entry_type == "fifo":
        os.mkfifo(target)
    else:
        open_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        open_socket.bind(target.as_posix())
    try:
        with pytest.raises(verifier.VerificationError, match="entry type"):
            verifier.verify_evidence_root(root)
    finally:
        if open_socket is not None:
            open_socket.close()


@pytest.mark.parametrize("mutation", ["content_tamper", "order_drift", "malformed"])
def test_staged_verifier_rejects_checksum_content_order_and_shape(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _write_c_only_root(root)
    checksum = root / "checksums.sha256"
    if mutation == "content_tamper":
        (root / "logs/00_cmake_configure.stdout.log").write_bytes(b"tampered\n")
        match = "checksum mismatch"
    elif mutation == "order_drift":
        checksum.write_text(
            "".join(reversed(checksum.read_text(encoding="ascii").splitlines(True))),
            encoding="ascii",
        )
        match = "checksum path order"
    else:
        lines = checksum.read_text(encoding="ascii").splitlines(True)
        lines[0] = f"{'0' * 64} {lines[0][66:]}"
        checksum.write_text("".join(lines), encoding="ascii")
        match = "malformed checksum"
    with pytest.raises(verifier.VerificationError, match=match):
        verifier.verify_evidence_root(root)


@pytest.mark.parametrize(
    ("relative", "value", "match"),
    [
        (
            "result.json",
            {"result": "runner_timeout", "schema": "h0_phase_a_execution_v1"},
            "manifest/result.json mismatch",
        ),
        (
            "verification/aggregate.json",
            {
                "document_type": "aggregate_verification",
                "result": "runner_timeout",
                "schema": "h0_phase_a_verifier_v1",
                "valid": True,
            },
            "stored aggregate differs",
        ),
    ],
)
def test_staged_verifier_rejects_manifest_file_and_aggregate_disagreement(
    tmp_path: Path, relative: str, value: object, match: str
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _write_c_only_root(root)
    (root / relative).write_bytes(parent.canonical_json_file_bytes(value))
    _rewrite_root_checksums(root)
    with pytest.raises(verifier.VerificationError, match=match):
        verifier.verify_evidence_root(root)


def test_staged_verifier_rejects_packet_verifier_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence.incomplete"
    root.mkdir()
    _patch_synthetic_domain_verifiers(monkeypatch)
    _write_complete_synthetic_root(root, "phase_a_pass")
    stored_path = root / "runs/02_capture_on_2/packet_verification.json"
    stored = verifier.load_json(stored_path, canonical_file=True)
    stored["report"]["semantic_digest_sha256"] = "2" * 64
    stored_path.write_bytes(parent.canonical_json_file_bytes(stored))
    _rewrite_root_checksums(root)
    with pytest.raises(verifier.VerificationError, match="packet verifier pass"):
        verifier.verify_evidence_root(root)


def test_finalization_runs_staged_verifier_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    predicates = _predicates()
    predicates["build_ok"] = False
    staged = tmp_path / contract["incomplete_root"]
    final = tmp_path / contract["evidence_root"]
    real_verify = verifier.verify_evidence_root
    observed: list[Path] = []

    def inspect_before_publication(root: Path) -> dict[str, object]:
        assert root == staged
        assert (root / "checksums.sha256").is_file()
        assert not final.exists()
        observed.append(root)
        return real_verify(root)

    monkeypatch.setattr(verifier, "verify_evidence_root", inspect_before_publication)
    published = parent._finalize_bundle_once(
        contract,
        started=0.0,
        result="build_failed",
        predicates=predicates,
        checkpoints=_binding(contract)["checkpoints"],
        comparison=None,
        build_identity=None,
        mutation_events=[],
        clock=lambda: 1.0,
    )
    assert observed == [staged]
    assert published == final
    assert final.is_dir()
    assert not staged.exists()


def test_controller_self_report_cannot_bypass_staged_root_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    predicates = _predicates()
    predicates["build_ok"] = False
    staged = tmp_path / contract["incomplete_root"]
    final = tmp_path / contract["evidence_root"]
    self_report = {
        "document_type": "aggregate_verification",
        "result": "build_failed",
        "schema": "h0_phase_a_verifier_v1",
        "valid": True,
    }
    monkeypatch.setattr(verifier, "verify_evidence", lambda _manifest: self_report)
    real_validate = parent._validate_directory_universe

    def inject_self_reported_success(root: Path, required) -> None:
        real_validate(root, required)
        identity = verifier.load_json(root / "build_identity.json", canonical_file=True)
        identity["controller_claimed_success"] = True
        (root / "build_identity.json").write_bytes(
            parent.canonical_json_file_bytes(identity)
        )
        _rewrite_root_checksums(root)

    monkeypatch.setattr(
        parent, "_validate_directory_universe", inject_self_reported_success
    )
    with pytest.raises(verifier.VerificationError, match="not-produced identity"):
        parent._finalize_bundle_once(
            contract,
            started=0.0,
            result="build_failed",
            predicates=predicates,
            checkpoints=_binding(contract)["checkpoints"],
            comparison=None,
            build_identity=None,
            mutation_events=[],
            clock=lambda: 1.0,
        )
    assert staged.is_dir()
    assert not final.exists()


def test_staged_reconstruction_must_equal_stored_aggregate_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    predicates = _predicates()
    predicates["build_ok"] = False
    final = tmp_path / contract["evidence_root"]
    monkeypatch.setattr(
        verifier,
        "verify_evidence_root",
        lambda _root: {"valid": False},
    )
    with pytest.raises(parent.ContractError, match="staged-root reconstruction"):
        parent._finalize_bundle_once(
            contract,
            started=0.0,
            result="build_failed",
            predicates=predicates,
            checkpoints=_binding(contract)["checkpoints"],
            comparison=None,
            build_identity=None,
            mutation_events=[],
            clock=lambda: 1.0,
        )
    assert not final.exists()


def test_staged_verification_crossing_deadline_prevents_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    predicates = _predicates()
    predicates["build_ok"] = False
    final = tmp_path / contract["evidence_root"]
    now = {"value": 3599.0}
    real_verify = verifier.verify_evidence_root

    def crossing_verify(root: Path) -> dict[str, object]:
        value = real_verify(root)
        now["value"] = 3600.0
        return value

    monkeypatch.setattr(verifier, "verify_evidence_root", crossing_verify)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        parent._finalize_bundle_once(
            contract,
            started=0.0,
            result="build_failed",
            predicates=predicates,
            checkpoints=_binding(contract)["checkpoints"],
            comparison=None,
            build_identity=None,
            mutation_events=[],
            clock=lambda: now["value"],
        )
    assert not final.exists()


def test_inotify_observes_bound_mutation(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_bytes(b"before")
    with parent.BoundInputMonitor([bound]) as monitor:
        monitor.assert_clean()
        bound.write_bytes(b"after")
        with pytest.raises(parent.DriftError, match="mutation"):
            monitor.assert_clean()


def test_inotify_watches_logical_symlink_and_allows_future_output_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"bound")
    logical = tmp_path / "logical"
    logical.symlink_to(target)
    future_output = tmp_path / "future" / "output"
    with parent.BoundInputMonitor([logical], ignored_roots=[future_output]) as monitor:
        monitor.assert_clean()
        logical.unlink()
        logical.symlink_to(target)
        with pytest.raises(parent.DriftError, match="mutation"):
            monitor.assert_clean()


def test_active_wait_kills_process_group_on_continuous_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = tmp_path / "bound"
    bound.write_bytes(b"before")
    terminated: list[object] = []

    class Process:
        def wait(self, timeout=None):
            bound.write_bytes(b"after")
            raise subprocess.TimeoutExpired("fixture", timeout)

    process = Process()
    monkeypatch.setattr(parent, "_terminate_process_group", terminated.append)
    with parent.BoundInputMonitor([bound]) as monitor:
        with pytest.raises(parent.DriftError, match="while child fixture was active"):
            parent._wait_with_monitor(
                process,
                started=parent.time.monotonic(),
                monitor=monitor,
                stage="child fixture",
            )
    assert terminated == [process]


def test_termination_prefers_full_tracee_tree_cleanup() -> None:
    calls: list[str] = []

    class Process:
        pid = 12345

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def terminate_tree() -> None:
            calls.append("tree")

    parent._terminate_process_group(Process())
    assert calls == ["tree"]


@pytest.mark.parametrize("artifact_name", ["result.json", "checksums.sha256"])
def test_final_artifact_fsync_crossing_deadline_is_rejected(
    tmp_path: Path, artifact_name: str
) -> None:
    now = {"value": 3599.0}
    path = tmp_path / artifact_name

    def clock() -> float:
        return now["value"]

    if artifact_name == "result.json":

        def crossing_write(target: Path, value: object) -> None:
            parent._write_canonical_fsync(target, value)
            now["value"] = 3600.0

        action = crossing_write
        payload: object = {"result": "phase_a_pass"}
    else:

        def crossing_write(target: Path, value: object) -> None:
            assert isinstance(value, bytes)
            parent._write_bytes_fsync(target, value)
            now["value"] = 3600.0

        action = crossing_write
        payload = b"0" * 64 + b"  result.json\n"

    with pytest.raises(TimeoutError, match="deadline exhausted"):
        parent._deadline_checked_call(0.0, clock, action, path, payload)
    assert path.is_file()


def test_publication_rename_crossing_deadline_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = tmp_path / "evidence.incomplete"
    final = tmp_path / "evidence"
    incomplete.mkdir()
    (incomplete / "result.json").write_bytes(b"fixture")
    now = {"value": 3599.0}
    real_replace = parent.os.replace
    replace_calls = 0

    def crossing_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        real_replace(source, destination)
        if replace_calls == 1:
            now["value"] = 3600.0

    monkeypatch.setattr(parent.os, "replace", crossing_replace)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        parent._publish_evidence_root(
            incomplete, final, started=0.0, clock=lambda: now["value"]
        )
    assert replace_calls == 2
    assert incomplete.is_dir()
    assert not final.exists()


def test_parent_fsync_crossing_deadline_rolls_back_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = tmp_path / "evidence.incomplete"
    final = tmp_path / "evidence"
    incomplete.mkdir()
    (incomplete / "result.json").write_bytes(b"fixture")
    now = {"value": 3599.0}
    real_fsync = parent._fsync_directory
    fsync_calls = 0

    def crossing_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(path)
        if fsync_calls == 1:
            now["value"] = 3600.0

    monkeypatch.setattr(parent, "_fsync_directory", crossing_fsync)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        parent._publish_evidence_root(
            incomplete, final, started=0.0, clock=lambda: now["value"]
        )
    assert fsync_calls == 2
    assert incomplete.is_dir()
    assert not final.exists()


def test_expired_finalization_publishes_only_runner_timeout_envelope(
    tmp_path: Path,
) -> None:
    contract = _launch_contract(tmp_path)
    predicates = _predicates()
    checkpoints = [
        {
            "digest": contract["bound_inputs"]["digest"],
            "events_after": [],
            "events_before": [],
            "inventory_equal": True,
            "monotonic_ns": 0,
            "name": "T0",
            "state": "completed",
        }
    ] + [parent._not_reached_checkpoint(name) for name in parent.CHECKPOINTS[1:]]
    result = parent._finalize_bundle(
        contract,
        started=0.0,
        result="phase_a_pass",
        predicates=predicates,
        checkpoints=checkpoints,
        comparison=None,
        build_identity=None,
        mutation_events=[],
        clock=lambda: 3600.0,
    )
    final = tmp_path / contract["evidence_root"]
    assert result == "runner_timeout"
    assert final.is_dir()
    assert not (tmp_path / contract["incomplete_root"]).exists()
    assert parent.read_canonical_json(final / "result.json")["result"] == result
    assert parent._artifact_inventory(final) == sorted(
        parent.C_PATHS, key=lambda value: value.encode("utf-8")
    )
    assert verifier.verify_evidence_root(final)["result"] == result


def test_parent_fsync_timeout_after_four_completed_children_republishes_valid_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _launch_contract(tmp_path)
    incomplete = tmp_path / contract["incomplete_root"]
    incomplete.mkdir()
    parent._ensure_not_run_slots(contract, incomplete)
    for run_id in parent.RUN_IDS:
        invocation_path = incomplete / "runs" / run_id / "invocation.json"
        invocation = parent.read_canonical_json(invocation_path)
        invocation["confinement_plan_digest"] = "1" * 64
        invocation["confinement_probe_passed"] = True
        invocation["state"] = "completed"
        invocation["result"] = "run_completed"
        invocation["runtime_inputs_digest"] = "2" * 64
        parent._replace_canonical_fsync(invocation_path, invocation)
    for relative in parent.D_PATHS:
        path = incomplete / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic\n")
    semantic_digest = "1" * 64
    for relative in parent.V_PATHS:
        path = incomplete / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            parent.canonical_json_file_bytes(
                {
                    "report": {"semantic_digest_sha256": semantic_digest},
                    "state": "pass",
                }
            )
        )
    now = {"value": 3599.0}
    real_fsync = parent._fsync_directory
    fsync_calls = 0

    def crossing_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        real_fsync(path)
        if fsync_calls == 1:
            now["value"] = 3600.0

    # This deadline-only fixture intentionally uses synthetic D/V placeholders;
    # the staged-root verifier has separate byte/schema reconstruction tests.
    monkeypatch.setattr(
        verifier,
        "verify_evidence_root",
        lambda root: parent.read_canonical_json(root / "verification/aggregate.json"),
    )
    monkeypatch.setattr(parent, "_fsync_directory", crossing_fsync)
    result = parent._finalize_bundle(
        contract,
        started=0.0,
        result="phase_a_pass",
        predicates=_predicates(),
        checkpoints=_binding(contract)["checkpoints"],
        comparison={"state": "equal"},
        build_identity=None,
        mutation_events=[],
        clock=lambda: now["value"],
    )
    final = tmp_path / contract["evidence_root"]
    assert fsync_calls == 3
    assert result == "runner_timeout"
    assert final.is_dir()
    assert not incomplete.exists()
    assert parent.read_canonical_json(final / "manifest.json")["result"] == result
    assert parent.read_canonical_json(final / "result.json")["result"] == result
    assert not any((final / relative).exists() for relative in parent.D_PATHS)
    assert not any((final / relative).exists() for relative in parent.V_PATHS)
    assert verifier.verify_evidence_root(final)["result"] == result
