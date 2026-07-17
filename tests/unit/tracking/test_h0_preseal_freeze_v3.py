"""RC2 authority-chain and independent v3 JSON admissions; no Phase A runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[3] / "scripts" / "tools"
sys.path.insert(0, TOOLS.as_posix())

import verify_h0_preseal_freeze as verifier  # noqa: E402
import build_h0_preseal_freeze as assembler  # noqa: E402


def test_v3_has_one_ordered_implementation_universe_and_no_output_switch() -> None:
    assert assembler.FREEZE_SCHEMA_VERSION == "h0_preseal_freeze_v3"
    assert assembler.IMPLEMENTATION_IDENTITIES == verifier.IMPLEMENTATIONS
    assert [path for path, _identity in assembler.IMPLEMENTATION_IDENTITIES] == [
        "scripts/tools/run_h0_phase_a.py",
        "scripts/tools/run_h0_phase_a_child.py",
        "scripts/tools/h0_phase_a_execution_schema_v1.json",
        "scripts/tools/h0_runtime_confinement.py",
        "scripts/tools/verify_h0_phase_a.py",
        "scripts/tools/export_headline_bridge_decision_trace.py",
        "scripts/tools/verify_headline_bridge_decision_trace.py",
        "scripts/tools/build_h0_preseal_freeze.py",
        "scripts/tools/check_h0_bridge_decision_trace_contract.py",
        "scripts/tools/h0_bridge_decision_trace_schema_v2.json",
        "scripts/tools/verify_h0_preseal_freeze.py",
    ]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _chain(
    tmp_path: Path, *, owner_date: str = "2026-07-17"
) -> tuple[Path, str, str, str, dict[str, str]]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "h0@example.invalid")
    _git(root, "config", "user.name", "H0 test")
    declaration = root / verifier.DECLARATION_PATH
    declaration.parent.mkdir(parents=True)
    declaration.write_text("base declaration\n", encoding="utf-8")
    instrumentation = _commit(root, "instrumentation")
    artifact = {"instrumentation_head": instrumentation}
    freeze_path = root / verifier.freeze_path(instrumentation)
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(verifier.canonical_json(artifact) + b"\n")
    freeze = _commit(root, "freeze")
    with declaration.open("a", encoding="utf-8") as output:
        output.write(
            f"| {owner_date} | `{instrumentation}` | `{freeze}` | `SEALED` |\n"
        )
    seal = _commit(root, "seal")
    return root, instrumentation, freeze, seal, artifact


def test_valid_synthetic_i_freeze_seal_execution_chain_passes(tmp_path: Path) -> None:
    root, instrumentation, freeze, seal, artifact = _chain(tmp_path)
    assert verifier.verify_authority_landing(root, artifact) == {
        "instrumentation_head": instrumentation,
        "freeze_commit": freeze,
        "seal_commit": seal,
        "execution_checkout": seal,
    }


@pytest.mark.parametrize("where", ["freeze-extra", "seal-extra"])
def test_post_head_extra_path_fails_closed(tmp_path: Path, where: str) -> None:
    root, instrumentation, freeze, _seal, artifact = _chain(tmp_path)
    if where == "freeze-extra":
        _git(root, "reset", "--hard", instrumentation)
        path = root / verifier.freeze_path(instrumentation)
        path.parent.mkdir(parents=True)
        path.write_bytes(verifier.canonical_json(artifact) + b"\n")
        (root / "scripts").mkdir()
        (root / "scripts" / "runtime.py").write_text("extra\n", encoding="utf-8")
        freeze = _commit(root, "invalid freeze")
        declaration = root / verifier.DECLARATION_PATH
        with declaration.open("a", encoding="utf-8") as output:
            output.write(
                f"| 2026-07-17 | `{instrumentation}` | `{freeze}` | `SEALED` |\n"
            )
        _commit(root, "seal")
    else:
        _git(root, "reset", "--hard", freeze)
        declaration = root / verifier.DECLARATION_PATH
        with declaration.open("a", encoding="utf-8") as output:
            output.write(
                f"| 2026-07-17 | `{instrumentation}` | `{freeze}` | `SEALED` |\n"
            )
        (root / "docs" / "extra.md").write_text("extra\n", encoding="utf-8")
        _commit(root, "invalid seal")
    with pytest.raises(verifier.VerificationError, match="post-head delta"):
        verifier.verify_authority_landing(root, artifact)


def test_wrong_or_duplicate_seal_row_fails_closed(tmp_path: Path) -> None:
    root, instrumentation, freeze, _seal, artifact = _chain(tmp_path)
    _git(root, "reset", "--hard", freeze)
    declaration = root / verifier.DECLARATION_PATH
    with declaration.open("a", encoding="utf-8") as output:
        output.write(f"| 2026-07-17 | `{instrumentation}` | `{freeze}` | `SEALED` |\n")
        output.write(f"| 2026-07-17 | `{instrumentation}` | `{freeze}` | `SEALED` |\n")
    _commit(root, "duplicate seal")
    with pytest.raises(verifier.VerificationError, match="seal row"):
        verifier.verify_authority_landing(root, artifact)


@pytest.mark.parametrize("owner_date", ["2026-02-30", "2026-13-01", "0000-01-01"])
def test_non_calendar_owner_event_date_fails_closed(
    tmp_path: Path, owner_date: str
) -> None:
    root, _instrumentation, _freeze, _seal, artifact = _chain(
        tmp_path, owner_date=owner_date
    )
    with pytest.raises(verifier.VerificationError, match="ISO calendar date"):
        verifier.verify_authority_landing(root, artifact)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda controller, inventory: controller["tool_paths"].update(
                {"git": "/other/git"}
            ),
            r"which\(\) selection",
        ),
        (
            lambda controller, inventory: controller["library_dirs"].update(
                {"cuda_library_dir": "/other/cuda"}
            ),
            "Python/nvcc derivation",
        ),
        (
            lambda controller, inventory: controller.update(
                {"gpu": {"uuid": "GPU-other"}}
            ),
            "NVML selection",
        ),
        (
            lambda controller, inventory: inventory.update({"tool_runtime": []}),
            "host expansion",
        ),
    ],
)
def test_host_execution_input_substitution_fails_against_independent_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    match: str,
) -> None:
    expected = {
        "tool_paths": {
            "git": "/selected/git",
            "ldd": "/selected/ldd",
            "readelf": "/selected/readelf",
            "uv": "/selected/uv",
        },
        "library_dirs": {
            "cuda_library_dir": "/selected/cuda",
            "pytorch_library_dir": "/selected/torch",
            "tensorrt_library_dir": "/selected/tensorrt",
        },
        "gpu": {"uuid": "GPU-selected"},
        "tool_runtime": [
            {
                "length": 1,
                "logical_path": "/selected/git",
                "realpath": "/selected/git",
                "sha256": "a" * 64,
                "symlink_chain": [],
            }
        ],
    }
    controller = {
        "tool_paths": dict(expected["tool_paths"]),
        "library_dirs": dict(expected["library_dirs"]),
        "gpu": dict(expected["gpu"]),
    }
    inventory = {"tool_runtime": list(expected["tool_runtime"])}
    monkeypatch.setattr(
        verifier, "_independent_host_execution_inputs", lambda _root: expected
    )
    mutate(controller, inventory)
    with pytest.raises(verifier.VerificationError, match=match):
        verifier._verify_host_execution_inputs(controller, inventory, tmp_path)


def _independent_host_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    root = tmp_path / "root"
    python = root / ".venv/bin/python"
    pyvenv_config = root / ".venv/pyvenv.cfg"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python\n")
    pyvenv_config.write_bytes(b"home = /fixture/python\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("git", "ldd", "readelf", "uv"):
        (tools / name).write_bytes(name.encode("utf-8"))
    nvcc = tmp_path / "cuda/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_bytes(b"nvcc\n")
    cuda_lib = tmp_path / "cuda/lib64"
    cuda_lib.mkdir()
    (cuda_lib / "cudart.so").write_bytes(b"cuda\n")
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "torch.so").write_bytes(b"torch\n")
    tensorrt_lib = tmp_path / "tensorrt"
    tensorrt_lib.mkdir()
    (tensorrt_lib / "nvinfer.so").write_bytes(b"tensorrt\n")

    def physical(command: str) -> Path:
        return nvcc if command == "nvcc" else tools / command

    def run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{torch_lib}\n{tensorrt_lib}\n",
        )

    monkeypatch.setattr(verifier, "_physical_executable", physical)
    monkeypatch.setattr(
        verifier, "_independently_selected_gpu", lambda: {"uuid": "GPU"}
    )
    monkeypatch.setattr(verifier.subprocess, "run", run)
    return root, pyvenv_config


def test_independent_host_expansion_binds_physical_pyvenv_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pyvenv_config = _independent_host_fixture(tmp_path, monkeypatch)
    host = verifier._independent_host_execution_inputs(root)
    record = next(
        item
        for item in host["tool_runtime"]
        if item["logical_path"] == pyvenv_config.as_posix()
    )
    assert record == verifier._host_file_record(pyvenv_config)


@pytest.mark.parametrize("state", ["missing", "symlink"])
def test_independent_host_expansion_rejects_missing_or_symlink_pyvenv_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    root, pyvenv_config = _independent_host_fixture(tmp_path, monkeypatch)
    if state == "missing":
        pyvenv_config.unlink()
    else:
        target = tmp_path / "foreign-pyvenv.cfg"
        target.write_bytes(b"foreign\n")
        pyvenv_config.unlink()
        pyvenv_config.symlink_to(target)
    with pytest.raises(verifier.VerificationError, match=r"\.venv/pyvenv\.cfg"):
        verifier._independent_host_execution_inputs(root)


def test_independent_host_expansion_rejects_pyvenv_config_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pyvenv_config = _independent_host_fixture(tmp_path, monkeypatch)
    expected = verifier._independent_host_execution_inputs(root)
    inventory = {"tool_runtime": [dict(record) for record in expected["tool_runtime"]]}
    record = next(
        item
        for item in inventory["tool_runtime"]
        if item["logical_path"] == pyvenv_config.as_posix()
    )
    record["sha256"] = "0" * 64
    controller = {
        "tool_paths": dict(expected["tool_paths"]),
        "library_dirs": dict(expected["library_dirs"]),
        "gpu": dict(expected["gpu"]),
    }
    with pytest.raises(verifier.VerificationError, match="host expansion"):
        verifier._verify_host_execution_inputs(controller, inventory, root)


def test_execution_checkout_must_be_exact_seal_commit(tmp_path: Path) -> None:
    root, _instrumentation, _freeze, seal, artifact = _chain(tmp_path)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_authority_landing(
            root, artifact, checkout=_git(root, "rev-parse", f"{seal}^")
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"instrumentation_head":"a","instrumentation_head":"b"}\n',
        b'{ "instrumentation_head": "a" }\n',
    ],
)
def test_duplicate_and_noncanonical_json_are_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    artifact = tmp_path / "h0_preseal_freeze_v3.json"
    artifact.write_bytes(payload)
    with pytest.raises(verifier.VerificationError):
        verifier.load_canonical_json(artifact)
