"""RC2 authority-chain and independent v3 JSON admissions; no Phase A runs."""

# scope: system
# function: contract
# lifecycle: active

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
            lambda controller, inventory: controller.update(
                {"build_tool_binding": {"fixture": "substituted"}}
            ),
            "build-tool binding",
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
            "cmake": "/selected/cmake",
            "cxx": "/selected/cxx",
            "git": "/selected/git",
            "ldd": "/selected/ldd",
            "nvcc": "/selected/nvcc",
            "readelf": "/selected/readelf",
            "uv": "/selected/uv",
        },
        "build_tool_binding": {"fixture": "binding"},
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
        "build_tool_binding": dict(expected["build_tool_binding"]),
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
    for name in ("git", "ldd", "readelf", "uv", "c++", "cmake"):
        (tools / name).write_bytes(name.encode("utf-8"))
    (tools / "loader").write_bytes(b"loader\n")
    purelib = tmp_path / "purelib"
    nvcc = purelib / "nvidia/cu13/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_bytes(b"nvcc\n")
    cuda_lib = purelib / "nvidia/cu13/lib"
    cuda_lib.mkdir()
    (cuda_lib / "libcudart.so.13").write_bytes(b"cuda\n")
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "torch.so").write_bytes(b"torch\n")
    tensorrt_lib = tmp_path / "tensorrt"
    tensorrt_lib.mkdir()
    (tensorrt_lib / "nvinfer.so").write_bytes(b"tensorrt\n")

    def physical(command: str, **_kwargs) -> Path:
        return tools / command

    def run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{torch_lib}\n{tensorrt_lib}\n{purelib}\n",
        )

    monkeypatch.setattr(verifier, "_physical_executable", physical)
    monkeypatch.setattr(
        verifier,
        "_independent_loader_closure",
        lambda *_args: [verifier._host_file_record(tools / "loader")],
    )
    monkeypatch.setattr(
        verifier, "_independently_selected_gpu", lambda: {"uuid": "GPU"}
    )
    # Fixtures fully mock subprocess.run for the torch/library query; R5
    # interpreter-runtime discovery is exercised by dedicated tests instead.
    monkeypatch.setattr(
        verifier, "_discover_python_interpreter_runtime_paths", lambda _python: []
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
    nvcc = root.parent / "purelib/nvidia/cu13/bin/nvcc"
    assert host["tool_paths"]["nvcc"] == nvcc.as_posix()
    assert next(
        item for item in host["tool_runtime"] if item["logical_path"] == nvcc.as_posix()
    ) == verifier._host_file_record(nvcc)


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
        "build_tool_binding": dict(expected["build_tool_binding"]),
        "library_dirs": dict(expected["library_dirs"]),
        "gpu": dict(expected["gpu"]),
    }
    with pytest.raises(verifier.VerificationError, match="host expansion"):
        verifier._verify_host_execution_inputs(controller, inventory, root)


def test_independent_discovery_admits_fixture_interpreter_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: independent expansion must admit discovered interpreter-runtime paths."""
    root, _pyvenv_config = _independent_host_fixture(tmp_path, monkeypatch)
    runtime = tmp_path / "interp-runtime"
    runtime.mkdir()
    extra = runtime / "encodings_stub.py"
    extra.write_bytes(b"encodings\n")
    host_paths = [extra.as_posix(), "/etc/ld.so.cache"]
    # Keep only paths that exist on this host for the fixture record builder.
    present = [path for path in host_paths if Path(path).is_file()]
    assert present, "fixture requires at least one discoverable host path"
    monkeypatch.setattr(
        verifier,
        "_discover_python_interpreter_runtime_paths",
        lambda _python: present,
    )
    host = verifier._independent_host_execution_inputs(root)
    logicals = {item["logical_path"] for item in host["tool_runtime"]}
    for path in present:
        assert path in logicals
        record = next(
            item for item in host["tool_runtime"] if item["logical_path"] == path
        )
        assert record == verifier._host_file_record(Path(path))


def test_omitted_or_mutated_interpreter_runtime_path_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omission or mutation of a discovered interpreter-runtime path is fail-closed."""
    root, _pyvenv_config = _independent_host_fixture(tmp_path, monkeypatch)
    runtime = tmp_path / "interp-runtime"
    runtime.mkdir()
    extra = runtime / "locale_alias_stub"
    extra.write_bytes(b"locale\n")
    discovered = [extra.as_posix()]
    monkeypatch.setattr(
        verifier,
        "_discover_python_interpreter_runtime_paths",
        lambda _python: discovered,
    )
    expected = verifier._independent_host_execution_inputs(root)
    controller = {
        "tool_paths": dict(expected["tool_paths"]),
        "build_tool_binding": dict(expected["build_tool_binding"]),
        "library_dirs": dict(expected["library_dirs"]),
        "gpu": dict(expected["gpu"]),
    }

    omitted = {
        "tool_runtime": [
            dict(record)
            for record in expected["tool_runtime"]
            if record["logical_path"] != extra.as_posix()
        ]
    }
    with pytest.raises(verifier.VerificationError, match="host expansion"):
        verifier._verify_host_execution_inputs(controller, omitted, root)

    mutated = {"tool_runtime": [dict(record) for record in expected["tool_runtime"]]}
    target = next(
        item
        for item in mutated["tool_runtime"]
        if item["logical_path"] == extra.as_posix()
    )
    target["sha256"] = "0" * 64
    with pytest.raises(verifier.VerificationError, match="host expansion"):
        verifier._verify_host_execution_inputs(controller, mutated, root)


def test_assembler_tool_runtime_matches_independent_expansion_on_host() -> None:
    """Positive host regression: freeze assembler tool_runtime == independent verifier.

    Builds the same candidate set the freeze assembler admits (including R5
    interpreter-runtime discovery) and asserts byte-exact inventory equality
    against the verifier's independent reconstruction — not mere counts.
    """
    import run_h0_phase_a as controller

    root = Path(__file__).resolve().parents[3]
    python = root / ".venv/bin/python"
    if not python.is_file() or python.is_symlink():
        pytest.skip("physical frozen .venv/bin/python unavailable")
    pyvenv_config = root / ".venv/pyvenv.cfg"
    if not pyvenv_config.is_file() or pyvenv_config.is_symlink():
        pytest.skip("physical frozen .venv/pyvenv.cfg unavailable")

    tool_paths = {
        name: assembler._physical_executable(name).as_posix()
        for name in ("git", "ldd", "readelf", "uv")
    }
    build_tool_bound_inputs = assembler.derive_build_tool_bound_inputs(
        root, ldd_path=Path(tool_paths["ldd"])
    )
    build_tool_binding = build_tool_bound_inputs["build_tool_binding"]
    tool_paths.update(
        {
            item["role"]: item["record"]["realpath"]
            for item in build_tool_binding["tools"]
        }
    )
    query = subprocess.run(
        [
            python.as_posix(),
            "-I",
            "-c",
            "import pathlib,sysconfig,torch; "
            "print((pathlib.Path(torch.__file__).resolve().parent/'lib').as_posix()); "
            "import tensorrt_libs; "
            "print(pathlib.Path(tensorrt_libs.__file__).resolve().parent.as_posix()); "
            "print(pathlib.Path(sysconfig.get_path('purelib')).resolve().as_posix())",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(query) != 3:
        pytest.skip("could not derive runtime library directories")
    cuda_root = Path(query[2]).resolve(strict=True) / "nvidia/cu13"
    nvcc = cuda_root / "bin/nvcc"
    if nvcc.is_symlink() or not nvcc.is_file():
        pytest.skip("frozen venv CUDA compiler unavailable")
    tool_paths["nvcc"] = nvcc.as_posix()
    libraries = {
        "tensorrt_library_dir": Path(query[1]).resolve(strict=True).as_posix(),
        "pytorch_library_dir": Path(query[0]).resolve(strict=True).as_posix(),
        "cuda_library_dir": (cuda_root / "lib").resolve(strict=True).as_posix(),
    }
    non_build = [
        Path(path) for name, path in tool_paths.items() if name not in {"cmake", "cxx"}
    ] + [python, pyvenv_config]
    tool_candidates = list(non_build)
    for directory in libraries.values():
        tool_candidates.extend(
            sorted(
                path
                for path in Path(directory).rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        )
    for logical in controller.discover_python_interpreter_runtime_paths(python):
        tool_candidates.append(Path(logical))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in tool_candidates:
        try:
            real = path.resolve(strict=True).as_posix()
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        deduped.append(path)
    other = [
        controller.external_input_record(root, path.as_posix()) for path in deduped
    ]
    assembler_tool_runtime = sorted(
        [*other, *build_tool_bound_inputs["tool_runtime"]],
        key=lambda record: record["logical_path"].encode("utf-8"),
    )
    independent = verifier._independent_host_execution_inputs(root)
    independent_tool_runtime = independent["tool_runtime"]
    assert len(assembler_tool_runtime) == len(independent_tool_runtime)
    assert assembler_tool_runtime == independent_tool_runtime
    assert any(
        item["logical_path"].endswith("ld.so.cache")
        or "encodings" in item["logical_path"]
        or "python3.12" in item["logical_path"]
        for item in independent_tool_runtime
    )


def test_execution_checkout_must_be_exact_seal_commit(tmp_path: Path) -> None:
    root, _instrumentation, _freeze, seal, artifact = _chain(tmp_path)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_authority_landing(
            root, artifact, checkout=_git(root, "rev-parse", f"{seal}^")
        )


def test_multiparent_execution_checkout_is_landing_mismatch_not_structural(
    tmp_path: Path,
) -> None:
    """Merge / multi-parent HEAD cannot be S; classify as non-current, not abort.

    Regression for Seal requalification on GitHub merge-commit main: discovery
    previously raised a hard VerificationError from _single_parent(HEAD), which
    aborted mixed-corpus classification and failed controlled-host qualification
    at landing_discovery_dry_run. Declaration A7.RC2 allows I to be any
    reviewable commit (including merges); only F and S must be ordinary
    one-parent commits. ``LandingMismatchError`` is the non-current signal
    caught by ``verify_current_landing_candidate``.
    """
    root, instrumentation, _freeze, seal, artifact = _chain(tmp_path)
    # Build a second parent and a merge commit (multi-parent execution checkout).
    _git(root, "checkout", "-b", "side", instrumentation)
    (root / "side.txt").write_text("side\n", encoding="utf-8")
    side = _commit(root, "side parent")
    _git(root, "checkout", seal)
    _git(root, "merge", "--no-ff", "-m", "merge side", side)
    merge = _git(root, "rev-parse", "HEAD")
    parents = _git(root, "show", "-s", "--format=%P", merge).split()
    assert len(parents) == 2

    with pytest.raises(
        verifier.LandingMismatchError, match="ordinary one-parent"
    ) as excinfo:
        verifier.verify_authority_landing(root, artifact, checkout=merge)
    # Discovery catches LandingMismatchError only; hard structural VerificationError
    # (non-subclass path) would abort the whole corpus.
    assert type(excinfo.value) is verifier.LandingMismatchError
    assert issubclass(verifier.LandingMismatchError, verifier.VerificationError)


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
