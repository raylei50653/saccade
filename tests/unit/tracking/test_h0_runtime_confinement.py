"""Kernel-level runtime input confinement and native loader admissions."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import h0_runtime_confinement as confinement  # noqa: E402
import verify_h0_phase_a as independent_verifier  # noqa: E402


pytestmark = pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to deadlocks.*:DeprecationWarning"
)


def _identity(path: Path, logical: Path | None = None) -> dict[str, Any]:
    real = path.resolve(strict=True)
    data = real.read_bytes()
    return {
        "length": len(data),
        "logical_path": (logical or path).as_posix(),
        "realpath": real.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "symlink_chain": [],
    }


def _ldd_inputs(executable: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ldd", executable.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [executable, Path("/etc/ld.so.cache")]
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "linux-vdso" in stripped or "not found" in stripped:
            continue
        candidate = (
            stripped.split("=>", 1)[1].strip().split(" ", 1)[0]
            if "=>" in stripped
            else stripped.split(" ", 1)[0]
        )
        if candidate.startswith("/"):
            paths.append(Path(candidate))
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = _identity(path, path)
        unique.setdefault(record["realpath"], record)
    return list(unique.values())


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.fail("native runtime confinement tests require cc")
    root = tmp_path_factory.mktemp("h0-native-confinement")
    (root / ".venv/bin").mkdir(parents=True)
    plugin_source = root / "plugin.c"
    runner_source = root / "runner.c"
    plugin = root / "native_plugin.so"
    runner = root / "native_runner"
    model = root / "model.bin"
    sequence = root / "sequence/frame.bin"
    probe = root / "evidence.incomplete/_runtime_confinement_denial_probe"
    run_dir = root / "evidence.incomplete/runs/00_capture_off"
    for run_id in independent_verifier.RUN_IDS:
        (root / "evidence.incomplete/runs" / run_id).mkdir(parents=True)
    plugin_source.write_text(
        "int h0_plugin_value(void) { return 7; }\n", encoding="utf-8"
    )
    runner_source.write_text(
        """
        #include <dlfcn.h>
        #include <errno.h>
        #include <fcntl.h>
        #include <sys/mman.h>
        #include <sys/stat.h>
        #include <sys/syscall.h>
        #include <unistd.h>

        int main(int argc, char **argv) {
            if (argc != 4) return 80;
            int denied = syscall(SYS_open, argv[1], O_RDONLY);
            if (denied >= 0 || errno != EACCES) return 81;
            void *plugin = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
            if (!plugin) return 82;
            int (*value)(void) = (int (*)(void))dlsym(plugin, "h0_plugin_value");
            if (!value || value() != 7) return 83;
            int fd = open(argv[3], O_RDONLY);
            if (fd < 0) return 84;
            struct stat info;
            if (fstat(fd, &info) != 0 || info.st_size < 1) return 85;
            void *mapped = mmap(0, info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapped == MAP_FAILED || ((unsigned char *)mapped)[0] != 0x5a) return 86;
            return 0;
        }
        """,
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", plugin_source, "-o", plugin], check=True
    )
    subprocess.run([compiler, runner_source, "-ldl", "-o", runner], check=True)
    model.write_bytes(b"Z-bound-model\n")
    sequence.parent.mkdir()
    sequence.write_bytes(b"Z-bound-sequence\n")
    probe.write_bytes(b"this read must be denied\n")
    return {
        "model": model,
        "plugin": plugin,
        "probe": probe,
        "root": root,
        "run_dir": run_dir,
        "runner": runner,
        "sequence": sequence,
    }


def _plan(
    fixture: dict[str, Path],
    *,
    bind_plugin: bool = True,
    bind_model: bool = True,
    data_kind: str = "model",
) -> dict[str, Any]:
    root = fixture["root"]
    runner_record = _identity(fixture["runner"])
    artifacts = []
    if bind_plugin:
        plugin = _identity(fixture["plugin"])
        artifacts.append(
            {
                "length": plugin["length"],
                "path": fixture["plugin"].relative_to(root).as_posix(),
                "sha256": plugin["sha256"],
            }
        )
    models = (
        [_identity(fixture["model"])] if bind_model and data_kind == "model" else []
    )
    sequence_files = []
    if bind_model and data_kind == "sequence":
        record = _identity(fixture["sequence"])
        sequence_files.append(
            {
                "length": record["length"],
                "path": fixture["sequence"].relative_to(root / "sequence").as_posix(),
                "sha256": record["sha256"],
            }
        )
    inventory = {
        "models_engines": models,
        "repository": [],
        "sequence": {"files": sequence_files, "root": "sequence"},
        "tool_runtime": _ldd_inputs(fixture["runner"]),
    }
    build_identity = {
        "artifacts": artifacts,
        "python": {
            "length": runner_record["length"],
            "path": runner_record["realpath"],
            "sha256": runner_record["sha256"],
        },
    }
    return confinement.build_plan(
        root=root,
        incomplete=root / "evidence.incomplete",
        inventory=inventory,
        build_identity=build_identity,
        denial_probe=fixture["probe"],
        run_ids=independent_verifier.RUN_IDS,
    )


def _run(
    fixture: dict[str, Path], plan: dict[str, Any], model_argument: Path
) -> tuple[int, dict[str, Any]]:
    root = fixture["root"]
    vector = [
        fixture["runner"].as_posix(),
        fixture["probe"].as_posix(),
        fixture["plugin"].as_posix(),
        model_argument.as_posix(),
    ]
    stdout_path = root / f"stdout-{os.urandom(8).hex()}"
    stderr_path = root / f"stderr-{os.urandom(8).hex()}"
    with (
        open(os.devnull, "rb", buffering=0) as stdin,
        open(stdout_path, "xb", buffering=0) as stdout,
        open(stderr_path, "xb", buffering=0) as stderr,
    ):
        process = confinement.spawn_confined(
            vector,
            cwd=root,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            plan=plan,
        )
        returncode = process.wait(timeout=10)
        return returncode, process.runtime_attestation()


def test_bound_native_plugin_model_and_mmap_are_attested(
    native_fixture: dict[str, Path],
) -> None:
    returncode, attestation = _run(
        native_fixture, _plan(native_fixture), native_fixture["model"]
    )
    assert returncode == 0
    assert attestation["state"] == "complete"
    assert attestation["denial_probe_observed"] is True
    records = {record["realpath"]: record for record in attestation["regular_files"]}
    assert "shared_library" in records[native_fixture["plugin"].as_posix()]["roles"]
    assert "mmap_read" in records[native_fixture["model"].as_posix()]["operations"]
    assert (
        "interpreter_or_executable"
        in records[native_fixture["runner"].as_posix()]["roles"]
    )
    assert any(
        "startup_mapping" in record["operations"]
        and "shared_library" in record["roles"]
        for record in records.values()
    )


def test_independent_verifier_rebuilds_the_native_runtime_inventory(
    native_fixture: dict[str, Path],
) -> None:
    plan = _plan(native_fixture)
    returncode, attestation = _run(native_fixture, plan, native_fixture["model"])
    assert returncode == 0
    runner_record = _identity(native_fixture["runner"])
    plugin_record = _identity(native_fixture["plugin"])
    build_identity = {
        "artifacts": [
            {
                "length": plugin_record["length"],
                "path": native_fixture["plugin"]
                .relative_to(native_fixture["root"])
                .as_posix(),
                "sha256": plugin_record["sha256"],
            }
        ],
        "python": {
            "length": runner_record["length"],
            "path": runner_record["realpath"],
            "sha256": runner_record["sha256"],
        },
    }
    controller = {
        "bound_inputs": {
            "models_engines": [_identity(native_fixture["model"])],
            "repository": [],
            "sequence": {"files": [], "root": "sequence"},
            "tool_runtime": _ldd_inputs(native_fixture["runner"]),
        },
        "incomplete_root": "evidence.incomplete",
        "repository_root": native_fixture["root"].as_posix(),
    }
    invocation = {
        "confinement_plan_digest": plan["digest"],
        "confinement_probe_passed": True,
        "result": "run_completed",
        "runtime_inputs_digest": independent_verifier.digest(attestation),
    }
    independent_verifier._verify_runtime_inputs(
        attestation, invocation, controller, build_identity
    )
    tampered = dict(attestation)
    tampered["regular_files"] = [
        dict(record) for record in attestation["regular_files"]
    ]
    tampered["regular_files"][0]["sha256"] = "0" * 64
    invocation["runtime_inputs_digest"] = independent_verifier.digest(tampered)
    with pytest.raises(
        independent_verifier.VerificationError,
        match="actual runtime file identity",
    ):
        independent_verifier._verify_runtime_inputs(
            tampered, invocation, controller, build_identity
        )
    tampered_resource = dict(attestation)
    tampered_resource["resources"] = [
        {
            "kind": "procfs",
            "operations": ["openat"],
            "path": "/etc/passwd",
        },
        *attestation["resources"],
    ]
    invocation["runtime_inputs_digest"] = independent_verifier.digest(tampered_resource)
    with pytest.raises(
        independent_verifier.VerificationError,
        match="absent from confinement plan",
    ):
        independent_verifier._verify_runtime_inputs(
            tampered_resource, invocation, controller, build_identity
        )


@pytest.mark.parametrize("missing", ["plugin", "model"])
def test_unbound_native_plugin_or_model_fails_closed(
    native_fixture: dict[str, Path], missing: str
) -> None:
    plan = _plan(
        native_fixture,
        bind_plugin=missing != "plugin",
        bind_model=missing != "model",
    )
    returncode, attestation = _run(native_fixture, plan, native_fixture["model"])
    assert returncode == -9
    assert attestation["state"] == "rejected"
    assert any(
        violation["reason"] == "unbound_regular_file"
        for violation in attestation["violations"]
    )


def test_bound_and_unbound_sequence_file_use_the_same_boundary(
    native_fixture: dict[str, Path],
) -> None:
    bound_code, bound_attestation = _run(
        native_fixture,
        _plan(native_fixture, data_kind="sequence"),
        native_fixture["sequence"],
    )
    assert bound_code == 0
    sequence_record = next(
        record
        for record in bound_attestation["regular_files"]
        if record["realpath"] == native_fixture["sequence"].as_posix()
    )
    assert "sequence" in sequence_record["bindings"]
    unbound_code, unbound_attestation = _run(
        native_fixture,
        _plan(native_fixture, bind_model=False, data_kind="sequence"),
        native_fixture["sequence"],
    )
    assert unbound_code == -9
    assert any(
        violation["path"] == native_fixture["sequence"].as_posix()
        and violation["reason"] == "unbound_regular_file"
        for violation in unbound_attestation["violations"]
    )


def test_native_path_traversal_fails_before_open(
    native_fixture: dict[str, Path],
) -> None:
    traversal = native_fixture["run_dir"] / "../../../model.bin"
    returncode, attestation = _run(native_fixture, _plan(native_fixture), traversal)
    assert returncode == -9
    assert any(
        violation["reason"] == "non_canonical_path"
        for violation in attestation["violations"]
    )


def test_output_symlink_cannot_hide_a_bound_input(
    native_fixture: dict[str, Path],
) -> None:
    alias = native_fixture["run_dir"] / "model-alias.bin"
    alias.symlink_to(native_fixture["model"])
    try:
        returncode, attestation = _run(native_fixture, _plan(native_fixture), alias)
    finally:
        alias.unlink()
    assert returncode == -9
    assert any(
        violation["reason"] == "unbound_output_alias"
        for violation in attestation["violations"]
    )


def test_symlink_substitution_fails_closed(
    native_fixture: dict[str, Path],
) -> None:
    plan = _plan(native_fixture)
    model = native_fixture["model"]
    original = native_fixture["root"] / "model.bound"
    substitute = native_fixture["root"] / "model.substitute"
    model.rename(original)
    substitute.write_bytes(original.read_bytes())
    model.symlink_to(substitute)
    try:
        returncode, attestation = _run(native_fixture, plan, model)
    finally:
        model.unlink()
        original.rename(model)
        substitute.unlink()
    assert returncode == -9
    assert attestation["state"] == "rejected"
    assert any(
        violation["reason"] in {"unbound_regular_file", "runtime_identity_drift"}
        for violation in attestation["violations"]
    )


def _repository_record(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    return {
        "git_object": hashlib.sha1(data, usedforsecurity=False).hexdigest(),
        "git_type": "blob",
        "kind": "regular",
        "length": len(data),
        "mode": "100644",
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_bound_and_unbound_python_imports_use_the_os_boundary(tmp_path: Path) -> None:
    python = Path(sys.executable).resolve(strict=True)
    (tmp_path / ".venv/bin").mkdir(parents=True)
    incomplete = tmp_path / "evidence.incomplete"
    (incomplete / "runs/r").mkdir(parents=True)
    probe = incomplete / "_runtime_confinement_denial_probe"
    probe.write_bytes(b"deny\n")
    script = tmp_path / "import_runner.py"
    allowed_module = tmp_path / "allowed_runtime_module.py"
    unbound_module = tmp_path / "unbound_runtime_module.py"
    allowed_module.write_text("VALUE = 7\n", encoding="utf-8")
    unbound_module.write_text("VALUE = 9\n", encoding="utf-8")
    script.write_text(
        """
import importlib
import os
import pathlib
import sys

if os.environ.get("H0_DISCOVER") != "1":
    try:
        pathlib.Path(sys.argv[1]).read_bytes()
    except PermissionError:
        pass
    else:
        raise SystemExit(91)
sys.path.insert(0, pathlib.Path(__file__).parent.as_posix())
module = importlib.import_module(sys.argv[2])
raise SystemExit(0 if module.VALUE in {7, 9} else 92)
""",
        encoding="utf-8",
    )
    discovery = subprocess.run(
        [
            python.as_posix(),
            "-I",
            "-B",
            "-S",
            "-v",
            script.as_posix(),
            probe.as_posix(),
            "allowed_runtime_module",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "H0_DISCOVER": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
        },
    )
    runtime_paths = [python, Path("/etc/ld.so.cache")]
    locale_archive = Path("/usr/lib/locale/locale-archive")
    if locale_archive.is_file():
        runtime_paths.append(locale_archive)
    locale_alias = Path("/usr/share/locale/locale.alias")
    if locale_alias.is_file():
        runtime_paths.append(locale_alias)
    gconv_cache = Path("/usr/lib/gconv/gconv-modules.cache")
    if gconv_cache.is_file():
        runtime_paths.append(gconv_cache)
    utc_zone = Path("/usr/share/zoneinfo/UTC")
    if utc_zone.is_file():
        runtime_paths.append(utc_zone)
    for system_file in (
        "/etc/group",
        "/etc/host.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/resolv.conf",
    ):
        path = Path(system_file)
        if path.is_file():
            runtime_paths.append(path)
    c_utf8 = Path("/usr/lib/locale/C.utf8")
    if c_utf8.is_dir():
        runtime_paths.extend(path for path in c_utf8.rglob("*") if path.is_file())
    runtime_paths.extend(
        Path(match)
        for match in re.findall(r"['\"](/[^'\"]+)['\"]", discovery.stderr)
        if Path(match).is_file()
        and Path(match) not in {script, allowed_module, unbound_module}
    )
    runtime_records = _ldd_inputs(python)
    known = {record["realpath"] for record in runtime_records}
    for path in runtime_paths:
        record = _identity(path, path)
        if record["realpath"] not in known:
            runtime_records.append(record)
            known.add(record["realpath"])
    python_record = _identity(python)
    inventory = {
        "models_engines": [],
        "repository": [
            _repository_record(tmp_path, path) for path in (allowed_module, script)
        ],
        "sequence": {"files": [], "root": "sequence"},
        "tool_runtime": runtime_records,
    }
    build_identity = {
        "artifacts": [],
        "python": {
            "length": python_record["length"],
            "path": python_record["realpath"],
            "sha256": python_record["sha256"],
        },
    }
    plan = confinement.build_plan(
        root=tmp_path,
        incomplete=incomplete,
        inventory=inventory,
        build_identity=build_identity,
        denial_probe=probe,
        run_ids=("r",),
    )

    def run(module: str) -> tuple[int, dict[str, Any]]:
        vector = [
            python.as_posix(),
            "-I",
            "-B",
            "-S",
            script.as_posix(),
            probe.as_posix(),
            module,
        ]
        with (
            open(os.devnull, "rb", buffering=0) as stdin,
            open(tmp_path / f"{module}.stdout", "xb", buffering=0) as stdout,
            open(tmp_path / f"{module}.stderr", "xb", buffering=0) as stderr,
        ):
            process = confinement.spawn_confined(
                vector,
                cwd=tmp_path,
                env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                plan=plan,
            )
            code = process.wait(timeout=10)
            return code, process.runtime_attestation()

    allowed_code, allowed_attestation = run("allowed_runtime_module")
    assert allowed_code == 0, allowed_attestation["violations"][0]
    assert allowed_attestation["state"] == "complete"
    assert any(
        "python_module" in record["roles"]
        and record["realpath"] == allowed_module.as_posix()
        for record in allowed_attestation["regular_files"]
    )
    unbound_code, unbound_attestation = run("unbound_runtime_module")
    assert unbound_code == -9
    assert unbound_attestation["state"] == "rejected"
    assert any(
        violation["path"] == unbound_module.as_posix()
        and violation["reason"] == "unbound_regular_file"
        for violation in unbound_attestation["violations"]
    )
