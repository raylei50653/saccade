#!/usr/bin/env python3
"""Run the repeatable, non-authoritative H0 substrate qualification gate.

This tool intentionally is *not* a controller mode.  It never discovers a
freeze, reads a research sequence, writes an H0 evidence root, or produces an
H0 terminal.  It exercises the same host build and extension substrate with a
synthetic runner so ordinary engineering failures are found before a seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "h0_phase_a_qualification_v1"
REPORT_NAME = "qualification_summary.json"
GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
STEP_NAMES = (
    "configure",
    "build",
    "build_identity",
    "runtime_closure",
    "cuda_runtime_confinement",
    "extension_load",
    "t1_verdict_semantics",
    "runner_launch_preflight",
    "failure_envelope_serialization",
)


class QualificationError(RuntimeError):
    """The repeatable substrate gate did not complete."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _regular_file_record(path: Path) -> dict[str, Any]:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"qualification input is not a regular file: {path}")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise QualificationError(f"qualification input changed while hashing: {path}")
    return {
        "length": len(data),
        "path": path.as_posix(),
        "sha256": sha256_bytes(data),
    }


def _write_report(workspace: Path, report: Mapping[str, Any]) -> Path:
    identity = {
        "repository_head_sha",
        "repository_tree_sha",
        "requested_ref",
    }
    if not identity.issubset(report):
        raise QualificationError("qualification report lacks repository identity")
    if report.get("result") == "passed" and (
        not isinstance(report.get("requested_ref"), str)
        or not report["requested_ref"]
        or not isinstance(report.get("repository_head_sha"), str)
        or not GIT_OBJECT_ID.fullmatch(report["repository_head_sha"])
        or not isinstance(report.get("repository_tree_sha"), str)
        or not GIT_OBJECT_ID.fullmatch(report["repository_tree_sha"])
    ):
        raise QualificationError(
            "successful qualification report lacks resolved identity"
        )
    path = workspace / REPORT_NAME
    path.write_bytes(canonical_json_bytes(report) + b"\n")
    return path


def _workspace(root: Path, raw: Path) -> Path:
    candidate = raw.expanduser().resolve(strict=False)
    authoritative_build = root / "build/h0_phase_a"
    evidence = root / "docs/modules/semantic/research/evidence"
    if (
        candidate in {root, authoritative_build}
        or candidate == evidence
        or evidence in candidate.parents
    ):
        raise QualificationError(
            "qualification workspace overlaps authoritative H0 state"
        )
    if root in candidate.parents:
        allowed = root / "build/h0_qualification"
        if candidate != allowed and allowed not in candidate.parents:
            raise QualificationError(
                "in-repository qualification workspace must be below build/h0_qualification"
            )
    if candidate.exists():
        if not candidate.is_dir() or candidate.is_symlink() or any(candidate.iterdir()):
            raise QualificationError("qualification workspace must be absent or empty")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _tool(name: str) -> Path:
    value = shutil.which(name)
    if value is None:
        raise QualificationError(f"required qualification tool is absent: {name}")
    path = Path(value).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise QualificationError(
            f"qualification tool is not a physical regular file: {name}"
        )
    return path


def _venv_runtime_libraries(python: Path) -> dict[str, Path]:
    """Derive the frozen runtime library directories from the venv itself.

    The CUDA toolchain and runtime tree are uv-locked venv content (issue
    #214); nothing here may fall back to PATH or the rolling system toolkit.
    Mirrors the controller-input derivation in build_h0_preseal_freeze.py.
    """
    result = subprocess.run(
        [
            python.as_posix(),
            "-I",
            "-B",
            "-c",
            "import pathlib,sysconfig,torch; print((pathlib.Path(torch.__file__).resolve().parent/'lib').as_posix()); import tensorrt_libs; print(pathlib.Path(tensorrt_libs.__file__).resolve().parent.as_posix()); print(pathlib.Path(sysconfig.get_path('purelib')).resolve().as_posix())",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise QualificationError(
            "qualification venv could not derive the runtime library directories"
        )
    lines = result.stdout.decode("utf-8", errors="strict").splitlines()
    if len(lines) != 3:
        raise QualificationError(
            "qualification venv derived an unexpected library-directory shape"
        )
    cuda_root = Path(lines[2]).resolve(strict=True) / "nvidia/cu13"
    nvcc = cuda_root / "bin/nvcc"
    if nvcc.is_symlink() or not nvcc.is_file():
        raise QualificationError(
            "frozen venv CUDA compiler is absent or unsafe (run `uv sync --frozen`)"
        )
    libraries = {
        "tensorrt_library_dir": Path(lines[1]),
        "pytorch_library_dir": Path(lines[0]),
        "cuda_library_dir": cuda_root / "lib",
        "nvcc": nvcc,
    }
    for name in ("tensorrt_library_dir", "pytorch_library_dir", "cuda_library_dir"):
        directory = libraries[name]
        if directory.is_symlink() or not directory.is_dir():
            raise QualificationError(
                f"qualification runtime library directory is absent or unsafe: {name}"
            )
    return libraries


def _assert_cuda_confinement(
    artifacts: Iterable[Path], cuda_library_dir: Path, library_path: str
) -> None:
    """Fail closed if a CUDA library the frozen venv provides resolves elsewhere.

    Sonames the venv tree does not provide (e.g. system OpenCV's own CUDA-side
    dependencies) are outside this check's authority and stay recorded in the
    runtime closure instead.
    """
    cuda_library_dir = cuda_library_dir.resolve(strict=True)
    provided = {entry.name for entry in cuda_library_dir.iterdir() if entry.is_file()}
    environment = {"LD_LIBRARY_PATH": library_path, "PATH": "/usr/bin:/bin"}
    for artifact in artifacts:
        result = subprocess.run(
            ["ldd", artifact.as_posix()],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        if result.returncode:
            raise QualificationError(
                f"ldd rejected qualification artifact during CUDA confinement: {artifact}"
            )
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            if "=>" not in line:
                continue
            left, right = line.split("=>", 1)
            soname = left.strip().split(" ", 1)[0]
            target = right.strip().split(" ", 1)[0]
            if soname not in provided or not target.startswith("/"):
                continue
            real = Path(target).resolve(strict=True)
            if not real.is_relative_to(cuda_library_dir):
                raise QualificationError(
                    f"CUDA runtime dependency escaped the frozen venv tree: {soname} -> {real}"
                )


def _git_object_id(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "-C", root.as_posix(), "rev-parse", "--verify", revision],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    value = result.stdout.decode("utf-8", errors="strict").strip()
    if result.returncode or not GIT_OBJECT_ID.fullmatch(value):
        raise QualificationError(f"cannot resolve qualification {revision}")
    return value


def _repository_identity(root: Path, requested_ref: str) -> dict[str, str]:
    if not requested_ref:
        raise QualificationError("qualification requested ref is empty")
    return {
        "repository_head_sha": _git_object_id(root, "HEAD^{commit}"),
        "repository_tree_sha": _git_object_id(root, "HEAD^{tree}"),
        "requested_ref": requested_ref,
    }


def _run(
    vector: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> None:
    try:
        result = subprocess.run(
            list(vector),
            cwd=root,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_bytes(exc.stdout or b"")
        stderr_path.write_bytes(exc.stderr or b"")
        raise QualificationError(
            f"qualification command timed out: {vector[0]}"
        ) from exc
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode:
        raise QualificationError(
            f"qualification command exited {result.returncode}: {vector[0]}"
        )


def _cmake_cache(cache: Path) -> dict[str, str]:
    if cache.is_symlink() or not cache.is_file():
        raise QualificationError("qualification CMakeCache.txt is absent or unsafe")
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if ":" not in left:
            continue
        key, _type = left.split(":", 1)
        values[key] = value
    return values


def _dynamic_closure(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ldd", path.as_posix()],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise QualificationError(f"ldd rejected qualification artifact: {path}")
    records: list[dict[str, Any]] = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if "not found" in line:
            raise QualificationError(
                f"qualification closure has missing dependency: {line}"
            )
        if "=>" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        if target.startswith("/"):
            records.append(_regular_file_record(Path(target).resolve(strict=True)))
    return sorted(records, key=lambda record: record["path"].encode("utf-8"))


def _build_identity(root: Path, build: Path, python: Path) -> dict[str, Any]:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise QualificationError("Python EXT_SUFFIX is unavailable")
    extension = build / f"saccade_tracking_ext{suffix}"
    plugin = build / "libsaccade_scan_plugin.so"
    cache = build / "CMakeCache.txt"
    cache_values = _cmake_cache(cache)
    # The project declares `project(Saccade CXX CUDA)` and has no C sources,
    # so CMake does not populate CMAKE_C_COMPILER. Match the authoritative
    # controller's C++/CUDA build identity requirements.
    compiler_keys = ("CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER")
    if any(not cache_values.get(key) for key in compiler_keys):
        raise QualificationError("qualification CMake cache lacks C++/CUDA compilers")
    compilers = {
        key.removeprefix("CMAKE_")
        .removesuffix("_COMPILER")
        .lower(): _regular_file_record(Path(cache_values[key]).resolve(strict=True))
        for key in compiler_keys
    }
    artifacts = [_regular_file_record(extension), _regular_file_record(plugin)]
    return {
        "artifacts": artifacts,
        "cmake_cache_sha256": sha256_file(cache),
        "compilers": compilers,
        "python": _regular_file_record(python.resolve(strict=True)),
        "python_ext_suffix": suffix,
    }


def _qualification_inventory(root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    inputs = [
        _regular_file_record(root / "CMakeLists.txt"),
        _regular_file_record(root / "scripts/tools/run_h0_phase_a.py"),
        _regular_file_record(
            root / "scripts/tools/h0_phase_a_execution_schema_v1.json"
        ),
        *identity["artifacts"],
        *identity["compilers"].values(),
        identity["python"],
    ]
    records = sorted(inputs, key=lambda record: record["path"].encode("utf-8"))
    return {
        "algorithm": "h0_qualification_inputs_v1",
        "digest": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def _failure_probe() -> dict[str, Any]:
    """Exercise the truthful failure-envelope shape without an H0 terminal."""
    import run_h0_phase_a as controller

    checkpoint = controller._failed_checkpoint(
        "T1",
        cause="inventory_mismatch",
        inventory_comparison_executed=True,
        inventory_equal=False,
        observed_digest="0" * 64,
    )
    failure = controller._failure_record(
        "qualification",
        controller.CheckpointDriftError(
            "T1", "synthetic qualification inequality", checkpoint_record=checkpoint
        ),
    )
    # The report serializer is deliberately exercised over the controller's
    # exact producer output, while the enclosing result remains non-authoritative.
    serialized = json.loads(
        canonical_json_bytes({"failure": failure, "row": checkpoint})
    )
    return {
        "failure": serialized["failure"],
        "qualification_only": True,
        "row": serialized["row"],
    }


def qualification_runner_argv(build: Path, extension: Path, plugin: Path) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-B",
        (ROOT / "scripts/tools/qualify_h0_phase_a_child.py").as_posix(),
        "--build-dir",
        build.as_posix(),
        "--extension",
        extension.as_posix(),
        "--plugin",
        plugin.as_posix(),
    ]


def _step(name: str, action: Callable[[], None], steps: list[dict[str, str]]) -> None:
    action()
    steps.append({"name": name, "state": "passed"})


def run_qualification(
    root: Path,
    workspace: Path,
    *,
    requested_ref: str,
    timeout: float,
) -> dict[str, Any]:
    """Run every pre-seal substrate step without entering H0 authority."""
    root = root.resolve(strict=True)
    workspace = _workspace(root, workspace)
    logs = workspace / "logs"
    logs.mkdir()
    environment_root = workspace / "environment"
    for name in ("home", "tmp", "xdg-cache"):
        (environment_root / name).mkdir(parents=True)
    steps: list[dict[str, str]] = []
    identity: dict[str, Any] | None = None
    closure: dict[str, list[dict[str, Any]]] | None = None
    t1_verdict_semantics: dict[str, Any] | None = None
    repository_identity: dict[str, str | None] = {
        "repository_head_sha": None,
        "repository_tree_sha": None,
        "requested_ref": requested_ref,
    }
    try:
        repository_identity = _repository_identity(root, requested_ref)
        uv = _tool("uv")
        python = root / ".venv/bin/python"
        if python.is_symlink() or not python.is_file():
            raise QualificationError("qualification Python is absent or unsafe")
        libraries = _venv_runtime_libraries(python)
        nvcc = libraries["nvcc"]
        build = workspace / "build"
        environment = {
            "CUDACXX": nvcc.as_posix(),
            "HOME": (environment_root / "home").as_posix(),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": f"{root}/.venv/bin:/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": (environment_root / "tmp").as_posix(),
            "TZ": "UTC",
            "XDG_CACHE_HOME": (environment_root / "xdg-cache").as_posix(),
        }
        configure = [
            uv.as_posix(),
            "run",
            "--frozen",
            "cmake",
            "--fresh",
            "-S",
            root.as_posix(),
            "-B",
            build.as_posix(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_NATIVE_TESTS=OFF",
            "-DSACCADE_ENABLE_NVTX=ON",
            f"-DPython3_EXECUTABLE={python.as_posix()}",
        ]
        _step(
            "configure",
            lambda: _run(
                configure,
                root=root,
                environment=environment,
                stdout_path=logs / "configure.stdout.log",
                stderr_path=logs / "configure.stderr.log",
                timeout=timeout,
            ),
            steps,
        )
        build_vector = [
            uv.as_posix(),
            "run",
            "--frozen",
            "cmake",
            "--build",
            build.as_posix(),
            "--target",
            "saccade_tracking_ext",
            "saccade_scan_plugin",
            "--parallel",
            "1",
        ]
        _step(
            "build",
            lambda: _run(
                build_vector,
                root=root,
                environment=environment,
                stdout_path=logs / "build.stdout.log",
                stderr_path=logs / "build.stderr.log",
                timeout=timeout,
            ),
            steps,
        )
        identity = _build_identity(root, build, python)
        steps.append({"name": "build_identity", "state": "passed"})
        closure = {
            artifact["path"]: _dynamic_closure(Path(artifact["path"]))
            for artifact in identity["artifacts"]
        }
        steps.append({"name": "runtime_closure", "state": "passed"})
        extension = Path(identity["artifacts"][0]["path"])
        plugin = Path(identity["artifacts"][1]["path"])
        # Controller closure semantics (run_h0_phase_a BUILD template):
        # build : tensorrt : pytorch : cuda — no ambient passthrough.
        library_path = ":".join(
            (
                build.as_posix(),
                libraries["tensorrt_library_dir"].as_posix(),
                libraries["pytorch_library_dir"].as_posix(),
                libraries["cuda_library_dir"].as_posix(),
            )
        )
        _step(
            "cuda_runtime_confinement",
            lambda: _assert_cuda_confinement(
                (extension, plugin), libraries["cuda_library_dir"], library_path
            ),
            steps,
        )
        load_script = (
            "import ctypes,pathlib,sys;"
            f"sys.path.insert(0,{build.as_posix()!r});"
            "import saccade_tracking_ext;"
            f"assert pathlib.Path(saccade_tracking_ext.__file__).resolve()==pathlib.Path({extension.as_posix()!r}).resolve();"
            f"ctypes.CDLL({plugin.as_posix()!r},mode=ctypes.RTLD_LOCAL)"
        )
        runtime_environment = {
            **environment,
            "LD_LIBRARY_PATH": library_path,
            "SACCADE_BUILD_PATH": build.as_posix(),
            "H0_QUALIFICATION_MODE": "1",
        }
        runner = [python.as_posix(), "-I", "-B", "-c", load_script]
        _step(
            "extension_load",
            lambda: _run(
                runner,
                root=root,
                environment=runtime_environment,
                stdout_path=logs / "extension_load.stdout.log",
                stderr_path=logs / "extension_load.stderr.log",
                timeout=timeout,
            ),
            steps,
        )
        import run_h0_phase_a as controller

        before = _qualification_inventory(root, identity)
        after = _qualification_inventory(root, identity)
        t1_verdict_semantics = controller._checkpoint_inventory_verdict(
            "T1", before, after
        )
        steps.append({"name": "t1_verdict_semantics", "state": "passed"})
        runner_preflight = qualification_runner_argv(build, extension, plugin)
        _step(
            "runner_launch_preflight",
            lambda: _run(
                runner_preflight,
                root=root,
                environment=runtime_environment,
                stdout_path=logs / "runner_preflight.stdout.log",
                stderr_path=logs / "runner_preflight.stderr.log",
                timeout=timeout,
            ),
            steps,
        )
        failure_probe = _failure_probe()
        if failure_probe["failure"]["stage"] != "checkpoint_T1":
            raise QualificationError("synthetic failure envelope stage drift")
        steps.append({"name": "failure_envelope_serialization", "state": "passed"})
    except BaseException as exc:
        report = {
            "authority": "non_authoritative",
            "capture": "forbidden",
            "error": str(exc) or exc.__class__.__name__,
            "phase_b": "forbidden",
            "research_inputs": "forbidden",
            **repository_identity,
            "result": "failed",
            "schema": SCHEMA,
            "steps": steps,
            "terminal_claim": "forbidden",
            "workspace": workspace.as_posix(),
        }
        _write_report(workspace, report)
        if isinstance(exc, QualificationError):
            raise
        raise QualificationError(str(exc) or exc.__class__.__name__) from exc
    report = {
        "authority": "non_authoritative",
        "build_identity": identity,
        "capture": "forbidden",
        "failure_envelope_probe": failure_probe,
        "phase_b": "forbidden",
        **repository_identity,
        "research_inputs": "forbidden",
        "result": "passed",
        "runtime_closure": closure,
        "schema": SCHEMA,
        "steps": steps,
        "t1_verdict_semantics": t1_verdict_semantics,
        "terminal_claim": "forbidden",
        "workspace": workspace.as_posix(),
    }
    _write_report(workspace, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--requested-ref", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    started = time.monotonic()
    try:
        report = run_qualification(
            ROOT,
            args.workspace,
            requested_ref=args.requested_ref,
            timeout=args.timeout,
        )
    except QualificationError as exc:
        print(f"H0 qualification rejected: {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {"elapsed_seconds": round(elapsed, 3), "result": report["result"]},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
