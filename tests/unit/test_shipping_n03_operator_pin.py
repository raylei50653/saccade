"""The runtime's N03 pin record (#549 S2-2, P-N03 rev 3d).

docs/reference/model_bundle_s2_2_n03_offline_proposal_549.md. The pin
(``shipping/n03_operator_pin.json``) is compiled into the loader; configure and
install refuse a pin that is not the tracked realization attestation's
``op_library``. Here, without a build:

* the record names one digest and size under policy ``n03-no-origin-v1``;
* it is the tracked attestation's ``op_library`` (sha256 and bytes);
* ``shipping/cmake/n03_pin.cmake`` (the rule configure and install apply)
  refuses a pin of another digest or size than the attestation's, and a
  malformed pin (skips without ``cmake``);
* the package checker grants the no-pin exception only to proven ancestors
  of the last pre-rev-3d main, with real-git history controls (#585); unknown
  commits, shallow history and unreadable required pins refuse, including
  end to end through ``manifest_head`` with synthetic source files;
* where the build has the attested operator library, its bytes are the pin's.
"""

# scope: unit
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PIN = REPO / "shipping" / "n03_operator_pin.json"
ATTESTATION = REPO / "configs" / "shipping" / "mamba_head_realization.attestation.json"
OPERATOR = REPO / "build" / "libsaccade_scan_torchop.so"
RULE = REPO / "shipping" / "cmake" / "n03_pin.cmake"
SHA = "b" * 64


def _package_checker() -> ModuleType:
    sys.path.insert(0, str(REPO / "scripts" / "native"))
    path = REPO / "scripts" / "native" / "check_shipping_package.py"
    spec = importlib.util.spec_from_file_location("check_shipping_package", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule(
    tmp_path: Path, pin: Any, op: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    (tmp_path / "att.json").write_text(json.dumps({"op_library": op}))
    return subprocess.run(
        [cmake, f"-DPIN={tmp_path / 'pin.json'}", f"-DATTESTATION={tmp_path / 'att.json'}", "-P", str(RULE)],
        capture_output=True,
        text=True,
    )  # fmt: skip


def test_the_pin_is_one_digest_under_the_admission_policy() -> None:
    pin = json.loads(PIN.read_text())
    assert pin["schema"] == "saccade.n03_operator_pin/v1"
    assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    assert isinstance(pin["bytes"], int) and pin["bytes"] > 0
    assert pin["admission_policy"] == "n03-no-origin-v1"
    assert pin["attestation"] == ATTESTATION.relative_to(REPO).as_posix()
    assert (REPO / pin["decision"]).is_file()


def test_the_pin_is_the_attested_operator_library() -> None:
    pin = json.loads(PIN.read_text())
    op = json.loads(ATTESTATION.read_text())["op_library"]
    assert (pin["sha256"], pin["bytes"]) == (op["sha256"], op["bytes"])


def test_the_rule_accepts_a_pin_that_is_the_attestations(tmp_path: Path) -> None:
    r = _rule(tmp_path, {"sha256": SHA, "bytes": 5}, {"sha256": SHA, "bytes": 5})
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "op",
    [{"sha256": "e" * 64, "bytes": 5}, {"sha256": SHA, "bytes": 6}, {"sha256": SHA}],
)
def test_the_rule_refuses_a_pin_that_is_not_the_attestations(
    tmp_path: Path, op: dict[str, Any]
) -> None:
    r = _rule(tmp_path, {"sha256": SHA, "bytes": 5}, op)
    assert r.returncode != 0
    assert "is not the runtime's N03 pin" in r.stderr, r.stderr


@pytest.mark.parametrize(
    "pin",
    [{"sha256": SHA.upper(), "bytes": 5}, {"sha256": SHA[:63], "bytes": 5}, {"sha256": SHA, "bytes": 0}, {"bytes": 5}],
)  # fmt: skip
def test_the_rule_refuses_a_malformed_pin(tmp_path: Path, pin: Any) -> None:
    r = _rule(tmp_path, pin, {"sha256": SHA, "bytes": 5})
    assert r.returncode != 0
    assert "is not an N03 pin" in r.stderr, r.stderr


def test_the_package_checker_refuses_an_operator_that_is_not_the_pin() -> None:
    pkg = _package_checker()
    pin = json.dumps({"sha256": SHA, "bytes": 5}).encode()
    pkg.check_n03_pin("share/saccade/build/op.so", SHA, pin)
    pkg.check_n03_pin("share/saccade/build/op.so", "e" * 64, None)  # before rev 3d
    for bad in (pin, b"not json", b"[]"):
        with pytest.raises(pkg.PackageError, match="N03 pin|is not JSON"):
            pkg.check_n03_pin("share/saccade/build/op.so", "e" * 64, bad)


# ── real-git history: only proven pre-rev-3d ancestors may omit the pin ────────

_OP_SHA = hashlib.sha256(b"op").hexdigest()
_PIN = json.dumps({"sha256": _OP_SHA, "bytes": 2}).encode()


@pytest.fixture
def pin_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, dict[str, str], Path]:
    repo = tmp_path / "history"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "--initial-branch=main")
    git("config", "user.name", "N03 test")
    git("config", "user.email", "n03-test@example.invalid")
    git("config", "commit.gpgsign", "false")

    def commit(label: str) -> str:
        (repo / "history.txt").write_text(label)
        git("add", ".")
        git("commit", "-q", "-m", label)
        return git("rev-parse", "HEAD")

    commits = {"ancestor": commit("ancestor"), "boundary": commit("boundary")}
    commits["descendant"] = commit("descendant without a pin")
    (repo / "shipping").mkdir()
    (repo / "shipping/n03_operator_pin.json").write_bytes(_PIN)
    commits["pinned"] = commit("descendant with a pin")
    git("switch", "-q", "-c", "sibling", commits["ancestor"])
    commits["sibling"] = commit("sibling without a pin")
    pkg = _package_checker()
    monkeypatch.setattr(pkg, "REPO", repo)
    monkeypatch.setattr(pkg, "N03_PIN_PRE_REV3D_MAIN", commits["boundary"])
    return pkg, commits, repo


@pytest.mark.parametrize("label", ["ancestor", "boundary"])
def test_package_pin_exception_requires_proven_pre_rev_3d_ancestry(
    pin_history: tuple[ModuleType, dict[str, str], Path], label: str
) -> None:
    pkg, commits, _ = pin_history
    commit = commits[label]
    assert pkg.read_n03_pin(lambda path: pkg._git_show(commit, path), commit) is None


@pytest.mark.parametrize("label", ["descendant", "sibling"])
def test_package_refuses_a_non_ancestor_without_a_readable_pin(
    pin_history: tuple[ModuleType, dict[str, str], Path], label: str
) -> None:
    pkg, commits, _ = pin_history
    commit = commits[label]
    with pytest.raises(pkg.PackageError, match="cannot be read.*requires the N03 pin"):
        pkg.read_n03_pin(lambda path: pkg._git_show(commit, path), commit)


def test_package_reads_a_non_ancestors_pin(
    pin_history: tuple[ModuleType, dict[str, str], Path],
) -> None:
    pkg, commits, _ = pin_history
    commit = commits["pinned"]
    assert pkg.read_n03_pin(lambda path: pkg._git_show(commit, path), commit) == _PIN


def test_package_refuses_an_unknown_source_commit_even_with_a_readable_pin(
    pin_history: tuple[ModuleType, dict[str, str], Path],
) -> None:
    pkg, _, _ = pin_history
    with pytest.raises(pkg.PackageError, match="git ancestry check failed"):
        pkg.read_n03_pin(lambda _: _PIN, "1" * 40)


def test_package_refuses_an_unavailable_history_boundary(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg, commits, _ = pin_history
    monkeypatch.setattr(pkg, "N03_PIN_PRE_REV3D_MAIN", "1" * 40)
    with pytest.raises(pkg.PackageError, match="git ancestry check failed"):
        pkg.read_n03_pin(lambda _: _PIN, commits["pinned"])


def test_package_refuses_a_shallow_repository_even_with_both_commits_available(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg, commits, repo = pin_history
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth=3",
            "--branch=main",
            repo.as_uri(),
            str(shallow),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(pkg, "REPO", shallow)
    assert pkg._git("rev-parse", "--is-shallow-repository").strip() == "true"
    pkg._git("cat-file", "-e", f"{commits['boundary']}^{{commit}}")
    with pytest.raises(pkg.PackageError, match="shallow repository"):
        pkg.read_n03_pin(lambda _: _PIN, commits["pinned"])


# ── manifest_head end to end: real history, synthetic source/package files ────


def _source(pin: bytes | None) -> dict[str, bytes]:
    """The repository files manifest_head reads at the source commit, up to the
    N03 pin check (``pin`` None: no pin file)."""
    pkg = _package_checker()
    att = {
        "op_library": {"path": "build/libsaccade_scan_torchop.so", "sha256": _OP_SHA}
    }
    files = {
        "pyproject.toml": b'[project]\nversion = "0.0.0"\n',
        "shipping/third_party_set.json": json.dumps(
            {"entries": [{"wheel": "w-1", "soname": "libx.so.1", "sha256": "a" * 64}]}
        ).encode(),
        "shipping/entrypoint_pin.json": json.dumps({"sha256": "c" * 64}).encode(),
        pkg.g2.ATTESTATION: json.dumps(att).encode(),
    }
    for role in pkg.TRACKED_ROLES:
        files.setdefault(pkg.MODEL_ROOT_ROLES[role], b"tracked " + role.encode())
    if pin is not None:
        files[pkg.N03_OPERATOR_PIN] = pin
    return files


def _manifest_head(
    monkeypatch: pytest.MonkeyPatch,
    pkg: ModuleType,
    commit: str,
    source: dict[str, bytes],
    op_sha: str = _OP_SHA,
) -> str:
    """manifest_head's refusal (PackageError text) for a package whose model
    root matches `source`; the vendor object is missing on purpose, so a
    package that passes the N03 check fails at the next one."""
    import check_shipping_bundle as bundle

    monkeypatch.setattr(
        bundle, "THIRD_PARTY_SET", pkg.REPO / "shipping/third_party_set.json"
    )
    monkeypatch.setattr(
        bundle, "ENTRYPOINT_PIN", pkg.REPO / "shipping/entrypoint_pin.json"
    )

    def show(source_commit: str, path: str) -> bytes:
        assert source_commit == commit
        if path not in source:
            raise subprocess.CalledProcessError(
                128, ["git", "show", f"{commit}:{path}"]
            )
        return source[path]

    monkeypatch.setattr(pkg, "_git_show", show)
    mr = pkg.g2.MODEL_ROOT
    files = {f"{mr}/{p}": {"sha256": "d" * 64} for p in pkg.MODEL_ROOT_ROLES.values()}
    for role in pkg.TRACKED_ROLES:
        path = pkg.MODEL_ROOT_ROLES[role]
        files[f"{mr}/{path}"] = {"sha256": hashlib.sha256(source[path]).hexdigest()}
    files[f"{mr}/build/libsaccade_scan_torchop.so"] = {"sha256": op_sha}
    with pytest.raises(pkg.PackageError) as e:
        pkg.manifest_head(commit, True, True, files)
    return str(e.value)


def test_manifest_head_passes_the_pinned_operator(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg, commits, _ = pin_history
    error = _manifest_head(monkeypatch, pkg, commits["pinned"], _source(_PIN))
    assert error == "lib/vendor/libx.so.1 is not in the package"  # past the N03 check


@pytest.mark.parametrize("label", ["descendant", "sibling"])
def test_manifest_head_refuses_an_unreadable_pin_outside_pre_rev_3d_history(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    pkg, commits, _ = pin_history
    commit = commits[label]
    error = _manifest_head(monkeypatch, pkg, commit, _source(None))
    assert error == (
        f"shipping/n03_operator_pin.json cannot be read at {commit[:12]}, which requires the N03 pin"
    )


@pytest.mark.parametrize("pin", [json.dumps({"sha256": "e" * 64}).encode(), b"{"])
def test_manifest_head_refuses_an_operator_that_is_not_the_pin(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    pin: bytes,
) -> None:
    pkg, commits, _ = pin_history
    error = _manifest_head(monkeypatch, pkg, commits["pinned"], _source(pin))
    assert "N03 pin" in error or "is not JSON" in error, error


@pytest.mark.parametrize("label", ["ancestor", "boundary"])
def test_manifest_head_skips_the_pin_before_rev_3d(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    pkg, commits, _ = pin_history
    error = _manifest_head(monkeypatch, pkg, commits[label], _source(None))
    assert error == "lib/vendor/libx.so.1 is not in the package"


def test_manifest_head_refuses_an_unknown_source_commit(
    pin_history: tuple[ModuleType, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg, _, _ = pin_history
    error = _manifest_head(monkeypatch, pkg, "1" * 40, _source(_PIN))
    assert "git ancestry check failed" in error


@pytest.mark.skipif(not OPERATOR.exists(), reason="no build/libsaccade_scan_torchop.so")
def test_the_built_operator_library_is_the_pin() -> None:
    pin = json.loads(PIN.read_text())
    data = OPERATOR.read_bytes()
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (
        pin["sha256"],
        pin["bytes"],
    )
