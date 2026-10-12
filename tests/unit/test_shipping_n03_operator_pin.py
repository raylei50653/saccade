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
* the package checker refuses a packaged operator that is not the pin, and,
  end to end through ``manifest_head`` on a fake source commit, an unreadable
  pin once the commit's build compiles it (never taken for an older commit);
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


# ── manifest_head end to end, on a fake source commit ─────────────────────────

_COMMIT = "1" * 40
_OP_SHA = hashlib.sha256(b"op").hexdigest()


def _source(cmake: bytes, pin: Any) -> dict[str, bytes]:
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
        "shipping/CMakeLists.txt": cmake,
    }
    for role in pkg.TRACKED_ROLES:
        files.setdefault(pkg.MODEL_ROOT_ROLES[role], b"tracked " + role.encode())
    if pin is not None:
        files[pkg.N03_OPERATOR_PIN] = pin
    return files


def _manifest_head(
    monkeypatch: pytest.MonkeyPatch, source: dict[str, bytes], op_sha: str = _OP_SHA
) -> str:
    """manifest_head's refusal (PackageError text) for a package whose model
    root matches `source`; the vendor object is missing on purpose, so a
    package that passes the N03 check fails at the next one."""
    pkg = _package_checker()

    def show(commit: str, path: str) -> bytes:
        assert commit == _COMMIT
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
        pkg.manifest_head(_COMMIT, True, True, files)
    return str(e.value)


_REV3D_CMAKE = (
    b"set(SACCADE_N03_OPERATOR_PIN ${CMAKE_CURRENT_SOURCE_DIR}/n03_operator_pin.json)\n"
)
_PIN = json.dumps({"sha256": _OP_SHA, "bytes": 2}).encode()


def test_manifest_head_passes_the_pinned_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _manifest_head(monkeypatch, _source(_REV3D_CMAKE, _PIN))
    assert error == "lib/vendor/libx.so.1 is not in the package"  # past the N03 check


def test_manifest_head_refuses_an_unreadable_pin_once_the_build_compiles_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _manifest_head(monkeypatch, _source(_REV3D_CMAKE, None))
    assert error == (
        f"shipping/n03_operator_pin.json cannot be read at {_COMMIT[:12]}, whose build compiles it"
    )


@pytest.mark.parametrize("pin", [json.dumps({"sha256": "e" * 64}).encode(), b"{"])
def test_manifest_head_refuses_an_operator_that_is_not_the_pin(
    monkeypatch: pytest.MonkeyPatch, pin: bytes
) -> None:
    error = _manifest_head(monkeypatch, _source(_REV3D_CMAKE, pin))
    assert "N03 pin" in error or "is not JSON" in error, error


def test_manifest_head_skips_the_pin_before_rev_3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source commit whose shipping CMake compiles no pin has none to check."""
    error = _manifest_head(monkeypatch, _source(b"# before rev 3d\n", None))
    assert error == "lib/vendor/libx.so.1 is not in the package"


def test_the_repository_build_compiles_the_pin() -> None:
    pkg = _package_checker()
    cmake = (REPO / "shipping" / "CMakeLists.txt").read_bytes()
    assert pkg.N03_PIN_BUILD_MARKER in cmake


@pytest.mark.skipif(not OPERATOR.exists(), reason="no build/libsaccade_scan_torchop.so")
def test_the_built_operator_library_is_the_pin() -> None:
    pin = json.loads(PIN.read_text())
    data = OPERATOR.read_bytes()
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (
        pin["sha256"],
        pin["bytes"],
    )
