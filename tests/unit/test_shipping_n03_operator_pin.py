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
* the package checker's ``check_n03_pin`` refuses a packaged operator that is
  not the pin, and ``manifest_head`` calls it;
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
    with pytest.raises(pkg.PackageError, match="is not the runtime's N03 pin"):
        pkg.check_n03_pin("share/saccade/build/op.so", "e" * 64, pin)
    source = (REPO / "scripts" / "native" / "check_shipping_package.py").read_text()
    body = source[source.index("def manifest_head(") :]
    assert 'check_n03_pin(op_file, entry(op_file)["sha256"], n03_pin)' in body


@pytest.mark.skipif(not OPERATOR.exists(), reason="no build/libsaccade_scan_torchop.so")
def test_the_built_operator_library_is_the_pin() -> None:
    pin = json.loads(PIN.read_text())
    data = OPERATOR.read_bytes()
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (
        pin["sha256"],
        pin["bytes"],
    )
