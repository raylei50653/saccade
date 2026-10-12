"""The runtime's N03 pin record (#549 S2-2, P-N03 rev 3d).

docs/reference/model_bundle_s2_2_n03_offline_proposal_549.md. The pin
(``shipping/n03_operator_pin.json``) is compiled into the loader; configure and
install refuse a pin that is not the tracked realization attestation's
``op_library``. Here, without a build:

* the record names one digest and size under policy ``n03-no-origin-v1``;
* it is the tracked attestation's ``op_library`` (sha256 and bytes);
* the package checker compares the packaged operator with it;
* where the build has the attested operator library, its bytes are the pin's.
"""

# scope: unit
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PIN = REPO / "shipping" / "n03_operator_pin.json"
ATTESTATION = REPO / "configs" / "shipping" / "mamba_head_realization.attestation.json"
OPERATOR = REPO / "build" / "libsaccade_scan_torchop.so"


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


def test_the_package_checker_reads_the_pin() -> None:
    source = (REPO / "scripts" / "native" / "check_shipping_package.py").read_text()
    assert 'N03_OPERATOR_PIN = "shipping/n03_operator_pin.json"' in source
    assert "is not the runtime's N03 pin" in source


@pytest.mark.skipif(not OPERATOR.exists(), reason="no build/libsaccade_scan_torchop.so")
def test_the_built_operator_library_is_the_pin() -> None:
    pin = json.loads(PIN.read_text())
    data = OPERATOR.read_bytes()
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (
        pin["sha256"],
        pin["bytes"],
    )
