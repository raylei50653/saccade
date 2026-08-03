"""Successor measurement controller consumption ordering and fail-closed CLI."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import h2_successor_authorization as authority  # noqa: E402
import run_h2_execution as producer  # noqa: E402
import run_h2_successor_measurement as controller  # noqa: E402


def _grant(request: dict[str, Any], authorization_id: str = "a" * 64) -> dict[str, Any]:
    return {
        "authority": request["authority"],
        "authorization_id": authorization_id,
        "execution_domain": request["execution_domain"],
        "execution_id": request["execution_id"],
        "execution_semantics_projection_digest": request[
            "execution_semantics_projection_digest"
        ],
        "issued_by": request["requested_issuer"],
        "phase": request["phase"],
        "resolved_run_spec_digest": request["resolved_run_spec_digest"],
        "schema": authority.GRANT_SCHEMA,
    }


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _Stages:
    def __init__(self) -> None:
        self.abandoned = False

    def abandon(self) -> None:
        self.abandoned = True


class _Execution:
    def __init__(self, ledger: Path, authorization_id: str, *, fail: bool = False):
        self.ledger = ledger
        self.authorization_id = authorization_id
        self.authorization_binding_digest: str | None = None
        self.stages = _Stages()
        self.fail = fail
        self.produced = False

    def produce(self, root: Path) -> dict[str, Any]:
        assert (self.ledger / f"{self.authorization_id}.json").is_file()
        assert self.authorization_binding_digest is not None
        self.produced = True
        if self.fail:
            raise producer.ProducerError("injected producer failure")
        root.mkdir(parents=True)
        return {
            "authority": "exactly_once_measurement",
            "result": "measurement_pass",
            "terminal": None,
        }


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail: bool = False,
) -> tuple[Path, Path, Path, _Execution]:
    ledger = tmp_path / "ledger"
    packet = tmp_path / "packet"
    build = tmp_path / "build"
    build.mkdir()
    request, _ = controller.request_for(execution_id="measurement-1", ledger=ledger)
    grant = _grant(request)
    grant_path = tmp_path / "grant.json"
    _write(grant_path, grant)
    execution = _Execution(ledger, grant["authorization_id"], fail=fail)
    monkeypatch.setattr(controller, "assert_exact_clean_head", lambda _: None)
    monkeypatch.setattr(
        producer,
        "configured_execution",
        lambda **_: execution,
    )
    return ledger, packet, grant_path, execution


def test_receipt_is_durable_before_the_producer_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger, packet, grant, execution = _setup(monkeypatch, tmp_path)
    incomplete, result = controller.execute(
        execution_id="measurement-1",
        grant_path=grant,
        packet=packet,
        selected_base="a" * 40,
        build_dir=tmp_path / "build",
        ledger=ledger,
        require_controlled_domain=False,
    )
    assert execution.produced
    assert result["result"] == "measurement_pass"
    assert (incomplete / authority.RECEIPT_NAME).is_file()
    assert len(list(ledger.glob("*.json"))) == 1


def test_a_spent_grant_cannot_launch_a_second_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger, packet, grant, first = _setup(monkeypatch, tmp_path)
    controller.execute(
        execution_id="measurement-1",
        grant_path=grant,
        packet=packet,
        selected_base="a" * 40,
        build_dir=tmp_path / "build",
        ledger=ledger,
        require_controlled_domain=False,
    )
    second = _Execution(ledger, "a" * 64)
    monkeypatch.setattr(producer, "configured_execution", lambda **_: second)
    with pytest.raises(authority.AuthorizationError, match="already consumed"):
        controller.execute(
            execution_id="measurement-1",
            grant_path=grant,
            packet=tmp_path / "packet-2",
            selected_base="a" * 40,
            build_dir=tmp_path / "build",
            ledger=ledger,
            require_controlled_domain=False,
        )
    assert not second.produced
    assert second.stages.abandoned
    assert first.produced


def test_failure_after_consumption_preserves_spent_incomplete_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger, packet, grant, execution = _setup(monkeypatch, tmp_path, fail=True)
    with pytest.raises(producer.ProducerError, match="injected"):
        controller.execute(
            execution_id="measurement-1",
            grant_path=grant,
            packet=packet,
            selected_base="a" * 40,
            build_dir=tmp_path / "build",
            ledger=ledger,
            require_controlled_domain=False,
        )
    incomplete = Path(f"{packet}.incomplete")
    assert (incomplete / authority.RECEIPT_NAME).is_file()
    assert len(list(ledger.glob("*.json"))) == 1
    assert execution.stages.abandoned


def test_grant_mismatch_creates_no_packet_or_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger, packet, grant_path, _ = _setup(monkeypatch, tmp_path)
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    grant["resolved_run_spec_digest"] = "f" * 64
    _write(grant_path, grant)
    with pytest.raises(authority.AuthorizationError, match="run_spec"):
        controller.execute(
            execution_id="measurement-1",
            grant_path=grant_path,
            packet=packet,
            selected_base="a" * 40,
            build_dir=tmp_path / "build",
            ledger=ledger,
            require_controlled_domain=False,
        )
    assert not Path(f"{packet}.incomplete").exists()
    assert not ledger.exists()


def test_exact_head_and_clean_checkout_are_preconsumption_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller,
        "_git",
        lambda *args: "b" * 40 if args == ("rev-parse", "HEAD") else " M file",
    )
    with pytest.raises(controller.MeasurementControllerError, match="current HEAD"):
        controller.assert_exact_clean_head("a" * 40)
    with pytest.raises(controller.MeasurementControllerError, match="clean"):
        controller.assert_exact_clean_head("b" * 40)


def test_production_cli_exposes_no_ledger_override() -> None:
    source = (_TOOLS / "run_h2_successor_measurement.py").read_text(encoding="utf-8")
    assert 'add_argument("--authorization-ledger"' not in source
    assert "ledger=authority.controlled_ledger()" in source
