"""Fail-closed contracts for the Issue #112 v2 terminal verifier.

These fixtures pin the choices that would otherwise silently drift and change a
terminal: the agreement denominator, the inclusive decision boundary, the tie
policy, the quantile method, partition conservation, and the non-compensatory
box logic.

`s0 = w·½(fwd+bwd) + (1−w)·dist_h` with `w = sqrt(clip(speed/0.12, 0, 1))`, so
setting `lost_exit_speed = 0` gives `w = 0` and `s0 = dist_h` exactly. Every
fixture below uses that to control the proxy score directly.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/tools/run_d0_runtime_shadow_fidelity.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("d0_v2_verifier", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_study(
    tmp_path: Path,
    events: list[dict[str, Any]],
    *,
    shadow: bool = True,
    overflow: int = 0,
) -> tuple[Path, Path]:
    """Build a synthetic study + substrate from (partition, s0, bdist) events."""
    study = tmp_path / "study"
    substrate = tmp_path / "substrate"
    study.mkdir(parents=True)
    substrate.mkdir(parents=True)

    pair_rows = []
    cap_rows = []
    for i, ev in enumerate(events):
        gid_lost, gid_cand = 100 + 2 * i, 101 + 2 * i
        seq = ev.get("seq", "MOT17-02-SDP")
        # w = 0 (speed 0) => s0 == dist_h exactly.
        if ev["partition"] == "matched":
            pair_rows.append(
                {
                    "seq": seq,
                    "lost_id": gid_lost,
                    "cand_id": gid_cand,
                    "lost_exit_speed": 0.0,
                    "fwd_resid": 0.0,
                    "bwd_resid": 0.0,
                    "dist_h": ev["s0"],
                    "gap": ev.get("gap", 10),
                }
            )
        keyed = ev["partition"] != "unemitted"
        cap_rows.append(
            {
                "event_key": f"{seq}|{gid_lost}|{gid_cand}" if keyed else "",
                "event_key_version": "d0_event_key_v2_global",
                "partition": ev["partition"],
                "seq": seq,
                "lost_global_id": gid_lost if keyed else -1,
                "cand_global_id": gid_cand if keyed else -1,
                "lost_local_id": gid_lost,
                "cand_local_id": gid_cand,
                "gap": ev.get("gap", 10),
                "la": ev.get("la", 13),
                "bdist": ev["bdist"],
                "dist_h": ev["bdist"],
                "fwd_r": ev["bdist"],
                "bwd_r": ev["bdist"],
            }
        )

    with (study / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    with gzip.open(study / "capture.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cap_rows[0].keys()))
        w.writeheader()
        w.writerows(cap_rows)

    (study / "capture.csv.gz.manifest.json").write_text(
        json.dumps(
            {
                "overflow_events": overflow,
                "provenance": {
                    "shadow": shadow,
                    "capture_contract": "d0_runtime_cuda_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    (substrate / "_global_id_map.txt").write_text("", encoding="utf-8")
    (substrate / "MOT17-02-SDP.txt").write_text("", encoding="utf-8")
    return study, substrate


def _matched(n: int, *, s0: float, bdist: float, **kw: Any) -> list[dict[str, Any]]:
    return [{"partition": "matched", "s0": s0, "bdist": bdist, **kw} for _ in range(n)]


def _faithful(n: int = 1200) -> list[dict[str, Any]]:
    """A proxy that tracks the runtime value exactly -> all boxes pass."""
    out = []
    for i in range(n):
        v = 0.1 + (i % 40) * 0.05
        out.append({"partition": "matched", "s0": v, "bdist": v, "gap": 5 + i % 20})
    return out


def test_faithful_proxy_yields_t1(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful())
    m = runner.run(study, substrate, None)
    assert m["boxes"] == {"B1": True, "B2": True, "B3": True}
    assert m["terminal"] == "T1_PROXY_FAITHFUL"


def test_unmatched_partitions_never_enter_the_agreement_denominator(
    tmp_path: Path,
) -> None:
    """cohort_gap / unemitted are limits on extrapolation, not missing values."""
    runner = _load_runner()
    base = _faithful()
    clean = runner.run(*_write_study(tmp_path / "a", base), None)

    # Add wildly disagreeing non-matched events. F1/F2/F3 must not move at all.
    polluted_events = base + [
        {"partition": "cohort_gap", "s0": 0.0, "bdist": 99.0, "gap": 3},
        {"partition": "cohort_gap", "s0": 0.0, "bdist": 99.0, "gap": 3},
        {"partition": "unemitted", "s0": 0.0, "bdist": 99.0, "gap": 3},
    ]
    polluted = runner.run(*_write_study(tmp_path / "b", polluted_events), None)

    assert polluted["F1_decision_agreement"] == clean["F1_decision_agreement"]
    assert polluted["F3_spearman_rho"] == clean["F3_spearman_rho"]
    assert (
        polluted["F2_numeric_error"]["absdelta_q95"]
        == clean["F2_numeric_error"]["absdelta_q95"]
    )
    assert polluted["validity"]["V5_matched_n"] == clean["validity"]["V5_matched_n"]
    assert polluted["partition"]["cohort_gap"] == 2
    assert polluted["partition"]["unemitted"] == 1


def test_decision_boundary_is_inclusive(tmp_path: Path) -> None:
    """Production accepts on `<= 0.4`. A strict `<` would flip these events."""
    runner = _load_runner()
    # Both sides sit exactly on the threshold: they must agree (both accept).
    events = _matched(1200, s0=0.4, bdist=0.4)
    m = runner.run(*_write_study(tmp_path, events), None)
    assert m["F1_confusion"]["both_accept"] == 1200
    assert m["F1_confusion"]["both_reject"] == 0
    assert m["F1_decision_agreement"] == 1.0


def test_boundary_disagreement_is_not_netted(tmp_path: Path) -> None:
    """proxy-accept-only and runtime-accept-only are different failures."""
    runner = _load_runner()
    events = (
        _matched(1000, s0=0.1, bdist=0.1)
        + _matched(30, s0=0.3, bdist=0.5)  # proxy accepts, runtime rejects
        + _matched(20, s0=0.5, bdist=0.3)  # runtime accepts, proxy rejects
    )
    m = runner.run(*_write_study(tmp_path, events), None)
    c = m["F1_confusion"]
    assert c["proxy_accept_only"] == 30
    assert c["runtime_accept_only"] == 20
    # They must not cancel: agreement counts both as disagreements.
    assert m["F1_decision_agreement"] == pytest.approx(1000 / 1050)


def test_quantile_method_is_type7_linear(tmp_path: Path) -> None:
    """A different quantile method would move B2 and could move the terminal."""
    runner = _load_runner()
    # 1200 events; |delta| = 0 for 1140, = 1.0 for 60 => q95 straddles the step.
    events = _matched(1140, s0=0.1, bdist=0.1) + _matched(60, s0=1.1, bdist=0.1)
    m = runner.run(*_write_study(tmp_path, events), None)
    import numpy as np

    expected = float(
        np.quantile(np.array([0.0] * 1140 + [1.0] * 60), 0.95, method="linear")
    )
    assert m["F2_numeric_error"]["absdelta_q95"] == pytest.approx(expected)
    assert m["conventions"]["quantile_method"] == "linear"


def test_rank_ties_use_average_ranks(tmp_path: Path) -> None:
    """Tie handling changes Spearman and therefore B3."""
    runner = _load_runner()
    from scipy import stats as sps

    # Heavy ties on both sides.
    events = _matched(600, s0=0.2, bdist=0.2) + _matched(600, s0=0.2, bdist=0.9)
    m = runner.run(*_write_study(tmp_path, events), None)
    s0 = [0.2] * 1200
    bd = [0.2] * 600 + [0.9] * 600
    expected = sps.spearmanr(s0, bd).statistic
    # All-constant s0 => rho is nan under average ranks; the point is that the
    # verifier reports scipy's average-rank result rather than inventing one.
    assert (m["F3_spearman_rho"] != m["F3_spearman_rho"]) == (expected != expected)
    assert m["conventions"]["tie_policy"] == "average_ranks"


def test_boxes_are_non_compensatory(tmp_path: Path) -> None:
    """A near-perfect rank correlation must not rescue a failed threshold box."""
    runner = _load_runner()
    # Monotone but offset: rank agreement is perfect, decisions flip constantly.
    events = []
    for i in range(1200):
        v = 0.01 * (i % 60)
        events.append(
            {"partition": "matched", "s0": v, "bdist": v + 0.35, "gap": 5 + i % 10}
        )
    m = runner.run(*_write_study(tmp_path, events), None)
    assert m["boxes"]["B3"] is True  # ordering preserved exactly
    assert m["boxes"]["B1"] is False or m["boxes"]["B2"] is False
    assert m["terminal"] == "T2_PROXY_UNFAITHFUL"
    assert m["box_bars"]["non_compensatory"] is True


def test_non_shadow_capture_is_unresolved(tmp_path: Path) -> None:
    """A committing bridge rewrites the ids we join on -- refuse, don't report."""
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful(), shadow=False)
    with pytest.raises(runner.ValidityFailure, match="not a shadow capture"):
        runner.run(study, substrate, None)


def test_overflowed_capture_is_unresolved(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful(), overflow=3)
    with pytest.raises(runner.ValidityFailure, match="overflow"):
        runner.run(study, substrate, None)


def test_partition_conservation_is_enforced(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful())
    # Corrupt one partition label to a value outside the frozen partition set.
    path = study / "capture.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["partition"] = "somethingelse"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with pytest.raises(runner.ValidityFailure, match="does not conserve"):
        runner.run(study, substrate, None)


def test_hash_mismatch_is_unresolved_not_a_finding(tmp_path: Path) -> None:
    """V6: a changed substrate is a validity failure, never a fidelity result."""
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful())
    with pytest.raises(runner.ValidityFailure, match="frozen inputs changed"):
        runner.run(study, substrate, {"pairs.csv": "0" * 64})


def test_matched_below_minimum_is_unresolved(tmp_path: Path) -> None:
    runner = _load_runner()
    study, substrate = _write_study(tmp_path, _faithful(n=50))
    with pytest.raises(runner.ValidityFailure, match="matched N"):
        runner.run(study, substrate, None)
