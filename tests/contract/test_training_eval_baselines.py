"""Contract for #421 deliverable 4: a baseline row exists only behind a ``paired`` verdict.

Deliverable 3 froze how a measurement is taken and inventoried the campaign;
this contract pins how the inventory becomes baseline rows:

* the campaign inventory is the only input naming runs and the contract the
  only input naming pairs — a campaign under another contract sha, an
  incomplete campaign, a pair set that differs from
  ``declared.prepared_comparisons`` or a non-prepared pair id refuses;
* every lhs × rhs formal combination goes through ``validate-pair`` plus the
  per-side gate (clean tree, bench lease, campaign commit, listed formal run,
  campaign identity); one ``not_paired`` combination and there is no row;
* a row's deltas are read against the print precision and both sides'
  runtime-repeat observed ranges, never upgraded to an effect; the pair's
  ``remaining_confounds`` are copied from the contract and must agree with
  the matrix row — a pair without them, or a report row that lost them, is
  refused by the builder / the audit;
* plain-GT2 ↔ T3→T1 readings are carried as historical only, with the
  matrix confounds (``warmup_epochs`` 5→3, seed for s) and no row;
* the committed report audits clean against the committed contract, campaign
  and matrix, and the committed doc is exactly its rendering.
"""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.provenance import run_manifest as rm
from scripts.provenance import training_eval_contract as tec
from tests.contract.test_training_eval_contract import (
    C1,
    CONTRACT,
    DECLARED,
    S42_RECIPE,
    S_RECIPE,
    _identity,
    _run_dir,
    _synthetic_contract,
)

REPO = Path(__file__).resolve().parents[2]
MATRIX = tec.load_matrix(REPO / tec.DEFAULT_MATRIX)
LHS_METRICS = {
    "IDF1": 76.7,
    "MOTA": 77.8,
    "HOTA": 68.7,
    "DetA": 69.9,
    "AssA": 67.7,
    "IDs": 512.0,
    "FP": 4246.0,
    "FN": 20161.0,
}
RHS_METRICS = {
    "IDF1": 78.3,
    "MOTA": 77.9,
    "HOTA": 70.0,
    "DetA": 69.9,
    "AssA": 69.9,
    "IDs": 429.0,
    "FP": 3471.0,
    "FN": 20940.0,
}


def _d4_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The synthetic two-recipe / one-pair contract with C1 as the only prepared pair."""
    contract = _synthetic_contract(tmp_path, monkeypatch)
    contract["declared"]["prepared_comparisons"] = [C1]
    contract["frozen"]["contract_sha256"] = tec.contract_sha256(contract)
    monkeypatch.setattr(tec, "HISTORICAL_COMPARISONS", {})
    return contract


def _campaign(
    tmp_path: Path,
    contract: dict[str, Any],
    *,
    rhs_repeat_metrics: list[dict[str, float]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """A run root with n=3 repeat runs + report and 3 formal runs per recipe, with
    distinct metrics per side, and its campaign inventory."""
    root = tmp_path / "results"
    for recipe, metrics in ((S42_RECIPE, LHS_METRICS), (S_RECIPE, RHS_METRICS)):
        recipe_root = root / tec._slug(recipe)
        recipe_root.mkdir(parents=True)
        rep = _identity(contract, recipe, stage="repeat")
        per_run = (
            rhs_repeat_metrics
            if recipe == S_RECIPE and rhs_repeat_metrics
            else [metrics] * 3
        )
        dirs = [
            _run_dir(
                tmp_path,
                f"repeat-0-r0{i}",
                rep,
                parent=recipe_root,
                metrics=per_run[i - 1],
            )
            for i in range(1, 4)
        ]
        (recipe_root / "repeat-0-report.json").write_text(
            json.dumps(tec.repeat_report(dirs, contract))
        )
        formal = _identity(contract, recipe)
        for i in range(1, 4):
            _run_dir(
                tmp_path, f"formal-1-r0{i}", formal, parent=recipe_root, metrics=metrics
            )
    campaign = tec.campaign_inventory(contract, root)
    assert campaign["complete"], campaign["reasons"]
    return root, campaign


# --------------------------------------------------------------------------- rows


def test_baseline_row_forms_from_paired_verdicts_and_reads_deltas_against_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _d4_contract(tmp_path, monkeypatch)
    _, campaign = _campaign(
        tmp_path,
        contract,
        rhs_repeat_metrics=[
            RHS_METRICS,
            {**RHS_METRICS, "MOTA": 77.7},
            {**RHS_METRICS, "MOTA": 78.1},
        ],
    )
    report = tec.baseline_report(contract, campaign, MATRIX)
    assert report["schema"] == tec.BASELINES_SCHEMA
    assert report["complete"] and report["baseline_rows"] == [C1]
    assert report["contract_sha256"] == contract["frozen"]["contract_sha256"]
    assert report["campaign_sha256"] == tec.sha256_json(campaign)
    pair = report["pairs"][C1]
    assert (
        pair["verdict"] == "paired" and pair["n_paired"] == pair["n_combinations"] == 9
    )
    assert [c["name"] for c in pair["remaining_confounds"]] == list(
        contract["pairs"][C1]["remaining_confounds"]
    )
    assert pair["remaining_confounds"][0]["severity"] == "common_mode"
    row = pair["baseline_row"]
    assert row["baseline_side"] == "lhs" and row["baseline_recipe"] == S42_RECIPE
    assert row["treatment_recipe"] == S_RECIPE
    assert set(tec.REQUIRED_METRIC_KEYS) <= set(row["metrics"])
    hota = row["metrics"]["HOTA"]
    assert hota["delta_rhs_minus_lhs"] == pytest.approx(1.3)
    assert hota["print_precision"] == 0.1
    assert hota["lhs_observed_range"] == 0.0 and hota["rhs_observed_range"] == 0.0
    assert hota["reading"] == "above_runtime_repeat_range"
    assert row["metrics"]["DetA"]["reading"] == "no_observed_difference"
    # MOTA moved by 0.1 but the rhs repeat set spans 0.4: within range, not resolved
    mota = row["metrics"]["MOTA"]
    assert mota["rhs_observed_range"] == pytest.approx(0.4)
    assert mota["reading"] == "within_runtime_repeat_range"
    ids = row["metrics"]["IDs"]
    assert ids["print_precision"] == 1.0 and ids["delta_rhs_minus_lhs"] == -83.0
    fps = row["throughput_fps"]
    assert fps["reading"] == "no_observed_difference" and fps["print_precision"] == 0.01
    assert fps["profile"] == "whole_graph_serial"
    assert row["repeat_n"] == {"lhs": 3, "rhs": 3}
    assert set(row["observed_differences"]) == {
        "artifacts.mamba_ckpt",
        "config.mamba_ckpt",
    }
    # readings are computed, not asserted: a lone C1 has no seed reference
    readings = report["readings"]
    assert readings["explicit_vs_implicit_shared_gt1"]["complete"] is False
    assert readings["seed_replicates"]["complete"] is False
    assert report["historical"] == []
    text = json.dumps(report).lower()
    assert "effect" not in text.replace("effect_claim", "").replace(
        "causal effect", ""
    ).replace("training effect", "").replace("effect size", "")
    assert tec.audit_baseline_report(report, contract, campaign, MATRIX) == []
    md = tec.render_baselines_markdown(report)
    assert "<!-- doc-status: active -->" in md
    assert f"`{C1}`" in md and "deployed_backbone_teacher_mismatch" in md
    assert "68.7→70.0 (+1.3 >range)" in md
    assert "(+0.1 ≤range)" in md


def test_no_row_without_a_formal_run_or_with_a_not_paired_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _d4_contract(tmp_path, monkeypatch)
    root, campaign = _campaign(tmp_path, contract)

    missing = json.loads(json.dumps(campaign))
    missing["pairs"][C1]["rhs_formal_runs"] = []
    report = tec.baseline_report(contract, missing, MATRIX)
    assert not report["complete"] and report["baseline_rows"] == []
    pair = report["pairs"][C1]
    assert pair["verdict"] == "not_paired" and pair["baseline_row"] is None
    assert any("no formal run" in r for r in pair["reasons"])
    assert pair["remaining_confounds"], "confounds stay attached even without a row"

    # one formal run whose record identity does not match its manifest: that
    # combination is not_paired and the whole pair forms no row
    recipe_root = root / tec._slug(S_RECIPE)
    bad = _run_dir(
        tmp_path,
        "formal-1-r04",
        _identity(contract, S_RECIPE),
        parent=recipe_root,
        record_identity="0" * 64,
        metrics=RHS_METRICS,
    )
    tampered = json.loads(json.dumps(campaign))
    tampered["pairs"][C1]["rhs_formal_runs"].append(str(bad))
    report = tec.baseline_report(contract, tampered, MATRIX)
    pair = report["pairs"][C1]
    assert pair["verdict"] == "not_paired" and pair["baseline_row"] is None
    assert pair["n_combinations"] == 12 and pair["n_paired"] == 9
    assert any("not_paired" in r for r in pair["reasons"])
    bad_combo = [
        c
        for c in pair["combinations"]
        if c["lhs_run"] != c["rhs_run"] and str(bad) == c["rhs_run"]
    ]
    assert bad_combo and any("record identity" in r for r in bad_combo[0]["reasons"])


def test_side_gate_refuses_dirty_tree_wrong_lease_foreign_commit_and_unlisted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``validate-pair`` compares the two sides to each other, so two dirty runs
    agree in ``dirty``; the deliverable-4 gate refuses each side on its own."""
    contract = _d4_contract(tmp_path, monkeypatch)
    root, campaign = _campaign(tmp_path, contract)
    lhs_run = campaign["pairs"][C1]["lhs_formal_runs"][0]

    monkeypatch.setattr(rm, "_git_dirty", lambda: True)
    dirty_lhs = _identity(contract, S42_RECIPE)
    dirty_rhs = _identity(contract, S_RECIPE)
    monkeypatch.setattr(rm, "_git_dirty", lambda: False)
    assert dirty_lhs["pairing"]["dirty"] is True
    other = tmp_path / "elsewhere"
    lhs_d = _run_dir(
        tmp_path, "formal-9-r01", dirty_lhs, parent=other, metrics=LHS_METRICS
    )
    rhs_d = _run_dir(
        tmp_path, "formal-9-r02", dirty_rhs, parent=other, metrics=RHS_METRICS
    )
    assert tec.validate_pair(contract, C1, lhs_d, rhs_d)["verdict"] == "paired"
    verdict = tec.d4_pair_verdict(contract, campaign, C1, str(lhs_d), str(rhs_d))
    assert verdict["verdict"] == "not_paired"
    assert any("lhs: dirty tree" in r for r in verdict["reasons"])
    assert any("rhs: dirty tree" in r for r in verdict["reasons"])
    assert any(
        "not a formal run the campaign inventory lists" in r for r in verdict["reasons"]
    )

    identity = tec.run_identity(Path(lhs_run))
    record = tec.run_record(Path(lhs_run))
    assert tec.d4_side_gate("lhs", identity, record, campaign, contract, lhs_run) == []
    wrong_lease = {**record, "lease": "gpu0"}
    assert any(
        "lease 'gpu0'" in r
        for r in tec.d4_side_gate(
            "lhs", identity, wrong_lease, campaign, contract, lhs_run
        )
    )
    foreign = json.loads(json.dumps(identity))
    foreign["pairing"]["commit"] = "cafebabe"
    assert any(
        "campaign commit" in r
        for r in tec.d4_side_gate("lhs", foreign, record, campaign, contract, lhs_run)
    )
    drifted = json.loads(json.dumps(identity))
    drifted["identity_sha256"] = "f" * 64
    assert any(
        "identity differs" in r
        for r in tec.d4_side_gate("lhs", drifted, record, campaign, contract, lhs_run)
    )


# --------------------------------------------------------------------------- inputs


def test_campaign_under_another_contract_or_incomplete_or_wrong_pair_set_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _d4_contract(tmp_path, monkeypatch)
    _, campaign = _campaign(tmp_path, contract)

    foreign = {**campaign, "contract_sha256": "0" * 64}
    with pytest.raises(tec.ContractError, match="different contract"):
        tec.baseline_report(contract, foreign, MATRIX)

    incomplete = {**campaign, "complete": False, "reasons": ["x"]}
    with pytest.raises(tec.ContractError, match="incomplete"):
        tec.baseline_report(contract, incomplete, MATRIX)

    extra = json.loads(json.dumps(campaign))
    extra["pairs"]["Z9.not_prepared"] = extra["pairs"][C1]
    with pytest.raises(tec.ContractError, match="differ from the prepared pairs"):
        tec.baseline_report(contract, extra, MATRIX)

    with pytest.raises(tec.ContractError, match="not a prepared pair"):
        tec.baseline_pair(contract, campaign, tec._row_by_id(MATRIX), "Z9.not_prepared")

    # the contract's own pair list must be the declared one
    drifted = json.loads(json.dumps(contract))
    drifted["declared"]["prepared_comparisons"] = [C1, "Z9.not_prepared"]
    with pytest.raises(tec.ContractError, match="prepared_comparisons"):
        tec.baseline_report(drifted, campaign, MATRIX)

    # a non-prepared id through validate-pair is not_paired by construction
    lhs, rhs = (
        campaign["pairs"][C1]["lhs_formal_runs"][0],
        campaign["pairs"][C1]["rhs_formal_runs"][0],
    )
    assert (
        tec.validate_pair(contract, "Z9.not_prepared", Path(lhs), Path(rhs))["verdict"]
        == "not_paired"
    )


def test_lost_or_disagreeing_confounds_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _d4_contract(tmp_path, monkeypatch)
    _, campaign = _campaign(tmp_path, contract)
    rows = tec._row_by_id(MATRIX)

    stripped = json.loads(json.dumps(contract))
    del stripped["pairs"][C1]["remaining_confounds"]
    with pytest.raises(tec.ContractError, match="no remaining_confounds"):
        tec._pair_confounds(stripped, rows, C1)

    emptied = json.loads(json.dumps(contract))
    emptied["pairs"][C1]["remaining_confounds"] = []
    with pytest.raises(tec.ContractError, match="!= matrix"):
        tec._pair_confounds(emptied, rows, C1)

    # a report whose row dropped the confound after the fact fails the audit
    report = tec.baseline_report(contract, campaign, MATRIX)
    assert tec.audit_baseline_report(report, contract, campaign, MATRIX) == []
    lost = json.loads(json.dumps(report))
    lost["pairs"][C1]["remaining_confounds"] = []
    assert any(
        "remaining_confounds differ" in r
        for r in tec.audit_baseline_report(lost, contract, campaign, MATRIX)
    )
    # ... as does a row that was hand-edited, a row without a paired verdict,
    # a swapped baseline side, or a report on another campaign
    edited = json.loads(json.dumps(report))
    edited["pairs"][C1]["baseline_row"]["metrics"]["HOTA"]["delta_rhs_minus_lhs"] = 9.9
    assert any(
        "does not recompute" in r
        for r in tec.audit_baseline_report(edited, contract, campaign, MATRIX)
    )
    unpaired = json.loads(json.dumps(report))
    unpaired["pairs"][C1]["verdict"] = "not_paired"
    assert any(
        "without a paired verdict" in r
        for r in tec.audit_baseline_report(unpaired, contract, campaign, MATRIX)
    )
    swapped = json.loads(json.dumps(report))
    swapped["pairs"][C1]["baseline_row"]["baseline_recipe"] = S_RECIPE
    assert any(
        "baseline side" in r
        for r in tec.audit_baseline_report(swapped, contract, campaign, MATRIX)
    )
    other = json.loads(json.dumps(campaign))
    other["generated_at"] = "2000-01-01T00:00:00+00:00"
    assert any(
        "different campaign" in r
        for r in tec.audit_baseline_report(report, contract, other, MATRIX)
    )


def test_delta_reading_never_upgrades_to_an_effect() -> None:
    r = tec.delta_reading(
        70.0, 70.0, precision=0.1, lhs_observed_range=0.0, rhs_observed_range=0.0
    )
    assert r["reading"] == "no_observed_difference" and r["delta_rhs_minus_lhs"] == 0.0
    r = tec.delta_reading(
        70.0, 70.3, precision=0.1, lhs_observed_range=0.0, rhs_observed_range=0.5
    )
    assert (
        r["reading"] == "within_runtime_repeat_range"
        and r["runtime_repeat_floor"] == 0.5
    )
    r = tec.delta_reading(
        70.0, 70.6, precision=0.1, lhs_observed_range=0.0, rhs_observed_range=0.5
    )
    assert r["reading"] == "above_runtime_repeat_range"
    r = tec.delta_reading(
        70.0, 70.04, precision=0.1, lhs_observed_range=None, rhs_observed_range=None
    )
    assert r["reading"] == "no_observed_difference"
    assert "effect" not in json.dumps(r)
    assert (
        tec.metric_print_precision("IDs") == 1.0
        and tec.metric_print_precision("HOTA") == 0.1
    )


# --------------------------------------------------------------------------- committed snapshot


def test_committed_baselines_report_audits_clean_and_matches_its_doc() -> None:
    """The committed deliverable-4 report is the report of the committed contract,
    campaign and matrix: 15/15 prepared pairs paired on every combination, every
    row carrying its contract confounds, every delta recomputing, the two
    plain-GT2 ↔ T3→T1 readings historical only, and the doc its exact rendering."""
    report = json.loads((REPO / tec.DEFAULT_BASELINES_OUT).read_text())
    campaign = tec.load_campaign(REPO / tec.DEFAULT_CAMPAIGN_OUT)
    assert tec.audit_baseline_report(report, CONTRACT, campaign, MATRIX) == []
    assert report["deliverable"] == 4 and report["complete"]
    assert (
        report["n_pairs"]
        == report["n_paired"]
        == len(DECLARED["prepared_comparisons"])
        == 15
    )
    assert list(report["pairs"]) == DECLARED["prepared_comparisons"]
    for cid, pair in report["pairs"].items():
        assert (
            pair["verdict"] == "paired"
            and pair["n_paired"] == pair["n_combinations"] == 9
        ), cid
        assert [c["name"] for c in pair["remaining_confounds"]] == CONTRACT["pairs"][
            cid
        ]["remaining_confounds"]
        row = pair["baseline_row"]
        assert row["baseline_recipe"] == CONTRACT["pairs"][cid]["lhs_recipe"]
        assert row["repeat_n"] == {"lhs": 6, "rhs": 6}
        assert row["repeat_distinct_outputs_max"] == {"lhs": 1, "rhs": 1}
        for key in tec.REQUIRED_METRIC_KEYS:
            assert row["metrics"][key]["lhs_observed_range"] == 0.0
            assert row["metrics"][key]["rhs_observed_range"] == 0.0
        if cid.startswith(("B1.", "B2.", "C", "D")):
            assert "deployed_backbone_teacher_mismatch" in [
                c["name"] for c in pair["remaining_confounds"]
            ], cid
    # the historical readings are exactly the two plain-GT2 <-> T3->T1 pairings, with their confounds
    hist = {h["comparison_id"]: h for h in report["historical"]}
    assert set(hist) == {
        "C6.s.plain_gt2_vs_t3t1_unpaired_original",
        "C9.m.plain_gt2_vs_t3t1",
    }
    for h in hist.values():
        assert (
            h["baseline_row"] is None and h["validate_pair"]["verdict"] == "not_paired"
        )
        assert h["schedule_confound_keys"] == ["warmup_epochs"]
        assert "training_schedule" in [c["name"] for c in h["training_confounds"]]
    assert "training_seed" in [
        c["name"]
        for c in hist["C6.s.plain_gt2_vs_t3t1_unpaired_original"]["training_confounds"]
    ]
    assert hist["C9.m.plain_gt2_vs_t3t1"]["training_seed_axis"] == "matched"
    # readings recompute from the rows and stay observational
    assert report["readings"] == tec.group_readings(report["pairs"])
    assert report["readings"]["explicit_vs_implicit_shared_gt1"]["complete"]
    assert report["readings"]["seed_replicates"]["complete"]
    md = (REPO / tec.DEFAULT_BASELINES_MD).read_text()
    assert md == tec.render_baselines_markdown(report)
    assert "not an effect" in md and "historical only" in md
    assert "warmup_epochs" in md
