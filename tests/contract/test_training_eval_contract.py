"""Contract for the #421 shared eval contract: a run is pairable only when its identity says so.

Deliverable 2 decided *which* node pairs can be attributed; this contract
decides *whether two concrete runs are the same measurement* of such a
pair.  These tests pin the fail-closed predicates:

* the resolved-config mirror of ``mot17.py`` (preset merge + argv) matches
  the golden config fixture, so a moved argparse default or preset key is a
  different ``resolved_config_sha256``;
* ``runtime_identity`` refuses preset drift, argv/config drift, artifact sha
  drift, a missing artifact, a sequence outside the frozen key and a shell
  carrying a ``SACCADE_*`` hatch;
* ``validate-pair`` is ``paired`` only for two complete **formal** runs of
  the declared recipes, bound to this contract, differing in nothing but
  the pair's ``allowed_differences``; a missing ``runtime_identity``, a
  non-formal stage, a contract mismatch, an incomplete run or an extra
  difference is ``not_paired``;
* a repeat report needs one identity, full-sequence runs and ``min_runs``;
  a formal run needs a clean tree, the bench lease and a same-identity
  repeat report; the eager tracker policy never re-selects the preset;
* ``run_manifest`` v3 accepts ``runtime_identity`` on a production claim
  only, and the committed contract + doc are exactly what the tool derives
  (freshness, with frozen artifact shas carried over where the workspace
  lacks the files).
"""

# scope: detection
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.provenance import run_manifest as rm
from scripts.provenance import training_eval_contract as tec

REPO = Path(__file__).resolve().parents[2]
CONTRACT = tec.load_contract(tec.DEFAULT_CONTRACT)
DECLARED = CONTRACT["declared"]
S_RECIPE = "wg:mamba_whole_graph:s.t3t1_phase_b"
S42_RECIPE = "wg:mamba_whole_graph:s.implicit_s42"
C1 = "C1.s.explicit_vs_implicit_shared_gt1_s42"


# --------------------------------------------------------------------------- helpers


def _fake_environment(_declared: Any) -> dict[str, Any]:
    return {
        "hostname": "testhost",
        "platform": "linux-test",
        "python": "3.12",
        "packages": {"torch": "x"},
        "gpu": "GPU-T",
        "driver": "0",
        "cuda_visible_devices": None,
        "build_dir": "/b",
        "native_extensions": {"saccade_tracking_ext.so": "aa"},
        "trackeval": {"root": "third_party/TrackEval", "git_tree": "t", "dirty": False},
    }


def _synthetic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The committed contract with every artifact / dataset re-pointed at tmp files.

    Presets stay the real ones (copied), so ``resolve_config`` is the real
    merge; artifacts and images are small synthetic files whose sha256 the
    contract is re-frozen against.
    """
    root = tmp_path / "repo"
    (root / "configs" / "presets").mkdir(parents=True)
    for name in ("mamba_whole_graph", "mamba_whole_graph_m", "mamba_pyt_backbone"):
        (root / "configs" / "presets" / f"{name}.yaml").write_bytes(
            (REPO / "configs" / "presets" / f"{name}.yaml").read_bytes()
        )
    contract = json.loads(json.dumps(CONTRACT))
    contract["recipes"] = {
        k: v for k, v in contract["recipes"].items() if k in (S_RECIPE, S42_RECIPE)
    }
    contract["pairs"] = {C1: contract["pairs"][C1]}
    for recipe in contract["recipes"].values():
        for key, entry in recipe["artifacts"].items():
            if entry:
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{key}:{entry['path']}".encode())
                entry["sha256"] = tec.sha256_file(path)
                entry["size_bytes"] = path.stat().st_size
    ds = DECLARED["dataset"]
    seq_root = root / ds["data_root"] / ds["split"]
    for seq in ds["sequences"]:
        d = seq_root / seq
        (d / "gt").mkdir(parents=True)
        (d / "img1").mkdir()
        (d / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=8\nimHeight=4\nimExt=.jpg\n"
        )
        (d / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,1,1\n")
        for i in (1, 2):
            (d / "img1" / f"{i:06d}.jpg").write_bytes(f"{seq}-{i}".encode())
    monkeypatch.setattr(tec, "REPO_ROOT", root)
    key = tec.dataset_key(ds, ds["sequences"])
    contract["dataset_key"] = key
    contract["frozen"]["contract_sha256"] = tec.contract_sha256(contract)
    monkeypatch.setattr(tec, "environment_identity", _fake_environment)
    monkeypatch.setattr(rm, "_git_head", lambda: "deadbeef")
    monkeypatch.setattr(rm, "_git_dirty", lambda: False)
    for var in list(os.environ):
        if var.startswith("SACCADE_"):
            monkeypatch.delenv(var)
    return contract


def _identity(
    contract: dict[str, Any], recipe: str, stage: str = "formal"
) -> dict[str, Any]:
    return tec.runtime_identity(
        contract, recipe, stage=stage, sequences=DECLARED["dataset"]["sequences"]
    )


def _run_dir(
    tmp_path: Path,
    name: str,
    identity: dict[str, Any] | None,
    *,
    complete: bool = True,
    record_identity: str | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    run = tmp_path / "runs" / name
    rm.open_run(run, produced_by="eval", runtime_identity=identity)
    for seq in DECLARED["dataset"]["sequences"]:
        (run / f"{seq}.txt").write_text("1,1,0,0,1,1,1,-1,-1,-1\n")
    numeric = metrics or {k: 1.0 for k in tec.REQUIRED_METRIC_KEYS}
    record = {
        "schema": tec.RUN_RECORD_SCHEMA,
        "recipe_id": identity["pairing"]["recipe_id"] if identity else None,
        "stage": identity["stage"] if identity else "formal",
        "identity_sha256": record_identity
        or (identity["identity_sha256"] if identity else "x"),
        "complete": complete,
        "incomplete_reasons": [] if complete else ["exit code 1"],
        "metrics": {"raw": {}, "numeric": numeric},
    }
    (run / tec.RUN_RECORD_FILENAME).write_text(json.dumps(record))
    return run


# --------------------------------------------------------------------------- config resolution


def test_resolved_config_mirrors_the_golden_preset_snapshot() -> None:
    golden = json.loads(
        (REPO / "tests/fixtures/golden_config_mamba_whole_graph.json").read_text()
    )
    resolved = tec.resolve_config(["--preset", "mamba_whole_graph"])
    for key, value in golden["config"].items():
        assert resolved.get(key) == value, key


def test_recipe_argv_may_not_reroute_config_through_unpinned_files() -> None:
    for flag in (
        "--config",
        "--module-motion",
        "--output",
        "--max-frames",
        "--double-buffer",
    ):
        with pytest.raises(tec.ContractError, match="may not carry"):
            tec.check_recipe_argv(["--preset", "mamba_whole_graph", flag, "x"])


def test_eager_tracker_policy_never_reselects_the_preset() -> None:
    argv, overrides = tec.eager_tracker_policy_argv(DECLARED)
    assert "--preset" not in argv and "preset" not in overrides
    assert overrides["private_continuation_enabled"] is True
    # every override is expressible on the CLI and round-trips
    resolved = tec.resolve_config(
        ["--preset", DECLARED["tracker_policy"]["eager_preset"], *argv]
    )
    for key, value in overrides.items():
        assert resolved[key] == value, key


# --------------------------------------------------------------------------- identity drift


def test_identity_binds_cleanly_on_the_frozen_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    identity = _identity(contract, S_RECIPE)
    assert identity["schema"] == tec.IDENTITY_SCHEMA
    assert identity["pairing"]["dataset"]["subset"] is False
    assert (
        identity["pairing"]["artifacts"]["mamba_ckpt"]
        == contract["recipes"][S_RECIPE]["artifacts"]["mamba_ckpt"]["sha256"]
    )
    smoke = tec.runtime_identity(
        contract,
        S_RECIPE,
        stage="smoke",
        sequences=DECLARED["dataset"]["smoke_sequences"],
    )
    assert smoke["pairing"]["dataset"]["subset"] is True
    assert smoke["identity_sha256"] != identity["identity_sha256"]


def test_preset_drift_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    preset = tec.REPO_ROOT / contract["recipes"][S_RECIPE]["preset_path"]
    preset.write_text(preset.read_text() + "\n# drift\n")
    with pytest.raises(tec.ContractError, match="preset .* drifted"):
        _identity(contract, S_RECIPE)


def test_config_drift_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    contract["recipes"][S_RECIPE]["argv"] = [
        *contract["recipes"][S_RECIPE]["argv"],
        "--match-thresh",
        "0.51",
    ]
    with pytest.raises(tec.ContractError, match="resolved config drifted"):
        _identity(contract, S_RECIPE)


def test_artifact_sha_drift_and_absence_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    ckpt = (
        tec.REPO_ROOT / contract["recipes"][S_RECIPE]["artifacts"]["mamba_ckpt"]["path"]
    )
    ckpt.write_bytes(b"retrained")
    with pytest.raises(tec.ContractError, match="mamba_ckpt=.* drifted"):
        _identity(contract, S_RECIPE)
    ckpt.unlink()
    with pytest.raises(tec.ContractError, match="mamba_ckpt=.* is missing"):
        _identity(contract, S_RECIPE)


def test_dataset_drift_and_unknown_sequence_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    ds = DECLARED["dataset"]
    frame = (
        tec.REPO_ROOT
        / ds["data_root"]
        / ds["split"]
        / ds["sequences"][0]
        / "img1"
        / "000001.jpg"
    )
    frame.write_bytes(b"re-encoded")
    with pytest.raises(tec.ContractError, match="dataset drifted"):
        _identity(contract, S_RECIPE)
    with pytest.raises(tec.ContractError, match="not in the frozen dataset key"):
        tec.runtime_identity(
            contract, S_RECIPE, stage="formal", sequences=["MOT17-99-SDP"]
        )


def test_stray_saccade_hatch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    monkeypatch.setenv("SACCADE_STABILITY_W", "0.5")
    with pytest.raises(tec.ContractError, match="SACCADE_STABILITY_W"):
        _identity(contract, S_RECIPE)


# --------------------------------------------------------------------------- pairing


def test_treatment_only_difference_is_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    lhs = _run_dir(tmp_path, "lhs", _identity(contract, S42_RECIPE))
    rhs = _run_dir(tmp_path, "rhs", _identity(contract, S_RECIPE))
    verdict = tec.validate_pair(contract, C1, lhs, rhs)
    assert verdict["verdict"] == "paired", verdict["reasons"]
    assert set(verdict["observed_differences"]) == {
        "config.mamba_ckpt",
        "artifacts.mamba_ckpt",
    }
    assert verdict["variance_axis"] == "treatment"


def test_difference_outside_the_treatment_is_not_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    lhs_id = _identity(contract, S42_RECIPE)
    rhs_id = _identity(contract, S_RECIPE)
    # the same recipe run on a different environment / commit / tracker key
    rhs_id["resolved_config"]["match_thresh"] = 0.51
    lhs = _run_dir(tmp_path, "lhs", lhs_id)
    rhs = _run_dir(tmp_path, "rhs", rhs_id)
    verdict = tec.validate_pair(contract, C1, lhs, rhs)
    assert verdict["verdict"] == "not_paired"
    assert any("config.match_thresh" in r for r in verdict["reasons"])

    rhs_id = _identity(contract, S_RECIPE)
    rhs_id["pairing"]["environment"]["driver"] = "other"
    rhs_id["pairing"]["commit"] = "cafebabe"
    rhs2 = _run_dir(tmp_path, "rhs2", rhs_id)
    verdict = tec.validate_pair(contract, C1, lhs, rhs2)
    assert verdict["verdict"] == "not_paired"
    assert any("environment.driver" in r and "commit" in r for r in verdict["reasons"])


def test_missing_identity_wrong_stage_contract_or_incomplete_is_not_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    lhs = _run_dir(tmp_path, "lhs", _identity(contract, S42_RECIPE))

    no_identity = _run_dir(tmp_path, "no_id", None)
    v = tec.validate_pair(contract, C1, lhs, no_identity)
    assert v["verdict"] == "not_paired" and any(
        "no runtime_identity" in r for r in v["reasons"]
    )

    repeat = _run_dir(tmp_path, "repeat", _identity(contract, S_RECIPE, stage="repeat"))
    v = tec.validate_pair(contract, C1, lhs, repeat)
    assert v["verdict"] == "not_paired" and any(
        "not pairable" in r for r in v["reasons"]
    )

    foreign = _identity(contract, S_RECIPE)
    foreign["pairing"]["contract_sha256"] = "0" * 64
    v = tec.validate_pair(contract, C1, lhs, _run_dir(tmp_path, "foreign", foreign))
    assert v["verdict"] == "not_paired" and any(
        "different contract" in r for r in v["reasons"]
    )

    incomplete = _run_dir(
        tmp_path, "incomplete", _identity(contract, S_RECIPE), complete=False
    )
    v = tec.validate_pair(contract, C1, lhs, incomplete)
    assert v["verdict"] == "not_paired" and any("incomplete" in r for r in v["reasons"])

    swapped = tec.validate_pair(
        contract, C1, _run_dir(tmp_path, "rhs", _identity(contract, S_RECIPE)), lhs
    )
    assert swapped["verdict"] == "not_paired" and any(
        "recipe" in r for r in swapped["reasons"]
    )

    assert tec.validate_pair(contract, "Z9.nope", lhs, lhs)["verdict"] == "not_paired"


def test_reconstructed_manifest_is_never_pairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _synthetic_contract(tmp_path, monkeypatch)
    run = tmp_path / "runs" / "recon"
    run.mkdir(parents=True)
    (run / "MOT17-02-SDP.txt").write_text("x")
    rm.attach_reconstructed_manifest(
        run,
        rm.build_reconstructed_manifest("recon", commit="c", backfill_sources=["log"]),
    )
    with pytest.raises(tec.ContractError, match="not a production claim"):
        tec.run_identity(run)


# --------------------------------------------------------------------------- procedure


def test_repeat_report_requires_one_full_identity_and_min_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    ident = _identity(contract, S_RECIPE, stage="repeat")
    dirs = [_run_dir(tmp_path, f"r{i}", ident) for i in range(3)]
    report = tec.repeat_report(dirs, contract)
    assert (
        report["complete"]
        and report["n_runs"] == 3
        and report["byte_identity"]["all_identical"]
    )
    assert "deterministic" not in json.dumps(report).lower().replace(
        "never 'deterministic'", ""
    )
    assert report["variance_axis"] == "runtime_repeat"

    short = tec.repeat_report(dirs[:2], contract)
    assert not short["complete"] and any("min_runs" in r for r in short["reasons"])

    other = _run_dir(tmp_path, "other", _identity(contract, S42_RECIPE, stage="repeat"))
    mixed = tec.repeat_report([*dirs, other], contract)
    assert not mixed["complete"] and any(
        "different identities" in r for r in mixed["reasons"]
    )

    smoke = tec.runtime_identity(
        contract,
        S_RECIPE,
        stage="smoke",
        sequences=DECLARED["dataset"]["smoke_sequences"],
    )
    subset = [_run_dir(tmp_path, f"s{i}", smoke) for i in range(3)]
    sub_report = tec.repeat_report(subset, contract)
    assert not sub_report["complete"] and any(
        "subset" in r for r in sub_report["reasons"]
    )

    formal = [
        _run_dir(tmp_path, f"f{i}", _identity(contract, S_RECIPE)) for i in range(3)
    ]
    formal_report = tec.repeat_report(formal, contract)
    assert not formal_report["complete"]
    assert any("repeat-stage runs only" in r for r in formal_report["reasons"])


def test_formal_preflight_needs_clean_tree_lease_and_same_identity_repeat_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _synthetic_contract(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tec.REPO_ROOT))
    root = tmp_path / "results"
    monkeypatch.setattr(rm, "_git_dirty", lambda: True)
    with pytest.raises(tec.ContractError, match="clean working tree"):
        tec.preflight(
            contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
        )
    monkeypatch.setattr(rm, "_git_dirty", lambda: False)
    with pytest.raises(tec.ContractError, match="requires the 'machine-bench' lease"):
        tec.preflight(contract, S_RECIPE, stage="formal", root=root, lease="gpu0")
    with pytest.raises(tec.ContractError, match="repeat report"):
        tec.preflight(
            contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
        )
    # a same-identity repeat report unlocks it; a foreign one does not
    ident = _identity(contract, S_RECIPE, stage="repeat")
    dirs = [_run_dir(tmp_path, f"r{i}", ident) for i in range(3)]
    report = tec.repeat_report(dirs, contract)
    recipe_root = root / tec._slug(S_RECIPE)
    recipe_root.mkdir(parents=True)
    foreign = dict(report, identity_sha256="f" * 64)
    (recipe_root / "repeat-0-report.json").write_text(json.dumps(foreign))
    with pytest.raises(tec.ContractError, match="repeat report"):
        tec.preflight(
            contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
        )
    (recipe_root / "repeat-1-report.json").write_text(json.dumps(report))
    pre = tec.preflight(
        contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
    )
    assert pre["identity"]["identity_sha256"] == ident["identity_sha256"]
    # smoke never needs any of it, and runs the smoke subset
    monkeypatch.setattr(rm, "_git_dirty", lambda: True)
    smoke = tec.preflight(contract, S_RECIPE, stage="smoke", root=root, lease="gpu0")
    assert smoke["sequences"] == DECLARED["dataset"]["smoke_sequences"]


def test_dirty_repeat_is_refused_and_cannot_unlock_a_clean_formal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat on edited sources must not become the same-identity evidence a formal run needs.

    Two guards, either of which suffices: the repeat preflight refuses a dirty
    tree outright, and ``dirty`` is bound into the pairing identity so a
    report assembled from dirty runs carries a different identity_sha256
    from the clean formal run at the same HEAD.
    """
    contract = _synthetic_contract(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tec.REPO_ROOT))
    root = tmp_path / "results"

    monkeypatch.setattr(rm, "_git_dirty", lambda: True)
    with pytest.raises(
        tec.ContractError, match="repeat run requires a clean working tree"
    ):
        tec.preflight(contract, S_RECIPE, stage="repeat", root=root, lease="gpu0")
    # smoke may still run dirty
    tec.preflight(contract, S_RECIPE, stage="smoke", root=root, lease="gpu0")

    dirty_ident = _identity(contract, S_RECIPE, stage="repeat")
    assert dirty_ident["pairing"]["dirty"] is True
    dirty_runs = [_run_dir(tmp_path, f"d{i}", dirty_ident) for i in range(3)]
    dirty_report = tec.repeat_report(dirty_runs, contract)
    recipe_root = root / tec._slug(S_RECIPE)
    recipe_root.mkdir(parents=True)
    (recipe_root / "repeat-dirty-report.json").write_text(json.dumps(dirty_report))

    monkeypatch.setattr(rm, "_git_dirty", lambda: False)
    clean_ident = _identity(contract, S_RECIPE, stage="repeat")
    assert clean_ident["identity_sha256"] != dirty_ident["identity_sha256"]
    with pytest.raises(tec.ContractError, match="repeat report"):
        tec.preflight(
            contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
        )

    clean_runs = [_run_dir(tmp_path, f"c{i}", clean_ident) for i in range(3)]
    (recipe_root / "repeat-clean-report.json").write_text(
        json.dumps(tec.repeat_report(clean_runs, contract))
    )
    pre = tec.preflight(
        contract, S_RECIPE, stage="formal", root=root, lease="machine-bench"
    )
    assert pre["identity"]["identity_sha256"] == clean_ident["identity_sha256"]


def test_metrics_parse_keeps_digits_and_flags_missing_hota() -> None:
    stdout = (
        "noise\n=== OVERALL METRICS ===\n  IDF1: 78.3%\n  MOTA: 80.1%\n  HOTA: 66.0%\n  DetA: 70.0%\n"
        "  AssA: 62.0%\n  IDs: 412\n  FP: 10\n  FN: 20\n  Rcll: 90.0%\n  Prcn: 95.0%\n[mlflow] skip\n"
    )
    parsed = tec.parse_overall_metrics(stdout)
    assert parsed["numeric"]["IDF1"] == 78.3 and parsed["numeric"]["IDs"] == 412
    assert all(k in parsed["numeric"] for k in tec.REQUIRED_METRIC_KEYS)
    without_hota = tec.parse_overall_metrics(stdout.replace("  HOTA: 66.0%\n", ""))
    assert "HOTA" not in without_hota["numeric"]
    assert tec.parse_overall_metrics("no block") == {}


# --------------------------------------------------------------------------- manifest v3


def test_manifest_v3_carries_runtime_identity_on_production_only(
    tmp_path: Path,
) -> None:
    identity = {"schema": tec.IDENTITY_SCHEMA, "pairing": {"x": 1}}
    rm.open_run(tmp_path / "r", produced_by="eval", runtime_identity=identity)
    payload = rm.read_manifest(tmp_path / "r")
    assert payload["schema_version"] == 3 and payload["runtime_identity"] == identity

    v2 = dict(payload, schema_version=2)
    with pytest.raises(rm.ManifestError, match="may not carry runtime_identity"):
        rm.validate_manifest(v2)

    recon = rm.build_reconstructed_manifest("r", commit="c", backfill_sources=["s"])
    recon["runtime_identity"] = identity
    with pytest.raises(rm.ManifestError, match="production-only"):
        rm.validate_manifest(recon)

    for bad in ({}, {"schema": ""}, {"pairing": {}}, "x"):
        with pytest.raises(rm.ManifestError, match="runtime_identity"):
            rm.validate_manifest(dict(payload, runtime_identity=bad))


# --------------------------------------------------------------------------- committed contract


def test_committed_contract_is_fresh() -> None:
    """The JSON and the doc are exactly what the tool derives from the committed matrix.

    Artifact shas are carried over from the frozen contract where this
    workspace lacks the files (CI), so preset / argparse / CLI / matrix drift
    is still caught there; the dataset key is only re-hashed with
    ``--verify-dataset``.
    """
    reasons = tec.check(
        tec.DEFAULT_CONTRACT,
        tec.DEFAULT_MATRIX,
        tec.DEFAULT_INVENTORY,
        tec.DEFAULT_MD_OUT,
    )
    assert reasons == []


def test_committed_contract_prepares_the_required_rows_consistently() -> None:
    pairs, recipes = CONTRACT["pairs"], CONTRACT["recipes"]
    for required in ("C1", "C2", "C3", "E1", "E2", "E3", "E4", "B1", "B2", "B3", "B4"):
        assert any(cid.startswith(required + ".") for cid in pairs), required
    matrix = tec.load_matrix(REPO / tec.DEFAULT_MATRIX)
    rows = {r["comparison_id"]: r for r in matrix["comparisons"]}
    for cid, pair in pairs.items():
        assert rows[cid]["classification"] == pair["classification"]
        lhs, rhs = recipes[pair["lhs_recipe"]], recipes[pair["rhs_recipe"]]
        assert lhs["execution_profile"] == rhs["execution_profile"], cid
        assert set(pair["frozen_differences"]) <= set(pair["allowed_differences"]), cid
        if pair["classification"] == "controlled":
            assert set(pair["allowed_differences"]) == tec._axis_paths(
                pair["treatment_axes"], DECLARED
            )
        if rows[cid]["design"] == "seed_replicate":
            assert pair["variance_axis"] == "training_seed"
    for rid, recipe in recipes.items():
        argv = recipe["argv"]
        assert (
            "--detector" in argv
            and "--sequences" not in argv
            and "--output" not in argv
        )
        if recipe["execution_profile"] == "eager_pytorch":
            assert "--no-compile" in argv and "--private-continuation" in argv, rid
            if recipe["node_kind"] == "mamba_ckpt":
                assert "--no-temporal" in argv and "--no-mamba-trt" in argv, rid
            else:
                assert "--teacher-head-ckpt" in argv and "--mamba-ckpt" in argv, rid
        else:
            assert "--no-compile" not in argv, rid
        head = recipe["binding_facts"]["head_source"]
        assert recipe["binding_facts"]["effective_T"] == 1
        if "ckpt-head" in rid or recipe["execution_profile"] == "eager_pytorch":
            assert head["checkpoint_head_deployed"] is True, rid
    # the m preset-as-is row measures the fixed engine head, and says so
    assert (
        recipes["wg:mamba_whole_graph_m:m.t3t1_phase_b"]["binding_facts"][
            "head_source"
        ]["kind"]
        == "trt_engine"
    )
    ds = DECLARED["dataset"]
    assert (
        set(ds["smoke_sequences"]) < set(ds["sequences"]) and ds["max_frames"] is None
    )
    assert DECLARED["procedure"]["repeat"]["requires_clean_tree"] is True
    assert DECLARED["procedure"]["formal"]["requires_clean_tree"] is True
    assert set(CONTRACT["dataset_key"]["sequences"]) == set(ds["sequences"])
