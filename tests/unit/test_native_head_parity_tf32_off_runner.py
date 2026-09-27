"""The #465 PR-2R runner is PR-2's gate with only the TF32-off head swapped in.

``scripts/eval/diagnostics/native_head_parity_tf32_off.py`` implements
``docs/reference/native_runtime_head_parity_tf32_off_declaration.md``. That
declaration keeps PR-2's thresholds, tolerance policy, arms and terminal order
and changes only the head under test, so the runner is pinned here to (a) its
own declaration blob and frozen artifact identity, (b) PR-2's decision
constants and decision functions, unchanged, and (c) a command line with no
option that could change the study identity.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "scripts" / "eval" / "diagnostics"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIAG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("native_head_parity_tf32_off")
PR2 = _load("native_head_parity")


def test_declaration_blob_is_the_frozen_one():
    blob = subprocess.run(
        ["git", "hash-object", R.DECLARATION],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert blob == R.FROZEN_DECLARATION_BLOB  # an amendment must move this constant
    assert R.DECLARATION != PR2.DECLARATION


def test_frozen_artifact_identity_appears_in_the_declaration():
    text = (REPO / R.DECLARATION).read_text()
    assert R.HEAD_STEM.endswith("_notf32")
    for value in (
        R.HEAD_ENGINE,
        R.HEAD_LINEAGE,
        R.HEAD_LINEAGE_SHA256,
        R.HEAD_ONNX_SHA256,
        R.CKPT_SHA256,
        R.EXPECTED_PRECISION,
        f"--precision {R.EXPECTED_PRECISION} --check",
        ",".join(R.SEQUENCES),
        f"`--mamba-head-engine {R.HEAD_ENGINE}`",
    ):
        assert value in text, value
    assert R.EXPECTED_BUILDER_FLAGS == {"fp16": False, "tf32": False}
    assert R.ARMS["A_T"] == ["--mamba-head-engine", R.HEAD_ENGINE]


@pytest.mark.parametrize(
    "name",
    [
        "HEAD_ONNX_SHA256",
        "CKPT",
        "CKPT_SHA256",
        "BACKBONE",
        "PRESET_NAME",
        "SEQUENCES",
        "TOTAL_FRAMES",
        "LEASE",
        "SCORE_FLOOR",
        "L1_SCORE_MAX",
        "L1_BOX_MAX_PX",
        "V2_FRAMES",
        "L1_PAIRS",
        "L1_DECISION_PAIR",
        "RUN_ORDER",
        "L2_METRICS",
        "TOL_FLOOR",
        "TOL_CAP",
        "TERMINALS",
    ],
)
def test_decision_constants_equal_pr2(name):
    assert getattr(R, name) == getattr(PR2, name)


@pytest.mark.parametrize(
    "name",
    [
        "tolerance",
        "l1_verdict",
        "l2_verdict",
        "decide_terminal",
        "metrics_from_counts",
        "first_divergent_frame",
        "hist_quantile",
    ],
)
def test_decision_functions_equal_pr2(name):
    assert inspect.getsource(getattr(R, name)) == inspect.getsource(getattr(PR2, name))


def test_only_the_arm_engine_differs_from_pr2():
    assert R.ARMS.keys() == PR2.ARMS.keys()
    assert R.ARMS["A_C"] == PR2.ARMS["A_C"] == []
    assert R.ARMS["A_N"] == PR2.ARMS["A_N"] == ["--no-compile"]
    assert R.HEAD_ENGINE != PR2.HEAD_ENGINE


def test_cli_exposes_no_study_identity_option():
    seen: list[str] = []
    real = argparse.ArgumentParser.add_argument

    def spy(self, *names, **kwargs):
        seen.extend(n for n in names if n.startswith("--"))
        return real(self, *names, **kwargs)

    with mock.patch.object(argparse.ArgumentParser, "add_argument", spy):
        with mock.patch("sys.argv", ["x", "--help"]), pytest.raises(SystemExit):
            R.main()
    assert set(seen) - {"--help"} == {"--smoke-frames", "--_l1-worker", "--_sequences"}
