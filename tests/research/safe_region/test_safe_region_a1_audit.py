"""Mutation tests for the A1 read-only audit: defect sensitivity, not just PASS.

Each test copies the real conversion pack, tampers exactly one semantic fact,
and asserts the corresponding audit section flips to FAIL. Skipped when the
pack is not present (out/ is untracked, so CI skips; run locally / pre-push).
"""

# scope: eval
# function: contract
# lifecycle: quarantined
# lifecycle-note: safe-region A1 study CLOSED (A1_ACCEPTED_WITH_LIMITS);
#   DISPOSITION.md proposes T3 — generic packet checkers + the sealed pack supersede it.

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
PACK = REPO / "out/signal_study/m_b1_5_safe_region_asset_r1_20260710"
T0 = (
    REPO
    / "docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710"
)
Q45 = REPO / "docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710"

pytestmark = pytest.mark.skipif(
    not PACK.exists(), reason="conversion pack not present (out/ is untracked)"
)


def _audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_safe_region_a1_audit", REPO / "scripts/tools/run_safe_region_a1_audit.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def pack_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "pack"
    shutil.copytree(PACK, dst)
    return dst


def _statuses(pack: Path) -> dict[str, str]:
    mod = _audit_module()
    p = mod._load_pack(pack)
    results: list[dict[str, str]] = []
    mod.audit_s0(p, results)
    mod.audit_s1(p, T0, Q45, results)
    mod.audit_q1(p, results)
    mod.audit_n1(p, results)
    return {r["check_id"]: r["status"] for r in results}


def test_untampered_pack_passes(pack_copy: Path) -> None:
    st = _statuses(pack_copy)
    assert set(st.values()) == {"PASS"}, {k: v for k, v in st.items() if v != "PASS"}


def test_component_alias_tamper_fails_s1(pack_copy: Path) -> None:
    f = pack_copy / "region_assets.csv"
    df = pd.read_csv(f)
    df.loc[0, "t0_component_id_alias"] = "S::bogus::high_tail::comp99"
    df.to_csv(f, index=False)
    st = _statuses(pack_copy)
    assert st["S1.7"] == "FAIL"


def test_per_sequence_support_tamper_fails_s1(pack_copy: Path) -> None:
    f = pack_copy / "region_coordinate_membership.csv"
    df = pd.read_csv(f)
    seq = json.loads(df.loc[0, "per_sequence_neg_json"])
    key = sorted(seq)[0]
    seq[key] = int(seq[key]) + 1
    df.loc[0, "per_sequence_neg_json"] = json.dumps(seq)
    df.to_csv(f, index=False)
    st = _statuses(pack_copy)
    assert st["S1.8"] == "FAIL"


def test_forbidden_promotion_removal_fails_n1(pack_copy: Path) -> None:
    f = pack_copy / "region_claim_contract.json"
    contract = json.loads(f.read_text())
    contract["forbidden_promotions"].remove("A0_to_A1_self_accept")
    f.write_text(json.dumps(contract))
    st = _statuses(pack_copy)
    assert st["N1.promotions"] == "FAIL"
    assert st["N1.maturity"] == "FAIL"


def test_self_promoted_maturity_fails_s0(pack_copy: Path) -> None:
    f = pack_copy / "region_asset_manifest.json"
    manifest = json.loads(f.read_text())
    manifest["maturity_declared"] = "A1"
    f.write_text(json.dumps(manifest))
    st = _statuses(pack_copy)
    assert st["S0.3"] == "FAIL"
