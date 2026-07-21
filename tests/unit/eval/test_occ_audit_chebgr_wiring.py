"""WP1b: config / evaluator wiring for occ-exit Cheb-GR probe (default-off)."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

_eval_dir = Path(__file__).resolve().parents[3] / "scripts" / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

from config.lifecycle import LifecycleConfig  # noqa: E402
from mot17_args import build_parser  # noqa: E402
from saccade.perception.eval.clean_fifo_bank import CleanFifoBank  # noqa: E402
from saccade.perception.eval.config import (  # noqa: E402
    EvalConfig,
    parse_eval_config,
)
from saccade.perception.eval.cpp_runner import run_eval_cpp  # noqa: E402
from saccade.perception.eval.evaluator import run_eval  # noqa: E402
from saccade.perception.eval.occ_audit import (  # noqa: E402
    occ_exit_audit_lines_from_bank,
    plan_occ_audit_episodes,
)
from saccade.perception.eval.post_merge import _parse_mot_lines  # noqa: E402
from saccade.perception.eval.utils import append_dict_csv  # noqa: E402

_DIM = 16

_CHEBGR_CFG_KEYS = (
    "occ_audit_chebgr_probe",
    "occ_audit_chebgr_max_cost",
    "occ_audit_chebgr_margin",
    "occ_audit_chebgr_pool_frac",
    "occ_audit_chebgr_lambda",
    "occ_audit_chebgr_k2",
    "occ_audit_chebgr_max_fwd",
    "occ_audit_chebgr_fuse_lambda",
)

_CHEBGR_PASS_KEYS = (
    "chebgr_probe",
    "chebgr_max_cost",
    "chebgr_margin",
    "chebgr_pool_frac",
    "chebgr_lambda",
    "chebgr_k2",
    "chebgr_max_fwd",
    "chebgr_fuse_lambda",
)


def _identity_emb(seed: int, n: int = 5) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    center = F.normalize(torch.randn(_DIM, generator=g), dim=0)
    return [
        F.normalize(center + 0.02 * torch.randn(_DIM, generator=g), dim=0)
        for _ in range(n)
    ]


def _make_substrate_with_occ() -> list[str]:
    lines: list[str] = []
    for f in range(1, 11):
        lines.append(f"{f},1,0,0,10,20,0.9,-1,-1,-1")
    for f in range(11, 15):
        lines.append(f"{f},1,0,0,10,20,0.9,-1,-1,-1")
        lines.append(f"{f},99,0,5,10,20,0.9,-1,-1,-1")
    for f in range(15, 20):
        lines.append(f"{f},1,0,0,10,20,0.9,-1,-1,-1")
    return lines


# --- 1. config defaults stay off ------------------------------------------------


def test_config_chebgr_defaults_are_false_or_neutral():
    lc = LifecycleConfig()
    assert lc.occ_audit_chebgr_probe is False
    assert lc.occ_audit_chebgr_max_cost == 0.45
    assert lc.occ_audit_chebgr_margin == 0.0
    assert lc.occ_audit_chebgr_pool_frac == 0.3
    assert lc.occ_audit_chebgr_lambda == 2.0
    assert lc.occ_audit_chebgr_k2 == 6
    assert lc.occ_audit_chebgr_max_fwd == 50
    assert lc.occ_audit_chebgr_fuse_lambda == 0.3

    parser = build_parser()
    args, _ = parser.parse_known_args([])
    assert args.occ_audit_chebgr_probe is False

    cfg = parse_eval_config(
        output="/tmp/wp1b",
        data_root="datasets/MOT17",
        split="train",
        sequences="MOT17-04-SDP",
        conf_threshold=0.1,
        reid_mode="off",
        reid_model="mnv4",
        profile_stages=False,
        kwargs={},
    )
    assert isinstance(cfg, EvalConfig)
    assert cfg.occ_audit_chebgr_probe is False
    assert cfg.lifecycle.occ_audit_chebgr_probe is False


# --- 2. argparse / YAML can set the flags --------------------------------------


def test_argparse_sets_chebgr_probe_flags():
    parser = build_parser()
    args, _ = parser.parse_known_args(
        [
            "--occ-audit-chebgr-probe",
            "--occ-audit-chebgr-max-cost",
            "0.33",
            "--occ-audit-chebgr-margin",
            "0.12",
            "--occ-audit-chebgr-pool-frac",
            "0.25",
            "--occ-audit-chebgr-lambda",
            "1.5",
            "--occ-audit-chebgr-k2",
            "4",
            "--occ-audit-chebgr-max-fwd",
            "40",
            "--occ-audit-chebgr-fuse-lambda",
            "0.4",
        ]
    )
    assert args.occ_audit_chebgr_probe is True
    assert args.occ_audit_chebgr_max_cost == 0.33
    assert args.occ_audit_chebgr_margin == 0.12
    assert args.occ_audit_chebgr_pool_frac == 0.25
    assert args.occ_audit_chebgr_lambda == 1.5
    assert args.occ_audit_chebgr_k2 == 4
    assert args.occ_audit_chebgr_max_fwd == 40
    assert args.occ_audit_chebgr_fuse_lambda == 0.4


def test_yaml_lifecycle_sets_chebgr_probe():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lifecycle.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "occ_audit_chebgr_probe": True,
                    "occ_audit_chebgr_max_cost": 0.31,
                }
            )
        )
        lc = LifecycleConfig.from_yaml(path)
        assert lc.occ_audit_chebgr_probe is True
        assert lc.occ_audit_chebgr_max_cost == 0.31


def test_parse_eval_config_kwargs_chebgr_probe():
    cfg = parse_eval_config(
        output="/tmp/wp1b",
        data_root="datasets/MOT17",
        split="train",
        sequences="MOT17-04-SDP",
        conf_threshold=0.1,
        reid_mode="off",
        reid_model="mnv4",
        profile_stages=False,
        kwargs={
            "occ_audit_chebgr_probe": True,
            "occ_audit_chebgr_max_cost": 0.28,
            "occ_audit_chebgr_k2": 8,
        },
    )
    assert cfg.occ_audit_chebgr_probe is True
    assert cfg.occ_audit_chebgr_max_cost == 0.28
    assert cfg.occ_audit_chebgr_k2 == 8
    for key in _CHEBGR_CFG_KEYS:
        assert hasattr(cfg, key)


# --- 3. evaluator / cpp_runner pass-through call sites -------------------------


def test_evaluator_and_cpp_runner_pass_chebgr_params():
    for fn in (run_eval, run_eval_cpp):
        src = inspect.getsource(fn)
        for key in _CHEBGR_PASS_KEYS:
            assert key in src, f"{fn.__name__} missing pass-through for {key}"
        for key in _CHEBGR_CFG_KEYS:
            assert key in src, f"{fn.__name__} missing cfg.{key} read"


# --- 4. CSV columns only when probe on ----------------------------------------


def test_csv_chebgr_columns_only_when_probe_enabled():
    lines = _make_substrate_with_occ()
    records = _parse_mot_lines(lines)
    episodes = plan_occ_audit_episodes(
        records,
        appearance_occlusion_cov=0.3,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
    )
    ep = episodes[0]
    identity_a = _identity_emb(42)
    identity_b = _identity_emb(99)
    bank = CleanFifoBank(fifo_n=20, stride=1, decide_n=5)
    for i, f in enumerate(range(1, 11)):
        bank.store(1, identity_a[i % len(identity_a)], f)
        bank.store(99, identity_b[i % len(identity_b)], f)
    audit_embs = {(1, f): identity_b[f % len(identity_b)] for f in ep.audit_frames}
    common = dict(
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        appearance_occlusion_cov=0.3,
    )

    with tempfile.TemporaryDirectory() as td:
        off_path = Path(td) / "off.csv"
        on_path = Path(td) / "on.csv"

        log_off: list[dict] = []
        occ_exit_audit_lines_from_bank(
            lines, bank, audit_embs, decision_log=log_off, chebgr_probe=False, **common
        )
        append_dict_csv(off_path, [{"seq": "x", **r} for r in log_off])

        log_on: list[dict] = []
        occ_exit_audit_lines_from_bank(
            lines, bank, audit_embs, decision_log=log_on, chebgr_probe=True, **common
        )
        append_dict_csv(on_path, [{"seq": "x", **r} for r in log_on])

        off_header = off_path.read_text().splitlines()[0].split(",")
        on_header = on_path.read_text().splitlines()[0].split(",")

        assert not any(h.startswith("chebgr_") for h in off_header)
        assert "flag_delta" not in off_header
        assert "chebgr_self_cost" in on_header
        assert "chebgr_flag" in on_header
        assert "flag_delta" in on_header
        assert "chebgr_margin" in on_header
