#!/usr/bin/env python3
"""Minimal instrumented vs uninstrumented observer-effect check (issue #363).

Question: does enabling per-stage fingerprinting materially perturb the
execution conditions relevant to historical Block S MOT divergence?

This is not a rate study and not a localization session.  Each arm is n=1.
MOT identity of the pair is recorded as artifact identity only; it is not
an observer-effect verdict input.  CONDITION2_RULES and
LOCALIZATION_BUDGET_RUNS are not consulted and not moved.

Default flags match Block S: ``--preset baseline --detector SDP
--no-gpu-decode --sequences MOT17-02-SDP``.
"""
# status: diagnostic

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SCRIPT_DIR))

from eval_stage_fingerprint import (  # noqa: E402
    ALLOCATOR_RESERVED_MATERIAL_BYTES,
    OBSERVER_EFFECT_SITES,
    StageFingerprintCollector,
    install_eval_hooks,
    measure_allocator_reserved_growth,
    read_observer_effect,
)
from check_eval_repeat_identity import merge_eval_flags  # noqa: E402

ARMS = ("uninstrumented", "instrumented")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _cuda_allocator_snapshot() -> dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0}
        stats = torch.cuda.memory_stats()
    except Exception:
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0) or 0),
        "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0) or 0),
    }


def install_runtime_probe(output_path: Path) -> Any:
    """Count device-wide synchronizes and allocator occupancy around run_eval."""

    import torch
    import saccade.perception.eval.runner as runner

    original_sync = torch.cuda.synchronize
    original_run_eval = runner.run_eval
    counts = {"synchronize": 0}
    probe: dict[str, Any] = {
        "syncs_before_run_eval": 0,
        "syncs_after_run_eval": 0,
        "syncs_after_process": 0,
        "mem_before_run_eval": {"allocated_bytes": 0, "reserved_bytes": 0},
        "mem_after_run_eval": {"allocated_bytes": 0, "reserved_bytes": 0},
        "mem_after_process": {"allocated_bytes": 0, "reserved_bytes": 0},
    }

    def counting_sync(*args: Any, **kwargs: Any) -> Any:
        counts["synchronize"] += 1
        return original_sync(*args, **kwargs)

    def wrapped_run_eval(**kwargs: Any) -> Any:
        probe["syncs_before_run_eval"] = counts["synchronize"]
        probe["mem_before_run_eval"] = _cuda_allocator_snapshot()
        result = original_run_eval(**kwargs)
        probe["syncs_after_run_eval"] = counts["synchronize"]
        probe["mem_after_run_eval"] = _cuda_allocator_snapshot()
        return result

    torch.cuda.synchronize = counting_sync
    runner.run_eval = wrapped_run_eval

    def undo() -> None:
        probe["syncs_after_process"] = counts["synchronize"]
        probe["mem_after_process"] = _cuda_allocator_snapshot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        torch.cuda.synchronize = original_sync
        runner.run_eval = original_run_eval

    return undo


def cmd_child(*, arm: str, fingerprint_dir: Path, forwarded: Sequence[str]) -> int:
    if arm not in ARMS:
        print(f"unknown arm {arm!r}", file=sys.stderr)
        return 2
    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    collector: StageFingerprintCollector | None = None
    undo_hooks = None
    if arm == "instrumented":
        collector = StageFingerprintCollector(fingerprint_dir, include_payloads=True)
        undo_hooks = install_eval_hooks(collector)
    undo_probe = install_runtime_probe(fingerprint_dir / "observer_probe.json")
    original_argv = sys.argv[:]
    eval_script_dir = str(_ROOT / "scripts" / "eval")
    inserted = eval_script_dir not in sys.path
    if inserted:
        sys.path.insert(0, eval_script_dir)
    exit_code = 0
    try:
        sys.argv = [str(_ROOT / "scripts" / "eval" / "mot17.py"), *forwarded]
        runpy.run_path(
            str(_ROOT / "scripts" / "eval" / "mot17.py"), run_name="__main__"
        )
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            exit_code = 1
    finally:
        sys.argv = original_argv
        if inserted:
            sys.path.remove(eval_script_dir)
        if undo_hooks is not None:
            undo_hooks()
        if collector is not None:
            collector.finalize()
        undo_probe()
    return exit_code


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_compare(*, artifact_dir: Path, forwarded: Sequence[str]) -> int:
    eval_flags = merge_eval_flags(forwarded)
    root = artifact_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    exits: dict[str, int] = {}
    for arm in ARMS:
        arm_dir = root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_dir = arm_dir / "stage_fingerprint"
        output_dir = arm_dir / "eval"
        cmd = [
            python,
            str(_SCRIPT_DIR / "check_eval_observer_effect.py"),
            "child",
            "--arm",
            arm,
            "--fingerprint-dir",
            str(fingerprint_dir),
            "--",
            *eval_flags,
            "--output",
            str(output_dir),
        ]
        print(f"observer-effect: {arm} → {arm_dir}")
        proc = subprocess.run(cmd, cwd=_ROOT)
        exits[arm] = int(proc.returncode)
        if proc.returncode != 0:
            print(f"  {arm} eval exit {proc.returncode}", file=sys.stderr)

    un_probe = _load_json(
        root / "uninstrumented" / "stage_fingerprint" / "observer_probe.json"
    )
    inst_probe = _load_json(
        root / "instrumented" / "stage_fingerprint" / "observer_probe.json"
    )
    measured_path = (
        root / "instrumented" / "stage_fingerprint" / "observer_effect_measured.json"
    )
    measured = _load_json(measured_path) if measured_path.is_file() else {}

    un_sync = int(un_probe["syncs_after_run_eval"])
    inst_sync = int(inst_probe["syncs_after_run_eval"])
    extra_sync = inst_sync - un_sync
    un_reserved = int(un_probe["mem_after_run_eval"]["reserved_bytes"])
    inst_reserved = int(inst_probe["mem_after_run_eval"]["reserved_bytes"])
    n_reserved_increases = int(measured.get("n_reserved_increases_on_clone", 0) or 0)
    reserved_grew = measure_allocator_reserved_growth(
        uninstrumented_reserved_bytes=un_reserved,
        instrumented_reserved_bytes=inst_reserved,
        n_reserved_increases_on_clone=n_reserved_increases,
    )
    producing_path_gpu_clone = int(measured.get("n_clone_samples", 0) or 0) > 0
    inst_sync_after = int(inst_probe["syncs_after_process"])
    post_eval_synchronize = inst_sync_after > inst_sync

    reading = read_observer_effect(
        extra_device_sync_during_frame_loop=extra_sync > 0,
        allocator_reserved_grew_from_snapshots=reserved_grew,
        producing_path_gpu_clone=producing_path_gpu_clone,
        post_eval_synchronize=post_eval_synchronize,
    )
    payload = {
        "kind": reading.kind,
        "allowed_claim": reading.allowed_claim,
        "issue_close": reading.issue_close,
        "mechanism_claim": reading.mechanism_claim,
        "condition_1_advanced": reading.condition_1_advanced,
        "condition_2_advanced": reading.condition_2_advanced,
        "localization_budget_reopened": reading.localization_budget_reopened,
        "n_per_arm": 1,
        "not": [
            "divergence_rate",
            "localization_session",
            "condition_2_reading",
            "causal_mechanism",
        ],
        "eval_exits": exits,
        "sites": OBSERVER_EFFECT_SITES,
        "syncs": {
            "uninstrumented_during_run_eval": un_sync,
            "instrumented_during_run_eval": inst_sync,
            "extra_during_run_eval": extra_sync,
            "instrumented_after_process": inst_sync_after,
            "material_if_extra_during_run_eval_gt": 0,
        },
        "allocator": {
            "uninstrumented_reserved_after_run_eval": un_reserved,
            "instrumented_reserved_after_run_eval": inst_reserved,
            "reserved_delta": inst_reserved - un_reserved,
            "material_bytes": ALLOCATOR_RESERVED_MATERIAL_BYTES,
            "n_reserved_increases_on_clone": n_reserved_increases,
            "grew_from_snapshots": reserved_grew,
            "measured": measured,
        },
        "reading": reading.to_dict(),
    }
    (root / "observer_effect.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"observer-effect: kind={reading.kind} "
        f"extra_sync={extra_sync} reserved_delta={inst_reserved - un_reserved} "
        f"reserved_increases={n_reserved_increases} "
        f"clones={measured.get('n_clone_samples', 0)}"
    )
    print(f"  allowed_claim: {reading.allowed_claim}")
    if any(code != 0 for code in exits.values()):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    compare = sub.add_parser(
        "run",
        help="n=1 uninstrumented and n=1 instrumented Block S, then read",
    )
    compare.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="default: out/determinism/eval_observer_effect_<timestamp>/",
    )

    child = sub.add_parser("child", help=argparse.SUPPRESS)
    child.add_argument("--arm", choices=ARMS, required=True)
    child.add_argument("--fingerprint-dir", type=Path, required=True)

    if not argv_list or argv_list[0] in ("-h", "--help"):
        parser.print_help()
        return 0 if argv_list and argv_list[0] in ("-h", "--help") else 2

    if argv_list[0] == "child":
        args, forwarded = child.parse_known_args(argv_list[1:])
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return cmd_child(
            arm=args.arm,
            fingerprint_dir=args.fingerprint_dir,
            forwarded=forwarded,
        )
    if argv_list[0] == "run":
        args, forwarded = compare.parse_known_args(argv_list[1:])
        artifact = args.artifact_dir or Path(
            f"out/determinism/eval_observer_effect_{_timestamp()}"
        )
        return cmd_compare(artifact_dir=artifact, forwarded=forwarded)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
