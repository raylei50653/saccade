#!/usr/bin/env python3
"""Exploratory #465 study r2: in which tracker block does R_T first diverge structurally from R_C?

Implements ``docs/research/studies/tracker_block_divergence_465_r2/`` (contract
§20.11, tier ``exploratory``; the result is not citable by a formal chain). It
supersedes ``tracker_block_divergence_465`` (r1), whose only attempt was
invalid on ``V_REPEAT``: r1 required the two replays of an arm to produce
byte-identical dumps, but the order of a track's ``CND`` rows is the order of
its candidate slots, which ``atomicAdd`` allocates (race-ordered; the producer
promises no order).

r2 changes exactly one thing (declaration §r2, §3): ``V_REPEAT`` compares the
dump segments of a repeat pair under :func:`canonical_segment`, where each
maximal run of consecutive ``CND`` lines is a multiset of complete line bytes
and every other line is compared exactly, in place. Everything else -- the
replay worker, the dump parser, the A/P/E comparisons, the ladder, the
terminal rule and the report -- is r1's code, imported unchanged from the r1
runner (whose blob, and the r2 stage study runner's, this binding pins). The
canonical form is used only by :func:`repeat_problem`; the raw dumps are
stored as written and the cross-arm comparison reads them exactly as r1 does.

Usage (the one formal attempt; clean tree at the freeze tag, under a lease)::

  .venv/bin/python tools/resctl.py run gpu0 -- \\
      .venv/bin/python scripts/eval/diagnostics/tracker_block_divergence_465_r2.py \\
      --raw-out <raw record directory named in declaration §8>
"""

# status: experiment

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "tools"))

from research_study import StudyBinding, open_frozen_study  # noqa: E402

R1_RUNNER = "scripts/eval/diagnostics/tracker_block_divergence_465.py"
S2_RUNNER = "scripts/eval/diagnostics/tf32_off_head_drift_stage_r2.py"


def _load_r1() -> Any:
    """The r1 runner (frozen with r1; every function but V_REPEAT is reused unchanged)."""
    spec = importlib.util.spec_from_file_location(
        "tracker_block_divergence_465", project_root / R1_RUNNER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r1 = _load_r1()
s2 = r1.s2

# --- frozen by the declaration ----------------------------------------------
STUDY_ID = "tracker_block_divergence_465_r2"
BINDING = StudyBinding(
    study_id=STUDY_ID,
    runner_file=__file__,
    freeze_tag=f"freeze/{STUDY_ID}/1",
    pinned_blobs={
        f"docs/research/studies/{STUDY_ID}/study.yaml": "e507a0921808bbd803b3b7b455fefee7303b8063",
        f"docs/research/studies/{STUDY_ID}/declaration.md": "3f9ef82f061144c41a02ef1869adcce758f41530",
        # The reused inference surface: r1's runner and the comparator it loads.
        R1_RUNNER: "4dd2f0d6f06f2c483f79a9e1cf67e270c08e52ab",
        S2_RUNNER: "13e59ca7568d37ac665a89c7b111b782f69dfdd9",
    },
)

# Unchanged from r1 (declaration §0–§2, §4–§6 are r1's text).
SEQUENCE_FRAMES = r1.SEQUENCE_FRAMES
REPLAYS = r1.REPLAYS
REF_RUN = r1.REF_RUN
REPEAT_PAIRS = r1.REPEAT_PAIRS
COMPARED = r1.COMPARED
PRIMARY = r1.PRIMARY
TERMINAL_BY_BLOCK = r1.TERMINAL_BY_BLOCK
SPLIT = r1.SPLIT
VALIDITY_ORDER = r1.VALIDITY_ORDER
LEASE_RESOURCES = r1.LEASE_RESOURCES
DUMP_ENV = r1.DUMP_ENV
PRESET_NAME = r1.PRESET_NAME
IOU_MIN = r1.IOU_MIN
SUPPORT_MIN = r1.SUPPORT_MIN
Invalid = r1.Invalid
decide = r1.decide
held_lease = r1.held_lease

CND_PREFIX = b"CND,"


# --------------------------------------------------------------------------
# V_REPEAT (the r2 change; unit-tested on synthetic data)
# --------------------------------------------------------------------------
def canonical_segment(data: bytes) -> list[tuple[str, Any]]:
    """The repeat-equivalence form of one dump segment (declaration §3, item 5).

    The segment is split on ``\\n``. Each maximal run of consecutive ``CND``
    lines (one track's candidate rows: the writer emits a track's ``TRK`` line
    and then its ``CND`` lines) becomes the multiset of its complete line
    bytes, represented as the sorted tuple of those lines so multiplicities
    count. Every other line -- ``GMC``, ``TRK``, anything else, and the
    remainder after the last ``\\n`` -- stays an exact, positioned element. A
    ``CND`` line therefore never moves across a ``TRK`` or ``GMC`` line.
    """
    out: list[tuple[str, Any]] = []
    run: list[bytes] = []
    for line in data.split(b"\n"):
        if line.startswith(CND_PREFIX):
            run.append(line)
            continue
        if run:
            out.append(("CND", tuple(sorted(run))))
            run = []
        out.append(("LINE", line))
    if run:
        out.append(("CND", tuple(sorted(run))))
    return out


def repeat_problem(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    """First V_REPEAT violation between two replays of one arm, or ``None``."""
    for seq, n in SEQUENCE_FRAMES.items():
        for f in range(1, n + 1):
            sa, sb = a["segments"][seq][f], b["segments"][seq][f]
            if sa != sb and canonical_segment(sa) != canonical_segment(sb):
                return f"{seq} f{f}: dump differs beyond CND row order"
            if a["states"][seq][f] != b["states"][seq][f]:
                return f"{seq} f{f}: step-end state differs"
    return None


# --------------------------------------------------------------------------
# attempt (r1's evaluate with the V_REPEAT check replaced)
# --------------------------------------------------------------------------
def evaluate(study: Any, raw_out: Path) -> dict[str, Any]:
    # V_COMPLETE / V_FORMAT over every input before any replay launches (§3).
    members = set(study.input_members("r2"))
    need = {arm for _, arm in REPLAYS}
    for arm in sorted(need):
        for seq in SEQUENCE_FRAMES:
            if f"l2/{arm}.evidence/{seq}.npz" not in members:
                raise Invalid(
                    "V_COMPLETE", f"r2 member l2/{arm}.evidence/{seq}.npz missing"
                )
    ti: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    r2_out: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = {}
    for arm in sorted(need):
        ti[arm], r2_out[arm], paths[arm] = {}, {}, {}
        for seq, n in SEQUENCE_FRAMES.items():
            member = f"l2/{arm}.evidence/{seq}.npz"
            try:
                path = study.input_file("r2", member)
                ev = np.load(path)
            except Exception as exc:  # noqa: BLE001
                raise Invalid("V_COMPLETE", f"{member}: {exc}") from exc
            try:
                ti[arm][seq] = s2.stage_view(ev, "tracker_input", n)
                r2_out[arm][seq] = r1.r2_tracker_full(ev, n)
            except (KeyError, ValueError) as exc:
                raise Invalid("V_FORMAT", f"{member}: {exc}") from exc
            for f, rows in ti[arm][seq].items():
                if (
                    rows.ndim != 2
                    or rows.shape[1] != 6
                    or not np.all(np.isfinite(rows))
                ):
                    raise Invalid(
                        "V_FORMAT", f"{member} f{f}: tracker_input rows malformed"
                    )
            paths[arm][seq] = str(path)

    replays: dict[str, dict[str, Any]] = {}
    for run, arm in REPLAYS:
        run_dir = raw_out / run.replace("#", "_")
        run_dir.mkdir(parents=True, exist_ok=False)
        inputs_json = run_dir / "inputs.json"
        inputs_json.write_text(
            json.dumps(paths[arm], indent=2) + "\n", encoding="utf-8"
        )
        log = run_dir / "stdout.log"
        with log.open("wb") as fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-inputs",
                    str(inputs_json),
                    "--worker-out",
                    str(run_dir),
                ],
                cwd=project_root,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            raise Invalid(
                "V_REPLAY", f"{run}: worker exit {proc.returncode} (see {log.name})"
            )
        rec = r1.load_replay(run_dir, ti[arm])
        emits = r1.load_emits(run_dir)
        for seq, n in SEQUENCE_FRAMES.items():
            for f in range(1, n + 1):
                if not r1.emits_bit_equal(emits[seq][f], r2_out[arm][seq][f]):
                    raise Invalid(
                        "V_REPLAY",
                        f"{run}: {seq} f{f} tracker output differs from {arm}'s r2 capture",
                    )
        rec["emits"] = emits
        replays[run] = rec

    # The r2 change: CND rows within a track compare as a multiset (§3 item 5).
    for first_run, second_run in REPEAT_PAIRS:
        problem = repeat_problem(replays[first_run], replays[second_run])
        if problem is not None:
            raise Invalid("V_REPEAT", f"{problem}, {first_run} vs {second_run}")

    # From here on, r1 unchanged: the raw (not canonical) dumps are compared.
    ref_arm = dict(REPLAYS)[REF_RUN]
    by_run: dict[str, dict[str, Any]] = {}
    for run in COMPARED:
        arm = dict(REPLAYS)[run]
        by_run[run] = {}
        for seq, n in SEQUENCE_FRAMES.items():
            dr, do = replays[REF_RUN]["dumps"][seq], replays[run]["dumps"][seq]
            outr = {f: (v[0], v[2]) for f, v in replays[REF_RUN]["emits"][seq].items()}
            outo = {f: (v[0], v[2]) for f, v in replays[run]["emits"][seq].items()}
            sr, so = replays[REF_RUN]["states"][seq], replays[run]["states"][seq]
            views = (dr, do, sr, so, ti[ref_arm][seq], ti[arm][seq], outr, outo)
            lad = r1.ladder(n, *views)
            lad["first_detail"] = (
                r1.explain_first(lad["first"], *views)
                if lad["first"] is not None
                else None
            )
            gmc_diff = [f for f in range(1, n + 1) if dr[f]["gmc"] != do[f]["gmc"]]
            lad["gmc_row_mismatch_frames"] = len(gmc_diff)
            by_run[run][seq] = lad
    return by_run


def _report_md(by_run: Mapping[str, Any], terminal: str) -> str:
    text = r1._report_md(by_run, terminal)
    head, sep, rest = text.partition("\n")
    return head.replace(r1.STUDY_ID, STUDY_ID, 1) + sep + rest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-out", help="new raw record directory (declaration §8)")
    parser.add_argument("--worker-inputs", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_inputs or args.worker_out:
        if not (args.worker_inputs and args.worker_out):
            parser.error("internal worker flags are malformed")
        return r1.run_worker(Path(args.worker_inputs), Path(args.worker_out))
    if not args.raw_out:
        parser.error("--raw-out is required")
    lease = held_lease()
    if lease is None or lease["resource"] not in LEASE_RESOURCES:
        print(
            f"refused: run under `resctl run` holding one of {LEASE_RESOURCES}",
            file=sys.stderr,
        )
        return 2  # before the freeze opens: no attempt is consumed
    if DUMP_ENV in os.environ:
        print(
            f"refused: {DUMP_ENV} is set in the caller's environment", file=sys.stderr
        )
        return 2

    study = open_frozen_study(BINDING)  # no data path exists before this passes
    raw_out = (project_root / args.raw_out).resolve()
    from scripts.provenance.run_manifest import open_run

    open_run(
        raw_out,
        produced_by="diagnostic",
        preset=PRESET_NAME,
        detector="SDP",
        cmdline=[
            Path(__file__).relative_to(project_root).as_posix(),
            "--raw-out",
            args.raw_out,
        ],
    )
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        # The attempt directory is created only after the replays, so every
        # mot17.py child sees the clean freeze tree.
        by_run = evaluate(study, raw_out)
    except Exception as exc:  # noqa: BLE001 -- every failure is recorded, never dropped
        payload = study.payload_dir()
        criterion, detail = (
            (exc.criterion, exc.detail)
            if isinstance(exc, Invalid)
            else ("V_RUNNER", f"{type(exc).__name__}: {exc}")
        )
        manifest = r1.write_raw_manifest(raw_out)
        (payload / "invalid.json").write_text(
            json.dumps(
                {
                    "criterion": criterion,
                    "detail": detail,
                    "raw_out": args.raw_out,
                    "raw_manifest_sha256": r1._sha256(manifest),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        record = study.record("invalid", invalid_criterion=criterion)
        print(f"INVALID {criterion}: {detail}; recorded {record}")
        return 1
    blocks = {
        seq: (r["first"]["block"] if r["first"] is not None else None)
        for seq, r in by_run[PRIMARY].items()
    }
    terminal = decide(blocks)
    manifest = r1.write_raw_manifest(raw_out)
    payload = study.payload_dir()
    result = {
        "study_id": STUDY_ID,
        "terminal": terminal,
        "primary_first_blocks": blocks,
        "iou_min": IOU_MIN,
        "support_min": SUPPORT_MIN,
        "lease": lease,
        "started_utc": started,
        "raw_out": args.raw_out,
        "raw_manifest_sha256": r1._sha256(manifest),
        "runs": by_run,
    }
    (payload / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (payload / "report.md").write_text(_report_md(by_run, terminal), encoding="utf-8")
    record = study.record("valid", terminal=terminal)
    print(f"VALID terminal={terminal}; recorded {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
